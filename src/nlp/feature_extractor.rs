use ndarray::Array2;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::{normalize_text, tokenize};

pub type DenseMatrix = Array2<f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastTextConfig {
    pub min_n: usize,
    pub max_n: usize,
    pub dimensions: usize,
}

#[derive(Debug, Clone)]
pub struct FastTextConfigBuilder {
    min_n: usize,
    max_n: usize,
    dimensions: usize,
}

impl FastTextConfigBuilder {
    pub fn new() -> Self {
        Self { min_n: 3, max_n: 5, dimensions: 32 }
    }
    pub fn min_n(mut self, value: usize) -> Self { self.min_n = value; self }
    pub fn max_n(mut self, value: usize) -> Self { self.max_n = value; self }
    pub fn dimensions(mut self, value: usize) -> Self { self.dimensions = value.max(1); self }
    pub fn build(self) -> Result<FastTextConfig, crate::error::VecEyesError> {
        if self.min_n == 0 || self.max_n == 0 {
            return Err(crate::error::VecEyesError::InvalidConfig("FastText n-gram sizes must be >= 1".into()));
        }
        if self.min_n > self.max_n {
            return Err(crate::error::VecEyesError::InvalidConfig("FastText min_n cannot be greater than max_n".into()));
        }
        Ok(FastTextConfig { min_n: self.min_n, max_n: self.max_n, dimensions: self.dimensions.max(1) })
    }
}

#[derive(Debug, Clone)]
pub struct TfIdfModel {
    pub vocab: Vec<String>,
    pub token_to_index: HashMap<String, usize>,
    pub idf: Vec<f32>,
    pub min_df: usize,
    pub max_df_ratio: f32,
}

const DEFAULT_MIN_DF: usize = 1;
const DEFAULT_MAX_DF_RATIO: f32 = 0.95;
const DEFAULT_STOPWORDS: &[&str] = &[
    "the", "is", "to", "a", "an", "and", "or", "of", "for", "in", "on", "at", "by", "with", "from",
    "that", "this", "it", "be", "as", "are", "was", "were",
];

pub fn fit_tfidf(texts: &[String]) -> TfIdfModel {
    fit_tfidf_with_config(texts, DEFAULT_MIN_DF, DEFAULT_MAX_DF_RATIO)
}

pub fn fit_tfidf_with_config(texts: &[String], min_df: usize, max_df_ratio: f32) -> TfIdfModel {
    let mut df: HashMap<String, usize> = HashMap::new();
    let stopwords: HashSet<&str> = DEFAULT_STOPWORDS.iter().copied().collect();
    for text in texts {
        let normalized = normalize_text(text);
        let tokens = tokenize(&normalized);
        let mut seen = HashSet::new();
        for token in &tokens {
            if stopwords.contains(token.as_str()) { continue; }
            if seen.insert(token.clone()) {
                *df.entry(token.clone()).or_insert(0) += 1;
            }
        }
    }
    let n_docs = texts.len().max(1) as f32;
    let mut vocab: Vec<String> = df.keys().filter(|token| {
        let freq = *df.get(*token).unwrap_or(&0);
        freq >= min_df && (freq as f32 / n_docs) <= max_df_ratio
    }).cloned().collect();
    vocab.sort();
    let mut token_to_index = HashMap::new();
    for (idx, token) in vocab.iter().enumerate() { token_to_index.insert(token.clone(), idx); }
    let mut idf = vec![0.0; vocab.len()];
    for token in &vocab {
        let index = token_to_index[token];
        let doc_freq = *df.get(token).unwrap_or(&1) as f32;
        idf[index] = ((n_docs + 1.0) / (doc_freq + 1.0)).ln() + 1.0;
    }
    TfIdfModel { vocab, token_to_index, idf, min_df, max_df_ratio }
}

/// Transforms texts into a TF-IDF matrix.
/// Uses sublinear TF (`1 + ln(count)`) so the scale is consistent between
/// training and single-document inference regardless of document length.
pub fn transform_tfidf(model: &TfIdfModel, texts: &[String]) -> DenseMatrix {
    let rows = texts.len();
    let cols = model.vocab.len();
    let mut matrix = Array2::<f32>::zeros((rows, cols));
    for row in 0..rows {
        let normalized = normalize_text(&texts[row]);
        let tokens = tokenize(&normalized);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for token in tokens {
            if let Some(&index) = model.token_to_index.get(&token) {
                *counts.entry(index).or_insert(0) += 1;
            }
        }
        if counts.is_empty() { continue; }
        for (index, count) in counts {
            // sublinear TF: avoids large counts dominating and is invariant to
            // document length normalization, making train/inference consistent.
            let tf = 1.0 + (count as f32).ln();
            matrix[[row, index]] = tf * model.idf[index];
        }
        l2_normalize_row(&mut matrix, row);
    }
    matrix
}

#[derive(Debug, Clone)]
pub struct WordEmbeddingModel {
    pub dims: usize,
    pub vectors: HashMap<String, Vec<f32>>,
    pub fasttext: Option<FastTextConfig>,
}

impl WordEmbeddingModel {
    pub fn train_word2vec(texts: &[String], dims: usize) -> Self {
        let vectors = train_context_embeddings(texts, dims, None);
        Self { dims, vectors, fasttext: None }
    }
    pub fn train_fasttext(texts: &[String], dims: usize, config: FastTextConfig) -> Self {
        let vectors = train_context_embeddings(texts, dims, Some(&config));
        Self { dims, vectors, fasttext: Some(config) }
    }
    pub fn vector_for(&self, token: &str) -> Vec<f32> {
        if let Some(v) = self.vectors.get(token) { return v.clone(); }
        if let Some(config) = &self.fasttext { return fasttext_vector(token, self.dims, config); }
        vec![0.0; self.dims]
    }
}

pub fn dense_matrix_from_texts(model: &WordEmbeddingModel, texts: &[String]) -> DenseMatrix {
    dense_matrix_from_texts_with_tfidf(model, texts, None)
}

pub fn dense_matrix_from_texts_with_tfidf(
    model: &WordEmbeddingModel,
    texts: &[String],
    tfidf_model: Option<&TfIdfModel>,
) -> DenseMatrix {
    let mut matrix = Array2::<f32>::zeros((texts.len(), model.dims));
    for row in 0..texts.len() {
        let normalized = normalize_text(&texts[row]);
        let tokens = tokenize(&normalized);
        if tokens.is_empty() { continue; }
        let mut sum = vec![0.0f32; model.dims];
        let mut weight_sum = 0.0f32;
        for token in tokens {
            let vector = model.vector_for(&token);
            let idf_weight = tfidf_model
                .and_then(|m| m.token_to_index.get(&token).map(|&idx| m.idf[idx]))
                .unwrap_or(1.0);
            weight_sum += idf_weight;
            for d in 0..model.dims { sum[d] += vector[d] * idf_weight; }
        }
        let denom = weight_sum.max(1e-6);
        for d in 0..model.dims { matrix[[row, d]] = sum[d] / denom; }
        l2_normalize_row(&mut matrix, row);
    }
    matrix
}

fn l2_normalize_row(matrix: &mut DenseMatrix, row: usize) {
    let norm = (0..matrix.shape()[1])
        .map(|c| { let v = matrix[[row, c]]; v * v })
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for c in 0..matrix.shape()[1] { matrix[[row, c]] /= norm; }
    }
}

/// Skip-gram with separate W_in / W_out matrices and true random negative sampling.
///
/// Using two embedding matrices is the standard Skip-gram formulation (Mikolov 2013):
/// W_in encodes the center word, W_out encodes context words.  Sharing a single
/// matrix couples the gradients and degrades quality.  We return W_in as the
/// final word representation, which is the common practice.
fn train_context_embeddings(
    texts: &[String],
    dims: usize,
    fasttext: Option<&FastTextConfig>,
) -> HashMap<String, Vec<f32>> {
    let dims = dims.max(1);
    let window = 2usize;
    let n_negatives = 5usize;
    let epochs = 6usize;
    let learning_rate = 0.05f32;

    // --- build vocab + corpus ---
    let mut vocab_counts: HashMap<String, usize> = HashMap::new();
    let mut flat_tokens: Vec<Vec<String>> = Vec::new();
    for text in texts {
        let tokens = tokenize(&normalize_text(text));
        if !tokens.is_empty() {
            for t in &tokens { *vocab_counts.entry(t.clone()).or_insert(0) += 1; }
            flat_tokens.push(tokens);
        }
    }
    let mut vocab_list: Vec<String> = vocab_counts.keys().cloned().collect();
    vocab_list.sort();
    if vocab_list.is_empty() { return HashMap::new(); }

    let vocab_to_idx: HashMap<String, usize> = vocab_list
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i))
        .collect();
    let corpus: Vec<Vec<usize>> = flat_tokens
        .iter()
        .filter_map(|tokens| {
            let idx: Vec<usize> = tokens.iter().filter_map(|t| vocab_to_idx.get(t).copied()).collect();
            if idx.is_empty() { None } else { Some(idx) }
        })
        .collect();

    // --- two separate embedding matrices ---
    // W_in: center-word embeddings, initialised deterministically for reproducibility.
    // W_out: context embeddings, zero-initialised (standard practice).
    let mut w_in: Vec<Vec<f32>> = vocab_list.iter().map(|t| deterministic_vector(t, dims)).collect();
    let mut w_out = vec![vec![0.0f32; dims]; vocab_list.len()];

    // Noise distribution P(w)^(3/4) for negative sampling (Mikolov 2013)
    let mut noise_cdf = Vec::with_capacity(vocab_list.len());
    let mut running = 0.0f32;
    for token in &vocab_list {
        running += (*vocab_counts.get(token).unwrap_or(&1) as f32).powf(0.75);
        noise_cdf.push(running);
    }
    let noise_total = running.max(1e-6);

    let mut rng = StdRng::seed_from_u64(0x57A9_C0DE);

    for _ in 0..epochs {
        // shuffle corpus order each epoch to avoid order bias
        let mut order: Vec<usize> = (0..corpus.len()).collect();
        order.shuffle(&mut rng);

        for &ci in &order {
            let tokens = &corpus[ci];
            for (pos, &center) in tokens.iter().enumerate() {
                let start = pos.saturating_sub(window);
                let end = (pos + window + 1).min(tokens.len());
                for ctx_pos in start..end {
                    if ctx_pos == pos { continue; }
                    let ctx = tokens[ctx_pos];
                    update_skipgram(&mut w_in, &mut w_out, center, ctx, learning_rate, 1.0);
                    for _ in 0..n_negatives {
                        let mass = rng.random_range(0.0f32..noise_total);
                        let neg = noise_cdf
                            .binary_search_by(|p| p.partial_cmp(&mass).unwrap_or(std::cmp::Ordering::Greater))
                            .unwrap_or_else(|i| i)
                            .min(noise_cdf.len().saturating_sub(1));
                        if neg != ctx {
                            update_skipgram(&mut w_in, &mut w_out, center, neg, learning_rate, 0.0);
                        }
                    }
                }
            }
        }
    }

    // Optional FastText subword enrichment applied to W_in
    if let Some(cfg) = fasttext {
        for (token, vec) in vocab_list.iter().zip(w_in.iter_mut()) {
            let sub = fasttext_vector(token, dims, cfg);
            for d in 0..dims { vec[d] += sub[d] * 0.15; }
        }
    }

    // L2-normalise the final input embeddings
    for vec in &mut w_in {
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 { for v in vec.iter_mut() { *v /= norm; } }
    }

    vocab_list.into_iter().zip(w_in).collect()
}

/// Separate W_in / W_out gradient update for one (center, context/negative) pair.
fn update_skipgram(
    w_in: &mut [Vec<f32>],
    w_out: &mut [Vec<f32>],
    center: usize,
    ctx: usize,
    lr: f32,
    target: f32,
) {
    if center >= w_in.len() || ctx >= w_out.len() { return; }
    let dot: f32 = w_in[center].iter().zip(w_out[ctx].iter()).map(|(a, b)| a * b).sum();
    let error = target - (1.0 / (1.0 + (-dot).exp()));
    let dims = w_in[center].len();
    for d in 0..dims {
        let g_in  = lr * error * w_out[ctx][d];
        let g_out = lr * error * w_in[center][d];
        w_in[center][d]  += g_in;
        w_out[ctx][d]    += g_out;
    }
}

fn deterministic_vector(token: &str, dims: usize) -> Vec<f32> {
    let mut seed = 0u64;
    for b in token.as_bytes() { seed = seed.wrapping_mul(131).wrapping_add(*b as u64 + 17); }
    let mut vector = vec![0.0; dims];
    for item in vector.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let raw = ((seed >> 32) as u32) as f32 / u32::MAX as f32;
        *item = raw * 2.0 - 1.0;
    }
    vector
}

fn fasttext_vector(token: &str, dims: usize, cfg: &FastTextConfig) -> Vec<f32> {
    let token = format!("<{}>", token);
    let chars: Vec<char> = token.chars().collect();
    let mut sum = vec![0.0f32; dims];
    let mut count = 0.0f32;
    for n in cfg.min_n..=cfg.max_n {
        if n > chars.len() { continue; }
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            let v = deterministic_vector(&gram, dims);
            for d in 0..dims { sum[d] += v[d]; }
            count += 1.0;
        }
    }
    if count == 0.0 { return deterministic_vector(&token, dims); }
    for item in sum.iter_mut() { *item /= count; }
    sum
}
