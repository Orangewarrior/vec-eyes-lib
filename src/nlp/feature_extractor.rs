use ndarray::Array2;
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

pub fn transform_tfidf(model: &TfIdfModel, texts: &[String]) -> DenseMatrix {
    let rows = texts.len();
    let cols = model.vocab.len();
    let mut matrix = Array2::<f32>::zeros((rows, cols));
    for row in 0..texts.len() {
        let normalized = normalize_text(&texts[row]);
        let tokens = tokenize(&normalized);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for token in tokens {
            if let Some(index) = model.token_to_index.get(&token) { *counts.entry(*index).or_insert(0) += 1; }
        }
        let total_terms: usize = counts.values().sum();
        if total_terms == 0 { continue; }
        for (index, count) in counts {
            let tf = count as f32 / total_terms as f32;
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

pub fn dense_matrix_from_texts_with_tfidf(model: &WordEmbeddingModel, texts: &[String], tfidf_model: Option<&TfIdfModel>) -> DenseMatrix {
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
            for idx in 0..model.dims { sum[idx] += vector[idx] * idf_weight; }
        }
        let denom = weight_sum.max(1e-6);
        for idx in 0..model.dims { matrix[[row, idx]] = sum[idx] / denom; }
        l2_normalize_row(&mut matrix, row);
    }
    matrix
}

fn l2_normalize_row(matrix: &mut DenseMatrix, row: usize) {
    let mut norm = 0.0f32;
    for col in 0..matrix.shape()[1] { let v = matrix[[row, col]]; norm += v * v; }
    norm = norm.sqrt();
    if norm > 0.0 { for col in 0..matrix.shape()[1] { matrix[[row, col]] /= norm; } }
}

fn train_context_embeddings(texts: &[String], dims: usize, fasttext: Option<&FastTextConfig>) -> HashMap<String, Vec<f32>> {
    let dims = dims.max(1);
    let window = 2usize;
    let negative_samples = 2usize;
    let epochs = 6usize;
    let learning_rate = 0.05f32;
    let mut corpus: Vec<Vec<usize>> = Vec::new();
    let mut vocab_counts: HashMap<String, usize> = HashMap::new();
    let mut flat_tokens: Vec<Vec<String>> = Vec::new();

    for text in texts {
        let normalized = normalize_text(text);
        let tokens = tokenize(&normalized);
        if !tokens.is_empty() {
            for token in &tokens { *vocab_counts.entry(token.clone()).or_insert(0) += 1; }
            flat_tokens.push(tokens);
        }
    }
    let mut vocab_list: Vec<String> = vocab_counts.keys().cloned().collect();
    vocab_list.sort();
    if vocab_list.is_empty() { return HashMap::new(); }
    let vocab_to_idx: HashMap<String, usize> = vocab_list.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
    for tokens in &flat_tokens {
        let indices: Vec<usize> = tokens.iter().filter_map(|t| vocab_to_idx.get(t).copied()).collect();
        if !indices.is_empty() { corpus.push(indices); }
    }
    let mut vectors: Vec<Vec<f32>> = vocab_list.iter().map(|token| deterministic_vector(token, dims)).collect();
    let mut cdf = Vec::with_capacity(vocab_list.len());
    let mut running = 0.0f32;
    for token in &vocab_list {
        let count = *vocab_counts.get(token).unwrap_or(&1) as f32;
        running += count.powf(0.75);
        cdf.push(running);
    }
    let total_mass = running.max(1e-6);
    for epoch in 0..epochs {
        for tokens in &corpus {
            for (idx, &center_idx) in tokens.iter().enumerate() {
                let start = idx.saturating_sub(window);
                let end = (idx + window + 1).min(tokens.len());
                for ctx_pos in start..end {
                    if ctx_pos == idx { continue; }
                    let context_idx = tokens[ctx_pos];
                    update_pair(&mut vectors, center_idx, context_idx, learning_rate, 1.0);
                    for neg_round in 0..negative_samples {
                        let sample_key = format!("{}:{}:{}", epoch, idx, neg_round);
                        let sample_mass = (stable_hash(&sample_key) as f32 / u64::MAX as f32) * total_mass;
                        let neg_idx = match cdf.binary_search_by(|probe| probe.partial_cmp(&sample_mass).unwrap_or(std::cmp::Ordering::Greater)) {
                            Ok(i) => i,
                            Err(i) => i.min(cdf.len().saturating_sub(1)),
                        };
                        if neg_idx != context_idx {
                            update_pair(&mut vectors, center_idx, neg_idx, learning_rate * 0.5, 0.0);
                        }
                    }
                }
            }
        }
    }
    if let Some(cfg) = fasttext {
        for (token, vector) in vocab_list.iter().zip(vectors.iter_mut()) {
            let subword = fasttext_vector(token, dims, cfg);
            for dim in 0..dims { vector[dim] += subword[dim] * 0.15; }
        }
    }
    for vector in &mut vectors {
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 { for value in vector.iter_mut() { *value /= norm; } }
    }
    vocab_list.into_iter().zip(vectors.into_iter()).collect()
}

fn update_pair(vectors: &mut [Vec<f32>], center_idx: usize, context_idx: usize, learning_rate: f32, target: f32) {
    if center_idx == context_idx || center_idx >= vectors.len() || context_idx >= vectors.len() { return; }
    let center_old = vectors[center_idx].clone();
    let context_old = vectors[context_idx].clone();
    let dot = center_old.iter().zip(context_old.iter()).map(|(a, b)| a * b).sum::<f32>();
    let pred = 1.0 / (1.0 + (-dot).exp());
    let error = target - pred;
    for dim in 0..center_old.len() {
        vectors[center_idx][dim] += learning_rate * error * context_old[dim];
        vectors[context_idx][dim] += learning_rate * error * center_old[dim];
    }
}

fn stable_hash(token: &str) -> u64 {
    token.as_bytes().iter().fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(*b as u64 + 17))
}

fn deterministic_vector(token: &str, dims: usize) -> Vec<f32> {
    let mut seed = 0u64;
    for b in token.as_bytes() { seed = seed.wrapping_mul(131).wrapping_add(*b as u64 + 17); }
    let mut vector = vec![0.0; dims];
    for item in vector.iter_mut().take(dims) {
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
            let gram: String = chars[i..i+n].iter().collect();
            let vec = deterministic_vector(&gram, dims);
            for idx in 0..dims { sum[idx] += vec[idx]; }
            count += 1.0;
        }
    }
    if count == 0.0 { return deterministic_vector(&token, dims); }
    for item in sum.iter_mut().take(dims) { *item /= count; }
    sum
}
