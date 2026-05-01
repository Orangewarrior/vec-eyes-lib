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
        Self {
            min_n: 3,
            max_n: 5,
            dimensions: 32,
        }
    }
    pub fn min_n(mut self, value: usize) -> Self {
        self.min_n = value;
        self
    }
    pub fn max_n(mut self, value: usize) -> Self {
        self.max_n = value;
        self
    }
    pub fn dimensions(mut self, value: usize) -> Self {
        self.dimensions = value.max(1);
        self
    }
    pub fn build(self) -> Result<FastTextConfig, crate::error::VecEyesError> {
        if self.min_n == 0 || self.max_n == 0 {
            return Err(crate::error::VecEyesError::InvalidConfig(
                "FastText n-gram sizes must be >= 1".into(),
            ));
        }
        if self.min_n > self.max_n {
            return Err(crate::error::VecEyesError::InvalidConfig(
                "FastText min_n cannot be greater than max_n".into(),
            ));
        }
        Ok(FastTextConfig {
            min_n: self.min_n,
            max_n: self.max_n,
            dimensions: self.dimensions.max(1),
        })
    }
}

impl Default for FastTextConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

pub fn fit_tfidf<S: AsRef<str>>(texts: &[S]) -> TfIdfModel {
    fit_tfidf_with_config(texts, DEFAULT_MIN_DF, DEFAULT_MAX_DF_RATIO)
}

pub fn fit_tfidf_with_config<S: AsRef<str>>(
    texts: &[S],
    min_df: usize,
    max_df_ratio: f32,
) -> TfIdfModel {
    let mut df: HashMap<String, usize> = HashMap::new();
    let stopwords: HashSet<&str> = DEFAULT_STOPWORDS.iter().copied().collect();
    for text in texts {
        let normalized = normalize_text(text.as_ref());
        let tokens = tokenize(&normalized);
        let mut seen = HashSet::new();
        for token in &tokens {
            if stopwords.contains(token.as_str()) {
                continue;
            }
            if seen.insert(token.clone()) {
                *df.entry(token.clone()).or_insert(0) += 1;
            }
        }
    }
    let n_docs = texts.len().max(1) as f32;
    let mut vocab: Vec<String> = df
        .keys()
        .filter(|token| {
            let freq = *df.get(*token).unwrap_or(&0);
            freq >= min_df && (freq as f32 / n_docs) <= max_df_ratio
        })
        .cloned()
        .collect();
    vocab.sort();
    let mut token_to_index = HashMap::new();
    for (idx, token) in vocab.iter().enumerate() {
        token_to_index.insert(token.clone(), idx);
    }
    let mut idf = vec![0.0f32; vocab.len()];
    for token in &vocab {
        let index = token_to_index[token];
        let doc_freq = *df.get(token).unwrap_or(&1) as f32;
        idf[index] = ((n_docs + 1.0) / (doc_freq + 1.0)).ln() + 1.0;
    }
    TfIdfModel {
        vocab,
        token_to_index,
        idf,
        min_df,
        max_df_ratio,
    }
}

/// Sublinear TF-IDF transform with L2-normalised rows.
///
/// Uses `tf = 1 + ln(count)` (Salton & Buckley 1988) so high-frequency terms
/// don't dominate and the scale is consistent across document lengths.
pub fn transform_tfidf<S: AsRef<str>>(model: &TfIdfModel, texts: &[S]) -> DenseMatrix {
    use rayon::prelude::*;
    let cols = model.vocab.len();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let mut matrix = Array2::<f32>::zeros((refs.len(), cols));
    matrix
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .zip(refs.par_iter())
        .for_each(|(mut row, &text)| {
            let normalized = normalize_text(text);
            let tokens = tokenize(&normalized);
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for token in tokens {
                if let Some(&index) = model.token_to_index.get(&token) {
                    *counts.entry(index).or_insert(0) += 1;
                }
            }
            if counts.is_empty() {
                return;
            }
            for (index, count) in counts {
                row[index] = (1.0 + (count as f32).ln()) * model.idf[index];
            }
            let norm_sq: f32 = row.iter().map(|v| v * v).sum();
            let norm = norm_sq.sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|v| v / norm);
            }
        });
    matrix
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WordEmbeddingModel {
    pub dims: usize,
    pub vectors: HashMap<String, Vec<f32>>,
    pub fasttext: Option<FastTextConfig>,
}

impl WordEmbeddingModel {
    pub fn train_word2vec<S: AsRef<str>>(texts: &[S], dims: usize) -> Self {
        let vectors = train_context_embeddings(texts, dims, None);
        Self {
            dims,
            vectors,
            fasttext: None,
        }
    }
    pub fn train_fasttext<S: AsRef<str>>(texts: &[S], dims: usize, config: FastTextConfig) -> Self {
        let vectors = train_context_embeddings(texts, dims, Some(&config));
        Self {
            dims,
            vectors,
            fasttext: Some(config),
        }
    }
    pub fn vector_for(&self, token: &str) -> Vec<f32> {
        if let Some(v) = self.vectors.get(token) {
            return v.clone();
        }
        if let Some(config) = &self.fasttext {
            return fasttext_vector(token, self.dims, config);
        }
        vec![0.0; self.dims]
    }
}

pub fn dense_matrix_from_texts<S: AsRef<str>>(
    model: &WordEmbeddingModel,
    texts: &[S],
) -> DenseMatrix {
    dense_matrix_from_texts_with_tfidf(model, texts, None)
}

/// IDF-weighted average of word vectors with L2-normalised rows.
///
/// When `tfidf_model` is supplied it must be the model fitted on the **training**
/// corpus (never re-fitted on the probe document) so IDF weights are stationary.
pub fn dense_matrix_from_texts_with_tfidf<S: AsRef<str>>(
    model: &WordEmbeddingModel,
    texts: &[S],
    tfidf_model: Option<&TfIdfModel>,
) -> DenseMatrix {
    use rayon::prelude::*;
    let dims = model.dims;
    let refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let mut matrix = Array2::<f32>::zeros((refs.len(), dims));
    matrix
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .zip(refs.par_iter())
        .for_each(|(mut row, &text)| {
            let normalized = normalize_text(text);
            let tokens = tokenize(&normalized);
            if tokens.is_empty() {
                return;
            }
            // Per-row accumulator — avoids shared mutable state across threads.
            let mut acc = vec![0f32; dims];
            let mut weight_sum = 0.0f32;
            for token in &tokens {
                let vector = model.vector_for(token);
                let idf_weight = tfidf_model
                    .and_then(|m| m.token_to_index.get(token).map(|&idx| m.idf[idx]))
                    .unwrap_or(1.0);
                weight_sum += idf_weight;
                // Fused scaled accumulation — SIMD-vectorised by LLVM.
                acc.iter_mut()
                    .zip(vector.iter().take(dims))
                    .for_each(|(a, &v)| *a += v * idf_weight);
            }
            let denom = weight_sum.max(1e-6);
            row.iter_mut()
                .zip(acc.iter())
                .for_each(|(r, &a)| *r = a / denom);
            let norm_sq: f32 = row.iter().map(|v| v * v).sum();
            let norm = norm_sq.sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|v| v / norm);
            }
        });
    matrix
}

// ── Skip-gram training ────────────────────────────────────────────────────────

/// Skip-gram with separate W_in / W_out matrices and true random negative sampling.
///
/// Two embedding matrices (Mikolov 2013): W_in encodes center words, W_out
/// encodes context words.  Only W_in is returned as the final representation.
fn train_context_embeddings<S: AsRef<str>>(
    texts: &[S],
    dims: usize,
    fasttext: Option<&FastTextConfig>,
) -> HashMap<String, Vec<f32>> {
    let dims = dims.max(1);
    let window = 2usize;
    let n_negatives = 5usize;
    let epochs = 6usize;
    let learning_rate = 0.05f32;

    // ── vocab + indexed corpus ──────────────────────────────────────────────
    let mut vocab_counts: HashMap<String, usize> = HashMap::new();
    let mut flat_tokens: Vec<Vec<String>> = Vec::new();
    for text in texts {
        let tokens = tokenize(&normalize_text(text.as_ref()));
        if !tokens.is_empty() {
            for t in &tokens {
                *vocab_counts.entry(t.clone()).or_insert(0) += 1;
            }
            flat_tokens.push(tokens);
        }
    }
    let mut vocab_list: Vec<String> = vocab_counts.keys().cloned().collect();
    vocab_list.sort();
    if vocab_list.is_empty() {
        return HashMap::new();
    }

    let vocab_to_idx: HashMap<String, usize> = vocab_list
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i))
        .collect();
    let corpus: Vec<Vec<usize>> = flat_tokens
        .iter()
        .filter_map(|tokens| {
            let idx: Vec<usize> = tokens
                .iter()
                .filter_map(|t| vocab_to_idx.get(t).copied())
                .collect();
            if idx.is_empty() {
                None
            } else {
                Some(idx)
            }
        })
        .collect();

    // ── embedding matrices ─────────────────────────────────────────────────
    // W_in: deterministic init; W_out: zero-init (standard Skip-gram).
    let mut w_in: Vec<Vec<f32>> = vocab_list
        .iter()
        .map(|t| deterministic_vector(t, dims))
        .collect();
    let mut w_out = vec![vec![0.0f32; dims]; vocab_list.len()];

    // Unigram noise distribution P(w)^0.75 stored as a CDF for O(log V) sampling.
    let mut noise_cdf = Vec::with_capacity(vocab_list.len());
    let mut running = 0.0f32;
    for token in &vocab_list {
        running += (*vocab_counts.get(token).unwrap_or(&1) as f32).powf(0.75);
        noise_cdf.push(running);
    }
    let noise_total = running.max(1e-6);

    let mut rng = StdRng::seed_from_u64(0x57A9_C0DE);

    for _ in 0..epochs {
        let mut order: Vec<usize> = (0..corpus.len()).collect();
        order.shuffle(&mut rng);

        for &ci in &order {
            let tokens = &corpus[ci];
            for (pos, &center) in tokens.iter().enumerate() {
                let start = pos.saturating_sub(window);
                let end = (pos + window + 1).min(tokens.len());
                for (ctx_pos, &ctx) in tokens.iter().enumerate().take(end).skip(start) {
                    if ctx_pos == pos {
                        continue;
                    }
                    let scale = skipgram_scale(&w_in, &w_out, center, ctx, learning_rate, 1.0);
                    update_skipgram(&mut w_in, &mut w_out, center, ctx, scale);
                    for _ in 0..n_negatives {
                        let mass = rng.random_range(0.0f32..noise_total);
                        let neg = noise_cdf
                            .binary_search_by(|p| {
                                p.partial_cmp(&mass).unwrap_or(std::cmp::Ordering::Greater)
                            })
                            .unwrap_or_else(|i| i)
                            .min(noise_cdf.len().saturating_sub(1));
                        if neg != ctx {
                            let scale =
                                skipgram_scale(&w_in, &w_out, center, neg, learning_rate, 0.0);
                            update_skipgram(&mut w_in, &mut w_out, center, neg, scale);
                        }
                    }
                }
            }
        }
    }

    // Optional FastText subword enrichment — blend subword mean into W_in.
    if let Some(cfg) = fasttext {
        use rayon::prelude::*;
        vocab_list
            .par_iter()
            .zip(w_in.par_iter_mut())
            .for_each(|(token, vec)| {
                let sub = fasttext_vector(token, dims, cfg);
                vec.iter_mut()
                    .zip(sub.iter())
                    .for_each(|(v, &s)| *v += s * 0.15);
            });
    }

    // L2-normalise W_in before returning.
    {
        use rayon::prelude::*;
        w_in.par_iter_mut().for_each(|vec| {
            let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                vec.iter_mut().for_each(|v| *v *= inv);
            }
        });
    }

    vocab_list.into_iter().zip(w_in).collect()
}

/// Compute `lr × error` for one (center, ctx/neg) pair.
///
/// Pre-fusing this scalar lets `update_skipgram` receive a single `scale`
/// argument and avoids recomputing the sigmoid in the hot path.
#[inline(always)]
fn skipgram_scale(
    w_in: &[Vec<f32>],
    w_out: &[Vec<f32>],
    center: usize,
    ctx: usize,
    lr: f32,
    target: f32,
) -> f32 {
    let dot: f32 = w_in[center]
        .iter()
        .zip(w_out[ctx].iter())
        .map(|(a, b)| a * b)
        .sum();
    lr * (target - 1.0 / (1.0 + (-dot).exp()))
}

/// Simultaneous W_in / W_out gradient update for one (center, context) pair.
///
/// Strategy: copy the pre-update `w_out[ctx]` into a stack buffer (embeddings
/// are typically 32–256 dims, so 2 KB max), then run two independent passes.
/// Each pass is a pure `zip` with no aliasing, so LLVM can emit SIMD
/// multiply-add instructions without any unsafe code.
#[inline(always)]
fn update_skipgram(
    w_in: &mut [Vec<f32>],
    w_out: &mut [Vec<f32>],
    center: usize,
    ctx: usize,
    scale: f32,
) {
    if center >= w_in.len() || ctx >= w_out.len() {
        return;
    }
    let dims = w_in[center].len().min(w_out[ctx].len());

    // Stack buffer avoids a heap allocation on every call (≤ 512 dims = 2 KB).
    const STACK_CAP: usize = 512;
    if dims <= STACK_CAP {
        let mut buf = [0.0f32; STACK_CAP];
        buf[..dims].copy_from_slice(&w_out[ctx][..dims]); // snapshot of w_out

        // Pass 1 (SIMD): w_out[ctx] += scale * w_in[center]  (cross-param borrow — no conflict)
        for (wo, &wi) in w_out[ctx][..dims]
            .iter_mut()
            .zip(w_in[center][..dims].iter())
        {
            *wo += scale * wi;
        }
        // Pass 2 (SIMD): w_in[center] += scale * original w_out (from stack buf)
        for (wi, &wo) in w_in[center][..dims].iter_mut().zip(buf[..dims].iter()) {
            *wi += scale * wo;
        }
    } else {
        // Heap fallback for unusually large embeddings (> 512 dims).
        let wo_orig: Vec<f32> = w_out[ctx].clone();
        for (wo, &wi) in w_out[ctx].iter_mut().zip(w_in[center].iter()) {
            *wo += scale * wi;
        }
        for (wi, &wo) in w_in[center].iter_mut().zip(wo_orig.iter()) {
            *wi += scale * wo;
        }
    }
}

fn deterministic_vector(token: &str, dims: usize) -> Vec<f32> {
    let mut seed = 0u64;
    for b in token.as_bytes() {
        seed = seed.wrapping_mul(131).wrapping_add(*b as u64 + 17);
    }
    let mut vector = vec![0.0f32; dims];
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
        if n > chars.len() {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            let v = deterministic_vector(&gram, dims);
            for (s, gv) in sum.iter_mut().zip(v.iter()) {
                *s += gv;
            }
            count += 1.0;
        }
    }
    if count == 0.0 {
        return deterministic_vector(&token, dims);
    }
    sum.iter_mut().for_each(|v| *v /= count);
    sum
}
