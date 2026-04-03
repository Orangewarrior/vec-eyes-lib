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

#[derive(Debug, Clone)]
pub struct TfIdfModel {
    pub vocab: Vec<String>,
    pub token_to_index: HashMap<String, usize>,
    pub idf: Vec<f32>,
}

pub fn fit_tfidf(texts: &[String]) -> TfIdfModel {
    let mut df: HashMap<String, usize> = HashMap::new();

    for text in texts {
        let normalized = normalize_text(text);
        let tokens = tokenize(&normalized);
        let mut seen = HashSet::new();
        for token in &tokens {
            if seen.insert(token.clone()) {
                *df.entry(token.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut vocab: Vec<String> = df.keys().cloned().collect();
    vocab.sort();

    let mut token_to_index = HashMap::new();
    for (idx, token) in vocab.iter().enumerate() {
        token_to_index.insert(token.clone(), idx);
    }

    let n_docs = texts.len() as f32;
    let mut idf = vec![0.0; vocab.len()];
    for token in &vocab {
        let index = token_to_index[token];
        let doc_freq = *df.get(token).unwrap_or(&1) as f32;
        idf[index] = ((n_docs + 1.0) / (doc_freq + 1.0)).ln() + 1.0;
    }

    TfIdfModel { vocab, token_to_index, idf }
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
            if let Some(index) = model.token_to_index.get(&token) {
                *counts.entry(*index).or_insert(0) += 1;
            }
        }

        let total_terms: usize = counts.values().sum();
        if total_terms == 0 {
            continue;
        }

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
        if let Some(v) = self.vectors.get(token) {
            return v.clone();
        }

        if let Some(config) = &self.fasttext {
            return fasttext_vector(token, self.dims, config);
        }

        vec![0.0; self.dims]
    }
}

pub fn dense_matrix_from_texts(model: &WordEmbeddingModel, texts: &[String]) -> DenseMatrix {
    let mut matrix = Array2::<f32>::zeros((texts.len(), model.dims));
    for row in 0..texts.len() {
        let normalized = normalize_text(&texts[row]);
        let tokens = tokenize(&normalized);
        if tokens.is_empty() {
            continue;
        }

        let mut sum = vec![0.0f32; model.dims];
        let token_count = tokens.len() as f32;
        for token in tokens {
            let vector = model.vector_for(&token);
            for idx in 0..model.dims {
                sum[idx] += vector[idx];
            }
        }

        for idx in 0..model.dims {
            matrix[[row, idx]] = sum[idx] / token_count;
        }

        l2_normalize_row(&mut matrix, row);
    }
    matrix
}

fn l2_normalize_row(matrix: &mut DenseMatrix, row: usize) {
    let mut norm = 0.0f32;
    for col in 0..matrix.shape()[1] {
        let v = matrix[[row, col]];
        norm += v * v;
    }
    norm = norm.sqrt();
    if norm > 0.0 {
        for col in 0..matrix.shape()[1] {
            matrix[[row, col]] /= norm;
        }
    }
}


fn train_context_embeddings(
    texts: &[String],
    dims: usize,
    fasttext: Option<&FastTextConfig>,
) -> HashMap<String, Vec<f32>> {
    let dims = dims.max(1);
    let window = 2usize;
    let negative_samples = 2usize;
    let epochs = 6usize;
    let learning_rate = 0.05f32;
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut corpus: Vec<Vec<String>> = Vec::new();
    let mut vocab = HashSet::new();

    for text in texts {
        let normalized = normalize_text(text);
        let tokens = tokenize(&normalized);
        if !tokens.is_empty() {
            for token in &tokens {
                vocab.insert(token.clone());
                vectors.entry(token.clone()).or_insert_with(|| deterministic_vector(token, dims));
            }
            corpus.push(tokens);
        }
    }

    let vocab_list: Vec<String> = vocab.into_iter().collect();
    if vocab_list.is_empty() {
        return vectors;
    }

    for _ in 0..epochs {
        for tokens in &corpus {
            for (idx, token) in tokens.iter().enumerate() {
                let start = idx.saturating_sub(window);
                let end = (idx + window + 1).min(tokens.len());
                for (ctx_idx, context) in tokens[start..end].iter().enumerate() {
                    let absolute_ctx_idx = start + ctx_idx;
                    if absolute_ctx_idx == idx {
                        continue;
                    }
                    update_skipgram_pair(&mut vectors, token, context, dims, learning_rate, 1.0);
                    for neg in 0..negative_samples {
                        let neg_idx = (stable_hash(token) as usize + absolute_ctx_idx + neg) % vocab_list.len();
                        let negative = &vocab_list[neg_idx];
                        if negative != context {
                            update_skipgram_pair(&mut vectors, token, negative, dims, learning_rate * 0.5, 0.0);
                        }
                    }
                }
            }
        }
    }

    if let Some(cfg) = fasttext {
        for (token, vector) in vectors.iter_mut() {
            let subword = fasttext_vector(token, dims, cfg);
            for dim in 0..dims {
                vector[dim] += subword[dim] * 0.15;
            }
        }
    }

    for vector in vectors.values_mut() {
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in vector.iter_mut() {
                *value /= norm;
            }
        }
    }

    vectors
}

fn update_skipgram_pair(
    vectors: &mut HashMap<String, Vec<f32>>,
    center: &str,
    context: &str,
    dims: usize,
    learning_rate: f32,
    target: f32,
) {
    let center_vec = vectors.get(center).cloned().unwrap_or_else(|| deterministic_vector(center, dims));
    let context_vec = vectors.get(context).cloned().unwrap_or_else(|| deterministic_vector(context, dims));
    let dot = center_vec.iter().zip(context_vec.iter()).map(|(a, b)| a * b).sum::<f32>();
    let pred = 1.0 / (1.0 + (-dot).exp());
    let error = target - pred;

    let mut new_center = center_vec.clone();
    let mut new_context = context_vec.clone();
    for dim in 0..dims {
        new_center[dim] += learning_rate * error * context_vec[dim];
        new_context[dim] += learning_rate * error * center_vec[dim];
    }
    vectors.insert(center.to_string(), new_center);
    vectors.insert(context.to_string(), new_context);
}

fn stable_hash(token: &str) -> u64 {
    token
        .as_bytes()
        .iter()
        .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(*b as u64 + 17))
}

fn deterministic_vector(token: &str, dims: usize) -> Vec<f32> {
    let mut seed = 0u64;
    for b in token.as_bytes() {
        seed = seed.wrapping_mul(131).wrapping_add(*b as u64 + 17);
    }

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
        if n > chars.len() {
            continue;
        }
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            let vec = deterministic_vector(&gram, dims);
            for idx in 0..dims {
                sum[idx] += vec[idx];
            }
            count += 1.0;
        }
    }

    if count == 0.0 {
        return deterministic_vector(&token, dims);
    }

    for item in sum.iter_mut().take(dims) {
        *item /= count;
    }
    sum
}
