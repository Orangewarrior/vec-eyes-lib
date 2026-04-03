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
    let mut docs_tokens: Vec<Vec<String>> = Vec::new();

    for text in texts {
        let normalized = normalize_text(text);
        let tokens = tokenize(&normalized);
        let mut seen = HashSet::new();
        for token in &tokens {
            if seen.insert(token.clone()) {
                *df.entry(token.clone()).or_insert(0) += 1;
            }
        }
        docs_tokens.push(tokens);
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
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for text in texts {
        let normalized = normalize_text(text);
        let tokens = tokenize(&normalized);
        for (idx, token) in tokens.iter().enumerate() {
            let entry = vectors.entry(token.clone()).or_insert_with(|| vec![0.0; dims]);
            let start = idx.saturating_sub(window);
            let end = (idx + window + 1).min(tokens.len());

            for (ctx_idx, context) in tokens[start..end].iter().enumerate() {
                let absolute_ctx_idx = start + ctx_idx;
                if absolute_ctx_idx == idx {
                    continue;
                }

                let base = if let Some(cfg) = fasttext {
                    fasttext_vector(context, dims, cfg)
                } else {
                    deterministic_vector(context, dims)
                };
                let distance = idx.abs_diff(absolute_ctx_idx).max(1) as f32;
                let weight = 1.0 / distance;

                for dim in 0..dims {
                    entry[dim] += base[dim] * weight;
                }
            }

            if let Some(cfg) = fasttext {
                let subword = fasttext_vector(token, dims, cfg);
                for dim in 0..dims {
                    entry[dim] += subword[dim] * 0.1;
                }
            }
        }
    }

    for vector in vectors.values_mut() {
        let mut norm = 0.0f32;
        for value in vector.iter() {
            norm += value * value;
        }
        let norm = norm.sqrt();
        if norm > 0.0 {
            for value in vector.iter_mut() {
                *value /= norm;
            }
        }
    }

    vectors
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
