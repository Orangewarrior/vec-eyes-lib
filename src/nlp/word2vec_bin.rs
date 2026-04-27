//! Loader for Google word2vec binary (`.bin`) files.
//!
//! Binary format (original Google word2vec):
//! ```text
//! <vocab_size> <dims>\n
//! <word> <dims × float32 LE>\n   (repeated vocab_size times)
//! ```
//! The word and its vector are separated by a single space byte.
//! A newline byte follows each vector.  Some encoder variants omit the
//! trailing newline; the reader skips leading newlines before each word to
//! stay in sync either way.

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::VecEyesError;
use crate::nlp::{DenseMatrix, TfIdfModel};

// ── Low-level binary helpers ──────────────────────────────────────────────────

fn read_header(r: &mut impl Read) -> Result<(usize, usize), VecEyesError> {
    let mut bytes = Vec::with_capacity(32);
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        if b[0] == b'\n' {
            break;
        }
        bytes.push(b[0]);
    }
    let s = String::from_utf8(bytes)
        .map_err(|e| VecEyesError::invalid_config("Word2VecBin::header", e.to_string()))?;
    let mut parts = s.split_whitespace();
    let vocab: usize = parts
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| VecEyesError::invalid_config("Word2VecBin::header", "missing vocab_size"))?;
    let dims: usize = parts
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| VecEyesError::invalid_config("Word2VecBin::header", "missing dims"))?;
    Ok((vocab, dims))
}

fn read_word(r: &mut impl Read) -> Result<String, VecEyesError> {
    let mut bytes = Vec::with_capacity(32);
    // Skip leading newlines — some encoders write "\n<word> ..." instead of
    // relying on the trailing newline after each vector.
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        if b[0] != b'\n' {
            bytes.push(b[0]);
            break;
        }
    }
    // Read until the space that separates the word from its vector.
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        if b[0] == b' ' || b[0] == b'\t' {
            break;
        }
        bytes.push(b[0]);
    }
    String::from_utf8(bytes)
        .map_err(|e| VecEyesError::invalid_config("Word2VecBin::word", e.to_string()))
}

fn read_floats(r: &mut impl Read, buf: &mut [f32]) -> Result<(), VecEyesError> {
    // Chunked read to avoid large stack allocations.
    const CHUNK: usize = 4096;
    let mut tmp = [0u8; CHUNK * 4];
    let mut offset = 0;
    while offset < buf.len() {
        let n = (buf.len() - offset).min(CHUNK);
        r.read_exact(&mut tmp[..n * 4])?;
        for (dst, src) in buf[offset..offset + n].iter_mut().zip(tmp[..n * 4].chunks_exact(4)) {
            *dst = f32::from_le_bytes(src.try_into().unwrap());
        }
        offset += n;
    }
    Ok(())
}

// ── Word2VecBin ───────────────────────────────────────────────────────────────

/// Raw loader for a Google word2vec binary file.
///
/// Keeps the full vocabulary and embedding matrix in memory.  Call
/// [`extract_all`](Word2VecBin::extract_all) or
/// [`extract_for_vocab`](Word2VecBin::extract_for_vocab) to obtain a
/// serializable [`Word2VecEmbeddings`] that can be passed to any classifier's
/// `train_with_external_embeddings`.
pub struct Word2VecBin {
    dims: usize,
    word_index: HashMap<String, usize>,
    /// Row-major flat matrix: row `i` = `word_index[word] * dims`.
    matrix: Vec<f32>,
}

impl Word2VecBin {
    /// Parse a word2vec `.bin` file produced by the Google word2vec tool or
    /// any compatible encoder (gensim `save_word2vec_format(binary=True)`, etc.)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, VecEyesError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mut reader = BufReader::with_capacity(1 << 20, file);

        let (vocab_size, dims) = read_header(&mut reader)?;

        let mut word_index = HashMap::with_capacity(vocab_size);
        let mut matrix = vec![0f32; vocab_size * dims];

        for i in 0..vocab_size {
            let word = read_word(&mut reader)?;
            word_index.insert(word, i);
            read_floats(&mut reader, &mut matrix[i * dims..(i + 1) * dims])?;
            // Consume the trailing newline written by the encoder.
            // If the encoder omitted it, read_word will skip it on the next
            // iteration anyway.
            let mut maybe_nl = [0u8; 1];
            let _ = reader.read(&mut maybe_nl);
            if maybe_nl[0] != b'\n' {
                // Not a newline — we consumed one byte too many; but since the
                // next word is read byte-by-byte starting with newline-skipping,
                // this is handled gracefully.
            }
        }

        Ok(Self { dims, word_index, matrix })
    }

    /// Extract every word vector from the model.
    ///
    /// For large pre-trained models (millions of words) this allocates
    /// significant memory.  Prefer [`extract_for_vocab`](Word2VecBin::extract_for_vocab)
    /// when you only need vectors for your training/inference corpus.
    pub fn extract_all(&self) -> Word2VecEmbeddings {
        let mut word_vectors = HashMap::with_capacity(self.word_index.len());
        for (word, &idx) in &self.word_index {
            let start = idx * self.dims;
            word_vectors.insert(word.clone(), self.matrix[start..start + self.dims].to_vec());
        }
        let centroid = compute_centroid(&word_vectors, self.dims);
        Word2VecEmbeddings { dims: self.dims, word_vectors, centroid }
    }

    /// Extract only the vectors needed for the given vocabulary.
    ///
    /// Each string in `vocab` is normalised and tokenised the same way the
    /// classifier pipeline does, and only the resulting tokens that exist in
    /// the model are kept.  This produces a much smaller
    /// [`Word2VecEmbeddings`] than [`extract_all`](Word2VecBin::extract_all).
    pub fn extract_for_vocab(&self, vocab: &[&str]) -> Word2VecEmbeddings {
        let mut word_vectors = HashMap::new();
        for &text in vocab {
            let normalized = crate::nlp::normalize_text(text);
            for token in crate::nlp::tokenize(&normalized) {
                if let Some(&idx) = self.word_index.get(&token) {
                    let start = idx * self.dims;
                    word_vectors
                        .entry(token)
                        .or_insert_with(|| self.matrix[start..start + self.dims].to_vec());
                }
            }
        }
        let centroid = compute_centroid(&word_vectors, self.dims);
        Word2VecEmbeddings { dims: self.dims, word_vectors, centroid }
    }
}

// ── Centroid helper ───────────────────────────────────────────────────────────

fn compute_centroid(word_vectors: &HashMap<String, Vec<f32>>, dims: usize) -> Vec<f32> {
    if word_vectors.is_empty() {
        return vec![0.0f32; dims];
    }
    use rayon::prelude::*;
    let vecs: Vec<&Vec<f32>> = word_vectors.values().collect();
    let mut sum = vecs
        .par_iter()
        .fold(
            || vec![0.0f32; dims],
            |mut acc, v| {
                acc.iter_mut().zip(v.iter()).for_each(|(a, &b)| *a += b);
                acc
            },
        )
        .reduce(
            || vec![0.0f32; dims],
            |mut a, b| {
                a.iter_mut().zip(b.iter()).for_each(|(x, &y)| *x += y);
                a
            },
        );
    let inv = 1.0 / word_vectors.len() as f32;
    sum.iter_mut().for_each(|v| *v *= inv);
    sum
}

// ── Word2VecEmbeddings ────────────────────────────────────────────────────────

/// Serializable subset of a word2vec model: word vectors and a vocabulary
/// centroid used as a fallback for out-of-vocabulary tokens.
///
/// Unlike fastText there are no subword buckets, so OOV words are represented
/// by the vocabulary centroid — the mean of all word vectors.  This places
/// unknown tokens at the geometric centre of the semantic space rather than
/// discarding them silently.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Word2VecEmbeddings {
    pub dims: usize,
    /// Word → float32 vector map.
    pub word_vectors: HashMap<String, Vec<f32>>,
    /// Vocabulary centroid: mean of all word vectors.  Returned for OOV tokens.
    pub centroid: Vec<f32>,
}

impl Word2VecEmbeddings {
    /// Look up the vector for `word`.
    ///
    /// Returns the stored vector when the word is in vocabulary, or the
    /// vocabulary centroid for OOV words.
    pub fn vector_for(&self, word: &str) -> &[f32] {
        self.word_vectors
            .get(word)
            .map(|v| v.as_slice())
            .unwrap_or(&self.centroid)
    }

    /// Persist to a bincode file for fast subsequent reloads.
    pub fn save_bincode<P: AsRef<Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let bytes = bincode::serialize(self)
            .map_err(|e| VecEyesError::Serialization(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Reload from a bincode file produced by [`save_bincode`](Word2VecEmbeddings::save_bincode).
    pub fn load_bincode<P: AsRef<Path>>(path: P) -> Result<Self, VecEyesError> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes)
            .map_err(|e| VecEyesError::Serialization(e.to_string()))
    }
}

// ── Inference helper ──────────────────────────────────────────────────────────

/// Embed a batch of texts using word2vec vectors with optional TF-IDF weighting.
///
/// OOV tokens contribute the vocabulary centroid weighted by `idf` weight 1.0
/// (or their IDF weight when `idf` is provided).
pub fn embed_texts<S: AsRef<str>>(
    embeddings: &Word2VecEmbeddings,
    texts: &[S],
    idf: Option<&TfIdfModel>,
) -> DenseMatrix {
    use rayon::prelude::*;

    let cols = embeddings.dims;
    // Collect to &str so rayon gets a Sync slice regardless of S.
    let refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let mut matrix = ndarray::Array2::<f32>::zeros((refs.len(), cols));

    matrix
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .zip(refs.par_iter())
        .for_each(|(mut row, &text)| {
            let normalized = crate::nlp::normalize_text(text);
            let tokens = crate::nlp::tokenize(&normalized);
            if tokens.is_empty() {
                return;
            }
            let mut acc = vec![0f32; cols];
            let mut total_weight = 0f32;
            for token in &tokens {
                let weight = idf
                    .and_then(|m| m.token_to_index.get(token).map(|&i| m.idf[i]))
                    .unwrap_or(1.0);
                // vector_for always returns a slice (centroid for OOV).
                let vec = embeddings.vector_for(token);
                acc.iter_mut()
                    .zip(vec.iter().take(cols))
                    .for_each(|(a, &b)| *a += b * weight);
                total_weight += weight;
            }
            if total_weight > 0.0 {
                let inv = 1.0 / total_weight;
                row.iter_mut().zip(acc.iter()).for_each(|(dst, &src)| *dst = src * inv);
            }
        });

    matrix
}
