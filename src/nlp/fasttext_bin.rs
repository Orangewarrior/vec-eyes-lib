use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::VecEyesError;
use crate::nlp::{DenseMatrix, TfIdfModel};

const FASTTEXT_MAGIC: i32 = 793712314;
const FASTTEXT_VERSION: i32 = 12;

// ── Low-level binary helpers ─────────────────────────────────────────────────

fn read_i32(r: &mut impl Read) -> Result<i32, VecEyesError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> Result<i64, VecEyesError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64, VecEyesError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_u8(r: &mut impl Read) -> Result<u8, VecEyesError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_cstring(r: &mut impl Read) -> Result<String, VecEyesError> {
    let mut bytes = Vec::new();
    loop {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        if b[0] == 0 {
            break;
        }
        bytes.push(b[0]);
    }
    String::from_utf8(bytes)
        .map_err(|e| VecEyesError::invalid_config("FastTextBin::read_cstring", e.to_string()))
}

fn read_f32_chunk(r: &mut impl Read, buf: &mut [f32]) -> Result<(), VecEyesError> {
    const CHUNK: usize = 8192;
    let mut tmp = [0u8; CHUNK * 4];
    let mut offset = 0;
    while offset < buf.len() {
        let n = (buf.len() - offset).min(CHUNK);
        r.read_exact(&mut tmp[..n * 4])?;
        for i in 0..n {
            buf[offset + i] = f32::from_le_bytes([
                tmp[i * 4],
                tmp[i * 4 + 1],
                tmp[i * 4 + 2],
                tmp[i * 4 + 3],
            ]);
        }
        offset += n;
    }
    Ok(())
}

// ── FNV-1a hash matching fastText C++ exactly ────────────────────────────────

fn fnv1a(bytes: &[u8]) -> u32 {
    const PRIME: u32 = 16777619;
    const OFFSET: u32 = 2166136261;
    let mut h = OFFSET;
    for &b in bytes {
        // Sign-extend byte → i8 → i32 → u32: matches fastText's uint32_t(int8_t(c))
        h ^= b as i8 as i32 as u32;
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ── Subword character n-gram hashes ─────────────────────────────────────────

/// Compute subword bucket indices for `word`, matching fastText's algorithm exactly.
///
/// `word` is wrapped in `<` / `>` before hashing.  Single boundary characters
/// (`<` alone or `>` alone) are skipped.  Hashes are offset by `nwords` so they
/// index directly into the full embedding matrix.
pub(crate) fn subword_hashes(
    word: &str,
    minn: usize,
    maxn: usize,
    bucket: usize,
    nwords: usize,
) -> Vec<usize> {
    if minn == 0 || maxn == 0 || bucket == 0 {
        return Vec::new();
    }
    let padded = format!("<{}>", word);
    let bytes = padded.as_bytes();
    let len = bytes.len();
    let mut hashes = Vec::new();
    let mut i = 0;
    while i < len {
        // Skip UTF-8 continuation bytes in outer loop (only visit codepoint starts)
        if (bytes[i] & 0xC0) == 0x80 {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut n = 0usize;
        while j < len && n < maxn {
            // Advance j past one codepoint
            j += 1;
            while j < len && (bytes[j] & 0xC0) == 0x80 {
                j += 1;
            }
            n += 1;
            // Skip n==1 ngrams at the word boundaries (stand-alone '<' or '>')
            if n >= minn && !(n == 1 && (i == 0 || j == len)) {
                let h = (fnv1a(&bytes[i..j]) as usize) % bucket;
                hashes.push(h + nwords);
            }
        }
        i += 1;
    }
    hashes
}

// ── FastTextBin ─────────────────────────────────────────────────────────────

/// Temporary loader for a fastText `.bin` model (v12, non-quantized).
///
/// Call [`extract_all`](FastTextBin::extract_all) or
/// [`extract_for_vocab`](FastTextBin::extract_for_vocab) to obtain a
/// serializable [`FastTextEmbeddings`] suitable for save/load with bincode.
pub struct FastTextBin {
    pub dims: usize,
    pub minn: usize,
    pub maxn: usize,
    pub(crate) bucket: usize,
    pub(crate) nwords: usize,
    word_index: HashMap<String, usize>,
    matrix: Vec<f32>,
    total_rows: usize,
}

impl FastTextBin {
    /// Load a fastText binary model from `path`.
    ///
    /// Only the **input** matrix (word vectors + subword bucket vectors) is
    /// loaded; the output matrix is not needed for embedding inference.
    /// Quantized models (`.ftz`) are not supported.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, VecEyesError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mut r = BufReader::new(file);

        let magic = read_i32(&mut r)?;
        if magic != FASTTEXT_MAGIC {
            return Err(VecEyesError::invalid_config(
                "FastTextBin::load",
                format!(
                    "bad magic 0x{:x} — not a fastText v12 .bin file",
                    magic
                ),
            ));
        }
        let version = read_i32(&mut r)?;
        if version != FASTTEXT_VERSION {
            return Err(VecEyesError::invalid_config(
                "FastTextBin::load",
                format!(
                    "unsupported fastText version {version} (expected {FASTTEXT_VERSION})"
                ),
            ));
        }

        // Args struct (order matches fastText C++ Args::save)
        let _lr = read_f64(&mut r)?;
        let dim = read_i32(&mut r)? as usize;
        let _ws = read_i32(&mut r)?;
        let _epoch = read_i32(&mut r)?;
        let _min_count = read_i32(&mut r)?;
        let _neg = read_i32(&mut r)?;
        let _word_ngrams = read_i32(&mut r)?;
        let _loss = read_i32(&mut r)?;
        let _model = read_i32(&mut r)?;
        let bucket = read_i32(&mut r)? as usize;
        let minn = read_i32(&mut r)? as usize;
        let maxn = read_i32(&mut r)? as usize;
        let _lr_update = read_i32(&mut r)?;
        let _t = read_f64(&mut r)?;

        // Dictionary header
        let dict_size = read_i32(&mut r)? as usize;
        let nwords = read_i32(&mut r)? as usize;
        let _nlabels = read_i32(&mut r)?;
        let _ntokens = read_i64(&mut r)?;
        let pruneidx_size = read_i64(&mut r)?;

        // Dictionary entries (null-terminated strings + metadata)
        let mut word_index: HashMap<String, usize> = HashMap::with_capacity(nwords);
        for i in 0..dict_size {
            let word = read_cstring(&mut r)?;
            let _count = read_i64(&mut r)?;
            let entry_type = read_u8(&mut r)?; // 0 = WORD, 1 = LABEL
            if pruneidx_size >= 0 && entry_type == 0 {
                let _pruneidx = read_i32(&mut r)?;
            }
            if entry_type == 0 {
                word_index.insert(word, i);
            }
        }

        // Quantization flag
        let quant = read_u8(&mut r)?;
        if quant != 0 {
            return Err(VecEyesError::unsupported(
                "FastTextBin::load",
                "quantized fastText models (.ftz) are not supported — use the full .bin model",
            ));
        }

        // Input matrix: rows = nwords + bucket, cols = dim
        let mat_rows = read_i64(&mut r)? as usize;
        let mat_cols = read_i64(&mut r)? as usize;
        if mat_cols != dim {
            return Err(VecEyesError::invalid_config(
                "FastTextBin::load",
                format!("matrix cols {mat_cols} != args dim {dim}"),
            ));
        }

        let mut matrix = vec![0f32; mat_rows * mat_cols];
        read_f32_chunk(&mut r, &mut matrix)?;

        Ok(Self {
            dims: dim,
            minn,
            maxn,
            bucket,
            nwords,
            word_index,
            matrix,
            total_rows: mat_rows,
        })
    }

    fn row(&self, idx: usize) -> &[f32] {
        let s = idx * self.dims;
        &self.matrix[s..s + self.dims]
    }

    /// Extract the complete model (all word vectors + all bucket vectors) into a
    /// serializable [`FastTextEmbeddings`].  For models with millions of buckets
    /// this allocates substantial memory — prefer [`extract_for_vocab`] when only
    /// a known vocabulary is needed.
    pub fn extract_all(&self) -> FastTextEmbeddings {
        let mut word_vectors: HashMap<String, Vec<f32>> =
            HashMap::with_capacity(self.nwords);
        for (word, &idx) in &self.word_index {
            if idx < self.total_rows {
                word_vectors.insert(word.clone(), self.row(idx).to_vec());
            }
        }

        let bucket_start = self.nwords;
        let bucket_end = (bucket_start + self.bucket).min(self.total_rows);
        let actual = bucket_end - bucket_start;
        let mut bucket_vectors: HashMap<usize, Vec<f32>> = HashMap::with_capacity(actual);
        for i in 0..actual {
            bucket_vectors.insert(i, self.row(bucket_start + i).to_vec());
        }

        FastTextEmbeddings {
            dims: self.dims,
            minn: self.minn,
            maxn: self.maxn,
            bucket_size: self.bucket,
            nwords: self.nwords,
            word_vectors,
            bucket_vectors,
        }
    }

    /// Extract word vectors for `vocab` plus only the subword bucket vectors
    /// required for OOV composition — produces a much smaller model than
    /// [`extract_all`](FastTextBin::extract_all).
    pub fn extract_for_vocab(&self, vocab: &[&str]) -> FastTextEmbeddings {
        let mut word_vectors: HashMap<String, Vec<f32>> =
            HashMap::with_capacity(vocab.len());
        let mut needed: HashSet<usize> = HashSet::new();

        for &word in vocab {
            if let Some(&idx) = self.word_index.get(word) {
                if idx < self.total_rows {
                    word_vectors.insert(word.to_string(), self.row(idx).to_vec());
                }
            } else {
                for h in
                    subword_hashes(word, self.minn, self.maxn, self.bucket, self.nwords)
                {
                    needed.insert(h - self.nwords);
                }
            }
        }

        let mut bucket_vectors: HashMap<usize, Vec<f32>> =
            HashMap::with_capacity(needed.len());
        for bi in needed {
            let row_idx = self.nwords + bi;
            if row_idx < self.total_rows {
                bucket_vectors.insert(bi, self.row(row_idx).to_vec());
            }
        }

        FastTextEmbeddings {
            dims: self.dims,
            minn: self.minn,
            maxn: self.maxn,
            bucket_size: self.bucket,
            nwords: self.nwords,
            word_vectors,
            bucket_vectors,
        }
    }
}

// ── FastTextEmbeddings ───────────────────────────────────────────────────────

/// Serializable, inference-ready fastText embedding model.
///
/// Obtained by calling [`FastTextBin::extract_all`] or
/// [`FastTextBin::extract_for_vocab`], then serialized with bincode for fast
/// reloading without re-parsing the original `.bin` file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FastTextEmbeddings {
    pub dims: usize,
    pub minn: usize,
    pub maxn: usize,
    pub(crate) bucket_size: usize,
    pub(crate) nwords: usize,
    word_vectors: HashMap<String, Vec<f32>>,
    /// Sparse bucket vectors keyed by bucket index (0..bucket_size).
    bucket_vectors: HashMap<usize, Vec<f32>>,
}

impl FastTextEmbeddings {
    /// Return the embedding vector for `word`.
    ///
    /// For in-vocabulary words, returns the stored vector directly.  For OOV
    /// words, computes the vector by averaging subword character n-gram bucket
    /// vectors (real subword composition, not a zero fallback).  Returns `None`
    /// only when no subword information is available.
    pub fn vector_for(&self, word: &str) -> Option<Vec<f32>> {
        if let Some(v) = self.word_vectors.get(word) {
            return Some(v.clone());
        }
        if self.maxn == 0 || self.bucket_size == 0 {
            return None;
        }
        let hashes =
            subword_hashes(word, self.minn, self.maxn, self.bucket_size, self.nwords);
        if hashes.is_empty() {
            return None;
        }
        let mut vec = vec![0f32; self.dims];
        let mut count = 0usize;
        for h in hashes {
            let bi = h - self.nwords;
            if let Some(bv) = self.bucket_vectors.get(&bi) {
                for (i, &v) in bv.iter().enumerate().take(self.dims) {
                    vec[i] += v;
                }
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let n = count as f32;
        for v in &mut vec {
            *v /= n;
        }
        Some(vec)
    }

    /// Number of in-vocabulary words stored in this model.
    pub fn vocab_size(&self) -> usize {
        self.word_vectors.len()
    }

    /// Returns `true` if `word` has an exact in-vocabulary vector.
    pub fn contains(&self, word: &str) -> bool {
        self.word_vectors.contains_key(word)
    }

    /// Save to a bincode file for fast reloading.
    pub fn save_bincode<P: AsRef<Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let bytes = bincode::serialize(self)
            .map_err(|e| VecEyesError::Serialization(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load from a bincode file previously written by [`save_bincode`](FastTextEmbeddings::save_bincode).
    pub fn load_bincode<P: AsRef<Path>>(path: P) -> Result<Self, VecEyesError> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes)
            .map_err(|e| VecEyesError::Serialization(e.to_string()))
    }
}

// ── Text embedding helper ────────────────────────────────────────────────────

/// Embed a slice of texts using [`FastTextEmbeddings`], optionally weighted by
/// a TF-IDF model fitted on the training corpus.
///
/// Each text is tokenized, each token is resolved to a vector (with subword OOV
/// fallback), weighted by its IDF score, and averaged into one row.
pub fn embed_texts<S: AsRef<str>>(
    embeddings: &FastTextEmbeddings,
    texts: &[S],
    idf: Option<&TfIdfModel>,
) -> DenseMatrix {
    let rows = texts.len();
    let cols = embeddings.dims;
    let mut matrix = ndarray::Array2::<f32>::zeros((rows, cols));
    for (row, text) in texts.iter().enumerate() {
        let normalized = crate::nlp::normalize_text(text.as_ref());
        let tokens = crate::nlp::tokenize(&normalized);
        if tokens.is_empty() {
            continue;
        }
        let mut row_vec = vec![0f32; cols];
        let mut total_weight = 0f32;
        for token in &tokens {
            let weight = idf
                .and_then(|m| m.token_to_index.get(token).map(|&i| m.idf[i]))
                .unwrap_or(1.0);
            if let Some(vec) = embeddings.vector_for(token) {
                for (i, &v) in vec.iter().enumerate().take(cols) {
                    row_vec[i] += v * weight;
                }
                total_weight += weight;
            }
        }
        if total_weight > 0.0 {
            for (col, &v) in row_vec.iter().enumerate() {
                matrix[[row, col]] = v / total_weight;
            }
        }
    }
    matrix
}
