//! Unified external-embeddings type covering both fastText and word2vec models.
//!
//! [`ExternalEmbeddings`] is the single type accepted by every classifier's
//! `train_with_external_embeddings` method.  Construct it by wrapping either a
//! [`FastTextEmbeddings`] or a [`Word2VecEmbeddings`]:
//!
//! ```rust,ignore
//! // From a fastText .bin file
//! let emb = ExternalEmbeddings::FastText(
//!     FastTextBin::load("model.bin")?.extract_for_vocab(&vocab)
//! );
//!
//! // From a word2vec .bin file
//! let emb = ExternalEmbeddings::Word2Vec(
//!     Word2VecBin::load("vectors.bin")?.extract_for_vocab(&vocab)
//! );
//!
//! // Both are used the same way downstream
//! let classifier = KnnClassifier::train_with_external_embeddings(
//!     &samples, emb, DistanceMetric::Cosine, 5, None, false,
//! )?;
//! ```

use crate::nlp::fasttext_bin::FastTextEmbeddings;
use crate::nlp::word2vec_bin::Word2VecEmbeddings;
use crate::nlp::{DenseMatrix, TfIdfModel};

/// Unified external-embeddings wrapper.
///
/// Dispatches to the appropriate embedding function at inference time so the
/// rest of the classifier pipeline does not need to know which format was used.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExternalEmbeddings {
    /// Embeddings extracted from a fastText `.bin` file.
    /// Supports subword OOV composition via character n-gram bucket vectors.
    FastText(FastTextEmbeddings),
    /// Embeddings extracted from a Google word2vec `.bin` file.
    /// OOV tokens fall back to the vocabulary centroid.
    Word2Vec(Word2VecEmbeddings),
}

impl ExternalEmbeddings {
    /// Dimensionality of the stored embedding vectors.
    pub fn dims(&self) -> usize {
        match self {
            Self::FastText(e) => e.dims,
            Self::Word2Vec(e) => e.dims,
        }
    }

    /// Persist to a single bincode file for fast reloads.
    pub fn save_bincode<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), crate::error::VecEyesError> {
        let bytes = bincode::serialize(self)
            .map_err(|e| crate::error::VecEyesError::Serialization(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load from a bincode file produced by [`save_bincode`](ExternalEmbeddings::save_bincode).
    pub fn load_bincode<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, crate::error::VecEyesError> {
        let bytes = crate::security::read_file_limited(
            path.as_ref(),
            crate::security::DEFAULT_MAX_MODEL_BYTES,
            "ExternalEmbeddings::load_bincode",
        )?;
        let embeddings: Self = bincode::deserialize(&bytes)
            .map_err(|e| crate::error::VecEyesError::Serialization(e.to_string()))?;
        if embeddings.dims() == 0 {
            return Err(crate::error::VecEyesError::invalid_config(
                "ExternalEmbeddings::load_bincode",
                "embedding dimensions must be >= 1",
            ));
        }
        Ok(embeddings)
    }
}

/// Embed a batch of texts using whichever embedding format is stored.
///
/// Dispatches to [`fasttext_bin::embed_texts`] or [`word2vec_bin::embed_texts`]
/// respectively, so callers are decoupled from the concrete format.
pub fn embed_external<S: AsRef<str>>(
    embeddings: &ExternalEmbeddings,
    texts: &[S],
    idf: Option<&TfIdfModel>,
) -> DenseMatrix {
    match embeddings {
        ExternalEmbeddings::FastText(ft) => crate::nlp::fasttext_bin::embed_texts(ft, texts, idf),
        ExternalEmbeddings::Word2Vec(w2v) => crate::nlp::word2vec_bin::embed_texts(w2v, texts, idf),
    }
}
