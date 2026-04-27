pub mod normalizer;
pub mod tokenizer;
pub mod feature_extractor;
pub mod fasttext_bin;
pub mod word2vec_bin;
pub mod external_embeddings;

pub use fasttext_bin::{FastTextBin, FastTextEmbeddings};
pub use word2vec_bin::{Word2VecBin, Word2VecEmbeddings};
pub use external_embeddings::ExternalEmbeddings;

pub use feature_extractor::{
    dense_matrix_from_texts, dense_matrix_from_texts_with_tfidf, fit_tfidf, fit_tfidf_with_config, transform_tfidf, DenseMatrix, FastTextConfig,
    FastTextConfigBuilder, TfIdfModel, WordEmbeddingModel,
};
pub use normalizer::{decode_obfuscated_text, normalize_text, normalize_text_with_options, set_security_normalization_enabled, SecurityNormalizationOptions};
pub use tokenizer::tokenize;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NlpOption {
    #[serde(alias = "count", alias = "Count")]
    Count,
    #[serde(alias = "tf-idf", alias = "TF-IDF", alias = "Tfidf", alias = "tfidf")]
    TfIdf,
    #[serde(alias = "word2vec", alias = "Word2Vec")]
    Word2Vec,
    #[serde(alias = "fasttext", alias = "FastText")]
    FastText,
}
