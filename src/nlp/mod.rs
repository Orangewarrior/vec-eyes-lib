pub mod normalizer;
pub mod tokenizer;
pub mod feature_extractor;

pub use feature_extractor::{
    dense_matrix_from_texts, fit_tfidf, transform_tfidf, DenseMatrix, FastTextConfig,
    FastTextConfigBuilder, TfIdfModel, WordEmbeddingModel,
};
pub use normalizer::normalize_text;
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
