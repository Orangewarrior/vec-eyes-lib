//! Factory types for selecting a classifier and CLI method.

use crate::bayes::BayesBuilder;
use crate::error::{VecEyesError, VecEyesResult};
use crate::knn::{DistanceMetric, KnnBuilder};
use crate::nlp::RepresentationKind;

#[derive(Debug, Clone, Copy)]
pub enum ClassifierKind {
    NaiveBayes,
    KNN,
}

#[derive(Debug, Clone)]
pub enum ClassifierBuilder {
    NaiveBayes(BayesBuilder),
    KNN(KnnBuilder),
}

pub struct ClassifierFactory;

impl ClassifierFactory {
    pub fn builder(kind: ClassifierKind, representation: RepresentationKind) -> VecEyesResult<ClassifierBuilder> {
        match kind {
            ClassifierKind::NaiveBayes => Ok(ClassifierBuilder::NaiveBayes(BayesBuilder::new())),
            ClassifierKind::KNN => {
                let metric = match representation {
                    RepresentationKind::FastText | RepresentationKind::Word2Vec => DistanceMetric::Cosine,
                    _ => DistanceMetric::Cosine,
                };
                Ok(ClassifierBuilder::KNN(KnnBuilder::new().metric(metric)))
            }
        }
    }
}

pub struct MethodFactory;

impl MethodFactory {
    pub fn parse(method: &str) -> VecEyesResult<(ClassifierKind, DistanceMetric)> {
        let normalized = method.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "bayes-count" | "bayes-tfidf" => Ok((ClassifierKind::NaiveBayes, DistanceMetric::Cosine)),
            "knn-cosine" => Ok((ClassifierKind::KNN, DistanceMetric::Cosine)),
            "knn-euclidean" => Ok((ClassifierKind::KNN, DistanceMetric::Euclidean)),
            "knn-manhattan" => Ok((ClassifierKind::KNN, DistanceMetric::Manhattan)),
            "knn-minkowski" => Ok((ClassifierKind::KNN, DistanceMetric::Minkowski { p: 3.0 })),
            _ => Err(VecEyesError::UnsupportedMethod(method.to_string())),
        }
    }

    pub fn representation_for(method: &str, nlp_opt: &str) -> VecEyesResult<RepresentationKind> {
        let method = method.trim().to_ascii_lowercase();
        let nlp_opt = nlp_opt.trim().to_ascii_lowercase();
        if method.starts_with("bayes") {
            if method.ends_with("tfidf") || nlp_opt == "tfidf" {
                Ok(RepresentationKind::TfIdf)
            } else {
                Ok(RepresentationKind::Count)
            }
        } else {
            match nlp_opt.as_str() {
                "fasttext" => Ok(RepresentationKind::FastText),
                _ => Ok(RepresentationKind::Word2Vec),
            }
        }
    }
}
