pub(crate) mod core;
mod distance;

use crate::builders::Builder;
use crate::classifier::{ClassificationResult, Classifier, ClassifierBuilder, ClassifierFactory, ClassifierMethod};
use crate::config::ScoreSumMode;
use crate::dataset::{load_training_samples, TrainingSample};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::matcher::{RuleMatcher, ScoringEngine};
use crate::nlp::{dense_matrix_from_texts, DenseMatrix, NlpOption, WordEmbeddingModel};
use std::path::Path;

/// Entry point for KNN-based classifiers.
pub struct KnnModule;

impl KnnModule {
    #[inline]
    pub fn builder() -> KnnBuilder {
        KnnBuilder::new()
    }

    #[inline]
    pub fn cosine() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnCosine)
    }

    #[inline]
    pub fn euclidean() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnEuclidean)
    }

    #[inline]
    pub fn manhattan() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnManhattan)
    }

    #[inline]
    pub fn minkowski() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnMinkowski)
    }
}

pub use distance::DistanceMetric;
pub(crate) use distance::{euclidean_distance_squared, manhattan_distance, minkowski_distance};

#[derive(Debug, Clone)]
pub struct KnnBuilder {
    nlp: NlpOption,
    samples: Vec<TrainingSample>,
    metric: DistanceMetric,
    dims: usize,
    k: Option<usize>,
    p: Option<f32>,
    threads: Option<usize>,
    normalize_features: bool,
}

impl Builder<KnnClassifier> for KnnBuilder {
    fn new() -> Self {
        Self {
            nlp: NlpOption::Word2Vec,
            samples: Vec::new(),
            metric: DistanceMetric::Cosine,
            dims: 32,
            k: None,
            p: None,
            threads: None,
            normalize_features: false,
        }
    }

    fn build(self) -> Result<KnnClassifier, VecEyesError> {
        let k = crate::classifier::require_k(self.k)?;
        if let DistanceMetric::Minkowski(_) = self.metric {
            crate::classifier::require_p(self.p)?;
        }
        KnnClassifier::train(&self.samples, self.nlp, self.metric, self.dims, k, self.threads, self.normalize_features)
    }
}

impl KnnBuilder {

pub fn new() -> Self {
    Self {
        nlp: NlpOption::Word2Vec,
        samples: Vec::new(),
        metric: DistanceMetric::Cosine,
        dims: 32,
        k: None,
        p: None,
        threads: None,
        normalize_features: false,
    }
}

pub fn build(self) -> Result<KnnClassifier, VecEyesError> {
    <Self as Builder<KnnClassifier>>::build(self)
}

    pub fn nlp(mut self, nlp: NlpOption) -> Self {
        self.nlp = nlp;
        self
    }

    pub fn samples(mut self, samples: Vec<TrainingSample>) -> Self {
        self.samples = samples;
        self
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    pub fn p(mut self, p: f32) -> Self {
        self.p = Some(p);
        self
    }

    pub fn threads(mut self, threads: Option<usize>) -> Self {
        self.threads = threads;
        self
    }

    pub fn normalize_features(mut self, normalize_features: bool) -> Self {
        self.normalize_features = normalize_features;
        self
    }

    pub fn cosine(mut self) -> Self {
        self.metric = DistanceMetric::Cosine;
        self
    }

    pub fn euclidean(mut self) -> Self {
        self.metric = DistanceMetric::Euclidean;
        self
    }

    pub fn manhattan(mut self) -> Self {
        self.metric = DistanceMetric::Manhattan;
        self
    }

    pub fn minkowski(mut self, p: f32) -> Self {
        self.metric = DistanceMetric::Minkowski(p);
        self.p = Some(p);
        self
    }

    pub fn pipeline(mut self, pipeline: crate::compat::NlpPipeline) -> Self {
        self.nlp = match pipeline.representation {
            crate::compat::RepresentationKind::Count => NlpOption::Count,
            crate::compat::RepresentationKind::TfIdf => NlpOption::TfIdf,
            crate::compat::RepresentationKind::Word2Vec => NlpOption::Word2Vec,
            crate::compat::RepresentationKind::FastText => NlpOption::FastText,
        };
        if let Some(cfg) = pipeline.fasttext_config {
            self.dims = cfg.dimensions.max(1);
        }
        self
    }

    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric.clone();
        if let DistanceMetric::Minkowski(p) = metric {
            self.p = Some(p);
        }
        self
    }

    pub fn fit_from_directories<P: AsRef<Path>>(
        mut self,
        hot_path: Option<P>,
        cold_path: Option<P>,
        hot_label: ClassificationLabel,
    ) -> Result<KnnClassifier, VecEyesError> {
        let hot_root = hot_path.ok_or_else(|| VecEyesError::invalid_config("classifier::KnnBuilder::fit_from_directories", "hot directory is required"))?;
        let cold_root = cold_path.ok_or_else(|| VecEyesError::invalid_config("classifier::KnnBuilder::fit_from_directories", "cold directory is required"))?;
        let mut samples = load_training_samples(hot_root.as_ref(), hot_label, true)?;
        samples.extend(load_training_samples(
            cold_root.as_ref(),
            ClassificationLabel::BlockList,
            true,
        )?);
        self.samples = samples;
        self.build()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DenseFeatureModel {
    Word2Vec(WordEmbeddingModel),
    FastText(WordEmbeddingModel),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnnClassifier {
    metric: DistanceMetric,
    threads: Option<usize>,
    labels: Vec<ClassificationLabel>,
    matrix: DenseMatrix,
    model: DenseFeatureModel,
    k: usize,
    normalize_features: bool,
    feature_mean: Option<Vec<f32>>,
    feature_std: Option<Vec<f32>>,
}

impl KnnClassifier {
    pub(crate) fn from_parts(
        metric: DistanceMetric,
        threads: Option<usize>,
        labels: Vec<ClassificationLabel>,
        matrix: DenseMatrix,
        model: DenseFeatureModel,
        k: usize,
        normalize_features: bool,
        feature_mean: Option<Vec<f32>>,
        feature_std: Option<Vec<f32>>,
    ) -> Self {
        Self { metric, threads, labels, matrix, model, k, normalize_features, feature_mean, feature_std }
    }

    pub(crate) fn metric(&self) -> &DistanceMetric { &self.metric }
    pub(crate) fn threads(&self) -> Option<usize> { self.threads }
    pub(crate) fn labels(&self) -> &Vec<ClassificationLabel> { &self.labels }
    pub(crate) fn matrix(&self) -> &DenseMatrix { &self.matrix }
    pub(crate) fn k(&self) -> usize { self.k }

    pub fn train(
        samples: &[TrainingSample],
        nlp: NlpOption,
        metric: DistanceMetric,
        dims: usize,
        k: usize,
        threads: Option<usize>,
        normalize_features: bool,
    ) -> Result<Self, VecEyesError> {
        core::train(samples, nlp, metric, dims, k, threads, normalize_features)
    }

    /// Persist the trained model to a JSON file.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let json = serde_json::to_string(self)
            .map_err(|e| VecEyesError::invalid_config("KnnClassifier::save", e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a previously saved model from a JSON file.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VecEyesError> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| VecEyesError::invalid_config("KnnClassifier::load", e.to_string()))
    }

    pub(crate) fn matrix_for_text(&self, text: &str) -> DenseMatrix {
        let texts = [text];
        match &self.model {
            DenseFeatureModel::Word2Vec(inner) => { let mut m = dense_matrix_from_texts(inner, &texts); self.apply_feature_normalization(&mut m); m },
            DenseFeatureModel::FastText(inner) => { let mut m = dense_matrix_from_texts(inner, &texts); self.apply_feature_normalization(&mut m); m },
        }
    }

    fn score_neighbors(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        core::score_neighbors(self, text)
    }

    fn apply_feature_normalization(&self, matrix: &mut DenseMatrix) {
        if !self.normalize_features { return; }
        if let (Some(mean), Some(std)) = (&self.feature_mean, &self.feature_std) {
            for row in 0..matrix.shape()[0] {
                for col in 0..matrix.shape()[1] {
                    matrix[[row, col]] = (matrix[[row, col]] - mean[col]) / std[col].max(1e-6);
                }
            }
        }
    }
}


impl Classifier for KnnClassifier {
    fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult {
        let mut labels = self.score_neighbors(text);
        let hits = if score_sum_mode.is_on() {
            let (boost, hits) = ScoringEngine::compute_rule_boost(text, matchers);
            for (_, score) in &mut labels {
                *score = ScoringEngine::merge_scores(*score, boost, score_sum_mode);
            }
            hits
        } else {
            matchers.iter().flat_map(|m| m.find_matches(text)).collect()
        };
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ClassificationResult { labels, extra_hits: hits }
    }
}

impl From<KnnClassifier> for Box<dyn Classifier> {
    fn from(value: KnnClassifier) -> Self {
        Box::new(value)
    }
}
