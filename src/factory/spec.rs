//! Typed `ClassifierSpec` factory — eliminates runtime `require_*_config` checks.
//!
//! Each factory method returns a method-specific builder that only exposes
//! the relevant hyper-parameters.  The method type is known at compile time,
//! so no `MethodKind` enum or defensive runtime validation is needed.
//!
//! # Example
//! ```rust,ignore
//! use vec_eyes_lib::factory::ClassifierSpec;
//! use vec_eyes_lib::advanced_models::RandomForestConfig;
//! use vec_eyes_lib::nlp::NlpOption;
//!
//! let classifier = ClassifierSpec::random_forest(RandomForestConfig::default())
//!     .nlp(NlpOption::FastText)
//!     .hot_label(ClassificationLabel::Spam)
//!     .training_data("hot/", "cold/")
//!     .build()?;
//! ```

use std::path::Path;

use crate::advanced_models::{
    AdvancedClassifier, AdvancedMethod, AdvancedModelConfig, GradientBoostingConfig,
    IsolationForestConfig, LogisticRegressionConfig, RandomForestConfig, SvmConfig,
};
use crate::classifier::Classifier;
use crate::classifiers::bayes::BayesClassifier;
use crate::classifiers::knn::{DistanceMetric, KnnClassifier};
use crate::dataset::{load_training_samples, TrainingSample};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::NlpOption;

/// Entry point for the typed classifier factory.
pub struct ClassifierSpec;

impl ClassifierSpec {
    pub fn bayes() -> BayesSpec { BayesSpec::new() }
    pub fn knn_cosine(k: usize) -> KnnSpec { KnnSpec::new(DistanceMetric::Cosine, k, None) }
    pub fn knn_euclidean(k: usize) -> KnnSpec { KnnSpec::new(DistanceMetric::Euclidean, k, None) }
    pub fn knn_manhattan(k: usize) -> KnnSpec { KnnSpec::new(DistanceMetric::Manhattan, k, None) }
    pub fn knn_minkowski(k: usize, p: f32) -> KnnSpec { KnnSpec::new(DistanceMetric::Minkowski(p), k, Some(p)) }
    pub fn logistic_regression(cfg: LogisticRegressionConfig) -> AdvancedSpec {
        AdvancedSpec::new(AdvancedMethod::LogisticRegression, AdvancedModelConfig { logistic: Some(cfg), ..Default::default() })
    }
    pub fn random_forest(cfg: RandomForestConfig) -> AdvancedSpec {
        AdvancedSpec::new(AdvancedMethod::RandomForest, AdvancedModelConfig { random_forest: Some(cfg), ..Default::default() })
    }
    pub fn svm(cfg: SvmConfig) -> AdvancedSpec {
        AdvancedSpec::new(AdvancedMethod::Svm, AdvancedModelConfig { svm: Some(cfg), ..Default::default() })
    }
    pub fn gradient_boosting(cfg: GradientBoostingConfig) -> AdvancedSpec {
        AdvancedSpec::new(AdvancedMethod::GradientBoosting, AdvancedModelConfig { gradient_boosting: Some(cfg), ..Default::default() })
    }
    pub fn isolation_forest(cfg: IsolationForestConfig) -> AdvancedSpec {
        AdvancedSpec::new(AdvancedMethod::IsolationForest, AdvancedModelConfig { isolation_forest: Some(cfg), ..Default::default() })
    }
}

// ── Shared training-data builder ─────────────────────────────────────────────

struct TrainingData {
    samples: Option<Vec<TrainingSample>>,
    hot_path: Option<std::path::PathBuf>,
    cold_path: Option<std::path::PathBuf>,
    hot_label: ClassificationLabel,
    cold_label: ClassificationLabel,
    recursive: bool,
}

impl TrainingData {
    fn new() -> Self {
        Self {
            samples: None,
            hot_path: None,
            cold_path: None,
            hot_label: ClassificationLabel::WebAttack,
            cold_label: ClassificationLabel::RawData,
            recursive: true,
        }
    }

    fn resolve(self) -> Result<Vec<TrainingSample>, VecEyesError> {
        if let Some(samples) = self.samples { return Ok(samples); }
        let hot = self.hot_path.ok_or_else(|| VecEyesError::invalid_config("ClassifierSpec", "hot_path or samples() is required"))?;
        let cold = self.cold_path.ok_or_else(|| VecEyesError::invalid_config("ClassifierSpec", "cold_path or samples() is required"))?;
        let mut s = load_training_samples(&hot, self.hot_label, self.recursive)?;
        s.extend(load_training_samples(&cold, self.cold_label, self.recursive)?);
        Ok(s)
    }
}

// ── BayesSpec ─────────────────────────────────────────────────────────────────

pub struct BayesSpec {
    nlp: NlpOption,
    threads: Option<usize>,
    data: TrainingData,
}

impl BayesSpec {
    fn new() -> Self { Self { nlp: NlpOption::Count, threads: None, data: TrainingData::new() } }

    pub fn nlp(mut self, nlp: NlpOption) -> Self { self.nlp = nlp; self }
    pub fn threads(mut self, t: usize) -> Self { self.threads = Some(t); self }
    pub fn hot_label(mut self, l: ClassificationLabel) -> Self { self.data.hot_label = l; self }
    pub fn cold_label(mut self, l: ClassificationLabel) -> Self { self.data.cold_label = l; self }
    pub fn recursive(mut self, r: bool) -> Self { self.data.recursive = r; self }
    pub fn samples(mut self, s: Vec<TrainingSample>) -> Self { self.data.samples = Some(s); self }
    pub fn training_data<P: AsRef<Path>, Q: AsRef<Path>>(mut self, hot: P, cold: Q) -> Self {
        self.data.hot_path = Some(hot.as_ref().to_path_buf());
        self.data.cold_path = Some(cold.as_ref().to_path_buf());
        self
    }

    pub fn build(self) -> Result<BayesClassifier, VecEyesError> {
        let samples = self.data.resolve()?;
        BayesClassifier::train(&samples, self.nlp, self.threads)
    }

    pub fn build_boxed(self) -> Result<Box<dyn Classifier>, VecEyesError> {
        Ok(Box::new(self.build()?))
    }
}

// ── KnnSpec ───────────────────────────────────────────────────────────────────

pub struct KnnSpec {
    metric: DistanceMetric,
    k: usize,
    nlp: NlpOption,
    dims: usize,
    threads: Option<usize>,
    normalize: bool,
    data: TrainingData,
}

impl KnnSpec {
    fn new(metric: DistanceMetric, k: usize, _p: Option<f32>) -> Self {
        Self { metric, k, nlp: NlpOption::Word2Vec, dims: 32, threads: None, normalize: false, data: TrainingData::new() }
    }

    pub fn nlp(mut self, nlp: NlpOption) -> Self { self.nlp = nlp; self }
    pub fn dims(mut self, d: usize) -> Self { self.dims = d; self }
    pub fn threads(mut self, t: usize) -> Self { self.threads = Some(t); self }
    pub fn normalize_features(mut self, n: bool) -> Self { self.normalize = n; self }
    pub fn hot_label(mut self, l: ClassificationLabel) -> Self { self.data.hot_label = l; self }
    pub fn cold_label(mut self, l: ClassificationLabel) -> Self { self.data.cold_label = l; self }
    pub fn recursive(mut self, r: bool) -> Self { self.data.recursive = r; self }
    pub fn samples(mut self, s: Vec<TrainingSample>) -> Self { self.data.samples = Some(s); self }
    pub fn training_data<P: AsRef<Path>, Q: AsRef<Path>>(mut self, hot: P, cold: Q) -> Self {
        self.data.hot_path = Some(hot.as_ref().to_path_buf());
        self.data.cold_path = Some(cold.as_ref().to_path_buf());
        self
    }

    pub fn build(self) -> Result<KnnClassifier, VecEyesError> {
        let samples = self.data.resolve()?;
        KnnClassifier::train(&samples, self.nlp, self.metric, self.dims, self.k, self.threads, self.normalize)
    }

    pub fn build_boxed(self) -> Result<Box<dyn Classifier>, VecEyesError> {
        Ok(Box::new(self.build()?))
    }
}

// ── AdvancedSpec ─────────────────────────────────────────────────────────────

pub struct AdvancedSpec {
    method: AdvancedMethod,
    config: AdvancedModelConfig,
    nlp: NlpOption,
    data: TrainingData,
}

impl AdvancedSpec {
    fn new(method: AdvancedMethod, config: AdvancedModelConfig) -> Self {
        Self { method, config, nlp: NlpOption::Word2Vec, data: TrainingData::new() }
    }

    pub fn nlp(mut self, nlp: NlpOption) -> Self { self.nlp = nlp; self }
    pub fn threads(mut self, t: usize) -> Self { self.config.threads = Some(t); self }
    pub fn embedding_dims(mut self, d: usize) -> Self { self.config.embedding_dimensions = Some(d.max(1)); self }
    pub fn hot_label(mut self, l: ClassificationLabel) -> Self { self.data.hot_label = l; self }
    pub fn cold_label(mut self, l: ClassificationLabel) -> Self { self.data.cold_label = l; self }
    pub fn recursive(mut self, r: bool) -> Self { self.data.recursive = r; self }
    pub fn samples(mut self, s: Vec<TrainingSample>) -> Self { self.data.samples = Some(s); self }
    pub fn training_data<P: AsRef<Path>, Q: AsRef<Path>>(mut self, hot: P, cold: Q) -> Self {
        self.data.hot_path = Some(hot.as_ref().to_path_buf());
        self.data.cold_path = Some(cold.as_ref().to_path_buf());
        self
    }

    pub fn build(self) -> Result<AdvancedClassifier, VecEyesError> {
        let hot_label = self.data.hot_label.clone();
        let cold_label = self.data.cold_label.clone();
        let samples = self.data.resolve()?;
        AdvancedClassifier::train(self.method, &samples, self.nlp, hot_label, cold_label, &self.config)
    }

    pub fn build_boxed(self) -> Result<Box<dyn Classifier>, VecEyesError> {
        Ok(Box::new(self.build()?))
    }
}
