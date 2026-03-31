use crate::advanced_models::{
    AdvancedClassifier, AdvancedMethod, AdvancedModelConfig, GradientBoostingConfig,
    IsolationForestConfig, LogisticRegressionConfig, RandomForestConfig, RandomForestMaxFeatures,
    RandomForestMode, SvmConfig, SvmKernel,
};
use crate::config::{RulesFile, ScoreSumMode};
use crate::dataset::{load_training_samples, read_text_file, TrainingSample};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::matcher::{AlertHit, RuleMatcher, ScoringEngine};
use crate::parallel::install_pool;
use crate::nlp::{
    dense_matrix_from_texts, fit_tfidf, transform_tfidf, DenseMatrix, FastTextConfigBuilder,
    NlpOption, TfIdfModel, WordEmbeddingModel,
};
use chrono::Utc;
use ndarray::Axis;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MethodKind {
    #[serde(alias = "bayes", alias = "Bayes")]
    Bayes,
    #[serde(alias = "knn-cosine", alias = "KnnCosine")]
    KnnCosine,
    #[serde(alias = "knn-euclidean", alias = "KnnEuclidean")]
    KnnEuclidean,
    #[serde(alias = "knn-manhattan", alias = "KnnManhattan")]
    KnnManhattan,
    #[serde(alias = "knn-minkowski", alias = "KnnMinkowski")]
    KnnMinkowski,
    #[serde(alias = "logistic-regression", alias = "LogisticRegression")]
    LogisticRegression,
    #[serde(alias = "random-forest", alias = "RandomForest")]
    RandomForest,
    #[serde(alias = "isolation-forest", alias = "IsolationForest")]
    IsolationForest,
    #[serde(alias = "svm", alias = "Svm")]
    Svm,
    #[serde(alias = "gradient-boosting", alias = "GradientBoosting")]
    GradientBoosting,
}

impl MethodKind {
    pub fn is_knn(&self) -> bool {
matches!(self, Self::KnnCosine | Self::KnnEuclidean | Self::KnnManhattan | Self::KnnMinkowski)
    }

    pub fn requires_p(&self) -> bool {
        matches!(self, Self::KnnMinkowski)
    }
}

#[derive(Debug, Clone)]
pub enum ClassifierMethod {
    Bayes,
    KnnCosine,
    KnnEuclidean,
    KnnManhattan,
    KnnMinkowski,
    LogisticRegression,
    RandomForest,
    IsolationForest,
    Svm,
    GradientBoosting,
}

impl From<MethodKind> for ClassifierMethod {
    fn from(value: MethodKind) -> Self {
        match value {
            MethodKind::Bayes => Self::Bayes,
            MethodKind::KnnCosine => Self::KnnCosine,
            MethodKind::KnnEuclidean => Self::KnnEuclidean,
            MethodKind::KnnManhattan => Self::KnnManhattan,
            MethodKind::KnnMinkowski => Self::KnnMinkowski,
            MethodKind::LogisticRegression => Self::LogisticRegression,
            MethodKind::RandomForest => Self::RandomForest,
            MethodKind::IsolationForest => Self::IsolationForest,
            MethodKind::Svm => Self::Svm,
            MethodKind::GradientBoosting => Self::GradientBoosting,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub labels: Vec<(ClassificationLabel, f32)>,
    pub extra_hits: Vec<AlertHit>,
}

pub trait Classifier {
    fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult;
}

pub struct ClassifierFactory;

impl ClassifierFactory {
    pub fn builder() -> ClassifierBuilder {
        ClassifierBuilder::new()
    }
}

pub struct ClassifierBuilder {
    method: Option<ClassifierMethod>,
    nlp: Option<NlpOption>,
    hot_label: Option<ClassificationLabel>,
    cold_label: Option<ClassificationLabel>,
    hot_path: Option<PathBuf>,
    cold_path: Option<PathBuf>,
    recursive: bool,
    threads: Option<usize>,
    k: Option<usize>,
    p: Option<f32>,
    advanced: AdvancedModelConfig,
}

impl ClassifierBuilder {
    pub fn new() -> Self {
        Self {
            method: None,
            nlp: None,
            hot_label: None,
            cold_label: None,
            hot_path: None,
            cold_path: None,
            recursive: true,
            threads: None,
            k: None,
            p: None,
            advanced: AdvancedModelConfig::default(),
        }
    }

    pub fn method(mut self, method: ClassifierMethod) -> Self {
        self.method = Some(method);
        self
    }

    pub fn nlp(mut self, nlp: NlpOption) -> Self {
        self.nlp = Some(nlp);
        self
    }

    pub fn hot_label(mut self, label: ClassificationLabel) -> Self {
        self.hot_label = Some(label);
        self
    }

    pub fn cold_label(mut self, label: ClassificationLabel) -> Self {
        self.cold_label = Some(label);
        self
    }

    pub fn hot_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.hot_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn cold_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.cold_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn threads(mut self, threads: Option<usize>) -> Self {
        self.threads = threads;
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

    pub fn logistic_config(mut self, learning_rate: f32, epochs: usize, lambda: Option<f32>) -> Self {
        self.advanced.logistic = Some(LogisticRegressionConfig {
            learning_rate,
            epochs,
            lambda: lambda.unwrap_or(1e-3),
        });
        self
    }

    pub fn random_forest_config(mut self, n_trees: usize, max_depth: Option<usize>, min_samples_split: Option<usize>) -> Self {
        self.advanced.random_forest = Some(RandomForestConfig {
            n_trees,
            max_depth: max_depth.unwrap_or(6),
            min_samples_split: min_samples_split.unwrap_or(2),
            ..RandomForestConfig::default()
        });
        self
    }

    pub fn random_forest_mode(mut self, mode: RandomForestMode) -> Self {
        let mut cfg = self.advanced.random_forest.take().unwrap_or_default();
        cfg.mode = mode;
        self.advanced.random_forest = Some(cfg);
        self
    }

    pub fn random_forest_full_config(
        mut self,
        mode: RandomForestMode,
        n_trees: usize,
        max_depth: Option<usize>,
        max_features: Option<RandomForestMaxFeatures>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        bootstrap: Option<bool>,
        oob_score: Option<bool>,
    ) -> Self {
        self.advanced.random_forest = Some(RandomForestConfig {
            mode,
            n_trees,
            max_depth: max_depth.unwrap_or(6),
            max_features: max_features.unwrap_or(RandomForestMaxFeatures::Sqrt),
            min_samples_split: min_samples_split.unwrap_or(2),
            min_samples_leaf: min_samples_leaf.unwrap_or(1),
            bootstrap: bootstrap.unwrap_or(true),
            oob_score: oob_score.unwrap_or(false),
        });
        self
    }

    pub fn svm_config(mut self, kernel: SvmKernel, c: f32, learning_rate: Option<f32>, epochs: Option<usize>, gamma: Option<f32>, degree: Option<usize>, coef0: Option<f32>) -> Self {
        self.advanced.svm = Some(SvmConfig {
            kernel,
            c,
            learning_rate: learning_rate.unwrap_or(0.08),
            epochs: epochs.unwrap_or(40),
            gamma: gamma.unwrap_or(0.35),
            degree: degree.unwrap_or(2),
            coef0: coef0.unwrap_or(0.0),
        });
        self
    }

    pub fn gradient_boosting_config(mut self, n_estimators: usize, learning_rate: f32, max_depth: Option<usize>) -> Self {
        self.advanced.gradient_boosting = Some(GradientBoostingConfig {
            n_estimators,
            learning_rate,
            max_depth: max_depth.unwrap_or(1),
        });
        self
    }

    pub fn isolation_forest_config(mut self, n_trees: usize, contamination: f32, subsample_size: Option<usize>) -> Self {
        self.advanced.isolation_forest = Some(IsolationForestConfig {
            n_trees,
            contamination,
            subsample_size: subsample_size.unwrap_or(64),
        });
        self
    }

    pub fn from_rules_file(mut self, rules: &RulesFile) -> Self {
        self.method = Some(rules.method.clone().into());
        self.nlp = Some(rules.nlp.clone());
        self.hot_path = Some(rules.hot_test_path.clone());
        self.cold_path = Some(rules.cold_test_path.clone());
        self.hot_label = Some(rules.hot_label.clone().unwrap_or(ClassificationLabel::WebAttack));
        self.cold_label = Some(rules.cold_label.clone().unwrap_or(ClassificationLabel::RawData));
        self.recursive = rules.recursive_way.is_on();
        self.threads = rules.threads;
        self.k = rules.k;
        self.p = rules.p;
        self.advanced.logistic = match (rules.logistic_learning_rate, rules.logistic_epochs) {
            (Some(learning_rate), Some(epochs)) => Some(LogisticRegressionConfig {
                learning_rate,
                epochs,
                lambda: rules.logistic_lambda.unwrap_or(1e-3),
            }),
            _ => None,
        };
        self.advanced.random_forest = rules.random_forest_n_trees.map(|n_trees| RandomForestConfig {
            mode: rules.random_forest_mode.clone().unwrap_or(RandomForestMode::Standard),
            n_trees,
            max_depth: rules.random_forest_max_depth.unwrap_or(6),
            max_features: rules.random_forest_max_features.clone().unwrap_or(RandomForestMaxFeatures::Sqrt),
            min_samples_split: rules.random_forest_min_samples_split.unwrap_or(2),
            min_samples_leaf: rules.random_forest_min_samples_leaf.unwrap_or(1),
            bootstrap: rules.random_forest_bootstrap.unwrap_or(true),
            oob_score: rules.random_forest_oob_score.unwrap_or(false),
        });
        self.advanced.svm = match (rules.svm_kernel.clone(), rules.svm_c) {
            (Some(kernel), Some(c)) => Some(SvmConfig {
                kernel,
                c,
                learning_rate: rules.svm_learning_rate.unwrap_or(0.08),
                epochs: rules.svm_epochs.unwrap_or(40),
                gamma: rules.svm_gamma.unwrap_or(0.35),
                degree: rules.svm_degree.unwrap_or(2),
                coef0: rules.svm_coef0.unwrap_or(0.0),
            }),
            _ => None,
        };
        self.advanced.gradient_boosting = match (rules.gradient_boosting_n_estimators, rules.gradient_boosting_learning_rate) {
            (Some(n_estimators), Some(learning_rate)) => Some(GradientBoostingConfig {
                n_estimators,
                learning_rate,
                max_depth: rules.gradient_boosting_max_depth.unwrap_or(1),
            }),
            _ => None,
        };
        self.advanced.threads = rules.threads;
        self.advanced.isolation_forest = match (rules.isolation_forest_n_trees, rules.isolation_forest_contamination) {
            (Some(n_trees), Some(contamination)) => Some(IsolationForestConfig {
                n_trees,
                contamination,
                subsample_size: rules.isolation_forest_subsample_size.unwrap_or(64),
            }),
            _ => None,
        };
        self
    }

    pub fn build(self) -> Result<Box<dyn Classifier>, VecEyesError> {
        let method = self.method.ok_or_else(|| VecEyesError::InvalidConfig("missing method".into()))?;
        let nlp = self.nlp.ok_or_else(|| VecEyesError::InvalidConfig("missing nlp option".into()))?;
        let hot_label = self.hot_label.ok_or_else(|| VecEyesError::InvalidConfig("missing hot label".into()))?;
        let cold_label = self.cold_label.unwrap_or(ClassificationLabel::RawData);

        let mut samples = Vec::new();

        if let Some(hot_path) = &self.hot_path {
            let mut hot = load_training_samples(hot_path, hot_label.clone(), self.recursive)?;
            samples.append(&mut hot);
        }
        if let Some(cold_path) = &self.cold_path {
            let mut cold = load_training_samples(cold_path, cold_label.clone(), self.recursive)?;
            samples.append(&mut cold);
        }

        match method {
            ClassifierMethod::Bayes => Ok(Box::new(BayesBuilder::new().nlp(nlp).samples(samples).threads(self.threads).build()?)),
            ClassifierMethod::KnnCosine => {
                let k = require_k(self.k)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).threads(self.threads).k(k).cosine().build()?))
            }
            ClassifierMethod::KnnEuclidean => {
                let k = require_k(self.k)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).threads(self.threads).k(k).euclidean().build()?))
            }
            ClassifierMethod::KnnManhattan => {
                let k = require_k(self.k)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).threads(self.threads).k(k).manhattan().build()?))
            }
            ClassifierMethod::KnnMinkowski => {
                let k = require_k(self.k)?;
                let p = require_p(self.p)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).threads(self.threads).k(k).p(p).minkowski(p).build()?))
            }
            ClassifierMethod::LogisticRegression => {
                require_logistic_config(&self.advanced)?;
                Ok(Box::new(AdvancedClassifier::train(
                    AdvancedMethod::LogisticRegression,
                    &samples,
                    nlp,
                    hot_label,
                    cold_label,
                    &self.advanced,
                )?))
            },
            ClassifierMethod::RandomForest => {
                require_random_forest_config(&self.advanced)?;
                Ok(Box::new(AdvancedClassifier::train(
                    AdvancedMethod::RandomForest,
                    &samples,
                    nlp,
                    hot_label,
                    cold_label,
                    &self.advanced,
                )?))
            },
            ClassifierMethod::IsolationForest => {
                require_isolation_forest_config(&self.advanced)?;
                Ok(Box::new(AdvancedClassifier::train(
                    AdvancedMethod::IsolationForest,
                    &samples,
                    nlp,
                    hot_label,
                    cold_label,
                    &self.advanced,
                )?))
            },
            ClassifierMethod::Svm => {
                require_svm_config(&self.advanced)?;
                Ok(Box::new(AdvancedClassifier::train(
                    AdvancedMethod::Svm,
                    &samples,
                    nlp,
                    hot_label,
                    cold_label,
                    &self.advanced,
                )?))
            },
            ClassifierMethod::GradientBoosting => {
                require_gradient_boosting_config(&self.advanced)?;
                Ok(Box::new(AdvancedClassifier::train(
                    AdvancedMethod::GradientBoosting,
                    &samples,
                    nlp,
                    hot_label,
                    cold_label,
                    &self.advanced,
                )?))
            },
        }
    }
}

fn require_k(value: Option<usize>) -> Result<usize, VecEyesError> {
    let k = value.ok_or_else(|| VecEyesError::InvalidConfig("KNN requires field 'k' and it must be passed explicitly".into()))?;
    if k == 0 {
        return Err(VecEyesError::InvalidConfig("KNN requires k >= 1".into()));
    }
    Ok(k)
}

fn require_p(value: Option<f32>) -> Result<f32, VecEyesError> {
    let p = value.ok_or_else(|| VecEyesError::InvalidConfig("KNN Minkowski requires field 'p'".into()))?;
    if p <= 0.0 {
        return Err(VecEyesError::InvalidConfig("KNN Minkowski requires p > 0".into()));
    }
    Ok(p)
}

fn require_logistic_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.logistic {
        Some(cfg) if cfg.learning_rate > 0.0 && cfg.epochs > 0 => Ok(()),
        _ => Err(VecEyesError::InvalidConfig(
            "LogisticRegression requires logistic_learning_rate and logistic_epochs".into(),
        )),
    }
}

fn require_random_forest_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.random_forest {
        Some(cfg) if cfg.n_trees > 0 => Ok(()),
        _ => Err(VecEyesError::InvalidConfig(
            "RandomForest requires random_forest_n_trees".into(),
        )),
    }
}

fn require_svm_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.svm {
        Some(cfg) if cfg.c > 0.0 && cfg.epochs > 0 && cfg.learning_rate > 0.0 => Ok(()),
        _ => Err(VecEyesError::InvalidConfig(
            "Svm requires svm_kernel and svm_c plus valid training defaults".into(),
        )),
    }
}

fn require_gradient_boosting_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.gradient_boosting {
        Some(cfg) if cfg.n_estimators > 0 && cfg.learning_rate > 0.0 => Ok(()),
        _ => Err(VecEyesError::InvalidConfig(
            "GradientBoosting requires gradient_boosting_n_estimators and gradient_boosting_learning_rate".into(),
        )),
    }
}

fn require_isolation_forest_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.isolation_forest {
        Some(cfg) if cfg.n_trees > 0 && cfg.contamination > 0.0 && cfg.contamination < 0.5 => Ok(()),
        _ => Err(VecEyesError::InvalidConfig(
            "IsolationForest requires isolation_forest_n_trees and isolation_forest_contamination".into(),
        )),
    }
}

#[derive(Debug, Clone)]
pub struct BayesBuilder {
    nlp: NlpOption,
    samples: Vec<TrainingSample>,
    threads: Option<usize>,
}

impl BayesBuilder {
    pub fn new() -> Self {
        Self { nlp: NlpOption::Count, samples: Vec::new(), threads: None }
    }

    pub fn nlp(mut self, nlp: NlpOption) -> Self {
        self.nlp = nlp;
        self
    }

    pub fn samples(mut self, samples: Vec<TrainingSample>) -> Self {
        self.samples = samples;
        self
    }

    pub fn threads(mut self, threads: Option<usize>) -> Self {
        self.threads = threads;
        self
    }

    pub fn build(self) -> Result<BayesClassifier, VecEyesError> {
        BayesClassifier::train(&self.samples, self.nlp, self.threads)
    }
}

#[derive(Debug, Clone)]
pub enum BayesFeature {
    Count,
    TfIdf(TfIdfModel),
}

#[derive(Debug, Clone)]
pub struct BayesClassifier {
    nlp: NlpOption,
    threads: Option<usize>,
    labels: Vec<ClassificationLabel>,
    token_scores: HashMap<ClassificationLabel, HashMap<String, f32>>,
    priors: HashMap<ClassificationLabel, f32>,
    tfidf: Option<TfIdfModel>,
}

impl BayesClassifier {
    pub fn train(samples: &[TrainingSample], nlp: NlpOption, threads: Option<usize>) -> Result<Self, VecEyesError> {
        let mut token_scores: HashMap<ClassificationLabel, HashMap<String, f32>> = HashMap::new();
        let mut label_counts: HashMap<ClassificationLabel, usize> = HashMap::new();
        let texts: Vec<String> = samples.iter().map(|s| s.text.clone()).collect();
        let tfidf = if nlp == NlpOption::TfIdf { Some(fit_tfidf(&texts)) } else { None };

        for sample in samples {
            *label_counts.entry(sample.label.clone()).or_insert(0) += 1;
            let entry = token_scores.entry(sample.label.clone()).or_default();
            let normalized = crate::nlp::normalize_text(&sample.text);
            let tokens = crate::nlp::tokenize(&normalized);
            for token in tokens {
                *entry.entry(token).or_insert(0.0) += 1.0;
            }
        }

        let total = samples.len() as f32;
        let mut priors = HashMap::new();
        let mut labels = Vec::new();
        for (label, count) in label_counts {
            labels.push(label.clone());
            priors.insert(label, count as f32 / total.max(1.0));
        }

        Ok(Self { nlp, threads, labels, token_scores, priors, tfidf })
    }

    fn base_scores(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        let normalized = crate::nlp::normalize_text(text);
        let tokens = crate::nlp::tokenize(&normalized);
        let labels = self.labels.clone();
        let tfidf_matrix = if self.nlp == NlpOption::TfIdf {
            self.tfidf.as_ref().map(|model| transform_tfidf(model, &[text.to_string()]))
        } else {
            None
        };

        let raw = install_pool(self.threads, || {
            use rayon::prelude::*;
            labels
                .par_iter()
                .map(|label| {
                    let prior = self.priors.get(label).copied().unwrap_or(0.01).ln();
                    let token_map = self.token_scores.get(label);
                    let mut score = prior;

                    if self.nlp == NlpOption::TfIdf {
                        if let (Some(model), Some(matrix)) = (&self.tfidf, &tfidf_matrix) {
                            for token in &tokens {
                                if let Some(index) = model.token_to_index.get(token) {
                                    let weight = matrix[[0, *index]].max(1e-6);
                                    let token_count = token_map.and_then(|m| m.get(token)).copied().unwrap_or(1.0);
                                    score += (token_count * weight).ln();
                                }
                            }
                        }
                    } else {
                        for token in &tokens {
                            let token_count = token_map.and_then(|m| m.get(token)).copied().unwrap_or(1.0);
                            score += token_count.ln();
                        }
                    }

                    (label.clone(), score)
                })
                .collect::<Vec<_>>()
        });

        softmax_scores(&raw)
    }
}

impl Classifier for BayesClassifier {
    fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult {
        let mut labels = self.base_scores(text);
        let (boost, hits) = ScoringEngine::compute_rule_boost(text, matchers);
        if score_sum_mode.is_on() {
            for (_, score) in &mut labels {
                *score = ScoringEngine::merge_scores(*score, boost, score_sum_mode);
            }
        }
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ClassificationResult { labels, extra_hits: hits }
    }
}

#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Minkowski(f32),
}

#[derive(Debug, Clone)]
pub struct KnnBuilder {
    nlp: NlpOption,
    samples: Vec<TrainingSample>,
    metric: DistanceMetric,
    dims: usize,
    k: Option<usize>,
    p: Option<f32>,
    threads: Option<usize>,
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
        }
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

    pub fn build(self) -> Result<KnnClassifier, VecEyesError> {
        let k = require_k(self.k)?;
        if let DistanceMetric::Minkowski(_) = self.metric {
            require_p(self.p)?;
        }
        KnnClassifier::train(&self.samples, self.nlp, self.metric, self.dims, k, self.threads)
    }
}

#[derive(Debug, Clone)]
pub enum DenseFeatureModel {
    Word2Vec(WordEmbeddingModel),
    FastText(WordEmbeddingModel),
}

#[derive(Debug, Clone)]
pub struct KnnClassifier {
    metric: DistanceMetric,
    threads: Option<usize>,
    labels: Vec<ClassificationLabel>,
    matrix: DenseMatrix,
    model: DenseFeatureModel,
    k: usize,
}

impl KnnClassifier {
    pub fn train(
        samples: &[TrainingSample],
        nlp: NlpOption,
        metric: DistanceMetric,
        dims: usize,
        k: usize,
        threads: Option<usize>,
    ) -> Result<Self, VecEyesError> {
        let texts: Vec<String> = samples.iter().map(|s| s.text.clone()).collect();
        let labels: Vec<ClassificationLabel> = samples.iter().map(|s| s.label.clone()).collect();

        let model = match nlp {
            NlpOption::Word2Vec => DenseFeatureModel::Word2Vec(WordEmbeddingModel::train_word2vec(&texts, dims)),
            NlpOption::FastText => {
                let config = FastTextConfigBuilder::new().build();
                DenseFeatureModel::FastText(WordEmbeddingModel::train_fasttext(&texts, dims, config))
            }
            _ => return Err(VecEyesError::InvalidConfig("KNN requires Word2Vec or FastText".into())),
        };

        let matrix = match &model {
            DenseFeatureModel::Word2Vec(inner) => dense_matrix_from_texts(inner, &texts),
            DenseFeatureModel::FastText(inner) => dense_matrix_from_texts(inner, &texts),
        };

        Ok(Self { metric, threads, labels, matrix, model, k })
    }

    fn matrix_for_text(&self, text: &str) -> DenseMatrix {
        let texts = vec![text.to_string()];
        match &self.model {
            DenseFeatureModel::Word2Vec(inner) => dense_matrix_from_texts(inner, &texts),
            DenseFeatureModel::FastText(inner) => dense_matrix_from_texts(inner, &texts),
        }
    }

    fn score_neighbors(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        let probe = self.matrix_for_text(text);
        let probe_row = probe.index_axis(Axis(0), 0);
        let probe_vec = probe_row.to_vec();
        let mut ranked: Vec<(f32, ClassificationLabel)> = install_pool(self.threads, || {
            use rayon::prelude::*;
            (0..self.matrix.shape()[0])
                .into_par_iter()
                .map(|row| {
                    let candidate = self.matrix.index_axis(Axis(0), row);
                    let candidate_vec = candidate.to_vec();
                    let distance = match self.metric {
                        DistanceMetric::Cosine => cosine_distance(&probe_vec, &candidate_vec),
                        DistanceMetric::Euclidean => euclidean_distance(&probe_vec, &candidate_vec),
                        DistanceMetric::Manhattan => manhattan_distance(&probe_vec, &candidate_vec),
                        DistanceMetric::Minkowski(p) => minkowski_distance(&probe_vec, &candidate_vec, p),
                    };
                    (distance, self.labels[row].clone())
                })
                .collect::<Vec<_>>()
        });

        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut best: HashMap<ClassificationLabel, f32> = HashMap::new();
        let limit = self.k.min(ranked.len());
        for (distance, label) in ranked.into_iter().take(limit) {
            let score = (1.0 / (distance + 1e-6)).min(1000.0);
            *best.entry(label).or_insert(0.0) += score;
        }

        let mut raw = Vec::new();
        for (label, score) in best {
            raw.push((label, score));
        }
        softmax_scores(&raw)
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
        let (boost, hits) = ScoringEngine::compute_rule_boost(text, matchers);
        if score_sum_mode.is_on() {
            for (_, score) in &mut labels {
                *score = ScoringEngine::merge_scores(*score, boost, score_sum_mode);
            }
        }
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ClassificationResult { labels, extra_hits: hits }
    }
}

pub fn run_rules_pipeline(
    rules: &RulesFile,
    classify_objects: &Path,
) -> Result<crate::report::ClassificationReport, VecEyesError> {
    rules.validate()?;

    if !classify_objects.exists() {
        return Err(VecEyesError::InvalidConfig(format!(
            "classify_objects path does not exist: {}",
            classify_objects.display()
        )));
    }

    let builder = ClassifierFactory::builder().from_rules_file(rules);
    let classifier = builder.build()?;
    let matchers = ScoringEngine::matchers_from_rules_file(rules)?;
    let mut report = crate::report::ClassificationReport::new(rules.report_name.clone().unwrap_or_else(|| "Vec-Eyes Report".to_string()));

    let files = crate::dataset::collect_files_recursively(classify_objects, rules.recursive_way.is_on())?;
    for file in files {
        let text = read_text_file(&file)?;
        let result = classifier.classify_text(&text, rules.score_sum, &matchers);
        let labels: Vec<String> = result.labels.iter().map(|(l, s)| format!("{}:{:.2}", l, s)).collect();
        let top_score = result.labels.first().map(|x| x.1).unwrap_or(0.0);
        let hit_titles: Vec<String> = result.extra_hits.iter().map(|h| h.title.clone()).collect();
        report.records.push(crate::report::ClassificationRecord {
            title_object: file.file_name().and_then(|x| x.to_str()).unwrap_or("object").to_string(),
            name_file_dataset: file.to_string_lossy().to_string(),
            classify_names_list: labels.join(","),
            date_of_occurrence: Utc::now(),
            score_percent: top_score,
            match_titles: hit_titles.join(","),
        });
    }

    Ok(report)
}

pub(crate) fn softmax_scores(input: &[(ClassificationLabel, f32)]) -> Vec<(ClassificationLabel, f32)> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut max_score = f32::NEG_INFINITY;
    for (_, score) in input {
        if *score > max_score {
            max_score = *score;
        }
    }

    let mut sum = 0.0f32;
    let mut exp_values = Vec::new();
    for (label, score) in input {
        let value = (*score - max_score).exp();
        sum += value;
        exp_values.push((label.clone(), value));
    }

    let mut output = Vec::new();
    for (label, value) in exp_values {
        let pct = if sum > 0.0 { (value / sum) * 100.0 } else { 0.0 };
        output.push((label, pct));
    }
    output
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for idx in 0..a.len() {
        dot += a[idx] * b[idx];
        na += a[idx] * a[idx];
        nb += b[idx] * b[idx];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        let d = a[idx] - b[idx];
        acc += d * d;
    }
    acc.sqrt()
}

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        acc += (a[idx] - b[idx]).abs();
    }
    acc
}

fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        acc += (a[idx] - b[idx]).abs().powf(p);
    }
    acc.powf(1.0 / p)
}
