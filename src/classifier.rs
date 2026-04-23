use crate::advanced_models::{
    AdvancedClassifier, AdvancedMethod, AdvancedModelConfig, GradientBoostingConfig,
    IsolationForestConfig, LogisticRegressionConfig, RandomForestConfig, RandomForestMaxFeatures,
    RandomForestMode, SvmConfig, SvmKernel,
};
use crate::builders::Builder;
use crate::config::{RulesFile, ScoreSumMode};
use crate::dataset::{load_training_samples, read_text_file_limited};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::matcher::{AlertHit, RuleMatcher, ScoringEngine};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub use crate::classifiers::bayes::{BayesBuilder, BayesClassifier, BayesFeature};
pub use crate::classifiers::knn::{DenseFeatureModel, DistanceMetric, KnnBuilder, KnnClassifier};

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

pub type ClassifierMethod = MethodKind;

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub labels: Vec<(ClassificationLabel, f32)>,
    pub extra_hits: Vec<AlertHit>,
}

impl ClassificationResult {
    pub fn top_label(&self) -> Option<&ClassificationLabel> { self.labels.first().map(|(l, _)| l) }
    pub fn top_score(&self) -> f32 { self.labels.first().map(|(_, s)| *s).unwrap_or(0.0) }
    pub fn is_hot(&self, threshold: f32) -> bool { self.top_score() >= threshold }
    pub fn rule_hits(&self) -> &[AlertHit] { &self.extra_hits }
}

#[derive(Debug, Clone)]
pub struct TokenContribution {
    pub token: String,
    pub contribution: f32,
}

pub trait ExplainableClassifier: Classifier {
    fn explain(&self, text: &str) -> Vec<TokenContribution>;
}

pub struct EnsembleClassifier {
    members: Vec<(Box<dyn Classifier>, f32)>,
}

impl EnsembleClassifier {
    /// Build an ensemble.  Weights are normalised to sum to 1.0 automatically
    /// so callers can supply raw relative weights (e.g. `0.7` / `0.3`) or
    /// probabilities — the result is the same.
    pub fn new(members: Vec<(Box<dyn Classifier>, f32)>) -> Self {
        let total: f32 = members.iter().map(|(_, w)| w.abs()).sum::<f32>().max(1e-9);
        let members = members.into_iter().map(|(c, w)| (c, w / total)).collect();
        Self { members }
    }
}

impl Classifier for EnsembleClassifier {
    fn classify_text(&self, text: &str, score_sum_mode: ScoreSumMode, matchers: &[Box<dyn RuleMatcher>]) -> ClassificationResult {
        use std::collections::HashMap;
        let mut acc: HashMap<ClassificationLabel, f32> = HashMap::new();
        let mut hits = Vec::new();
        for (classifier, weight) in &self.members {
            let result = classifier.classify_text(text, score_sum_mode, matchers);
            // Sum weighted log-probs: each member already emits calibrated
            // probabilities via softmax; taking their ln converts back to
            // log-space before combining, which is mathematically correct for
            // a product-of-experts ensemble.
            for (label, score) in result.labels {
                *acc.entry(label).or_insert(0.0) += weight * score.max(1e-9).ln();
            }
            hits.extend(result.extra_hits);
        }
        let mut labels: Vec<(ClassificationLabel, f32)> = acc.into_iter().collect();
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let labels = crate::math::softmax_scores(&labels);
        ClassificationResult { labels, extra_hits: hits }
    }
}

pub trait Classifier {
    fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult;

    /// Classify a batch of texts.
    ///
    /// The default implementation is sequential. Concrete types that are
    /// `Send + Sync` (e.g. [`AdvancedClassifier`]) override this to amortise
    /// the NLP pipeline and parallelize with rayon.
    fn classify_batch(
        &self,
        texts: &[&str],
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> Vec<ClassificationResult> {
        texts.iter()
            .map(|t| self.classify_text(t, score_sum_mode, matchers))
            .collect()
    }
}

pub struct ClassifierFactory;

impl ClassifierFactory {
    pub fn builder() -> ClassifierBuilder {
        ClassifierBuilder::new()
    }
}

pub struct ClassifierBuilder {
    method: Option<ClassifierMethod>,
    nlp: Option<crate::nlp::NlpOption>,
    hot_label: Option<ClassificationLabel>,
    cold_label: Option<ClassificationLabel>,
    hot_path: Option<PathBuf>,
    cold_path: Option<PathBuf>,
    recursive: bool,
    threads: Option<usize>,
    k: Option<usize>,
    p: Option<f32>,
    advanced: AdvancedModelConfig,
    /// Pre-loaded samples that bypass hot_path / cold_path file loading.
    preloaded_samples: Option<Vec<crate::dataset::TrainingSample>>,
}

impl Builder<Box<dyn Classifier>> for ClassifierBuilder {
    fn new() -> Self {
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
            preloaded_samples: None,
        }
    }

    fn build(self) -> Result<Box<dyn Classifier>, VecEyesError> {
        let method = self.method.ok_or_else(|| VecEyesError::invalid_config("classifier::ClassifierBuilder::build", "missing method; call .method(...) before .build()"))?;
        let nlp = self.nlp.ok_or_else(|| VecEyesError::invalid_config("classifier::ClassifierBuilder::build", "missing nlp option; call .nlp(...) before .build()"))?;
        let hot_label = self.hot_label.clone().unwrap_or(ClassificationLabel::WebAttack);
        let cold_label = self.cold_label.clone().unwrap_or(ClassificationLabel::RawData);

        let samples = if let Some(preloaded) = self.preloaded_samples {
            preloaded
        } else {
            let hot_path = self.hot_path.ok_or_else(|| VecEyesError::invalid_config("classifier::ClassifierBuilder::build", "missing hot training path; call .hot_path(...) or .samples(...) before .build()"))?;
            let cold_path = self.cold_path.ok_or_else(|| VecEyesError::invalid_config("classifier::ClassifierBuilder::build", "missing cold training path; call .cold_path(...) or .samples(...) before .build()"))?;
            let mut s = load_training_samples(&hot_path, hot_label.clone(), self.recursive)?;
            s.extend(load_training_samples(&cold_path, cold_label.clone(), self.recursive)?);
            s
        };

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
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).threads(self.threads).k(k).minkowski(p).build()?))
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
            }
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
            }
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
            }
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
            }
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
            }
        }
    }
}

impl ClassifierBuilder {

pub fn new() -> Self {
    <Self as Builder<Box<dyn Classifier>>>::new()
}

pub fn build(self) -> Result<Box<dyn Classifier>, VecEyesError> {
    <Self as Builder<Box<dyn Classifier>>>::build(self)
}


    pub fn method(mut self, method: ClassifierMethod) -> Self {
        self.method = Some(method);
        self
    }

    pub fn nlp(mut self, nlp: crate::nlp::NlpOption) -> Self {
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

    pub fn embedding_dimensions(mut self, dimensions: usize) -> Self {
        self.advanced.embedding_dimensions = Some(dimensions.max(1));
        self
    }

    pub fn advanced_config(mut self, advanced: AdvancedModelConfig) -> Self {
        self.advanced = advanced;
        self
    }

/// Supply pre-loaded training samples directly, bypassing `hot_path` / `cold_path`.
///
/// When this is set, `hot_path` and `cold_path` are ignored.  `hot_label` and
/// `cold_label` still contribute as defaults when samples have no override.
pub fn samples(mut self, samples: Vec<crate::dataset::TrainingSample>) -> Self {
    self.preloaded_samples = Some(samples);
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
        let cfg = self.advanced.random_forest.get_or_insert_with(RandomForestConfig::default);
        cfg.mode = mode;
        self
    }

    #[allow(clippy::too_many_arguments)]
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
            random_seed: None,
        });
        self
    }

    #[allow(clippy::too_many_arguments)]
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

    pub fn from_rules_file(self, rules: &RulesFile) -> Self {
        rules.apply_to_builder(self)
    }
}

pub(crate) fn require_k(value: Option<usize>) -> Result<usize, VecEyesError> {
    let k = value.ok_or_else(|| VecEyesError::invalid_config("classifier::KNN", "field 'k' is required and must be passed explicitly"))?;
    if k == 0 {
        return Err(VecEyesError::invalid_config("classifier::KNN", "k must be >= 1"));
    }
    Ok(k)
}

pub(crate) fn require_p(value: Option<f32>) -> Result<f32, VecEyesError> {
    let p = value.ok_or_else(|| VecEyesError::invalid_config("classifier::KNN Minkowski", "field 'p' is required"))?;
    if p <= 0.0 {
        return Err(VecEyesError::invalid_config("classifier::KNN Minkowski", "p must be > 0"));
    }
    Ok(p)
}

fn require_logistic_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.logistic {
        Some(cfg) if cfg.learning_rate > 0.0 && cfg.epochs > 0 => Ok(()),
        _ => Err(VecEyesError::invalid_config("classifier::AdvancedClassifier::LogisticRegression", "logistic_learning_rate and logistic_epochs must be configured with valid positive values")),
    }
}

fn require_random_forest_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.random_forest {
        Some(cfg) if cfg.n_trees > 0 => Ok(()),
        _ => Err(VecEyesError::invalid_config("classifier::AdvancedClassifier::RandomForest", "random_forest_n_trees must be configured and >= 1")),
    }
}

fn require_svm_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.svm {
        Some(cfg) if cfg.c > 0.0 && cfg.epochs > 0 && cfg.learning_rate > 0.0 => Ok(()),
        _ => Err(VecEyesError::invalid_config("classifier::AdvancedClassifier::Svm", "svm_kernel and svm_c must be configured; training defaults must be positive")),
    }
}

fn require_gradient_boosting_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.gradient_boosting {
        Some(cfg) if cfg.n_estimators > 0 && cfg.learning_rate > 0.0 => Ok(()),
        _ => Err(VecEyesError::invalid_config("classifier::AdvancedClassifier::GradientBoosting", "gradient_boosting_n_estimators and gradient_boosting_learning_rate must be configured with valid positive values")),
    }
}

fn require_isolation_forest_config(config: &AdvancedModelConfig) -> Result<(), VecEyesError> {
    match &config.isolation_forest {
        Some(cfg) if cfg.n_trees > 0 && cfg.contamination > 0.0 && cfg.contamination < 0.5 => Ok(()),
        _ => Err(VecEyesError::invalid_config("classifier::AdvancedClassifier::IsolationForest", "isolation_forest_n_trees must be >= 1 and contamination must be in (0, 0.5)")),
    }
}

pub fn run_rules_pipeline(
    rules: &RulesFile,
    classify_objects: &Path,
) -> Result<crate::report::ClassificationReport, VecEyesError> {
    rules.validate()?;

    if !classify_objects.exists() {
        return Err(VecEyesError::invalid_config("classifier::run_rules_pipeline", format!("classify_objects path does not exist: {}", classify_objects.display())));
    }

    let builder = ClassifierFactory::builder().from_rules_file(rules);
    let classifier = builder.build()?;
    let matchers = ScoringEngine::matchers_from_rules_file(rules)?;
    let mut report = crate::report::ClassificationReport::new(rules.report_name.clone().unwrap_or_else(|| "Vec-Eyes Report".to_string()));

    let max_file_bytes = rules.max_file_bytes.unwrap_or(crate::dataset::DEFAULT_MAX_FILE_BYTES);
    let files = crate::dataset::collect_files_recursively(classify_objects, rules.recursive_way.is_on())?;
    for file in files {
        let text = read_text_file_limited(&file, max_file_bytes)?;
        let result = classifier.classify_text(&text, rules.score_sum, &matchers);
        let labels: Vec<String> = result.labels.iter().map(|(l, s)| format!("{}:{:.2}", l, s)).collect();
        let top_score = result.labels.first().map(|x| x.1).unwrap_or(0.0);
        let hit_titles: Vec<String> = result.extra_hits.iter().map(|h| h.title.clone()).collect();
        report.records.push(crate::report::ClassificationRecord {
            title_object: file.file_name().and_then(|x| x.to_str()).unwrap_or("object").to_string(),
            name_file_dataset: file.to_string_lossy().to_string(),
            date_of_occurrence: Utc::now(),
            score_percent: top_score * 100.0,
            match_titles: hit_titles.join(","),
            classify_names_list: labels.join(","),
        });
    }

    Ok(report)
}
