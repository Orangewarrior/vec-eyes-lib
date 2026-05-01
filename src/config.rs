use crate::advanced_models::{
    AdvancedModelConfig, GradientBoostingConfig, IsolationForestConfig, LogisticRegressionConfig,
    RandomForestConfig, RandomForestMaxFeatures, RandomForestMode, SvmConfig, SvmKernel,
};
use crate::classifier::{ClassifierBuilder, MethodKind};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::NlpOption;
use crate::security::MAX_CONFIG_THREADS;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MAX_RULES_FILE_BYTES: u64 = 512 * 1024;

// ── Utility enums (unchanged) ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursiveMode {
    #[serde(alias = "ON", alias = "on")]
    #[default]
    On,
    #[serde(alias = "OFF", alias = "off")]
    Off,
}
impl RecursiveMode {
    pub fn is_on(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreSumMode {
    #[serde(alias = "ON", alias = "on")]
    On,
    #[serde(alias = "OFF", alias = "off")]
    #[default]
    Off,
}
impl ScoreSumMode {
    pub fn is_on(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtraMatchEngine {
    #[serde(alias = "Vectorscan", alias = "vectorscan")]
    Vectorscan,
    #[serde(alias = "Regex", alias = "regex", alias = "regexmatch")]
    Regex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraMatchConfig {
    #[serde(default)]
    pub recursive_way: RecursiveMode,
    pub engine: ExtraMatchEngine,
    pub path: PathBuf,
    #[serde(default)]
    pub score_add_points: f32,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

// ── Sub-configs ───────────────────────────────────────────────────────────────

/// Training data paths, labels, and pipeline behaviour flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub hot_test_path: PathBuf,
    pub cold_test_path: PathBuf,
    #[serde(default)]
    pub hot_label: Option<ClassificationLabel>,
    #[serde(default)]
    pub cold_label: Option<ClassificationLabel>,
    #[serde(default)]
    pub recursive_way: RecursiveMode,
    #[serde(default)]
    pub score_sum: ScoreSumMode,
}

/// NLP feature-extraction settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub nlp: NlpOption,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub embedding_dimensions: Option<usize>,
    #[serde(default)]
    pub security_normalize_obfuscation: Option<bool>,
}

// ── Default helpers for serde ─────────────────────────────────────────────────
fn rf_max_depth_default() -> usize {
    6
}
fn rf_min_samples_split_default() -> usize {
    2
}
fn rf_min_samples_leaf_default() -> usize {
    1
}
fn rf_bootstrap_default() -> bool {
    true
}
fn lr_lambda_default() -> f32 {
    1e-3
}
fn svm_lr_default() -> f32 {
    0.08
}
fn svm_epochs_default() -> usize {
    40
}
fn svm_gamma_default() -> f32 {
    0.35
}
fn svm_degree_default() -> usize {
    2
}
fn gb_max_depth_default() -> usize {
    1
}
fn if_subsample_size_default() -> usize {
    64
}

// ── ModelConfig ───────────────────────────────────────────────────────────────

/// Classifier method and its full parameter set in a single typed block.
///
/// The `method` field acts as the tag; all other fields are the method's
/// parameters.  Unknown fields in YAML are silently ignored.
///
/// ```yaml
/// model:
///   method: RandomForest
///   n_trees: 61
///   max_depth: 10
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum ModelConfig {
    #[serde(alias = "bayes", alias = "Bayes")]
    Bayes,

    #[serde(alias = "knn-cosine", alias = "KnnCosine")]
    KnnCosine { k: usize },

    #[serde(alias = "knn-euclidean", alias = "KnnEuclidean")]
    KnnEuclidean { k: usize },

    #[serde(alias = "knn-manhattan", alias = "KnnManhattan")]
    KnnManhattan { k: usize },

    #[serde(alias = "knn-minkowski", alias = "KnnMinkowski")]
    KnnMinkowski { k: usize, p: f32 },

    #[serde(alias = "logistic-regression", alias = "LogisticRegression")]
    LogisticRegression {
        learning_rate: f32,
        epochs: usize,
        #[serde(default = "lr_lambda_default")]
        lambda: f32,
    },

    #[serde(alias = "random-forest", alias = "RandomForest")]
    RandomForest {
        n_trees: usize,
        #[serde(default = "rf_max_depth_default")]
        max_depth: usize,
        #[serde(default)]
        mode: RandomForestMode,
        #[serde(default)]
        max_features: RandomForestMaxFeatures,
        #[serde(default = "rf_min_samples_split_default")]
        min_samples_split: usize,
        #[serde(default = "rf_min_samples_leaf_default")]
        min_samples_leaf: usize,
        #[serde(default = "rf_bootstrap_default")]
        bootstrap: bool,
        #[serde(default)]
        oob_score: bool,
        #[serde(default)]
        random_seed: Option<u64>,
    },

    #[serde(alias = "svm", alias = "Svm")]
    Svm {
        kernel: SvmKernel,
        c: f32,
        #[serde(default = "svm_lr_default")]
        learning_rate: f32,
        #[serde(default = "svm_epochs_default")]
        epochs: usize,
        #[serde(default = "svm_gamma_default")]
        gamma: f32,
        #[serde(default = "svm_degree_default")]
        degree: usize,
        #[serde(default)]
        coef0: f32,
    },

    #[serde(alias = "gradient-boosting", alias = "GradientBoosting")]
    GradientBoosting {
        n_estimators: usize,
        learning_rate: f32,
        #[serde(default = "gb_max_depth_default")]
        max_depth: usize,
    },

    #[serde(alias = "isolation-forest", alias = "IsolationForest")]
    IsolationForest {
        n_trees: usize,
        contamination: f32,
        #[serde(default = "if_subsample_size_default")]
        subsample_size: usize,
    },
}

impl ModelConfig {
    pub fn method_kind(&self) -> MethodKind {
        match self {
            Self::Bayes => MethodKind::Bayes,
            Self::KnnCosine { .. } => MethodKind::KnnCosine,
            Self::KnnEuclidean { .. } => MethodKind::KnnEuclidean,
            Self::KnnManhattan { .. } => MethodKind::KnnManhattan,
            Self::KnnMinkowski { .. } => MethodKind::KnnMinkowski,
            Self::LogisticRegression { .. } => MethodKind::LogisticRegression,
            Self::RandomForest { .. } => MethodKind::RandomForest,
            Self::Svm { .. } => MethodKind::Svm,
            Self::GradientBoosting { .. } => MethodKind::GradientBoosting,
            Self::IsolationForest { .. } => MethodKind::IsolationForest,
        }
    }

    pub fn to_advanced_config(
        &self,
        threads: Option<usize>,
        embedding_dimensions: Option<usize>,
    ) -> AdvancedModelConfig {
        let mut cfg = AdvancedModelConfig {
            threads,
            embedding_dimensions,
            ..Default::default()
        };
        match self {
            Self::LogisticRegression {
                learning_rate,
                epochs,
                lambda,
            } => {
                cfg.logistic = Some(LogisticRegressionConfig {
                    learning_rate: *learning_rate,
                    epochs: *epochs,
                    lambda: *lambda,
                });
            }
            Self::RandomForest {
                n_trees,
                max_depth,
                mode,
                max_features,
                min_samples_split,
                min_samples_leaf,
                bootstrap,
                oob_score,
                random_seed,
            } => {
                cfg.random_forest = Some(RandomForestConfig {
                    mode: mode.clone(),
                    n_trees: *n_trees,
                    max_depth: *max_depth,
                    max_features: max_features.clone(),
                    min_samples_split: *min_samples_split,
                    min_samples_leaf: *min_samples_leaf,
                    bootstrap: *bootstrap,
                    oob_score: *oob_score,
                    random_seed: *random_seed,
                });
            }
            Self::Svm {
                kernel,
                c,
                learning_rate,
                epochs,
                gamma,
                degree,
                coef0,
            } => {
                cfg.svm = Some(SvmConfig {
                    kernel: kernel.clone(),
                    c: *c,
                    learning_rate: *learning_rate,
                    epochs: *epochs,
                    gamma: *gamma,
                    degree: *degree,
                    coef0: *coef0,
                });
            }
            Self::GradientBoosting {
                n_estimators,
                learning_rate,
                max_depth,
            } => {
                cfg.gradient_boosting = Some(GradientBoostingConfig {
                    n_estimators: *n_estimators,
                    learning_rate: *learning_rate,
                    max_depth: *max_depth,
                });
            }
            Self::IsolationForest {
                n_trees,
                contamination,
                subsample_size,
            } => {
                cfg.isolation_forest = Some(IsolationForestConfig {
                    n_trees: *n_trees,
                    contamination: *contamination,
                    subsample_size: *subsample_size,
                });
            }
            _ => {}
        }
        cfg
    }

    fn validate(&self) -> Result<(), VecEyesError> {
        match self {
            Self::KnnCosine { k } | Self::KnnEuclidean { k } | Self::KnnManhattan { k } => {
                if *k == 0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: k must be >= 1".into(),
                    ));
                }
            }
            Self::KnnMinkowski { k, p } => {
                if *k == 0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: k must be >= 1".into(),
                    ));
                }
                if *p <= 0.0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: p must be > 0 for knn-minkowski".into(),
                    ));
                }
            }
            Self::LogisticRegression {
                learning_rate,
                epochs,
                ..
            } => {
                if *learning_rate <= 0.0 || *epochs == 0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: learning_rate must be > 0 and epochs >= 1".into(),
                    ));
                }
            }
            Self::RandomForest {
                n_trees,
                min_samples_leaf,
                oob_score,
                bootstrap,
                ..
            } => {
                if *n_trees == 0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: n_trees must be >= 1".into(),
                    ));
                }
                if *min_samples_leaf == 0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: min_samples_leaf must be >= 1".into(),
                    ));
                }
                if *oob_score && !bootstrap {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: oob_score requires bootstrap = true".into(),
                    ));
                }
            }
            Self::Svm { c, .. } => {
                if *c <= 0.0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: svm c must be > 0".into(),
                    ));
                }
            }
            Self::GradientBoosting {
                n_estimators,
                learning_rate,
                ..
            } => {
                if *n_estimators == 0 || *learning_rate <= 0.0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: n_estimators >= 1 and learning_rate > 0".into(),
                    ));
                }
            }
            Self::IsolationForest {
                n_trees,
                contamination,
                ..
            } => {
                if *n_trees == 0 || *contamination <= 0.0 || *contamination >= 0.5 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: n_trees >= 1 and contamination in (0, 0.5)".into(),
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn apply_to_builder(&self, mut builder: ClassifierBuilder) -> ClassifierBuilder {
        builder = builder.method(self.method_kind());
        match self {
            Self::KnnCosine { k } | Self::KnnEuclidean { k } | Self::KnnManhattan { k } => {
                builder = builder.k(*k);
            }
            Self::KnnMinkowski { k, p } => {
                builder = builder.k(*k).p(*p);
            }
            _ => {}
        }
        builder
    }
}

// ── RulesFile ─────────────────────────────────────────────────────────────────

/// Top-level YAML pipeline configuration.
///
/// ```yaml
/// report_name: My pipeline   # optional
///
/// data:
///   hot_test_path: data/hot
///   cold_test_path: data/cold
///   hot_label: SPAM
///   cold_label: RAW_DATA
///   recursive_way: On
///   score_sum: Off
///
/// pipeline:
///   nlp: TfIdf
///   threads: 4
///
/// model:
///   method: RandomForest
///   n_trees: 61
///   max_depth: 10
///
/// extra_match:
///   - engine: Regex
///     path: rules/my_rules.txt
///     score_add_points: 20
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulesFile {
    #[serde(default)]
    pub report_name: Option<String>,
    pub data: DataConfig,
    pub pipeline: PipelineConfig,
    pub model: ModelConfig,
    #[serde(default)]
    pub extra_match: Vec<ExtraMatchConfig>,
    #[serde(default)]
    pub csv_output: Option<PathBuf>,
    #[serde(default)]
    pub json_output: Option<PathBuf>,
    #[serde(default)]
    pub max_file_bytes: Option<u64>,
}

fn validate_maybe_relative_path(path: &Path, allowed_base: &Path) -> Result<(), VecEyesError> {
    if path.is_absolute() {
        crate::security::sanitize_existing_path(path)?;
        return Ok(());
    }
    let current_dir = std::env::current_dir()?;
    let cwd_candidate = current_dir.join(path);
    if cwd_candidate.exists() {
        crate::security::sanitize_existing_path_with_base(&cwd_candidate, &current_dir)?;
        return Ok(());
    }
    let base_candidate = allowed_base.join(path);
    crate::security::sanitize_existing_path_with_base(&base_candidate, allowed_base)?;
    Ok(())
}

impl RulesFile {
    pub fn validate(&self) -> Result<(), VecEyesError> {
        if self.pipeline.threads == Some(0) {
            return Err(VecEyesError::invalid_config(
                "config::RulesFile::validate",
                "threads must be >= 1",
            ));
        }
        if self
            .pipeline
            .threads
            .is_some_and(|threads| threads > MAX_CONFIG_THREADS)
        {
            return Err(VecEyesError::invalid_config(
                "config::RulesFile::validate",
                format!("threads must be <= {MAX_CONFIG_THREADS}"),
            ));
        }
        if self.pipeline.embedding_dimensions == Some(0) {
            return Err(VecEyesError::InvalidConfig(
                "YAML validation error: embedding_dimensions must be >= 1".into(),
            ));
        }
        for extra in &self.extra_match {
            if extra.score_add_points < 0.0 {
                return Err(VecEyesError::invalid_config(
                    "config::RulesFile::validate",
                    format!(
                        "extra_match score_add_points must be >= 0.0 for {}",
                        extra.path.display()
                    ),
                ));
            }
        }
        self.model.validate()
    }

    pub fn validate_paths_against(&self, allowed_base: &Path) -> Result<(), VecEyesError> {
        validate_maybe_relative_path(&self.data.hot_test_path, allowed_base)?;
        validate_maybe_relative_path(&self.data.cold_test_path, allowed_base)?;
        for extra in &self.extra_match {
            validate_maybe_relative_path(&extra.path, allowed_base)?;
        }
        if let Some(path) = &self.csv_output {
            crate::security::sanitize_output_path_with_base(path, allowed_base)?;
        }
        if let Some(path) = &self.json_output {
            crate::security::sanitize_output_path_with_base(path, allowed_base)?;
        }
        Ok(())
    }

    pub fn validate_with_base(&self, allowed_base: &Path) -> Result<(), VecEyesError> {
        self.validate_paths_against(allowed_base)?;
        self.validate()
    }

    pub fn apply_to_builder(&self, mut builder: ClassifierBuilder) -> ClassifierBuilder {
        crate::nlp::set_security_normalization_enabled(
            self.pipeline
                .security_normalize_obfuscation
                .unwrap_or(false),
        );
        builder = builder
            .nlp(self.pipeline.nlp.clone())
            .hot_path(self.data.hot_test_path.clone())
            .cold_path(self.data.cold_test_path.clone())
            .hot_label(
                self.data
                    .hot_label
                    .clone()
                    .unwrap_or(ClassificationLabel::WebAttack),
            )
            .cold_label(
                self.data
                    .cold_label
                    .clone()
                    .unwrap_or(ClassificationLabel::RawData),
            )
            .recursive(self.data.recursive_way.is_on())
            .threads(self.pipeline.threads)
            .embedding_dimensions(self.pipeline.embedding_dimensions.unwrap_or(32));
        builder = self.model.apply_to_builder(builder);
        builder.advanced_config(
            self.model
                .to_advanced_config(self.pipeline.threads, self.pipeline.embedding_dimensions),
        )
    }

    pub fn from_yaml_path<P: AsRef<Path>>(path: P) -> Result<Self, VecEyesError> {
        let path_ref = path.as_ref();
        let resolved_path = if path_ref.is_absolute() {
            path_ref.to_path_buf()
        } else {
            std::env::current_dir()?.join(path_ref)
        };
        let allowed_base = resolved_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::from_yaml_path_with_base(&resolved_path, &allowed_base)
    }

    pub fn from_yaml_path_with_base<P: AsRef<Path>>(
        path: P,
        allowed_base: &Path,
    ) -> Result<Self, VecEyesError> {
        let metadata = std::fs::metadata(&path)?;
        if metadata.len() > MAX_RULES_FILE_BYTES {
            return Err(VecEyesError::invalid_config(
                "config::RulesFile::from_yaml_path_with_base",
                format!(
                    "rules file {} exceeds the maximum allowed size of {} bytes",
                    path.as_ref().display(),
                    MAX_RULES_FILE_BYTES
                ),
            ));
        }
        let content = std::fs::read_to_string(&path)?;
        let value: serde_yaml::Value = serde_yaml::from_str(&content)?;
        if !matches!(value, serde_yaml::Value::Mapping(_)) {
            return Err(VecEyesError::invalid_config(
                "config::RulesFile::from_yaml_path_with_base",
                format!(
                    "expected a YAML mapping at root for {}",
                    path.as_ref().display()
                ),
            ));
        }
        let rules: Self = serde_yaml::from_value(value)?;
        rules.validate_with_base(allowed_base)?;
        Ok(rules)
    }
}
