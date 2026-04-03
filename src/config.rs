use crate::advanced_models::{
    AdvancedModelConfig, GradientBoostingConfig, IsolationForestConfig,
    LogisticRegressionConfig, RandomForestConfig, RandomForestMaxFeatures, RandomForestMode,
    SvmConfig, SvmKernel,
};
use crate::classifier::{ClassifierBuilder, MethodKind};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::NlpOption;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MAX_RULES_FILE_BYTES: u64 = 512 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecursiveMode {
    #[serde(alias = "ON", alias = "on")]
    On,
    #[serde(alias = "OFF", alias = "off")]
    Off,
}

impl Default for RecursiveMode {
    fn default() -> Self {
        Self::On
    }
}

impl RecursiveMode {
    pub fn is_on(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreSumMode {
    #[serde(alias = "ON", alias = "on")]
    On,
    #[serde(alias = "OFF", alias = "off")]
    Off,
}

impl Default for ScoreSumMode {
    fn default() -> Self {
        Self::Off
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulesFile {
    pub report_name: Option<String>,
    pub method: MethodKind,
    pub nlp: NlpOption,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub embedding_dimensions: Option<usize>,
    #[serde(default)]
    pub security_normalize_obfuscation: Option<bool>,
    #[serde(default)]
    pub csv_output: Option<PathBuf>,
    #[serde(default)]
    pub json_output: Option<PathBuf>,
    #[serde(default)]
    pub recursive_way: RecursiveMode,
    pub hot_test_path: PathBuf,
    pub cold_test_path: PathBuf,
    #[serde(default)]
    pub hot_label: Option<ClassificationLabel>,
    #[serde(default)]
    pub cold_label: Option<ClassificationLabel>,
    #[serde(default)]
    pub score_sum: ScoreSumMode,
    #[serde(default)]
    pub extra_match: Vec<ExtraMatchConfig>,
    #[serde(default)]
    pub k: Option<usize>,
    #[serde(default)]
    pub p: Option<f32>,
    #[serde(default)]
    pub logistic_learning_rate: Option<f32>,
    #[serde(default)]
    pub logistic_epochs: Option<usize>,
    #[serde(default)]
    pub logistic_lambda: Option<f32>,
    #[serde(default)]
    pub random_forest_n_trees: Option<usize>,
    #[serde(default)]
    pub random_forest_mode: Option<RandomForestMode>,
    #[serde(default)]
    pub random_forest_max_depth: Option<usize>,
    #[serde(default)]
    pub random_forest_max_features: Option<RandomForestMaxFeatures>,
    #[serde(default)]
    pub random_forest_min_samples_split: Option<usize>,
    #[serde(default)]
    pub random_forest_min_samples_leaf: Option<usize>,
    #[serde(default)]
    pub random_forest_bootstrap: Option<bool>,
    #[serde(default)]
    pub random_forest_oob_score: Option<bool>,
    #[serde(default)]
    pub random_forest_seed: Option<u64>,
    #[serde(default)]
    pub svm_kernel: Option<SvmKernel>,
    #[serde(default)]
    pub svm_c: Option<f32>,
    #[serde(default)]
    pub svm_learning_rate: Option<f32>,
    #[serde(default)]
    pub svm_epochs: Option<usize>,
    #[serde(default)]
    pub svm_gamma: Option<f32>,
    #[serde(default)]
    pub svm_degree: Option<usize>,
    #[serde(default)]
    pub svm_coef0: Option<f32>,
    #[serde(default)]
    pub gradient_boosting_n_estimators: Option<usize>,
    #[serde(default)]
    pub gradient_boosting_learning_rate: Option<f32>,
    #[serde(default)]
    pub gradient_boosting_max_depth: Option<usize>,
    #[serde(default)]
    pub isolation_forest_n_trees: Option<usize>,
    #[serde(default)]
    pub isolation_forest_contamination: Option<f32>,
    #[serde(default)]
    pub isolation_forest_subsample_size: Option<usize>,
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
        if self.threads == Some(0) {
            return Err(VecEyesError::invalid_config(
                "config::RulesFile::validate",
                "threads must be >= 1",
            ));
        }

        if self.embedding_dimensions == Some(0) {
            return Err(VecEyesError::InvalidConfig(
                "YAML validation error: embedding_dimensions must be >= 1".into(),
            ));
        }

        for extra in &self.extra_match {
            if extra.score_add_points < 0.0 {
                return Err(VecEyesError::invalid_config(
                    "config::RulesFile::validate",
                    format!("extra_match score_add_points must be >= 0.0 for {}", extra.path.display()),
                ));
            }
        }


        if self.method.is_knn() {
            let k = self.k.ok_or_else(|| {
                VecEyesError::InvalidConfig(
                    "YAML validation error: field 'k' is required for every KNN method".into(),
                )
            })?;

            if k == 0 {
                return Err(VecEyesError::InvalidConfig(
                    "YAML validation error: field 'k' must be >= 1".into(),
                ));
            }

            if self.method.requires_p() {
                let p = self.p.ok_or_else(|| {
                    VecEyesError::InvalidConfig(
                        "YAML validation error: field 'p' is required for knn-minkowski".into(),
                    )
                })?;

                if p <= 0.0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: field 'p' must be > 0 for knn-minkowski".into(),
                    ));
                }
            }
        }

        match self.method {
            MethodKind::LogisticRegression => {
                let lr = self.logistic_learning_rate.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'logistic_learning_rate' is required for logistic regression".into(),
                ))?;
                let epochs = self.logistic_epochs.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'logistic_epochs' is required for logistic regression".into(),
                ))?;
                if lr <= 0.0 || epochs == 0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: logistic_learning_rate must be > 0 and logistic_epochs must be >= 1".into(),
                    ));
                }
            }
            MethodKind::RandomForest => {
                let n_trees = self.random_forest_n_trees.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'random_forest_n_trees' is required for random forest".into(),
                ))?;
                if n_trees == 0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: random_forest_n_trees must be >= 1".into(),
                    ));
                }
                if let Some(min_leaf) = self.random_forest_min_samples_leaf {
                    if min_leaf == 0 {
                        return Err(VecEyesError::InvalidConfig(
                            "YAML validation error: random_forest_min_samples_leaf must be >= 1".into(),
                        ));
                    }
                }
                if self.random_forest_oob_score == Some(true)
                    && self.random_forest_bootstrap == Some(false)
                {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: random_forest_oob_score requires random_forest_bootstrap = true".into(),
                    ));
                }
            }
            MethodKind::Svm => {
                let kernel = self.svm_kernel.clone().ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'svm_kernel' is required for svm".into(),
                ))?;
                let c = self.svm_c.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'svm_c' is required for svm".into(),
                ))?;
                if c <= 0.0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: svm_c must be > 0".into(),
                    ));
                }
                match kernel {
                    SvmKernel::Linear
                    | SvmKernel::Rbf
                    | SvmKernel::Polynomial
                    | SvmKernel::Sigmoid => {}
                }
            }
            MethodKind::GradientBoosting => {
                let n = self.gradient_boosting_n_estimators.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'gradient_boosting_n_estimators' is required for gradient boosting".into(),
                ))?;
                let lr = self.gradient_boosting_learning_rate.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'gradient_boosting_learning_rate' is required for gradient boosting".into(),
                ))?;
                if n == 0 || lr <= 0.0 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: gradient_boosting_n_estimators must be >= 1 and gradient_boosting_learning_rate must be > 0".into(),
                    ));
                }
            }
            MethodKind::IsolationForest => {
                let n = self.isolation_forest_n_trees.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'isolation_forest_n_trees' is required for isolation forest".into(),
                ))?;
                let contamination = self.isolation_forest_contamination.ok_or_else(|| VecEyesError::InvalidConfig(
                    "YAML validation error: field 'isolation_forest_contamination' is required for isolation forest".into(),
                ))?;
                if n == 0 || contamination <= 0.0 || contamination >= 0.5 {
                    return Err(VecEyesError::InvalidConfig(
                        "YAML validation error: isolation_forest_n_trees must be >= 1 and contamination must be in (0, 0.5)".into(),
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }

    pub fn validate_paths_against(&self, allowed_base: &Path) -> Result<(), VecEyesError> {
        validate_maybe_relative_path(&self.hot_test_path, allowed_base)?;
        validate_maybe_relative_path(&self.cold_test_path, allowed_base)?;
        for extra in &self.extra_match {
            validate_maybe_relative_path(&extra.path, allowed_base)?;
        }
        Ok(())
    }

    pub fn validate_with_base(&self, allowed_base: &Path) -> Result<(), VecEyesError> {
        self.validate_paths_against(allowed_base)?;
        self.validate()
    }

    pub fn advanced_model_config(&self) -> AdvancedModelConfig {        AdvancedModelConfig {
            threads: self.threads,
            embedding_dimensions: self.embedding_dimensions,
            logistic: match (self.logistic_learning_rate, self.logistic_epochs) {
                (Some(learning_rate), Some(epochs)) => Some(LogisticRegressionConfig {
                    learning_rate,
                    epochs,
                    lambda: self.logistic_lambda.unwrap_or(1e-3),
                }),
                _ => None,
            },
            random_forest: self.random_forest_n_trees.map(|n_trees| RandomForestConfig {
                mode: self.random_forest_mode.clone().unwrap_or(RandomForestMode::Standard),
                n_trees,
                max_depth: self.random_forest_max_depth.unwrap_or(6),
                max_features: self
                    .random_forest_max_features
                    .clone()
                    .unwrap_or(RandomForestMaxFeatures::Sqrt),
                min_samples_split: self.random_forest_min_samples_split.unwrap_or(2),
                min_samples_leaf: self.random_forest_min_samples_leaf.unwrap_or(1),
                bootstrap: self.random_forest_bootstrap.unwrap_or(true),
                oob_score: self.random_forest_oob_score.unwrap_or(false),
                random_seed: self.random_forest_seed,
            }),
            svm: match (self.svm_kernel.clone(), self.svm_c) {
                (Some(kernel), Some(c)) => Some(SvmConfig {
                    kernel,
                    c,
                    learning_rate: self.svm_learning_rate.unwrap_or(0.08),
                    epochs: self.svm_epochs.unwrap_or(40),
                    gamma: self.svm_gamma.unwrap_or(0.35),
                    degree: self.svm_degree.unwrap_or(2),
                    coef0: self.svm_coef0.unwrap_or(0.0),
                }),
                _ => None,
            },
            gradient_boosting: match (
                self.gradient_boosting_n_estimators,
                self.gradient_boosting_learning_rate,
            ) {
                (Some(n_estimators), Some(learning_rate)) => Some(GradientBoostingConfig {
                    n_estimators,
                    learning_rate,
                    max_depth: self.gradient_boosting_max_depth.unwrap_or(1),
                }),
                _ => None,
            },
            isolation_forest: match (
                self.isolation_forest_n_trees,
                self.isolation_forest_contamination,
            ) {
                (Some(n_trees), Some(contamination)) => Some(IsolationForestConfig {
                    n_trees,
                    contamination,
                    subsample_size: self.isolation_forest_subsample_size.unwrap_or(64),
                }),
                _ => None,
            },
        }
    }

    pub fn apply_to_builder(&self, mut builder: ClassifierBuilder) -> ClassifierBuilder {
        crate::nlp::set_security_normalization_enabled(self.security_normalize_obfuscation.unwrap_or(false));
        builder = builder
            .method(self.method.clone())
            .nlp(self.nlp.clone())
            .hot_path(self.hot_test_path.clone())
            .cold_path(self.cold_test_path.clone())
            .hot_label(self.hot_label.clone().unwrap_or(ClassificationLabel::WebAttack))
            .cold_label(self.cold_label.clone().unwrap_or(ClassificationLabel::RawData))
            .recursive(self.recursive_way.is_on())
            .threads(self.threads)
            .embedding_dimensions(self.embedding_dimensions.unwrap_or(32));
        if let Some(k) = self.k {
            builder = builder.k(k);
        }
        if let Some(p) = self.p {
            builder = builder.p(p);
        }
        builder.advanced_config(self.advanced_model_config())
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
