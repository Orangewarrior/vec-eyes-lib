use crate::advanced_models::{RandomForestMaxFeatures, RandomForestMode, SvmKernel};
use crate::classifier::MethodKind;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::NlpOption;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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

impl RulesFile {
    pub fn validate(&self) -> Result<(), VecEyesError> {
        if self.threads == Some(0) {
            return Err(VecEyesError::InvalidConfig("threads must be >= 1".into()));
        }

        if !self.hot_test_path.exists() {
            return Err(VecEyesError::InvalidConfig(format!(
                "hot_test_path does not exist: {}",
                self.hot_test_path.display()
            )));
        }

        if !self.cold_test_path.exists() {
            return Err(VecEyesError::InvalidConfig(format!(
                "cold_test_path does not exist: {}",
                self.cold_test_path.display()
            )));
        }

        for extra in &self.extra_match {
            if !extra.path.exists() {
                return Err(VecEyesError::InvalidConfig(format!(
                    "extra_match path does not exist: {}",
                    extra.path.display()
                )));
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
                if self.random_forest_oob_score == Some(true) && self.random_forest_bootstrap == Some(false) {
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
                    SvmKernel::Linear | SvmKernel::Rbf | SvmKernel::Polynomial | SvmKernel::Sigmoid => {}
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

    /// Loads a YAML rules file from disk and validates it before returning it.
    pub fn from_yaml_path<P: AsRef<Path>>(path: P) -> Result<Self, VecEyesError> {
        let content = std::fs::read_to_string(path)?;
        let rules: Self = serde_yaml::from_str(&content)?;
        rules.validate()?;
        Ok(rules)
    }

}
