use crate::classifier::MethodKind;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::NlpOption;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

        Ok(())
    }
}
