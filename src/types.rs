use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::labels::ClassificationLabel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampleOrigin {
    Inline,
    DatasetFile(PathBuf),
    ClassifyObject(PathBuf),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSample {
    pub label: ClassificationLabel,
    pub text: String,
    pub source_name: String,
    pub origin: SampleOrigin,
}

impl RawSample {
    pub fn new(label: ClassificationLabel, text: impl Into<String>, source_name: impl Into<String>) -> Self {
        Self {
            label,
            text: text.into(),
            source_name: source_name.into(),
            origin: SampleOrigin::Inline,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationOutcome {
    pub label: ClassificationLabel,
    pub score: f64,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub title_object: String,
    pub dataset_name: String,
    pub classify_names_list: Vec<ClassificationLabel>,
    pub date_of_occurrence: DateTime<Utc>,
}
