use crate::error::VecEyesError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationRecord {
    pub title_object: String,
    pub name_file_dataset: String,
    pub classify_names_list: String,
    pub date_of_occurrence: DateTime<Utc>,
    pub score_percent: f32,
    pub match_titles: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    pub report_name: String,
    pub records: Vec<ClassificationRecord>,
}

impl ClassificationReport {
    pub fn new(report_name: String) -> Self {
        Self { report_name, records: Vec::new() }
    }

    pub fn write_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let mut writer = csv::WriterBuilder::new().delimiter(b';').from_path(path)?;
        writer.write_record([
            "title object",
            "name_file_dataset",
            "classify_names_list",
            "date of occurrence",
            "score_percent",
            "match_titles",
        ])?;
        for record in &self.records {
            writer.write_record([
                &record.title_object,
                &record.name_file_dataset,
                &record.classify_names_list,
                &record.date_of_occurrence.to_rfc3339(),
                &format!("{:.2}", record.score_percent),
                &record.match_titles,
            ])?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn write_json<P: AsRef<Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content).map_err(|err| {
            VecEyesError::invalid_config(
                "report::ClassificationReport::write_json",
                format!("failed to write JSON report to {}: {err}", path.display()),
            )
        })?;
        Ok(())
    }
}
