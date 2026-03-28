//! High-level classification engine.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use chrono::Utc;

use crate::alerts::{AlertMatcher, AlertRuleMatch};
use crate::bayes::NaiveBayesClassifier;
use crate::error::VecEyesResult;
use crate::filesystem::collect_files_recursively;
use crate::knn::KnnClassifier;
use crate::labels::ClassificationLabel;
use crate::report::{ClassificationReport, CsvReportWriter, JsonReportWriter};

#[derive(Debug, Clone)]
pub enum ModelHandle {
    Bayes(NaiveBayesClassifier),
    Knn(KnnClassifier),
}

impl From<NaiveBayesClassifier> for ModelHandle {
    fn from(value: NaiveBayesClassifier) -> Self { Self::Bayes(value) }
}

impl From<KnnClassifier> for ModelHandle {
    fn from(value: KnnClassifier) -> Self { Self::Knn(value) }
}

#[derive(Default)]
pub struct OutputWriters {
    pub csv: Option<CsvReportWriter>,
    pub json: Option<JsonReportWriter>,
}

impl OutputWriters {
    pub fn disabled() -> Self { Self::default() }
}

pub struct EngineBuilder {
    model: Option<ModelHandle>,
    alerts: Option<AlertMatcher>,
    output: OutputWriters,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self { model: None, alerts: None, output: OutputWriters::disabled() }
    }
}

impl EngineBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn model(mut self, value: ModelHandle) -> Self { self.model = Some(value); self }
    pub fn alerts(mut self, value: AlertMatcher) -> Self { self.alerts = Some(value); self }
    pub fn output(mut self, value: OutputWriters) -> Self { self.output = value; self }
    pub fn build(self) -> VecEyesResult<VecEyesEngine> {
        Ok(VecEyesEngine {
            model: self.model.expect("engine requires a model"),
            alerts: self.alerts,
            output: self.output,
        })
    }
}

pub struct VecEyesEngine {
    model: ModelHandle,
    alerts: Option<AlertMatcher>,
    output: OutputWriters,
}

impl VecEyesEngine {
    pub fn classify_text(&self, text: &str, object_title: &str) -> VecEyesResult<ClassificationReport> {
        let mut scores = match &self.model {
            ModelHandle::Bayes(model) => model.predict_scores(text)?,
            ModelHandle::Knn(model) => model.predict_scores(text)?,
        };

        let alert_matches = if let Some(alerts) = &self.alerts {
            alerts.boost_scores(text, &mut scores)?
        } else {
            Vec::<AlertRuleMatch>::new()
        };

        let mut classifications: Vec<(ClassificationLabel, f64)> = scores.into_iter().collect();
        classifications.sort_by(|left, right| right.1.total_cmp(&left.1));
        classifications.retain(|(_, score)| *score >= 5.0);

        Ok(ClassificationReport {
            object_title: object_title.to_string(),
            dataset_name: object_title.to_string(),
            classifications,
            alert_matches,
            event_time: Utc::now(),
        })
    }

    pub fn classify_file(&mut self, path: impl AsRef<Path>) -> VecEyesResult<ClassificationReport> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)?;
        let title = path.file_name().map(|v| v.to_string_lossy().to_string()).unwrap_or_else(|| path.display().to_string());
        let mut report = self.classify_text(&text, &title)?;
        report.dataset_name = path.display().to_string();
        self.write_outputs(&report)?;
        Ok(report)
    }

    pub fn classify_directory(&mut self, path: impl AsRef<Path>) -> VecEyesResult<Vec<ClassificationReport>> {
        let mut reports = Vec::new();
        for file in collect_files_recursively(path)? {
            reports.push(self.classify_file(file)?);
        }
        Ok(reports)
    }

    fn write_outputs(&mut self, report: &ClassificationReport) -> VecEyesResult<()> {
        if let Some(csv) = &mut self.output.csv {
            csv.write_report(report)?;
        }
        if let Some(json) = &mut self.output.json {
            json.push(report.clone())?;
        }
        Ok(())
    }
}
