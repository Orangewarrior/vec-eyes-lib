//! Reporting helpers.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::error::VecEyesResult;
use crate::labels::ClassificationLabel;
use crate::types::AuditRecord;
use crate::alerts::AlertRuleMatch;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    pub object_title: String,
    pub dataset_name: String,
    pub classifications: Vec<(ClassificationLabel, f64)>,
    pub alert_matches: Vec<AlertRuleMatch>,
    pub event_time: chrono::DateTime<Utc>,
}

pub struct CsvReportWriter {
    writer: csv::Writer<BufWriter<File>>,
}

impl CsvReportWriter {
    pub fn new(path: impl AsRef<Path>) -> VecEyesResult<Self> {
        let file = File::create(path)?;
        let writer = csv::WriterBuilder::new()
            .delimiter(b';')
            .from_writer(BufWriter::new(file));
        Ok(Self { writer })
    }

    pub fn write_report(&mut self, report: &ClassificationReport) -> VecEyesResult<()> {
        let record = AuditRecord {
            title_object: report.object_title.clone(),
            dataset_name: report.dataset_name.clone(),
            classify_names_list: report.classifications.iter().map(|(label, _)| *label).collect(),
            date_of_occurrence: report.event_time,
        };
        self.writer.serialize(record)?;
        self.writer.flush()?;
        Ok(())
    }
}

pub struct JsonReportWriter {
    path: std::path::PathBuf,
    reports: Vec<ClassificationReport>,
}

impl JsonReportWriter {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf(), reports: Vec::new() }
    }

    pub fn push(&mut self, report: ClassificationReport) -> VecEyesResult<()> {
        self.reports.push(report);
        let file = File::create(&self.path)?;
        serde_json::to_writer_pretty(BufWriter::new(file), &self.reports)?;
        Ok(())
    }
}
