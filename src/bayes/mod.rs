//! Naive Bayes classifier.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use ndarray::Array1;

use crate::error::{VecEyesError, VecEyesResult};
use crate::labels::ClassificationLabel;
use crate::nlp::{FeatureMatrix, NlpPipeline, RepresentationKind};
use crate::types::RawSample;

#[derive(Debug, Clone)]
pub struct BayesBuilder {
    alpha: f64,
    pipeline: Option<NlpPipeline>,
}

impl Default for BayesBuilder {
    fn default() -> Self {
        Self { alpha: 1.0, pipeline: None }
    }
}

impl BayesBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn alpha(mut self, value: f64) -> Self { self.alpha = value; self }
    pub fn pipeline(mut self, pipeline: NlpPipeline) -> Self { self.pipeline = Some(pipeline); self }

    pub fn build(self) -> VecEyesResult<NaiveBayesClassifier> {
        let pipeline = self.pipeline.ok_or(VecEyesError::InvalidConfig("Bayes requires an NLP pipeline"))?;
        match pipeline.representation() {
            RepresentationKind::Count | RepresentationKind::TfIdf => {}
            other => {
                return Err(VecEyesError::IncompatibleRepresentation {
                    classifier: "NaiveBayes",
                    representation: other.as_str(),
                });
            }
        }
        Ok(NaiveBayesClassifier {
            alpha: self.alpha,
            pipeline,
            class_priors: HashMap::new(),
            feature_log_probs: HashMap::new(),
            feature_names: Vec::new(),
        })
    }

    pub fn fit(self, samples: &[RawSample]) -> VecEyesResult<NaiveBayesClassifier> {
        self.build()?.fit(samples)
    }

    pub fn fit_from_file(self, path: impl AsRef<Path>) -> VecEyesResult<NaiveBayesClassifier> {
        let model = self.build()?;
        let samples = model.pipeline.read_tagged_text_file(path)?;
        model.fit(&samples)
    }

    pub fn fit_from_directories(
        self,
        hot: Option<impl AsRef<Path>>,
        cold: Option<impl AsRef<Path>>,
        hot_label: ClassificationLabel,
    ) -> VecEyesResult<NaiveBayesClassifier> {
        let model = self.build()?;
        let samples = model.pipeline.read_hot_and_cold_directories(hot, cold, hot_label)?;
        model.fit(&samples)
    }
}

#[derive(Debug, Clone)]
pub struct NaiveBayesClassifier {
    alpha: f64,
    pipeline: NlpPipeline,
    class_priors: HashMap<ClassificationLabel, f64>,
    feature_log_probs: HashMap<ClassificationLabel, Array1<f64>>,
    feature_names: Vec<String>,
}

impl NaiveBayesClassifier {
    pub fn fit(mut self, samples: &[RawSample]) -> VecEyesResult<Self> {
        let matrix = self.pipeline.fit_transform(samples)?;
        self.fit_matrix(&matrix)?;
        Ok(self)
    }

    fn fit_matrix(&mut self, matrix: &FeatureMatrix) -> VecEyesResult<()> {
        if matrix.labels.is_empty() {
            return Err(VecEyesError::EmptyDataset);
        }

        self.feature_names = matrix.feature_names.clone();
        let mut classes = BTreeSet::new();
        for label in &matrix.labels {
            classes.insert(*label);
        }

        for class in classes {
            let mut row_indices = Vec::new();
            for (row_index, label) in matrix.labels.iter().enumerate() {
                if *label == class {
                    row_indices.push(row_index);
                }
            }

            let prior = row_indices.len() as f64 / matrix.labels.len() as f64;
            self.class_priors.insert(class, prior.ln());

            let mut class_feature_sum = Array1::<f64>::zeros(matrix.rows.ncols());
            for row_index in row_indices {
                class_feature_sum += &matrix.rows.row(row_index);
            }

            let smoothed_total = class_feature_sum.sum() + self.alpha * matrix.rows.ncols() as f64;
            let mut log_probs = Array1::<f64>::zeros(matrix.rows.ncols());
            for feature_index in 0..matrix.rows.ncols() {
                let numerator = class_feature_sum[feature_index] + self.alpha;
                log_probs[feature_index] = (numerator / smoothed_total).ln();
            }
            self.feature_log_probs.insert(class, log_probs);
        }

        Ok(())
    }

    pub fn predict_scores(&self, text: &str) -> VecEyesResult<HashMap<ClassificationLabel, f64>> {
        if self.feature_log_probs.is_empty() {
            return Err(VecEyesError::ModelNotFitted);
        }

        let row = self.pipeline.transform_text(text, &self.feature_names)?;
        let vector = row.row(0).to_owned();
        let mut raw_scores = HashMap::new();

        for (label, log_prior) in &self.class_priors {
            if let Some(log_probs) = self.feature_log_probs.get(label) {
                let score = *log_prior + vector.dot(log_probs);
                raw_scores.insert(*label, score);
            }
        }

        Ok(normalize_log_scores(raw_scores))
    }

    pub fn pipeline(&self) -> &NlpPipeline { &self.pipeline }
}

fn normalize_log_scores(raw_scores: HashMap<ClassificationLabel, f64>) -> HashMap<ClassificationLabel, f64> {
    let max_score = raw_scores.values().fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let mut exps = HashMap::new();
    let mut total = 0.0;
    for (label, value) in raw_scores {
        let exp = (value - max_score).exp();
        total += exp;
        exps.insert(label, exp);
    }

    let mut normalized = HashMap::new();
    for (label, value) in exps {
        normalized.insert(label, (value / total) * 100.0);
    }
    normalized
}
