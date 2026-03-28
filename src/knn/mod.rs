//! KNN classifier for dense vectors.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::{VecEyesError, VecEyesResult};
use crate::labels::ClassificationLabel;
use crate::nlp::{FeatureMatrix, NlpPipeline, RepresentationKind};
use crate::types::RawSample;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Minkowski { p: f64 },
    Cosine,
}

#[derive(Debug, Clone)]
pub struct KnnBuilder {
    k: usize,
    metric: DistanceMetric,
    distance_weighted: bool,
    pipeline: Option<NlpPipeline>,
}

impl Default for KnnBuilder {
    fn default() -> Self {
        Self { k: 3, metric: DistanceMetric::Cosine, distance_weighted: true, pipeline: None }
    }
}

impl KnnBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn k(mut self, value: usize) -> Self { self.k = value.max(1); self }
    pub fn metric(mut self, value: DistanceMetric) -> Self { self.metric = value; self }
    pub fn distance_weighted(mut self, value: bool) -> Self { self.distance_weighted = value; self }
    pub fn pipeline(mut self, pipeline: NlpPipeline) -> Self { self.pipeline = Some(pipeline); self }

    pub fn build(self) -> VecEyesResult<KnnClassifier> {
        let pipeline = self.pipeline.ok_or(VecEyesError::InvalidConfig("KNN requires an NLP pipeline"))?;
        match pipeline.representation() {
            RepresentationKind::Word2Vec | RepresentationKind::FastText => {}
            other => {
                return Err(VecEyesError::IncompatibleRepresentation {
                    classifier: "KNN",
                    representation: other.as_str(),
                });
            }
        }

        Ok(KnnClassifier {
            k: self.k,
            metric: self.metric,
            distance_weighted: self.distance_weighted,
            pipeline,
            train_rows: None,
            train_labels: Vec::new(),
            feature_names: Vec::new(),
        })
    }

    pub fn fit(self, samples: &[RawSample]) -> VecEyesResult<KnnClassifier> { self.build()?.fit(samples) }

    pub fn fit_from_file(self, path: impl AsRef<Path>) -> VecEyesResult<KnnClassifier> {
        let model = self.build()?;
        let samples = model.pipeline.read_tagged_text_file(path)?;
        model.fit(&samples)
    }

    pub fn fit_from_directories(
        self,
        hot: Option<impl AsRef<Path>>,
        cold: Option<impl AsRef<Path>>,
        hot_label: ClassificationLabel,
    ) -> VecEyesResult<KnnClassifier> {
        let model = self.build()?;
        let samples = model.pipeline.read_hot_and_cold_directories(hot, cold, hot_label)?;
        model.fit(&samples)
    }
}

#[derive(Debug, Clone)]
pub struct KnnClassifier {
    k: usize,
    metric: DistanceMetric,
    distance_weighted: bool,
    pipeline: NlpPipeline,
    train_rows: Option<Array2<f64>>,
    train_labels: Vec<ClassificationLabel>,
    feature_names: Vec<String>,
}

impl KnnClassifier {
    pub fn fit(mut self, samples: &[RawSample]) -> VecEyesResult<Self> {
        let matrix = self.pipeline.fit_transform(samples)?;
        self.fit_matrix(matrix);
        Ok(self)
    }

    fn fit_matrix(&mut self, matrix: FeatureMatrix) {
        self.train_rows = Some(matrix.rows);
        self.train_labels = matrix.labels;
        self.feature_names = matrix.feature_names;
    }

    pub fn predict_scores(&self, text: &str) -> VecEyesResult<HashMap<ClassificationLabel, f64>> {
        let train_rows = self.train_rows.as_ref().ok_or(VecEyesError::ModelNotFitted)?;
        let query = self.pipeline.transform_text(text, &self.feature_names)?;
        let query = query.row(0);

        let mut distances = Vec::with_capacity(train_rows.nrows());
        for row_index in 0..train_rows.nrows() {
            let train = train_rows.row(row_index);
            let distance = match self.metric {
                DistanceMetric::Euclidean => euclidean(&query, &train),
                DistanceMetric::Manhattan => manhattan(&query, &train),
                DistanceMetric::Minkowski { p } => minkowski(&query, &train, p),
                DistanceMetric::Cosine => cosine_distance(&query, &train),
            };
            distances.push((distance, self.train_labels[row_index]));
        }
        distances.sort_by(|left, right| left.0.total_cmp(&right.0));

        let mut votes: HashMap<ClassificationLabel, f64> = HashMap::new();
        for (distance, label) in distances.into_iter().take(self.k) {
            let weight = if self.distance_weighted {
                1.0 / (distance + 1e-9)
            } else {
                1.0
            };
            *votes.entry(label).or_insert(0.0) += weight;
        }

        let total: f64 = votes.values().sum();
        if total > 0.0 {
            for value in votes.values_mut() {
                *value = (*value / total) * 100.0;
            }
        }

        Ok(votes)
    }

    pub fn pipeline(&self) -> &NlpPipeline { &self.pipeline }
}

fn euclidean(query: &ndarray::ArrayView1<'_, f64>, train: &ndarray::ArrayView1<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for index in 0..query.len() {
        let delta = query[index] - train[index];
        sum += delta * delta;
    }
    sum.sqrt()
}

fn manhattan(query: &ndarray::ArrayView1<'_, f64>, train: &ndarray::ArrayView1<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for index in 0..query.len() {
        sum += (query[index] - train[index]).abs();
    }
    sum
}

fn minkowski(query: &ndarray::ArrayView1<'_, f64>, train: &ndarray::ArrayView1<'_, f64>, p: f64) -> f64 {
    let power = p.max(1.0);
    let mut sum = 0.0;
    for index in 0..query.len() {
        sum += (query[index] - train[index]).abs().powf(power);
    }
    sum.powf(1.0 / power)
}

fn cosine_distance(query: &ndarray::ArrayView1<'_, f64>, train: &ndarray::ArrayView1<'_, f64>) -> f64 {
    let mut dot = 0.0;
    let mut query_norm = 0.0;
    let mut train_norm = 0.0;
    for index in 0..query.len() {
        dot += query[index] * train[index];
        query_norm += query[index] * query[index];
        train_norm += train[index] * train[index];
    }
    let denom = query_norm.sqrt() * train_norm.sqrt();
    if denom <= 0.0 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}
