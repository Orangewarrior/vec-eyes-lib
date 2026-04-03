use ndarray::Axis;
use rayon::prelude::*;
use rand::{rngs::StdRng, RngExt, SeedableRng};

use crate::advanced_models::LabelEncoder;
use crate::classifier::softmax_scores;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

#[derive(Debug, Clone)]
struct RegStump {
    feature: usize,
    threshold: f32,
    left_value: f32,
    right_value: f32,
}

impl RegStump {
    fn predict_row(&self, row: &[f32]) -> f32 {
        if row[self.feature] <= self.threshold { self.left_value } else { self.right_value }
    }
}

fn fit_regression_stump(x: &DenseMatrix, residual: &[f32]) -> RegStump {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut best = RegStump { feature: 0, threshold: 0.0, left_value: 0.0, right_value: 0.0 };
    let mut best_loss = f32::INFINITY;
    let mut rng = StdRng::seed_from_u64(0x51A7E);

    for feature in 0..cols {
        let min_v = (0..rows).map(|r| x[[r, feature]]).fold(f32::INFINITY, f32::min);
        let max_v = (0..rows).map(|r| x[[r, feature]]).fold(f32::NEG_INFINITY, f32::max);
        if !min_v.is_finite() || !max_v.is_finite() || (max_v - min_v).abs() < 1e-6 {
            continue;
        }
        let candidates = 12usize.min(rows.max(1));
        let thresholds: Vec<f32> = (0..candidates)
            .map(|idx| {
                let jitter = rng.random_range(0.0..1.0);
                min_v + ((idx as f32 + jitter) / candidates as f32) * (max_v - min_v)
            })
            .collect();
        for threshold in thresholds {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for row in 0..rows {
                if x[[row, feature]] <= threshold { left.push(residual[row]); } else { right.push(residual[row]); }
            }
            if left.is_empty() || right.is_empty() { continue; }
            let left_mean = left.iter().sum::<f32>() / left.len() as f32;
            let right_mean = right.iter().sum::<f32>() / right.len() as f32;
            let mut loss = 0.0;
            for row in 0..rows {
                let pred = if x[[row, feature]] <= threshold { left_mean } else { right_mean };
                let diff = residual[row] - pred;
                loss += diff * diff;
            }
            if loss < best_loss {
                best_loss = loss;
                best = RegStump { feature, threshold, left_value: left_mean, right_value: right_mean };
            }
        }
    }
    best
}

#[derive(Debug, Clone)]
struct BinaryGradientBoosting {
    stumps: Vec<RegStump>,
    base_score: f32,
    learning_rate: f32,
}

impl BinaryGradientBoosting {
    fn fit(x: &DenseMatrix, y: &[f32], rounds: usize, learning_rate: f32) -> Self {
        let positive = y.iter().sum::<f32>().max(1.0);
        let negative = (y.len() as f32 - positive).max(1.0);
        let base_score = (positive / negative).ln();
        let mut logits = vec![base_score; y.len()];
        let mut stumps = Vec::new();
        for _ in 0..rounds {
            let residual: Vec<f32> = (0..y.len())
                .map(|idx| {
                    let p = 1.0 / (1.0 + (-logits[idx]).exp());
                    y[idx] - p
                })
                .collect();
            let stump = fit_regression_stump(x, &residual);
            for row in 0..y.len() {
                let row_vec = x.index_axis(Axis(0), row).to_vec();
                logits[row] += learning_rate * stump.predict_row(&row_vec);
            }
            stumps.push(stump);
        }
        Self { stumps, base_score, learning_rate }
    }

    fn predict_score(&self, row: &[f32]) -> f32 {
        let mut logit = self.base_score;
        for stump in &self.stumps {
            logit += self.learning_rate * stump.predict_row(row);
        }
        1.0 / (1.0 + (-logit).exp())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GradientBoostingModel {
    encoder: LabelEncoder,
    models: Vec<BinaryGradientBoosting>,
}

impl GradientBoostingModel {
    pub(crate) fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], rounds: usize, learning_rate: f32, threads: Option<usize>) -> Result<Self, VecEyesError> {
        let encoder = LabelEncoder::fit(samples);
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect::<Result<_, _>>()?;
        let models: Vec<BinaryGradientBoosting> = install_pool(threads, || {
            (0..encoder.labels.len())
                .into_par_iter()
                .map(|class_id| {
                    let targets: Vec<f32> = y_idx.iter().map(|&idx| if idx == class_id { 1.0 } else { 0.0 }).collect();
                    BinaryGradientBoosting::fit(matrix, &targets, rounds, learning_rate)
                })
                .collect()
        });
        Ok(Self { encoder, models })
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0).to_vec();
        let raw: Vec<(ClassificationLabel, f32)> = self.models
            .iter()
            .enumerate()
            .map(|(idx, model)| (self.encoder.decode(idx), model.predict_score(&row).max(1e-6).ln()))
            .collect();
        softmax_scores(&raw)
    }
}
