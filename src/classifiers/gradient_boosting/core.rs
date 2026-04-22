use ndarray::Axis;
use rayon::prelude::*;

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

/// Fit a regression stump using sorted quantile splits.
///
/// For each feature the unique values are sorted, midpoints between adjacent
/// unique values form the candidate thresholds (up to 32 candidates for
/// efficiency — equivalent to a 32-bin histogram).  This is deterministic and
/// finds strictly better splits than random jitter over `[min, max]`.
fn fit_regression_stump(x: &DenseMatrix, residual: &[f32]) -> RegStump {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut best = RegStump { feature: 0, threshold: 0.0, left_value: 0.0, right_value: 0.0 };
    let mut best_loss = f32::INFINITY;
    let lambda = 1.0f32; // L2 leaf regulariser

    for feature in 0..cols {
        // Collect sorted (value, residual_index) pairs for this feature
        let mut order: Vec<usize> = (0..rows).collect();
        order.sort_by(|&a, &b| x[[a, feature]].partial_cmp(&x[[b, feature]]).unwrap_or(std::cmp::Ordering::Equal));

        // Build midpoint candidates between adjacent unique values (max 32)
        let mut candidates: Vec<f32> = Vec::new();
        let mut prev = x[[order[0], feature]];
        for &row in order.iter().skip(1) {
            let v = x[[row, feature]];
            if (v - prev).abs() > 1e-7 {
                candidates.push((prev + v) * 0.5);
                prev = v;
            }
        }
        if candidates.is_empty() { continue; }

        // Sub-sample to at most 32 candidates for efficiency with wide feature sets
        let step = (candidates.len() / 32).max(1);
        let candidates: Vec<f32> = candidates.into_iter().step_by(step).collect();

        for threshold in candidates {
            let (mut l_sum, mut l_cnt, mut r_sum, mut r_cnt) = (0.0f32, 0usize, 0.0f32, 0usize);
            for r in 0..rows {
                if x[[r, feature]] <= threshold { l_sum += residual[r]; l_cnt += 1; }
                else                            { r_sum += residual[r]; r_cnt += 1; }
            }
            if l_cnt == 0 || r_cnt == 0 { continue; }
            let l_mean = l_sum / (l_cnt as f32 + lambda);
            let r_mean = r_sum / (r_cnt as f32 + lambda);
            let loss: f32 = (0..rows)
                .map(|r| {
                    let pred = if x[[r, feature]] <= threshold { l_mean } else { r_mean };
                    let d = residual[r] - pred;
                    d * d
                })
                .sum();
            if loss < best_loss {
                best_loss = loss;
                best = RegStump { feature, threshold, left_value: l_mean, right_value: r_mean };
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
        let mut stumps = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let residual: Vec<f32> = logits
                .iter()
                .zip(y.iter())
                .map(|(&logit, &target)| target - 1.0 / (1.0 + (-logit).exp()))
                .collect();
            let stump = fit_regression_stump(x, &residual);
            for row in 0..y.len() {
                logits[row] += learning_rate * stump.predict_row(x.index_axis(Axis(0), row).as_slice().unwrap_or(&[]));
            }
            stumps.push(stump);
        }
        Self { stumps, base_score, learning_rate }
    }

    fn predict_score(&self, row: &[f32]) -> f32 {
        let logit = self.stumps.iter().fold(self.base_score, |acc, s| acc + self.learning_rate * s.predict_row(row));
        1.0 / (1.0 + (-logit).exp())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GradientBoostingModel {
    encoder: LabelEncoder,
    models: Vec<BinaryGradientBoosting>,
}

impl GradientBoostingModel {
    pub(crate) fn fit(
        matrix: &DenseMatrix,
        samples: &[TrainingSample],
        rounds: usize,
        learning_rate: f32,
        threads: Option<usize>,
    ) -> Result<Self, VecEyesError> {
        let encoder = LabelEncoder::fit(samples);
        let y_idx: Vec<usize> = samples
            .iter()
            .map(|s| encoder.encode(&s.label))
            .collect::<Result<_, _>>()?;
        let models: Vec<BinaryGradientBoosting> = install_pool(threads, || {
            (0..encoder.labels.len())
                .into_par_iter()
                .map(|class_id| {
                    let targets: Vec<f32> = y_idx.iter().map(|&i| if i == class_id { 1.0 } else { 0.0 }).collect();
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
            .map(|(idx, model)| (self.encoder.decode(idx), model.predict_score(&row).max(1e-9).ln()))
            .collect();
        softmax_scores(&raw)
    }
}
