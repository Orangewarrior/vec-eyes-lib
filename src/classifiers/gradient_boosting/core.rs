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
    #[inline(always)]
    fn predict(&self, xv: f32) -> f32 {
        if xv <= self.threshold { self.left_value } else { self.right_value }
    }
}

/// Regression stump fitted via sorted quantile midpoints (≤ 32 per feature).
///
/// Extracting each feature column into a contiguous `Vec<f32>` before scanning
/// converts the otherwise stride-D memory access pattern of row-major ndarray
/// into a sequential scan, enabling hardware prefetching and SIMD auto-vec.
fn fit_regression_stump(x: &DenseMatrix, residual: &[f32]) -> RegStump {
    let rows = x.shape()[0];
    let cols = x.shape()[1];
    let mut best = RegStump { feature: 0, threshold: 0.0, left_value: 0.0, right_value: 0.0 };
    let mut best_loss = f32::INFINITY;
    let lambda = 1.0f32;

    for feature in 0..cols {
        // Contiguous column copy → sequential memory access in all inner loops.
        let col: Vec<f32> = x.column(feature).iter().copied().collect();

        let mut order: Vec<usize> = (0..rows).collect();
        order.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap_or(std::cmp::Ordering::Equal));

        // Midpoint candidates between adjacent unique values (max 32).
        let mut candidates: Vec<f32> = Vec::new();
        let mut prev = col[order[0]];
        for &r in order.iter().skip(1) {
            let v = col[r];
            if (v - prev).abs() > 1e-7 {
                candidates.push((prev + v) * 0.5);
                prev = v;
            }
        }
        if candidates.is_empty() { continue; }
        let step = (candidates.len() / 32).max(1);
        let candidates: Vec<f32> = candidates.into_iter().step_by(step).collect();

        for threshold in candidates {
            // Single sequential pass: accumulate left/right sums (SIMD-friendly).
            let (mut l_sum, mut l_cnt, mut r_sum, mut r_cnt) =
                (0.0f32, 0usize, 0.0f32, 0usize);
            for (&xv, &rv) in col.iter().zip(residual.iter()) {
                if xv <= threshold { l_sum += rv; l_cnt += 1; }
                else               { r_sum += rv; r_cnt += 1; }
            }
            if l_cnt == 0 || r_cnt == 0 { continue; }
            let l_mean = l_sum / (l_cnt as f32 + lambda);
            let r_mean = r_sum / (r_cnt as f32 + lambda);

            // Loss: one more sequential pass over the contiguous column.
            let loss: f32 = col.iter().zip(residual.iter()).map(|(&xv, &rv)| {
                let pred = if xv <= threshold { l_mean } else { r_mean };
                let d = rv - pred;
                d * d
            }).sum();

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

            // Logit update using the pre-extracted column — one sequential pass.
            let col: Vec<f32> = x.column(stump.feature).iter().copied().collect();
            let lr = learning_rate;
            for (logit, &xv) in logits.iter_mut().zip(col.iter()) {
                *logit += lr * stump.predict(xv);
            }
            stumps.push(stump);
        }
        Self { stumps, base_score, learning_rate }
    }

    fn predict_score(&self, row: &[f32]) -> f32 {
        let logit = self.stumps.iter().fold(self.base_score, |acc, s| {
            acc + self.learning_rate * s.predict(row[s.feature])
        });
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
                    let targets: Vec<f32> = y_idx
                        .iter()
                        .map(|&i| if i == class_id { 1.0 } else { 0.0 })
                        .collect();
                    BinaryGradientBoosting::fit(matrix, &targets, rounds, learning_rate)
                })
                .collect()
        });
        Ok(Self { encoder, models })
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.row(0);
        let row_slice = row.as_slice().unwrap_or(&[]);
        let raw: Vec<(ClassificationLabel, f32)> = self
            .models
            .iter()
            .enumerate()
            .map(|(idx, model)| {
                (self.encoder.decode(idx), model.predict_score(row_slice).max(1e-9).ln())
            })
            .collect();
        softmax_scores(&raw)
    }
}
