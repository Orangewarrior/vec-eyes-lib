use rayon::prelude::*;

use crate::advanced_models::LabelEncoder;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::math::softmax_scores;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

const MAX_SPLIT_CANDIDATES: usize = 32;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum RegTree {
    Leaf(f32),
    Split {
        feature: usize,
        threshold: f32,
        left: Box<RegTree>,
        right: Box<RegTree>,
    },
}

#[derive(Debug, Clone)]
struct SplitCandidate {
    feature: usize,
    threshold: f32,
    left_value: f32,
    right_value: f32,
    loss: f32,
}

impl RegTree {
    #[inline(always)]
    fn predict(&self, row: &[f32]) -> f32 {
        match self {
            Self::Leaf(value) => *value,
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row.get(*feature).copied().unwrap_or_default() <= *threshold {
                    left.predict(row)
                } else {
                    right.predict(row)
                }
            }
        }
    }
}

fn mean_leaf(residual: &[f32], rows: &[usize]) -> f32 {
    if rows.is_empty() {
        return 0.0;
    }
    let sum: f32 = rows.iter().map(|&row| residual[row]).sum();
    sum / rows.len() as f32
}

fn candidate_thresholds(x: &DenseMatrix, rows: &[usize], feature: usize) -> Vec<f32> {
    let mut values: Vec<f32> = rows.iter().map(|&idx| x[[idx, feature]]).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    if values.len() < 2 {
        return Vec::new();
    }
    let step = ((values.len() - 1) / MAX_SPLIT_CANDIDATES).max(1);
    values
        .windows(2)
        .step_by(step)
        .map(|w| (w[0] + w[1]) * 0.5)
        .collect()
}

fn best_split(x: &DenseMatrix, residual: &[f32], rows: &[usize]) -> Option<SplitCandidate> {
    let cols = x.shape()[1];
    let lambda = 1.0f32;
    let best = (0..cols)
        .into_par_iter()
        .filter_map(|feature| {
            let mut local_best: Option<SplitCandidate> = None;
            for threshold in candidate_thresholds(x, rows, feature) {
                // Single sequential pass: accumulate left/right sums (SIMD-friendly).
                let (mut l_sum, mut l_cnt, mut r_sum, mut r_cnt) = (0.0f32, 0usize, 0.0f32, 0usize);
                for &row in rows {
                    if x[[row, feature]] <= threshold {
                        l_sum += residual[row];
                        l_cnt += 1;
                    } else {
                        r_sum += residual[row];
                        r_cnt += 1;
                    }
                }
                if l_cnt == 0 || r_cnt == 0 {
                    continue;
                }
                let l_mean = l_sum / (l_cnt as f32 + lambda);
                let r_mean = r_sum / (r_cnt as f32 + lambda);

                let loss: f32 = rows
                    .iter()
                    .map(|&row| {
                        let pred = if x[[row, feature]] <= threshold {
                            l_mean
                        } else {
                            r_mean
                        };
                        let d = residual[row] - pred;
                        d * d
                    })
                    .sum();

                if local_best
                    .as_ref()
                    .map(|best| loss < best.loss)
                    .unwrap_or(true)
                {
                    local_best = Some(SplitCandidate {
                        feature,
                        threshold,
                        left_value: l_mean,
                        right_value: r_mean,
                        loss,
                    });
                }
            }
            local_best
        })
        .reduce_with(|a, b| if a.loss <= b.loss { a } else { b });
    best
}

fn build_regression_tree(
    x: &DenseMatrix,
    residual: &[f32],
    rows: &[usize],
    depth: usize,
    max_depth: usize,
) -> RegTree {
    if rows.is_empty() || depth >= max_depth || rows.len() <= 2 {
        return RegTree::Leaf(mean_leaf(residual, rows));
    }

    let Some(split) = best_split(x, residual, rows) else {
        return RegTree::Leaf(mean_leaf(residual, rows));
    };

    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for &row in rows {
        if x[[row, split.feature]] <= split.threshold {
            left_rows.push(row);
        } else {
            right_rows.push(row);
        }
    }
    if left_rows.is_empty() || right_rows.is_empty() {
        return RegTree::Leaf(mean_leaf(residual, rows));
    }

    let left = if depth + 1 >= max_depth {
        RegTree::Leaf(split.left_value)
    } else {
        build_regression_tree(x, residual, &left_rows, depth + 1, max_depth)
    };
    let right = if depth + 1 >= max_depth {
        RegTree::Leaf(split.right_value)
    } else {
        build_regression_tree(x, residual, &right_rows, depth + 1, max_depth)
    };

    RegTree::Split {
        feature: split.feature,
        threshold: split.threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BinaryGradientBoosting {
    trees: Vec<RegTree>,
    base_score: f32,
    learning_rate: f32,
}

impl BinaryGradientBoosting {
    fn fit(
        x: &DenseMatrix,
        y: &[f32],
        rounds: usize,
        learning_rate: f32,
        max_depth: usize,
    ) -> Self {
        let positive = y.iter().sum::<f32>().max(1.0);
        let negative = (y.len() as f32 - positive).max(1.0);
        let base_score = (positive / negative).ln();
        let mut logits = vec![base_score; y.len()];
        let mut trees = Vec::with_capacity(rounds);
        let rows: Vec<usize> = (0..y.len()).collect();
        let max_depth = max_depth.max(1);

        for _ in 0..rounds {
            let residual: Vec<f32> = logits
                .iter()
                .zip(y.iter())
                .map(|(&logit, &target)| target - 1.0 / (1.0 + (-logit).exp()))
                .collect();

            let tree = build_regression_tree(x, &residual, &rows, 0, max_depth);

            for (row_idx, logit) in logits.iter_mut().enumerate() {
                let row = x.row(row_idx);
                *logit += learning_rate * tree.predict(row.as_slice().unwrap_or(&[]));
            }
            trees.push(tree);
        }
        Self {
            trees,
            base_score,
            learning_rate,
        }
    }

    fn predict_score(&self, row: &[f32]) -> f32 {
        let logit = self.trees.iter().fold(self.base_score, |acc, tree| {
            acc + self.learning_rate * tree.predict(row)
        });
        1.0 / (1.0 + (-logit).exp())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        max_depth: usize,
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
                    BinaryGradientBoosting::fit(matrix, &targets, rounds, learning_rate, max_depth)
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
                (
                    self.encoder.decode(idx),
                    model.predict_score(row_slice).max(1e-9).ln(),
                )
            })
            .collect();
        softmax_scores(&raw)
    }
}
