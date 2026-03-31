use ndarray::Axis;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::advanced_models::{LabelEncoder, RandomForestConfig, RandomForestMode};
use crate::classifier::softmax_scores;
use crate::dataset::TrainingSample;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

#[derive(Debug, Clone)]
enum TreeNode {
    Leaf(Vec<f32>),
    Split {
        feature: usize,
        threshold: f32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict(&self, row: &[f32]) -> Vec<f32> {
        match self {
            Self::Leaf(values) => values.clone(),
            Self::Split { feature, threshold, left, right } => {
                if row[*feature] <= *threshold { left.predict(row) } else { right.predict(row) }
            }
        }
    }
}

fn class_distribution(y: &[usize], num_classes: usize) -> Vec<f32> {
    let mut counts = vec![0.0f32; num_classes];
    if y.is_empty() { return counts; }
    for &label in y { counts[label] += 1.0; }
    let inv = 1.0 / y.len() as f32;
    for item in &mut counts { *item *= inv; }
    counts
}

fn gini(y: &[usize], num_classes: usize) -> f32 {
    let dist = class_distribution(y, num_classes);
    1.0 - dist.iter().map(|p| p * p).sum::<f32>()
}

#[derive(Debug, Clone, Copy)]
enum SplitStrategy {
    Standard,
    ExtraTrees,
}

fn build_tree(
    x: &DenseMatrix,
    y: &[usize],
    rows: &[usize],
    num_classes: usize,
    depth: usize,
    max_depth: usize,
    min_leaf: usize,
    min_samples_split: usize,
    feature_budget: usize,
    strategy: SplitStrategy,
    rng: &mut StdRng,
) -> TreeNode {
    if rows.is_empty() || depth >= max_depth || rows.len() < min_samples_split.max(2) || rows.len() <= min_leaf {
        let labels: Vec<usize> = rows.iter().map(|&idx| y[idx]).collect();
        return TreeNode::Leaf(class_distribution(&labels, num_classes));
    }

    let first = y[rows[0]];
    if rows.iter().all(|&idx| y[idx] == first) {
        let labels: Vec<usize> = rows.iter().map(|&idx| y[idx]).collect();
        return TreeNode::Leaf(class_distribution(&labels, num_classes));
    }

    let features = x.shape()[1];
    let mut all_features: Vec<usize> = (0..features).collect();
    all_features.shuffle(rng);
    all_features.truncate(feature_budget.max(1).min(features));

    let parent_labels: Vec<usize> = rows.iter().map(|&idx| y[idx]).collect();
    let parent_gini = gini(&parent_labels, num_classes);
    let mut best_gain = -1.0f32;
    let mut best_split: Option<(usize, f32, Vec<usize>, Vec<usize>)> = None;

    for &feature in &all_features {
        let mut values: Vec<f32> = rows.iter().map(|&idx| x[[idx, feature]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        if values.len() < 2 {
            continue;
        }

        let thresholds: Vec<f32> = match strategy {
            SplitStrategy::Standard => values.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect(),
            SplitStrategy::ExtraTrees => {
                let min = *values.first().unwrap_or(&0.0);
                let max = *values.last().unwrap_or(&min);
                if (max - min).abs() < 1e-6 {
                    Vec::new()
                } else {
                    let tries = values.len().min(8).max(2);
                    (0..tries).map(|_| rng.gen_range(min..max)).collect()
                }
            }
        };

        for threshold in thresholds {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &row in rows {
                if x[[row, feature]] <= threshold { left.push(row); } else { right.push(row); }
            }
            if left.len() < min_leaf || right.len() < min_leaf {
                continue;
            }
            let left_labels: Vec<usize> = left.iter().map(|&idx| y[idx]).collect();
            let right_labels: Vec<usize> = right.iter().map(|&idx| y[idx]).collect();
            let gain = parent_gini
                - (left.len() as f32 / rows.len() as f32) * gini(&left_labels, num_classes)
                - (right.len() as f32 / rows.len() as f32) * gini(&right_labels, num_classes);

            if gain > best_gain {
                best_gain = gain;
                best_split = Some((feature, threshold, left, right));
            }
        }
    }

    if let Some((feature, threshold, left, right)) = best_split {
        TreeNode::Split {
            feature,
            threshold,
            left: Box::new(build_tree(
                x, y, &left, num_classes, depth + 1, max_depth, min_leaf, min_samples_split, feature_budget, strategy, rng,
            )),
            right: Box::new(build_tree(
                x, y, &right, num_classes, depth + 1, max_depth, min_leaf, min_samples_split, feature_budget, strategy, rng,
            )),
        }
    } else {
        let labels: Vec<usize> = rows.iter().map(|&idx| y[idx]).collect();
        TreeNode::Leaf(class_distribution(&labels, num_classes))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RandomForestModel {
    trees: Vec<TreeNode>,
    encoder: LabelEncoder,
    pub(crate) oob_score: Option<f32>,
}

impl RandomForestModel {
    pub(crate) fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], config: &RandomForestConfig, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect();
        let rows: Vec<usize> = (0..matrix.shape()[0]).collect();
        let num_classes = encoder.labels.len();
        let feature_budget = config.max_features.resolve(matrix.shape()[1]);
        let strategy = match config.mode {
            RandomForestMode::ExtraTrees => SplitStrategy::ExtraTrees,
            RandomForestMode::Standard | RandomForestMode::Balanced => SplitStrategy::Standard,
        };

        let built: Vec<(TreeNode, Option<Vec<usize>>)> = install_pool(threads, || {
            (0..config.n_trees)
                .into_par_iter()
                .map(|tree_id| {
                    let mut rng = StdRng::seed_from_u64(0xC0FFEE + tree_id as u64 * 17);
                    let boot_rows = if config.bootstrap {
                        match config.mode {
                            RandomForestMode::Balanced => Self::balanced_bootstrap(&y_idx, rows.len(), &mut rng),
                            RandomForestMode::Standard | RandomForestMode::ExtraTrees => {
                                (0..rows.len()).map(|_| rng.gen_range(0..rows.len())).collect::<Vec<_>>()
                            }
                        }
                    } else {
                        rows.clone()
                    };
                    let oob_rows = if config.bootstrap && config.oob_score {
                        let mut used = vec![false; rows.len()];
                        for &idx in &boot_rows {
                            used[idx] = true;
                        }
                        Some(rows.iter().copied().filter(|idx| !used[*idx]).collect::<Vec<_>>())
                    } else {
                        None
                    };
                    let tree = build_tree(
                        matrix,
                        &y_idx,
                        &boot_rows,
                        num_classes,
                        0,
                        config.max_depth.max(1),
                        config.min_samples_leaf.max(1),
                        config.min_samples_split.max(2),
                        feature_budget,
                        strategy,
                        &mut rng,
                    );
                    (tree, oob_rows)
                })
                .collect()
        });

        let trees: Vec<TreeNode> = built.iter().map(|(tree, _)| tree.clone()).collect();
        let oob_score = if config.oob_score && config.bootstrap {
            let mut correct = 0usize;
            let mut seen = 0usize;
            for (tree, oob_rows) in &built {
                if let Some(rows) = oob_rows {
                    for &row_idx in rows {
                        let row = matrix.index_axis(Axis(0), row_idx).to_vec();
                        let prediction = tree.predict(&row);
                        let best = prediction
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|(idx, _)| idx)
                            .unwrap_or(0);
                        if best == y_idx[row_idx] {
                            correct += 1;
                        }
                        seen += 1;
                    }
                }
            }
            if seen > 0 { Some(correct as f32 / seen as f32) } else { None }
        } else {
            None
        };

        Self { trees, encoder, oob_score }
    }

    fn balanced_bootstrap(y_idx: &[usize], rows: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, cls) in y_idx.iter().enumerate() {
            grouped.entry(*cls).or_default().push(idx);
        }
        let mut output = Vec::with_capacity(rows);
        let mut groups: Vec<Vec<usize>> = grouped.into_values().collect();
        groups.sort_by_key(|v| v.first().copied().unwrap_or(usize::MAX));
        while output.len() < rows {
            for group in &groups {
                if group.is_empty() {
                    continue;
                }
                let pick = group[rng.gen_range(0..group.len())];
                output.push(pick);
                if output.len() >= rows {
                    break;
                }
            }
        }
        output
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0).to_vec();
        let mut acc = vec![0.0f32; self.encoder.labels.len()];
        for tree in &self.trees {
            let dist = tree.predict(&row);
            for (idx, value) in dist.into_iter().enumerate() {
                acc[idx] += value;
            }
        }
        let raw: Vec<(ClassificationLabel, f32)> = acc
            .into_iter()
            .enumerate()
            .map(|(idx, value)| (self.encoder.decode(idx), value.max(1e-6).ln()))
            .collect();
        softmax_scores(&raw)
    }
}
