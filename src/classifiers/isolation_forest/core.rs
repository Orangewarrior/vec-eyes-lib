use ndarray::Axis;
use rand::prelude::*;
use rayon::prelude::*;

use crate::classifier::softmax_scores;
use crate::labels::ClassificationLabel;
use crate::nlp::DenseMatrix;
use crate::parallel::install_pool;

#[derive(Debug, Clone)]
struct IsolationNode {
    feature: usize,
    threshold: f32,
    left: Option<Box<IsolationNode>>,
    right: Option<Box<IsolationNode>>,
    size: usize,
}

impl IsolationNode {
    fn leaf(size: usize) -> Self {
        Self { feature: 0, threshold: 0.0, left: None, right: None, size }
    }

    fn path_length(&self, row: &[f32], depth: f32) -> f32 {
        match (&self.left, &self.right) {
            (Some(left), Some(right)) => {
                if row[self.feature] <= self.threshold { left.path_length(row, depth + 1.0) } else { right.path_length(row, depth + 1.0) }
            }
            _ => depth + c_factor(self.size as f32),
        }
    }
}

fn c_factor(n: f32) -> f32 {
    if n <= 1.0 { 0.0 } else { 2.0 * (n - 1.0).ln() + 0.57721566 - 2.0 * (n - 1.0) / n }
}

fn build_isolation_tree(x: &DenseMatrix, rows: &[usize], depth: usize, max_depth: usize, rng: &mut StdRng) -> IsolationNode {
    if rows.len() <= 1 || depth >= max_depth {
        return IsolationNode::leaf(rows.len());
    }
    let cols = x.shape()[1];
    let feature = rng.gen_range(0..cols);
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &row in rows {
        let v = x[[row, feature]];
        if v < min_v { min_v = v; }
        if v > max_v { max_v = v; }
    }
    if (max_v - min_v).abs() < 1e-6 {
        return IsolationNode::leaf(rows.len());
    }
    let threshold = rng.gen_range(min_v..max_v);
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for &row in rows {
        if x[[row, feature]] <= threshold { left_rows.push(row); } else { right_rows.push(row); }
    }
    if left_rows.is_empty() || right_rows.is_empty() {
        return IsolationNode::leaf(rows.len());
    }
    IsolationNode {
        feature,
        threshold,
        left: Some(Box::new(build_isolation_tree(x, &left_rows, depth + 1, max_depth, rng))),
        right: Some(Box::new(build_isolation_tree(x, &right_rows, depth + 1, max_depth, rng))),
        size: rows.len(),
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IsolationForestModel {
    trees: Vec<IsolationNode>,
    threshold: f32,
    anomaly_label: ClassificationLabel,
    normal_label: ClassificationLabel,
    subsample_size: usize,
}

impl IsolationForestModel {
    pub(crate) fn fit(cold_matrix: &DenseMatrix, anomaly_label: ClassificationLabel, normal_label: ClassificationLabel, n_trees: usize, contamination: f32, subsample_size: usize, threads: Option<usize>) -> Self {
        let rows: Vec<usize> = (0..cold_matrix.shape()[0]).collect();
        let max_depth = ((rows.len().max(2) as f32).log2().ceil() as usize).max(2);
        let trees: Vec<IsolationNode> = install_pool(threads, || {
            (0..n_trees)
                .into_par_iter()
                .map(|seed| {
                    let mut rng = StdRng::seed_from_u64(0xBADC0DE + seed as u64 * 13);
                    let sample_size = rows.len().min(subsample_size.max(8));
                    let subset: Vec<usize> = (0..sample_size).map(|_| rows[rng.gen_range(0..rows.len())]).collect();
                    build_isolation_tree(cold_matrix, &subset, 0, max_depth, &mut rng)
                })
                .collect()
        });

        let avg_c = c_factor(rows.len().min(subsample_size.max(8)) as f32).max(1e-6);
        let mut scores = Vec::new();
        for row_idx in 0..cold_matrix.shape()[0] {
            let row = cold_matrix.index_axis(Axis(0), row_idx).to_vec();
            let mut path_sum = 0.0;
            for tree in &trees {
                path_sum += tree.path_length(&row, 0.0);
            }
            let avg_path = path_sum / trees.len().max(1) as f32;
            let score = 2f32.powf(-avg_path / avg_c);
            scores.push(score);
        }
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pos = ((scores.len() as f32) * (1.0 - contamination.clamp(0.001, 0.49))).floor() as usize;
        let threshold = scores.get(pos.min(scores.len().saturating_sub(1))).copied().unwrap_or(0.6).max(0.55);

        Self { trees, threshold, anomaly_label, normal_label, subsample_size }
    }

    pub(crate) fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0).to_vec();
        let avg_c = c_factor(self.subsample_size.max(8) as f32).max(1e-6);
        let mut path_sum = 0.0;
        for tree in &self.trees {
            path_sum += tree.path_length(&row, 0.0);
        }
        let avg_path = path_sum / self.trees.len().max(1) as f32;
        let score = 2f32.powf(-avg_path / avg_c);
        let anomaly_pct = ((score / self.threshold.max(1e-6)).min(1.5) * 100.0).min(100.0);
        let normal_pct = (100.0 - anomaly_pct).max(0.0);
        let raw = vec![
            (self.anomaly_label.clone(), anomaly_pct.max(1e-6).ln()),
            (self.normal_label.clone(), normal_pct.max(1e-6).ln()),
        ];
        softmax_scores(&raw)
    }
}
