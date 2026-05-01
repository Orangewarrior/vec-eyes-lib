//! Standard ML evaluation metrics.
//!
//! All functions operate on label slices so they work with any classifier
//! output — there is no dependency on the classifier internals.
//!
//! # Binary metrics
//! [`precision`], [`recall`], [`f1`], [`roc_auc`] each require a designated
//! positive class label.
//!
//! # Multi-class metrics
//! [`accuracy`], [`macro_f1`], [`confusion_matrix`] work with any number of
//! classes.

use std::collections::HashMap;
use std::hash::Hash;

/// Fraction of predictions that match the ground-truth labels.
pub fn accuracy<L: Eq>(y_true: &[L], y_pred: &[L]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have equal length"
    );
    if y_true.is_empty() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| t == p)
        .count();
    correct as f32 / y_true.len() as f32
}

/// Precision for one class: TP / (TP + FP).
pub fn precision<L: Eq>(y_true: &[L], y_pred: &[L], positive: &L) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have equal length"
    );
    let tp = y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == &positive && t == &positive)
        .count();
    let fp = y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == &positive && t != &positive)
        .count();
    let denom = tp + fp;
    if denom == 0 {
        0.0
    } else {
        tp as f32 / denom as f32
    }
}

/// Recall for one class: TP / (TP + FN).
pub fn recall<L: Eq>(y_true: &[L], y_pred: &[L], positive: &L) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have equal length"
    );
    let tp = y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == &positive && t == &positive)
        .count();
    let fn_ = y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p != &positive && t == &positive)
        .count();
    let denom = tp + fn_;
    if denom == 0 {
        0.0
    } else {
        tp as f32 / denom as f32
    }
}

/// F1 score for one class: harmonic mean of precision and recall.
pub fn f1<L: Eq>(y_true: &[L], y_pred: &[L], positive: &L) -> f32 {
    let p = precision(y_true, y_pred, positive);
    let r = recall(y_true, y_pred, positive);
    if p + r < 1e-9 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

/// Macro-average F1: unweighted mean of per-class F1 scores.
pub fn macro_f1<L: Eq + Hash + Clone>(y_true: &[L], y_pred: &[L]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have equal length"
    );
    let classes: std::collections::HashSet<_> = y_true.iter().chain(y_pred.iter()).collect();
    if classes.is_empty() {
        return 0.0;
    }
    let sum: f32 = classes.iter().map(|c| f1(y_true, y_pred, c)).sum();
    sum / classes.len() as f32
}

/// Weighted F1: class F1 weighted by support (true class frequency).
pub fn weighted_f1<L: Eq + Hash + Clone>(y_true: &[L], y_pred: &[L]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have equal length"
    );
    if y_true.is_empty() {
        return 0.0;
    }
    let mut support: HashMap<_, usize> = HashMap::new();
    for label in y_true {
        *support.entry(label).or_insert(0) += 1;
    }
    let total = y_true.len() as f32;
    support
        .iter()
        .map(|(label, &count)| f1(y_true, y_pred, label) * count as f32)
        .sum::<f32>()
        / total
}

/// Area under the ROC curve for binary classification.
///
/// `scores` is a slice of `(predicted_score, is_positive_class)` pairs.
/// Uses the trapezoidal rule over all unique thresholds.
pub fn roc_auc(scores: &[(f32, bool)]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let positives = scores.iter().filter(|&&(_, p)| p).count();
    let negatives = scores.len() - positives;
    if positives == 0 || negatives == 0 {
        return 0.5;
    }

    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut auc = 0.0f32;
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_tp = 0usize;
    let mut prev_fp = 0usize;

    for &(_, is_positive) in &sorted {
        if is_positive {
            tp += 1;
        } else {
            fp += 1;
        }
        let tpr = tp as f32 / positives as f32;
        let fpr = fp as f32 / negatives as f32;
        let prev_tpr = prev_tp as f32 / positives as f32;
        let prev_fpr = prev_fp as f32 / negatives as f32;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;
        prev_tp = tp;
        prev_fp = fp;
    }
    auc.clamp(0.0, 1.0)
}

/// Confusion matrix as a `Vec<Vec<usize>>` indexed by `labels`.
///
/// `result[i][j]` = number of samples with true label `labels[i]` predicted
/// as `labels[j]`.
pub fn confusion_matrix<L: Eq + Hash>(y_true: &[L], y_pred: &[L], labels: &[L]) -> Vec<Vec<usize>> {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have equal length"
    );
    let n = labels.len();
    let label_idx: HashMap<_, usize> = labels.iter().enumerate().map(|(i, l)| (l, i)).collect();
    let mut matrix = vec![vec![0usize; n]; n];
    for (t, p) in y_true.iter().zip(y_pred.iter()) {
        if let (Some(&ti), Some(&pi)) = (label_idx.get(t), label_idx.get(p)) {
            matrix[ti][pi] += 1;
        }
    }
    matrix
}

/// Summary of precision, recall, and F1 per class plus overall accuracy.
#[derive(Debug, Clone)]
pub struct ClassificationReport<L> {
    pub per_class: Vec<ClassMetrics<L>>,
    pub accuracy: f32,
    pub macro_f1: f32,
    pub weighted_f1: f32,
}

#[derive(Debug, Clone)]
pub struct ClassMetrics<L> {
    pub label: L,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub support: usize,
}

/// Build a full [`ClassificationReport`] from predictions.
pub fn classification_report<L: Eq + Hash + Clone>(
    y_true: &[L],
    y_pred: &[L],
) -> ClassificationReport<L> {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "y_true and y_pred must have equal length"
    );
    let mut support: HashMap<&L, usize> = HashMap::new();
    for l in y_true {
        *support.entry(l).or_insert(0) += 1;
    }
    let mut classes: Vec<&L> = support.keys().copied().collect();
    // Stable ordering: sorted by support descending so dominant class is first.
    classes.sort_by(|a, b| support[b].cmp(&support[a]));
    let per_class = classes
        .iter()
        .map(|label| ClassMetrics {
            label: (*label).clone(),
            precision: precision(y_true, y_pred, label),
            recall: recall(y_true, y_pred, label),
            f1: f1(y_true, y_pred, label),
            support: *support.get(label).unwrap_or(&0),
        })
        .collect();
    ClassificationReport {
        per_class,
        accuracy: accuracy(y_true, y_pred),
        macro_f1: macro_f1(y_true, y_pred),
        weighted_f1: weighted_f1(y_true, y_pred),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_all_correct() {
        assert_eq!(accuracy(&[1, 2, 3], &[1, 2, 3]), 1.0);
    }

    #[test]
    fn accuracy_none_correct() {
        assert_eq!(accuracy(&[1, 2, 3], &[4, 5, 6]), 0.0);
    }

    #[test]
    fn precision_recall_f1_binary() {
        let y_true = vec![true, true, false, false, true];
        let y_pred = vec![true, false, false, true, true];
        let p = precision(&y_true, &y_pred, &true);
        let r = recall(&y_true, &y_pred, &true);
        let f = f1(&y_true, &y_pred, &true);
        assert!((p - 2.0 / 3.0).abs() < 1e-5);
        assert!((r - 2.0 / 3.0).abs() < 1e-5);
        assert!((f - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn roc_auc_perfect() {
        let scores = vec![(1.0f32, true), (0.9, true), (0.4, false), (0.2, false)];
        assert!((roc_auc(&scores) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn confusion_matrix_basic() {
        let labels = vec!['a', 'b'];
        let y_true = vec!['a', 'a', 'b', 'b'];
        let y_pred = vec!['a', 'b', 'a', 'b'];
        let cm = confusion_matrix(&y_true, &y_pred, &labels);
        assert_eq!(cm[0][0], 1); // a→a
        assert_eq!(cm[0][1], 1); // a→b
        assert_eq!(cm[1][0], 1); // b→a
        assert_eq!(cm[1][1], 1); // b→b
    }
}
