use crate::classifier::{softmax_scores, ClassificationResult, Classifier};
use crate::config::ScoreSumMode;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::matcher::{RuleMatcher, ScoringEngine};
use crate::parallel::install_pool;
use crate::nlp::{
    dense_matrix_from_texts, fit_tfidf, normalize_text, tokenize, DenseMatrix, FastTextConfigBuilder,
    NlpOption, TfIdfModel, WordEmbeddingModel,
};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SvmKernel {
    #[serde(alias = "linear", alias = "Linear")]
    Linear,
    #[serde(alias = "rbf", alias = "RBF")]
    Rbf,
    #[serde(alias = "polynomial", alias = "Polynomial", alias = "poly")]
    Polynomial,
    #[serde(alias = "sigmoid", alias = "Sigmoid")]
    Sigmoid,
}

#[derive(Debug, Clone)]
pub struct LogisticRegressionConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub lambda: f32,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self { learning_rate: 0.25, epochs: 180, lambda: 1e-3 }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RandomForestMode {
    #[serde(alias = "standard", alias = "Standard", alias = "standart", alias = "Standart")]
    Standard,
    #[serde(alias = "balanced", alias = "Balanced")]
    Balanced,
    #[serde(alias = "extra_trees", alias = "ExtraTrees", alias = "extra-trees")]
    ExtraTrees,
}

impl Default for RandomForestMode {
    fn default() -> Self { Self::Standard }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RandomForestMaxFeatures {
    #[serde(alias = "sqrt", alias = "Sqrt")]
    Sqrt,
    #[serde(alias = "log2", alias = "Log2")]
    Log2,
    #[serde(alias = "all", alias = "All", alias = "auto", alias = "Auto")]
    All,
    #[serde(alias = "half", alias = "Half")]
    Half,
}

impl Default for RandomForestMaxFeatures {
    fn default() -> Self { Self::Sqrt }
}

impl RandomForestMaxFeatures {
    fn resolve(&self, total_features: usize) -> usize {
        let total_features = total_features.max(1);
        match self {
            Self::Sqrt => (total_features as f32).sqrt().round() as usize,
            Self::Log2 => (total_features as f32).log2().round() as usize,
            Self::All => total_features,
            Self::Half => (total_features / 2).max(1),
        }
        .clamp(1, total_features)
    }
}

#[derive(Debug, Clone)]
pub struct RandomForestConfig {
    pub mode: RandomForestMode,
    pub n_trees: usize,
    pub max_depth: usize,
    pub max_features: RandomForestMaxFeatures,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub bootstrap: bool,
    pub oob_score: bool,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            mode: RandomForestMode::Standard,
            n_trees: 21,
            max_depth: 6,
            max_features: RandomForestMaxFeatures::Sqrt,
            min_samples_split: 2,
            min_samples_leaf: 1,
            bootstrap: true,
            oob_score: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SvmConfig {
    pub kernel: SvmKernel,
    pub c: f32,
    pub learning_rate: f32,
    pub epochs: usize,
    pub gamma: f32,
    pub degree: usize,
    pub coef0: f32,
}

impl Default for SvmConfig {
    fn default() -> Self {
        Self {
            kernel: SvmKernel::Linear,
            c: 1.0,
            learning_rate: 0.08,
            epochs: 40,
            gamma: 0.35,
            degree: 2,
            coef0: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradientBoostingConfig {
    pub n_estimators: usize,
    pub learning_rate: f32,
    pub max_depth: usize,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self { n_estimators: 24, learning_rate: 0.2, max_depth: 1 }
    }
}

#[derive(Debug, Clone)]
pub struct IsolationForestConfig {
    pub n_trees: usize,
    pub contamination: f32,
    pub subsample_size: usize,
}

impl Default for IsolationForestConfig {
    fn default() -> Self {
        Self { n_trees: 64, contamination: 0.05, subsample_size: 64 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AdvancedModelConfig {
    pub threads: Option<usize>,
    pub logistic: Option<LogisticRegressionConfig>,
    pub random_forest: Option<RandomForestConfig>,
    pub svm: Option<SvmConfig>,
    pub gradient_boosting: Option<GradientBoostingConfig>,
    pub isolation_forest: Option<IsolationForestConfig>,
}


#[derive(Debug, Clone)]
pub enum AdvancedMethod {
    LogisticRegression,
    RandomForest,
    IsolationForest,
    Svm,
    GradientBoosting,
}

#[derive(Debug, Clone)]
enum FeaturePipeline {
    Count(TfIdfModel),
    TfIdf(TfIdfModel),
    Word2Vec(WordEmbeddingModel),
    FastText(WordEmbeddingModel),
}

impl FeaturePipeline {
    fn fit(samples: &[TrainingSample], nlp: NlpOption, dims: usize) -> Result<(Self, DenseMatrix), VecEyesError> {
        let texts: Vec<String> = samples.iter().map(|s| s.text.clone()).collect();
        match nlp {
            NlpOption::Count => {
                let model = fit_tfidf(&texts);
                let matrix = transform_count(&model, &texts);
                Ok((Self::Count(model), matrix))
            }
            NlpOption::TfIdf => {
                let model = fit_tfidf(&texts);
                let matrix = crate::nlp::transform_tfidf(&model, &texts);
                Ok((Self::TfIdf(model), matrix))
            }
            NlpOption::Word2Vec => {
                let model = WordEmbeddingModel::train_word2vec(&texts, dims);
                let matrix = dense_matrix_from_texts(&model, &texts);
                Ok((Self::Word2Vec(model), matrix))
            }
            NlpOption::FastText => {
                let cfg = FastTextConfigBuilder::new().build();
                let model = WordEmbeddingModel::train_fasttext(&texts, dims, cfg);
                let matrix = dense_matrix_from_texts(&model, &texts);
                Ok((Self::FastText(model), matrix))
            }
        }
    }

    fn transform_text(&self, text: &str) -> DenseMatrix {
        let texts = vec![text.to_string()];
        match self {
            Self::Count(model) => transform_count(model, &texts),
            Self::TfIdf(model) => crate::nlp::transform_tfidf(model, &texts),
            Self::Word2Vec(model) | Self::FastText(model) => dense_matrix_from_texts(model, &texts),
        }
    }
}

fn transform_count(model: &TfIdfModel, texts: &[String]) -> DenseMatrix {
    let rows = texts.len();
    let cols = model.vocab.len();
    let mut matrix = Array2::<f32>::zeros((rows, cols));

    for row in 0..texts.len() {
        let normalized = normalize_text(&texts[row]);
        let tokens = tokenize(&normalized);
        for token in tokens {
            if let Some(index) = model.token_to_index.get(&token) {
                matrix[[row, *index]] += 1.0;
            }
        }
        l2_normalize_row(&mut matrix, row);
    }

    matrix
}

fn l2_normalize_row(matrix: &mut DenseMatrix, row: usize) {
    let mut norm = 0.0f32;
    for col in 0..matrix.shape()[1] {
        let v = matrix[[row, col]];
        norm += v * v;
    }
    norm = norm.sqrt();
    if norm > 0.0 {
        for col in 0..matrix.shape()[1] {
            matrix[[row, col]] /= norm;
        }
    }
}

#[derive(Debug, Clone)]
struct LabelEncoder {
    labels: Vec<ClassificationLabel>,
    to_idx: HashMap<ClassificationLabel, usize>,
}

impl LabelEncoder {
    fn fit(samples: &[TrainingSample]) -> Self {
        let mut labels: Vec<ClassificationLabel> = samples.iter().map(|s| s.label.clone()).collect();
        labels.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        labels.dedup();
        let mut to_idx = HashMap::new();
        for (idx, label) in labels.iter().enumerate() {
            to_idx.insert(label.clone(), idx);
        }
        Self { labels, to_idx }
    }

    fn encode(&self, label: &ClassificationLabel) -> usize {
        *self.to_idx.get(label).unwrap_or(&0)
    }

    fn decode(&self, idx: usize) -> ClassificationLabel {
        self.labels.get(idx).cloned().unwrap_or(ClassificationLabel::RawData)
    }
}

#[derive(Debug, Clone)]
struct LogisticOVR {
    weights: Array2<f32>,
    bias: Array1<f32>,
    encoder: LabelEncoder,
}

impl LogisticOVR {
    fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], epochs: usize, lr: f32, lambda: f32, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let classes = encoder.labels.len();
        let features = matrix.shape()[1];
        let x = matrix.clone();
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect();

        let models: Vec<(Vec<f32>, f32)> = install_pool(threads, || {
            (0..classes)
            .into_par_iter()
            .map(|class_id| {
                let mut w = vec![0.0f32; features];
                let mut b = 0.0f32;
                for _ in 0..epochs {
                    let mut grad_w = vec![0.0f32; features];
                    let mut grad_b = 0.0f32;
                    for row in 0..x.shape()[0] {
                        let y = if y_idx[row] == class_id { 1.0 } else { 0.0 };
                        let mut z = b;
                        for col in 0..features {
                            z += x[[row, col]] * w[col];
                        }
                        let pred = 1.0 / (1.0 + (-z).exp());
                        let diff = pred - y;
                        for col in 0..features {
                            grad_w[col] += diff * x[[row, col]];
                        }
                        grad_b += diff;
                    }
                    let inv_n = 1.0 / (x.shape()[0].max(1) as f32);
                    for col in 0..features {
                        grad_w[col] = grad_w[col] * inv_n + lambda * w[col];
                        w[col] -= lr * grad_w[col];
                    }
                    b -= lr * grad_b * inv_n;
                }
                (w, b)
            })
            .collect()
        });

        let mut weights = Array2::<f32>::zeros((classes, features));
        let mut bias = Array1::<f32>::zeros(classes);
        for (class_id, (w, b)) in models.into_iter().enumerate() {
            for col in 0..features {
                weights[[class_id, col]] = w[col];
            }
            bias[class_id] = b;
        }

        Self { weights, bias, encoder }
    }

    fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0);
        let mut raw = Vec::new();
        for class_id in 0..self.encoder.labels.len() {
            let mut z = self.bias[class_id];
            for col in 0..row.len() {
                z += self.weights[[class_id, col]] * row[col];
            }
            let p = 1.0 / (1.0 + (-z).exp());
            raw.push((self.encoder.decode(class_id), p.max(1e-6).ln()));
        }
        softmax_scores(&raw)
    }
}

fn apply_svm_kernel_map(matrix: &DenseMatrix, config: &SvmConfig) -> DenseMatrix {
    let mut mapped = matrix.clone();
    match config.kernel {
        SvmKernel::Linear => mapped,
        SvmKernel::Rbf => {
            for row in 0..mapped.shape()[0] {
                for col in 0..mapped.shape()[1] {
                    let value = mapped[[row, col]];
                    mapped[[row, col]] = (-config.gamma * value * value).exp();
                }
            }
            mapped
        }
        SvmKernel::Polynomial => {
            for row in 0..mapped.shape()[0] {
                for col in 0..mapped.shape()[1] {
                    let value = mapped[[row, col]];
                    mapped[[row, col]] = (config.gamma * value + config.coef0).powi(config.degree as i32);
                }
            }
            mapped
        }
        SvmKernel::Sigmoid => {
            for row in 0..mapped.shape()[0] {
                for col in 0..mapped.shape()[1] {
                    let value = mapped[[row, col]];
                    mapped[[row, col]] = (config.gamma * value + config.coef0).tanh();
                }
            }
            mapped
        }
    }
}

#[derive(Debug, Clone)]
struct LinearSvmOVR {
    weights: Array2<f32>,
    bias: Array1<f32>,
    encoder: LabelEncoder,
    config: SvmConfig,
}

impl LinearSvmOVR {
    fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], config: &SvmConfig, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let x = apply_svm_kernel_map(matrix, config);
        let classes = encoder.labels.len();
        let features = x.shape()[1];
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect();
        let lambda = 1.0 / config.c.max(1e-6);

        let models: Vec<(Vec<f32>, f32)> = install_pool(threads, || {
            (0..classes)
            .into_par_iter()
            .map(|class_id| {
                let mut w = vec![0.0f32; features];
                let mut b = 0.0f32;
                for epoch in 0..config.epochs {
                    let local_lr = config.learning_rate / (1.0 + epoch as f32 * 0.05);
                    for row in 0..x.shape()[0] {
                        let y = if y_idx[row] == class_id { 1.0 } else { -1.0 };
                        let mut margin = b;
                        for col in 0..features {
                            margin += x[[row, col]] * w[col];
                        }
                        let signed = y * margin;
                        for col in 0..features {
                            w[col] *= 1.0 - local_lr * lambda;
                        }
                        if signed < 1.0 {
                            for col in 0..features {
                                w[col] += local_lr * y * x[[row, col]];
                            }
                            b += local_lr * y;
                        }
                    }
                }
                (w, b)
            })
            .collect()
        });

        let mut weights = Array2::<f32>::zeros((classes, features));
        let mut bias = Array1::<f32>::zeros(classes);
        for (class_id, (w, b)) in models.into_iter().enumerate() {
            for col in 0..features {
                weights[[class_id, col]] = w[col];
            }
            bias[class_id] = b;
        }
        Self { weights, bias, encoder, config: config.clone() }
    }

    fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let mapped = apply_svm_kernel_map(probe, &self.config);
        let row = mapped.index_axis(Axis(0), 0);
        let mut raw = Vec::new();
        for class_id in 0..self.encoder.labels.len() {
            let mut margin = self.bias[class_id];
            for col in 0..row.len() {
                margin += self.weights[[class_id, col]] * row[col];
            }
            raw.push((self.encoder.decode(class_id), margin));
        }
        softmax_scores(&raw)
    }
}

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
        if values.len() > 12 {
            let step = values.len() / 12;
            values = values.into_iter().step_by(step.max(1)).collect();
        }

        let thresholds: Vec<f32> = match strategy {
            SplitStrategy::Standard => values,
            SplitStrategy::ExtraTrees => {
                if values.is_empty() {
                    Vec::new()
                } else {
                    let min_v = *values.first().unwrap_or(&0.0);
                    let max_v = *values.last().unwrap_or(&min_v);
                    let tries = 4usize.min(values.len().max(1));
                    (0..tries)
                        .map(|_| if (max_v - min_v).abs() < 1e-6 { min_v } else { rng.gen_range(min_v..max_v) })
                        .collect()
                }
            }
        };

        for threshold in thresholds {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &row_idx in rows {
                if x[[row_idx, feature]] <= threshold { left.push(row_idx); } else { right.push(row_idx); }
            }
            if left.len() < min_leaf || right.len() < min_leaf { continue; }
            let left_labels: Vec<usize> = left.iter().map(|&idx| y[idx]).collect();
            let right_labels: Vec<usize> = right.iter().map(|&idx| y[idx]).collect();
            let left_g = gini(&left_labels, num_classes);
            let right_g = gini(&right_labels, num_classes);
            let weighted = (left.len() as f32 / rows.len() as f32) * left_g + (right.len() as f32 / rows.len() as f32) * right_g;
            let gain = parent_gini - weighted;
            if gain > best_gain {
                best_gain = gain;
                best_split = Some((feature, threshold, left, right));
            }
        }
    }

    if let Some((feature, threshold, left_rows, right_rows)) = best_split {
        if best_gain <= 0.0 {
            return TreeNode::Leaf(class_distribution(&parent_labels, num_classes));
        }
        let left = build_tree(x, y, &left_rows, num_classes, depth + 1, max_depth, min_leaf, min_samples_split, feature_budget, strategy, rng);
        let right = build_tree(x, y, &right_rows, num_classes, depth + 1, max_depth, min_leaf, min_samples_split, feature_budget, strategy, rng);
        TreeNode::Split { feature, threshold, left: Box::new(left), right: Box::new(right) }
    } else {
        TreeNode::Leaf(class_distribution(&parent_labels, num_classes))
    }
}

#[derive(Debug, Clone)]
struct RandomForestModel {
    trees: Vec<TreeNode>,
    encoder: LabelEncoder,
    oob_score: Option<f32>,
}

impl RandomForestModel {
    fn balanced_bootstrap(y_idx: &[usize], rows: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for (row_idx, class_id) in y_idx.iter().copied().enumerate() {
            by_class.entry(class_id).or_default().push(row_idx);
        }
        if by_class.is_empty() {
            return (0..rows).collect();
        }
        let target_per_class = ((rows as f32) / by_class.len() as f32).ceil() as usize;
        let mut sample = Vec::with_capacity(rows);
        let mut class_ids: Vec<usize> = by_class.keys().copied().collect();
        class_ids.sort_unstable();
        for class_id in class_ids {
            let bucket = &by_class[&class_id];
            for _ in 0..target_per_class {
                let picked = bucket[rng.gen_range(0..bucket.len())];
                sample.push(picked);
                if sample.len() >= rows {
                    break;
                }
            }
            if sample.len() >= rows { break; }
        }
        while sample.len() < rows {
            sample.push(rng.gen_range(0..rows));
        }
        sample
    }

    fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], config: &RandomForestConfig, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect();
        let rows = matrix.shape()[0];
        let features = matrix.shape()[1];
        let feature_budget = config.max_features.resolve(features);
        let strategy = match config.mode {
            RandomForestMode::ExtraTrees => SplitStrategy::ExtraTrees,
            RandomForestMode::Standard | RandomForestMode::Balanced => SplitStrategy::Standard,
        };

        let tree_results: Vec<(TreeNode, Option<Vec<usize>>)> = install_pool(threads, || {
            (0..config.n_trees)
            .into_par_iter()
            .map(|seed| {
                let mut rng = StdRng::seed_from_u64(0x5EED_u64 + seed as u64 * 17);
                let bootstrap: Vec<usize> = match config.mode {
                    RandomForestMode::Balanced => Self::balanced_bootstrap(&y_idx, rows, &mut rng),
                    RandomForestMode::Standard | RandomForestMode::ExtraTrees => {
                        if config.bootstrap {
                            (0..rows).map(|_| rng.gen_range(0..rows)).collect()
                        } else {
                            let mut sample: Vec<usize> = (0..rows).collect();
                            sample.shuffle(&mut rng);
                            sample
                        }
                    }
                };
                let oob = if config.oob_score {
                    let inbag: std::collections::HashSet<usize> = bootstrap.iter().copied().collect();
                    let remaining: Vec<usize> = (0..rows).filter(|idx| !inbag.contains(idx)).collect();
                    Some(remaining)
                } else {
                    None
                };
                let tree = build_tree(
                    matrix,
                    &y_idx,
                    &bootstrap,
                    encoder.labels.len(),
                    0,
                    config.max_depth,
                    config.min_samples_leaf.max(1),
                    config.min_samples_split.max(2),
                    feature_budget,
                    strategy,
                    &mut rng,
                );
                (tree, oob)
            })
            .collect()
        });

        let trees: Vec<TreeNode> = tree_results.iter().map(|(tree, _)| tree.clone()).collect();
        let oob_score = if config.oob_score {
            // Real OOB estimation: aggregate predictions per sample across all trees
            // for which the sample was out-of-bag, then evaluate the majority class.
            let mut per_row_votes = vec![vec![0.0f32; encoder.labels.len()]; rows];
            let mut per_row_counts = vec![0usize; rows];
            for (tree, oob_rows) in &tree_results {
                if let Some(rows_for_tree) = oob_rows {
                    for &row_idx in rows_for_tree {
                        let row = matrix.index_axis(Axis(0), row_idx).to_vec();
                        let pred = tree.predict(&row);
                        for (class_idx, value) in pred.into_iter().enumerate() {
                            per_row_votes[row_idx][class_idx] += value;
                        }
                        per_row_counts[row_idx] += 1;
                    }
                }
            }

            let mut correct = 0usize;
            let mut total = 0usize;
            for row_idx in 0..rows {
                if per_row_counts[row_idx] == 0 {
                    continue;
                }
                total += 1;
                let best_idx = per_row_votes[row_idx]
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                if best_idx == y_idx[row_idx] {
                    correct += 1;
                }
            }
            if total > 0 { Some(correct as f32 / total as f32) } else { None }
        } else {
            None
        };

        Self { trees, encoder, oob_score }
    }

    fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0).to_vec();
        let mut votes = vec![0.0f32; self.encoder.labels.len()];
        for tree in &self.trees {
            let local = tree.predict(&row);
            for idx in 0..votes.len() {
                votes[idx] += local.get(idx).copied().unwrap_or(0.0);
            }
        }
        let raw: Vec<(ClassificationLabel, f32)> = votes
            .into_iter()
            .enumerate()
            .map(|(idx, v)| (self.encoder.decode(idx), v.max(1e-6).ln()))
            .collect();
        softmax_scores(&raw)
    }
}

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

    for feature in 0..cols {
        let mut values: Vec<f32> = (0..rows).map(|r| x[[r, feature]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        if values.len() > 12 {
            let step = values.len() / 12;
            values = values.into_iter().step_by(step.max(1)).collect();
        }
        for threshold in values {
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
struct GradientBoostingModel {
    encoder: LabelEncoder,
    models: Vec<BinaryGradientBoosting>,
}

impl GradientBoostingModel {
    fn fit(matrix: &DenseMatrix, samples: &[TrainingSample], rounds: usize, learning_rate: f32, threads: Option<usize>) -> Self {
        let encoder = LabelEncoder::fit(samples);
        let y_idx: Vec<usize> = samples.iter().map(|s| encoder.encode(&s.label)).collect();
        let models: Vec<BinaryGradientBoosting> = install_pool(threads, || {
            (0..encoder.labels.len())
            .into_par_iter()
            .map(|class_id| {
                let targets: Vec<f32> = y_idx.iter().map(|&idx| if idx == class_id { 1.0 } else { 0.0 }).collect();
                BinaryGradientBoosting::fit(matrix, &targets, rounds, learning_rate)
            })
            .collect()
        });
        Self { encoder, models }
    }

    fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        let row = probe.index_axis(Axis(0), 0).to_vec();
        let raw: Vec<(ClassificationLabel, f32)> = self.models
            .iter()
            .enumerate()
            .map(|(idx, model)| (self.encoder.decode(idx), model.predict_score(&row).max(1e-6).ln()))
            .collect();
        softmax_scores(&raw)
    }
}

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
struct IsolationForestModel {
    trees: Vec<IsolationNode>,
    threshold: f32,
    anomaly_label: ClassificationLabel,
    normal_label: ClassificationLabel,
    subsample_size: usize,
}

impl IsolationForestModel {
    fn fit(cold_matrix: &DenseMatrix, anomaly_label: ClassificationLabel, normal_label: ClassificationLabel, n_trees: usize, contamination: f32, subsample_size: usize, threads: Option<usize>) -> Self {
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

    fn predict_scores(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
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

#[derive(Debug, Clone)]
enum AdvancedInner {
    Logistic(LogisticOVR),
    RandomForest(RandomForestModel),
    Svm(LinearSvmOVR),
    GradientBoosting(GradientBoostingModel),
    IsolationForest(IsolationForestModel),
}

#[derive(Debug, Clone)]
pub struct AdvancedClassifier {
    pipeline: FeaturePipeline,
    inner: AdvancedInner,
}

impl AdvancedClassifier {

    pub fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult {
        <Self as Classifier>::classify_text(self, text, score_sum_mode, matchers)
    }

    pub fn train(
        method: AdvancedMethod,
        samples: &[TrainingSample],
        nlp: NlpOption,
        hot_label: ClassificationLabel,
        cold_label: ClassificationLabel,
        config: &AdvancedModelConfig,
    ) -> Result<Self, VecEyesError> {
        let dims = 32;
        match method {
            AdvancedMethod::IsolationForest => {
                if !matches!(nlp, NlpOption::Word2Vec | NlpOption::FastText) {
                    return Err(VecEyesError::InvalidConfig(
                        "IsolationForest currently requires Word2Vec or FastText embeddings".into(),
                    ));
                }
                let params = config.isolation_forest.clone().unwrap_or_default();
                let cold_only: Vec<TrainingSample> = samples.iter().filter(|s| s.label == cold_label).cloned().collect();
                let basis = if cold_only.is_empty() { samples.to_vec() } else { cold_only };
                let (pipeline, matrix) = FeaturePipeline::fit(&basis, nlp, dims)?;
                let model = IsolationForestModel::fit(
                    &matrix,
                    hot_label,
                    cold_label,
                    params.n_trees,
                    params.contamination,
                    params.subsample_size,
                    config.threads,
                );
                Ok(Self { pipeline, inner: AdvancedInner::IsolationForest(model) })
            }
            AdvancedMethod::LogisticRegression => {
                let params = config.logistic.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = LogisticOVR::fit(&matrix, samples, params.epochs, params.learning_rate, params.lambda, config.threads);
                Ok(Self { pipeline, inner: AdvancedInner::Logistic(model) })
            }
            AdvancedMethod::Svm => {
                let params = config.svm.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = LinearSvmOVR::fit(&matrix, samples, &params, config.threads);
                Ok(Self { pipeline, inner: AdvancedInner::Svm(model) })
            }
            AdvancedMethod::RandomForest => {
                let params = config.random_forest.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = RandomForestModel::fit(&matrix, samples, &params, config.threads);
                Ok(Self { pipeline, inner: AdvancedInner::RandomForest(model) })
            }
            AdvancedMethod::GradientBoosting => {
                let params = config.gradient_boosting.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = GradientBoostingModel::fit(&matrix, samples, params.n_estimators, params.learning_rate, config.threads);
                Ok(Self { pipeline, inner: AdvancedInner::GradientBoosting(model) })
            }
        }
    }


    pub fn random_forest_oob_score(&self) -> Option<f32> {
        match &self.inner {
            AdvancedInner::RandomForest(model) => model.oob_score,
            _ => None,
        }
    }

    fn base_scores(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        let probe = self.pipeline.transform_text(text);
        match &self.inner {
            AdvancedInner::Logistic(model) => model.predict_scores(&probe),
            AdvancedInner::RandomForest(model) => model.predict_scores(&probe),
            AdvancedInner::Svm(model) => model.predict_scores(&probe),
            AdvancedInner::GradientBoosting(model) => model.predict_scores(&probe),
            AdvancedInner::IsolationForest(model) => model.predict_scores(&probe),
        }
    }
}

impl Classifier for AdvancedClassifier {
    fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult {
        let mut labels = self.base_scores(text);
        let (boost, hits) = ScoringEngine::compute_rule_boost(text, matchers);
        if score_sum_mode.is_on() {
            for (_, score) in &mut labels {
                *score = ScoringEngine::merge_scores(*score, boost, score_sum_mode);
            }
        }
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ClassificationResult { labels, extra_hits: hits }
    }
}
