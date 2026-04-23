use crate::classifier::{ClassificationResult, Classifier};
use ndarray;
use crate::config::ScoreSumMode;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::matcher::{RuleMatcher, ScoringEngine};
use crate::nlp::{
    dense_matrix_from_texts_with_tfidf, fit_tfidf, normalize_text, tokenize, DenseMatrix,
    FastTextConfigBuilder, NlpOption, TfIdfModel, WordEmbeddingModel,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::classifiers::gradient_boosting::core::GradientBoostingModel;
use crate::classifiers::isolation_forest::core::IsolationForestModel;
use crate::classifiers::logistic_regression::core::LogisticOVR;
use crate::classifiers::random_forest::core::RandomForestModel;
use crate::classifiers::svm::core::LinearSvmOVR;

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    pub(crate) fn resolve(&self, total_features: usize) -> usize {
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomForestConfig {
    pub mode: RandomForestMode,
    pub n_trees: usize,
    pub max_depth: usize,
    pub max_features: RandomForestMaxFeatures,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub bootstrap: bool,
    pub oob_score: bool,
    pub random_seed: Option<u64>,
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
            random_seed: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AdvancedModelConfig {
    pub threads: Option<usize>,
    pub embedding_dimensions: Option<usize>,
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

/// Feature extraction pipeline stored alongside trained models.
///
/// Word2Vec and FastText variants carry the IDF model fitted on the **training**
/// corpus so inference never has to refit TF-IDF on a single probe document
/// (which would make every IDF weight degenerate and lose all discriminative
/// power from the training distribution).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum FeaturePipeline {
    Count(TfIdfModel),
    TfIdf(TfIdfModel),
    Word2Vec { model: WordEmbeddingModel, idf: TfIdfModel },
    FastText { model: WordEmbeddingModel, idf: TfIdfModel },
}

impl FeaturePipeline {
    fn fit(samples: &[TrainingSample], nlp: NlpOption, dims: usize) -> Result<(Self, DenseMatrix), VecEyesError> {
        let texts: Vec<&str> = samples.iter().map(|s| s.text.as_str()).collect();
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
                let idf = fit_tfidf(&texts);
                let model = WordEmbeddingModel::train_word2vec(&texts, dims);
                let matrix = dense_matrix_from_texts_with_tfidf(&model, &texts, Some(&idf));
                Ok((Self::Word2Vec { model, idf }, matrix))
            }
            NlpOption::FastText => {
                let cfg = FastTextConfigBuilder::new().build().expect("default FastTextConfigBuilder must be valid");
                let idf = fit_tfidf(&texts);
                let model = WordEmbeddingModel::train_fasttext(&texts, dims, cfg);
                let matrix = dense_matrix_from_texts_with_tfidf(&model, &texts, Some(&idf));
                Ok((Self::FastText { model, idf }, matrix))
            }
        }
    }

    fn transform_text(&self, text: &str) -> DenseMatrix {
        let texts = [text];
        self.transform_batch(&texts)
    }

    fn transform_batch<S: AsRef<str>>(&self, texts: &[S]) -> DenseMatrix {
        match self {
            Self::Count(model)  => transform_count(model, texts),
            Self::TfIdf(model)  => crate::nlp::transform_tfidf(model, texts),
            // Use the stored training IDF — never refit on the probe document.
            Self::Word2Vec { model, idf } => dense_matrix_from_texts_with_tfidf(model, texts, Some(idf)),
            Self::FastText { model, idf } => dense_matrix_from_texts_with_tfidf(model, texts, Some(idf)),
        }
    }
}

fn transform_count<S: AsRef<str>>(model: &TfIdfModel, texts: &[S]) -> DenseMatrix {
    let rows = texts.len();
    let cols = model.vocab.len();
    let mut matrix = ndarray::Array2::<f32>::zeros((rows, cols));

    for row in 0..texts.len() {
        let normalized = normalize_text(texts[row].as_ref());
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct LabelEncoder {
    pub(crate) labels: Vec<ClassificationLabel>,
    to_idx: HashMap<ClassificationLabel, usize>,
}

impl LabelEncoder {
    pub(crate) fn fit(samples: &[TrainingSample]) -> Self {
        let mut labels: Vec<ClassificationLabel> = samples.iter().map(|s| s.label.clone()).collect();
        labels.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        // sort must precede dedup — dedup only removes consecutive duplicates
        labels.dedup();
        let mut to_idx = HashMap::new();
        for (idx, label) in labels.iter().enumerate() {
            to_idx.insert(label.clone(), idx);
        }
        Self { labels, to_idx }
    }


    pub(crate) fn encode(&self, label: &ClassificationLabel) -> Result<usize, VecEyesError> {
        self.to_idx.get(label).copied().ok_or_else(|| {
            VecEyesError::invalid_config(
                "advanced_models::LabelEncoder::encode",
                format!("unseen label: {}", label.as_str()),
            )
        })
    }

    pub(crate) fn decode(&self, idx: usize) -> ClassificationLabel {
        self.labels.get(idx).cloned().unwrap_or(ClassificationLabel::RawData)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum AdvancedInner {
    Logistic(LogisticOVR),
    RandomForest(RandomForestModel),
    Svm(LinearSvmOVR),
    GradientBoosting(GradientBoostingModel),
    IsolationForest(IsolationForestModel),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
        let dims = config.embedding_dimensions.unwrap_or(32).max(1);
        match method {
            AdvancedMethod::IsolationForest => {
                if !matches!(nlp, NlpOption::Word2Vec | NlpOption::FastText) {
                    return Err(VecEyesError::InvalidConfig(
                        "IsolationForest currently requires Word2Vec or FastText embeddings".into(),
                    ));
                }
                let params = config.isolation_forest.clone().unwrap_or_default();
                // Train on the full dataset so the isolation tree sees the real data
                // distribution.  Derive contamination from the observed hot fraction
                // when ground-truth labels are available; fall back to the user value
                // when all samples share the same label.
                let hot_count = samples.iter().filter(|s| s.label == hot_label).count();
                let contamination = if hot_count > 0 && hot_count < samples.len() {
                    (hot_count as f32 / samples.len() as f32).clamp(0.001, 0.49)
                } else {
                    params.contamination
                };
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = IsolationForestModel::fit(
                    &matrix,
                    hot_label,
                    cold_label,
                    params.n_trees,
                    contamination,
                    params.subsample_size,
                    config.threads,
                );
                Ok(Self { pipeline, inner: AdvancedInner::IsolationForest(model) })
            }
            AdvancedMethod::LogisticRegression => {
                let params = config.logistic.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = LogisticOVR::fit(&matrix, samples, params.epochs, params.learning_rate, params.lambda, config.threads)?;
                Ok(Self { pipeline, inner: AdvancedInner::Logistic(model) })
            }
            AdvancedMethod::Svm => {
                let params = config.svm.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = LinearSvmOVR::fit(&matrix, samples, &params, config.threads)?;
                Ok(Self { pipeline, inner: AdvancedInner::Svm(model) })
            }
            AdvancedMethod::RandomForest => {
                let params = config.random_forest.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = RandomForestModel::fit(&matrix, samples, &params, config.threads)?;
                Ok(Self { pipeline, inner: AdvancedInner::RandomForest(model) })
            }
            AdvancedMethod::GradientBoosting => {
                let params = config.gradient_boosting.clone().unwrap_or_default();
                let (pipeline, matrix) = FeaturePipeline::fit(samples, nlp, dims)?;
                let model = GradientBoostingModel::fit(&matrix, samples, params.n_estimators, params.learning_rate, config.threads)?;
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
        self.score_probe(&probe)
    }

    fn score_probe(&self, probe: &DenseMatrix) -> Vec<(ClassificationLabel, f32)> {
        match &self.inner {
            AdvancedInner::Logistic(model) => model.predict_scores(probe),
            AdvancedInner::RandomForest(model) => model.predict_scores(probe),
            AdvancedInner::Svm(model) => model.predict_scores(probe),
            AdvancedInner::GradientBoosting(model) => model.predict_scores(probe),
            AdvancedInner::IsolationForest(model) => model.predict_scores(probe),
        }
    }

    /// Persist the trained model to a JSON file.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let json = serde_json::to_string(self)
            .map_err(|e| VecEyesError::invalid_config("AdvancedClassifier::save", e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a previously saved model from a JSON file.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VecEyesError> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| VecEyesError::invalid_config("AdvancedClassifier::load", e.to_string()))
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
        let hits = if score_sum_mode.is_on() {
            let (boost, hits) = ScoringEngine::compute_rule_boost(text, matchers);
            for (_, score) in &mut labels {
                *score = ScoringEngine::merge_scores(*score, boost, score_sum_mode);
            }
            hits
        } else {
            matchers.iter().flat_map(|m| m.find_matches(text)).collect()
        };
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ClassificationResult { labels, extra_hits: hits }
    }

    fn classify_batch(
        &self,
        texts: &[&str],
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> Vec<ClassificationResult> {
        use rayon::prelude::*;
        if texts.is_empty() { return Vec::new(); }
        // Transform entire batch to matrix in one NLP pipeline call — the main
        // speedup: token lookup + IDF weighting happens once for all N texts.
        let batch_matrix = self.pipeline.transform_batch(texts);
        texts.par_iter().enumerate().map(|(i, &text)| {
            let single = batch_matrix.slice(ndarray::s![i..i+1, ..]).to_owned();
            let mut labels = self.score_probe(&single);
            let hits = if score_sum_mode.is_on() {
                let (boost, hits) = ScoringEngine::compute_rule_boost(text, matchers);
                for (_, score) in &mut labels {
                    *score = ScoringEngine::merge_scores(*score, boost, score_sum_mode);
                }
                hits
            } else {
                matchers.iter().flat_map(|m| m.find_matches(text)).collect()
            };
            labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ClassificationResult { labels, extra_hits: hits }
        }).collect()
    }
}
