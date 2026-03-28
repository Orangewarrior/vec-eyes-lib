use crate::config::{RulesFile, ScoreSumMode};
use crate::dataset::{load_training_samples, read_text_file, TrainingSample};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::matcher::{AlertHit, RuleMatcher, ScoringEngine};
use crate::nlp::{
    dense_matrix_from_texts, fit_tfidf, transform_tfidf, DenseMatrix, FastTextConfigBuilder,
    NlpOption, TfIdfModel, WordEmbeddingModel,
};
use chrono::Utc;
use ndarray::Axis;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MethodKind {
    #[serde(alias = "bayes", alias = "Bayes")]
    Bayes,
    #[serde(alias = "knn-cosine", alias = "KnnCosine")]
    KnnCosine,
    #[serde(alias = "knn-euclidean", alias = "KnnEuclidean")]
    KnnEuclidean,
    #[serde(alias = "knn-manhattan", alias = "KnnManhattan")]
    KnnManhattan,
    #[serde(alias = "knn-minkowski", alias = "KnnMinkowski")]
    KnnMinkowski,
}

impl MethodKind {
    pub fn is_knn(&self) -> bool {
        !matches!(self, Self::Bayes)
    }

    pub fn requires_p(&self) -> bool {
        matches!(self, Self::KnnMinkowski)
    }
}

#[derive(Debug, Clone)]
pub enum ClassifierMethod {
    Bayes,
    KnnCosine,
    KnnEuclidean,
    KnnManhattan,
    KnnMinkowski,
}

impl From<MethodKind> for ClassifierMethod {
    fn from(value: MethodKind) -> Self {
        match value {
            MethodKind::Bayes => Self::Bayes,
            MethodKind::KnnCosine => Self::KnnCosine,
            MethodKind::KnnEuclidean => Self::KnnEuclidean,
            MethodKind::KnnManhattan => Self::KnnManhattan,
            MethodKind::KnnMinkowski => Self::KnnMinkowski,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub labels: Vec<(ClassificationLabel, f32)>,
    pub extra_hits: Vec<AlertHit>,
}

pub trait Classifier {
    fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult;
}

pub struct ClassifierFactory;

impl ClassifierFactory {
    pub fn builder() -> ClassifierBuilder {
        ClassifierBuilder::new()
    }
}

pub struct ClassifierBuilder {
    method: Option<ClassifierMethod>,
    nlp: Option<NlpOption>,
    hot_label: Option<ClassificationLabel>,
    cold_label: Option<ClassificationLabel>,
    hot_path: Option<PathBuf>,
    cold_path: Option<PathBuf>,
    recursive: bool,
    threads: Option<usize>,
    k: Option<usize>,
    p: Option<f32>,
}

impl ClassifierBuilder {
    pub fn new() -> Self {
        Self {
            method: None,
            nlp: None,
            hot_label: None,
            cold_label: None,
            hot_path: None,
            cold_path: None,
            recursive: true,
            threads: None,
            k: None,
            p: None,
        }
    }

    pub fn method(mut self, method: ClassifierMethod) -> Self {
        self.method = Some(method);
        self
    }

    pub fn nlp(mut self, nlp: NlpOption) -> Self {
        self.nlp = Some(nlp);
        self
    }

    pub fn hot_label(mut self, label: ClassificationLabel) -> Self {
        self.hot_label = Some(label);
        self
    }

    pub fn cold_label(mut self, label: ClassificationLabel) -> Self {
        self.cold_label = Some(label);
        self
    }

    pub fn hot_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.hot_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn cold_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.cold_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn threads(mut self, threads: Option<usize>) -> Self {
        self.threads = threads;
        self
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    pub fn p(mut self, p: f32) -> Self {
        self.p = Some(p);
        self
    }

    pub fn from_rules_file(mut self, rules: &RulesFile) -> Self {
        self.method = Some(rules.method.clone().into());
        self.nlp = Some(rules.nlp.clone());
        self.hot_path = Some(rules.hot_test_path.clone());
        self.cold_path = Some(rules.cold_test_path.clone());
        self.hot_label = Some(rules.hot_label.clone().unwrap_or(ClassificationLabel::WebAttack));
        self.cold_label = Some(rules.cold_label.clone().unwrap_or(ClassificationLabel::RawData));
        self.recursive = rules.recursive_way.is_on();
        self.threads = rules.threads;
        self.k = rules.k;
        self.p = rules.p;
        self
    }

    pub fn build(self) -> Result<Box<dyn Classifier>, VecEyesError> {
        let method = self.method.ok_or_else(|| VecEyesError::InvalidConfig("missing method".into()))?;
        let nlp = self.nlp.ok_or_else(|| VecEyesError::InvalidConfig("missing nlp option".into()))?;
        let hot_label = self.hot_label.ok_or_else(|| VecEyesError::InvalidConfig("missing hot label".into()))?;
        let cold_label = self.cold_label.unwrap_or(ClassificationLabel::RawData);

        let mut samples = Vec::new();

        if let Some(hot_path) = &self.hot_path {
            let mut hot = load_training_samples(hot_path, hot_label, self.recursive)?;
            samples.append(&mut hot);
        }
        if let Some(cold_path) = &self.cold_path {
            let mut cold = load_training_samples(cold_path, cold_label, self.recursive)?;
            samples.append(&mut cold);
        }

        match method {
            ClassifierMethod::Bayes => Ok(Box::new(BayesBuilder::new().nlp(nlp).samples(samples).build()?)),
            ClassifierMethod::KnnCosine => {
                let k = require_k(self.k)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).k(k).cosine().build()?))
            }
            ClassifierMethod::KnnEuclidean => {
                let k = require_k(self.k)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).k(k).euclidean().build()?))
            }
            ClassifierMethod::KnnManhattan => {
                let k = require_k(self.k)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).k(k).manhattan().build()?))
            }
            ClassifierMethod::KnnMinkowski => {
                let k = require_k(self.k)?;
                let p = require_p(self.p)?;
                Ok(Box::new(KnnBuilder::new().nlp(nlp).samples(samples).k(k).p(p).minkowski(p).build()?))
            }
        }
    }
}

fn require_k(value: Option<usize>) -> Result<usize, VecEyesError> {
    let k = value.ok_or_else(|| VecEyesError::InvalidConfig("KNN requires field 'k' and it must be passed explicitly".into()))?;
    if k == 0 {
        return Err(VecEyesError::InvalidConfig("KNN requires k >= 1".into()));
    }
    Ok(k)
}

fn require_p(value: Option<f32>) -> Result<f32, VecEyesError> {
    let p = value.ok_or_else(|| VecEyesError::InvalidConfig("KNN Minkowski requires field 'p'".into()))?;
    if p <= 0.0 {
        return Err(VecEyesError::InvalidConfig("KNN Minkowski requires p > 0".into()));
    }
    Ok(p)
}

#[derive(Debug, Clone)]
pub struct BayesBuilder {
    nlp: NlpOption,
    samples: Vec<TrainingSample>,
}

impl BayesBuilder {
    pub fn new() -> Self {
        Self { nlp: NlpOption::Count, samples: Vec::new() }
    }

    pub fn nlp(mut self, nlp: NlpOption) -> Self {
        self.nlp = nlp;
        self
    }

    pub fn samples(mut self, samples: Vec<TrainingSample>) -> Self {
        self.samples = samples;
        self
    }

    pub fn build(self) -> Result<BayesClassifier, VecEyesError> {
        BayesClassifier::train(&self.samples, self.nlp)
    }
}

#[derive(Debug, Clone)]
pub enum BayesFeature {
    Count,
    TfIdf(TfIdfModel),
}

#[derive(Debug, Clone)]
pub struct BayesClassifier {
    nlp: NlpOption,
    labels: Vec<ClassificationLabel>,
    token_scores: HashMap<ClassificationLabel, HashMap<String, f32>>,
    priors: HashMap<ClassificationLabel, f32>,
    tfidf: Option<TfIdfModel>,
}

impl BayesClassifier {
    pub fn train(samples: &[TrainingSample], nlp: NlpOption) -> Result<Self, VecEyesError> {
        let mut token_scores: HashMap<ClassificationLabel, HashMap<String, f32>> = HashMap::new();
        let mut label_counts: HashMap<ClassificationLabel, usize> = HashMap::new();
        let texts: Vec<String> = samples.iter().map(|s| s.text.clone()).collect();
        let tfidf = if nlp == NlpOption::TfIdf { Some(fit_tfidf(&texts)) } else { None };

        for sample in samples {
            *label_counts.entry(sample.label.clone()).or_insert(0) += 1;
            let entry = token_scores.entry(sample.label.clone()).or_default();
            let normalized = crate::nlp::normalize_text(&sample.text);
            let tokens = crate::nlp::tokenize(&normalized);
            for token in tokens {
                *entry.entry(token).or_insert(0.0) += 1.0;
            }
        }

        let total = samples.len() as f32;
        let mut priors = HashMap::new();
        let mut labels = Vec::new();
        for (label, count) in label_counts {
            labels.push(label.clone());
            priors.insert(label, count as f32 / total.max(1.0));
        }

        Ok(Self { nlp, labels, token_scores, priors, tfidf })
    }

    fn base_scores(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        let normalized = crate::nlp::normalize_text(text);
        let tokens = crate::nlp::tokenize(&normalized);
        let mut raw = Vec::new();

        for label in &self.labels {
            let prior = self.priors.get(label).copied().unwrap_or(0.01).ln();
            let token_map = self.token_scores.get(label);
            let mut score = prior;

            if self.nlp == NlpOption::TfIdf {
                if let Some(model) = &self.tfidf {
                    let matrix = transform_tfidf(model, &[text.to_string()]);
                    for token in &tokens {
                        if let Some(index) = model.token_to_index.get(token) {
                            let weight = matrix[[0, *index]].max(1e-6);
                            let token_count = token_map.and_then(|m| m.get(token)).copied().unwrap_or(1.0);
                            score += (token_count * weight).ln();
                        }
                    }
                }
            } else {
                for token in &tokens {
                    let token_count = token_map.and_then(|m| m.get(token)).copied().unwrap_or(1.0);
                    score += token_count.ln();
                }
            }

            raw.push((label.clone(), score));
        }

        softmax_scores(&raw)
    }
}

impl Classifier for BayesClassifier {
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

#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Minkowski(f32),
}

#[derive(Debug, Clone)]
pub struct KnnBuilder {
    nlp: NlpOption,
    samples: Vec<TrainingSample>,
    metric: DistanceMetric,
    dims: usize,
    k: Option<usize>,
    p: Option<f32>,
}

impl KnnBuilder {
    pub fn new() -> Self {
        Self {
            nlp: NlpOption::Word2Vec,
            samples: Vec::new(),
            metric: DistanceMetric::Cosine,
            dims: 32,
            k: None,
            p: None,
        }
    }

    pub fn nlp(mut self, nlp: NlpOption) -> Self {
        self.nlp = nlp;
        self
    }

    pub fn samples(mut self, samples: Vec<TrainingSample>) -> Self {
        self.samples = samples;
        self
    }

    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    pub fn p(mut self, p: f32) -> Self {
        self.p = Some(p);
        self
    }

    pub fn cosine(mut self) -> Self {
        self.metric = DistanceMetric::Cosine;
        self
    }

    pub fn euclidean(mut self) -> Self {
        self.metric = DistanceMetric::Euclidean;
        self
    }

    pub fn manhattan(mut self) -> Self {
        self.metric = DistanceMetric::Manhattan;
        self
    }

    pub fn minkowski(mut self, p: f32) -> Self {
        self.metric = DistanceMetric::Minkowski(p);
        self.p = Some(p);
        self
    }

    pub fn build(self) -> Result<KnnClassifier, VecEyesError> {
        let k = require_k(self.k)?;
        if let DistanceMetric::Minkowski(_) = self.metric {
            require_p(self.p)?;
        }
        KnnClassifier::train(&self.samples, self.nlp, self.metric, self.dims, k)
    }
}

#[derive(Debug, Clone)]
pub enum DenseFeatureModel {
    Word2Vec(WordEmbeddingModel),
    FastText(WordEmbeddingModel),
}

#[derive(Debug, Clone)]
pub struct KnnClassifier {
    metric: DistanceMetric,
    labels: Vec<ClassificationLabel>,
    matrix: DenseMatrix,
    model: DenseFeatureModel,
    k: usize,
}

impl KnnClassifier {
    pub fn train(
        samples: &[TrainingSample],
        nlp: NlpOption,
        metric: DistanceMetric,
        dims: usize,
        k: usize,
    ) -> Result<Self, VecEyesError> {
        let texts: Vec<String> = samples.iter().map(|s| s.text.clone()).collect();
        let labels: Vec<ClassificationLabel> = samples.iter().map(|s| s.label.clone()).collect();

        let model = match nlp {
            NlpOption::Word2Vec => DenseFeatureModel::Word2Vec(WordEmbeddingModel::train_word2vec(&texts, dims)),
            NlpOption::FastText => {
                let config = FastTextConfigBuilder::new().build();
                DenseFeatureModel::FastText(WordEmbeddingModel::train_fasttext(&texts, dims, config))
            }
            _ => return Err(VecEyesError::InvalidConfig("KNN requires Word2Vec or FastText".into())),
        };

        let matrix = match &model {
            DenseFeatureModel::Word2Vec(inner) => dense_matrix_from_texts(inner, &texts),
            DenseFeatureModel::FastText(inner) => dense_matrix_from_texts(inner, &texts),
        };

        Ok(Self { metric, labels, matrix, model, k })
    }

    fn matrix_for_text(&self, text: &str) -> DenseMatrix {
        let texts = vec![text.to_string()];
        match &self.model {
            DenseFeatureModel::Word2Vec(inner) => dense_matrix_from_texts(inner, &texts),
            DenseFeatureModel::FastText(inner) => dense_matrix_from_texts(inner, &texts),
        }
    }

    fn score_neighbors(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        let probe = self.matrix_for_text(text);
        let probe_row = probe.index_axis(Axis(0), 0);
        let probe_vec = probe_row.to_vec();
        let mut ranked: Vec<(f32, ClassificationLabel)> = Vec::new();

        for row in 0..self.matrix.shape()[0] {
            let candidate = self.matrix.index_axis(Axis(0), row);
            let candidate_vec = candidate.to_vec();
            let distance = match self.metric {
                DistanceMetric::Cosine => cosine_distance(&probe_vec, &candidate_vec),
                DistanceMetric::Euclidean => euclidean_distance(&probe_vec, &candidate_vec),
                DistanceMetric::Manhattan => manhattan_distance(&probe_vec, &candidate_vec),
                DistanceMetric::Minkowski(p) => minkowski_distance(&probe_vec, &candidate_vec, p),
            };
            ranked.push((distance, self.labels[row].clone()));
        }

        ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut best: HashMap<ClassificationLabel, f32> = HashMap::new();
        let limit = self.k.min(ranked.len());
        for (distance, label) in ranked.into_iter().take(limit) {
            let score = (1.0 / (distance + 1e-6)).min(1000.0);
            *best.entry(label).or_insert(0.0) += score;
        }

        let mut raw = Vec::new();
        for (label, score) in best {
            raw.push((label, score));
        }
        softmax_scores(&raw)
    }
}

impl Classifier for KnnClassifier {
    fn classify_text(
        &self,
        text: &str,
        score_sum_mode: ScoreSumMode,
        matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult {
        let mut labels = self.score_neighbors(text);
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

pub fn run_rules_pipeline(
    rules: &RulesFile,
    classify_objects: &Path,
) -> Result<crate::report::ClassificationReport, VecEyesError> {
    rules.validate()?;

    if !classify_objects.exists() {
        return Err(VecEyesError::InvalidConfig(format!(
            "classify_objects path does not exist: {}",
            classify_objects.display()
        )));
    }

    let builder = ClassifierFactory::builder().from_rules_file(rules);
    let classifier = builder.build()?;
    let matchers = ScoringEngine::matchers_from_rules_file(rules)?;
    let mut report = crate::report::ClassificationReport::new(rules.report_name.clone().unwrap_or_else(|| "Vec-Eyes Report".to_string()));

    let files = crate::dataset::collect_files_recursively(classify_objects, rules.recursive_way.is_on())?;
    for file in files {
        let text = read_text_file(&file)?;
        let result = classifier.classify_text(&text, rules.score_sum, &matchers);
        let labels: Vec<String> = result.labels.iter().map(|(l, s)| format!("{}:{:.2}", l, s)).collect();
        let top_score = result.labels.first().map(|x| x.1).unwrap_or(0.0);
        let hit_titles: Vec<String> = result.extra_hits.iter().map(|h| h.title.clone()).collect();
        report.records.push(crate::report::ClassificationRecord {
            title_object: file.file_name().and_then(|x| x.to_str()).unwrap_or("object").to_string(),
            name_file_dataset: file.to_string_lossy().to_string(),
            classify_names_list: labels.join(","),
            date_of_occurrence: Utc::now(),
            score_percent: top_score,
            match_titles: hit_titles.join(","),
        });
    }

    Ok(report)
}

fn softmax_scores(input: &[(ClassificationLabel, f32)]) -> Vec<(ClassificationLabel, f32)> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut max_score = f32::NEG_INFINITY;
    for (_, score) in input {
        if *score > max_score {
            max_score = *score;
        }
    }

    let mut sum = 0.0f32;
    let mut exp_values = Vec::new();
    for (label, score) in input {
        let value = (*score - max_score).exp();
        sum += value;
        exp_values.push((label.clone(), value));
    }

    let mut output = Vec::new();
    for (label, value) in exp_values {
        let pct = if sum > 0.0 { (value / sum) * 100.0 } else { 0.0 };
        output.push((label, pct));
    }
    output
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for idx in 0..a.len() {
        dot += a[idx] * b[idx];
        na += a[idx] * a[idx];
        nb += b[idx] * b[idx];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        let d = a[idx] - b[idx];
        acc += d * d;
    }
    acc.sqrt()
}

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        acc += (a[idx] - b[idx]).abs();
    }
    acc
}

fn minkowski_distance(a: &[f32], b: &[f32], p: f32) -> f32 {
    let mut acc = 0.0f32;
    for idx in 0..a.len() {
        acc += (a[idx] - b[idx]).abs().powf(p);
    }
    acc.powf(1.0 / p)
}
