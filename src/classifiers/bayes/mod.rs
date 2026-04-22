pub(crate) mod core;

use std::collections::HashMap;

use crate::builders::Builder;
use crate::classifier::{ClassificationResult, Classifier, ClassifierBuilder, ClassifierFactory, ClassifierMethod, ExplainableClassifier, TokenContribution};
use crate::config::ScoreSumMode;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::matcher::{RuleMatcher, ScoringEngine};
use crate::nlp::{NlpOption, TfIdfModel};

/// Entry point for the Naive Bayes classifier family.
pub struct BayesModule;

impl BayesModule {
    #[inline]
    pub fn builder() -> BayesBuilder {
        BayesBuilder::new()
    }

    #[inline]
    pub fn factory() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::Bayes)
    }
}

#[derive(Debug, Clone)]
pub struct BayesBuilder {
    nlp: NlpOption,
    samples: Vec<TrainingSample>,
    threads: Option<usize>,
}

impl Builder<BayesClassifier> for BayesBuilder {
    fn new() -> Self {
        Self { nlp: NlpOption::Count, samples: Vec::new(), threads: None }
    }

    fn build(self) -> Result<BayesClassifier, VecEyesError> {
        BayesClassifier::train(&self.samples, self.nlp, self.threads)
    }
}

impl BayesBuilder {
    pub fn new() -> Self { <Self as Builder<BayesClassifier>>::new() }

    pub fn build(self) -> Result<BayesClassifier, VecEyesError> { <Self as Builder<BayesClassifier>>::build(self) }

    pub fn nlp(mut self, nlp: NlpOption) -> Self {
        self.nlp = nlp;
        self
    }

    pub fn samples(mut self, samples: Vec<TrainingSample>) -> Self {
        self.samples = samples;
        self
    }

    pub fn threads(mut self, threads: Option<usize>) -> Self {
        self.threads = threads;
        self
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
    threads: Option<usize>,
    labels: Vec<ClassificationLabel>,
    token_scores: HashMap<ClassificationLabel, HashMap<String, f32>>,
    priors: HashMap<ClassificationLabel, f32>,
    token_totals: HashMap<ClassificationLabel, f32>,
    vocab_size: usize,
    alpha: f32,
    tfidf: Option<TfIdfModel>,
}

impl BayesClassifier {
    pub(crate) fn from_parts(
        nlp: NlpOption,
        threads: Option<usize>,
        labels: Vec<ClassificationLabel>,
        token_scores: HashMap<ClassificationLabel, HashMap<String, f32>>,
        priors: HashMap<ClassificationLabel, f32>,
        token_totals: HashMap<ClassificationLabel, f32>,
        vocab_size: usize,
        alpha: f32,
        tfidf: Option<TfIdfModel>,
    ) -> Self {
        Self { nlp, threads, labels, token_scores, priors, token_totals, vocab_size, alpha, tfidf }
    }

    pub(crate) fn nlp_option(&self) -> NlpOption { self.nlp.clone() }
    pub(crate) fn threads(&self) -> Option<usize> { self.threads }
    pub(crate) fn labels(&self) -> &Vec<ClassificationLabel> { &self.labels }
    pub(crate) fn token_scores(&self) -> &HashMap<ClassificationLabel, HashMap<String, f32>> { &self.token_scores }
    pub(crate) fn priors(&self) -> &HashMap<ClassificationLabel, f32> { &self.priors }
    pub(crate) fn token_totals(&self) -> &HashMap<ClassificationLabel, f32> { &self.token_totals }
    pub(crate) fn vocab_size(&self) -> usize { self.vocab_size }
    pub(crate) fn alpha(&self) -> f32 { self.alpha }
    pub(crate) fn tfidf_model(&self) -> Option<&TfIdfModel> { self.tfidf.as_ref() }

    pub fn train(samples: &[TrainingSample], nlp: NlpOption, threads: Option<usize>) -> Result<Self, VecEyesError> {
        core::train(samples, nlp, threads)
    }

    fn base_scores(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        core::base_scores(self, text)
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
}

impl From<BayesClassifier> for Box<dyn Classifier> {
    fn from(value: BayesClassifier) -> Self {
        Box::new(value)
    }
}


impl ExplainableClassifier for BayesClassifier {
    fn explain(&self, text: &str) -> Vec<TokenContribution> {
        let normalized = crate::nlp::normalize_text(text);
        let tokens = crate::nlp::tokenize(&normalized);
        let top = self.base_scores(text).into_iter().max_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(label, _)| label);
        let Some(label) = top else { return Vec::new(); };
        let token_map = self.token_scores().get(&label);
        let total_tokens = self.token_totals().get(&label).copied().unwrap_or(0.0);
        let denominator = total_tokens + self.alpha() * self.vocab_size() as f32;
        let mut out = Vec::new();
        for token in tokens {
            let count = token_map.and_then(|m| m.get(&token)).copied().unwrap_or(0.0);
            let contribution = ((count + self.alpha()) / denominator.max(1e-6)).ln();
            out.push(TokenContribution { token, contribution });
        }
        out.sort_by(|a,b| b.contribution.partial_cmp(&a.contribution).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}
