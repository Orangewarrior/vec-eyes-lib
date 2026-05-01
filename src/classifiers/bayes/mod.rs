pub(crate) mod core;

use std::collections::HashMap;

use crate::builders::Builder;
use crate::classifier::{
    ClassificationResult, Classifier, ClassifierBuilder, ClassifierFactory, ClassifierMethod,
    ExplainableClassifier, TokenContribution,
};
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
        Self {
            nlp: NlpOption::Count,
            samples: Vec::new(),
            threads: None,
        }
    }

    fn build(self) -> Result<BayesClassifier, VecEyesError> {
        BayesClassifier::train(&self.samples, self.nlp, self.threads)
    }
}

impl BayesBuilder {
    pub fn new() -> Self {
        <Self as Builder<BayesClassifier>>::new()
    }

    pub fn build(self) -> Result<BayesClassifier, VecEyesError> {
        <Self as Builder<BayesClassifier>>::build(self)
    }

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

impl Default for BayesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum BayesFeature {
    Count,
    TfIdf(TfIdfModel),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

pub(crate) struct BayesParts {
    pub(crate) nlp: NlpOption,
    pub(crate) threads: Option<usize>,
    pub(crate) labels: Vec<ClassificationLabel>,
    pub(crate) token_scores: HashMap<ClassificationLabel, HashMap<String, f32>>,
    pub(crate) priors: HashMap<ClassificationLabel, f32>,
    pub(crate) token_totals: HashMap<ClassificationLabel, f32>,
    pub(crate) vocab_size: usize,
    pub(crate) alpha: f32,
    pub(crate) tfidf: Option<TfIdfModel>,
}

impl BayesClassifier {
    pub(crate) fn from_parts(parts: BayesParts) -> Self {
        Self {
            nlp: parts.nlp,
            threads: parts.threads,
            labels: parts.labels,
            token_scores: parts.token_scores,
            priors: parts.priors,
            token_totals: parts.token_totals,
            vocab_size: parts.vocab_size,
            alpha: parts.alpha,
            tfidf: parts.tfidf,
        }
    }

    pub(crate) fn nlp_option(&self) -> NlpOption {
        self.nlp.clone()
    }
    pub(crate) fn threads(&self) -> Option<usize> {
        self.threads
    }
    pub(crate) fn labels(&self) -> &Vec<ClassificationLabel> {
        &self.labels
    }
    pub(crate) fn token_scores(&self) -> &HashMap<ClassificationLabel, HashMap<String, f32>> {
        &self.token_scores
    }
    pub(crate) fn priors(&self) -> &HashMap<ClassificationLabel, f32> {
        &self.priors
    }
    pub(crate) fn token_totals(&self) -> &HashMap<ClassificationLabel, f32> {
        &self.token_totals
    }
    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub(crate) fn alpha(&self) -> f32 {
        self.alpha
    }
    pub(crate) fn tfidf_model(&self) -> Option<&TfIdfModel> {
        self.tfidf.as_ref()
    }

    pub fn train(
        samples: &[TrainingSample],
        nlp: NlpOption,
        threads: Option<usize>,
    ) -> Result<Self, VecEyesError> {
        core::train(samples, nlp, threads)
    }

    /// Persist the trained model to a JSON file.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let json = serde_json::to_string(self)
            .map_err(|e| VecEyesError::invalid_config("BayesClassifier::save", e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a previously saved model from a JSON file.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VecEyesError> {
        let json = crate::security::read_to_string_limited(
            path.as_ref(),
            crate::security::DEFAULT_MAX_MODEL_BYTES,
            "BayesClassifier::load",
        )?;
        let model: Self = serde_json::from_str(&json)
            .map_err(|e| VecEyesError::invalid_config("BayesClassifier::load", e.to_string()))?;
        model.validate_loaded("BayesClassifier::load")?;
        Ok(model)
    }

    /// Save the model as a bincode file (fast, compact).
    pub fn save_bincode<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), VecEyesError> {
        let bytes =
            bincode::serialize(self).map_err(|e| VecEyesError::Serialization(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a model from a bincode file.
    pub fn load_bincode<P: AsRef<std::path::Path>>(path: P) -> Result<Self, VecEyesError> {
        let bytes = crate::security::read_file_limited(
            path.as_ref(),
            crate::security::DEFAULT_MAX_MODEL_BYTES,
            "BayesClassifier::load_bincode",
        )?;
        let model: Self =
            bincode::deserialize(&bytes).map_err(|e| VecEyesError::Serialization(e.to_string()))?;
        model.validate_loaded("BayesClassifier::load_bincode")?;
        Ok(model)
    }

    fn base_scores(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
        core::base_scores(self, text)
    }

    fn validate_loaded(&self, context: &str) -> Result<(), VecEyesError> {
        if self
            .threads
            .is_some_and(|threads| threads > crate::security::MAX_CONFIG_THREADS)
        {
            return Err(VecEyesError::invalid_config(
                context,
                format!("threads must be <= {}", crate::security::MAX_CONFIG_THREADS),
            ));
        }
        if self.labels.is_empty() {
            return Err(VecEyesError::invalid_config(
                context,
                "labels cannot be empty",
            ));
        }
        if self.alpha <= 0.0 || !self.alpha.is_finite() {
            return Err(VecEyesError::invalid_config(
                context,
                "alpha must be a finite positive value",
            ));
        }
        for label in &self.labels {
            if !self.priors.contains_key(label)
                || !self.token_scores.contains_key(label)
                || !self.token_totals.contains_key(label)
            {
                return Err(VecEyesError::invalid_config(
                    context,
                    format!("missing Bayes state for label {}", label.as_str()),
                ));
            }
        }
        Ok(())
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
        ClassificationResult {
            labels,
            extra_hits: hits,
        }
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
        let top = self
            .base_scores(text)
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label);
        let Some(label) = top else {
            return Vec::new();
        };
        let token_map = self.token_scores().get(&label);
        let total_tokens = self.token_totals().get(&label).copied().unwrap_or(0.0);
        let denominator = total_tokens + self.alpha() * self.vocab_size() as f32;
        let mut out = Vec::new();
        for token in tokens {
            let count = token_map
                .and_then(|m| m.get(&token))
                .copied()
                .unwrap_or(0.0);
            let contribution = ((count + self.alpha()) / denominator.max(1e-6)).ln();
            out.push(TokenContribution {
                token,
                contribution,
            });
        }
        out.sort_by(|a, b| {
            b.contribution
                .partial_cmp(&a.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}
