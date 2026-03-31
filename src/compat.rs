
use crate::builders::Builder;
use crate::classifier::Classifier;
use crate::config::ScoreSumMode;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::FastTextConfig;
use regex::RegexBuilder;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum RepresentationKind {
    Count,
    TfIdf,
    Word2Vec,
    FastText,
}

#[derive(Debug, Clone)]
pub struct NlpPipeline {
    pub representation: RepresentationKind,
    pub fasttext_config: Option<FastTextConfig>,
}

pub struct NlpPipelineBuilder {
    representation: Option<RepresentationKind>,
    fasttext_config: Option<FastTextConfig>,
}

impl Builder<NlpPipeline> for NlpPipelineBuilder {
    fn new() -> Self {
        Self {
            representation: None,
            fasttext_config: None,
        }
    }

    fn build(self) -> Result<NlpPipeline, VecEyesError> {
        Ok(NlpPipeline {
            representation: self.representation.unwrap_or(RepresentationKind::Word2Vec),
            fasttext_config: self.fasttext_config,
        })
    }
}

impl NlpPipelineBuilder {
    pub fn new() -> Self { <Self as Builder<NlpPipeline>>::new() }
    pub fn build(self) -> Result<NlpPipeline, VecEyesError> { <Self as Builder<NlpPipeline>>::build(self) }

    pub fn representation(mut self, representation: RepresentationKind) -> Self {
        self.representation = Some(representation);
        self
    }

    pub fn fasttext_config(mut self, config: FastTextConfig) -> Self {
        self.fasttext_config = Some(config);
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct OutputWriters {
    disabled: bool,
}

impl OutputWriters {
    pub fn disabled() -> Self {
        Self { disabled: true }
    }

    pub fn is_disabled(&self) -> bool {
        self.disabled
    }
}

pub mod alerts {
    use super::*;

    #[derive(Debug, Clone, Deserialize)]
    struct AlertRule {
        title: String,
        description: String,
        match_rule: String,
        score: f32,
        #[serde(default)]
        target_labels: Vec<ClassificationLabel>,
    }

    #[derive(Debug, Clone)]
    pub struct AlertMatcher {
        compiled: Vec<CompiledAlertRule>,
    }

    #[derive(Debug, Clone)]
    struct CompiledAlertRule {
        title: String,
        description: String,
        score: f32,
        target_labels: Vec<ClassificationLabel>,
        regex: regex::Regex,
    }

    impl AlertMatcher {
        pub fn load_json_file<P: AsRef<Path>>(path: P) -> Result<Self, VecEyesError> {
            let content = std::fs::read_to_string(path)?;
            let rules: Vec<AlertRule> = serde_json::from_str(&content)?;
            let mut compiled = Vec::with_capacity(rules.len());
            for rule in rules {
                let regex = RegexBuilder::new(&rule.match_rule)
                    .size_limit(10_000_000)
                    .dfa_size_limit(2_000_000)
                    // .unicode(false)
                    .build()?;
                compiled.push(CompiledAlertRule {
                    title: rule.title,
                    description: rule.description,
                    score: rule.score,
                    target_labels: rule.target_labels,
                    regex,
                });
            }
            Ok(Self { compiled })
        }

        pub fn classify_labels(&self, text: &str) -> Vec<(ClassificationLabel, f32)> {
            let mut scores: HashMap<ClassificationLabel, f32> = HashMap::new();
            for rule in &self.compiled {
                if rule.regex.is_match(text) {
                    for label in &rule.target_labels {
                        let entry = scores.entry(label.clone()).or_insert(0.0);
                        *entry = entry.max(rule.score);
                    }
                }
            }
            let mut out: Vec<_> = scores.into_iter().collect();
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            out
        }

        pub fn has_alerts(&self, text: &str) -> bool {
            self.compiled.iter().any(|rule| rule.regex.is_match(text))
        }

        pub fn alert_titles(&self, text: &str) -> Vec<String> {
            let mut titles = Vec::new();
            for rule in &self.compiled {
                if rule.regex.is_match(text) {
                    titles.push(format!("{}: {}", rule.title, rule.description));
                }
            }
            titles
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineReport {
    pub classifications: Vec<(ClassificationLabel, f32)>,
}

pub struct EngineBuilder {
    model: Option<Box<dyn Classifier>>,
    alerts: Option<alerts::AlertMatcher>,
    output: Option<OutputWriters>,
}

impl Builder<Engine> for EngineBuilder {
    fn new() -> Self {
        Self {
            model: None,
            alerts: None,
            output: None,
        }
    }

    fn build(self) -> Result<Engine, VecEyesError> {
        Ok(Engine {
            model: self.model.ok_or_else(|| VecEyesError::invalid_config("compat::EngineBuilder::build", "model is required"))?,
            alerts: self.alerts,
            output: self.output.unwrap_or_default(),
        })
    }
}

impl EngineBuilder {
    pub fn new() -> Self { <Self as Builder<Engine>>::new() }
    pub fn build(self) -> Result<Engine, VecEyesError> { <Self as Builder<Engine>>::build(self) }

    pub fn model(mut self, model: Box<dyn Classifier>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn alerts(mut self, alerts: alerts::AlertMatcher) -> Self {
        self.alerts = Some(alerts);
        self
    }

    pub fn output(mut self, output: OutputWriters) -> Self {
        self.output = Some(output);
        self
    }
}

pub struct Engine {
    model: Box<dyn Classifier>,
    alerts: Option<alerts::AlertMatcher>,
    output: OutputWriters,
}

impl Engine {
    pub fn classify_text(&self, text: &str, _source: &str) -> Result<EngineReport, VecEyesError> {
        let mut labels = self.model.classify_text(text, ScoreSumMode::Off, &[]).labels;
        if let Some(alerts) = &self.alerts {
            for (label, score) in alerts.classify_labels(text) {
                if let Some(existing) = labels.iter_mut().find(|(existing, _)| *existing == label) {
                    existing.1 = existing.1.max(score);
                } else {
                    labels.push((label, score));
                }
            }
        }
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if self.output.is_disabled() {
            // intentionally noop for compat mode
        }
        Ok(EngineReport { classifications: labels })
    }
}
