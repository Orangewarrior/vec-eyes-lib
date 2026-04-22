use crate::config::{ExtraMatchConfig, ExtraMatchEngine, RulesFile, ScoreSumMode};
use crate::dataset::{collect_files_recursively, read_text_file};
use crate::error::VecEyesError;
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};
use std::path::Path;

const MAX_RULES_PER_RULESET: usize = 10_000;
const MAX_TEXT_RULE_LINE_LEN: usize = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRule {
    pub title: String,
    pub description: String,
    pub match_rule: String,
    pub score: f32,
}

pub type MatchRule = JsonRule;

#[derive(Debug, Clone)]
pub struct RuleSet {
    pub rules: Vec<MatchRule>,
}

#[derive(Debug, Clone)]
pub struct AlertHit {
    pub title: String,
    pub description: String,
    pub score: f32,
}

pub trait RuleMatcher: Send + Sync {
    fn find_matches(&self, text: &str) -> Vec<AlertHit>;
}

#[derive(Debug, Clone, Copy)]
pub enum MatcherBackend {
    Regex,
    #[cfg(feature = "vectorscan")]
    Vectorscan,
}

pub struct RegexMatcher {
    compiled: Vec<(MatchRule, Regex)>,
}

impl RegexMatcher {
    pub fn from_ruleset(rule_set: &RuleSet) -> Result<Self, VecEyesError> {
        let mut compiled = Vec::new();
        for rule in &rule_set.rules {
            compiled.push((rule.clone(), RegexBuilder::new(&rule.match_rule).size_limit(10_000_000).dfa_size_limit(2_000_000).build()?));
        }
        Ok(Self { compiled })
    }
}

impl RuleMatcher for RegexMatcher {
    fn find_matches(&self, text: &str) -> Vec<AlertHit> {
        let mut hits = Vec::new();
        for (rule, regex) in &self.compiled {
            if regex.is_match(text) {
                hits.push(AlertHit {
                    title: rule.title.clone(),
                    description: rule.description.clone(),
                    score: rule.score,
                });
            }
        }
        hits
    }
}

#[cfg(feature = "vectorscan")]
pub struct VectorScanMatcher {
    inner: RegexMatcher,
}

#[cfg(feature = "vectorscan")]
impl VectorScanMatcher {
    pub fn from_ruleset(rule_set: &RuleSet) -> Result<Self, VecEyesError> {
        let inner = RegexMatcher::from_ruleset(rule_set)?;
        Ok(Self { inner })
    }
}

#[cfg(feature = "vectorscan")]
impl RuleMatcher for VectorScanMatcher {
    fn find_matches(&self, text: &str) -> Vec<AlertHit> {
        // Current safe fallback implementation. The feature flag isolates the native
        // dependency from the default build, but this implementation remains usable
        // even when the host enables the feature before wiring native scanning paths.
        self.inner.find_matches(text)
    }
}

pub struct MatcherFactory;

impl MatcherFactory {
    pub fn build(
        backend: MatcherBackend,
        ruleset: &RuleSet,
    ) -> Result<Box<dyn RuleMatcher>, VecEyesError> {
        match backend {
            MatcherBackend::Regex => Ok(Box::new(RegexMatcher::from_ruleset(ruleset)?)),
            #[cfg(feature = "vectorscan")]
            MatcherBackend::Vectorscan => Ok(Box::new(VectorScanMatcher::from_ruleset(ruleset)?)),
        }
    }

    pub fn build_from_extra_match(
        extra: &ExtraMatchConfig,
    ) -> Result<Box<dyn RuleMatcher>, VecEyesError> {
        let rules = RuleSet::from_rule_path(&extra.path, extra.recursive_way.is_on(), extra.score_add_points, extra.title.clone(), extra.description.clone())?;
        match extra.engine {
            ExtraMatchEngine::Regex => Self::build(MatcherBackend::Regex, &rules),
            ExtraMatchEngine::Vectorscan => {
                #[cfg(feature = "vectorscan")]
                {
                    Self::build(MatcherBackend::Vectorscan, &rules)
                }
                #[cfg(not(feature = "vectorscan"))]
                {
                    Err(VecEyesError::unsupported("matcher::MatcherFactory::build_from_extra_match", "vectorscan feature not enabled; use regex fallback or build with --features vectorscan"))
                }
            }
        }
    }
}

impl RuleSet {
    pub fn from_json_file(path: &Path) -> Result<Self, VecEyesError> {
        let content = read_text_file(path)?;
        let value: serde_json::Value = serde_json::from_str(&content)?;
        if !value.is_array() {
            return Err(VecEyesError::invalid_config("matcher::RuleSet::from_json_file", format!("expected JSON array of rules in {}", path.display())));
        }
        let rules: Vec<MatchRule> = serde_json::from_value(value)?;
        Ok(Self { rules })
    }

    pub fn from_yaml_file(path: &Path) -> Result<Self, VecEyesError> {
        let content = read_text_file(path)?;
        let value: serde_yaml::Value = serde_yaml::from_str(&content)?;
        if !matches!(value, serde_yaml::Value::Sequence(_)) {
            return Err(VecEyesError::invalid_config("matcher::RuleSet::from_yaml_file", format!("expected YAML sequence of rules in {}", path.display())));
        }
        let rules: Vec<MatchRule> = serde_yaml::from_value(value)?;
        Ok(Self { rules })
    }

    pub fn from_rule_path(
        path: &Path,
        recursive: bool,
        score: f32,
        title: Option<String>,
        description: Option<String>,
    ) -> Result<Self, VecEyesError> {
        let mut rules = Vec::new();
        if path.is_file() {
            rules.extend(load_rules_from_single_file(path, score, title.clone(), description.clone())?);
        } else {
            let files = collect_files_recursively(path, recursive)?;
            for file in files {
                rules.extend(load_rules_from_single_file(&file, score, title.clone(), description.clone())?);
            }
        }
        if rules.len() > MAX_RULES_PER_RULESET {
            return Err(VecEyesError::invalid_config(
                "matcher::RuleSet::from_rule_path",
                format!("compiled {} rules which exceeds MAX_RULES_PER_RULESET={}", rules.len(), MAX_RULES_PER_RULESET),
            ));
        }
        Ok(Self { rules })
    }
}

fn load_rules_from_single_file(
    path: &Path,
    score: f32,
    title: Option<String>,
    description: Option<String>,
) -> Result<Vec<MatchRule>, VecEyesError> {
    let extension = path.extension().and_then(|x| x.to_str()).unwrap_or_default().to_ascii_lowercase();
    if extension == "json" {
        return Ok(RuleSet::from_json_file(path)?.rules);
    }
    if extension == "yaml" || extension == "yml" {
        return Ok(RuleSet::from_yaml_file(path)?.rules);
    }

    let content = read_text_file(path)?;
    let file_title = title.unwrap_or_else(|| {
        path.file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("rule")
            .to_string()
    });
    let file_description = description.unwrap_or_else(|| path.to_string_lossy().to_string());

    let mut rules = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.len() > MAX_TEXT_RULE_LINE_LEN {
            return Err(VecEyesError::invalid_config(
                "matcher::load_rules_from_single_file",
                format!("rule line in {} exceeds {} bytes", path.display(), MAX_TEXT_RULE_LINE_LEN),
            ));
        }

        rules.push(MatchRule {
            title: file_title.clone(),
            description: file_description.clone(),
            match_rule: trimmed.to_string(),
            score,
        });
    }

    Ok(rules)
}

pub struct ScoringEngine;

impl ScoringEngine {
    pub fn compute_rule_boost(text: &str, matchers: &[Box<dyn RuleMatcher>]) -> (f32, Vec<AlertHit>) {
        let mut total = 0.0f32;
        let mut hits = Vec::new();
        for matcher in matchers {
            let local_hits = matcher.find_matches(text);
            for hit in local_hits {
                total += hit.score;
                hits.push(hit);
            }
        }
        (total.min(100.0), hits)
    }

    /// Collects only the hits without computing the total boost score.
    /// Useful when ScoreSumMode::Off to avoid unnecessary score accumulation.
    pub fn find_matches_only(text: &str, matchers: &[Box<dyn RuleMatcher>]) -> Vec<AlertHit> {
        let mut hits = Vec::new();
        for matcher in matchers {
            hits.extend(matcher.find_matches(text));
        }
        hits
    }

    /// Merges a base probability in `[0,1]` with an additive rule boost.
    /// Rule scores are kept in a user-friendly `[0, 100]` scale in rule files;
    /// they are divided by 100 here so the result stays in `[0, 1]`.
    pub fn merge_scores(base: f32, rule_boost: f32, mode: ScoreSumMode) -> f32 {
        if mode.is_on() {
            (base + rule_boost / 100.0).min(1.0)
        } else {
            base
        }
    }

    pub fn matchers_from_rules_file(
        rules: &RulesFile,
    ) -> Result<Vec<Box<dyn RuleMatcher>>, VecEyesError> {
        let mut matchers = Vec::new();
        for extra in &rules.extra_match {
            matchers.push(MatcherFactory::build_from_extra_match(extra)?);
        }
        Ok(matchers)
    }
}
