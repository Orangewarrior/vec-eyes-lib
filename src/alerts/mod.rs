//! Alert rules and Vectorscan-backed matching.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use vectorscan_rs::{BlockDatabase, Flag, Pattern, Scan};

use crate::error::VecEyesResult;
use crate::labels::ClassificationLabel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub title: String,
    pub description: String,
    pub match_rule: String,
    pub score: u8,
    #[serde(default)]
    pub target_labels: Vec<ClassificationLabel>,
}

#[derive(Debug, Clone, Default)]
pub struct AlertRuleBuilder {
    title: Option<String>,
    description: Option<String>,
    match_rule: Option<String>,
    score: Option<u8>,
    target_labels: Vec<ClassificationLabel>,
}

impl AlertRuleBuilder {
    pub fn new() -> Self { Self::default() }
    pub fn title(mut self, value: impl Into<String>) -> Self { self.title = Some(value.into()); self }
    pub fn description(mut self, value: impl Into<String>) -> Self { self.description = Some(value.into()); self }
    pub fn match_rule(mut self, value: impl Into<String>) -> Self { self.match_rule = Some(value.into()); self }
    pub fn score(mut self, value: u8) -> Self { self.score = Some(value); self }
    pub fn target_label(mut self, value: ClassificationLabel) -> Self { self.target_labels.push(value); self }
    pub fn build(self) -> AlertRule {
        AlertRule {
            title: self.title.unwrap_or_else(|| "untitled-rule".to_string()),
            description: self.description.unwrap_or_default(),
            match_rule: self.match_rule.unwrap_or_default(),
            score: self.score.unwrap_or(25),
            target_labels: self.target_labels,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRuleMatch {
    pub title: String,
    pub score: f64,
    pub labels: Vec<ClassificationLabel>,
}

#[derive(Debug, Clone)]
pub struct AlertMatcher {
    rules: Vec<AlertRule>,
    database: BlockDatabase,
}

impl AlertMatcher {
    pub fn new(rules: Vec<AlertRule>) -> VecEyesResult<Self> {
        let mut patterns = Vec::new();
        for (index, rule) in rules.iter().enumerate() {
            let pattern = Pattern::new(
                rule.match_rule.as_bytes().to_vec(),
                Flag::CASELESS | Flag::SINGLEMATCH | Flag::UTF8,
                Some(index as u32),
            );
            patterns.push(pattern);
        }
        let database = BlockDatabase::new(patterns)?;
        Ok(Self { rules, database })
    }

    pub fn load_json_file(path: impl AsRef<Path>) -> VecEyesResult<Self> {
        let content = fs::read_to_string(path)?;
        let rules: Vec<AlertRule> = if content.trim_start().starts_with('[') {
            serde_json::from_str(&content)?
        } else {
            vec![serde_json::from_str(&content)?]
        };
        Self::new(rules)
    }

    pub fn merge_json_files(paths: &[impl AsRef<Path>]) -> VecEyesResult<Option<Self>> {
        let mut rules = Vec::new();
        for path in paths {
            let content = fs::read_to_string(path.as_ref())?;
            if content.trim_start().starts_with('[') {
                rules.extend(serde_json::from_str::<Vec<AlertRule>>(&content)?);
            } else {
                rules.push(serde_json::from_str::<AlertRule>(&content)?);
            }
        }
        if rules.is_empty() { Ok(None) } else { Ok(Some(Self::new(rules)?)) }
    }

    pub fn from_plain_list(
        path: impl AsRef<Path>,
        title_prefix: &str,
        description: &str,
        default_score: u8,
        target_labels: &[ClassificationLabel],
        token_kind: &str,
    ) -> VecEyesResult<Self> {
        let mut rules = Vec::new();
        let content = fs::read_to_string(path)?;
        for (index, line) in content.lines().enumerate() {
            let item = line.trim();
            if item.is_empty() || item.starts_with('#') {
                continue;
            }
            let escaped = escape_for_vectorscan(item);
            let match_rule = if token_kind.eq_ignore_ascii_case("ip") {
                format!(r"(?:^|[^0-9]){}(?:$|[^0-9])", escaped)
            } else {
                format!(r"(?:https?://)?[^\s]*{}[^\s]*", escaped)
            };
            let mut builder = AlertRuleBuilder::new()
                .title(format!("{}-{}", title_prefix, index + 1))
                .description(description)
                .match_rule(match_rule)
                .score(default_score);
            for label in target_labels {
                builder = builder.target_label(*label);
            }
            rules.push(builder.build());
        }
        Self::new(rules)
    }

    pub fn scan(&self, input: &str) -> VecEyesResult<Vec<AlertRuleMatch>> {
        let mut scanner = self.database.create_scanner()?;
        let mut matches_by_rule = BTreeMap::<usize, AlertRuleMatch>::new();

        scanner.scan(input.as_bytes(), |id, _from, _to, _flags| {
            let rule_index = id as usize;
            if let Some(rule) = self.rules.get(rule_index) {
                matches_by_rule.entry(rule_index).or_insert(AlertRuleMatch {
                    title: rule.title.clone(),
                    score: rule.score as f64,
                    labels: if rule.target_labels.is_empty() {
                        vec![ClassificationLabel::BlockList]
                    } else {
                        rule.target_labels.clone()
                    },
                });
            }
            Scan::Continue
        })?;

        Ok(matches_by_rule.into_values().collect())
    }

    pub fn boost_scores(
        &self,
        input: &str,
        class_scores: &mut HashMap<ClassificationLabel, f64>,
    ) -> VecEyesResult<Vec<AlertRuleMatch>> {
        let matches = self.scan(input)?;
        for rule_match in &matches {
            for label in &rule_match.labels {
                let entry = class_scores.entry(*label).or_insert(0.0);
                *entry = (*entry + rule_match.score).clamp(0.0, 100.0);
            }
        }
        Ok(matches)
    }
}

fn escape_for_vectorscan(input: &str) -> String {
    let mut output = String::new();
    for ch in input.chars() {
        match ch {
            '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\' => {
                output.push('\\');
                output.push(ch);
            }
            _ => output.push(ch),
        }
    }
    output
}
