//! Factory façade for selecting Vec-Eyes classifiers.
//!
//! This module keeps the public construction flow explicit and stable:
//! choose a method, configure NLP, load datasets, and build a classifier.

pub mod typed;

pub use crate::classifier::{
    BayesBuilder, ClassifierBuilder, ClassifierFactory, ClassifierMethod, DistanceMetric,
    KnnBuilder, MethodKind,
};

use crate::config::RulesFile;

/// Creates a [`ClassifierBuilder`] from a validated YAML rules file.
#[inline]
pub fn builder_from_rules(rules: &RulesFile) -> ClassifierBuilder {
    ClassifierFactory::builder().from_rules_file(rules)
}

pub use typed::TypedClassifierBuilder;
