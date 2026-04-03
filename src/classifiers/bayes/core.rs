
use std::collections::{HashMap, HashSet};

use crate::classifier::softmax_scores;
use crate::classifiers::bayes::BayesClassifier;
use crate::dataset::TrainingSample;
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::{fit_tfidf, transform_tfidf, NlpOption};
use crate::parallel::install_pool;

pub(crate) fn train(samples: &[TrainingSample], nlp: NlpOption, threads: Option<usize>) -> Result<BayesClassifier, VecEyesError> {
    let mut token_scores: HashMap<ClassificationLabel, HashMap<String, f32>> = HashMap::new();
    let mut token_totals: HashMap<ClassificationLabel, f32> = HashMap::new();
    let mut label_counts: HashMap<ClassificationLabel, usize> = HashMap::new();
    let mut vocab = HashSet::new();
    let texts: Vec<String> = samples.iter().map(|s| s.text.clone()).collect();
    let tfidf = if nlp == NlpOption::TfIdf { Some(fit_tfidf(&texts)) } else { None };

    for sample in samples {
        *label_counts.entry(sample.label.clone()).or_insert(0) += 1;
        let entry = token_scores.entry(sample.label.clone()).or_default();
        let normalized = crate::nlp::normalize_text(&sample.text);
        let tokens = crate::nlp::tokenize(&normalized);
        let total_entry = token_totals.entry(sample.label.clone()).or_insert(0.0);
        for token in tokens {
            vocab.insert(token.clone());
            *entry.entry(token).or_insert(0.0) += 1.0;
            *total_entry += 1.0;
        }
    }

    let total = samples.len() as f32;
    let mut priors = HashMap::new();
    let mut labels = Vec::new();
    for (label, count) in label_counts {
        labels.push(label.clone());
        priors.insert(label, count as f32 / total.max(1.0));
    }

    Ok(BayesClassifier::from_parts(
        nlp,
        threads,
        labels,
        token_scores,
        priors,
        token_totals,
        vocab.len().max(1),
        1.0,
        tfidf,
    ))
}

pub(crate) fn base_scores(model: &BayesClassifier, text: &str) -> Vec<(ClassificationLabel, f32)> {
    let normalized = crate::nlp::normalize_text(text);
    let tokens = crate::nlp::tokenize(&normalized);
    let labels = model.labels().clone();
    let tfidf_matrix = if model.nlp_option() == NlpOption::TfIdf {
        model.tfidf_model().as_ref().map(|m| transform_tfidf(m, &[text.to_string()]))
    } else {
        None
    };
    let vocab_size = model.vocab_size() as f32;
    let alpha = model.alpha();

    let raw = install_pool(model.threads(), || {
        use rayon::prelude::*;
        labels.par_iter().map(|label| {
            let prior = model.priors().get(label).copied().unwrap_or(0.01).ln();
            let token_map = model.token_scores().get(label);
            let total_tokens = model.token_totals().get(label).copied().unwrap_or(0.0);
            let denominator = total_tokens + alpha * vocab_size;
            let mut score = prior;

            if model.nlp_option() == NlpOption::TfIdf {
                if let (Some(tfidf), Some(matrix)) = (model.tfidf_model(), &tfidf_matrix) {
                    for token in &tokens {
                        let count = token_map.and_then(|m| m.get(token)).copied().unwrap_or(0.0);
                        let probability = ((count + alpha) / denominator.max(alpha * vocab_size)).max(1e-9);
                        let weight = tfidf
                            .token_to_index
                            .get(token)
                            .map(|index| matrix[[0, *index]].max(1e-6))
                            .unwrap_or(1e-6);
                        score += weight * probability.ln();
                    }
                }
            } else {
                for token in &tokens {
                    let count = token_map.and_then(|m| m.get(token)).copied().unwrap_or(0.0);
                    let probability = ((count + alpha) / denominator.max(alpha * vocab_size)).max(1e-9);
                    score += probability.ln();
                }
            }
            (label.clone(), score)
        }).collect::<Vec<_>>()
    });

    softmax_scores(&raw)
}
