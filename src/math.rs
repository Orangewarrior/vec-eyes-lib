use crate::labels::ClassificationLabel;

/// Stable softmax over (label, log-score) pairs.
///
/// Numerically stable via max-subtraction before exponentiation.  Input scores
/// may be raw logits, log-probabilities, or any monotone proxy — only relative
/// order matters for the output distribution.
pub(crate) fn softmax_scores(
    input: &[(ClassificationLabel, f32)],
) -> Vec<(ClassificationLabel, f32)> {
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
    let mut exp_values = Vec::with_capacity(input.len());
    for (label, score) in input {
        let value = (*score - max_score).exp();
        sum += value;
        exp_values.push((label.clone(), value));
    }

    exp_values
        .into_iter()
        .map(|(label, value)| (label, if sum > 0.0 { value / sum } else { 0.0 }))
        .collect()
}
