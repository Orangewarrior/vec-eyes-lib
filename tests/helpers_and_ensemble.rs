use vec_eyes_lib::matcher::RuleMatcher;
use vec_eyes_lib::{
    ClassificationLabel, ClassificationResult, Classifier, EnsembleClassifier, ScoreSumMode,
};

struct Dummy(&'static str, f32);
impl Classifier for Dummy {
    fn classify_text(
        &self,
        _text: &str,
        _score_sum_mode: ScoreSumMode,
        _matchers: &[Box<dyn RuleMatcher>],
    ) -> ClassificationResult {
        ClassificationResult {
            labels: vec![(ClassificationLabel::Custom(self.0.to_string()), self.1)],
            extra_hits: vec![],
        }
    }
}

#[test]
fn classification_result_helpers_work() {
    let result = ClassificationResult {
        labels: vec![(ClassificationLabel::Spam, 91.0)],
        extra_hits: vec![],
    };
    assert!(matches!(
        result.top_label(),
        Some(ClassificationLabel::Spam)
    ));
    assert!(result.is_hot(80.0));
    assert_eq!(result.top_score(), 91.0);
}

#[test]
fn ensemble_classifier_returns_weighted_vote() {
    let ensemble = EnsembleClassifier::new(vec![
        (Box::new(Dummy("hot", 90.0)), 2.0),
        (Box::new(Dummy("cold", 40.0)), 1.0),
    ]);
    let result = ensemble.classify_text("probe", ScoreSumMode::Off, &[]);
    assert!(!result.labels.is_empty());
}
