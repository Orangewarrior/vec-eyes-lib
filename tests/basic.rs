use std::fs;
use tempfile::tempdir;
use vec_eyes_lib::classifier::{run_rules_pipeline, ClassifierFactory, ClassifierMethod};
use vec_eyes_lib::config::{ExtraMatchConfig, ExtraMatchEngine, RecursiveMode, RulesFile, ScoreSumMode};
use vec_eyes_lib::labels::ClassificationLabel;
use vec_eyes_lib::matcher::MatcherFactory;
use vec_eyes_lib::nlp::NlpOption;

#[test]
fn default_build_works_with_regex_fallback() {
    let root = std::path::PathBuf::from("tests/data");
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::KnnCosine)
        .nlp(NlpOption::FastText)
        .k(3)
        .hot_label(ClassificationLabel::WebAttack)
        .cold_label(ClassificationLabel::RawData)
        .hot_path(root.join("hot_attack"))
        .cold_path(root.join("cold_regular"))
        .build()
        .unwrap();

    let rules_dir = tempdir().unwrap();
    let regex_path = rules_dir.path().join("rules.txt");
    fs::write(&regex_path, r"union\s+select
casino
").unwrap();

    let matcher = MatcherFactory::build_from_extra_match(&ExtraMatchConfig {
        recursive_way: RecursiveMode::Off,
        engine: ExtraMatchEngine::Regex,
        path: regex_path,
        score_add_points: 15.0,
        title: Some("attack".to_string()),
        description: Some("demo".to_string()),
    }).unwrap();

    let result = classifier.classify_text(
        "GET /?id=1 union select user from accounts",
        ScoreSumMode::On,
        &[matcher],
    );

    assert!(!result.labels.is_empty());
    assert!(!result.extra_hits.is_empty());
}

#[test]
fn yaml_pipeline_runs_and_exports_report() {
    let dir = tempdir().unwrap();
    let classify_dir = dir.path().join("classify");
    fs::create_dir_all(&classify_dir).unwrap();
    fs::write(classify_dir.join("sample.txt"), "union select from users").unwrap();

    let rules_file = RulesFile {
        report_name: Some("unit".to_string()),
        method: vec_eyes_lib::classifier::MethodKind::KnnCosine,
        nlp: NlpOption::FastText,
        threads: Some(2),
        csv_output: None,
        json_output: None,
        recursive_way: RecursiveMode::On,
        hot_test_path: std::path::PathBuf::from("tests/data/hot_attack"),
        cold_test_path: std::path::PathBuf::from("tests/data/cold_regular"),
        hot_label: Some(ClassificationLabel::WebAttack),
        cold_label: Some(ClassificationLabel::RawData),
        score_sum: ScoreSumMode::On,
        extra_match: vec![],
        k: Some(3),
        p: None,
    };

    let report = run_rules_pipeline(&rules_file, &classify_dir).unwrap();
    assert_eq!(report.records.len(), 1);
}

#[test]
fn yaml_validation_rejects_knn_without_k() {
    let rules_file = RulesFile {
        report_name: Some("unit".to_string()),
        method: vec_eyes_lib::classifier::MethodKind::KnnCosine,
        nlp: NlpOption::FastText,
        threads: Some(2),
        csv_output: None,
        json_output: None,
        recursive_way: RecursiveMode::On,
        hot_test_path: std::path::PathBuf::from("tests/data/hot_attack"),
        cold_test_path: std::path::PathBuf::from("tests/data/cold_regular"),
        hot_label: Some(ClassificationLabel::WebAttack),
        cold_label: Some(ClassificationLabel::RawData),
        score_sum: ScoreSumMode::On,
        extra_match: vec![],
        k: None,
        p: None,
    };

    assert!(rules_file.validate().is_err());
}
