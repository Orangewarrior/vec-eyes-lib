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
        embedding_dimensions: None,
        security_normalize_obfuscation: None,
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
        logistic_learning_rate: None,
        logistic_epochs: None,
        logistic_lambda: None,
        random_forest_n_trees: None,
        random_forest_mode: None,
        random_forest_max_depth: None,
        random_forest_max_features: None,
        random_forest_min_samples_split: None,
        random_forest_min_samples_leaf: None,
        random_forest_bootstrap: None,
        random_forest_oob_score: None,
        random_forest_seed: None,
        svm_kernel: None,
        svm_c: None,
        svm_learning_rate: None,
        svm_epochs: None,
        svm_gamma: None,
        svm_degree: None,
        svm_coef0: None,
        gradient_boosting_n_estimators: None,
        gradient_boosting_learning_rate: None,
        gradient_boosting_max_depth: None,
        isolation_forest_n_trees: None,
        isolation_forest_contamination: None,
        isolation_forest_subsample_size: None,
        max_file_bytes: None,
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
        embedding_dimensions: None,
        security_normalize_obfuscation: None,
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
        logistic_learning_rate: None,
        logistic_epochs: None,
        logistic_lambda: None,
        random_forest_n_trees: None,
        random_forest_mode: None,
        random_forest_max_depth: None,
        random_forest_max_features: None,
        random_forest_min_samples_split: None,
        random_forest_min_samples_leaf: None,
        random_forest_bootstrap: None,
        random_forest_oob_score: None,
        random_forest_seed: None,
        svm_kernel: None,
        svm_c: None,
        svm_learning_rate: None,
        svm_epochs: None,
        svm_gamma: None,
        svm_degree: None,
        svm_coef0: None,
        gradient_boosting_n_estimators: None,
        gradient_boosting_learning_rate: None,
        gradient_boosting_max_depth: None,
        isolation_forest_n_trees: None,
        isolation_forest_contamination: None,
        isolation_forest_subsample_size: None,
        max_file_bytes: None,
    };

    assert!(rules_file.validate().is_err());
}


#[test]
fn yaml_threads_field_is_validated_and_builds() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let hot = root.join("tests/data/hot_attack");
    let cold = root.join("tests/data/cold_regular");

    let rules = RulesFile {
        report_name: Some("Threads test".into()),
        method: vec_eyes_lib::classifier::MethodKind::KnnCosine,
        nlp: NlpOption::FastText,
        threads: Some(2),
        embedding_dimensions: None,
        security_normalize_obfuscation: None,
        csv_output: None,
        json_output: None,
        recursive_way: RecursiveMode::On,
        hot_test_path: hot,
        cold_test_path: cold,
        hot_label: Some(ClassificationLabel::WebAttack),
        cold_label: Some(ClassificationLabel::RawData),
        score_sum: ScoreSumMode::Off,
        extra_match: Vec::new(),
        k: Some(3),
        p: None,
        logistic_learning_rate: None,
        logistic_epochs: None,
        logistic_lambda: None,
        random_forest_n_trees: None,
        random_forest_mode: None,
        random_forest_max_depth: None,
        random_forest_max_features: None,
        random_forest_min_samples_split: None,
        random_forest_min_samples_leaf: None,
        random_forest_bootstrap: None,
        random_forest_oob_score: None,
        random_forest_seed: None,
        svm_kernel: None,
        svm_c: None,
        svm_learning_rate: None,
        svm_epochs: None,
        svm_gamma: None,
        svm_degree: None,
        svm_coef0: None,
        gradient_boosting_n_estimators: None,
        gradient_boosting_learning_rate: None,
        gradient_boosting_max_depth: None,
        isolation_forest_n_trees: None,
        isolation_forest_contamination: None,
        isolation_forest_subsample_size: None,
        max_file_bytes: None,
    };

    rules.validate()?;
    let classifier = ClassifierFactory::builder().from_rules_file(&rules).build()?;
    let result = classifier.classify_text("free bonus casino offer", ScoreSumMode::Off, &[]);
    assert!(!result.labels.is_empty());
    Ok(())
}
