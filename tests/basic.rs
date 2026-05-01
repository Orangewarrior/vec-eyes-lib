use std::fs;
use tempfile::tempdir;
use vec_eyes_lib::classifier::{run_rules_pipeline, ClassifierFactory, ClassifierMethod};
use vec_eyes_lib::config::{
    DataConfig, ExtraMatchConfig, ExtraMatchEngine, ModelConfig, PipelineConfig, RecursiveMode,
    RulesFile, ScoreSumMode,
};
use vec_eyes_lib::labels::ClassificationLabel;
use vec_eyes_lib::matcher::MatcherFactory;
use vec_eyes_lib::nlp::NlpOption;
use vec_eyes_lib::report::ClassificationReport;

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
    fs::write(
        &regex_path,
        r"union\s+select
casino
",
    )
    .unwrap();

    let matcher = MatcherFactory::build_from_extra_match(&ExtraMatchConfig {
        recursive_way: RecursiveMode::Off,
        engine: ExtraMatchEngine::Regex,
        path: regex_path,
        score_add_points: 15.0,
        title: Some("attack".to_string()),
        description: Some("demo".to_string()),
    })
    .unwrap();

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
        data: DataConfig {
            hot_test_path: std::path::PathBuf::from("tests/data/hot_attack"),
            cold_test_path: std::path::PathBuf::from("tests/data/cold_regular"),
            hot_label: Some(ClassificationLabel::WebAttack),
            cold_label: Some(ClassificationLabel::RawData),
            recursive_way: RecursiveMode::On,
            score_sum: ScoreSumMode::On,
        },
        pipeline: PipelineConfig {
            nlp: NlpOption::FastText,
            threads: Some(2),
            embedding_dimensions: None,
            security_normalize_obfuscation: None,
        },
        model: ModelConfig::KnnCosine { k: 3 },
        extra_match: vec![],
        csv_output: None,
        json_output: None,
        max_file_bytes: None,
    };

    let report = run_rules_pipeline(&rules_file, &classify_dir).unwrap();
    assert_eq!(report.records.len(), 1);
}

#[test]
fn yaml_validation_rejects_knn_without_k() {
    let rules_file = RulesFile {
        report_name: Some("unit".to_string()),
        data: DataConfig {
            hot_test_path: std::path::PathBuf::from("tests/data/hot_attack"),
            cold_test_path: std::path::PathBuf::from("tests/data/cold_regular"),
            hot_label: Some(ClassificationLabel::WebAttack),
            cold_label: Some(ClassificationLabel::RawData),
            recursive_way: RecursiveMode::On,
            score_sum: ScoreSumMode::On,
        },
        pipeline: PipelineConfig {
            nlp: NlpOption::FastText,
            threads: Some(2),
            embedding_dimensions: None,
            security_normalize_obfuscation: None,
        },
        model: ModelConfig::KnnCosine { k: 0 },
        extra_match: vec![],
        csv_output: None,
        json_output: None,
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
        data: DataConfig {
            hot_test_path: hot,
            cold_test_path: cold,
            hot_label: Some(ClassificationLabel::WebAttack),
            cold_label: Some(ClassificationLabel::RawData),
            recursive_way: RecursiveMode::On,
            score_sum: ScoreSumMode::Off,
        },
        pipeline: PipelineConfig {
            nlp: NlpOption::FastText,
            threads: Some(2),
            embedding_dimensions: None,
            security_normalize_obfuscation: None,
        },
        model: ModelConfig::KnnCosine { k: 3 },
        extra_match: Vec::new(),
        csv_output: None,
        json_output: None,
        max_file_bytes: None,
    };

    rules.validate()?;
    let classifier = ClassifierFactory::builder()
        .from_rules_file(&rules)
        .build()?;
    let result = classifier.classify_text("free bonus casino offer", ScoreSumMode::Off, &[]);
    assert!(!result.labels.is_empty());
    Ok(())
}

#[test]
fn yaml_validation_rejects_excessive_threads() {
    let rules_file = RulesFile {
        report_name: Some("unit".to_string()),
        data: DataConfig {
            hot_test_path: std::path::PathBuf::from("tests/data/hot_attack"),
            cold_test_path: std::path::PathBuf::from("tests/data/cold_regular"),
            hot_label: Some(ClassificationLabel::WebAttack),
            cold_label: Some(ClassificationLabel::RawData),
            recursive_way: RecursiveMode::On,
            score_sum: ScoreSumMode::On,
        },
        pipeline: PipelineConfig {
            nlp: NlpOption::FastText,
            threads: Some(257),
            embedding_dimensions: None,
            security_normalize_obfuscation: None,
        },
        model: ModelConfig::KnnCosine { k: 3 },
        extra_match: vec![],
        csv_output: None,
        json_output: None,
        max_file_bytes: None,
    };

    assert!(rules_file.validate().is_err());
}

#[test]
fn report_rejects_absolute_output_path() {
    let report = ClassificationReport::new("unit".to_string());
    let result = report.write_json("/tmp/vec-eyes-report.json");
    assert!(result.is_err());
}

#[test]
fn word2vec_loader_rejects_huge_header_without_allocating_matrix() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("huge-word2vec.bin");
    fs::write(&path, b"999999999 300\n").unwrap();

    let result = vec_eyes_lib::Word2VecBin::load(&path);
    assert!(result.is_err());
}
