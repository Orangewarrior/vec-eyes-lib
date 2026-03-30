use vec_eyes_lib::classifier::run_rules_pipeline;
use vec_eyes_lib::config::RulesFile;
use vec_eyes_lib::labels::ClassificationLabel;

fn assert_top_label(report: &vec_eyes_lib::report::ClassificationReport, expected: ClassificationLabel) {
    let record = report.records.first().expect("expected at least one classification record");
    let expected_prefix = format!("{}:", expected);
    let top = record.classify_names_list.split(',').next().unwrap_or("");
    assert!(
        top.starts_with(&expected_prefix),
        "expected top label prefix '{}', got '{}'",
        expected_prefix,
        record.classify_names_list
    );
}

fn assert_label_present(report: &vec_eyes_lib::report::ClassificationReport, expected: ClassificationLabel) {
    let record = report.records.first().expect("expected at least one classification record");
    let expected_prefix = format!("{}:", expected);
    assert!(
        record.classify_names_list.contains(&expected_prefix),
        "expected label '{}' to appear in ranking, got '{}'",
        expected_prefix,
        record.classify_names_list
    );
}

fn assert_any_rule_matched(report: &vec_eyes_lib::report::ClassificationReport) {
    let record = report.records.first().expect("expected at least one classification record");
    assert!(
        !record.match_titles.trim().is_empty(),
        "expected at least one matched rule title, got '{}'",
        record.match_titles
    );
}

#[test]
fn parses_real_yaml_file_and_validates_knn_requirements() {
    let rules = RulesFile::from_yaml_path("tests/data/rules/web_knn_fasttext_cosine.yaml").unwrap();
    assert!(rules.method.is_knn());
    assert_eq!(rules.k, Some(3));
    assert_eq!(rules.hot_label, Some(ClassificationLabel::WebAttack));
    assert_eq!(rules.cold_label, Some(ClassificationLabel::RawData));
    assert_eq!(rules.extra_match.len(), 1);

    let invalid = RulesFile::from_yaml_path("tests/data/rules/invalid_missing_k.yaml");
    assert!(invalid.is_err(), "KNN YAML without k must be rejected");
}

#[test]
fn web_attack_knn_fasttext_pipeline_uses_real_yaml_and_recursive_training_data() {
    let rules = RulesFile::from_yaml_path("tests/data/rules/web_knn_fasttext_cosine.yaml").unwrap();
    let report = run_rules_pipeline(&rules, std::path::Path::new("tests/data/web/classify")).unwrap();

    assert_eq!(report.records.len(), 1);
    assert_top_label(&report, ClassificationLabel::WebAttack);
    assert_any_rule_matched(&report);
    let record = &report.records[0];
    assert!(
        record.match_titles.to_lowercase().contains("web"),
        "expected a web-related matched rule title, got '{}'",
        record.match_titles
    );
}

#[test]
fn biology_knn_methods_digest_yaml_files_and_classify_virus_text() {
    let yaml_files = [
        "tests/data/rules/biology_knn_fasttext_cosine.yaml",
        "tests/data/rules/biology_knn_word2vec_euclidean.yaml",
        "tests/data/rules/biology_knn_fasttext_manhattan.yaml",
        "tests/data/rules/biology_knn_fasttext_minkowski.yaml",
    ];

    for yaml in yaml_files {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report = run_rules_pipeline(&rules, std::path::Path::new("tests/data/biology/classify")).unwrap();
        assert_eq!(report.records.len(), 1, "expected one record for {yaml}");
        assert_top_label(&report, ClassificationLabel::Virus);
    }
}

#[test]
fn biology_bayes_pipelines_digest_yaml_and_classify_virus_text() {
    let yaml_files = [
        "tests/data/rules/biology_bayes_count.yaml",
        "tests/data/rules/biology_bayes_tfidf.yaml",
    ];

    for yaml in yaml_files {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report = run_rules_pipeline(&rules, std::path::Path::new("tests/data/biology/classify")).unwrap();
        assert_eq!(report.records.len(), 1, "expected one record for {yaml}");
        assert_top_label(&report, ClassificationLabel::Virus);
    }
}

#[test]
fn financial_fraud_knn_methods_digest_yaml_and_classify_high_risk_transaction_text() {
    let yaml_files = [
        "tests/data/rules/fraud_knn_cosine.yaml",
        "tests/data/rules/fraud_knn_euclidean.yaml",
        "tests/data/rules/fraud_knn_manhattan.yaml",
        "tests/data/rules/fraud_knn_minkowski.yaml",
    ];

    for yaml in yaml_files {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report = run_rules_pipeline(&rules, std::path::Path::new("tests/data/fraud/classify")).unwrap();
        assert_eq!(report.records.len(), 1, "expected one record for {yaml}");
        // KNN ranking may vary slightly by distance metric, but the anomaly label must be
        // present in the final ranking for a clearly high-risk transaction sample.
        assert_label_present(&report, ClassificationLabel::Anomaly);
    }
}


#[test]
fn financial_fraud_bayes_pipelines_digest_yaml_and_classify_high_risk_transaction_text() {
    let yaml_files = [
        "tests/data/rules/fraud_bayes_count.yaml",
        "tests/data/rules/fraud_bayes_tfidf.yaml",
    ];

    for yaml in yaml_files {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report = run_rules_pipeline(&rules, std::path::Path::new("tests/data/fraud/classify")).unwrap();
        assert_eq!(report.records.len(), 1, "expected one record for {yaml}");
        assert_label_present(&report, ClassificationLabel::Anomaly);
    }
}
