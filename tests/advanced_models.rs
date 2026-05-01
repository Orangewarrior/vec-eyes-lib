use std::path::Path;
use vec_eyes_lib::advanced_models::{AdvancedClassifier, AdvancedMethod};
use vec_eyes_lib::classifier::{run_rules_pipeline, ClassifierFactory, ClassifierMethod};
use vec_eyes_lib::config::RulesFile;
use vec_eyes_lib::config::ScoreSumMode;
use vec_eyes_lib::dataset::load_training_samples;
use vec_eyes_lib::labels::ClassificationLabel;
use vec_eyes_lib::nlp::NlpOption;

fn assert_label_present(
    report: &vec_eyes_lib::report::ClassificationReport,
    expected: ClassificationLabel,
) {
    let prefix = format!("{}:", expected);
    let ranked = &report.records.first().expect("record").classify_names_list;
    assert!(
        ranked.contains(&prefix),
        "expected '{}' in '{}'",
        prefix,
        ranked
    );
}

#[test]
fn yaml_parses_new_methods() {
    for path in [
        "tests/data/rules/sms_logisticregression.yaml",
        "tests/data/rules/sms_randomforest.yaml",
        "tests/data/rules/sms_svm.yaml",
        "tests/data/rules/sms_gradientboosting.yaml",
        "tests/data/rules/fraud_isolation_forest.yaml",
    ] {
        let rules = RulesFile::from_yaml_path(path).unwrap();
        assert!(
            !rules.model.method_kind().is_knn(),
            "{path} should not be treated as knn"
        );
    }
}

#[test]
fn uci_sms_family_methods_classify_spam_like_probe() {
    let yamls = [
        "tests/data/rules/sms_logisticregression.yaml",
        "tests/data/rules/sms_randomforest.yaml",
        "tests/data/rules/sms_svm.yaml",
        "tests/data/rules/sms_gradientboosting.yaml",
    ];
    for yaml in yamls {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report = run_rules_pipeline(&rules, Path::new("tests/data/uci_sms/classify")).unwrap();
        assert_eq!(
            report.records.len(),
            1,
            "expected one output record for {yaml}"
        );
        assert_label_present(&report, ClassificationLabel::Spam);
    }
}

#[test]
fn uci_fraud_advanced_methods_detect_high_risk_text() {
    let yamls = [
        "tests/data/rules/fraud_logisticregression.yaml",
        "tests/data/rules/fraud_randomforest.yaml",
        "tests/data/rules/fraud_svm.yaml",
        "tests/data/rules/fraud_gradientboosting.yaml",
    ];
    for yaml in yamls {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report =
            run_rules_pipeline(&rules, Path::new("tests/data/uci_fraud/classify")).unwrap();
        assert_eq!(
            report.records.len(),
            1,
            "expected one output record for {yaml}"
        );
        assert_label_present(&report, ClassificationLabel::Anomaly);
    }
}

#[test]
fn uci_biology_advanced_methods_detect_virus_text() {
    let yamls = [
        "tests/data/rules/biology_logisticregression.yaml",
        "tests/data/rules/biology_randomforest.yaml",
        "tests/data/rules/biology_svm.yaml",
        "tests/data/rules/biology_gradientboosting.yaml",
    ];
    for yaml in yamls {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report =
            run_rules_pipeline(&rules, Path::new("tests/data/uci_biology/classify")).unwrap();
        assert_eq!(
            report.records.len(),
            1,
            "expected one output record for {yaml}"
        );
        assert_label_present(&report, ClassificationLabel::Virus);
    }
}

#[test]
fn isolation_forest_flags_fraud_probe_as_anomaly() {
    let rules = RulesFile::from_yaml_path("tests/data/rules/fraud_isolation_forest.yaml").unwrap();
    let report = run_rules_pipeline(&rules, Path::new("tests/data/uci_fraud/classify")).unwrap();
    assert_eq!(report.records.len(), 1);
    assert_label_present(&report, ClassificationLabel::Anomaly);
}

#[test]
fn isolation_forest_flags_virus_probe_as_virus() {
    let rules =
        RulesFile::from_yaml_path("tests/data/rules/biology_isolation_forest.yaml").unwrap();
    let report = run_rules_pipeline(&rules, Path::new("tests/data/uci_biology/classify")).unwrap();
    assert_eq!(report.records.len(), 1);
    assert_label_present(&report, ClassificationLabel::Virus);
}

#[test]
fn yaml_validation_rejects_missing_required_advanced_params() {
    assert!(
        RulesFile::from_yaml_path("tests/data/rules/invalid_logistic_missing_params.yaml").is_err()
    );
    assert!(RulesFile::from_yaml_path("tests/data/rules/invalid_svm_missing_kernel.yaml").is_err());
}

#[test]
fn polynomial_svm_yaml_is_accepted_and_classifies_fraud_probe() {
    let rules = RulesFile::from_yaml_path("tests/data/rules/fraud_svm_polynomial.yaml").unwrap();
    let report = run_rules_pipeline(&rules, Path::new("tests/data/uci_fraud/classify")).unwrap();
    assert_eq!(report.records.len(), 1);
    assert_label_present(&report, ClassificationLabel::Anomaly);
}

#[test]
fn builder_api_accepts_advanced_hyperparameters() {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::LogisticRegression)
        .nlp(vec_eyes_lib::nlp::NlpOption::TfIdf)
        .hot_label(ClassificationLabel::Spam)
        .cold_label(ClassificationLabel::RawData)
        .hot_path("tests/data/uci_sms/hot")
        .cold_path("tests/data/uci_sms/cold")
        .logistic_config(0.20, 120, Some(1e-3))
        .build()
        .unwrap();

    let result = classifier.classify_text(
        "urgent free prize claim bonus now",
        vec_eyes_lib::config::ScoreSumMode::Off,
        &[],
    );

    assert!(!result.labels.is_empty());
    assert!(result
        .labels
        .iter()
        .any(|(label, _)| *label == ClassificationLabel::Spam));
}

#[test]
fn random_forest_modes_yaml_are_accepted_and_classify_sms_probe() {
    let yamls = [
        "tests/data/rules/sms_randomforest.yaml",
        "tests/data/rules/sms_randomforest_balanced.yaml",
        "tests/data/rules/sms_randomforest_extratrees.yaml",
    ];
    for yaml in yamls {
        let rules = RulesFile::from_yaml_path(yaml).unwrap();
        let report = run_rules_pipeline(&rules, Path::new("tests/data/uci_sms/classify")).unwrap();
        assert_eq!(
            report.records.len(),
            1,
            "expected one output record for {yaml}"
        );
        assert_label_present(&report, ClassificationLabel::Spam);
    }
}

#[test]
fn random_forest_yaml_rejects_oob_without_bootstrap() {
    assert!(RulesFile::from_yaml_path(
        "tests/data/rules/invalid_randomforest_oob_without_bootstrap.yaml"
    )
    .is_err());
}

#[test]
fn builder_api_accepts_random_forest_modes() {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::RandomForest)
        .nlp(vec_eyes_lib::nlp::NlpOption::TfIdf)
        .hot_label(ClassificationLabel::Spam)
        .cold_label(ClassificationLabel::RawData)
        .hot_path("tests/data/uci_sms/hot")
        .cold_path("tests/data/uci_sms/cold")
        .random_forest_full_config(
            vec_eyes_lib::RandomForestMode::ExtraTrees,
            31,
            Some(8),
            Some(vec_eyes_lib::RandomForestMaxFeatures::Log2),
            Some(2),
            Some(1),
            Some(true),
            Some(true),
        )
        .build()
        .unwrap();

    let result = classifier.classify_text(
        "urgent free prize claim bonus now",
        vec_eyes_lib::config::ScoreSumMode::Off,
        &[],
    );

    assert!(!result.labels.is_empty());
    assert!(result
        .labels
        .iter()
        .any(|(label, _)| *label == ClassificationLabel::Spam));
}

#[test]
fn builder_api_accepts_isolation_forest_hyperparameters() {
    let classifier = ClassifierFactory::builder()
        .method(ClassifierMethod::IsolationForest)
        .nlp(vec_eyes_lib::nlp::NlpOption::FastText)
        .hot_label(ClassificationLabel::Anomaly)
        .cold_label(ClassificationLabel::RawData)
        .hot_path("tests/data/uci_fraud/hot")
        .cold_path("tests/data/uci_fraud/cold")
        .isolation_forest_config(64, 0.05, Some(32))
        .build()
        .unwrap();

    let result = classifier.classify_text(
        "urgent offshore wire transfer hidden beneficiary mule account",
        vec_eyes_lib::config::ScoreSumMode::Off,
        &[],
    );

    assert!(!result.labels.is_empty());
    assert!(result
        .labels
        .iter()
        .any(|(label, _)| *label == ClassificationLabel::Anomaly));
}

#[test]
fn modular_factory_namespaces_are_available() {
    let _ = vec_eyes_lib::factory::ClassifierFactory::builder();
    let _ = vec_eyes_lib::classifiers::bayes::BayesModule::factory();
    let _ = vec_eyes_lib::classifiers::knn::KnnModule::cosine();
    let _ = vec_eyes_lib::classifiers::logistic_regression::LogisticRegressionModule::factory();
    let _ = vec_eyes_lib::classifiers::random_forest::RandomForestModule::factory();
    let _ = vec_eyes_lib::classifiers::isolation_forest::IsolationForestModule::factory();
    let _ = vec_eyes_lib::classifiers::svm::SvmModule::factory();
    let _ = vec_eyes_lib::classifiers::gradient_boosting::GradientBoostingModule::factory();
}

#[test]
fn random_forest_oob_yaml_is_accepted_and_trains_real_oob_score() {
    let rules = RulesFile::from_yaml_path("tests/data/rules/sms_randomforest_oob.yaml").unwrap();
    rules.validate().unwrap();

    let mut samples = load_training_samples(
        Path::new("tests/data/uci_sms/hot"),
        ClassificationLabel::Spam,
        true,
    )
    .unwrap();
    let mut cold = load_training_samples(
        Path::new("tests/data/uci_sms/cold"),
        ClassificationLabel::RawData,
        true,
    )
    .unwrap();
    samples.append(&mut cold);

    let rf_config = match &rules.model {
        vec_eyes_lib::config::ModelConfig::RandomForest {
            mode,
            n_trees,
            max_depth,
            max_features,
            min_samples_split,
            min_samples_leaf,
            bootstrap,
            oob_score,
            random_seed,
        } => vec_eyes_lib::advanced_models::RandomForestConfig {
            mode: mode.clone(),
            n_trees: *n_trees,
            max_depth: *max_depth,
            max_features: max_features.clone(),
            min_samples_split: *min_samples_split,
            min_samples_leaf: *min_samples_leaf,
            bootstrap: *bootstrap,
            oob_score: *oob_score,
            random_seed: *random_seed,
        },
        _ => panic!("expected RandomForest model config"),
    };
    let classifier = AdvancedClassifier::train(
        AdvancedMethod::RandomForest,
        &samples,
        NlpOption::TfIdf,
        ClassificationLabel::Spam,
        ClassificationLabel::RawData,
        &vec_eyes_lib::advanced_models::AdvancedModelConfig {
            random_forest: Some(rf_config),
            ..Default::default()
        },
    )
    .unwrap();

    let oob = classifier.random_forest_oob_score();
    assert!(oob.is_some(), "expected real OOB score to be computed");
    let value = oob.unwrap();
    assert!(
        (0.0..=1.0).contains(&value),
        "expected OOB score in [0,1], got {}",
        value
    );

    let result = classifier.classify_text("urgent free prize claim now", ScoreSumMode::Off, &[]);
    assert!(!result.labels.is_empty());
    assert!(result
        .labels
        .iter()
        .any(|(label, _)| *label == ClassificationLabel::Spam));
}
