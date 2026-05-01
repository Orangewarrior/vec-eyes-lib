/// Save / load round-trip tests for every classifier in vec-eyes-lib.
///
/// Each test:
///   1. Trains a classifier on a small UCI-derived dataset.
///   2. Classifies a representative probe text → records the top label.
///   3. Persists the model (JSON, bincode, or split bincode) to a temp dir.
///   4. Loads the model back from disk.
///   5. Classifies the same probe again and asserts the top label is identical.
///   6. Checks that the saved file(s) exist and are non-empty.
///
/// Datasets used (in tests/data/):
///   uci_sms/     – spam-vs-ham texts  (label: Spam / RawData)
///   uci_fraud/   – fraud-vs-normal    (label: Anomaly / RawData)
///   uci_biology/ – virus-vs-human     (label: Virus / RawData)
use std::path::PathBuf;
use tempfile::tempdir;

use vec_eyes_lib::dataset::{load_training_samples, TrainingSample};
use vec_eyes_lib::{
    AdvancedClassifier, AdvancedModelConfig, BayesClassifier, ClassificationLabel, Classifier,
    DistanceMetric, GradientBoostingClassifier, GradientBoostingConfig, IsolationForestClassifier,
    IsolationForestConfig, KnnClassifier, LogisticClassifier, LogisticRegressionConfig, NlpOption,
    RandomForestClassifier, RandomForestConfig, ScoreSumMode, SvmClassifier, SvmConfig,
};

// ── Shared fixtures ───────────────────────────────────────────────────────────

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn sms_samples() -> Vec<TrainingSample> {
    let base = root().join("uci_sms");
    let mut s = load_training_samples(&base.join("hot"), ClassificationLabel::Spam, false).unwrap();
    s.extend(
        load_training_samples(&base.join("cold"), ClassificationLabel::RawData, false).unwrap(),
    );
    s
}

fn fraud_samples() -> Vec<TrainingSample> {
    let base = root().join("uci_fraud");
    let mut s =
        load_training_samples(&base.join("hot"), ClassificationLabel::Anomaly, false).unwrap();
    s.extend(
        load_training_samples(&base.join("cold"), ClassificationLabel::RawData, false).unwrap(),
    );
    s
}

fn biology_samples() -> Vec<TrainingSample> {
    let base = root().join("uci_biology");
    let mut s =
        load_training_samples(&base.join("hot"), ClassificationLabel::Virus, false).unwrap();
    s.extend(
        load_training_samples(&base.join("cold"), ClassificationLabel::RawData, false).unwrap(),
    );
    s
}

const SMS_PROBE: &str =
    "Free bonus voucher claim your cash reward now limited offer reply immediately";
const FRAUD_PROBE: &str =
    "urgent offshore wire transfer to new beneficiary with crypto conversion and mule account mismatch";
const BIO_PROBE: &str =
    "viral rna mutation in spike protein indicates infection related genome marker pattern";

fn top_label(clf: &dyn Classifier, text: &str) -> ClassificationLabel {
    clf.classify_text(text, ScoreSumMode::Off, &[])
        .labels
        .into_iter()
        .next()
        .expect("classifier must return at least one label")
        .0
}

fn assert_file_nonempty(path: &std::path::Path) {
    assert!(path.exists(), "file not found: {}", path.display());
    assert!(
        std::fs::metadata(path).unwrap().len() > 0,
        "file is empty: {}",
        path.display()
    );
}

// ── KnnClassifier ─────────────────────────────────────────────────────────────

#[test]
fn knn_json_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = KnnClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        DistanceMetric::Cosine,
        24,
        3,
        None,
        false,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("knn.json");
    clf.save(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = KnnClassifier::load(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "KNN JSON round-trip"
    );
}

#[test]
fn knn_bincode_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = KnnClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        DistanceMetric::Cosine,
        24,
        3,
        None,
        false,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("knn.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = KnnClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "KNN bincode round-trip"
    );
}

#[test]
fn knn_split_bincode_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = KnnClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        DistanceMetric::Cosine,
        24,
        3,
        None,
        false,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let nlp = dir.path().join("knn.nlp.bin");
    let ml = dir.path().join("knn.ml.bin");
    clf.save_split_bincode(&nlp, &ml).unwrap();
    assert_file_nonempty(&nlp);
    assert_file_nonempty(&ml);

    let loaded = KnnClassifier::load_split_bincode(&nlp, &ml).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "KNN split-bincode round-trip"
    );
}

#[test]
fn knn_euclidean_bincode_round_trip() {
    let samples = biology_samples();
    let clf = KnnClassifier::train(
        &samples,
        NlpOption::FastText,
        DistanceMetric::Euclidean,
        24,
        3,
        None,
        false,
    )
    .unwrap();
    let expected = top_label(&clf, BIO_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("knn_euc.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = KnnClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, BIO_PROBE),
        "KNN Euclidean bincode round-trip"
    );
}

// ── BayesClassifier ───────────────────────────────────────────────────────────

#[test]
fn bayes_json_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = BayesClassifier::train(&samples, NlpOption::TfIdf, None).unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("bayes.json");
    clf.save(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = BayesClassifier::load(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "Bayes JSON round-trip"
    );
}

#[test]
fn bayes_bincode_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = BayesClassifier::train(&samples, NlpOption::TfIdf, None).unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("bayes.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = BayesClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "Bayes bincode round-trip"
    );
}

#[test]
fn bayes_both_formats_produce_nonempty_files() {
    let samples = sms_samples();
    let clf = BayesClassifier::train(&samples, NlpOption::TfIdf, None).unwrap();

    let dir = tempdir().unwrap();
    let json_path = dir.path().join("bayes.json");
    let bin_path = dir.path().join("bayes.bin");
    clf.save(&json_path).unwrap();
    clf.save_bincode(&bin_path).unwrap();

    assert_file_nonempty(&json_path);
    assert_file_nonempty(&bin_path);

    // Both formats must round-trip to the same classification result
    let expected = top_label(&clf, SMS_PROBE);
    assert_eq!(
        expected,
        top_label(&BayesClassifier::load(&json_path).unwrap(), SMS_PROBE)
    );
    assert_eq!(
        expected,
        top_label(
            &BayesClassifier::load_bincode(&bin_path).unwrap(),
            SMS_PROBE
        )
    );
}

// ── LogisticClassifier ────────────────────────────────────────────────────────

#[test]
fn logistic_json_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = LogisticClassifier::train(
        &samples,
        NlpOption::TfIdf,
        LogisticRegressionConfig::default(),
        None,
        32,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("logistic.json");
    clf.save(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = LogisticClassifier::load(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "Logistic JSON round-trip"
    );
}

#[test]
fn logistic_bincode_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = LogisticClassifier::train(
        &samples,
        NlpOption::TfIdf,
        LogisticRegressionConfig::default(),
        None,
        32,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("logistic.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = LogisticClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "Logistic bincode round-trip"
    );
}

#[test]
fn logistic_split_bincode_round_trip_preserves_classification() {
    let samples = fraud_samples();
    let clf = LogisticClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        LogisticRegressionConfig::default(),
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, FRAUD_PROBE);

    let dir = tempdir().unwrap();
    let nlp = dir.path().join("logistic.nlp.bin");
    let ml = dir.path().join("logistic.ml.bin");
    clf.save_split_bincode(&nlp, &ml).unwrap();
    assert_file_nonempty(&nlp);
    assert_file_nonempty(&ml);

    let loaded = LogisticClassifier::load_split_bincode(&nlp, &ml).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, FRAUD_PROBE),
        "Logistic split-bincode round-trip"
    );
}

// ── SvmClassifier ─────────────────────────────────────────────────────────────

#[test]
fn svm_json_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf =
        SvmClassifier::train(&samples, NlpOption::TfIdf, SvmConfig::default(), None, 32).unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("svm.json");
    clf.save(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = SvmClassifier::load(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "SVM JSON round-trip"
    );
}

#[test]
fn svm_bincode_round_trip_preserves_classification() {
    let samples = biology_samples();
    let clf = SvmClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        SvmConfig::default(),
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, BIO_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("svm.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = SvmClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, BIO_PROBE),
        "SVM bincode round-trip"
    );
}

#[test]
fn svm_split_bincode_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf =
        SvmClassifier::train(&samples, NlpOption::TfIdf, SvmConfig::default(), None, 32).unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let nlp = dir.path().join("svm.nlp.bin");
    let ml = dir.path().join("svm.ml.bin");
    clf.save_split_bincode(&nlp, &ml).unwrap();
    assert_file_nonempty(&nlp);
    assert_file_nonempty(&ml);

    let loaded = SvmClassifier::load_split_bincode(&nlp, &ml).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "SVM split-bincode round-trip"
    );
}

// ── RandomForestClassifier ────────────────────────────────────────────────────

#[test]
fn random_forest_json_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = RandomForestClassifier::train(
        &samples,
        NlpOption::TfIdf,
        RandomForestConfig::default(),
        None,
        32,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("rf.json");
    clf.save(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = RandomForestClassifier::load(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "RandomForest JSON round-trip"
    );
}

#[test]
fn random_forest_bincode_round_trip_preserves_classification() {
    let samples = fraud_samples();
    let clf = RandomForestClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        RandomForestConfig::default(),
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, FRAUD_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("rf.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = RandomForestClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, FRAUD_PROBE),
        "RandomForest bincode round-trip"
    );
}

#[test]
fn random_forest_split_bincode_round_trip_preserves_classification() {
    let samples = biology_samples();
    let clf = RandomForestClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        RandomForestConfig::default(),
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, BIO_PROBE);

    let dir = tempdir().unwrap();
    let nlp = dir.path().join("rf.nlp.bin");
    let ml = dir.path().join("rf.ml.bin");
    clf.save_split_bincode(&nlp, &ml).unwrap();
    assert_file_nonempty(&nlp);
    assert_file_nonempty(&ml);

    let loaded = RandomForestClassifier::load_split_bincode(&nlp, &ml).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, BIO_PROBE),
        "RandomForest split-bincode round-trip"
    );
}

// ── GradientBoostingClassifier ────────────────────────────────────────────────

#[test]
fn gradient_boosting_json_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = GradientBoostingClassifier::train(
        &samples,
        NlpOption::TfIdf,
        GradientBoostingConfig::default(),
        None,
        32,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("gb.json");
    clf.save(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = GradientBoostingClassifier::load(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "GradientBoosting JSON round-trip"
    );
}

#[test]
fn gradient_boosting_bincode_round_trip_preserves_classification() {
    let samples = fraud_samples();
    let clf = GradientBoostingClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        GradientBoostingConfig::default(),
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, FRAUD_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("gb.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = GradientBoostingClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, FRAUD_PROBE),
        "GradientBoosting bincode round-trip"
    );
}

#[test]
fn gradient_boosting_split_bincode_round_trip_preserves_classification() {
    let samples = sms_samples();
    let clf = GradientBoostingClassifier::train(
        &samples,
        NlpOption::TfIdf,
        GradientBoostingConfig::default(),
        None,
        32,
    )
    .unwrap();
    let expected = top_label(&clf, SMS_PROBE);

    let dir = tempdir().unwrap();
    let nlp = dir.path().join("gb.nlp.bin");
    let ml = dir.path().join("gb.ml.bin");
    clf.save_split_bincode(&nlp, &ml).unwrap();
    assert_file_nonempty(&nlp);
    assert_file_nonempty(&ml);

    let loaded = GradientBoostingClassifier::load_split_bincode(&nlp, &ml).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, SMS_PROBE),
        "GradientBoosting split-bincode round-trip"
    );
}

// ── IsolationForestClassifier ─────────────────────────────────────────────────

#[test]
fn isolation_forest_json_round_trip_preserves_classification() {
    let samples = fraud_samples();
    let clf = IsolationForestClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        IsolationForestConfig::default(),
        ClassificationLabel::Anomaly,
        ClassificationLabel::RawData,
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, FRAUD_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("iforest.json");
    clf.save(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = IsolationForestClassifier::load(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, FRAUD_PROBE),
        "IsolationForest JSON round-trip"
    );
}

#[test]
fn isolation_forest_bincode_round_trip_preserves_classification() {
    let samples = fraud_samples();
    let clf = IsolationForestClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        IsolationForestConfig::default(),
        ClassificationLabel::Anomaly,
        ClassificationLabel::RawData,
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, FRAUD_PROBE);

    let dir = tempdir().unwrap();
    let path = dir.path().join("iforest.bin");
    clf.save_bincode(&path).unwrap();
    assert_file_nonempty(&path);

    let loaded = IsolationForestClassifier::load_bincode(&path).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, FRAUD_PROBE),
        "IsolationForest bincode round-trip"
    );
}

#[test]
fn isolation_forest_split_bincode_round_trip_preserves_classification() {
    let samples = fraud_samples();
    let clf = IsolationForestClassifier::train(
        &samples,
        NlpOption::FastText,
        IsolationForestConfig::default(),
        ClassificationLabel::Anomaly,
        ClassificationLabel::RawData,
        None,
        24,
    )
    .unwrap();
    let expected = top_label(&clf, FRAUD_PROBE);

    let dir = tempdir().unwrap();
    let nlp = dir.path().join("iforest.nlp.bin");
    let ml = dir.path().join("iforest.ml.bin");
    clf.save_split_bincode(&nlp, &ml).unwrap();
    assert_file_nonempty(&nlp);
    assert_file_nonempty(&ml);

    let loaded = IsolationForestClassifier::load_split_bincode(&nlp, &ml).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, FRAUD_PROBE),
        "IsolationForest split-bincode round-trip"
    );
}

// ── AdvancedClassifier (generic wrapper) ──────────────────────────────────────

#[test]
fn advanced_classifier_split_bincode_reloads_correctly() {
    let samples = biology_samples();
    let config = AdvancedModelConfig {
        logistic: Some(LogisticRegressionConfig::default()),
        embedding_dimensions: Some(24),
        ..Default::default()
    };
    let clf = AdvancedClassifier::train(
        vec_eyes_lib::advanced_models::AdvancedMethod::LogisticRegression,
        &samples,
        NlpOption::Word2Vec,
        ClassificationLabel::Virus,
        ClassificationLabel::RawData,
        &config,
    )
    .unwrap();
    let expected = top_label(&clf, BIO_PROBE);

    let dir = tempdir().unwrap();
    let nlp = dir.path().join("adv.nlp.bin");
    let ml = dir.path().join("adv.ml.bin");
    clf.save_split_bincode(&nlp, &ml).unwrap();
    assert_file_nonempty(&nlp);
    assert_file_nonempty(&ml);

    let loaded = AdvancedClassifier::load_split_bincode(&nlp, &ml).unwrap();
    assert_eq!(
        expected,
        top_label(&loaded, BIO_PROBE),
        "AdvancedClassifier split-bincode round-trip"
    );
}

// ── Cross-format size comparison ──────────────────────────────────────────────

#[test]
fn bincode_files_are_consistently_smaller_than_json() {
    let samples = fraud_samples();
    let clf = RandomForestClassifier::train(
        &samples,
        NlpOption::Word2Vec,
        RandomForestConfig::default(),
        None,
        24,
    )
    .unwrap();

    let dir = tempdir().unwrap();
    let json_path = dir.path().join("rf.json");
    let bin_path = dir.path().join("rf.bin");
    clf.save(&json_path).unwrap();
    clf.save_bincode(&bin_path).unwrap();

    let json_size = std::fs::metadata(&json_path).unwrap().len();
    let bin_size = std::fs::metadata(&bin_path).unwrap().len();
    assert!(
        bin_size < json_size,
        "bincode ({bin_size}B) should be smaller than JSON ({json_size}B)"
    );
}

// ── Error cases ───────────────────────────────────────────────────────────────

#[test]
fn load_nonexistent_file_returns_error() {
    let result = KnnClassifier::load_bincode("/tmp/this_file_does_not_exist_vec_eyes.bin");
    assert!(result.is_err(), "loading a missing file should return Err");
}

#[test]
fn load_corrupted_bincode_returns_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt.bin");
    std::fs::write(&path, b"this is not valid bincode data at all").unwrap();
    let result = KnnClassifier::load_bincode(&path);
    assert!(result.is_err(), "loading corrupt bincode should return Err");
}

#[test]
fn load_corrupted_json_returns_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupt.json");
    std::fs::write(&path, b"{not: valid json}").unwrap();
    let result = BayesClassifier::load(&path);
    assert!(result.is_err(), "loading corrupt JSON should return Err");
}
