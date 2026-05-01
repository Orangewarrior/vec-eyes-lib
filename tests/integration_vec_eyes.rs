#![allow(deprecated)]

use std::path::PathBuf;

use vec_eyes_lib::{
    alerts::AlertMatcher, ClassificationLabel, DistanceMetric, EngineBuilder,
    FastTextConfigBuilder, KnnBuilder, NlpPipelineBuilder, OutputWriters, RepresentationKind,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

#[test]
fn fasttext_knn_and_alert_rule_push_block_list() {
    let pipeline = NlpPipelineBuilder::new()
        .representation(RepresentationKind::FastText)
        .fasttext_config(FastTextConfigBuilder::new().dimensions(48).build().unwrap())
        .build()
        .unwrap();

    let model = KnnBuilder::new()
        .pipeline(pipeline)
        .k(3)
        .metric(DistanceMetric::Cosine)
        .fit_from_directories(
            Some(root().join("hot")),
            Some(root().join("cold")),
            ClassificationLabel::WebAttack,
        )
        .unwrap();

    let alerts = AlertMatcher::load_json_file(root().join("rules/block_rules.json")).unwrap();
    let engine = EngineBuilder::new()
        .model(model.into())
        .alerts(alerts)
        .output(OutputWriters::disabled())
        .build()
        .unwrap();

    let report = engine
        .classify_text(
            "GET /download?file=payload from 10.10.10.10 and http://bad.example/m.exe",
            "imap-buffer",
        )
        .unwrap();

    assert!(report
        .classifications
        .iter()
        .any(|(label, _)| *label == ClassificationLabel::BlockList));
}
