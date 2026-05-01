use vec_eyes_lib::{
    Builder, ClassificationLabel, ClassifierMethod, NlpOption, TypedClassifierBuilder,
};

#[test]
#[allow(deprecated)]
fn common_builder_trait_is_available_for_pipeline_and_engine() {
    let pipeline = <vec_eyes_lib::NlpPipelineBuilder as Builder<vec_eyes_lib::NlpPipeline>>::new()
        .representation(vec_eyes_lib::RepresentationKind::Word2Vec)
        .build()
        .unwrap();
    assert!(matches!(
        pipeline.representation,
        vec_eyes_lib::RepresentationKind::Word2Vec
    ));

    let _engine_builder =
        <vec_eyes_lib::EngineBuilder as Builder<vec_eyes_lib::compat::Engine>>::new();
}

#[test]
fn typed_classifier_builder_compiles_for_complete_configuration() {
    let _typed = TypedClassifierBuilder::new()
        .method(ClassifierMethod::Bayes)
        .nlp(NlpOption::Count)
        .training_data(
            "tests/data/biology/hot",
            ClassificationLabel::Virus,
            "tests/data/biology/cold",
            ClassificationLabel::RawData,
        )
        .threads(Some(2));
}
