#![doc = r#"
# Vec-Eyes library

Vec-Eyes is a behavior classification library focused on raw text, mail-like
buffers, HTTP traces, syscall traces, malware notes, and rule-driven scoring.

## Default build

The default build uses a pure Rust regex matcher and **does not require Boost**.

## Optional Vectorscan backend

Enable the native matcher only when you explicitly want it:

```bash
cargo test --features vectorscan
```

When the `vectorscan` feature is enabled you will need a C/C++ toolchain plus
Boost and CMake available on the host system.

## YAML-driven pipeline

The library can load a YAML pipeline file that defines:
- hot/cold training paths,
- recursive behavior,
- NLP mode,
- classifier method,
- output files,
- extra regex/Vectorscan matchers,
- score summing behavior.

"#]

pub mod advanced_models;
pub mod builders;
pub mod classifier;
pub mod classifiers;
pub mod compat;
pub mod config;
pub mod dataset;
pub mod error;
pub mod factory;
pub mod labels;
pub mod matcher;
pub(crate) mod math;
pub mod metrics;
pub mod nlp;
pub(crate) mod parallel;
pub mod report;
pub mod security;

pub use advanced_models::{
    AdvancedClassifier,
    AdvancedModelConfig,
    // Standalone typed classifiers — same API as KnnClassifier / BayesClassifier
    GradientBoostingClassifier,
    GradientBoostingConfig,
    IsolationForestClassifier,
    IsolationForestConfig,
    LogisticClassifier,
    LogisticRegressionConfig,
    RandomForestClassifier,
    RandomForestConfig,
    RandomForestMaxFeatures,
    RandomForestMode,
    SvmClassifier,
    SvmConfig,
    SvmKernel,
};
pub use classifier::{
    BayesBuilder, BayesClassifier, ClassificationResult, Classifier, ClassifierBuilder,
    ClassifierFactory, ClassifierMethod, DistanceMetric, EnsembleClassifier, EnsembleStrategy,
    ExplainableClassifier, KnnBuilder, KnnClassifier, MethodKind, TokenContribution,
};
pub use config::{
    DataConfig, ExtraMatchConfig, ExtraMatchEngine, ModelConfig, PipelineConfig, RecursiveMode,
    RulesFile, ScoreSumMode,
};
pub use dataset::{
    collect_files_recursively, load_training_samples, read_text_file, training_sample_iter,
};
pub use error::VecEyesError;
pub use labels::ClassificationLabel;
pub use matcher::{
    AlertHit, JsonRule, MatchRule, MatcherBackend, MatcherFactory, RuleMatcher, RuleSet,
    ScoringEngine,
};
pub use nlp::external_embeddings::ExternalEmbeddings;
pub use nlp::fasttext_bin::{FastTextBin, FastTextEmbeddings};
pub use nlp::word2vec_bin::{Word2VecBin, Word2VecEmbeddings};
pub use nlp::{FastTextConfig, FastTextConfigBuilder, NlpOption};
pub use report::{ClassificationRecord, ClassificationReport};

pub use dataset::read_text_file_limited;

#[allow(deprecated)]
pub use compat::{
    alerts, EngineBuilder, NlpPipeline, NlpPipelineBuilder, OutputWriters, RepresentationKind,
};
pub use factory::{AdvancedSpec, BayesSpec, ClassifierSpec, KnnSpec, TypedClassifierBuilder};

pub use builders::Builder;
