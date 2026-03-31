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
pub mod compat;
pub mod classifiers;
pub mod factory;
pub mod config;
pub mod dataset;
pub mod error;
pub mod labels;
pub mod matcher;
pub mod nlp;
pub mod report;
pub mod security;
pub(crate) mod parallel;

pub use classifier::{
    BayesBuilder, Classifier, ClassifierBuilder, ClassifierFactory, ClassifierMethod,
    DistanceMetric, KnnBuilder, MethodKind,
};
pub use advanced_models::{
    AdvancedClassifier, AdvancedModelConfig, GradientBoostingConfig, IsolationForestConfig,
    LogisticRegressionConfig, RandomForestConfig, RandomForestMaxFeatures, RandomForestMode,
    SvmConfig, SvmKernel,
};
pub use config::{
    ExtraMatchConfig, ExtraMatchEngine, RecursiveMode, RulesFile, ScoreSumMode,
};
pub use dataset::{collect_files_recursively, read_text_file, load_training_samples};
pub use error::VecEyesError;
pub use labels::ClassificationLabel;
pub use matcher::{
    AlertHit, JsonRule, MatchRule, MatcherBackend, MatcherFactory, RuleMatcher,
    RuleSet, ScoringEngine,
};
pub use nlp::{FastTextConfig, FastTextConfigBuilder, NlpOption};
pub use report::{ClassificationReport, ClassificationRecord};

pub use compat::{alerts, EngineBuilder, NlpPipeline, NlpPipelineBuilder, OutputWriters, RepresentationKind};
pub use factory::TypedClassifierBuilder;

pub use builders::Builder;
