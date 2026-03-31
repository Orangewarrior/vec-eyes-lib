use crate::advanced_models::IsolationForestConfig;
use crate::classifier::{ClassifierBuilder, ClassifierFactory, ClassifierMethod};

/// Entry point for Isolation Forest classifiers.
pub struct IsolationForestModule;

impl IsolationForestModule {
    #[inline]
    pub fn factory() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::IsolationForest)
    }

    #[inline]
    pub fn with_config(cfg: IsolationForestConfig) -> ClassifierBuilder {
        ClassifierFactory::builder()
            .method(ClassifierMethod::IsolationForest)
            .isolation_forest_config(cfg.n_trees, cfg.contamination, Some(cfg.subsample_size))
    }
}

pub use crate::advanced_models::IsolationForestConfig as Config;
