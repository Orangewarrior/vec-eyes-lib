pub(crate) mod core;
use crate::advanced_models::RandomForestConfig;
use crate::classifier::{ClassifierBuilder, ClassifierFactory, ClassifierMethod};

/// Entry point for Random Forest classifiers.
pub struct RandomForestModule;

impl RandomForestModule {
    #[inline]
    pub fn factory() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::RandomForest)
    }

    #[inline]
    pub fn with_config(cfg: RandomForestConfig) -> ClassifierBuilder {
        ClassifierFactory::builder()
            .method(ClassifierMethod::RandomForest)
            .random_forest_full_config(
                cfg.mode,
                cfg.n_trees,
                Some(cfg.max_depth),
                Some(cfg.max_features),
                Some(cfg.min_samples_split),
                Some(cfg.min_samples_leaf),
                Some(cfg.bootstrap),
                Some(cfg.oob_score),
            )
    }
}

pub use crate::advanced_models::{
    RandomForestConfig as Config, RandomForestMaxFeatures, RandomForestMode,
};
