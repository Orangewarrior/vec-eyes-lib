pub(crate) mod core;
use crate::advanced_models::GradientBoostingConfig;
use crate::classifier::{ClassifierBuilder, ClassifierFactory, ClassifierMethod};

/// Entry point for Gradient Boosting classifiers.
pub struct GradientBoostingModule;

impl GradientBoostingModule {
    #[inline]
    pub fn factory() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::GradientBoosting)
    }

    #[inline]
    pub fn with_config(cfg: GradientBoostingConfig) -> ClassifierBuilder {
        ClassifierFactory::builder()
            .method(ClassifierMethod::GradientBoosting)
            .gradient_boosting_config(cfg.n_estimators, cfg.learning_rate, Some(cfg.max_depth))
    }
}

pub use crate::advanced_models::GradientBoostingConfig as Config;
