use crate::advanced_models::LogisticRegressionConfig;
use crate::classifier::{ClassifierBuilder, ClassifierFactory, ClassifierMethod};

/// Entry point for logistic-regression classifiers.
pub struct LogisticRegressionModule;

impl LogisticRegressionModule {
    #[inline]
    pub fn factory() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::LogisticRegression)
    }

    #[inline]
    pub fn with_config(cfg: LogisticRegressionConfig) -> ClassifierBuilder {
        ClassifierFactory::builder()
            .method(ClassifierMethod::LogisticRegression)
            .logistic_config(cfg.learning_rate, cfg.epochs, Some(cfg.lambda))
    }
}

pub use crate::advanced_models::LogisticRegressionConfig as Config;
