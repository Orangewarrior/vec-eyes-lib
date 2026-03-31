use crate::advanced_models::SvmConfig;
use crate::classifier::{ClassifierBuilder, ClassifierFactory, ClassifierMethod};

/// Entry point for SVM classifiers.
pub struct SvmModule;

impl SvmModule {
    #[inline]
    pub fn factory() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::Svm)
    }

    #[inline]
    pub fn with_config(cfg: SvmConfig) -> ClassifierBuilder {
        ClassifierFactory::builder()
            .method(ClassifierMethod::Svm)
            .svm_config(
                cfg.kernel,
                cfg.c,
                Some(cfg.learning_rate),
                Some(cfg.epochs),
                Some(cfg.gamma),
                Some(cfg.degree),
                Some(cfg.coef0),
            )
    }
}

pub use crate::advanced_models::{SvmConfig as Config, SvmKernel};
