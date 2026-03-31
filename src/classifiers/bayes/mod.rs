use crate::classifier::{BayesBuilder, ClassifierBuilder, ClassifierFactory, ClassifierMethod};

/// Entry point for the Naive Bayes classifier family.
pub struct BayesModule;

impl BayesModule {
    #[inline]
    pub fn builder() -> BayesBuilder {
        BayesBuilder::new()
    }

    #[inline]
    pub fn factory() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::Bayes)
    }
}
