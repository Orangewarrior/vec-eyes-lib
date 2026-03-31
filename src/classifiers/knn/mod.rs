use crate::classifier::{ClassifierBuilder, ClassifierFactory, ClassifierMethod, KnnBuilder};

/// Entry point for KNN-based classifiers.
pub struct KnnModule;

impl KnnModule {
    #[inline]
    pub fn builder() -> KnnBuilder {
        KnnBuilder::new()
    }

    #[inline]
    pub fn cosine() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnCosine)
    }

    #[inline]
    pub fn euclidean() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnEuclidean)
    }

    #[inline]
    pub fn manhattan() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnManhattan)
    }

    #[inline]
    pub fn minkowski() -> ClassifierBuilder {
        ClassifierFactory::builder().method(ClassifierMethod::KnnMinkowski)
    }
}

pub use crate::classifier::DistanceMetric;
