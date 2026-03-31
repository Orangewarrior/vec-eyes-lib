use std::marker::PhantomData;
use std::path::Path;

use crate::classifier::{ClassifierBuilder, ClassifierMethod};
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::nlp::NlpOption;

pub struct Missing;
pub struct Present;

/// Compile-time safer builder that prevents building without key configuration.
pub struct TypedClassifierBuilder<M, N, D> {
    inner: ClassifierBuilder,
    _marker: PhantomData<(M, N, D)>,
}

impl TypedClassifierBuilder<Missing, Missing, Missing> {
    pub fn new() -> Self {
        Self {
            inner: ClassifierBuilder::new(),
            _marker: PhantomData,
        }
    }
}

impl<M, N, D> TypedClassifierBuilder<M, N, D> {
    fn from_inner<TM, TN, TD>(inner: ClassifierBuilder) -> TypedClassifierBuilder<TM, TN, TD> {
        TypedClassifierBuilder {
            inner,
            _marker: PhantomData,
        }
    }

    pub fn recursive(mut self, recursive: bool) -> Self {
        self.inner = self.inner.recursive(recursive);
        self
    }

    pub fn threads(mut self, threads: Option<usize>) -> Self {
        self.inner = self.inner.threads(threads);
        self
    }

    pub fn k(mut self, k: usize) -> Self {
        self.inner = self.inner.k(k);
        self
    }

    pub fn p(mut self, p: f32) -> Self {
        self.inner = self.inner.p(p);
        self
    }

    pub fn logistic_config(mut self, learning_rate: f32, epochs: usize, lambda: Option<f32>) -> Self {
        self.inner = self.inner.logistic_config(learning_rate, epochs, lambda);
        self
    }

    pub fn random_forest_full_config(
        mut self,
        mode: crate::advanced_models::RandomForestMode,
        n_trees: usize,
        max_depth: Option<usize>,
        max_features: Option<crate::advanced_models::RandomForestMaxFeatures>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        bootstrap: Option<bool>,
        oob_score: Option<bool>,
    ) -> Self {
        self.inner = self.inner.random_forest_full_config(mode, n_trees, max_depth, max_features, min_samples_split, min_samples_leaf, bootstrap, oob_score);
        self
    }

    pub fn isolation_forest_config(mut self, n_trees: usize, contamination: f32, subsample_size: Option<usize>) -> Self {
        self.inner = self.inner.isolation_forest_config(n_trees, contamination, subsample_size);
        self
    }

    pub fn svm_config(mut self, kernel: crate::advanced_models::SvmKernel, c: f32, learning_rate: Option<f32>, epochs: Option<usize>, gamma: Option<f32>, degree: Option<usize>, coef0: Option<f32>) -> Self {
        self.inner = self.inner.svm_config(kernel, c, learning_rate, epochs, gamma, degree, coef0);
        self
    }

    pub fn gradient_boosting_config(mut self, n_estimators: usize, learning_rate: f32, max_depth: Option<usize>) -> Self {
        self.inner = self.inner.gradient_boosting_config(n_estimators, learning_rate, max_depth);
        self
    }
}

impl<N, D> TypedClassifierBuilder<Missing, N, D> {
    pub fn method(self, method: ClassifierMethod) -> TypedClassifierBuilder<Present, N, D> {
        let inner = self.inner.method(method);
        TypedClassifierBuilder::<Present, N, D>::from_inner(inner)
    }
}

impl<M, D> TypedClassifierBuilder<M, Missing, D> {
    pub fn nlp(self, nlp: NlpOption) -> TypedClassifierBuilder<M, Present, D> {
        let inner = self.inner.nlp(nlp);
        TypedClassifierBuilder::<M, Present, D>::from_inner(inner)
    }
}

impl<M, N> TypedClassifierBuilder<M, N, Missing> {
    pub fn training_data<P1: AsRef<Path>, P2: AsRef<Path>>(
        self,
        hot_path: P1,
        hot_label: ClassificationLabel,
        cold_path: P2,
        cold_label: ClassificationLabel,
    ) -> TypedClassifierBuilder<M, N, Present> {
        let inner = self
            .inner
            .hot_path(hot_path)
            .hot_label(hot_label)
            .cold_path(cold_path)
            .cold_label(cold_label);

        TypedClassifierBuilder::<M, N, Present>::from_inner(inner)
    }
}

impl TypedClassifierBuilder<Present, Present, Present> {
    pub fn build(self) -> Result<Box<dyn crate::classifier::Classifier>, VecEyesError> {
        self.inner.build()
    }
}
