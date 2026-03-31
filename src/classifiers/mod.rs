//! Per-classifier entry points.
//!
//! The internal training code remains shared, but each classifier now has its
//! own dedicated module so callers can discover the API in a predictable,
//! non-spaghetti layout.

pub mod bayes;
pub mod gradient_boosting;
pub mod isolation_forest;
pub mod knn;
pub mod logistic_regression;
pub mod random_forest;
pub mod svm;
