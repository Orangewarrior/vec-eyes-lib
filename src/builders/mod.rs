use crate::error::VecEyesError;

/// Common trait for builder-style types in Vec-Eyes.
pub trait Builder<T>: Sized {
    fn new() -> Self;
    fn build(self) -> Result<T, VecEyesError>;
}
