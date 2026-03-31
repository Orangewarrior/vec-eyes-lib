
use crate::error::VecEyesError;
use std::path::{Path, PathBuf};

pub fn sanitize_existing_path(input: &Path) -> Result<PathBuf, VecEyesError> {
    input.canonicalize()
        .map_err(|e| VecEyesError::invalid_config("security::sanitize_existing_path", format!("path resolution failed for {}: {}", input.display(), e)))
}
