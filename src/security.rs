use crate::error::VecEyesError;
use std::path::{Component, Path, PathBuf};

/// Reject output paths containing `..` components (path-traversal guard).
///
/// Does **not** require the path to exist, making it suitable for output files
/// that will be created by the caller.  Absolute paths are allowed; only the
/// presence of `ParentDir` (`..`) components is checked.
pub fn sanitize_output_path(path: &Path) -> Result<(), VecEyesError> {
    if path.components().any(|c| c == Component::ParentDir) {
        return Err(VecEyesError::invalid_config(
            "security::sanitize_output_path",
            format!(
                "output path '{}' contains '..' which may escape the intended directory",
                path.display()
            ),
        ));
    }
    Ok(())
}

pub fn sanitize_existing_path(input: &Path) -> Result<PathBuf, VecEyesError> {
    input.canonicalize().map_err(|e| {
        VecEyesError::invalid_config(
            "security::sanitize_existing_path",
            format!("path resolution failed for {}: {}", input.display(), e),
        )
    })
}

pub fn sanitize_existing_path_with_base(
    input: &Path,
    allowed_base: &Path,
) -> Result<PathBuf, VecEyesError> {
    let canonical = input.canonicalize().map_err(|e| {
        VecEyesError::invalid_config(
            "security::sanitize_existing_path_with_base",
            format!("path resolution failed for {}: {}", input.display(), e),
        )
    })?;
    let allowed_base = allowed_base.canonicalize().map_err(|e| {
        VecEyesError::invalid_config(
            "security::sanitize_existing_path_with_base",
            format!(
                "allowed base resolution failed for {}: {}",
                allowed_base.display(),
                e
            ),
        )
    })?;

    if !canonical.starts_with(&allowed_base) {
        return Err(VecEyesError::invalid_config(
            "security::sanitize_existing_path_with_base",
            format!(
                "path {} escapes the allowed base directory {}",
                canonical.display(),
                allowed_base.display()
            ),
        ));
    }

    Ok(canonical)
}
