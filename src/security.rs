use crate::error::VecEyesError;
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const DEFAULT_MAX_MODEL_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_CONFIG_THREADS: usize = 256;

/// Reject absolute output paths and paths containing `..` components.
///
/// Does **not** require the path to exist, making it suitable for output files
/// that will be created by the caller.
pub fn sanitize_output_path(path: &Path) -> Result<(), VecEyesError> {
    if path.is_absolute() {
        return Err(VecEyesError::invalid_config(
            "security::sanitize_output_path",
            format!("absolute output path '{}' is not allowed", path.display()),
        ));
    }
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

/// Resolve an output path under an allowed base directory without requiring the
/// output file to exist yet.
pub fn sanitize_output_path_with_base(
    path: &Path,
    allowed_base: &Path,
) -> Result<PathBuf, VecEyesError> {
    sanitize_output_path(path)?;

    if path.is_absolute() {
        return Err(VecEyesError::invalid_config(
            "security::sanitize_output_path_with_base",
            format!("absolute output path '{}' is not allowed", path.display()),
        ));
    }

    let allowed_base = allowed_base.canonicalize().map_err(|e| {
        VecEyesError::invalid_config(
            "security::sanitize_output_path_with_base",
            format!(
                "allowed base resolution failed for {}: {}",
                allowed_base.display(),
                e
            ),
        )
    })?;
    let candidate = allowed_base.join(path);
    let parent = candidate.parent().unwrap_or(&allowed_base);
    let mut existing_ancestor = parent;
    while !existing_ancestor.exists() {
        existing_ancestor = existing_ancestor.parent().ok_or_else(|| {
            VecEyesError::invalid_config(
                "security::sanitize_output_path_with_base",
                format!("output path '{}' has no existing ancestor", path.display()),
            )
        })?;
    }
    let canonical_parent = existing_ancestor.canonicalize().map_err(|e| {
        VecEyesError::invalid_config(
            "security::sanitize_output_path_with_base",
            format!(
                "output ancestor resolution failed for {}: {}",
                existing_ancestor.display(),
                e
            ),
        )
    })?;

    if !canonical_parent.starts_with(&allowed_base) {
        return Err(VecEyesError::invalid_config(
            "security::sanitize_output_path_with_base",
            format!(
                "output path {} escapes the allowed base directory {}",
                candidate.display(),
                allowed_base.display()
            ),
        ));
    }

    if candidate.file_name().is_none() {
        return Err(VecEyesError::invalid_config(
            "security::sanitize_output_path_with_base",
            format!("output path '{}' does not name a file", path.display()),
        ));
    }
    Ok(candidate)
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

pub fn read_file_limited(
    path: &Path,
    max_bytes: u64,
    context: &str,
) -> Result<Vec<u8>, VecEyesError> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > max_bytes {
        return Err(VecEyesError::invalid_config(
            context,
            format!(
                "file {} exceeds the maximum allowed size of {} bytes",
                path.display(),
                max_bytes
            ),
        ));
    }
    Ok(fs::read(path)?)
}

pub fn read_to_string_limited(
    path: &Path,
    max_bytes: u64,
    context: &str,
) -> Result<String, VecEyesError> {
    let bytes = read_file_limited(path, max_bytes, context)?;
    String::from_utf8(bytes).map_err(|e| VecEyesError::invalid_config(context, e.to_string()))
}
