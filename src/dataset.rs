
use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use crate::security::sanitize_existing_path;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const MAX_DISCOVERED_FILES: usize = 10_000;
pub const DEFAULT_MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub label: ClassificationLabel,
    pub text: String,
    pub source_name: String,
}

pub fn read_text_file_limited(path: &Path, max_bytes: u64) -> Result<String, VecEyesError> {
    let canonical = sanitize_existing_path(path)?;
    let metadata = fs::metadata(&canonical)?;
    if metadata.len() > max_bytes {
        return Err(VecEyesError::invalid_config(
            "dataset::read_text_file_limited",
            format!(
                "file {} exceeds the maximum allowed size of {} bytes",
                canonical.display(),
                max_bytes
            ),
        ));
    }
    Ok(fs::read_to_string(canonical)?)
}

pub fn read_text_file(path: &Path) -> Result<String, VecEyesError> {
    read_text_file_limited(path, DEFAULT_MAX_FILE_BYTES)
}

pub fn collect_files_recursively(
    root: &Path,
    recursive: bool,
) -> Result<Vec<PathBuf>, VecEyesError> {
    collect_files_recursively_with_limit(root, recursive, DEFAULT_MAX_FILE_BYTES)
}

pub fn collect_files_recursively_with_limit(
    root: &Path,
    recursive: bool,
    max_bytes: u64,
) -> Result<Vec<PathBuf>, VecEyesError> {
    let mut files = Vec::new();
    if recursive {
        let root = sanitize_existing_path(root)?;
        for entry in WalkDir::new(&root).follow_links(false) {
            let entry = entry.map_err(|e| VecEyesError::invalid_config("dataset::collect_files_recursively_with_limit", e.to_string()))?;
            if entry.file_type().is_file() {
                let metadata = entry.metadata().map_err(|e| {
                    VecEyesError::invalid_config("dataset::collect_files_recursively_with_limit", format!("metadata read failed for {}: {}", entry.path().display(), e))
                })?;
                if metadata.len() > max_bytes {
                    return Err(VecEyesError::invalid_config(
                        "dataset::collect_files_recursively_with_limit",
                        format!(
                            "file {} exceeds the maximum allowed size of {} bytes",
                            entry.path().display(),
                            max_bytes
                        ),
                    ));
                }
                files.push(entry.path().to_path_buf());
                if files.len() > MAX_DISCOVERED_FILES {
                    return Err(VecEyesError::invalid_config(
                        "dataset::collect_files_recursively_with_limit",
                        format!("discovered more than {} files under {}", MAX_DISCOVERED_FILES, root.display()),
                    ));
                }
            }
        }
    } else {
        let root = sanitize_existing_path(root)?;
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let metadata = entry.metadata()?;
                if metadata.len() > max_bytes {
                    return Err(VecEyesError::invalid_config(
                        "dataset::collect_files_recursively_with_limit",
                        format!(
                            "file {} exceeds the maximum allowed size of {} bytes",
                            entry.path().display(),
                            max_bytes
                        ),
                    ));
                }
                files.push(entry.path());
                if files.len() > MAX_DISCOVERED_FILES {
                    return Err(VecEyesError::invalid_config(
                        "dataset::collect_files_recursively_with_limit",
                        format!("discovered more than {} files in {}", MAX_DISCOVERED_FILES, root.display()),
                    ));
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

pub fn load_training_samples(
    path: &Path,
    label: ClassificationLabel,
    recursive: bool,
) -> Result<Vec<TrainingSample>, VecEyesError> {
    let files = collect_files_recursively(path, recursive)?;
    let mut samples = Vec::new();
    for file in files {
        let text = read_text_file(&file)?;
        let source_name = file.to_string_lossy().to_string();
        samples.push(TrainingSample { label: label.clone(), text, source_name });
    }
    Ok(samples)
}
