//! Filesystem helpers for recursive dataset and object loading.

use std::fs;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::error::{VecEyesError, VecEyesResult};
use crate::labels::ClassificationLabel;
use crate::types::{RawSample, SampleOrigin};

pub fn collect_files_recursively(path: impl AsRef<Path>) -> VecEyesResult<Vec<PathBuf>> {
    let root = path.as_ref();
    if !root.exists() {
        return Err(VecEyesError::MissingPath(root.to_path_buf()));
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(root) {
        let entry = entry?;
        if entry.file_type().is_file() {
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

pub fn load_directory_as_samples(
    path: impl AsRef<Path>,
    label: ClassificationLabel,
) -> VecEyesResult<Vec<RawSample>> {
    let mut samples = Vec::new();

    for file in collect_files_recursively(path)? {
        let text = fs::read_to_string(&file)?;
        let source_name = file
            .file_name()
            .map(|item| item.to_string_lossy().to_string())
            .unwrap_or_else(|| file.display().to_string());

        samples.push(RawSample {
            label,
            text,
            source_name,
            origin: SampleOrigin::DatasetFile(file),
        });
    }

    Ok(samples)
}
