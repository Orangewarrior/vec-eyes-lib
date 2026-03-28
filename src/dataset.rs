use crate::error::VecEyesError;
use crate::labels::ClassificationLabel;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub label: ClassificationLabel,
    pub text: String,
    pub source_name: String,
}

pub fn read_text_file(path: &Path) -> Result<String, VecEyesError> {
    Ok(fs::read_to_string(path)?)
}

pub fn collect_files_recursively(
    root: &Path,
    recursive: bool,
) -> Result<Vec<PathBuf>, VecEyesError> {
    let mut files = Vec::new();
    if recursive {
        for entry in WalkDir::new(root) {
            let entry = entry.map_err(|e| VecEyesError::InvalidConfig(e.to_string()))?;
            if entry.file_type().is_file() {
                files.push(entry.path().to_path_buf());
            }
        }
    } else {
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                files.push(entry.path());
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
