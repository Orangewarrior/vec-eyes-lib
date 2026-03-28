use thiserror::Error;

#[derive(Debug, Error)]
pub enum VecEyesError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("yaml error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),

    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),
}
