use std::fmt;

/// Errors that can occur during spectral prosody analysis.
#[derive(Debug, Clone)]
pub enum ProsodyError {
    /// Not enough nodes to construct a graph.
    InsufficientNodes { got: usize, need: usize },
    /// Eigen-decomposition failed to converge.
    DecompositionFailed(String),
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Empty feature vector.
    EmptyFeature,
    /// Index out of range.
    IndexOutOfRange { index: usize, len: usize },
}

impl fmt::Display for ProsodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientNodes { got, need } => {
                write!(f, "need at least {need} nodes, got {got}")
            }
            Self::DecompositionFailed(msg) => write!(f, "decomposition failed: {msg}"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::EmptyFeature => write!(f, "feature vector is empty"),
            Self::IndexOutOfRange { index, len } => {
                write!(f, "index {index} out of range (len={len})")
            }
        }
    }
}

impl std::error::Error for ProsodyError {}
