use std::fmt;

/// Kernel-level construction/validation failure (the Rust analog of the
/// `ValueError`s raised by the Python kernel's `__post_init__` checks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryError(pub String);

impl GeometryError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GeometryError {}

pub type GeometryResult<T> = Result<T, GeometryError>;
