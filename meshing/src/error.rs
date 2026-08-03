use std::fmt;
use std::path::PathBuf;

pub type MeshResult<T> = Result<T, MeshError>;

#[derive(Debug)]
pub enum MeshError {
    UnsupportedDimension {
        domain: String,
        dimension: u8,
    },
    InvalidInput(String),
    InvalidFile(String),
    Capability(String),
    UnsupportedFileVersion(u32),
    Cancelled,
    LimitExceeded(String),
    Io(std::io::Error),
    Arrow(String),
    AtomicPublication {
        destination: PathBuf,
        reason: String,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDimension { domain, dimension } => write!(
                f,
                "built-in meshing supports only 2D and 3D domains; {domain:?} is {dimension}D"
            ),
            Self::InvalidInput(message)
            | Self::InvalidFile(message)
            | Self::Capability(message) => f.write_str(message),
            Self::UnsupportedFileVersion(version) => write!(
                f,
                "unsupported mesh schema version {version}; expected {}",
                crate::MESH_SCHEMA_VERSION
            ),
            Self::Cancelled => f.write_str("meshing cancelled"),
            Self::LimitExceeded(message) => f.write_str(message),
            Self::Io(error) => write!(f, "mesh I/O failed: {error}"),
            Self::Arrow(message) => write!(f, "Arrow IPC error: {message}"),
            Self::AtomicPublication {
                destination,
                reason,
            } => write!(
                f,
                "could not publish {} atomically: {reason}",
                destination.display()
            ),
        }
    }
}

impl std::error::Error for MeshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for MeshError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<arrow_schema::ArrowError> for MeshError {
    fn from(value: arrow_schema::ArrowError) -> Self {
        let message = value.to_string();
        if message.contains("configured memory mesh cap exceeded") {
            Self::LimitExceeded("configured memory mesh cap exceeded".into())
        } else {
            Self::Arrow(message)
        }
    }
}
