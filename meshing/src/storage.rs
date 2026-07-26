use std::io::{self, Write};
use std::ops::Deref;
use std::sync::Arc;

use crate::error::{MeshError, MeshResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshArtifact {
    #[cfg(not(target_arch = "wasm32"))]
    Native(std::path::PathBuf),
    Memory(MemoryArtifact),
}

/// Immutable in-memory artifact whose byte allocation is preserved when a
/// writer publishes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryArtifact(Arc<Vec<u8>>);

impl MemoryArtifact {
    pub fn from_vec(bytes: Vec<u8>) -> Self {
        Self(Arc::new(bytes))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<u8>> for MemoryArtifact {
    fn from(bytes: Vec<u8>) -> Self {
        Self::from_vec(bytes)
    }
}

impl From<Arc<[u8]>> for MemoryArtifact {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self::from_vec(bytes.as_ref().to_vec())
    }
}

impl AsRef<[u8]> for MemoryArtifact {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl Deref for MemoryArtifact {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

pub trait MeshStorage {
    type Writer: Write;

    fn begin(&mut self) -> MeshResult<Self::Writer>;
    fn publish(self, writer: Self::Writer) -> MeshResult<MeshArtifact>;
}

#[derive(Debug)]
pub struct CappedBuffer {
    bytes: Vec<u8>,
    cap: usize,
}

impl CappedBuffer {
    fn new(cap: usize) -> Self {
        Self {
            bytes: Vec::new(),
            cap,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl Write for CappedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.bytes.len().saturating_add(bytes.len()) > self.cap {
            return Err(io::Error::other("configured memory mesh cap exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryStorage {
    cap: usize,
}

impl MemoryStorage {
    pub fn new(cap: usize) -> MeshResult<Self> {
        if cap == 0 {
            return Err(MeshError::InvalidInput(
                "memory mesh cap must be positive".into(),
            ));
        }
        Ok(Self { cap })
    }

    pub const fn cap(&self) -> usize {
        self.cap
    }
}

impl MeshStorage for MemoryStorage {
    type Writer = CappedBuffer;

    fn begin(&mut self) -> MeshResult<Self::Writer> {
        Ok(CappedBuffer::new(self.cap))
    }

    fn publish(self, writer: Self::Writer) -> MeshResult<MeshArtifact> {
        Ok(MeshArtifact::Memory(writer.bytes.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publishing_preserves_the_vec_allocation() {
        let mut storage = MemoryStorage::new(1024).unwrap();
        let mut writer = storage.begin().unwrap();
        writer.write_all(&[7; 128]).unwrap();
        let allocation = writer.bytes.as_ptr();
        let MeshArtifact::Memory(artifact) = storage.publish(writer).unwrap() else {
            unreachable!()
        };
        assert_eq!(artifact.as_ptr(), allocation);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct NativeFileStorage {
    destination: std::path::PathBuf,
    candidate: std::path::PathBuf,
    published: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativeFileStorage {
    pub fn new(destination: impl AsRef<std::path::Path>) -> MeshResult<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let destination = destination.as_ref().to_path_buf();
        let parent = destination.parent().ok_or_else(|| {
            MeshError::InvalidInput("native mesh destination has no parent directory".into())
        })?;
        let name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| MeshError::InvalidInput("native mesh filename is invalid".into()))?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let candidate = parent.join(format!(".{name}.candidate-{}-{nonce}", std::process::id()));
        Ok(Self {
            destination,
            candidate,
            published: false,
        })
    }

    pub fn destination(&self) -> &std::path::Path {
        &self.destination
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl MeshStorage for NativeFileStorage {
    type Writer = std::fs::File;

    fn begin(&mut self) -> MeshResult<Self::Writer> {
        use std::fs::OpenOptions;

        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.candidate)
            .map_err(Into::into)
    }

    fn publish(mut self, mut writer: Self::Writer) -> MeshResult<MeshArtifact> {
        writer.flush()?;
        writer.sync_all()?;
        drop(writer);
        crate::file::validate_metadata_path(&self.candidate)?;
        std::fs::rename(&self.candidate, &self.destination).map_err(|error| {
            MeshError::AtomicPublication {
                destination: self.destination.clone(),
                reason: error.to_string(),
            }
        })?;
        if let Some(parent) = self.destination.parent() {
            if let Ok(directory) = std::fs::File::open(parent) {
                let _ = directory.sync_all();
            }
        }
        self.published = true;
        Ok(MeshArtifact::Native(self.destination.clone()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for NativeFileStorage {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.candidate);
        }
    }
}
