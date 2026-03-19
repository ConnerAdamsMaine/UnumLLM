pub mod cpu;
pub mod traits;

#[cfg(feature = "rocm")]
mod hip_runtime;

#[cfg(feature = "rocm")]
pub mod rocm;

use crate::error::OneBitError;
use crate::Result;

pub use cpu::CpuBackend;
pub use traits::ComputeBackend;

#[cfg(feature = "rocm")]
pub use rocm::RocmBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Rocm,
}

impl BackendKind {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "rocm" | "hip" => Ok(Self::Rocm),
            other => Err(OneBitError::Other(format!(
                "unknown device/backend `{other}`. Use `cpu` or `rocm`."
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Rocm => "rocm",
        }
    }
}

pub fn create_backend(kind: BackendKind) -> Result<Box<dyn ComputeBackend>> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(CpuBackend)),
        BackendKind::Rocm => {
            #[cfg(feature = "rocm")]
            {
                Ok(Box::new(RocmBackend::new()))
            }
            #[cfg(not(feature = "rocm"))]
            {
                Err(OneBitError::Other(
                    "ROCm backend requested, but this build was compiled without the `rocm` feature".into(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_kind_parse() {
        assert_eq!(BackendKind::parse("cpu").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::parse("rocm").unwrap(), BackendKind::Rocm);
        assert!(BackendKind::parse("cuda").is_err());
    }
}
