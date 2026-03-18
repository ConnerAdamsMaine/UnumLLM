use ndarray::{Array, IxDyn};

use crate::Result;
use crate::tensor::PackedTensor;

/// Backend trait for compute operations.
///
/// The current tree only ships a CPU backend. Accelerator-specific traits below
/// are capability placeholders, not working ROCm/CUDA/Metal/Vulkan backends.
pub trait ComputeBackend: Send + Sync {
    /// Backend name (e.g., "cpu", "cuda", "metal").
    fn name(&self) -> &str;

    /// Quantized matmul: packed (M x K) * dense (K x N) -> f32 (M x N).
    fn packed_matmul(
        &self,
        packed: &PackedTensor,
        dense: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>>;

    /// Dense matmul: f32 (M x K) * f32 (K x N) -> f32 (M x N).
    fn dense_matmul(
        &self,
        a: &Array<f32, IxDyn>,
        b: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>>;

    /// Dense matrix (B x K) * packed matrix^T (M x K)^T -> dense (B x M).
    fn packed_matmul_dense_left_transposed(
        &self,
        lhs: &Array<f32, IxDyn>,
        packed: &PackedTensor,
    ) -> Result<Array<f32, IxDyn>>;

    /// Quantized matvec: packed (M x N) * dense (N,) -> f32 (M,).
    fn packed_matvec(
        &self,
        packed: &PackedTensor,
        input: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>>;
}

/// Experimental accelerator trait stub for a future CUDA/HIP backend.
pub trait CudaOps {
    fn forward_cuda(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>>;
    fn backward_cuda(&self, grad_output: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>>;
}

/// Experimental accelerator trait stub for a future Metal backend.
pub trait MetalOps {
    fn forward_metal(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>>;
}

/// Experimental accelerator trait stub for a future Vulkan backend.
pub trait VulkanOps {
    fn forward_vulkan(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>>;
}
