"""
OneBitLLM - High-performance 1-bit quantized language models
"""

__version__ = "0.1.0"
__author__ = "OneBitLLM Team"
__license__ = "LicenseRef-OneBitLLM-Research-Only-1.0"

try:
    # Import native module directly (not _core)
    from . import onebitllm as _native
    ModelConfig = _native.PyModelConfig
    OneBitModel = _native.PyModel
    Tokenizer = _native.PyTokenizer
    GenerateConfig = _native.PyGenerateConfig
except ImportError:
    raise ImportError(
        "Failed to import onebitllm native module. "
        "Please ensure the package is built: pip install -e ."
    )

__all__ = [
    "ModelConfig",
    "OneBitModel",
    "Tokenizer",
    "GenerateConfig",
]
