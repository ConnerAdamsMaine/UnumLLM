use thiserror::Error;

#[derive(Error, Debug)]
pub enum OneBitError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Bitpack error: {0}")]
    Bitpack(String),

    #[error("Tensor operation error: {0}")]
    TensorOp(String),

    #[error("Autograd error: {0}")]
    Autograd(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model config error: {0}")]
    Config(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, OneBitError>;
