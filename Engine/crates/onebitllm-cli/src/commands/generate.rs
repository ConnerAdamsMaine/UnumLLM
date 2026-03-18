use anyhow::bail;
use clap::ArgAction;
use clap::Args;

use onebitllm_core::backend::{BackendKind, create_backend};

use super::bigram;

/// Arguments for the `generate` subcommand.
#[derive(Args)]
pub struct GenerateArgs {
    /// Path to the model (.obm file or checkpoint).
    #[arg(short, long)]
    pub model: String,

    /// Input prompt text.
    #[arg(short, long)]
    pub prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    /// Top-k sampling (0 = disabled).
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,

    /// Top-p (nucleus) sampling (1.0 = disabled).
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,

    /// Repetition penalty (1.0 = no penalty).
    #[arg(long, default_value_t = 1.0)]
    pub repetition_penalty: f32,

    /// Random seed for reproducible generation.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Execution device/backend (`cpu` or `rocm`).
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Stream output token by token.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub stream: bool,
}

pub fn run(args: GenerateArgs) -> anyhow::Result<()> {
    log::info!("Starting generation...");
    log::info!("  Model: {}", args.model);
    log::info!("  Max tokens: {}", args.max_tokens);
    log::info!("  Temperature: {}", args.temperature);
    log::info!("  Device: {}", args.device);

    // Check model file exists
    if !std::path::Path::new(&args.model).exists() {
        anyhow::bail!("Model file not found: {}", args.model);
    }

    // Build sampling config
    let _sampling = onebitllm_core::infer::SamplingConfig {
        temperature: args.temperature,
        top_k: if args.top_k > 0 { Some(args.top_k) } else { None },
        top_p: if args.top_p < 1.0 { Some(args.top_p) } else { None },
        repetition_penalty: if args.repetition_penalty > 1.0 {
            Some(args.repetition_penalty)
        } else {
            None
        },
        seed: args.seed,
    };

    let _gen_config = onebitllm_core::infer::GenerateConfig {
        max_new_tokens: args.max_tokens,
        sampling: _sampling.clone(),
        stop_tokens: Vec::new(),
    };

    let model_path = std::path::Path::new(&args.model);
    if let Ok(model) = bigram::load_bigram_model(model_path) {
        let config = &model.config;
        if bigram::is_bigram_architecture(&config.architecture) {
            let backend_kind = BackendKind::parse(&args.device)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let backend = create_backend(backend_kind)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let output = bigram::generate_bigram_loaded(
                &model,
                &args.prompt,
                args.max_tokens,
                _sampling,
                args.stream,
                backend.as_ref(),
            )?;
            if !args.stream {
                println!("{output}");
            }
            return Ok(());
        }
    }

    bail!(
        "The Rust CLI can parse generation settings, but model/tokenizer loading and the real \
generation pipeline are still unimplemented for this model. Real execution exists today only for \
bigram OBM models. Prompt was accepted, but no executable inference path exists yet for `{}`.",
        args.model
    )
}
