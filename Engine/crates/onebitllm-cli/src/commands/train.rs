use anyhow::bail;
use clap::Args;

/// Arguments for the `train` subcommand.
#[derive(Args)]
pub struct TrainArgs {
    /// Path to training data (directory or file).
    #[arg(short, long)]
    pub data: String,

    /// Path to model config JSON file.
    #[arg(short, long)]
    pub config: String,

    /// Output directory for checkpoints and final model.
    #[arg(short, long, default_value = "output")]
    pub output: String,

    /// Number of training epochs.
    #[arg(long, default_value_t = 3)]
    pub epochs: usize,

    /// Batch size.
    #[arg(short, long, default_value_t = 8)]
    pub batch_size: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,

    /// Weight decay for AdamW.
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f64,

    /// Maximum gradient norm for clipping.
    #[arg(long, default_value_t = 1.0)]
    pub max_grad_norm: f64,

    /// Warmup steps for learning rate scheduler.
    #[arg(long, default_value_t = 100)]
    pub warmup_steps: usize,

    /// Total training steps (0 = auto from data).
    #[arg(long, default_value_t = 0)]
    pub max_steps: usize,

    /// Checkpoint save frequency (in steps).
    #[arg(long, default_value_t = 500)]
    pub save_every: usize,

    /// Log frequency (in steps).
    #[arg(long, default_value_t = 10)]
    pub log_every: usize,

    /// Random seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Resume from checkpoint path.
    #[arg(long)]
    pub resume: Option<String>,
}

pub fn run(args: TrainArgs) -> anyhow::Result<()> {
    log::info!("Starting training...");
    log::info!("  Data: {}", args.data);
    log::info!("  Config: {}", args.config);
    log::info!("  Output: {}", args.output);
    log::info!("  Epochs: {}", args.epochs);
    log::info!("  Batch size: {}", args.batch_size);
    log::info!("  LR: {}", args.lr);

    if matches!(
        std::path::Path::new(&args.config)
            .extension()
            .and_then(|ext| ext.to_str()),
        Some(ext) if ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml")
    ) {
        bail!(
            "Model config must be JSON in the current CLI build. YAML parsing is not enabled for `{}`.",
            args.config
        );
    }

    // Load model config
    let config_data = std::fs::read_to_string(&args.config)?;
    if !config_data.trim_start().starts_with('{') {
        bail!(
            "Model config must be a JSON object. Refusing to treat non-JSON input as a model config: {}",
            args.config
        );
    }
    let _model_config = onebitllm_core::io::ModelConfig::from_json_str(&config_data)
        .map_err(|e| anyhow::anyhow!("Failed to load config: {e}"))?;

    log::info!("Loaded JSON model config from {}", args.config);

    if !std::path::Path::new(&args.data).exists() {
        bail!("Training data not found: {}", args.data);
    }

    if let Some(resume) = &args.resume {
        if !std::path::Path::new(resume).exists() {
            bail!("Resume checkpoint not found: {resume}");
        }
    }

    bail!(
        "The Rust CLI can parse the training config, but end-to-end training is still unimplemented. \
Validated inputs:\n  data: {}\n  config: {}\n  output: {}\n\
Expected missing pieces: data loading/tokenization, model assembly, optimizer/scheduler wiring, \
checkpoint writing, and a real training loop.\n\
No checkpoints or model files were written.",
        args.data,
        args.config,
        args.output
    )
}
