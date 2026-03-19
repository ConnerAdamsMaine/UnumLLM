use anyhow::bail;
use clap::Args;

use super::bigram;

/// Arguments for the `train` subcommand.
#[derive(Args)]
pub struct TrainArgs {
    /// Path to training data (TXT/CSV/JSON/JSONL file or directory).
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

    /// Optional teacher bigram model for distillation.
    #[arg(long)]
    pub teacher_model: Option<String>,

    /// Optional evaluation corpus (TXT/CSV/JSON/JSONL file or directory).
    #[arg(long)]
    pub eval_data: Option<String>,

    /// Training-time weight format (`fp32`, `binary`, or `ternary`).
    #[arg(long, default_value = "same-as-config")]
    pub train_weight_format: String,

    /// Output weight format (`same-as-train`, `fp32`, `binary`, or `ternary`).
    #[arg(long, default_value = "same-as-train")]
    pub save_weight_format: String,

    /// Blend factor for teacher distillation loss (0.0-1.0).
    #[arg(long, default_value_t = 0.0)]
    pub distill_alpha: f32,

    /// Temperature used when matching teacher next-token distributions.
    #[arg(long, default_value_t = 1.0)]
    pub distill_temperature: f32,
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
    if let Some(teacher_model) = &args.teacher_model {
        if !std::path::Path::new(teacher_model).exists() {
            bail!("Teacher model not found: {teacher_model}");
        }
    }
    if let Some(eval_data) = &args.eval_data {
        if !std::path::Path::new(eval_data).exists() {
            bail!("Evaluation corpus not found: {eval_data}");
        }
    }
    if !(0.0..=1.0).contains(&args.distill_alpha) {
        bail!("--distill-alpha must be between 0.0 and 1.0");
    }
    if args.distill_temperature <= 0.0 {
        bail!("--distill-temperature must be greater than 0");
    }

    let train_weight_format = if args
        .train_weight_format
        .eq_ignore_ascii_case("same-as-config")
    {
        _model_config.training_weight_format.clone()
    } else {
        args.train_weight_format.clone()
    };
    let save_weight_format =
        super::bigram::resolve_save_weight_format(&train_weight_format, &args.save_weight_format)?;

    if bigram::is_bigram_architecture(&_model_config.architecture) {
        let output_model = bigram::train_bigram(
            &_model_config,
            bigram::BigramTrainArgs {
                corpus_path: std::path::Path::new(&args.data),
                output_dir: std::path::Path::new(&args.output),
                resume_path: args.resume.as_deref().map(std::path::Path::new),
                teacher_model_path: args.teacher_model.as_deref().map(std::path::Path::new),
                eval_path: args.eval_data.as_deref().map(std::path::Path::new),
                epochs: args.epochs,
                batch_size: args.batch_size,
                lr: args.lr as f32,
                weight_decay: args.weight_decay as f32,
                max_grad_norm: args.max_grad_norm as f32,
                warmup_steps: args.warmup_steps,
                max_steps: args.max_steps,
                save_every: args.save_every,
                log_every: args.log_every,
                seed: args.seed,
                train_weight_format: &train_weight_format,
                save_weight_format: &save_weight_format,
                distill_alpha: args.distill_alpha,
                distill_temperature: args.distill_temperature,
            },
        )?;
        log::info!(
            "Finished real bigram training with train_weight_format={} save_weight_format={}. Model written to {}",
            train_weight_format,
            save_weight_format,
            output_model.display()
        );
        return Ok(());
    }

    bail!(
        "The Rust CLI can parse the training config, but end-to-end training is still unimplemented. \
Validated inputs:\n  data: {}\n  config: {}\n  output: {}\n\
Requested train_weight_format: {}\n  requested save_weight_format: {}\n\
Real execution exists today only for `architecture = bigram`. Other architectures are still missing: \
data loading/tokenization, model assembly, optimizer/scheduler wiring, checkpoint writing, and a real training loop.\n\
No checkpoints or model files were written.",
        args.data,
        args.config,
        args.output,
        train_weight_format,
        save_weight_format
    )
}
