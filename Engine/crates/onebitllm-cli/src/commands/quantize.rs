use anyhow::bail;
use clap::Args;

use super::bigram;

/// Arguments for the `quantize` subcommand.
#[derive(Args)]
pub struct QuantizeArgs {
    /// Path to input model (checkpoint or SafeTensors file).
    #[arg(short, long)]
    pub input: String,

    /// Path to output quantized model (.obm file).
    #[arg(short, long)]
    pub output: String,

    /// Quantization granularity: "per-tensor", "per-channel", or "per-group:N".
    #[arg(long, default_value = "per-tensor")]
    pub granularity: String,

    /// Group size for per-group quantization.
    #[arg(long, default_value_t = 128)]
    pub group_size: usize,

    /// Use zero-point (asymmetric quantization).
    #[arg(long, default_value_t = false)]
    pub use_zero_point: bool,

    /// Output weight format (`fp32`, `binary`, or `ternary`).
    #[arg(long, default_value = "binary")]
    pub target_weight_format: String,

    /// Optional evaluation corpus for before/after quality checks.
    #[arg(long)]
    pub eval_data: Option<String>,
}

fn parse_granularity(
    granularity: &str,
    group_size: usize,
) -> anyhow::Result<onebitllm_core::quant::QuantGranularity> {
    match granularity {
        "per-tensor" => Ok(onebitllm_core::quant::QuantGranularity::PerTensor),
        "per-channel" => Ok(onebitllm_core::quant::QuantGranularity::PerChannel),
        "per-group" => {
            if group_size == 0 {
                bail!("--group-size must be greater than 0 when using per-group quantization");
            }
            Ok(onebitllm_core::quant::QuantGranularity::PerGroup(
                group_size,
            ))
        }
        other => {
            if let Some(raw_size) = other.strip_prefix("per-group:") {
                let parsed_size = raw_size.parse::<usize>().map_err(|_| {
                    anyhow::anyhow!(
                        "Invalid per-group granularity `{other}`. Use `per-group:N` with a positive integer."
                    )
                })?;
                if parsed_size == 0 {
                    bail!("per-group granularity must use a group size greater than 0");
                }
                Ok(onebitllm_core::quant::QuantGranularity::PerGroup(
                    parsed_size,
                ))
            } else {
                bail!(
                    "Unknown granularity: {other}. Use per-tensor, per-channel, per-group, or per-group:N"
                )
            }
        }
    }
}

pub fn run(args: QuantizeArgs) -> anyhow::Result<()> {
    log::info!("Starting post-training quantization...");
    log::info!("  Input: {}", args.input);
    log::info!("  Output: {}", args.output);
    log::info!("  Granularity: {}", args.granularity);

    // Parse granularity
    let granularity = parse_granularity(&args.granularity, args.group_size)?;

    // Check input exists
    if !std::path::Path::new(&args.input).exists() {
        anyhow::bail!("Input file not found: {}", args.input);
    }

    let input_data = std::fs::read(&args.input)?;
    let input_size = input_data.len();
    let granularity_desc = match granularity {
        onebitllm_core::quant::QuantGranularity::PerTensor => "per-tensor".to_string(),
        onebitllm_core::quant::QuantGranularity::PerChannel => "per-channel".to_string(),
        onebitllm_core::quant::QuantGranularity::PerGroup(size) => {
            format!("per-group:{size}")
        }
    };

    let target_weight_format = bigram::normalize_weight_format(&args.target_weight_format)?;
    let input_path = std::path::Path::new(&args.input);
    let output_path = std::path::Path::new(&args.output);
    if args.use_zero_point {
        bail!(
            "bigram quantization does not support --use-zero-point; packed runtime export is symmetric only"
        );
    }
    if let Some(eval_data) = &args.eval_data {
        if !std::path::Path::new(eval_data).exists() {
            bail!("Evaluation corpus not found: {eval_data}");
        }
    }
    if bigram::load_bigram_obm(input_path).is_ok() {
        let source_model = bigram::load_bigram_model(input_path)?;
        let quant_metrics = bigram::quantize_bigram_model(
            input_path,
            output_path,
            &target_weight_format,
            granularity,
        )?;
        let deployed_model = bigram::load_bigram_model(output_path)?;
        log::info!(
            "Wrote converted bigram model with target_weight_format={} to {}",
            target_weight_format,
            output_path.display()
        );
        if let Some(metrics) = quant_metrics {
            log::info!(
                "conversion diagnostics mse={:.6} mean_abs_error={:.6} max_abs_error={:.6} exact_match_fraction={:.4}",
                metrics.mse,
                metrics.mean_abs_error,
                metrics.max_abs_error,
                metrics.exact_match_fraction
            );
        }
        if let Some(eval_data) = &args.eval_data {
            let eval_tokens = bigram::collect_corpus_bytes(std::path::Path::new(eval_data))?;
            let source_eval =
                bigram::evaluate_bigram_weight(&source_model.dense_weight, &eval_tokens)?;
            let deployed_eval =
                bigram::evaluate_bigram_weight(&deployed_model.dense_weight, &eval_tokens)?;
            log::info!(
                "source eval pairs={} loss={:.4} ppl={:.4} accuracy={:.4}",
                source_eval.pair_count,
                source_eval.avg_loss,
                source_eval.perplexity,
                source_eval.accuracy
            );
            log::info!(
                "deployed eval pairs={} loss={:.4} ppl={:.4} accuracy={:.4}",
                deployed_eval.pair_count,
                deployed_eval.avg_loss,
                deployed_eval.perplexity,
                deployed_eval.accuracy
            );
            let retention = bigram::compare_bigram_weights(
                &deployed_model.dense_weight,
                &source_model.dense_weight,
                &eval_tokens,
            )?;
            log::info!(
                "teacher retention pairs={} top1_agreement={:.4} avg_kl_divergence={:.6}",
                retention.pair_count,
                retention.top1_agreement,
                retention.avg_kl_divergence
            );
        }
        return Ok(());
    }

    bail!(
        "The Rust CLI can validate quantization settings, but real conversion is currently implemented only for bigram OBM models.\n\
Validated inputs:\n  input: {}\n  output: {}\n  granularity: {}\n  zero_point: {}\n  target_weight_format: {}\n  size: {:.2} MB\n\
No output file was written.",
        args.input,
        args.output,
        granularity_desc,
        args.use_zero_point,
        target_weight_format,
        input_size as f64 / (1024.0 * 1024.0)
    )
}
