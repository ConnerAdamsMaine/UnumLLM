use anyhow::bail;
use clap::Args;

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
            Ok(onebitllm_core::quant::QuantGranularity::PerGroup(group_size))
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
                Ok(onebitllm_core::quant::QuantGranularity::PerGroup(parsed_size))
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

    bail!(
        "The Rust CLI can validate quantization settings, but model loading and OBM export are still unimplemented.\n\
Validated inputs:\n  input: {}\n  output: {}\n  granularity: {}\n  zero_point: {}\n  size: {:.2} MB\n\
No output file was written.",
        args.input,
        args.output,
        granularity_desc,
        args.use_zero_point,
        input_size as f64 / (1024.0 * 1024.0)
    )
}
