use std::fs;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use clap::Args;

use onebitllm_core::backend::{create_backend, BackendKind};
use onebitllm_core::infer::SamplingConfig;

use super::bigram;

#[derive(Args, Clone)]
pub struct BenchmarkArgs {
    /// Path to the model (.obm file).
    #[arg(short, long)]
    pub model: String,

    /// Single prompt to benchmark. Ignored when --prompts-file is set.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Path to a file containing one prompt per line.
    #[arg(long)]
    pub prompts_file: Option<String>,

    /// Maximum number of new tokens to generate per request.
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,

    /// Number of measured requests.
    #[arg(long, default_value_t = 16)]
    pub requests: usize,

    /// Number of warmup requests run before measurement.
    #[arg(long, default_value_t = 2)]
    pub warmup: usize,

    /// Number of concurrent workers.
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,

    /// Execution device/backend (`cpu` or `rocm`).
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Sampling temperature (0.0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Top-k sampling (0 = disabled).
    #[arg(long, default_value_t = 0)]
    pub top_k: usize,

    /// Top-p (nucleus) sampling (1.0 = disabled).
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,

    /// Repetition penalty (1.0 = disabled).
    #[arg(long, default_value_t = 1.0)]
    pub repetition_penalty: f32,

    /// Optional base seed for deterministic request streams.
    #[arg(long)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone)]
struct RequestMetric {
    latency: Duration,
    generated_tokens: usize,
}

pub fn run(args: BenchmarkArgs) -> anyhow::Result<()> {
    if !std::path::Path::new(&args.model).exists() {
        bail!("Model file not found: {}", args.model);
    }
    if args.requests == 0 {
        bail!("--requests must be greater than 0");
    }
    if args.concurrency == 0 {
        bail!("--concurrency must be greater than 0");
    }

    let prompts = load_prompts(&args)?;
    let model_path = std::path::Path::new(&args.model);
    let load_start = Instant::now();
    let model = bigram::load_bigram_model(model_path)?;
    let load_latency = load_start.elapsed();
    let backend_kind = BackendKind::parse(&args.device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let warmup_backend = create_backend(backend_kind).map_err(|e| anyhow::anyhow!("{e}"))?;

    log::info!(
        "Loaded benchmark model {} in {:.2} ms",
        args.model,
        duration_ms(load_latency)
    );

    for warmup_idx in 0..args.warmup {
        let prompt = &prompts[warmup_idx % prompts.len()];
        let output = bigram::generate_bigram_loaded(
            &model,
            prompt,
            args.max_tokens,
            sampling_config(&args, request_seed(args.seed, warmup_idx))?,
            false,
            warmup_backend.as_ref(),
        )?;
        log::info!(
            "warmup {}/{} prompt_bytes={} generated_tokens={}",
            warmup_idx + 1,
            args.warmup,
            prompt.len(),
            output.len().saturating_sub(prompt.len())
        );
    }

    let bench_start = Instant::now();
    let metrics = if args.concurrency == 1 {
        run_benchmark_worker(model, prompts.clone(), args.clone(), 0, 1)?
    } else {
        let mut handles = Vec::with_capacity(args.concurrency);
        for worker_id in 0..args.concurrency {
            let worker_model = model.clone();
            let worker_prompts = prompts.clone();
            let worker_args = args.clone();
            let worker_count = worker_args.concurrency;
            handles.push(thread::spawn(move || {
                run_benchmark_worker(
                    worker_model,
                    worker_prompts,
                    worker_args,
                    worker_id,
                    worker_count,
                )
            }));
        }

        let mut combined = Vec::new();
        for handle in handles {
            combined.extend(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("benchmark worker panicked"))??,
            );
        }
        combined
    };
    let wall_time = bench_start.elapsed();

    if metrics.is_empty() {
        bail!("benchmark did not execute any measured requests");
    }

    let latencies = metrics
        .iter()
        .map(|metric| metric.latency)
        .collect::<Vec<_>>();
    let total_tokens = metrics
        .iter()
        .map(|metric| metric.generated_tokens)
        .sum::<usize>();
    let request_count = metrics.len();

    println!("load_ms={:.2}", duration_ms(load_latency));
    println!("requests={request_count}");
    println!("concurrency={}", args.concurrency);
    println!("generated_tokens={total_tokens}");
    println!("wall_ms={:.2}", duration_ms(wall_time));
    println!(
        "throughput_req_per_s={:.3}",
        request_count as f64 / wall_time.as_secs_f64()
    );
    println!(
        "throughput_tok_per_s={:.3}",
        total_tokens as f64 / wall_time.as_secs_f64()
    );
    println!("latency_p50_ms={:.2}", percentile_ms(&latencies, 0.50));
    println!("latency_p95_ms={:.2}", percentile_ms(&latencies, 0.95));
    println!("latency_p99_ms={:.2}", percentile_ms(&latencies, 0.99));
    println!(
        "latency_avg_ms={:.2}",
        latencies
            .iter()
            .map(|value| duration_ms(*value))
            .sum::<f64>()
            / request_count as f64
    );

    Ok(())
}

fn load_prompts(args: &BenchmarkArgs) -> anyhow::Result<Vec<String>> {
    if let Some(path) = &args.prompts_file {
        let prompts = fs::read_to_string(path)
            .with_context(|| format!("failed to read prompts file {path}"))?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if prompts.is_empty() {
            bail!("prompts file `{path}` did not contain any non-empty lines");
        }
        return Ok(prompts);
    }

    Ok(vec![args.prompt.clone().unwrap_or_else(|| {
        "The future of 1-bit models is ".into()
    })])
}

fn run_benchmark_worker(
    model: bigram::LoadedBigramModel,
    prompts: Vec<String>,
    args: BenchmarkArgs,
    worker_id: usize,
    worker_count: usize,
) -> anyhow::Result<Vec<RequestMetric>> {
    let mut metrics = Vec::new();
    let backend_kind = BackendKind::parse(&args.device).map_err(|e| anyhow::anyhow!("{e}"))?;
    let backend = create_backend(backend_kind).map_err(|e| anyhow::anyhow!("{e}"))?;

    for request_idx in (worker_id..args.requests).step_by(worker_count) {
        let prompt = &prompts[request_idx % prompts.len()];
        let start = Instant::now();
        let output = bigram::generate_bigram_loaded(
            &model,
            prompt,
            args.max_tokens,
            sampling_config(&args, request_seed(args.seed, request_idx))?,
            false,
            backend.as_ref(),
        )?;
        metrics.push(RequestMetric {
            latency: start.elapsed(),
            generated_tokens: output.len().saturating_sub(prompt.len()),
        });
    }

    Ok(metrics)
}

fn sampling_config(args: &BenchmarkArgs, seed: Option<u64>) -> anyhow::Result<SamplingConfig> {
    if args.temperature < 0.0 {
        bail!("--temperature must be greater than or equal to 0");
    }
    if args.top_p <= 0.0 || args.top_p > 1.0 {
        bail!("--top-p must be in the range (0, 1]");
    }
    if args.repetition_penalty < 1.0 {
        bail!("--repetition-penalty must be at least 1.0");
    }

    Ok(SamplingConfig {
        temperature: args.temperature,
        top_k: if args.top_k > 0 {
            Some(args.top_k)
        } else {
            None
        },
        top_p: if args.top_p < 1.0 {
            Some(args.top_p)
        } else {
            None
        },
        repetition_penalty: if args.repetition_penalty > 1.0 {
            Some(args.repetition_penalty)
        } else {
            None
        },
        seed,
    })
}

fn request_seed(base_seed: Option<u64>, request_idx: usize) -> Option<u64> {
    base_seed.map(|seed| seed.wrapping_add(request_idx as u64))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn percentile_ms(values: &[Duration], percentile: f64) -> f64 {
    let mut sorted = values
        .iter()
        .map(|value| duration_ms(*value))
        .collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let index = ((sorted.len().saturating_sub(1)) as f64 * percentile).round() as usize;
    sorted[index]
}
