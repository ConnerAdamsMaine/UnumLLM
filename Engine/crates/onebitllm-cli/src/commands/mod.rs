pub mod benchmark;
pub mod bigram;
pub mod generate;
pub mod quantize;
pub mod train;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Command {
    /// Train a 1-bit quantized model using quantization-aware training.
    Train(train::TrainArgs),
    /// Quantize an existing FP32 model to packed binary or ternary weights.
    Quantize(quantize::QuantizeArgs),
    /// Generate text from a trained model.
    Generate(generate::GenerateArgs),
    /// Benchmark deployed model latency and throughput on real prompts.
    Benchmark(benchmark::BenchmarkArgs),
}

pub fn run(cmd: Command) -> anyhow::Result<()> {
    match cmd {
        Command::Train(args) => train::run(args),
        Command::Quantize(args) => quantize::run(args),
        Command::Generate(args) => generate::run(args),
        Command::Benchmark(args) => benchmark::run(args),
    }
}
