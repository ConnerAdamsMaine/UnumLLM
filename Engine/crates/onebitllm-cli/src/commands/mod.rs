pub mod train;
pub mod quantize;
pub mod generate;

use clap::Subcommand;

#[derive(Subcommand)]
pub enum Command {
    /// Train a 1-bit quantized model using quantization-aware training.
    Train(train::TrainArgs),
    /// Quantize an existing FP32 model to ternary weights.
    Quantize(quantize::QuantizeArgs),
    /// Generate text from a trained model.
    Generate(generate::GenerateArgs),
}

pub fn run(cmd: Command) -> anyhow::Result<()> {
    match cmd {
        Command::Train(args) => train::run(args),
        Command::Quantize(args) => quantize::run(args),
        Command::Generate(args) => generate::run(args),
    }
}
