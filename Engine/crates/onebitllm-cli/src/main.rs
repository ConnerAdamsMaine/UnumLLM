mod commands;

use clap::Parser;

/// OneBitLLM — Train, quantize, and run 1-bit quantized language models.
#[derive(Parser)]
#[command(name = "onebitllm", version, about)]
struct Cli {
    #[command(subcommand)]
    command: commands::Command,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    commands::run(cli.command)
}
