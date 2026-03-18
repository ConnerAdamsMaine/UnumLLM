#!/usr/bin/env python3
"""Single Python wrapper for `onebitllm generate`."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import GenerateCommandConfig
from engine_client import stream_engine_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rust generation through the Python frontend")
    parser.add_argument("--engine", default=None, help="Path to the Rust CLI binary")
    parser.add_argument("--model", required=True, help="Path to the Rust model/checkpoint")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_generate_job(args: argparse.Namespace) -> GenerateCommandConfig:
    """Build a generate job from parsed CLI args."""

    return GenerateCommandConfig(
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        device=args.device,
        stream=args.stream,
    )


def validate_generate_inputs(job: GenerateCommandConfig) -> Path:
    """Validate Python-side paths before spawning Rust."""

    model_path = Path(job.model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def run_generate_command(
    job: GenerateCommandConfig,
    *,
    engine: str | None = None,
    on_output=None,
) -> int:
    """Run a Rust generation job from Python."""

    validate_generate_inputs(job)
    return stream_engine_command(
        job.build_args(),
        explicit_binary=engine,
        on_output=on_output,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the generate wrapper."""

    args = build_parser().parse_args(argv)
    try:
        job = build_generate_job(args)
        return run_generate_command(job, engine=args.engine)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
