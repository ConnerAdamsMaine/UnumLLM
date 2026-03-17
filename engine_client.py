#!/usr/bin/env python3
"""Helpers for locating and streaming commands to the Rust engine."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parent
ENGINE_ROOT = REPO_ROOT / "Engine"
REPO_ENGINE_TARGET = ENGINE_ROOT / "target"
CAPABILITY_ORDER = ("train", "generate", "quantize")
VALIDATE_ONLY_MARKERS = {
    "train": (
        "training config, but end-to-end training is still unimplemented",
        "no checkpoints or model files were written",
    ),
    "generate": (
        "generation settings, but model/tokenizer loading and the real generation pipeline are still unimplemented",
        "no executable inference path exists yet",
    ),
    "quantize": (
        "can validate quantization settings, but model loading and obm export are still unimplemented",
        "no output file was written",
    ),
}


@dataclass(slots=True)
class EngineCapability:
    name: str
    state: str
    summary: str
    detail: str = ""


@dataclass(slots=True)
class EngineStatus:
    binary_path: Path | None
    binary_source: str
    binary_summary: str
    binary_freshness: str
    bindings_available: bool
    bindings_summary: str
    capabilities: dict[str, EngineCapability] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _EngineBinaryCandidate:
    path: Path
    source: str


def _existing_repo_binary_candidates() -> list[_EngineBinaryCandidate]:
    candidates: list[_EngineBinaryCandidate] = []
    for relative_path, label in (
        ("debug/onebitllm", "repo debug build"),
        ("release/onebitllm", "repo release build"),
    ):
        candidate = REPO_ENGINE_TARGET / relative_path
        if candidate.is_file():
            candidates.append(_EngineBinaryCandidate(candidate.resolve(), label))
    return candidates


def _latest_repo_binary_candidate() -> _EngineBinaryCandidate | None:
    candidates = _existing_repo_binary_candidates()
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.path.stat().st_mtime)


def resolve_engine_binary(explicit_path: str | Path | None = None) -> Path:
    """Locate the Rust CLI binary."""

    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Rust engine CLI not found at explicit path: {candidate}")

    env_binary = os.environ.get("ONEBITLLM_ENGINE_BIN")
    if env_binary:
        candidate = Path(env_binary).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Rust engine CLI not found at ONEBITLLM_ENGINE_BIN={candidate}")

    repo_binary = _latest_repo_binary_candidate()
    if repo_binary is not None:
        return repo_binary.path

    path_binary = shutil.which("onebitllm")
    if path_binary:
        return Path(path_binary).resolve()

    raise FileNotFoundError(
        "Rust engine CLI not found. Build it with `cargo build --release -p onebitllm-cli` "
        "inside `Engine/` or set `ONEBITLLM_ENGINE_BIN`."
    )


def _describe_binary_source(binary_path: Path, explicit_path: str | Path | None = None) -> str:
    if explicit_path:
        return "explicit path"

    env_binary = os.environ.get("ONEBITLLM_ENGINE_BIN")
    if env_binary and binary_path == Path(env_binary).expanduser().resolve():
        return "ONEBITLLM_ENGINE_BIN"

    repo_binary = _latest_repo_binary_candidate()
    if repo_binary is not None and binary_path == repo_binary.path:
        return repo_binary.source

    return "PATH lookup"


def _latest_engine_source_mtime() -> float | None:
    latest_mtime = 0.0
    for path in ENGINE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".rs", ".toml"}:
            continue
        latest_mtime = max(latest_mtime, path.stat().st_mtime)
    return latest_mtime or None


def _binary_freshness(binary_path: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []

    try:
        binary_path.relative_to(REPO_ENGINE_TARGET)
    except ValueError:
        return ("external binary; repo freshness unavailable", warnings)

    latest_source_mtime = _latest_engine_source_mtime()
    if latest_source_mtime is None:
        return ("source freshness unavailable", warnings)

    binary_mtime = binary_path.stat().st_mtime
    if binary_mtime + 1 < latest_source_mtime:
        warnings.append(
            "Selected engine binary is older than the latest Rust source edits. "
            "Rebuild `Engine/` to avoid stale CLI behavior."
        )
        return ("stale", warnings)

    return ("fresh", warnings)


def _probe_command_args(command: str, tempdir: Path) -> list[str]:
    if command == "generate":
        model_path = tempdir / "probe-model.obm"
        model_path.write_text("{}", encoding="utf-8")
        return ["generate", "--model", str(model_path), "--prompt", "probe"]

    if command == "train":
        data_path = tempdir / "probe-data.txt"
        data_path.write_text("probe\n", encoding="utf-8")
        config_path = tempdir / "probe-config.json"
        config_path.write_text(
            (
                '{"architecture":"bitnet-b1.58","hidden_size":768,"num_layers":12,'
                '"num_attention_heads":12,"num_kv_heads":12,"intermediate_size":2048,'
                '"vocab_size":32000,"max_seq_len":2048,"rms_norm_eps":1e-5,'
                '"activation":"silu","positional_encoding":"rope","rope_theta":10000.0,'
                '"use_bias":false,"quant_group_size":0}\n'
            ),
            encoding="utf-8",
        )
        return ["train", "--data", str(data_path), "--config", str(config_path)]

    if command == "quantize":
        input_path = tempdir / "probe-model.bin"
        input_path.write_bytes(b"probe")
        output_path = tempdir / "probe-output.obm"
        return ["quantize", "--input", str(input_path), "--output", str(output_path)]

    raise ValueError(f"Unsupported probe command: {command}")


def _detect_missing_command(output: str, command: str) -> bool:
    lowered = output.lower()
    return (
        f"unrecognized subcommand '{command}'" in lowered
        or f"unrecognized subcommand `{command}`" in lowered
        or f"unknown subcommand '{command}'" in lowered
        or f"unknown subcommand `{command}`" in lowered
    )


def probe_engine_capability(binary_path: Path, command: str) -> EngineCapability:
    """Probe a single command to see whether it is real, validate-only, or missing."""

    try:
        with tempfile.TemporaryDirectory(prefix=f"onebitllm-{command}-probe-") as tempdir:
            result = subprocess.run(
                [str(binary_path), *_probe_command_args(command, Path(tempdir))],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=False,
                timeout=10,
            )
    except subprocess.TimeoutExpired:
        return EngineCapability(
            name=command,
            state="unknown",
            summary="probe timed out",
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        return EngineCapability(
            name=command,
            state="missing",
            summary=f"probe failed: {exc}",
        )

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    lowered = output.lower()
    if any(marker in lowered for marker in VALIDATE_ONLY_MARKERS[command]):
        return EngineCapability(
            name=command,
            state="validate-only",
            summary="input validation only; execution is unimplemented",
            detail=output,
        )

    if _detect_missing_command(output, command):
        return EngineCapability(
            name=command,
            state="missing",
            summary="command is not available in this binary",
            detail=output,
        )

    if result.returncode == 0:
        return EngineCapability(
            name=command,
            state="real",
            summary="command executed successfully during probe",
            detail=output,
        )

    return EngineCapability(
        name=command,
        state="unknown",
        summary=f"probe exited with code {result.returncode}",
        detail=output,
    )


def detect_engine_status(explicit_path: str | Path | None = None) -> EngineStatus:
    warnings: list[str] = []
    issues: list[str] = []
    binary_path: Path | None = None
    binary_source = "missing"
    binary_summary = "missing"
    binary_freshness = "missing"
    capabilities = {
        command: EngineCapability(command, "missing", "engine binary not available")
        for command in CAPABILITY_ORDER
    }

    try:
        binary_path = resolve_engine_binary(explicit_path)
        binary_source = _describe_binary_source(binary_path, explicit_path)
        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        version_output = result.stdout.strip() or result.stderr.strip() or binary_path.name
        binary_summary = f"{version_output} ({binary_source})"
        binary_freshness, freshness_warnings = _binary_freshness(binary_path)
        warnings.extend(freshness_warnings)
        capabilities = {
            command: probe_engine_capability(binary_path, command)
            for command in CAPABILITY_ORDER
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        issues.append(str(exc))

    bindings_available = False
    bindings_summary = "not installed"
    try:
        import onebitllm  # type: ignore

        bindings_available = True
        version_fn = getattr(onebitllm, "version", None)
        if callable(version_fn):
            bindings_summary = f"onebitllm {version_fn()}"
        else:
            bindings_summary = f"onebitllm from {getattr(onebitllm, '__file__', '?')}"
    except Exception as exc:  # pragma: no cover - diagnostics only
        issues.append(f"python bindings: {exc}")

    return EngineStatus(
        binary_path=binary_path,
        binary_source=binary_source,
        binary_summary=binary_summary,
        binary_freshness=binary_freshness,
        bindings_available=bindings_available,
        bindings_summary=bindings_summary,
        capabilities=capabilities,
        warnings=warnings,
        issues=issues,
    )


def format_engine_status(status: EngineStatus) -> str:
    """Render engine status for CLI or GUI display."""

    lines = [
        f"Rust CLI: {status.binary_summary}",
        f"Binary Path: {status.binary_path or 'missing'}",
        f"Binary Freshness: {status.binary_freshness}",
        f"Bindings: {status.bindings_summary}",
    ]
    for command in CAPABILITY_ORDER:
        capability = status.capabilities.get(command)
        if capability is None:
            continue
        lines.append(f"{command}: {capability.state} ({capability.summary})")
    if status.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in status.warnings)
    if status.issues:
        lines.append("Issues:")
        lines.extend(f"  - {issue}" for issue in status.issues)
    return "\n".join(lines)


def translate_engine_output(output: str) -> str | None:
    """Convert known Rust/Python engine failures into clearer UI text."""

    stripped = output.strip()
    if not stripped:
        return None

    lowered = stripped.lower()

    if "unexpected argument '--no-stream' found" in lowered:
        return (
            "This engine build only accepts `--stream`. Disable streaming by omitting the flag, "
            "or rebuild the Python frontend and Rust CLI together."
        )

    if "rust engine cli not found" in lowered:
        return stripped

    if "training config, but end-to-end training is still unimplemented" in lowered:
        return (
            "Rust train is validate-only in this build. The corpus/config were accepted, "
            "but no training loop, checkpoints, or model files exist yet."
        )

    if "generation settings, but model/tokenizer loading and the real generation pipeline are still unimplemented" in lowered:
        return (
            "Rust generate is validate-only in this build. The prompt/model path were accepted, "
            "but executable inference is still unimplemented."
        )

    if "can validate quantization settings, but model loading and obm export are still unimplemented" in lowered:
        return (
            "Rust quantize is validate-only in this build. Settings were parsed, "
            "but no quantized OBM file can be written yet."
        )

    for prefix, label in (
        ("model file not found:", "Model file not found"),
        ("training corpus not found:", "Training corpus not found"),
        ("training data not found:", "Training corpus not found"),
        ("resume checkpoint not found:", "Resume checkpoint not found"),
        ("model config json not found:", "Model config JSON not found"),
        ("failed to load config:", "Model config failed to parse"),
        ("model config must be json", "Model config must be JSON"),
    ):
        if prefix in lowered:
            tail = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
            return f"{label}: {tail}"

    return None


def stream_engine_command(
    args: Sequence[str],
    *,
    explicit_binary: str | Path | None = None,
    cwd: str | Path | None = None,
    on_output: Callable[[str], None] | None = None,
) -> int:
    """Run a Rust engine command and stream merged stdout/stderr."""

    binary = resolve_engine_binary(explicit_binary)
    command = [str(binary), *args]
    callback = on_output or print
    callback(f"$ {shlex.join(command)}")

    process = subprocess.Popen(
        command,
        cwd=str(Path(cwd) if cwd else REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        callback(line.rstrip("\n"))

    return process.wait()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for engine utilities."""

    parser = argparse.ArgumentParser(description="Rust engine helpers for the OneBitLLM Python frontend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Print Rust CLI and Python binding status")
    status.add_argument("--engine", default=None, help="Path to the Rust CLI binary")

    resolve_cmd = subparsers.add_parser("resolve-binary", help="Resolve the Rust CLI binary path")
    resolve_cmd.add_argument("--engine", default=None, help="Path to the Rust CLI binary")

    run = subparsers.add_parser("run", help="Run a Rust command and stream its output")
    run.add_argument("--engine", default=None, help="Path to the Rust CLI binary")
    run.add_argument("rust_args", nargs=argparse.REMAINDER, help="Arguments passed directly to the Rust CLI")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for engine helpers."""

    args = build_parser().parse_args(argv)

    if args.command == "status":
        status = detect_engine_status(args.engine)
        print(format_engine_status(status))
        return 0

    if args.command == "resolve-binary":
        print(resolve_engine_binary(args.engine))
        return 0

    if args.command == "run":
        rust_args = list(args.rust_args)
        if rust_args and rust_args[0] == "--":
            rust_args = rust_args[1:]
        if not rust_args:
            raise SystemExit("engine_client.py run requires Rust CLI arguments")
        return stream_engine_command(rust_args, explicit_binary=args.engine)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
