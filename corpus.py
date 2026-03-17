#!/usr/bin/env python3
"""Corpus download helpers for the Python frontend."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class CorpusPreset:
    name: str
    url: str
    default_destination: str
    description: str


CORPUS_PRESETS: dict[str, CorpusPreset] = {
    "tinyshakespeare": CorpusPreset(
        name="tinyshakespeare",
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        default_destination="corpora/tinyshakespeare.txt",
        description="Compact Shakespeare corpus used in many language-model demos.",
    ),
}


def resolve_download_url(preset_name: str | None, custom_url: str | None) -> str:
    if custom_url:
        return custom_url.strip()
    if preset_name and preset_name in CORPUS_PRESETS:
        return CORPUS_PRESETS[preset_name].url
    raise ValueError("Provide either a corpus preset or a custom URL.")


def default_destination_for_preset(preset_name: str) -> str:
    return CORPUS_PRESETS[preset_name].default_destination


def download_corpus(url: str, destination: str | Path, timeout: int = 60) -> Path:
    request = Request(url, headers={"User-Agent": "OneBitLLM Python Frontend"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()

    output = Path(destination).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a training corpus for OneBitLLM")
    parser.add_argument("--preset", choices=sorted(CORPUS_PRESETS), default="tinyshakespeare")
    parser.add_argument("--url", default=None, help="Override the preset URL")
    parser.add_argument("--output", default=None, help="Destination path")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--list-presets", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_presets:
        for preset in CORPUS_PRESETS.values():
            print(f"{preset.name}: {preset.description}")
            print(f"  url: {preset.url}")
            print(f"  default output: {preset.default_destination}")
        return 0

    destination = args.output or default_destination_for_preset(args.preset)
    url = resolve_download_url(args.preset, args.url)
    path = download_corpus(url, destination, timeout=args.timeout)
    print(f"Downloaded corpus to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
