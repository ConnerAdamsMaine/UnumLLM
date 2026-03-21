#!/usr/bin/env python3
"""Install or materialize optional datasets into a local ignored cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import request as urllib_request

DEFAULT_DATASET_ROOT = Path("dataset")

REDPAJAMA_SENTINEL = "redpajama"
REDPAJAMA_PARTITION = "head_middle"
REDPAJAMA_SNAPSHOTS = ("2023-06", "2022-49")
REDPAJAMA_LANGUAGES = ("en", "de")
REDPAJAMA_TEXT_FIELDS = ("raw_content", "text", "content")
REDPAJAMA_URL_BASE = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
REDPAJAMA_NUM_SHARDS = 5000

ULTRACHAT_DOWNLOADS = (
    (
        "questions_about_the_world.jsonl",
        "https://cloud.tsinghua.edu.cn/f/0a27393192ad46a5a081/?dl=1",
    ),
    (
        "writing_and_creation_part_1.jsonl",
        "https://cloud.tsinghua.edu.cn/f/57258a87846243218a9b/?dl=1",
    ),
    (
        "writing_and_creation_part_2.jsonl",
        "https://cloud.tsinghua.edu.cn/f/099b4dd71b82448fb7fb/?dl=1",
    ),
    (
        "assistance_on_existent_materials.jsonl",
        "https://cloud.tsinghua.edu.cn/f/1f7abdf2d2564cb4b338/?dl=1",
    ),
)

DATASET_DESCRIPTIONS = {
    "oasst1": "Export OpenAssistant/oasst1 chat trees into JSONL files for local training.",
    "redpajama": "Materialize a RedPajama text corpus into local files for OneBitLLM training.",
    "ultrachat": "Download the released UltraChat JSONL parts into the local dataset cache.",
    "the-pile": "Clone the upstream The Pile replication repo into the local dataset cache.",
    "fineweb-2": "Clone the upstream FineWeb 2 pipeline repo into the local dataset cache.",
}

CLONE_TARGETS = {
    "the-pile": {
        "url": "https://github.com/EleutherAI/The-Pile.git",
        "directory": "The-Pile",
    },
    "fineweb-2": {
        "url": "https://github.com/huggingface/fineweb-2.git",
        "directory": "FineWeb-2",
    },
}


def _parse_csv(values: Iterable[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return default

    parsed: list[str] = []
    for value in values:
        for item in value.split(","):
            cleaned = item.strip()
            if cleaned:
                parsed.append(cleaned)
    return tuple(parsed) or default


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_to_path(url: str, destination: Path, *, force: bool = False, timeout: int = 60) -> Path:
    if destination.exists() and not force:
        print(f"Reusing {destination}")
        return destination

    _ensure_parent(destination)
    request = urllib_request.Request(url, headers={"User-Agent": "OneBitLLM dataset installer"})
    print(f"Downloading {url} -> {destination}")
    with urllib_request.urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _load_huggingface_dataset():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - import depends on local env
        raise SystemExit(
            "This dataset installer needs the optional `datasets` package.\n"
            "Install it with: python -m pip install datasets"
        ) from exc
    return load_dataset


def normalize_oasst1_role(role: str) -> str | None:
    if role == "prompter":
        return "user"
    if role == "assistant":
        return "assistant"
    return None


def build_oasst1_index(split: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    nodes: dict[str, dict[str, Any]] = {}
    children: dict[str, list[str]] = defaultdict(list)

    for row in split:
        message_id = row["message_id"]
        parent_id = row["parent_id"]

        nodes[message_id] = {
            "message_id": message_id,
            "parent_id": parent_id,
            "role": row["role"],
            "text": row["text"],
            "lang": row.get("lang"),
        }

        if parent_id is not None:
            children[parent_id].append(message_id)

    return nodes, children


def extract_oasst1_paths(
    root_id: str,
    nodes: dict[str, dict[str, Any]],
    children: dict[str, list[str]],
) -> list[list[dict[str, Any]]]:
    paths: list[list[dict[str, Any]]] = []

    def dfs(message_id: str, path: list[dict[str, Any]]) -> None:
        node = nodes[message_id]
        path.append(node)

        child_ids = children.get(message_id, [])
        if not child_ids:
            paths.append(path.copy())
        else:
            for child_id in child_ids:
                dfs(child_id, path)

        path.pop()

    dfs(root_id, [])
    return paths


def oasst1_path_to_messages(path: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    for node in path:
        role = normalize_oasst1_role(node["role"])
        text = node["text"]

        if role is None or not isinstance(text, str):
            continue

        content = text.strip()
        if not content:
            continue

        messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return []

    has_user = any(message["role"] == "user" for message in messages)
    has_assistant = any(message["role"] == "assistant" for message in messages)
    if not has_user or not has_assistant:
        return []

    return messages


def export_oasst1(output_dir: str | Path, *, force: bool = False) -> list[Path]:
    load_dataset = _load_huggingface_dataset()
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    print("Loading OpenAssistant/oasst1 from Hugging Face")
    dataset = load_dataset("OpenAssistant/oasst1")

    written_paths: list[Path] = []
    for split_name, split in dataset.items():
        output_path = destination / f"{split_name}.jsonl"
        if output_path.exists() and not force:
            print(f"Reusing {output_path}")
            written_paths.append(output_path)
            continue

        print(f"Exporting {split_name} conversations to {output_path}")
        nodes, children = build_oasst1_index(split)
        root_ids = [message_id for message_id, node in nodes.items() if node["parent_id"] is None]

        written = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for root_id in root_ids:
                for path in extract_oasst1_paths(root_id, nodes, children):
                    messages = oasst1_path_to_messages(path)
                    if not messages:
                        continue
                    handle.write(json.dumps({"messages": messages}, ensure_ascii=False))
                    handle.write("\n")
                    written += 1

        print(f"Wrote {written} conversations to {output_path}")
        written_paths.append(output_path)

    return written_paths


def _extract_redpajama_text(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    for field in REDPAJAMA_TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _iter_redpajama_document_urls(
    *,
    partition: str,
    snapshots: tuple[str, ...],
    languages: tuple[str, ...],
    max_shards: int | None = None,
) -> list[str]:
    partitions = ("head", "middle") if partition == "head_middle" else (partition,)
    urls: list[str] = []
    for language in languages:
        for snapshot in snapshots:
            for partition_name in partitions:
                for shard_idx in range(REDPAJAMA_NUM_SHARDS):
                    if max_shards is not None and len(urls) >= max_shards:
                        return urls
                    base_tag = f"{snapshot}/{shard_idx:04d}/{language}_{partition_name}"
                    urls.append(f"{REDPAJAMA_URL_BASE}/documents/{base_tag}.json.gz")
    return urls


def materialize_redpajama_corpus(
    output_dir: str | Path,
    split_name: str,
    *,
    partition: str = REDPAJAMA_PARTITION,
    snapshots: tuple[str, ...] = REDPAJAMA_SNAPSHOTS,
    languages: tuple[str, ...] = REDPAJAMA_LANGUAGES,
    max_shards: int | None = None,
    max_docs: int | None = None,
    force: bool = False,
) -> Path:
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_path = output_root / f"{split_name}.txt"

    if corpus_path.is_file() and corpus_path.stat().st_size > 0 and not force:
        print(f"Reusing materialized RedPajama corpus at {corpus_path}")
        return corpus_path

    load_dataset = None
    try:
        from datasets import load_dataset as datasets_load_dataset  # type: ignore
    except ImportError:
        datasets_load_dataset = None
    else:
        load_dataset = datasets_load_dataset

    print(
        "Preparing RedPajama corpus with "
        f"partition={partition}, snapshots={list(snapshots)}, languages={list(languages)}"
    )

    if load_dataset is not None:
        try:
            dataset = load_dataset(
                "togethercomputer/RedPajama-Data-V2",
                name="default",
                partition=partition,
                snapshots=list(snapshots),
                languages=list(languages),
            )
            records = dataset.get("train") if isinstance(dataset, dict) else dataset
            if records is None:
                raise SystemExit("RedPajama loader did not return a usable dataset split.")

            written = 0
            with corpus_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    text = _extract_redpajama_text(record)
                    if not text:
                        continue
                    handle.write(text)
                    handle.write("\n")
                    written += 1
                    if max_docs is not None and written >= max_docs:
                        break

            if written == 0:
                raise SystemExit(
                    "RedPajama dataset loaded, but no text records were found in fields "
                    f"{', '.join(REDPAJAMA_TEXT_FIELDS)}."
                )

            print(f"Materialized {written} RedPajama documents to {corpus_path}")
            return corpus_path
        except RuntimeError as exc:
            if "dataset scripts are no longer supported" not in str(exc).lower():
                raise
            print(
                "Installed `datasets` cannot execute the RedPajama loader script. "
                "Falling back to direct shard download."
            )

    urls = _iter_redpajama_document_urls(
        partition=partition,
        snapshots=snapshots,
        languages=languages,
        max_shards=max_shards,
    )
    docs_written = 0
    files_downloaded = 0
    files_missing = 0

    print(
        "Streaming RedPajama document shards directly from Together storage. "
        f"Candidate shard files: {len(urls)}"
    )

    with corpus_path.open("w", encoding="utf-8") as handle:
        for idx, url in enumerate(urls, start=1):
            try:
                response = urllib_request.urlopen(url, timeout=60)
            except urllib_error.HTTPError as exc:
                if exc.code == 404:
                    files_missing += 1
                    continue
                raise SystemExit(f"Failed downloading {url}: HTTP {exc.code}") from exc
            except urllib_error.URLError as exc:
                raise SystemExit(f"Failed downloading {url}: {exc.reason}") from exc

            files_downloaded += 1
            with response:
                with gzip.GzipFile(fileobj=response) as gz_file:
                    with io.TextIOWrapper(gz_file, encoding="utf-8") as reader:
                        for line in reader:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            text = _extract_redpajama_text(record)
                            if not text:
                                continue
                            handle.write(text)
                            handle.write("\n")
                            docs_written += 1
                            if max_docs is not None and docs_written >= max_docs:
                                print(
                                    f"Reached max_docs={max_docs}; wrote {docs_written} documents to {corpus_path}"
                                )
                                return corpus_path

            if files_downloaded % 25 == 0:
                print(
                    f"Processed {files_downloaded} RedPajama shard files "
                    f"({files_missing} missing, {docs_written} docs written, last url {idx}/{len(urls)})"
                )

    if docs_written == 0:
        raise SystemExit(
            "RedPajama direct download completed, but no text records were written. "
            "If this machine has a modern `datasets` install and outbound network restrictions, "
            "either allow access to `data.together.xyz` or install `datasets<4`."
        )

    print(
        f"Materialized {docs_written} RedPajama documents from {files_downloaded} shard files "
        f"to {corpus_path}"
    )
    return corpus_path


def download_ultrachat(
    output_dir: str | Path,
    *,
    force: bool = False,
    merge_output: str | Path | None = None,
) -> list[Path]:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for filename, url in ULTRACHAT_DOWNLOADS:
        downloaded.append(_download_to_path(url, destination / filename, force=force))

    if merge_output:
        merge_path = Path(merge_output).expanduser()
        _ensure_parent(merge_path)
        with merge_path.open("w", encoding="utf-8") as out_handle:
            for part in downloaded:
                with part.open("r", encoding="utf-8") as in_handle:
                    shutil.copyfileobj(in_handle, out_handle)
        print(f"Merged UltraChat parts into {merge_path}")

    return downloaded


def clone_dataset_repo(
    dataset_name: str,
    output_root: str | Path,
    *,
    force: bool = False,
    ref: str | None = None,
) -> Path:
    spec = CLONE_TARGETS[dataset_name]
    destination = Path(output_root).expanduser() / spec["directory"]

    if destination.exists():
        if not force:
            print(f"Reusing existing clone at {destination}")
            return destination
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("git") is None:
        raise SystemExit("`git` is required to install this dataset source.")

    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd.extend(["--branch", ref, "--single-branch"])
    cmd.extend([spec["url"], str(destination)])
    print(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return destination


def install_dataset(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser()

    if args.dataset == "oasst1":
        export_paths = export_oasst1(root / "OASST1" / "oasst1_export", force=args.force)
        print("Installed OASST1:")
        for path in export_paths:
            print(path)
        return 0

    if args.dataset == "redpajama":
        partition = args.partition or REDPAJAMA_PARTITION
        snapshots = _parse_csv(args.snapshot, REDPAJAMA_SNAPSHOTS)
        languages = _parse_csv(args.language, REDPAJAMA_LANGUAGES)
        output = materialize_redpajama_corpus(
            root / "RedPajama",
            "train",
            partition=partition,
            snapshots=snapshots,
            languages=languages,
            max_shards=args.max_shards,
            max_docs=args.max_docs,
            force=args.force,
        )
        print(f"Installed RedPajama corpus at {output}")
        return 0

    if args.dataset == "ultrachat":
        merge_output = None
        if args.merge_output:
            merge_output = Path(args.merge_output).expanduser()
        download_paths = download_ultrachat(
            root / "UltraChat" / "raw",
            force=args.force,
            merge_output=merge_output,
        )
        print("Installed UltraChat parts:")
        for path in download_paths:
            print(path)
        return 0

    if args.dataset in CLONE_TARGETS:
        path = clone_dataset_repo(args.dataset, root, force=args.force, ref=args.repo_ref)
        print(f"Installed {args.dataset} source repo at {path}")
        return 0

    raise SystemExit(f"Unsupported dataset: {args.dataset}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install optional datasets into the local OneBitLLM dataset cache.",
    )
    parser.add_argument("dataset", nargs="?", choices=sorted(DATASET_DESCRIPTIONS))
    parser.add_argument("--root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--force", action="store_true", help="Overwrite existing local dataset files")
    parser.add_argument("--list", action="store_true", help="List supported datasets and exit")

    parser.add_argument(
        "--partition",
        default=None,
        help="RedPajama partition to materialize (default: head_middle)",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=None,
        help="RedPajama snapshot(s) to include; repeat or pass comma-separated values",
    )
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        help="RedPajama language(s) to include; repeat or pass comma-separated values",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Cap the number of RedPajama shard files fetched for smaller local runs",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Cap the number of RedPajama documents written",
    )
    parser.add_argument(
        "--merge-output",
        default=None,
        help="Optional merged JSONL output path for UltraChat downloads",
    )
    parser.add_argument(
        "--repo-ref",
        default=None,
        help="Optional git ref to clone for source-repo based datasets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(DATASET_DESCRIPTIONS):
            print(f"{name}: {DATASET_DESCRIPTIONS[name]}")
        return 0

    if not args.dataset:
        parser.error("dataset is required unless --list is used")

    return install_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
