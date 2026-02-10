#!/usr/bin/env python3
"""
Maintainer-only helper for regenerating golden conformance vectors.

This script is optional and not used by package runtime/tests in CI.
It depends on Python tokenizer stacks and writes JSON goldens consumed by Julia tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List


def _read_strings(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        lines.append(raw)
    return lines


def _build_hf_encoder(model_id: str) -> Callable[[str], List[int]]:
    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return lambda text: tok.encode(text, add_special_tokens=False)


def _build_tiktoken_encoder(name: str) -> Callable[[str], List[int]]:
    import tiktoken  # type: ignore

    enc = tiktoken.get_encoding(name)
    return lambda text: enc.encode(text)


def _render_spec(
    name: str,
    fmt: str,
    source: Dict[str, str],
    strings: List[str],
    encoder: Callable[[str], List[int]],
) -> Dict[str, object]:
    return {
        "name": name,
        "format": fmt,
        "source": source,
        "settings": {"add_special_tokens": False},
        "cases": [
            {
                "text": text,
                "expected_ids": encoder(text),
            }
            for text in strings
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tokenizer golden vectors")
    parser.add_argument(
        "--strings",
        default=str(Path(__file__).resolve().parents[2] / "test" / "golden" / "strings.txt"),
        help="Path to shared input strings file",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "test" / "golden"),
        help="Output directory for generated JSON specs",
    )
    args = parser.parse_args()

    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    strings = _read_strings(Path(args.strings).resolve())

    jobs = [
        (
            "golden_tiktoken_cl100k_base",
            "tiktoken",
            {"kind": "key", "value": "tiktoken_cl100k_base"},
            _build_tiktoken_encoder("cl100k_base"),
        ),
        (
            "golden_hf_gpt2",
            "bpe_gpt2",
            {"kind": "key", "value": "openai_gpt2_bpe"},
            _build_hf_encoder("gpt2"),
        ),
        (
            "golden_hf_roberta_base",
            "bpe_gpt2",
            {"kind": "key", "value": "roberta_base_bpe"},
            _build_hf_encoder("roberta-base"),
        ),
        (
            "golden_hf_bert_base_uncased",
            "wordpiece_vocab",
            {"kind": "key", "value": "bert_base_uncased_wordpiece"},
            _build_hf_encoder("bert-base-uncased"),
        ),
        (
            "golden_hf_xlm_roberta_base",
            "sentencepiece_model",
            {"kind": "key", "value": "xlm_roberta_base_sentencepiece_bpe"},
            _build_hf_encoder("xlm-roberta-base"),
        ),
    ]

    for file_stem, fmt, source, encoder in jobs:
        spec = _render_spec(file_stem, fmt, source, strings, encoder)
        target = outdir / f"{file_stem}.json"
        target.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
