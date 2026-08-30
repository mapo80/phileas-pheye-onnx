#!/usr/bin/env python3
"""Package a HuggingFace token-classification checkpoint as a model directory this module can drive.

The module reads a directory, never a HuggingFace repo id, so a checkpoint has to be laid out as:

    <modelDir>/
      onnx/model.onnx                     the exported graph  (input_ids, attention_mask -> logits)
      tokenizer.json                      the fast tokenizer
      config.json                         id2label, in BIO notation
      token_classification_config.json    the window this model was trained for

`token_classification_config.json` is what the detector needs and the HuggingFace config does not
record: the window in words the model is run with, and the threshold it was calibrated at.

`max_tokens` is the encoder's hard sub-token capacity, not the length it was distilled at. The two
are different numbers and only the first belongs here: the detector splits a window at `max_tokens`
because past it the graph cannot run at all, and splitting earlier than the reference pipeline does
would move the operating point the threshold was calibrated on. The distillation length is recorded
alongside as `trained_tokens`, for the record. `config.json` advertises the architecture's
positional limit (8192 for ModernBERT), which is not the length the weights were trained at, and
windowing a redaction model at the wrong length degrades it silently. The value is read here from
the training manifest instead of being guessed.

Usage:

    python scripts/package_token_classification_model.py \
        --checkpoint  /path/to/rizzo-pii-student-6x384g-v1.2.0 \
        --onnx        /path/to/6x384g-v3.onnx \
        --output      /path/to/model-dir \
        --max-words 120 --overlap-words 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="HuggingFace checkpoint directory (config.json, tokenizer.json, ...)")
    parser.add_argument("--onnx", type=Path, required=True, help="the exported ONNX graph")
    parser.add_argument("--output", type=Path, required=True, help="model directory to write")
    parser.add_argument("--max-words", type=int, default=120,
                        help="words per inference window (default: 120)")
    parser.add_argument("--overlap-words", type=int, default=20,
                        help="words shared between consecutive windows (default: 20)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="hard sub-token capacity; default: max_position_embeddings from config.json")
    parser.add_argument("--calibrated-threshold", type=float, default=None,
                        help="the decode threshold this model was calibrated at on a development split")
    args = parser.parse_args()

    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label") or {}
    if not id2label:
        raise SystemExit(f"{args.checkpoint}/config.json declares no id2label")

    max_tokens = args.max_tokens
    if max_tokens is None:
        if "max_position_embeddings" not in config:
            raise SystemExit("no --max-tokens given and config.json has no max_position_embeddings")
        max_tokens = int(config["max_position_embeddings"])

    trained_tokens = None
    manifest = args.checkpoint / "student_train.json"
    if manifest.is_file():
        trained_tokens = int(json.loads(manifest.read_text(encoding="utf-8"))["config"]["transfer"]["max_length"])

    if args.overlap_words >= args.max_words:
        raise SystemExit("--overlap-words must be smaller than --max-words")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "onnx").mkdir(exist_ok=True)

    shutil.copyfile(args.checkpoint / "config.json", args.output / "config.json")
    shutil.copyfile(args.checkpoint / "tokenizer.json", args.output / "tokenizer.json")
    if (args.checkpoint / "tokenizer_config.json").is_file():
        shutil.copyfile(args.checkpoint / "tokenizer_config.json", args.output / "tokenizer_config.json")
    shutil.copyfile(args.onnx, args.output / "onnx" / "model.onnx")

    window = {
        "_note": "Read by LocalTokenClassifierDetector. max_words/overlap_words are the window the "
                 "model is run with; max_tokens is the encoder's hard sub-token capacity, past "
                 "which the graph cannot run. trained_tokens is informational.",
        "max_words": args.max_words,
        "overlap_words": args.overlap_words,
        "max_tokens": max_tokens,
        "words_splitter_type": "whitespace",
        "aggregation": "simple",
    }
    if trained_tokens is not None:
        window["trained_tokens"] = trained_tokens
    if args.calibrated_threshold is not None:
        if not 0.0 <= args.calibrated_threshold <= 1.0:
            raise SystemExit("--calibrated-threshold must be within [0, 1]")
        window["calibrated_threshold"] = args.calibrated_threshold
    (args.output / "token_classification_config.json").write_text(
        json.dumps(window, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "checkpoint": str(args.checkpoint),
        "onnx_source": str(args.onnx),
        "onnx_sha256": sha256(args.output / "onnx" / "model.onnx"),
        "labels": len(id2label),
    }
    for name in ("student_build.json", "student_fidelity.json"):
        path = args.checkpoint / name
        if path.is_file():
            provenance[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"model directory -> {args.output}")
    print(f"  labels {len(id2label)}, window {args.max_words} words / {args.overlap_words} overlap, "
          f"capacity {max_tokens} sub-tokens, "
          f"calibrated threshold {args.calibrated_threshold if args.calibrated_threshold is not None else 'not declared'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
