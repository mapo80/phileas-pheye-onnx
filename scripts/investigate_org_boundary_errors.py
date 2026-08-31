#!/usr/bin/env python3
"""Why mmBERT's ORG exact-match F1 is low: quantifies the boundary-error mechanisms.

Reproduces the breakdown behind the README's "Why ORG's exact-match F1 is low" section. For
every gold ORG span the model does not match exactly, classifies the miss by comparing the
model's raw (pre-threshold) predictions and the reduced span against the gold span:

  trailing_period_fused   predicted span == gold span minus a trailing '.' (or run of them)
  leading_word_dropped    predicted span == gold span minus a leading word
  extra_context_included  predicted span strictly contains the gold span (or vice versa isn't
                           possible here since this branch is only reached on a mismatch)
  below_threshold_only    a raw prediction covers the gold span but every overlapping score is
                           under the decode threshold
  other                   anything not matching the above (used for manual inspection)

    python scripts/investigate_org_boundary_errors.py \
        --gold GOLD_DIR/test.jsonl \
        --mmbert-model PACKAGED_TOKEN_CLASSIFICATION_MODEL_DIR \
        --mmbert-weights HF_CHECKPOINT_DIR \
        --label ORG --threshold 0.98

For a model that exists only as a quantized ONNX export, with no torch checkpoint, pass
--mmbert-onnx instead of --mmbert-weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_token_classification_fixture import chunk_text, reduce_spans, onnx_scorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--mmbert-model", type=Path, required=True)
    parser.add_argument("--mmbert-weights", type=Path, default=None,
                        help="checkpoint holding the torch weights; required unless --mmbert-onnx is given")
    parser.add_argument("--mmbert-onnx", action="store_true",
                        help="drive the packaged model's own ONNX graph directly via onnxruntime")
    parser.add_argument("--label", default="ORG")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--show", type=int, default=6, help="example rows to print per bucket")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mmbert_onnx and args.mmbert_weights is None:
        raise SystemExit("--mmbert-weights is required unless --mmbert-onnx is given")

    window = json.loads((args.mmbert_model / "token_classification_config.json").read_text())
    max_words, overlap = window["max_words"], window["overlap_words"]

    if args.mmbert_onnx:
        score_chunk = onnx_scorer(args.mmbert_model, int(window["max_tokens"]))
    else:
        from transformers import AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(str(args.mmbert_model))
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]
        nlp = pipeline("token-classification", model=str(args.mmbert_weights), tokenizer=tokenizer,
                       aggregation_strategy="simple", device=-1)
        score_chunk = nlp

    documents = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]

    buckets: dict[str, list[tuple[str, str]]] = {
        "trailing_period_fused": [], "leading_word_dropped": [],
        "extra_context_included": [], "below_threshold_only": [], "other": [],
    }
    exact = 0
    total_gold = 0

    for document in documents:

        text = document["source_text"]
        gold = [(e["start"], e["end"], text[e["start"]:e["end"]])
                for e in document["entities"] if e["label"] == args.label]
        if not gold:
            continue
        total_gold += len(gold)

        raw = []
        chunks = chunk_text(text, max_words, overlap)
        for chunk, offset in chunks:
            for entity in score_chunk(chunk):
                if entity["entity_group"] != args.label:
                    continue
                raw.append({"label": args.label, "start": int(entity["start"]) + offset,
                            "end": int(entity["end"]) + offset, "score": float(entity["score"])})

        kept = [s for s in raw if s["score"] >= args.threshold]
        predicted = reduce_spans(kept, text)
        predicted_set = {(s["start"], s["end"]) for s in predicted}

        for start, end, value in gold:

            if (start, end) in predicted_set:
                exact += 1
                continue

            overlapping = [p for p in predicted if p["start"] < end and p["end"] > start]
            raw_overlapping = [r for r in raw if r["start"] < end and r["end"] > start]

            if not overlapping:
                if raw_overlapping and all(r["score"] < args.threshold for r in raw_overlapping):
                    buckets["below_threshold_only"].append(
                        (value, str([round(r["score"], 3) for r in raw_overlapping])))
                else:
                    buckets["other"].append((value, "no overlapping prediction at all"))
                continue

            candidate = overlapping[0]

            if candidate["start"] == start and candidate["end"] < end:
                missing = text[candidate["end"]:end]
                if missing and missing.strip(".") == "":
                    buckets["trailing_period_fused"].append((value, repr(missing)))
                    continue

            if candidate["start"] > start and candidate["end"] == end:
                buckets["leading_word_dropped"].append((value, repr(text[start:candidate["start"]])))
                continue

            if candidate["start"] <= start and candidate["end"] >= end:
                buckets["extra_context_included"].append((value, repr(text[candidate["start"]:candidate["end"]])))
                continue

            buckets["other"].append((value, repr(text[candidate["start"]:candidate["end"]])))

    mismatches = sum(len(v) for v in buckets.values())
    print(f"gold {args.label} spans: {total_gold} | exact: {exact} | mismatched: {mismatches}\n")
    for name, rows in buckets.items():
        pct = 100 * len(rows) / mismatches if mismatches else 0
        print(f"{name:24s} {len(rows):4d}  ({pct:4.1f}% of mismatches)")
    print()
    for name, rows in buckets.items():
        if rows:
            print(f"--- {name} ---")
            for value, note in rows[:args.show]:
                print(f"   {value!r:35s} {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
