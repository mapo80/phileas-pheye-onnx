#!/usr/bin/env python3
"""GLiNER vs mmBERT (token classification), matched protocol, same test split.

Reproduces the README's "GLiNER vs mmBERT" quality tables. Requires a Java build of this module
(`mvn -q test-compile dependency:build-classpath -Dmdep.outputFile=target/classpath.txt
-Dmdep.includeScope=test`, run once from the repository root) and the `transformers` Python package.

    python scripts/gliner_vs_mmbert_comparison.py \
        --gold GOLD_DIR/test.jsonl \
        --gliner-model GLINER_MODEL_DIR \
        --mmbert-model PACKAGED_TOKEN_CLASSIFICATION_MODEL_DIR \
        --mmbert-weights HF_CHECKPOINT_DIR \
        --output result.json

For a model that exists only as a quantized ONNX export, with no torch checkpoint, pass
--mmbert-onnx instead of --mmbert-weights.

Produces a fresh, apples-to-apples quality comparison on the finance-banking-gold test set for
the three canonical categories GLiNER's frozen production config supports (person, company,
postal address), scored exactly as the mmBERT student is scored elsewhere in this project:
entity-level exact character-span match.

Everything here is measured once, on this run, on this machine. No prior numbers are reused for
the fresh 3-category slice; the GLiNER config (labels, thresholds, decode strategy) is the exact
one already frozen for production by prior work in this workspace, run here against the CURRENT
local build of phileas-pheye-onnx rather than a previously published release.
"""
import base64
import json
import subprocess
import sys
from pathlib import Path
from collections import Counter

import argparse

HERE = Path(__file__).resolve().parent
PHILEAS_DIR = HERE.parent
sys.path.insert(0, str(HERE))

from generate_token_classification_fixture import chunk_text, reduce_spans, onnx_scorer  # noqa: E402

# GLiNER canonical label -> gold tag it stands in for (this project's own established mapping).
GLINER_TO_GOLD = {"person": "FULLNAME", "company": "ORG", "postal address": "STREET"}
CATEGORIES = ["FULLNAME", "ORG", "STREET"]


def load_docs():
    docs = []
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold_spans = {(e["label"], e["start"], e["end"]) for e in row["entities"] if e["label"] in CATEGORIES}
        docs.append({"id": row["id"], "text": row["source_text"], "gold": gold_spans})
    return docs


def run_gliner(docs, classpath):
    payload = "\n".join(
        json.dumps({"id": d["id"], "text_b64": base64.b64encode(d["text"].encode("utf-8")).decode("ascii")})
        for d in docs)
    result = subprocess.run(
        ["java", "-cp", classpath, "ai.philterd.phileas.pheye.onnx.GoldSpanDump", str(GLINER_MODEL)],
        input=payload, capture_output=True, text=True, cwd=str(PHILEAS_DIR))
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit("GoldSpanDump failed")
    predictions = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spans = set()
        for span in row["spans"]:
            gold_tag = GLINER_TO_GOLD.get(span["label"])
            if gold_tag:
                spans.add((gold_tag, span["start"], span["end"]))
        predictions[row["id"]] = spans
    return predictions


def run_mmbert(docs, threshold=0.98, use_onnx=False):
    """Score every document with the token-classification model.

    Two backends, sharing the same chunk_text/reduce_spans this project verifies Java against
    elsewhere: the HuggingFace pipeline (needs a loadable torch checkpoint) or onnxruntime direct
    against the packaged model's own graph (the only option for a model that exists solely as a
    quantized ONNX export, with no torch counterpart -- pass --mmbert-onnx for that case).
    """
    window = json.loads((MMBERT_MODEL / "token_classification_config.json").read_text())
    max_words, overlap = window["max_words"], window["overlap_words"]

    if use_onnx:
        score_chunk = onnx_scorer(MMBERT_MODEL, int(window["max_tokens"]))
    else:
        from transformers import AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(str(MMBERT_MODEL))
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]
        nlp = pipeline("token-classification", model=str(MMBERT_WEIGHTS), tokenizer=tokenizer,
                       aggregation_strategy="simple", device=-1)
        score_chunk = nlp

    predictions = {}
    for d in docs:
        text = d["text"]
        chunks = chunk_text(text, max_words, overlap)
        raw = []
        for chunk, offset in chunks:
            for entity in score_chunk(chunk):
                if entity["entity_group"] not in CATEGORIES:
                    continue
                score = float(entity["score"])
                if score >= threshold:
                    raw.append({"label": entity["entity_group"], "start": int(entity["start"]) + offset,
                                "end": int(entity["end"]) + offset, "score": score})
        predictions[d["id"]] = {(s["label"], s["start"], s["end"]) for s in reduce_spans(raw, text)}
    return predictions


def score(docs, predictions, label):
    tp = fp = fn = 0
    per_tag = {c: Counter() for c in CATEGORIES}
    for d in docs:
        gold = d["gold"]
        pred = predictions.get(d["id"], set())
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for tag in CATEGORIES:
            g = {s for s in gold if s[0] == tag}
            p = {s for s in pred if s[0] == tag}
            per_tag[tag]["tp"] += len(g & p)
            per_tag[tag]["fp"] += len(p - g)
            per_tag[tag]["fn"] += len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\n=== {label}: micro over {CATEGORIES} ===")
    print(f"  P={precision:.4f} R={recall:.4f} F1={f1:.4f}  (tp={tp} fp={fp} fn={fn})")
    for tag in CATEGORIES:
        c = per_tag[tag]
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"    {tag:10s} P={p:.4f} R={r:.4f} F1={f:.4f}  tp={c['tp']} fp={c['fp']} fn={c['fn']}")
    return {"precision": precision, "recall": recall, "f1": f1,
            "per_tag": {t: {"tp": per_tag[t]["tp"], "fp": per_tag[t]["fp"], "fn": per_tag[t]["fn"]} for t in CATEGORIES}}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, required=True,
                        help="finance-banking-gold test.jsonl (or any file with the same schema)")
    parser.add_argument("--gliner-model", type=Path, required=True, help="GLiNER model directory")
    parser.add_argument("--mmbert-model", type=Path, required=True,
                        help="packaged token-classification model directory")
    parser.add_argument("--mmbert-weights", type=Path, default=None,
                        help="the HuggingFace checkpoint holding the torch weights; required unless "
                             "--mmbert-onnx is given")
    parser.add_argument("--mmbert-onnx", action="store_true",
                        help="drive the packaged model's own ONNX graph directly via onnxruntime "
                             "instead of a torch checkpoint -- the only option for a model that "
                             "exists solely as a quantized ONNX export")
    parser.add_argument("--mmbert-threshold", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=None,
                        help="optional path to write the full result as JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.mmbert_onnx and args.mmbert_weights is None:
        raise SystemExit("--mmbert-weights is required unless --mmbert-onnx is given")
    global GOLD, GLINER_MODEL, MMBERT_MODEL, MMBERT_WEIGHTS
    GOLD, GLINER_MODEL = args.gold, args.gliner_model
    MMBERT_MODEL, MMBERT_WEIGHTS = args.mmbert_model, args.mmbert_weights

    docs = load_docs()
    print(f"test documents: {len(docs)}")
    total_gold = sum(len(d["gold"]) for d in docs)
    print(f"gold entities in {CATEGORIES}: {total_gold}")

    classpath = "target/test-classes:target/classes:" + (PHILEAS_DIR / "target" / "classpath.txt").read_text().strip()

    gliner_predictions = run_gliner(docs, classpath)
    mmbert_predictions = run_mmbert(docs, threshold=args.mmbert_threshold, use_onnx=args.mmbert_onnx)

    gliner_result = score(docs, gliner_predictions, "GLiNER (person/company/postal address, thr 0.60/0.40, FLAT_GREEDY)")
    mmbert_result = score(docs, mmbert_predictions, f"mmBERT (FULLNAME/ORG/STREET only, thr {args.mmbert_threshold})")

    out = {
        "documents": len(docs),
        "gold_entities": total_gold,
        "gliner": gliner_result,
        "mmbert_same_3_categories": mmbert_result,
    }
    if args.output:
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nfull result -> {args.output}")


if __name__ == "__main__":
    main()
