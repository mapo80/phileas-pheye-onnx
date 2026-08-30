#!/usr/bin/env python3
"""Emit the reference output of the Python token-classification pipeline, for the Java parity test.

The reference is the pipeline the model ships with in production: HuggingFace
`token-classification` with `aggregation_strategy="simple"`, run over word windows of
`max_words` with `overlap_words` shared words, offsets mapped back to the original text.
Two span-level normalisations from the reference application are applied here as well, because
the Java detector applies them: surrounding whitespace is trimmed, and a span that cuts a word in
half is widened to cover it.

Java must reproduce these spans from the same model directory. Any drift is a bug in one of the two.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORD = re.compile(r"\S+")

# 0 exercises the decode with nothing filtered out; 0.92 is the value this model was
# calibrated at on the development split.
THRESHOLDS = (0.0, 0.92)


def chunk_text(text: str, max_words: int, overlap: int) -> list[tuple[str, int]]:
    """Word-safe windows as (substring, absolute char offset). Mirrors the reference app."""
    words = list(WORD.finditer(text))
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        block = words[i:i + max_words]
        start, end = block[0].start(), block[-1].end()
        chunks.append((text[start:end], start))
        if i + max_words >= len(words):
            break
        i += step
    return chunks


def is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def reduce_spans(spans: list[dict], text: str) -> list[dict]:
    """The reference application's span reduction, model candidates only.

    Greedy by score then length, whitespace trimmed, widened to word boundaries, then coalesced.
    """
    order = sorted(spans, key=lambda e: (e["score"], e["end"] - e["start"]), reverse=True)
    kept: list[dict] = []
    for span in order:
        if any(span["start"] < k["end"] and k["start"] < span["end"] for k in kept):
            continue
        kept.append(dict(span))

    trimmed = []
    for span in kept:
        start, end = span["start"], span["end"]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            continue
        while start > 0 and is_word_char(text[start - 1]) and is_word_char(text[start]):
            start -= 1
        while end < len(text) and is_word_char(text[end]) and is_word_char(text[end - 1]):
            end += 1
        trimmed.append({**span, "start": start, "end": end})

    trimmed.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    merged: list[dict] = []
    for span in trimmed:
        if merged and span["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
            continue
        if merged and span["start"] == merged[-1]["end"] and span["label"] == merged[-1]["label"]:
            merged[-1]["end"] = span["end"]
            continue
        merged.append(span)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None,
                        help="checkpoint holding the torch weights; defaults to --model-dir")
    parser.add_argument("--texts", type=Path, required=True, help="JSON list of {id, text}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    window = json.loads((args.model_dir / "token_classification_config.json").read_text(encoding="utf-8"))
    max_words, overlap = int(window["max_words"]), int(window["overlap_words"])

    from transformers import AutoTokenizer, pipeline

    weights = args.weights or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]
    nlp = pipeline("token-classification", model=str(weights), tokenizer=tokenizer,
                   aggregation_strategy="simple", device=-1)

    documents = json.loads(args.texts.read_text(encoding="utf-8"))
    results = []
    for document in documents:
        text = document["text"]
        chunks = chunk_text(text, max_words, overlap)
        raw = []
        if chunks:
            outputs = nlp([chunk for chunk, _ in chunks])
            if isinstance(outputs, dict):
                outputs = [outputs]
            for (_, offset), entities in zip(chunks, outputs):
                for entity in entities:
                    raw.append({
                        "label": entity["entity_group"],
                        "start": int(entity["start"]) + offset,
                        "end": int(entity["end"]) + offset,
                        "score": round(float(entity["score"]), 6),
                    })
        results.append({
            "id": document["id"],
            "text": text,
            "words": len(WORD.findall(text)),
            "windows": len(chunks),
            "raw": raw,
            "by_threshold": {
                f"{threshold:g}": reduce_spans(
                    [s for s in raw if s["score"] >= threshold], text)
                for threshold in THRESHOLDS
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "_note": "Reference output of the Python pipeline. Regenerate with "
                 "scripts/generate_token_classification_fixture.py.",
        "model": args.model_dir.name,
        "window": {"max_words": max_words, "overlap_words": overlap},
        "thresholds": [f"{t:g}" for t in THRESHOLDS],
        "documents": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(len(d["by_threshold"]["0"]) for d in results)
    print(f"{len(results)} documents, {total} spans -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
