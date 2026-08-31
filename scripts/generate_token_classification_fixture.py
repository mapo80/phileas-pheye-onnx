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

# 0 exercises the decode with nothing filtered out. The second value is only a generic
# historical default; pass --threshold for the model actually being fixtured, since a
# calibrated threshold is a property of the model, not of this script.
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


def onnx_scorer(model_dir: Path, max_tokens: int):
    """A scorer function equivalent to the HuggingFace pipeline, driven by onnxruntime directly.

    For a model that exists only as a quantized ONNX graph -- no loadable PyTorch checkpoint --
    `transformers.pipeline` cannot be used at all. This reimplements exactly what it does for
    `aggregation_strategy="simple"` (per-subtoken softmax + argmax, then group consecutive
    subtokens while the type is unchanged and the tag is not a fresh `B-`, entity score = mean of
    member probabilities) directly against the graph, matching `LocalTokenClassifierDetector`'s
    own algorithm. Cross-checked equivalent to the pipeline-based path this replaces: on
    `rizzo-pii-student-6x384g`, Java driven by `LocalTokenClassifierDetector` matched this
    onnxruntime-direct reference exactly on the same corpus that it matched the pipeline-based
    reference on -- so by transitivity this reproduces `aggregation_strategy="simple"` exactly.

    Falls back to the sub-token windowing `LocalTokenClassifierDetector` uses when a chunk's own
    encoding exceeds `max_tokens` (a chunk holding one pathologically long "word").
    """
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    onnx_path = model_dir / "onnx" / "model.onnx"
    if not onnx_path.is_file():
        onnx_path = model_dir / "model.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def softmax(x: "np.ndarray") -> "np.ndarray":
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    def score_sub_window(ids: "np.ndarray", encoding, from_: int, to: int) -> list[dict]:
        sub_ids = ids[from_:to]
        mask = np.ones_like(sub_ids)
        logits = session.run(["logits"], {
            "input_ids": sub_ids.reshape(1, -1),
            "attention_mask": mask.reshape(1, -1),
        })[0][0]
        probabilities = softmax(logits)

        entities = []
        open_type = open_start = open_end = None
        open_total = 0.0
        open_count = 0
        for t in range(to - from_):
            absolute = from_ + t
            if encoding.special_tokens_mask[absolute] == 1:
                continue
            char_span = encoding.offsets[absolute]
            if char_span is None or char_span[0] < 0:
                continue
            best = int(probabilities[t].argmax())
            probability = float(probabilities[t][best])
            label = id2label[best]
            entity_type = "O" if label == "O" else label[2:]
            continues = (entity_type == open_type) and not label.startswith("B-")
            if open_type is not None and not continues:
                if open_type != "O":
                    entities.append({"entity_group": open_type, "start": open_start, "end": open_end,
                                     "score": open_total / open_count})
                open_type = None
            if open_type is None:
                open_type, open_start, open_total, open_count = entity_type, char_span[0], 0.0, 0
            open_end = char_span[1]
            open_total += probability
            open_count += 1
        if open_type is not None and open_type != "O":
            entities.append({"entity_group": open_type, "start": open_start, "end": open_end,
                             "score": open_total / open_count})
        return entities

    def score(chunk: str) -> list[dict]:
        encoding = tokenizer.encode(chunk)
        ids = np.array(encoding.ids, dtype=np.int64)
        length = len(ids)
        if length <= max_tokens:
            return score_sub_window(ids, encoding, 0, length)
        entities = []
        stride = max(max_tokens - max(max_tokens // 8, 1), 1)
        for start in range(0, length, stride):
            end = min(start + max_tokens, length)
            entities.extend(score_sub_window(ids, encoding, start, end))
            if end == length:
                break
        return entities

    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None,
                        help="checkpoint holding the torch weights; defaults to --model-dir. "
                             "Mutually exclusive with --onnx.")
    parser.add_argument("--onnx", action="store_true",
                        help="drive the model directory's own ONNX graph directly via onnxruntime "
                             "instead of a torch checkpoint -- the only option for a model that "
                             "exists solely as a quantized ONNX export.")
    parser.add_argument("--texts", type=Path, required=True, help="JSON list of {id, text}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds", default=None,
                        help="repeatable; overrides the default (0.0, 0.92)")
    args = parser.parse_args()

    window = json.loads((args.model_dir / "token_classification_config.json").read_text(encoding="utf-8"))
    max_words, overlap = int(window["max_words"]), int(window["overlap_words"])
    thresholds = tuple(args.thresholds) if args.thresholds else THRESHOLDS

    if args.onnx:
        if args.weights is not None:
            raise SystemExit("--weights and --onnx are mutually exclusive")
        score = onnx_scorer(args.model_dir, int(window["max_tokens"]))
    else:
        from transformers import AutoTokenizer, pipeline

        weights = args.weights or args.model_dir
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]
        nlp = pipeline("token-classification", model=str(weights), tokenizer=tokenizer,
                       aggregation_strategy="simple", device=-1)
        score = lambda chunk: nlp(chunk)  # noqa: E731

    documents = json.loads(args.texts.read_text(encoding="utf-8"))
    results = []
    for document in documents:
        text = document["text"]
        chunks = chunk_text(text, max_words, overlap)
        raw = []
        for chunk, offset in chunks:
            for entity in score(chunk):
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
                for threshold in thresholds
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "_note": "Reference output of the Python pipeline. Regenerate with "
                 "scripts/generate_token_classification_fixture.py.",
        "model": args.model_dir.name,
        "window": {"max_words": max_words, "overlap_words": overlap},
        "thresholds": [f"{t:g}" for t in thresholds],
        "documents": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(len(d["by_threshold"]["0"]) for d in results)
    print(f"{len(results)} documents, {total} spans -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
