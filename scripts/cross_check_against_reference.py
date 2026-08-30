#!/usr/bin/env python3
"""Compare this module's spans against the model's Python reference over a whole corpus.

The fixture parity test pins nine documents. This checks the same property at corpus scale, on
documents long enough to need many windows, which is where an implementation drifts: an off-by-one
in a window offset, a boundary entity counted twice, a tie broken the other way.

    python scripts/cross_check_against_reference.py \
        --model-dir  MODEL_DIR \
        --weights    HF_CHECKPOINT \
        --documents  corpus.jsonl        # one {"id", "text"} per line
        --threshold  0.92
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_token_classification_fixture import chunk_text, reduce_spans  # noqa: E402


def utf16_offsets(text: str) -> list[int]:
    """Map each code-point index to the UTF-16 code-unit index Java would use for it.

    Python indexes a string by code point; a Java String is indexed by UTF-16 code unit. The two
    coincide until the text holds a character outside the Basic Multilingual Plane. Comparing the
    two implementations' raw offsets across that boundary compares two different coordinate
    systems, so the reference is converted into Java's before anything is compared.
    """
    offsets, unit = [], 0
    for character in text:
        offsets.append(unit)
        unit += 2 if ord(character) > 0xFFFF else 1
    offsets.append(unit)
    return offsets


def python_reference(model_dir: Path, weights: Path, documents: list[dict],
                     threshold: float) -> dict[str, list[tuple]]:
    window = json.loads((model_dir / "token_classification_config.json").read_text(encoding="utf-8"))
    max_words, overlap = int(window["max_words"]), int(window["overlap_words"])

    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]
    nlp = pipeline("token-classification", model=str(weights), tokenizer=tokenizer,
                   aggregation_strategy="simple", device=-1)

    spans: dict[str, list[tuple]] = {}
    for index, document in enumerate(documents, start=1):
        text = document["text"]
        chunks = chunk_text(text, max_words, overlap)
        raw = []
        if chunks:
            for (_, offset), entities in zip(chunks, nlp([c for c, _ in chunks])):
                for entity in entities:
                    score = float(entity["score"])
                    if score >= threshold:
                        raw.append({"label": entity["entity_group"],
                                    "start": int(entity["start"]) + offset,
                                    "end": int(entity["end"]) + offset,
                                    "score": score})
        to_utf16 = utf16_offsets(text)
        spans[document["id"]] = [(s["label"], to_utf16[s["start"]], to_utf16[s["end"]])
                                 for s in reduce_spans(raw, text)]
        if index % 100 == 0 or index == len(documents):
            print(f"  python reference: {index}/{len(documents)}", flush=True)
    return spans


def java_spans(model_dir: Path, documents: list[dict], threshold: float,
               classpath: str) -> dict[str, list[tuple]]:
    payload = "\n".join(
        json.dumps({"id": d["id"],
                    "text_b64": base64.b64encode(d["text"].encode("utf-8")).decode("ascii")})
        for d in documents)
    result = subprocess.run(
        ["java", "-cp", classpath, "ai.philterd.phileas.pheye.onnx.SpanDump",
         str(model_dir), str(threshold)],
        input=payload, capture_output=True, text=True, check=True)
    spans = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spans[row["id"]] = [(s["label"], s["start"], s["end"]) for s in row.get("spans", [])]
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--classpath", default="target/test-classes:target/classes:@target/classpath.txt")
    args = parser.parse_args()

    classpath = args.classpath
    if "@" in classpath:
        prefix, _, file = classpath.partition("@")
        classpath = prefix + Path(file).read_text(encoding="utf-8").strip()

    documents = [json.loads(line) for line in args.documents.read_text(encoding="utf-8").splitlines() if line.strip()]
    astral = sum(1 for d in documents if any(ord(c) > 0xFFFF for c in d["text"]))
    print(f"{len(documents)} documents, threshold {args.threshold}"
          f" ({astral} with characters outside the Basic Multilingual Plane)")

    reference = python_reference(args.model_dir, args.weights, documents, args.threshold)
    actual = java_spans(args.model_dir, documents, args.threshold, classpath)

    mismatched = []
    reference_total = 0
    for document in documents:
        want = reference[document["id"]]
        got = actual.get(document["id"], [])
        reference_total += len(want)
        if want != got:
            mismatched.append((document["id"], want, got))

    print(f"\nreference spans: {reference_total}")
    print(f"documents compared: {len(documents)}")
    print(f"documents differing: {len(mismatched)}")
    for name, want, got in mismatched[:5]:
        print(f"  {name}\n    python: {want}\n    java:   {got}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
