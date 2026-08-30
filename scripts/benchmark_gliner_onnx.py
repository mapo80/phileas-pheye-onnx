#!/usr/bin/env python3
"""GLiNER ONNX latency, same protocol as this project's own mmBERT benchmark.

Mirrors rizzo-source/src/training/benchmark_student_onnx.py exactly: onnxruntime CPU,
intra_op=4, inter_op=1, batch=1, FP32, 10 warm-up + 50 timed iterations, at sequence
lengths 128/512/2048 sub-tokens -- so the numbers here are directly comparable to the
already-published mmBERT student numbers at those same three lengths.

GLiNER's graph takes six inputs rather than two, so "sequence length" here means the
same thing (total sub-tokens through the encoder) but the additional span-enumeration
tensors have to be sized too. words_mask assigns each token its own word (one sub-token
per word) purely as a fixed, reproducible way to size span_idx/span_mask -- an
upper bound on span-enumeration cost, not a claim about how many words natural text of
that length would contain (see the README for the real word/sub-token ratio measured
directly against this checkpoint: roughly 1.7-2.4 sub-tokens per word for Italian text).
"""

import statistics
import time
import numpy as np
import onnxruntime as ort

import argparse

MAX_WIDTH = 12
SEQUENCE_LENGTHS = [128, 512, 2048]
INTRA_OP, INTER_OP, BATCH, WARMUP, ITERATIONS = 4, 1, 1, 10, 50

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("model_onnx", help="path to the GLiNER onnx/model.onnx graph")
parser.add_argument("--max-width", type=int, default=MAX_WIDTH,
                    help="the model's gliner_config.json max_width (default: 12)")
args = parser.parse_args()
MAX_WIDTH = args.max_width
MODEL = args.model_onnx

options = ort.SessionOptions()
options.intra_op_num_threads = INTRA_OP
options.inter_op_num_threads = INTER_OP
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session = ort.InferenceSession(MODEL, options, providers=["CPUExecutionProvider"])

print(f"onnxruntime {ort.__version__}, provider {session.get_providers()}")
print(f"{'seq_len':>8} {'num_words':>10} {'median_ms':>10} {'mean_ms':>9} {'p95_ms':>9}")

for length in SEQUENCE_LENGTHS:
    num_words = length  # one sub-token per word: an upper bound on span-enumeration cost
    num_spans = num_words * MAX_WIDTH
    feed = {
        "input_ids": np.ones((BATCH, length), dtype=np.int64),
        "attention_mask": np.ones((BATCH, length), dtype=np.int64),
        "words_mask": np.arange(1, length + 1, dtype=np.int64).reshape(BATCH, length),
        "text_lengths": np.array([[num_words]], dtype=np.int64),
        "span_idx": np.zeros((BATCH, num_spans, 2), dtype=np.int64),
        "span_mask": np.zeros((BATCH, num_spans), dtype=bool),
    }
    for _ in range(WARMUP):
        session.run(None, feed)
    samples = []
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        session.run(None, feed)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    p95 = samples[int(0.95 * len(samples)) - 1]
    print(f"{length:>8} {num_words:>10} {statistics.median(samples):>10.1f} "
          f"{statistics.fmean(samples):>9.1f} {p95:>9.1f}")
