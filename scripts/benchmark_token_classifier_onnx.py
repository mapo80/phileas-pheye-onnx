#!/usr/bin/env python3
"""Token-classification ONNX latency, same protocol as scripts/benchmark_gliner_onnx.py.

onnxruntime CPU, intra_op=4, inter_op=1, batch=1, 10 warm-up + 50 timed iterations, at sequence
lengths 128/512/2048 sub-tokens -- directly comparable to the GLiNER numbers that script produces,
and to any other model benchmarked the same way (matched protocol is what makes the README's
"GLiNER vs mmBERT" performance table meaningful).
"""
import argparse
import statistics
import time

import numpy as np
import onnxruntime as ort

SEQUENCE_LENGTHS = [128, 512, 2048]
INTRA_OP, INTER_OP, BATCH, WARMUP, ITERATIONS = 4, 1, 1, 10, 50

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("model_onnx", help="path to the packaged model's onnx/model.onnx graph")
args = parser.parse_args()

options = ort.SessionOptions()
options.intra_op_num_threads = INTRA_OP
options.inter_op_num_threads = INTER_OP
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session = ort.InferenceSession(args.model_onnx, options, providers=["CPUExecutionProvider"])

print(f"onnxruntime {ort.__version__}, provider {session.get_providers()}")
print(f"{'seq_len':>8} {'median_ms':>10} {'mean_ms':>9} {'p95_ms':>9}")

for length in SEQUENCE_LENGTHS:
    feed = {
        "input_ids": np.ones((BATCH, length), dtype=np.int64),
        "attention_mask": np.ones((BATCH, length), dtype=np.int64),
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
    print(f"{length:>8} {statistics.median(samples):>10.1f} {statistics.fmean(samples):>9.1f} {p95:>9.1f}")
