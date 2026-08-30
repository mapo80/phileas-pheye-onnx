# Limitations and Accuracy

This module ports each supported model family's reference algorithm, and local inference is parity-tested against the Python implementation of each.

## A pathological window can silently lose everything in it

Read this one before the parity notes below: it is the most serious issue found while building this module, and it can affect a document that a caller specifically needs redacted correctly.

GLiNER windows by word count (`max_len`), which tracks sub-token count for ordinary language but not for a run of text with no whitespace and almost no vocabulary match — an embedded identifier, a base64 blob, a long run of digits. That kind of run stays a handful of "words" while inflating to thousands of sub-tokens. On this fork's own reference encoder (`gliner_multi_pii_v1_onnx`, mdeberta-v3-base with a BiLSTM head), measured directly: detections in the window degrade starting around 1,700 sub-tokens and collapse to **zero** by around 3,400 — two clearly-stated Italian names, `person` requested — with no exception, no error, nothing to tell the result apart from a clean document.

`LocalPhEyeOptions.maxSequenceTokens()` (default 1,024) bounds this: an over-long window is bisected on word boundaries — plain, non-overlapping, recursive — until every leaf is inside budget, and a single word whose own encoding alone still exceeds the ceiling is skipped rather than run through the encoder (no legitimate PII value is that long, and running the encoder over one that size risks the same quadratic-attention blowup the ceiling exists to bound). Confirmed on the same real weights: real names on both sides of a 200,000-character pathological run are both still found.

Building the fix surfaced a second bug worth naming: an earlier version widened each bisected half by the module's normal inter-window overlap, which — once a range had already been narrowed below that overlap — made both "halves" cover the entire parent range, so the recursion never made progress. This reproduced as an actual multi-minute hang against the real model, not a theoretical risk. See the README's "A pathological window can lose everything in it, silently" section for the full measured curve, the fix, and this second bug. `GlinerSequenceLengthSafetyTest` pins the mechanism fast and deterministically against the synthetic fixture; `GlinerLongSequenceDegradationTest` reproduces the original failure on real weights at the sizes it was measured at.

## Parity status

The detector is verified two ways. `LocalPhEyeDetectorParityTest` runs it against a tiny synthetic ONNX fixture (with real ONNX Runtime and the real DJL tokenizer) and asserts the deterministic spans, exercising the tensor wiring, the words-mask and span enumeration, the sigmoid/threshold/greedy-non-overlap decode, and the word-to-character offset mapping. `LocalPhEyeDetectorRealModelParityTest` runs it against a real exported GLiNER model (provided via `PHILEAS_GLINER_MODEL_DIR`) and asserts its spans match the Python `gliner.predict_entities` reference exactly: same offsets and labels, scores within tolerance. Both pass.

Verifying parity surfaced and fixed a real bug: `span_mask` is a boolean tensor in the GLiNER ONNX signature, and the detector had been feeding it as int64, which ONNX Runtime rejects. The detector now reproduces the Python reference on real weights.

For token-classification models, `LocalTokenClassifierParityTest` asserts span-for-span agreement with the Python `token-classification` pipeline on a committed fixture, at two thresholds, including documents long enough to need several windows. `scripts/cross_check_against_reference.py` runs the same comparison over a whole corpus, which is where an implementation drifts in practice.

Parity work on that path surfaced further real bugs. Three were about text that is not plain Latin, and all three failed the same silent way — a span with the right label and a plausible score, pointing at the wrong characters: tokenizer offsets are code-point indices while a Java `String` is indexed by UTF-16 code units, so one emoji shifted every later span; `Character.isWhitespace` excludes the non-breaking spaces Python's `str.isspace()` includes; and `Character.isLetterOrDigit` excludes the letter- and other-numbers `str.isalnum()` includes. `TextOffsets` holds all three, with `TextOffsetsTest` and `TokenClassifierUnicodeParityTest` covering them.

One more affected **both** families: DJL's default tokenizer loader truncates at 512 sub-tokens without saying so, which silently left the tail of a long window unexamined. Truncation is now disabled explicitly, and the GLiNER path additionally refuses a window whose text words did not all survive tokenization rather than scoring it with the tail missing. `TokenizersTest` pins both the fix and the default it works around.

## Accuracy

Detection with these models is probabilistic. A name detector will miss some names and flag some non-names, and accuracy depends on how close your text is to the data the model was trained on. Local inference does not change a model's accuracy: it runs the same model the remote PhEye service would, so the same calibration and recommended thresholds apply. Validate any model on your own text and set thresholds accordingly. You remain responsible for the personal data you process.

## Related

- [Phileas documentation](https://philterd.github.io/phileas/)
- [phileas-pheye-onnx on GitHub](https://github.com/philterd/phileas-pheye-onnx)
