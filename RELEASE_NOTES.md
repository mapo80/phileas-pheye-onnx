# phileas-pheye-onnx Release Notes

Notable changes to phileas-pheye-onnx, most recent first.

Full changelogs for each release are available in the [GitHub releases](https://github.com/philterd/phileas-pheye-onnx/releases).

## Version 1.4.0 - August 30, 2026

* Adds a second model family: BIO token classifiers, alongside the existing zero-shot GLiNER span
  models. `LocalTokenClassifierDetector` ports the HuggingFace `token-classification` pipeline
  (`aggregation_strategy="simple"`) with word windowing, and is parity-tested span for span against
  it, exact on a 529-document corpus. Which detector is built follows from the model directory's
  layout, so nothing in a policy changes for existing GLiNER callers; `LocalDetectorFactory` refuses
  a directory that matches neither layout, or both. A model directory may declare the threshold its
  weights were calibrated at (`calibrated_threshold`), applied by the detector's own constructor
  unless the caller set a threshold explicitly.
* **Fixes a silent, complete loss of detections in a GLiNER window containing a pathological run of
  text** (an embedded identifier, a base64 blob, a long run of digits with no whitespace). Word
  count and sub-token count are usually proportional, but an out-of-vocabulary run with no
  whitespace can inflate a modest word count to thousands of sub-tokens, and on this fork's own
  reference encoder that measurably degrades detection starting around 1,700 sub-tokens and
  collapses it to nothing by around 3,400 — with no exception, no error, nothing to distinguish the
  result from a clean document. `LocalPhEyeOptions.maxSequenceTokens()` (default 1,024) now bounds
  this: an over-long window is bisected on word boundaries, plain and non-overlapping, recursively,
  until every leaf is inside budget; a single word whose own encoding alone still exceeds the
  ceiling is skipped rather than risking the quadratic attention cost of running it. Confirmed on
  the same real weights the defect was found on: real names on both sides of a 200,000-character
  pathological run are both still found. An earlier version of the bisection extended each half by
  the module's normal inter-window overlap and hung for minutes against real weights once a range
  had narrowed below that overlap; see the README for the full measured degradation curve and this
  second bug.
* **Fixes a silent data leak on both paths.** DJL's default tokenizer loader truncates at 512
  sub-tokens without reporting it, so the tail of a long window was never scored and never
  reported. Truncation is now disabled explicitly, and the GLiNER path refuses a window whose text
  words did not all survive tokenization rather than scoring it with the tail missing.
* Handles text outside plain Latin correctly. A fast tokenizer's character offsets are code-point
  indices, while a Java `String` is indexed by UTF-16 code units, so one emoji earlier in a window
  shifted every later span by one; the non-breaking spaces were not trimmed off a span because
  `Character.isWhitespace` excludes them and Python's `str.isspace()` does not; letter- and
  other-numbers did not count as word characters when widening a partial span; and the
  whitespace-run splitter was not Unicode-aware, so a non-breaking space joined two words that the
  reference keeps apart and moved every window boundary after it.
* The label set is applied before the greedy reduce rather than after. Filtering afterwards let an
  unwanted entity suppress a wanted one over the same characters and leave nothing in its place.
* Native handles (ONNX session, tokenizer, session options) are released when the fail-closed
  constructor throws, and `close()` releases the tokenizer even if closing the session fails.
* Neither detector guesses a universal sub-token capacity for GLiNER: `max_len` counts words, the
  entity prompt shares the sequence with the text, and whether an over-long sequence even fails is
  architecture-specific (confirmed empirically: mdeberta-v3-base runs 384 words with 60 labels
  without error). Instead, an `OrtException` from ONNX Runtime is turned into an `IllegalStateException`
  naming the window's sub-token count and, on the token-classification path, the declared
  `max_tokens` — an actionable message in place of a bare "non-zero status code".
* README adds a freshly-measured GLiNER-vs-token-classification comparison (quality, matched-protocol
  latency, model size) and a mechanistic breakdown of the token classifier's low `ORG` exact-match
  F1: 88% of the mismatches are a tokenizer artifact, not missed entities.

## Version 1.0.0 - June 21, 2026

* First release. Adds optional local, on-device GLiNER inference for the Phileas PhEye filter via ONNX Runtime, so Phileas can run a model in-process instead of calling a remote PhEye service. The detector is discovered through the Phileas `PhEyeDetectorProvider` SPI, so adding this module to the classpath enables local inference when a PhEye filter sets a `modelPath`. Targets Phileas 4.1.0.
