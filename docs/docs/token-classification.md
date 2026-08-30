# Token-Classification Models

The module drives two model families. The first is zero-shot GLiNER, described in
[How It Works](how-it-works.md). The second is a **BIO token classifier**: an encoder with a
classification head over a fixed taxonomy, trained to label every sub-token of the text.

Which one you get is decided by the model directory's layout, not by configuration — see
[Model Directory](model-directory.md). Nothing else about using the module changes: the same
`modelPath`, the same SPI, the same `PhEyeSpan` results.

## Why a second family

The two answer different questions, and the trade is not subtle.

| | GLiNER | token classifier |
|---|---|---|
| Labels | zero-shot: the labels you pass *are* the prompt | fixed; baked into the weights |
| New label | pass it and see | retrain |
| Cost | grows with the number of labels: they share the sequence with the text | independent of the label count |
| Output | a score per (span, label) pair | one distribution per sub-token |
| Overlapping labels | routine — hence the decode strategies | impossible, one argmax per token |

If your taxonomy is fixed and known — the usual case for a compliance filter — a token classifier is
the cheaper half of that table, and can be trained directly on the taxonomy it will be scored on. If
you need to ask for a label nobody trained for, GLiNER is the only one of the two that can answer.

## The pipeline

`LocalTokenClassifierDetector` is a port of the model's Python reference pipeline
(HuggingFace `token-classification` with `aggregation_strategy="simple"`), step for step, so a
threshold calibrated in Python transfers unchanged:

1. split the text into whitespace-delimited words and window them, `max_words` at a time with
   `overlap_words` shared with the next window;
2. tokenize each window's exact substring, keeping character offsets, and run the model;
3. softmax per sub-token; take the argmax class;
4. group consecutive sub-tokens into entities: a run continues while the entity type is unchanged
   and the tag is not a fresh `B-`. The entity's score is the mean of its sub-tokens' probabilities;
5. drop entities scoring below the threshold for their label;
6. reduce the windows' entities to one non-overlapping set.

Step 6 is where the windows come back together, and each part of it fixes a defect you would
otherwise see in the output:

- **Greedy by score, then by length.** Windows overlap, so the same entity is normally found twice;
  the copies collide here and one survives. Where two windows disagree on an entity's boundaries,
  the more confident one wins.
- **Trim surrounding whitespace.** A sub-word tokenizer's offsets typically include the space before
  a word, so a span would otherwise start one character early.
- **Widen to word boundaries.** The model can label part of a word — `No` of `Novara` — and
  replacing only that part leaves `[CITY_1]vara`, which is still readable. Masking one character too
  many is the only acceptable error here.
- **Coalesce.** Widening can make two spans touch or overlap; left alone the same word would be
  masked twice.

## Labels

The taxonomy is fixed, so the labels you pass to the filter **select from it** rather than defining
it. Matching is case-insensitive, so `fullname` and `FULLNAME` both work.

Asking only for labels the model cannot emit is an **error**, not an empty result. The two are
indistinguishable downstream, and the empty one hides a misconfiguration behind what looks like a
clean document. `LocalTokenClassifierDetector.entityTypes()` lists what a loaded model can emit.

## Thresholds

`detectionThreshold` and the per-label thresholds work exactly as on the GLiNER path, keyed by the
taxonomy label:

```bash
-Dphileas.pheye.onnx.detectionThreshold=0.92
-Dphileas.pheye.onnx.threshold.fullname=0.85
```

The threshold applies to the **entity** score — the mean over the entity's sub-tokens — and a span
scoring exactly the threshold is kept, which is what the reference pipeline does.

A model directory may declare the value the model was calibrated at, as `calibrated_threshold`.
When it does, and when the caller has expressed no threshold of its own (neither programmatically
nor through the property or environment variable), that value is used in place of the library
default of 0.5. The library default exists to reproduce upstream GLiNER; applying it to a model
calibrated elsewhere is how a redaction component ends up quietly under- or over-detecting. **A
threshold you set is never overridden.**

`decodeStrategy` is not read on this path. The cross-label suppression it exists to control cannot
occur when the model picks exactly one class per sub-token.

## Long input

The three long-input modes apply here unchanged: `CHUNK` (the default) examines the whole document
in overlapping windows, `FAIL` refuses an over-long one rather than examining part of it, and
`TRUNCATE` reproduces the unsafe upstream behaviour. The unit is `max_words` words rather than
GLiNER's `max_len`.

```bash
-Dphileas.pheye.onnx.longTextMode=CHUNK
-Dphileas.pheye.onnx.chunkOverlapWords=32
```

One case is specific to sub-word models. A single "word" can be tens of kilobytes — an embedded
identifier, a base64 blob — and word-level windowing cannot break it up. Rather than truncate,
the detector falls back to windowing that word's *sub-tokens*, so every sub-token is still examined
and character offsets stay exact. This only triggers past the encoder's hard capacity
(`max_tokens`); splitting any earlier would change what the model sees relative to the reference
pipeline, and so move the operating point the threshold was calibrated on.

Because that split already enforces `max_tokens` before any call reaches the model, a rejection from
ONNX Runtime on this path means the declared `max_tokens` was itself too high for the encoder — see
[When ONNX Runtime rejects a window](#when-onnx-runtime-rejects-a-window).

## When ONNX Runtime rejects a window

Neither model family gets a fixed sub-token capacity guessed on its behalf. On the GLiNER side,
`gliner_config.json`'s `max_len` counts *words*, and the entity prompt shares the same sequence as
the text, so the same `max_len` produces a different sub-token count depending on the label list —
there is no single number to declare that would be right for every request. Whether an over-long
sequence even fails is architecture-specific besides: an encoder with absolute position embeddings
rejects one past its table size, but one with relative position buckets does not reject long input at
all. A real, currently supported GLiNER encoder (mdeberta-v3-base) runs 384 words with 60 labels
without error — well past where a 512-token model is usually sized — which is exactly the case a
declared cap would get wrong.

What both detectors do instead is add context to the encoder's own rejection when one does occur: an
`OrtException` becomes an `IllegalStateException` naming the window's sub-token count and, on this
path, the `max_tokens` the model directory declared — rather than a bare "non-zero status code"
message naming an internal graph node.

## Text that is not plain Latin

A tokenizer reports character offsets as indices into *code points* — how Python indexes a string —
while a Java `String` is indexed by UTF-16 *code units*. One emoji earlier in a window shifts every
later span. Similarly, `Character.isWhitespace` excludes the non-breaking spaces that Python's
`str.isspace()` includes, and `Character.isLetterOrDigit` excludes the letter- and other-numbers that
`str.isalnum()` includes.

All three are handled in `TextOffsets`. They matter because none of them fails loudly: the span comes
back with the right label and a plausible score, pointing at the wrong characters.

## Parity

`LocalTokenClassifierParityTest` asserts span-for-span agreement with the Python reference on a
committed fixture, at two thresholds, including multi-window documents.
`TokenClassifierUnicodeParityTest` covers the cases above on real weights. The fixture is regenerated
by `scripts/generate_token_classification_fixture.py` and never edited by hand.

For a new model, `scripts/cross_check_against_reference.py` runs the same comparison over a whole
corpus, which is where an implementation actually drifts: an off-by-one in a window offset, a
boundary entity counted twice, a tie broken the other way.
