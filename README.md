# phileas-pheye-onnx (mapo80 fork)

Local, in-process PII inference for the Phileas PhEye filter, via ONNX Runtime for Java.

This fork exists to make the module usable as the privacy component of a JVM service. What differs
from upstream, each difference driven by that goal:

| | upstream | this fork |
|---|---|---|
| Java | 25 bytecode | **Java 21** (class-file major 65, enforced in CI) |
| Phileas | 4.1.0 | **4.2.0** |
| Model families | GLiNER only | **GLiNER and BIO token classifiers** |
| ONNX decode threshold | hardcoded `0.5` | **configurable, global and per-label**, default `0.5`, or the model's own calibrated value |
| Overlapping labels | one flat greedy pass; a stronger ORGANIZATION deletes a correct PERSON | **`PER_LABEL_GREEDY`** keeps both for the caller's resolver |
| Input longer than the model's window | silently truncated | **overlapping windows** (or fail-closed) |
| Tokenizer | DJL default: silently truncates at 512 sub-tokens | **truncation off**; a window that loses words is refused |
| Non-Latin text | — | **code-point offsets converted**, Python's whitespace and word-character rules |
| Broken model directory | partly deferred | **fail-closed at construction** |

Everything else — the GLiNER prompt construction, tokenization, span enumeration, the six input
tensors, the logits reshape, the greedy non-overlap decode — is upstream's algorithm, unchanged and
parity-tested.

## Architecture

```
your JVM service
  └─ Phileas PhEyeFilter
       └─ PhEyeDetectorProvider (java.util.ServiceLoader)
            └─ LocalDetectorFactory                    ← picks by model directory layout
                 ├─ LocalPhEyeDetector                 ← GLiNER span model
                 └─ LocalTokenClassifierDetector       ← BIO token classifier
                      └─ ONNX Runtime Java
                           └─ model.onnx on the local filesystem
```

One process. No HTTP for inference, no Ph-Eye server, no Python, no sidecar, no child process,
CPU-only. Setting `modelPath` on the Phileas `PhEyeConfiguration` is what makes `PhEyeFilter` pick
this detector; leave it unset and Phileas uses its `RemotePhEyeDetector`, which calls a service over
HTTP.

## Two model families

The module drives two kinds of model, and the model directory's layout decides which — not a
configuration switch, because a caller that has to name the family gets to name it wrongly.

| | GLiNER (`LocalPhEyeDetector`) | token classifier (`LocalTokenClassifierDetector`) |
|---|---|---|
| Labels | zero-shot: the labels you pass *are* the prompt | fixed taxonomy, baked into the weights |
| A new label | pass it and see | retrain |
| Cost | grows with the label count: labels share the sequence with the text | independent of the label count |
| ONNX inputs | six tensors, including enumerated candidate spans | `input_ids`, `attention_mask` |
| Output | a score per (span, label) pair, sigmoid | a distribution over BIO classes per sub-token, softmax |
| Overlapping labels | routine — hence the decode strategies | impossible: one argmax per token |

For a fixed, known taxonomy — the usual case for a compliance filter — a token classifier is the
cheaper half of that table, and can be trained directly on the taxonomy it will be scored on. When
you need to ask for a label nobody trained for, GLiNER is the only one of the two that can answer.

`LocalDetectorFactory.open(modelDir)` builds whichever the directory describes; a directory matching
neither layout, or somehow both, is refused rather than half-loaded.

## GLiNER vs mmBERT: quality, performance, accuracy

The table above is the architectural picture. This is the measured one: an actual GLiNER checkpoint
against an actual token-classification checkpoint, same machine, same protocol, same test documents
where the comparison is apples-to-apples, with every place it stops being apples-to-apples called
out rather than glossed over.

**Models compared.** GLiNER: [`urchade/gliner_multi_pii-v1`](https://huggingface.co/urchade/gliner_multi_pii-v1)
(mdeberta-v3-base backbone, 288.95M parameters, 1,103.6 MB FP32 ONNX), zero-shot, run through this
module's `LocalPhEyeDetector` at the exact configuration a prior evaluation in this workspace froze
for production: labels `person`, `company`, `postal address`; decode threshold 0.60 globally, 0.40
for `person`; `FLAT_GREEDY`; `CHUNK`. mmBERT: this repository's own distilled student,
`rizzo-pii-student-6x384gf` v1.2.0, **INT8 dynamic, per-channel** (ModernBERT backbone, 6 of the
teacher's layers selected globally, factorized rank-128 embedding, 18.75M parameters, 19.1 MB ONNX),
fixed 45-class BIO taxonomy (22 entity types), run through `LocalTokenClassifierDetector` at its own
calibrated threshold, 0.98.

The INT8 graph is the one actually shipped, not a hypothetical: [an independent, extensively
documented evaluation](https://github.com/mapo80/rizzo-pii/releases/tag/student-models-v1.2.0) found
it statistically indistinguishable from its own FP32 baseline on exact entity F1 (paired bootstrap
95% CI `[-0.003632, +0.005477]`, containing zero) while being 3.93× smaller and 1.2–1.4× faster, and
this session independently reproduced that finding with its own calibration methodology before
promoting it: **0.7981 FP32 vs 0.8037 INT8**, dev-calibrated, measured once on test, model only. Both
figures are the *model alone*, no format or checksum gate — the fair comparison for what this module
actually ships, since it has no such gate itself.

These are not two sizes of the same thing: GLiNER is 15.4× the parameters and 57.8× the ONNX size of
the token classifier. That difference is a large part of what the numbers below show, and it's worth
holding in mind while reading them.

### Quality: the three categories both models can be scored on

GLiNER's frozen production config only asks for three labels, so a fair head-to-head is restricted
to what both models can be scored on: the taxonomy's `FULLNAME`, `ORG`, and `STREET`, matched to
GLiNER's `person`, `company`, and `postal address` respectively. Measured fresh, in this session, on
the same 156-document test split of the Finance & Banking Gold v1 corpus that `6x384gf`'s own
headline number was measured on (248 gold entities across the three tags), entity-level exact
character-span match, no format/checksum gate:

| | Precision | Recall | F1 | tp | fp | fn |
|---|---:|---:|---:|---:|---:|---:|
| **GLiNER** (3 labels, thr 0.60/0.40) | 0.7937 | 0.8065 | **0.8000** | 200 | 52 | 48 |
| **mmBERT `6x384gf` INT8** (same 3 tags only, thr 0.98) | 0.7826 | 0.7258 | 0.7531 | 180 | 50 | 68 |

| Tag | Gold | GLiNER P/R/F1 | mmBERT P/R/F1 |
|---|---:|---:|---:|
| `FULLNAME` (person) | 124 | 0.939 / 1.000 / **0.969** | 1.000 / 1.000 / **1.000** |
| `ORG` (company) | 92 | 0.854 / 0.826 / **0.840** | 0.333 / 0.272 / **0.299** |
| `STREET` (postal address) | 32 | 0.000 / 0.000 / **0.000** | 1.000 / 0.969 / **0.984** |

On this narrow slice GLiNER's overall F1 is higher, entirely on the strength of `ORG`: zero-shot
`company` finds organizations the fixed taxonomy's `ORG` class mostly doesn't. Investigated for this
README: of the 92 gold `ORG` spans, 25 match exactly and the other 67 break down as follows —

| Mechanism | Mismatches | What happens |
|---|---:|---|
| Trailing period fused away | 27 (40%) | The gold set formats some fields as `key=value;` with no space — e.g. `CedentePrestatore=Gruppo Nordest S.n.c.; IdFiscale=...`. The tokenizer merges the abbreviation's final `.` with the following `;` into one sub-token, which the model must label as a whole; it almost always calls that fused token `O`, so the predicted span is the gold span minus its last character. |
| Extra context absorbed | 20 (30%) | The mirror image: a sentence-ending `.` fused onto the abbreviation's own `.` (`"...S.r.l.."`) gets labelled `I-ORG` as a whole, extending the span by one character; separately, generic words that reliably precede a company mention in this data (`Impresa `) get pulled into the span. |
| Below the calibrated threshold | 17 (25%) | A correctly-bounded prediction scores 0.833–0.975 — under 0.98 — and is cut by calibration, not by a boundary error. Higher here than on the FP32-scale model this replaced, because a higher calibrated threshold (0.98, chosen for this smaller graph) always cuts more borderline-but-correct predictions, not because boundaries got worse. |
| Mixed: both boundaries off at once | 2 (3%) | An `Impresa ` prefix absorbed on the left *and* a trailing period lost on the right, on the same span. |
| A specific leading word dropped, cleanly | 1 (1%) | `Credito` — an ordinary Italian noun ("credit") reused here as a company-name prefix (`Credito Aurora S.p.A.`) — scores `O` even in a clean, unambiguous sentence: the model has not reliably learned to read it as part of a name rather than as the common word. |

The first two rows plus the mixed one (73% of mismatches) are boundary artifacts, not missed
entities — verified by inspecting the model's own per-sub-token probabilities: `I-ORG` above 0.94 on
every character of the entity except the fused punctuation token, which scores `O` at 0.98+. The
dominant one is specific to how this gold set formats some fields (`key=value;`, no space before the
punctuation); in every case the core company name is still correctly bounded and would still be
masked, off by one shared punctuation character or one generic precursor word. Exact-span-match F1
counts these identically to a fully missed entity, so 0.299 understates what a redaction pipeline
built on this model would actually achieve here. The calibration cutoffs (25%) are a real
precision/recall trade-off, not an error, and only `Credito` (1%) is a genuine boundary weakness.
`scripts/investigate_org_boundary_errors.py --mmbert-onnx` reproduces this exact breakdown against
any gold set and model directory in the same shape.

`FULLNAME`/`person` is close to a tie, both near-perfect. `STREET` is the opposite extreme: GLiNER
finds **zero** of the 32 gold spans under the `postal address` prompt, where the fixed-taxonomy model
gets all but one. This is not a fluke of this dataset — an earlier, differently-scoped evaluation in
this workspace (`aliasit-pii-gold-v1`, 114 documents, the phileas-pheye-onnx `models-v1.0.0` release)
found the same frozen config's `address` F1 at 0.120–0.160, its weakest category there too, against
`person` at 0.47–0.51. The likely reason generalizes across both datasets: this gold set's `STREET`
spans are bare street names and numbers (`Via Garibaldi 24`), and GLiNER's zero-shot `postal address`
prompt appears calibrated toward a fuller address string, not an isolated street name — a zero-shot
label's wording is not free of the assumptions behind it, and this is a concrete case of the "Labels"
section below biting in practice, not a hypothetical.

**Why not all 22 tags.** `6x384gf`'s full-taxonomy number — F1 0.8365 exact (FP32; the INT8 graph's
own full-taxonomy number was not separately measured with the format/checksum gate this session used
only for the model-only comparisons above), all 22 categories, format and checksum gate, calibrated
threshold — has no GLiNER counterpart in this comparison. GLiNER is zero-shot, so nothing stops
asking it for `AMOUNT`, `IBAN`, `CF`, or any other of the remaining 19 tags; nothing in this workspace
has *evaluated* it against them, because the frozen production config this section reuses was
deliberately scoped to the three categories a prior evaluation found GLiNER usable for. Presenting a
from-scratch 22-label zero-shot run as comparable to a model specifically distilled against that
exact taxonomy would overstate what either number means; the honest comparison is the one above, on
the categories both were actually measured on.

### Performance: same protocol, three sequence lengths

onnxruntime, `CPUExecutionProvider`, batch 1, 4 intra-op threads, 1 inter-op thread, 10 warm-up runs
discarded, 50 timed iterations, median reported, direct onnxruntime calls rather than through this
module's Java wrapper, for both models, measured in this session on the same machine so the ratio is
not exposed to cross-machine drift — the [quantization release
notes](https://github.com/mapo80/rizzo-pii/releases/tag/student-models-v1.2.0) measured that drift
directly for FP32-vs-INT8 on the same graph (one identical run timed 129.52 ms and 110.78 ms hours
apart) and found it larger than the difference the comparison was trying to measure, which is why
same-machine, same-session numbers matter here rather than citing each model's own previously
published figures at face value.

| Sequence length | GLiNER median | mmBERT `6x384gf` INT8 median | GLiNER ÷ mmBERT |
|---:|---:|---:|---:|
| 128 sub-tokens | 106.3 ms | 9.3 ms | 11.4× |
| 512 sub-tokens | 485.8 ms | 40.9 ms | 11.9× |
| 2,048 sub-tokens | 3,148.8 ms | 288.3 ms | 10.9× |

The ratio is essentially flat across all three lengths (~11.4×), tracking size more than any
algorithmic gap — GLiNER's own encoder is larger, and it runs FP32 against mmBERT's INT8 here on top
of that. Note the units: mmBERT's document-level throughput (linear in document length; see "Long
input" below) is a different, larger-scale measurement than this per-forward-pass benchmark; the two
are not directly comparable to each other, only within each table to its own model.

### Size

| | Parameters | ONNX |
|---|---:|---:|
| GLiNER (`gliner_multi_pii-v1`, FP32) | 288.95 M | 1,103.6 MB |
| mmBERT `6x384gf` v1.2.0 (INT8) | 18.75 M | 19.1 MB |
| Ratio | 15.4× | 57.8× |

The size ratio is larger than the parameter ratio because the two aren't measured at the same
precision: GLiNER here is FP32 (4 bytes/parameter), mmBERT is INT8 (roughly 1 byte/parameter plus
per-channel scales) — quantization is exactly why this comparison is no longer merely "fewer
parameters," on top of already having fewer of them.

### Reading the three tables together

- **On a fixed, known taxonomy** — the compliance-filter case the "Two model families" table above
  already argues for — mmBERT wins on cost by a very wide margin (15.4× fewer parameters, ~11.4×
  faster, 57.8× smaller on disk) and matches or beats GLiNER on two of the three categories once
  distilled, quantized and calibrated on that taxonomy specifically; `ORG`'s boundary problem (mostly
  a tokenizer artifact on this gold set's field formatting, detailed above) is the one open exception.
- **GLiNER's zero-shot flexibility has a real, measured cost attached**, not just an architectural
  one: an order of magnitude slower and dramatically larger for this pairing, for quality that is
  *better* on one category (`ORG`), *worse* on another (`STREET`, though only by one span here) with
  the frozen config used here, and untested against the sixteen categories mmBERT was actually
  distilled for.
- **Neither number is free-standing.** The `STREET` result specifically demonstrates why: it is not
  that GLiNER cannot find street names, it is that `postal address` was the wrong prompt for how this
  gold set annotates them. A different label choice, or per-category tuning of the sort this frozen
  config deliberately avoided, could move that number substantially — in either direction, and only
  another measurement would say which.

Full methodology for the quality table — dataset provenance, split integrity, negative controls — is
in `working/finance-banking-gold/MODEL_REPORT.md` (mmBERT's own repository), the [quantization
release notes](https://github.com/mapo80/rizzo-pii/releases/tag/student-models-v1.2.0) (the INT8
recipe and its own independent quality/speed evaluation), and
`working/confronto-rizzo-student-fastino-phileas.md` (the prior GLiNER evaluation this section's
config comes from), all outside this repository. The quality table reproduces with
`scripts/gliner_vs_mmbert_comparison.py --mmbert-onnx` against your own copy of the gold set and both
model directories; the performance table reproduces with `scripts/benchmark_gliner_onnx.py` and
`scripts/benchmark_token_classifier_onnx.py` against each graph directly.

## Requirements

- **Java 21** or newer at runtime (the artifact targets 21)
- Phileas 4.2.0
- A GLiNER or token-classification model directory on disk

## Maven

Two channels carry the same artifacts. Prefer the first: it needs no credentials.

### Without a token — the channel this project uses

```xml
<repositories>
  <repository>
    <id>mapo80-maven</id>
    <url>https://raw.githubusercontent.com/mapo80/phileas-pheye-onnx/maven-repo</url>
  </repository>
</repositories>

<dependency>
  <groupId>io.github.mapo80</groupId>
  <artifactId>phileas-pheye-onnx</artifactId>
  <version>1.2.0</version>
</dependency>
```

No `<server>` entry, no token, no `settings.xml`. The `maven-repo` branch holds a plain Maven
repository layout, and `raw.githubusercontent.com` serves a public repository anonymously. The
release workflow writes it on every tag and then verifies, with no `Authorization` header, that the
new version really is anonymously readable.

The `maven-repo` branch is protected: force pushes and deletions are disabled, with `enforce_admins`
on, so a published version is append-only and cannot be silently rewritten. That closes the one real
supply-chain weakness of serving a Maven repository from a git branch.

Note the remaining trade-off: `raw.githubusercontent.com` is not an artifact host, so there is no CDN
guarantee, no SLA, and no download audit, and a CI farm behind a single egress IP can hit GitHub's
rate limits. For production, proxy this repository through an internal Nexus/Artifactory or the
organisation's Azure Artifacts feed and let builds resolve from there.

### Maven Central (configured, not enabled)

Not currently used: the project is consumed internally, and Central versions can never be deleted,
so tying the coordinates to an immutable public repository buys nothing here. The setup below is
left in place should public distribution ever be wanted.

Central needs **nothing at all** from a consumer: no repository entry, no credentials, no
`settings.xml` — it is a default repository for every Maven installation. It is CDN-backed, has no
rate limits, and `mvnrepository.com` indexes it automatically (mvnrepository.com is a search index,
not a place you publish to).

`publish-central.yml` is ready and skips cleanly until four repository secrets exist:
`CENTRAL_TOKEN_USERNAME`, `CENTRAL_TOKEN_PASSWORD`, `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`.

Three one-time steps cannot be automated, because they need a human and a browser:

1. create an account at [central.sonatype.com](https://central.sonatype.com);
2. verify the `io.github.mapo80` namespace — the Portal issues a code and asks you to create a
   public repository with that name under the `mapo80` account, which proves you own it;
3. generate a GPG key, publish the public half to a keyserver, and store the private half in the
   secrets above.

The POM already satisfies every Central requirement (name, description, url, licenses, developers,
scm, sources and javadoc jars), so after those three steps a tag is all it takes.

### GitHub Packages (needs a token)

```xml
<repository>
  <id>github</id>
  <url>https://maven.pkg.github.com/mapo80/phileas-pheye-onnx</url>
</repository>
```

GitHub Packages requires authentication for Maven **even for a public package**, and exposes no
setting to allow anonymous reads. A consumer therefore needs a `~/.m2/settings.xml` server entry
with a token carrying `read:packages`:

```xml
<servers>
  <server>
    <id>github</id>
    <username>YOUR_GITHUB_USERNAME</username>
    <password>YOUR_TOKEN_WITH_read:packages</password>
  </server>
</servers>
```

Each tagged release also attaches the JAR, sources JAR, javadoc JAR and `.sha256` sums to the
[GitHub Release](https://github.com/mapo80/phileas-pheye-onnx/releases), for direct download when a
Maven repository is not wanted.

## Model directory

### GLiNER

```
<modelDir>/
├── tokenizer.json          HuggingFace fast tokenizer
├── gliner_config.json      span width, max length, prompt tokens
└── onnx/model.onnx         the exported GLiNER model  (or <modelDir>/model.onnx)
```

Any of the ONNX quantizations works as long as it is saved under that name. A missing or unreadable
file, a malformed config, a `words_splitter_type` other than `whitespace`, a non-positive
`max_len`/`max_width`, or an ONNX graph that is not a GLiNER span export all fail at construction.

### Token classification

```
<modelDir>/
├── tokenizer.json                        HuggingFace fast tokenizer
├── config.json                           HuggingFace model config; id2label is the taxonomy
├── token_classification_config.json      the inference window, and the calibrated threshold
└── onnx/model.onnx                       input_ids, attention_mask -> logits
```

```json
{
  "max_words": 120,
  "overlap_words": 20,
  "max_tokens": 8192,
  "words_splitter_type": "whitespace",
  "calibrated_threshold": 0.98
}
```

`config.json` is read for `id2label` only, and it is authoritative: it is the taxonomy the weights
were trained on, in BIO notation. Labels are ordered by class index rather than by their order in
the file — JSON objects have none, and a label read at the wrong index mislabels every document.

`token_classification_config.json` carries what the HuggingFace config does not, and is **required**
rather than defaulted:

- `max_words` / `overlap_words`: the inference window, in whitespace-delimited words. A model's
  `config.json` advertises its positional limit (`max_position_embeddings`, 8192 for ModernBERT),
  which is a ceiling, not the length the model is meant to be run at. Running a model at the wrong
  window degrades it quietly, so the number is declared rather than guessed.
- `max_tokens`: the encoder's hard sub-token capacity, past which the graph cannot run at all. A
  safety limit, not a quality knob.
- `calibrated_threshold` *(optional)*: see [Threshold](#threshold).

`scripts/package_token_classification_model.py` builds this layout from a HuggingFace checkpoint and
an exported graph.

The same fail-closed rule applies, plus: an `id2label` that is not BIO, a gap in its class indices,
a taxonomy of nothing but `O`, an `overlap_words` at least as wide as the window (the stride would be
zero and the windowing would never advance), and a graph whose class count disagrees with
`id2label` are all construction failures.

## Threshold

Two different thresholds are involved, and conflating them is the usual source of confusion:

- the **ONNX local decode threshold** decides which candidate spans leave the model at all. A span
  dropped here is invisible to everything downstream.
- Phileas's **per-label policy thresholds** filter spans the detector already emitted. They can only
  ever be more restrictive.

Only the first is this module's concern. Core Phileas owns `PhEyeConfiguration` and it has no field
for it, so when the detector is built through the SPI the value comes from a system property or
environment variable:

```bash
-Dphileas.pheye.onnx.detectionThreshold=0.20
# or: PHILEAS_PHEYE_ONNX_DETECTION_THRESHOLD=0.20
```

Constructing directly:

```java
try (var detector = new LocalPhEyeDetector(Path.of("/models/gliner"), 0.20)) {
    var spans = detector.detect(text, List.of("person", "organization", "address"), "ctx", 0);
}
```

The default stays `0.5`, so an unconfigured detector behaves exactly like upstream — pinned by a
parity test. **No model-specific value is baked in.** GLiNER models calibrate very differently:
some score a correct Italian person span around 0.45, others around 0.99. Pick the threshold per
model, on a validation set.

An unparseable value is a startup error, never a silent fallback to the default.

### The model's own calibrated threshold

A token-classification model directory may declare `calibrated_threshold`: the value that model was
calibrated at on a validation split. It is a property of the model, not of the library, which is why
it lives in the directory.

When the caller has expressed **no** threshold of its own — neither programmatically nor through the
property or environment variable — `LocalDetectorFactory` uses the declared value in place of the
library default. The library default of 0.5 exists to reproduce upstream GLiNER; applying it to a
model calibrated at 0.98 is how a redaction component ends up quietly over-detecting. A threshold
the caller did set is never overridden, and the distinction is explicit
(`LocalPhEyeOptions.thresholdExplicit()`) rather than inferred from the value.

### Per-label thresholds

GLiNER calibrates differently per label as well as per model, so each label can have its own floor,
with the global value as the fallback:

```bash
-Dphileas.pheye.onnx.detectionThreshold=0.40   # fallback for any label not listed
-Dphileas.pheye.onnx.threshold.person=0.20
-Dphileas.pheye.onnx.threshold.address=0.35
```

Programmatically:

```java
var options = LocalPhEyeOptions.of(0.40,
        Map.of("person", 0.20, "address", 0.35),
        LocalPhEyeOptions.DecodeStrategy.PER_LABEL_GREEDY);
var detector = new LocalPhEyeDetector(Path.of("/models/gliner"), options);
```

Label lookup is case-insensitive. Labels are free text, so per-label properties are discovered by
prefix rather than from a fixed list.

### Decode strategy: the cross-label suppression trap

GLiNER scores every (span, label) pair independently, so the same words routinely come back as both a
confident ORGANIZATION and a slightly less confident PERSON. Upstream runs **one** greedy pass across
all labels, so the ORGANIZATION wins and the PERSON is deleted — and the name is then never masked.
That is a leak, not a precision issue.

| Strategy | Behaviour |
|---|---|
| `FLAT_GREEDY` (default) | upstream: highest span wins across all labels |
| `PER_LABEL_GREEDY` | greedy within each label; different labels may overlap |

```bash
-Dphileas.pheye.onnx.decodeStrategy=PER_LABEL_GREEDY
```

`FLAT_GREEDY` remains the default so the parity guarantee with upstream holds. For redaction, prefer
`PER_LABEL_GREEDY` and let your own resolver reconcile overlaps: two overlapping classifications cost
a little precision, a suppressed PERSON costs a name.

This setting is **not read on the token-classification path**. A BIO classifier picks exactly one
class per sub-token, so the cross-label suppression these strategies exist to control cannot arise.

## Long input

A GLiNER model has a hard word limit (`max_len`, typically 384 words). Upstream truncates to it.
For a redaction component that is a data leak rather than a quality issue: the tail of the document
is never examined, so any personal data there is never masked, and nothing reports that.

This fork processes the whole input in overlapping windows and merges the results. Character offsets
always refer to the original text. Spans found in more than one window are deduplicated by the
existing greedy non-overlap decode, and the overlap is never smaller than `max_width - 1` words, so
an entity straddling a window boundary is still wholly inside one window.

```bash
-Dphileas.pheye.onnx.longTextMode=CHUNK          # default: examine everything
-Dphileas.pheye.onnx.longTextMode=FAIL           # refuse over-long input instead
-Dphileas.pheye.onnx.longTextMode=TRUNCATE       # upstream behaviour; unsafe for redaction
-Dphileas.pheye.onnx.chunkOverlapWords=32        # optional; derived from max_width when unset
-Dphileas.pheye.onnx.maxSequenceTokens=1024      # GLiNER only; see the next section
```

`LongInputWindowingTest` asserts the safety property directly: for a range of input sizes, every
word index is covered by at least one window. On real weights, `TokenClassifierLongInputTest` asserts
it end to end: over a 2,000-word document the detector finds the planted name in every one of the
forty paragraphs, `FAIL` refuses, and `TRUNCATE` is shown to lose the tail — which is why it is not
the default.

Note on quantization: with a low threshold, an INT8 model can label ordinary function words as
entities on long, low-information text, and those noise spans then win the greedy decode and crowd
out real names. If you process long documents, prefer FP32.

### The tokenizer truncates too, and says nothing

Both detectors decide what to examine by counting *words*. Words become an unpredictable number of
sub-tokens — a line of IBANs tokenizes to roughly one token per digit — so a window well inside its
word budget can still exceed the tokenizer's limit. DJL's `HuggingFaceTokenizer.newInstance(path)`
truncates at 512 sub-tokens by default and gives no indication that it did: the encoding simply
ends, and everything past the cut is never scored and never reported.

This module loads the tokenizer with truncation explicitly disabled, so an input that genuinely
will not fit either reaches the encoder in full — see below for what happens then — or is windowed
by the detector before it gets that far. The GLiNER path additionally refuses a window whose text
words did not all survive tokenization, rather than scoring it with the tail missing.
`TokenizersTest` pins both the fix and the DJL default it works around.

### A pathological window can lose everything in it, silently — this is the one to read

This is the most serious defect found while building this module, and the reason a caller worried
about long strings containing personal data should read this section rather than only the summary
table above.

GLiNER windows by **words** (`max_len`). Word count and sub-token count are usually proportional, but
not always: a run of text with no whitespace and almost no vocabulary matches — an embedded
identifier, a base64 blob, a long run of digits — is one or a handful of GLiNER "words" yet can
inflate to thousands of sub-tokens once tokenized, because an out-of-vocabulary run gets encoded
close to one sub-token per character. `max_len` never sees this coming, because it counts words.

Investigating exactly this shape of input against this module's own reference encoder
(`gliner_multi_pii_v1_onnx`, mdeberta-v3-base with a BiLSTM head — `"has_rnn": true` in its
`gliner_config.json`) found real degradation, measured directly, not inferred:

| Sub-tokens in the window | What happened |
|---:|---|
| up to ~1,400 | Both a name near the start and a name near the end of the window are found normally. |
| ~1,700 – ~3,000 | The name near the **end** of the window is lost. The one near the start still isn't. |
| ~3,400 and beyond | **Every detection in the window is gone.** Two clearly-stated Italian names, `person` requested, threshold 0.5 — nothing comes back. |

At no point does anything throw, log, or otherwise indicate a problem. A document that contains real
personal data produces a result indistinguishable from a document that contains none. For a
redaction component this is the worst failure mode there is: worse than a crash, because a crash at
least stops the pipeline from claiming success.

**The fix**: `LocalPhEyeOptions.maxSequenceTokens()` (default 1,024 — comfortably above what an
ordinary `max_len`-sized window of natural-language text encodes to even with a generous label list:
384 words of ordinary Italian financial prose measured at 651 sub-tokens with 3 labels and 919 with
22, roughly 1.7–2.4 sub-tokens per word, and with real margin below where degradation was measured to
begin). Before a window is scored, its
encoded length is checked against this ceiling. Over it, the word range is bisected — plain, exact,
non-overlapping bisection, `[from, mid)` and `[mid, to)` — and each half is checked and scored (or
bisected again) independently, recursively, until every leaf is inside budget or cannot be bisected
any further. A single word whose own encoding alone still exceeds the ceiling — the pathological
content itself, now isolated to its own leaf by the bisection around it — is skipped rather than run
through the encoder at all: no real PII value is anywhere near that long, and running the encoder
over a sequence that size risks the exact blowup the ceiling exists to bound, since a transformer's
attention cost is quadratic in sequence length. Confirmed on the same real weights: real names on
both sides of a single deliberately pathological 200,000-character run are both still found, in
about a second and a half, with nothing skipped except the pathological run itself.

Building this safety net surfaced a second bug in the process, worth naming because the instinct
that caused it is a natural one: the first version widened each bisected half by the module's normal
inter-window overlap, to protect an entity straddling the new internal boundary — reasonable-looking,
and wrong. Once bisection had already narrowed a range below that overlap, extending each half by it
covered the *entire* original range on both sides, so the two "halves" were the same window as their
parent and the recursion never made progress. This was not a theoretical concern: it reproduced as an
actual multi-minute hang against the real model before the exponential blow-up was traced and fixed.
Plain, unpadded bisection has no such failure mode — it cannot, because each half is provably smaller
than its parent — at the cost of a much narrower gap: an entity wide enough to straddle a bisection
point placed deep in the recursion, immediately next to whatever pathological content triggered it,
can be missed. That trade is what buys the guarantee that matters: this can no longer silently lose
everything in a window, or hang the process, regardless of what the window contains.

`GlinerSequenceLengthSafetyTest` pins the bisection mechanism deterministically and fast against the
committed synthetic fixture (including a regression test with a tight timeout for the exact hang
described above). `GlinerLongSequenceDegradationTest` reproduces the original failure on real weights,
at the actual sizes it was measured at, and confirms both names are now found.

### When a window is longer than the encoder can take

Neither detector declares a universal sub-token capacity for GLiNER, on purpose. `gliner_config.json`'s
`max_len` counts *words*, not sub-tokens, and the entity prompt (one `<<ENT>> label` pair per
requested label) shares the same sequence as the text, so the same `max_len` produces a different
sub-token count depending on how many labels are requested — there is no fixed number to declare that
would be right for every label list. Worse, whether an over-long sequence fails at all is
architecture-specific: an encoder with absolute position embeddings rejects one past its table size,
but one with relative position buckets does not reject long input at all — it does something worse,
which is the subject of the section above. `GlinerLongPromptTest` confirms the "runs without
rejecting" half of this empirically against this module's own reference model (mdeberta-v3-base): 384
words with 60 labels, well past the neighbourhood a 512-token model is usually sized for, runs without
a graph-level error (though, per the section above, not necessarily with usable results). Declaring a
fixed capacity would still be the wrong move even ignoring degradation: for an architecture with
absolute position embeddings it would be too conservative or too late depending on the guess, and
mdeberta-v3-base's own graph never rejects at all regardless of the guess.

What both detectors do instead is add context to the encoder's own rejection when one occurs, rather
than pretending to know a capacity they cannot verify: an `OrtException` from ONNX Runtime becomes an
`IllegalStateException` naming the sequence's sub-token count and, for the token-classification path,
the `max_tokens` value declared in the model directory — turning a bare "non-zero status code" message
naming an internal graph node into something a caller can act on. `OnnxRejectionMessageTest` pins the
wording; the token-classification path *does* have a declared, enforced `max_tokens`, and this is only
reached if that declared value is itself wrong for the encoder.

### Text that is not plain Latin

Three places where the obvious Java call disagrees with the Python the pipelines are specified by,
and all three fail the same way — a span with the right label and a plausible score, pointing at the
wrong characters, and no error anywhere:

- **Offsets.** A fast tokenizer reports character offsets as indices into *code points*, which is how
  a Python string is indexed. A Java `String` is indexed by UTF-16 *code units*. One emoji or one
  rarer CJK ideograph earlier in a window shifts every later span by one per surrogate pair.
- **Whitespace.** `Character.isWhitespace` deliberately excludes the non-breaking spaces; Python's
  `str.isspace()` includes them. Text extracted from HTML, PDF or Word is full of U+00A0.
- **Word characters.** `Character.isLetterOrDigit` counts only decimal digits; `str.isalnum()` also
  counts letter-numbers and other-numbers (`Ⅷ`, `½`, `①`), which decides how far a partial span is
  widened.

`TextOffsets` holds all three, and the whitespace-run splitter carries `UNICODE_CHARACTER_CLASS` for
the same reason the GLiNER one does. Parity is checked on a corpus built to contain them.

A single "word" can still be tens of kilobytes — an embedded identifier, a base64 blob — and no
amount of word-level windowing breaks it up. Past `max_tokens`, the token-classification detector
falls back to windowing that word's sub-tokens, so every sub-token is examined and character offsets
stay exact. It does not split any earlier: doing so would change what the model sees relative to the
reference pipeline, and so move the operating point the threshold was calibrated on.

## Labels

GLiNER is zero-shot: labels are the prompt, and their wording changes results. `PhEyeFilter` maps
`person` and `name` to `FilterType.PERSON` and everything else to `FilterType.OTHER`, while keeping
the raw label as the span's classification.

A token classifier's taxonomy is fixed, so the labels you pass **select from it** rather than
defining it, case-insensitively. Asking only for labels the model cannot emit is an **error**, not
an empty result: the two are indistinguishable downstream, and the empty one hides a
misconfiguration behind what looks like a clean document.
`LocalTokenClassifierDetector.entityTypes()` lists what a loaded model emits.

## Build

```bash
mvn clean verify                                          # Java 21

PHILEAS_GLINER_MODEL_DIR=/models/gliner mvn test           # GLiNER real-model tests
PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR=/models/tc mvn test     # token-classification real-model tests
```

Tests needing a model skip with a clear message when it is absent. CI runs `mvn clean verify` on
Java 21 and fails if any produced class file is newer than major 65.

## Parity with the reference pipeline

A port of a redaction pipeline is only useful if it agrees with the pipeline it ports, span for
span. Both families are checked against their Python reference:

- **GLiNER.** `LocalPhEyeDetectorParityTest` drives a committed synthetic ONNX fixture through the
  real runtime and tokenizer; `LocalPhEyeDetectorRealModelParityTest` reproduces
  `gliner.predict_entities` on real weights.
- **Token classification.** `LocalTokenClassifierParityTest` reproduces the HuggingFace
  `token-classification` pipeline (`aggregation_strategy="simple"`) on a committed fixture, at two
  thresholds, including multi-window documents. The fixture is generated by
  `scripts/generate_token_classification_fixture.py` and never edited by hand to match the Java.

For a whole corpus rather than a fixture — which is where an implementation actually drifts: an
off-by-one in a window offset, a boundary entity counted twice, a tie broken the other way:

```bash
mvn -q test-compile dependency:build-classpath -Dmdep.outputFile=target/classpath.txt -Dmdep.includeScope=test
python scripts/cross_check_against_reference.py \
    --model-dir MODEL_DIR --weights HF_CHECKPOINT \
    --documents corpus.jsonl --threshold 0.98
```

It runs the Python reference and this module over the same documents and reports every document
whose spans differ. Adopting a new model means getting a clean run out of it first.

## Verified distribution

`v1.2.0` was built, tested, published and consumed end to end:

| Step | Result |
|---|---|
| Java 21 CI (`build.yml`) | [run 32341662397](https://github.com/mapo80/phileas-pheye-onnx/actions/runs/32341662397) — success |
| Release workflow (`release.yml`) | [run 32342520568](https://github.com/mapo80/phileas-pheye-onnx/actions/runs/32342520568) — success |
| GitHub Packages — publish | `mvn deploy` BUILD SUCCESS in the release run |
| GitHub Packages — consume | [consumer-verify](https://github.com/mapo80/phileas-pheye-onnx/actions/workflows/consumer-verify.yml) resolves `1.2.0` from `maven.pkg.github.com` into an empty temporary repository, 8/8 tests green |
| GitHub Release | [v1.2.0](https://github.com/mapo80/phileas-pheye-onnx/releases/tag/v1.2.0) — jar, sources, javadoc, all with `.sha256` |
| Clean consumer | `examples/consumer-verification`: separate Java 21 project, not a Maven module of this one |

Consumption is verified in CI rather than from a developer machine on purpose: `maven.pkg.github.com`
requires the `read:packages` scope even for a public package, and GitHub exposes no API to mint a
token, so it cannot be automated locally. A workflow does not need one — the built-in `GITHUB_TOKEN`
already carries `packages: read`.

The consumer test checks what a third party actually cares about: the jar is Java 21 bytecode
(class-file major 65), local ONNX inference works, a custom per-label threshold is honoured, input
past `max_len` is still scanned with correct offsets, `FAIL` mode refuses over-long input, a broken
model directory is refused, and inference completes with an unroutable HTTP proxy configured.

## License

Apache 2.0, as upstream.
