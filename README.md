# phileas-pheye-onnx (mapo80 fork)

Local, in-process GLiNER inference for the Phileas PhEye filter, via ONNX Runtime for Java.

This fork exists to make the module usable as the privacy component of a JVM service. Three things
differ from upstream, each driven by that goal:

| | upstream | this fork |
|---|---|---|
| Java | 25 bytecode | **Java 21** (class-file major 65, enforced in CI) |
| Phileas | 4.1.0 | **4.2.0** |
| ONNX decode threshold | hardcoded `0.5` | **configurable, global and per-label**, default `0.5` |
| Overlapping labels | one flat greedy pass; a stronger ORGANIZATION deletes a correct PERSON | **`PER_LABEL_GREEDY`** keeps both for the caller's resolver |
| Input longer than `max_len` | silently truncated | **overlapping windows** (or fail-closed) |
| Broken model directory | partly deferred | **fail-closed at construction** |

Everything else — the GLiNER prompt construction, tokenization, span enumeration, the six input
tensors, the logits reshape, the greedy non-overlap decode — is upstream's algorithm, unchanged and
parity-tested.

## Architecture

```
your JVM service
  └─ Phileas PhEyeFilter
       └─ PhEyeDetectorProvider (java.util.ServiceLoader)
            └─ LocalPhEyeDetector          ← this module
                 └─ ONNX Runtime Java
                      └─ model.onnx on the local filesystem
```

One process. No HTTP for inference, no Ph-Eye server, no Python, no sidecar, no child process,
CPU-only. Setting `modelPath` on the Phileas `PhEyeConfiguration` is what makes `PhEyeFilter` pick
this detector; leave it unset and Phileas uses its `RemotePhEyeDetector`, which calls a service over
HTTP.

## Requirements

- **Java 21** or newer at runtime (the artifact targets 21)
- Phileas 4.2.0
- A GLiNER model directory on disk

## Maven

Published to GitHub Packages.

```xml
<repositories>
  <repository>
    <id>github</id>
    <url>https://maven.pkg.github.com/mapo80/phileas-pheye-onnx</url>
  </repository>
</repositories>

<dependency>
  <groupId>io.github.mapo80</groupId>
  <artifactId>phileas-pheye-onnx</artifactId>
  <version>1.2.0</version>
</dependency>
```

GitHub Packages requires authentication even for public packages, so add a `github` server to your
`~/.m2/settings.xml` with a personal access token that has `read:packages`:

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

```
<modelDir>/
├── tokenizer.json          HuggingFace fast tokenizer
├── gliner_config.json      span width, max length, prompt tokens
└── onnx/model.onnx         the exported GLiNER model  (or <modelDir>/model.onnx)
```

Any of the ONNX quantizations works as long as it is saved under that name. A missing or unreadable
file, a malformed config, a `words_splitter_type` other than `whitespace`, a non-positive
`max_len`/`max_width`, or an ONNX graph that is not a GLiNER span export all fail at construction.

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

## Long input

A GLiNER model has a hard word limit (`max_len`, typically 384 words). Upstream truncates to it.
For a redaction component that is a data leak rather than a quality issue: the tail of the document
is never examined, so any personal data there is never masked, and nothing reports that.

This fork processes the whole input in overlapping windows and merges the results. Character offsets
always refer to the original text. Spans found in more than one window are deduplicated by the
existing greedy non-overlap decode, and the overlap is never smaller than `max_width - 1` words, so
an entity straddling a window boundary is still wholly inside one window.

```bash
-Dphileas.pheye.onnx.longTextMode=CHUNK      # default: examine everything
-Dphileas.pheye.onnx.longTextMode=FAIL       # refuse over-long input instead
-Dphileas.pheye.onnx.longTextMode=TRUNCATE   # upstream behaviour; unsafe for redaction
-Dphileas.pheye.onnx.chunkOverlapWords=32    # optional; derived from max_width when unset
```

`LongInputWindowingTest` asserts the safety property directly: for a range of input sizes, every
word index is covered by at least one window.

Note on quantization: with a low threshold, an INT8 model can label ordinary function words as
entities on long, low-information text, and those noise spans then win the greedy decode and crowd
out real names. If you process long documents, prefer FP32.

## Labels

GLiNER is zero-shot: labels are the prompt, and their wording changes results. `PhEyeFilter` maps
`person` and `name` to `FilterType.PERSON` and everything else to `FilterType.OTHER`, while keeping
the raw label as the span's classification.

## Build

```bash
mvn clean verify                                  # Java 21
PHILEAS_GLINER_MODEL_DIR=/models/gliner mvn test   # also runs the real-model tests
```

Tests needing a model skip with a clear message when it is absent. CI runs `mvn clean verify` on
Java 21 and fails if any produced class file is newer than major 65.

## Verified distribution

`v1.2.0` was built, tested, published and consumed end to end:

| Step | Result |
|---|---|
| Java 21 CI (`build.yml`) | [run 32341662397](https://github.com/mapo80/phileas-pheye-onnx/actions/runs/32341662397) — success |
| Release workflow (`release.yml`) | [run 32342520568](https://github.com/mapo80/phileas-pheye-onnx/actions/runs/32342520568) — success |
| GitHub Packages | `mvn deploy` BUILD SUCCESS in the release run |
| GitHub Release | [v1.2.0](https://github.com/mapo80/phileas-pheye-onnx/releases/tag/v1.2.0) — jar, sources, javadoc, all with `.sha256` |
| Clean consumer | separate Java 21 project, empty temporary local repository, 8/8 tests green against the published jar |

The consumer test checks what a third party actually cares about: the jar is Java 21 bytecode
(class-file major 65), local ONNX inference works, a custom per-label threshold is honoured, input
past `max_len` is still scanned with correct offsets, `FAIL` mode refuses over-long input, a broken
model directory is refused, and inference completes with an unroutable HTTP proxy configured.

## License

Apache 2.0, as upstream.
