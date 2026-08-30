# phileas-pheye-onnx

`phileas-pheye-onnx` adds optional local, on-device inference to the [Phileas](https://philterd.github.io/phileas/) PhEye filter, using ONNX Runtime. It lets Phileas run a model in-process, with no separate PhEye server.

Two model families are supported: zero-shot **GLiNER** span models, and fixed-taxonomy **BIO token classifiers**. The [model directory](model-directory.md)'s layout decides which is loaded; nothing else about using the module changes.

Core Phileas talks to a [PhEye](https://www.philterd.ai/ph-eye/) service over HTTP. This module lets Phileas load and run the model itself instead. It is shipped as a separate artifact so that Phileas core stays lightweight: only applications that want local inference pull in the native ONNX Runtime and tokenizer dependencies.

## When to use local vs. remote PhEye

Phileas supports two ways to run PhEye detection. Both produce the same kind of detections; they differ in where the model runs.

- **Remote PhEye (default).** Phileas calls a PhEye HTTP service. Use this when you run PhEye as a shared service, want to scale model serving independently, or run the model on dedicated hardware (for example a GPU host) separate from the application.
- **Local PhEye (this module).** Phileas loads a model from a local directory and runs inference in-process. Use this when you want a single self-contained process, have no PhEye server to call, or need inference to stay entirely inside one application boundary with no network hop.

You select local inference per filter by setting a `modelPath` (see [Configuration](configuration.md)). With no `modelPath`, Phileas uses the remote HTTP detector exactly as before, so adding this module changes nothing until you point a filter at a model directory.

## Installing

Add the dependency alongside `phileas`:

```xml
<dependency>
    <groupId>ai.philterd</groupId>
    <artifactId>phileas-pheye-onnx</artifactId>
    <version>1.0.0</version>
</dependency>
```

Releases are published to Maven Central and resolve with no extra repository configuration.

This module is versioned independently of `phileas` (like [`phisql`](https://github.com/philterd/phisql)), because it has its own API and release cadence. It does implement an SPI from `phileas` core, so each release targets a specific `phileas` version: `phileas-pheye-onnx` `1.0.0` targets `phileas` `4.1.0`. Use a `phileas-pheye-onnx` build that matches the `phileas` you run.

Development builds are published as `-SNAPSHOT` versions to the Maven Central snapshot repository, which is not served from the default Maven Central repository. To use one, add that repository (`https://central.sonatype.com/repository/maven-snapshots/`) to your build. Snapshots are mutable and are periodically pruned, so pin a released version for anything you need to reproduce.

Adding the artifact to the classpath is all that is needed to enable local inference. The module registers its detector provider through `java.util.ServiceLoader`, so Phileas discovers it automatically. See [How It Works](how-it-works.md) for the mechanism.

## Next steps

- [How It Works](how-it-works.md): the SPI, service discovery, and the GLiNER pipeline this module reimplements.
- [Token-Classification Models](token-classification.md): the second model family, and when to prefer it.
- [Model Directory](model-directory.md): the files a local model directory must contain.
- [Configuration](configuration.md): pointing a PhEye filter at a local model.
- [Limitations and Accuracy](limitations.md): the current parity status, and what to confirm before production use.

## Related

- [Phileas documentation](https://philterd.github.io/phileas/), the core redaction library.
- [phileas-pheye-onnx on GitHub](https://github.com/philterd/phileas-pheye-onnx), the source for this module.
