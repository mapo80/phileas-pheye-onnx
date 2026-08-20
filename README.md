# Maven repository

A plain Maven repository layout, served straight from this branch. It exists so the artifact can be
consumed **without credentials**.

GitHub Packages requires authentication for Maven even when the package is public — there is no
setting to allow anonymous reads — so it cannot be used by a consumer that has no token. This branch
closes that gap.

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

No `<server>` entry, no token. `raw.githubusercontent.com` serves public repositories anonymously.

Written by the `release` workflow on every `vX.Y.Z` tag; do not commit here by hand.

For a wider audience prefer Maven Central, which is CDN-backed and has no rate limits. That needs a
Sonatype Central account and a GPG signing key, so it is a deliberate choice rather than something
CI can arrange on its own.
