# Consumer verification

Verifies the **published** artifact, not the one this repository builds.

Run by `.github/workflows/consumer-verify.yml`, which:

1. resolves `io.github.mapo80:phileas-pheye-onnx` from **GitHub Packages**, authenticating with the
   workflow's built-in `GITHUB_TOKEN` (which carries `packages: read`);
2. uses an **empty temporary local repository**, so nothing can be satisfied by a previous
   `mvn install`;
3. downloads a real GLiNER model from HuggingFace;
4. runs the tests with an **unroutable HTTP proxy**, so inference cannot reach the network.

That combination is what makes the result meaningful: if the package were missing, unreadable, built
for the wrong Java version, or broken, this module would fail.

It is deliberately not a Maven module of the parent: a reactor build would resolve the dependency
from the sibling `target/` directory and prove nothing.

To run it locally you need a token with `read:packages`:

```bash
mvn -s settings.xml -Dmaven.repo.local=$(mktemp -d) test
```
