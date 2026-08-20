package it.linksmt.consumer;

import ai.philterd.phileas.pheye.onnx.LocalPhEyeDetector;
import ai.philterd.phileas.pheye.onnx.LocalPhEyeOptions;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeSpan;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Proves the published artifact works for a third party.
 *
 * <p>Nothing here can pass by accident from the development checkout: the dependency comes from
 * GitHub Packages and the build uses an empty temporary local repository. If the artifact were
 * missing or broken, this module would not even compile.
 */
class PublishedArtifactTest {

    private static final List<String> LABELS = List.of("person", "organization", "address");

    private static Path modelDir() {
        final String dir = System.getenv("CONSUMER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(), "CONSUMER_MODEL_DIR not set");
        final Path path = Path.of(dir);
        assumeTrue(Files.isDirectory(path), "not a model directory: " + path);
        return path;
    }

    @Test
    @DisplayName("The published classes are the fork's, loaded from the GitHub Packages jar")
    void artifactIsTheFork() {
        // Both types exist only in the fork; upstream 1.0.0 has neither.
        assertEquals("ai.philterd.phileas.pheye.onnx.LocalPhEyeOptions", LocalPhEyeOptions.class.getName());
        assertEquals(0.5, LocalPhEyeOptions.DEFAULT_DETECTION_THRESHOLD);
        assertTrue(LocalPhEyeOptions.DecodeStrategy.valueOf("PER_LABEL_GREEDY") != null);

        final String source = LocalPhEyeDetector.class.getProtectionDomain()
                .getCodeSource().getLocation().toString();
        System.out.println("LocalPhEyeDetector loaded from: " + source);
        assertTrue(source.contains("phileas-pheye-onnx-1.2.0.jar"),
                "must come from the published 1.2.0 jar, got " + source);
    }

    @Test
    @DisplayName("The published jar is Java 21 bytecode")
    void bytecodeIsJava21() throws Exception {
        final var stream = LocalPhEyeDetector.class.getClassLoader()
                .getResourceAsStream("ai/philterd/phileas/pheye/onnx/LocalPhEyeDetector.class");
        assertTrue(stream != null, "class resource not found");
        try (stream) {
            final byte[] header = stream.readNBytes(8);
            final int major = ((header[6] & 0xFF) << 8) | (header[7] & 0xFF);
            System.out.println("class-file major version: " + major);
            assertTrue(major <= 65, "major " + major + " cannot be loaded by a Java 21 JVM");
        }
    }

    @Test
    @DisplayName("Local ONNX inference finds Gianluca Bellafronte")
    void detectsBellafronte() throws Exception {
        try (final LocalPhEyeDetector detector = new LocalPhEyeDetector(modelDir(), 0.20)) {
            final List<PhEyeSpan> spans = detector.detect(
                    "Il cliente Gianluca Bellafronte ha richiesto informazioni sul proprio conto.",
                    LABELS, "consumer", 0);
            assertTrue(spans.stream().anyMatch(s -> s.getText().equals("Gianluca Bellafronte")),
                    "expected the full name, got " + describe(spans));
        }
    }

    @Test
    @DisplayName("A custom per-label threshold and PER_LABEL_GREEDY are honoured")
    void customThresholdAndStrategy() throws Exception {

        final LocalPhEyeOptions options = LocalPhEyeOptions.of(0.50,
                Map.of("person", 0.10), LocalPhEyeOptions.DecodeStrategy.PER_LABEL_GREEDY);

        assertEquals(0.10, options.thresholdFor("person"));
        assertEquals(0.50, options.thresholdFor("organization"));

        try (final LocalPhEyeDetector low = new LocalPhEyeDetector(modelDir(), options);
             final LocalPhEyeDetector high = new LocalPhEyeDetector(modelDir(),
                     LocalPhEyeOptions.of(0.95, Map.of(), LocalPhEyeOptions.DecodeStrategy.PER_LABEL_GREEDY))) {

            final String text = "La pratica è intestata a Gianluca Bellafronte.";
            assertTrue(low.detect(text, LABELS, "c", 0).size()
                    >= high.detect(text, LABELS, "c", 0).size(),
                    "a lower threshold must not return fewer spans");
        }

    }

    @Test
    @DisplayName("Long input past max_len is still scanned, and offsets map to the original text")
    void longContextIsSafe() throws Exception {

        try (final LocalPhEyeDetector detector = new LocalPhEyeDetector(modelDir(),
                new LocalPhEyeOptions(0.20, LocalPhEyeOptions.LongTextMode.CHUNK, null))) {

            final int maxWords = detector.maxWords();
            final String text = filler(maxWords * 2, maxWords + 150);

            final List<PhEyeSpan> spans = detector.detect(text, LABELS, "consumer", 0);
            final PhEyeSpan hit = spans.stream()
                    .filter(s -> s.getText().contains("Bellafronte")).findFirst().orElse(null);

            assertTrue(hit != null, "name past max_len was not found; spans=" + spans.size());
            assertEquals("Gianluca Bellafronte", text.substring(hit.getStart(), hit.getEnd()).trim(),
                    "offsets must address the original document");
        }

    }

    @Test
    @DisplayName("FAIL mode refuses over-long input rather than half-reading it")
    void failClosedOnLongInput() throws Exception {
        try (final LocalPhEyeDetector detector = new LocalPhEyeDetector(modelDir(),
                new LocalPhEyeOptions(0.20, LocalPhEyeOptions.LongTextMode.FAIL, null))) {
            final String text = filler(detector.maxWords() * 2, 10);
            assertThrows(IllegalArgumentException.class, () -> detector.detect(text, LABELS, "c", 0));
        }
    }

    @Test
    @DisplayName("A broken model directory is refused, never a silent no-op detector")
    void failClosedOnBrokenModel() {
        assertThrows(Exception.class, () -> new LocalPhEyeDetector(Path.of("/nonexistent/model/dir")));
    }

    @Test
    @DisplayName("Inference performs no network I/O")
    void inferenceIsOffline() throws Exception {
        // The JVM is started with an unroutable proxy by run-consumer-test.sh; any HTTP attempt
        // would fail rather than silently succeed.
        try (final LocalPhEyeDetector detector = new LocalPhEyeDetector(modelDir(), 0.20)) {
            final List<PhEyeSpan> spans = detector.detect(
                    "Il beneficiario Gianluca Bellafronte risiede in Via Roma 15.", LABELS, "c", 0);
            assertFalse(spans.isEmpty(), "offline inference returned nothing");
        }
    }

    private static String filler(final int totalWords, final int namePosition) {
        final String[] vocabulary = {
                "la", "pratica", "risulta", "aperta", "presso", "la", "filiale", "di", "riferimento",
                "e", "il", "rapporto", "presenta", "un", "saldo", "disponibile", "regolare",
                "secondo", "quanto", "previsto", "dalle", "condizioni", "contrattuali", "vigenti"};
        final List<String> words = new ArrayList<>(totalWords);
        for (int i = 0; i < totalWords; i++) {
            words.add(vocabulary[i % vocabulary.length]);
        }
        final int start = Math.max(0, Math.min(namePosition, totalWords - 2));
        words.set(start, "Gianluca");
        words.set(start + 1, "Bellafronte");
        return String.join(" ", words);
    }

    private static String describe(final List<PhEyeSpan> spans) {
        final StringBuilder builder = new StringBuilder();
        spans.forEach(s -> builder.append(s.getLabel()).append("='").append(s.getText()).append("' "));
        return builder.toString();
    }

}
