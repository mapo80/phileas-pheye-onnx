/*
 *     Copyright 2025 Philterd, LLC @ https://www.philterd.ai
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.philterd.phileas.pheye.onnx;

import ai.philterd.phileas.pheye.onnx.LocalPhEyeOptions.LongTextMode;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeSpan;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The two behaviours this fork adds, verified against a real exported GLiNER model.
 *
 * <p>Set {@code PHILEAS_GLINER_MODEL_DIR} to a model directory to enable these; they skip
 * otherwise, since neither claim can be checked without a real model.
 *
 * <p><b>Use a well-calibrated model.</b> These tests were also run against the INT8 quantization of
 * gliner_multi_pii-v1, where they fail -- not because windowing is wrong, but because at the low
 * threshold that variant needs in order to find names at all, it also labels ordinary Italian
 * function words ("dalle", "condizioni contrattuali") as ORGANIZATION. Those noise spans then win
 * the greedy non-overlap and crowd the real name out. That is a property of the quantized model, and
 * a good reason to prefer FP32 for long inputs.
 */
class RealModelThresholdAndLongInputTest {

    /** A threshold low enough to expose the calibration of any of the GLiNER models we evaluate. */
    private static final double LOW_THRESHOLD = 0.20;

    private static final List<String> LABELS = List.of("person", "organization", "address");

    private static final String NAME = "Gianluca Bellafronte";

    private static Path modelDir() {
        final String dir = System.getenv("PHILEAS_GLINER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(),
                "PHILEAS_GLINER_MODEL_DIR is not set; skipping the real-model tests.");
        final Path path = Path.of(dir);
        assumeTrue(Files.isDirectory(path) && Files.exists(path.resolve("gliner_config.json")),
                "PHILEAS_GLINER_MODEL_DIR does not look like a GLiNER model directory: " + path);
        return path;
    }

    // ------------------------------------------------------------------ threshold

    @Test
    @DisplayName("At 0.5 the configurable detector returns exactly what the hardcoded 0.5 returned")
    void thresholdParityAtDefault() throws Exception {

        final Path dir = modelDir();
        final String text = "Il cliente Mario Rossi, residente in Via Roma 15 a Lecce, ha aperto un conto "
                + "presso la filiale di Alfa S.r.l. insieme a Maria De Luca.";

        try (final LocalPhEyeDetector atDefault = new LocalPhEyeDetector(dir);
             final LocalPhEyeDetector atLow = new LocalPhEyeDetector(dir, LOW_THRESHOLD)) {

            assertEquals(LocalPhEyeOptions.DEFAULT_DETECTION_THRESHOLD, atDefault.options().detectionThreshold(),
                    "the no-arg constructor must keep the upstream 0.5 default");

            final List<PhEyeSpan> viaDefault = atDefault.detect(text, LABELS, "ctx", 0);

            // Lowering the floor may only ADD spans. Everything above 0.5 must come back identically,
            // because the greedy decode places candidates in descending score order.
            final List<PhEyeSpan> aboveHalf = atLow.detect(text, LABELS, "ctx", 0).stream()
                    .filter(s -> s.getScore() > LocalPhEyeOptions.DEFAULT_DETECTION_THRESHOLD)
                    .collect(Collectors.toList());

            assertEquals(describe(viaDefault), describe(aboveHalf),
                    "the configurable detector diverges from the upstream 0.5 behaviour");

        }

    }

    @Test
    @DisplayName("A lower threshold is a superset: it never loses a detection the default found")
    void lowerThresholdIsASuperset() throws Exception {

        final Path dir = modelDir();
        final String text = "Il cliente Gianluca Bellafronte ha richiesto informazioni sul proprio conto.";

        try (final LocalPhEyeDetector atDefault = new LocalPhEyeDetector(dir);
             final LocalPhEyeDetector atLow = new LocalPhEyeDetector(dir, LOW_THRESHOLD)) {

            final List<PhEyeSpan> high = atDefault.detect(text, LABELS, "ctx", 0);
            final List<PhEyeSpan> low = atLow.detect(text, LABELS, "ctx", 0);

            assertTrue(low.size() >= high.size(), "lowering the threshold reduced the number of spans");
            for (final PhEyeSpan span : high) {
                assertTrue(low.stream().anyMatch(s -> s.getStart() == span.getStart()
                                && s.getEnd() == span.getEnd() && s.getLabel().equals(span.getLabel())),
                        "span " + span.getText() + " disappeared when the threshold was lowered");
            }

        }

    }

    // ------------------------------------------------------------------ long input

    @Test
    @DisplayName("A name past the model's word limit is still found in CHUNK mode")
    void nameBeyondMaxLenIsFound() throws Exception {

        final Path dir = modelDir();

        try (final LocalPhEyeDetector detector = new LocalPhEyeDetector(dir,
                new LocalPhEyeOptions(LOW_THRESHOLD, LongTextMode.CHUNK, null))) {

            final int maxWords = detector.maxWords();

            // Well past the limit, and also right at the window boundary, which is the case a naive
            // chunker gets wrong.
            for (final int position : List.of(maxWords + 120, maxWords - 1, maxWords * 2 + 40)) {

                final String text = filler(maxWords * 3, position);
                final List<PhEyeSpan> spans = detector.detect(text, LABELS, "ctx", 0);

                assertTrue(spans.stream().anyMatch(s -> s.getText().contains("Bellafronte")),
                        "no PERSON found with the name at word " + position + " of " + (maxWords * 3)
                                + " (max_len " + maxWords + "); spans=" + describe(spans));

                // The offsets must point at the name in the ORIGINAL text, not inside a window.
                final PhEyeSpan span = spans.stream()
                        .filter(s -> s.getText().contains("Bellafronte")).findFirst().orElseThrow();
                assertEquals(NAME, text.substring(span.getStart(), span.getEnd()).trim(),
                        "the reported offsets do not map back onto the original text");

            }

        }

    }

    @Test
    @DisplayName("TRUNCATE mode demonstrably loses that name: the reason CHUNK is the default")
    void truncateModeLosesTheTail() throws Exception {

        final Path dir = modelDir();

        try (final LocalPhEyeDetector chunking = new LocalPhEyeDetector(dir,
                new LocalPhEyeOptions(LOW_THRESHOLD, LongTextMode.CHUNK, null));
             final LocalPhEyeDetector truncating = new LocalPhEyeDetector(dir,
                     new LocalPhEyeOptions(LOW_THRESHOLD, LongTextMode.TRUNCATE, null))) {

            final int maxWords = chunking.maxWords();
            final String text = filler(maxWords * 2, maxWords + 200);

            assertTrue(chunking.detect(text, LABELS, "ctx", 0).stream()
                            .anyMatch(s -> s.getText().contains("Bellafronte")),
                    "CHUNK mode should find the name past the limit");

            assertFalse(truncating.detect(text, LABELS, "ctx", 0).stream()
                            .anyMatch(s -> s.getText().contains("Bellafronte")),
                    "TRUNCATE mode unexpectedly found text beyond max_len");

        }

    }

    @Test
    @DisplayName("FAIL mode refuses over-long input instead of half-reading it")
    void failModeRefusesLongInput() throws Exception {

        final Path dir = modelDir();

        try (final LocalPhEyeDetector detector = new LocalPhEyeDetector(dir,
                new LocalPhEyeOptions(LOW_THRESHOLD, LongTextMode.FAIL, null))) {

            final String text = filler(detector.maxWords() * 2, 10);
            assertThrows(IllegalArgumentException.class, () -> detector.detect(text, LABELS, "ctx", 0));

        }

    }

    // ------------------------------------------------------------------ helpers

    /**
     * Neutral Italian banking prose, {@code totalWords} long, with the test name at
     * {@code namePosition}.
     *
     * <p>The vocabulary is deliberately varied. An earlier version of this helper repeated a single
     * word, and the model responded by labelling hundreds of those repetitions as PERSON and
     * ORGANIZATION -- degenerate low-information input drives GLiNER into nonsense, and the garbage
     * spans then crowded out the real name in the greedy decode. Realistic filler keeps the test
     * measuring windowing rather than that pathology.
     */
    private static String filler(final int totalWords, final int namePosition) {

        final String[] vocabulary = {
                "la", "pratica", "risulta", "aperta", "presso", "la", "filiale", "di", "riferimento",
                "e", "il", "rapporto", "presenta", "un", "saldo", "disponibile", "regolare",
                "secondo", "quanto", "previsto", "dalle", "condizioni", "contrattuali", "vigenti",
                "l", "operazione", "e", "stata", "registrata", "nella", "giornata", "contabile",
                "con", "valuta", "coerente", "rispetto", "alla", "data", "di", "esecuzione",
                "il", "documento", "allegato", "riporta", "gli", "estremi", "necessari",
                "per", "la", "verifica", "amministrativa", "successiva"
        };

        final List<String> words = new ArrayList<>(totalWords);
        for (int i = 0; i < totalWords; i++) {
            words.add(vocabulary[i % vocabulary.length]);
        }

        // The name occupies two words, so it also exercises a span straddling a window edge.
        final int start = Math.max(0, Math.min(namePosition, totalWords - 2));
        words.set(start, "Gianluca");
        words.set(start + 1, "Bellafronte");

        return String.join(" ", words);

    }

    private static String describe(final List<PhEyeSpan> spans) {
        return spans.stream()
                .map(s -> s.getLabel() + "[" + s.getStart() + "," + s.getEnd() + "]='" + s.getText() + "'")
                .collect(Collectors.joining(", "));
    }

}
