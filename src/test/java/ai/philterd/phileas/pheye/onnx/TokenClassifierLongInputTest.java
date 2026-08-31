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

import ai.philterd.phileas.services.filters.ai.pheye.PhEyeSpan;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * How the token-classification detector behaves on input longer than one inference window.
 *
 * <p>The model labels a window of {@code max_words} words at a time. What happens to word
 * {@code max_words + 1} is the difference between a redaction component and a leak, so it is
 * asserted here on real weights rather than argued about: {@code CHUNK} examines everything,
 * {@code FAIL} refuses, and {@code TRUNCATE} is shown to lose the tail -- which is why it is not the
 * default.
 */
class TokenClassifierLongInputTest {

    /** One paragraph, carrying a name, an address and a date. Repeated to build long documents. */
    private static final String PARAGRAPH =
            "Il ricorrente Giovanni Battista Lombardi, nato il 27/11/1972 a Napoli, residente in "
                    + "Via Chiaia 88, 80132 Napoli, ha depositato la memoria il 15/04/2024 presso la "
                    + "cancelleria, allegando la documentazione richiesta e la ricevuta di pagamento del "
                    + "contributo unificato dovuto per il grado di giudizio in corso di trattazione. ";

    private static Path modelDirectory() {
        final String dir = System.getenv("PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(),
                "Set PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR to a token-classification model directory.");
        return Path.of(dir);
    }

    private static String repeated(final int times) {
        return PARAGRAPH.repeat(times);
    }

    private static LocalPhEyeOptions options(final LocalPhEyeOptions.LongTextMode mode) {
        return new LocalPhEyeOptions(0.98, mode, null);
    }

    @Test
    @DisplayName("CHUNK finds the name in every repetition, from the first to the last")
    void chunkExaminesTheWholeDocument() throws Exception {

        // 40 repetitions is roughly 2,000 words: about 20 windows, so nothing here fits in one pass.
        final int repetitions = 40;
        final String text = repeated(repetitions);

        try (final var detector = new LocalTokenClassifierDetector(modelDirectory(),
                options(LocalPhEyeOptions.LongTextMode.CHUNK))) {

            final List<PhEyeSpan> spans = detector.detect(text, detector.entityTypes(), "", 0);

            for (int i = 0; i < repetitions; i++) {
                final int from = i * PARAGRAPH.length();
                final int to = from + PARAGRAPH.length();
                final int repetition = i;
                assertTrue(spans.stream().anyMatch(s -> "FULLNAME".equals(s.getLabel())
                                && s.getStart() >= from && s.getEnd() <= to),
                        () -> "no FULLNAME found in repetition " + repetition + " of " + repetitions
                                + "; that stretch of the document was not examined");
            }

            // Every span must still quote the text it points at: an off-by-one in the window offset
            // would show up here as a shifted substring rather than as a missing span.
            for (final PhEyeSpan span : spans) {
                assertEquals(text.substring(span.getStart(), span.getEnd()), span.getText());
            }

        }

    }

    @Test
    @DisplayName("An entity split across a window boundary is recovered whole, and once")
    void entitiesStraddlingAWindowBoundaryAreRecovered() throws Exception {

        try (final var detector = new LocalTokenClassifierDetector(modelDirectory(),
                options(LocalPhEyeOptions.LongTextMode.CHUNK))) {

            // Place the name so its first word is the last word of window 1 and the rest is not:
            // only the overlap into window 2 can recover it whole.
            final int filler = detector.maxWords() - 1;
            final String prefix = "parola ".repeat(filler);
            final String text = prefix + "Giovanni Battista Lombardi ha depositato la memoria.";

            final List<PhEyeSpan> spans = detector.detect(text, List.of("FULLNAME"), "", 0);

            assertEquals(1, spans.size(), spans.toString());
            assertEquals("Giovanni Battista Lombardi", spans.get(0).getText());
            assertEquals(prefix.length(), spans.get(0).getStart());
        }

    }

    @Test
    @DisplayName("FAIL refuses an over-long document instead of examining part of it")
    void failModeRefusesOverLongInput() throws Exception {

        try (final var detector = new LocalTokenClassifierDetector(modelDirectory(),
                options(LocalPhEyeOptions.LongTextMode.FAIL))) {

            final Exception e = assertThrows(IllegalArgumentException.class,
                    () -> detector.detect(repeated(10), detector.entityTypes(), "", 0));
            assertTrue(e.getMessage().contains("longTextMode is FAIL"), e.getMessage());

            // A document that fits is still processed normally in this mode.
            assertFalse(detector.detect(PARAGRAPH, detector.entityTypes(), "", 0).isEmpty());
        }

    }

    @Test
    @DisplayName("TRUNCATE loses the tail, which is why CHUNK is the default")
    void truncateLosesTheTail() throws Exception {

        final int repetitions = 20;
        final String text = repeated(repetitions);

        try (final var chunked = new LocalTokenClassifierDetector(modelDirectory(),
                options(LocalPhEyeOptions.LongTextMode.CHUNK));
             final var truncated = new LocalTokenClassifierDetector(modelDirectory(),
                     options(LocalPhEyeOptions.LongTextMode.TRUNCATE))) {

            final int tail = (repetitions - 1) * PARAGRAPH.length();

            assertTrue(chunked.detect(text, chunked.entityTypes(), "", 0).stream()
                            .anyMatch(s -> s.getStart() >= tail),
                    "CHUNK must reach the last paragraph");
            assertTrue(truncated.detect(text, truncated.entityTypes(), "", 0).stream()
                            .noneMatch(s -> s.getStart() >= tail),
                    "TRUNCATE was expected to stop at the first window; if it no longer does, this test"
                            + " is asserting the wrong thing, not the code being safer");
        }

    }

    @Test
    @DisplayName("A caller-supplied overlap is used in place of the model directory's")
    void callerSuppliedOverlapIsUsed() throws Exception {

        final LocalPhEyeOptions wide = new LocalPhEyeOptions(0.98,
                LocalPhEyeOptions.LongTextMode.CHUNK, 60);

        try (final var detector = new LocalTokenClassifierDetector(modelDirectory(), wide)) {
            assertEquals(60, detector.chunkOverlapWords());
            // A wider overlap must not change what is found, only how often each window sees it.
            final String text = repeated(6);
            assertFalse(detector.detect(text, detector.entityTypes(), "", 0).isEmpty());
        }

    }

    @Test
    @DisplayName("An overlap wide enough to stall the windowing is clamped, not obeyed")
    void overlapIsClampedBelowTheWindow() throws Exception {

        // Honouring this literally would make the stride zero and the loop never advance.
        final LocalPhEyeOptions absurd = new LocalPhEyeOptions(0.98,
                LocalPhEyeOptions.LongTextMode.CHUNK, 10_000);

        try (final var detector = new LocalTokenClassifierDetector(modelDirectory(), absurd)) {
            assertEquals(detector.maxWords() - 1, detector.chunkOverlapWords());
            assertFalse(detector.detect(repeated(4), detector.entityTypes(), "", 0).isEmpty());
        }

    }

    @Test
    @DisplayName("A single multi-kilobyte token blob is scanned rather than truncated")
    void oversizedSingleWordIsStillScanned() throws Exception {

// One "word" of tens of kilobytes -- an embedded identifier or base64 blob -- encodes to
        // more sub-tokens than the encoder can take at all. Windowing by words cannot split it, so
        // the detector falls back to sub-token windows. The property under test is that this
        // terminates and keeps offsets exact, not that anything is found inside the blob.
        final String blob = "A1b2C3d4E5f6G7h8".repeat(1200);
        final String text = "Allegato " + blob + " trasmesso da Giovanni Battista Lombardi il 15/04/2024.";

        try (final var detector = new LocalTokenClassifierDetector(modelDirectory(),
                options(LocalPhEyeOptions.LongTextMode.CHUNK))) {

            final List<PhEyeSpan> spans = detector.detect(text, detector.entityTypes(), "", 0);

            for (final PhEyeSpan span : spans) {
                assertEquals(text.substring(span.getStart(), span.getEnd()), span.getText());
            }
            assertTrue(spans.stream().anyMatch(s -> "FULLNAME".equals(s.getLabel())
                            && "Giovanni Battista Lombardi".equals(s.getText())),
                    () -> "the text after the blob was not examined: " + spans);
        }

    }

}
