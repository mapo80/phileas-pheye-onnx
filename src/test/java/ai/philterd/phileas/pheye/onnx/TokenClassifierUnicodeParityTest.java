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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Real-weight regressions for text that is not plain BMP Latin.
 *
 * <p>None of these fails loudly when it breaks. A span comes back with the right label and a
 * plausible score, pointing at the wrong characters — so a redaction component masks the wrong
 * characters and reports success. The fixture corpus is Italian financial prose and never exercises
 * any of it; every case here was a live defect found by review rather than by a failing test.
 */
class TokenClassifierUnicodeParityTest {

    private static final char NBSP = 0x00A0;

    private static Path modelDirectory() {
        final String dir = System.getenv("PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(),
                "Set PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR to a token-classification model directory.");
        return Path.of(dir);
    }

    private static LocalTokenClassifierDetector detector() throws Exception {
        return new LocalTokenClassifierDetector(modelDirectory(), LocalPhEyeOptions.withThreshold(0.98));
    }

    @Test
    @DisplayName("An emoji earlier in the text does not shift the spans after it")
    void supplementaryCharactersDoNotShiftSpans() throws Exception {

        // The tokenizer offsets are code-point indices; a Java String is indexed by code units. One
        // emoji is one surrogate pair, so every later span used to come back one character early.
        final String text = "😀 Il sig. Mario Rossi, residente a Roma, ha firmato il contratto.";

        try (final var detector = detector()) {
            final List<PhEyeSpan> spans = detector.detect(text, List.of("FULLNAME", "CITY"), "", 0);

            assertFalse(spans.isEmpty(), "expected at least the name to be found");
            for (final PhEyeSpan span : spans) {
                assertEquals(text.substring(span.getStart(), span.getEnd()), span.getText(),
                        () -> "span " + span.getLabel() + " does not quote the text it points at");
            }
            assertTrue(spans.stream().anyMatch(s -> "Mario Rossi".equals(s.getText())),
                    () -> "expected exactly 'Mario Rossi': " + spans.stream().map(PhEyeSpan::getText).toList());
        }

    }

    @Test
    @DisplayName("A non-breaking space separates words, as it does for the reference")
    void nonBreakingSpaceSeparatesWords() {

        // Python's \S is Unicode-aware, so U+00A0 is a separator there. Java's is not unless asked,
        // and the two disagreeing means different words, different windows, different results.
        final String text = "Mario" + NBSP + "Rossi";

        assertEquals(List.of("Mario", "Rossi"),
                WordsSplitter.splitOnRuns(text).stream().map(WordsSplitter.Word::text).toList());
    }

    @Test
    @DisplayName("A non-breaking space before a name is trimmed off the span")
    void nonBreakingSpaceIsTrimmedFromSpans() throws Exception {

        // This tokenizer's offsets include the separator before a word. Character.isWhitespace says
        // U+00A0 is not whitespace, so the trim used to leave it in and the mask started a
        // character early — on text out of HTML or PDF, which is where NBSP comes from.
        final String text = "Il sig." + NBSP + "Mario Rossi abita a Roma.";

        try (final var detector = detector()) {
            for (final PhEyeSpan span : detector.detect(text, List.of("FULLNAME"), "", 0)) {
                assertEquals(text.substring(span.getStart(), span.getEnd()), span.getText());
                assertFalse(TextOffsets.isSpace(span.getText().charAt(0)),
                        () -> "span starts on whitespace: " + span.getText().codePoints().boxed().toList());
            }
        }

    }

    @Test
    @DisplayName("Asking for one label never loses a span that asking for all of them finds")
    void narrowingTheLabelSetNeverLosesSpans() throws Exception {

        // The reduce is greedy across labels, so filtering after it would let an unwanted entity
        // suppress a wanted one and leave nothing in its place. That failure is silent and open.
        final String text = "Il sig. Mario Rossi, residente in Via Garibaldi 24, 00185 Roma (RM), "
                + "amministratore della Edilnord S.r.l., ha versato euro 12.500,00 il 12/06/1985.";

        try (final var detector = detector()) {

            final List<PhEyeSpan> all = detector.detect(text, detector.entityTypes(), "", 0);

            for (final String label : List.of("FULLNAME", "CITY", "ORG", "STREET", "AMOUNT", "DATE")) {

                final List<String> fromAll = all.stream()
                        .filter(s -> label.equals(s.getLabel())).map(PhEyeSpan::getText).toList();
                final List<String> fromOne = detector.detect(text, List.of(label), "", 0).stream()
                        .map(PhEyeSpan::getText).toList();

                assertTrue(fromOne.containsAll(fromAll),
                        () -> "asking only for " + label + " lost " + fromAll + ", found " + fromOne);
            }

        }

    }

    @Test
    @DisplayName("The model's calibrated threshold applies to a directly constructed detector too")
    void calibratedThresholdAppliesWithoutTheFactory() throws Exception {

        final Path dir = modelDirectory();
        final Double calibrated = TokenClassifierConfig.load(dir).calibratedThreshold;
        assumeTrue(calibrated != null, "this model directory declares no calibrated_threshold");

        try (final var detector = new LocalTokenClassifierDetector(dir)) {
            assertEquals(calibrated, detector.options().detectionThreshold(), 1e-9);
        }
        try (final var detector = new LocalTokenClassifierDetector(dir, LocalPhEyeOptions.defaults())) {
            assertEquals(calibrated, detector.options().detectionThreshold(), 1e-9);
        }

        // A threshold the caller chose is still honoured.
        try (final var detector = new LocalTokenClassifierDetector(dir, LocalPhEyeOptions.withThreshold(0.30))) {
            assertEquals(0.30, detector.options().detectionThreshold(), 1e-9);
        }

    }

}
