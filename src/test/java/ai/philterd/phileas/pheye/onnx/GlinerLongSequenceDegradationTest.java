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
import org.junit.jupiter.api.Timeout;

import java.nio.file.Path;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The defect {@link LocalPhEyeOptions#maxSequenceTokens()} exists to close, reproduced and confirmed
 * fixed on the real weights it was found on.
 *
 * <p>A run of low-information text with no whitespace -- an embedded identifier, a base64 blob, a
 * long run of digits -- stays a handful of GLiNER "words" (the splitter groups it as one or a few
 * contiguous runs) but inflates sub-token count far out of proportion, because it has almost no
 * vocabulary matches. Investigating this fork's own reference encoder found the model's output
 * degrading gradually from roughly 1,700 sub-tokens in one window and collapsing to zero detections
 * by roughly 3,400 -- with <b>no exception, no error, nothing</b> to distinguish the failure from a
 * clean document. That is a silent, complete loss of real personal data around the pathological
 * content, which is precisely the class of failure this whole fork exists to close.
 *
 * <p>{@link GlinerSequenceLengthSafetyTest} pins the bisection mechanism fast and deterministically
 * against the synthetic fixture. This test is the other half: it runs the actual failing shape of
 * input against the actual model it failed on, at actual measured failure sizes, and asserts the
 * real names on both sides of the pathological run are still found. Gated on
 * {@code PHILEAS_GLINER_MODEL_DIR}; skips otherwise, since it needs the real weights the defect was
 * measured against.
 */
class GlinerLongSequenceDegradationTest {

    private static final String BEFORE = "Il sig. Mario Rossi, nato a Milano, residente in Via "
            + "Garibaldi 24, ha depositato il documento.";
    private static final String AFTER = "Il documento e' stato controfirmato dall'avvocato Giulia "
            + "Bianchi presso lo studio di Torino.";

    /** A deterministic run of characters with no vocabulary matches and no whitespace to split on. */
    private static String pathologicalRun(final int characters) {
        final String alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        final RandomGenerator random = RandomGeneratorFactory.of("Random").create(20260830L);
        final StringBuilder text = new StringBuilder(characters);
        for (int i = 0; i < characters; i++) {
            text.append(alphabet.charAt(random.nextInt(alphabet.length())));
        }
        return text.toString();
    }

    private static Path modelDirectory() {
        final String dir = System.getenv("PHILEAS_GLINER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(), "Set PHILEAS_GLINER_MODEL_DIR to run this.");
        return Path.of(dir);
    }

    @Test
    @DisplayName("Names on both sides of a pathological run are found at sizes that used to lose both")
    @Timeout(30)
    void bothNamesSurviveAcrossMeasuredFailureSizes() throws Exception {

        try (LocalPhEyeDetector detector =
                     new LocalPhEyeDetector(modelDirectory(), LocalPhEyeOptions.withThreshold(0.5))) {

            // 2,500 characters measured at the point where the first name started being lost;
            // 5,000 and 20,000 measured at total collapse -- zero detections, either name. 200,000
            // is two orders of magnitude past that collapse point, to confirm the fix does not merely
            // push the failure further out.
            for (final int characters : new int[]{2_500, 5_000, 20_000, 200_000}) {

                final String text = BEFORE + " " + pathologicalRun(characters) + " " + AFTER;

                final List<PhEyeSpan> spans = assertDoesNotThrow(
                        () -> detector.detect(text, List.of("person"), "", 0),
                        () -> "threw on a " + characters + "-character pathological run");

                for (final PhEyeSpan span : spans) {
                    assertEquals(text.substring(span.getStart(), span.getEnd()), span.getText());
                }

                assertTrue(spans.stream().anyMatch(s -> "Mario Rossi".equals(s.getText())),
                        () -> "'Mario Rossi' was lost at " + characters + " pathological characters: " + spans);
                assertTrue(spans.stream().anyMatch(s -> "Giulia Bianchi".equals(s.getText())),
                        () -> "'Giulia Bianchi' was lost at " + characters + " pathological characters: " + spans);

            }

        }

    }

    @Test
    @DisplayName("Detection over a pathological run completes quickly, not just eventually")
    @Timeout(15)
    void completesQuickly() throws Exception {

        // A generous bound: this fork measured well under two seconds for 200,000 characters.
        // What this guards against is a return of quadratic-or-worse blowup in a different shape
        // than the one already fixed -- a slow pass here is as much a regression as an exception.
        try (LocalPhEyeDetector detector =
                     new LocalPhEyeDetector(modelDirectory(), LocalPhEyeOptions.withThreshold(0.5))) {
            final String text = BEFORE + " " + pathologicalRun(200_000) + " " + AFTER;
            assertDoesNotThrow(() -> detector.detect(text, List.of("person"), "", 0));
        }

    }

}
