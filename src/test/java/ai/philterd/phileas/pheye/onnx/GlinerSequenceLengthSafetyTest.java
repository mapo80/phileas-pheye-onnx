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

import java.net.URL;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The mechanics of {@link LocalPhEyeOptions#maxSequenceTokens()} against the committed synthetic
 * fixture, forced to trigger by setting the ceiling far below what any real window needs.
 *
 * <p>This exists to pin the mechanism fast and deterministically, without a multi-hundred-MB model:
 * the fixture's graph produces a known, fixed span (word 0, width 1, label 0; see
 * {@code gliner-fixture/README.md}) regardless of content, so a leaf window that ends up covering
 * global word 0 must find it, whatever bisection happened to reach that leaf. This is the
 * regression test for two real defects found while building this safety net, both confirmed against
 * real weights before being fixed here:
 *
 * <ol>
 *   <li>An earlier version widened each bisected half by the top-level {@code chunkOverlapWords}.
 *       Once a range had already been narrowed below that overlap, both halves came back covering
 *       the entire parent range, so the recursion never made progress -- an actual hang against real
 *       weights, reproduced here in miniature via {@link #recursionTerminatesQuicklyEvenWhenForced()}
 *       with a generous {@link Timeout} as a tripwire against reintroducing it.</li>
 *   <li>A single word whose own encoding exceeds the ceiling cannot be shrunk by bisecting on word
 *       count at all, and running the encoder over it anyway risks the same blowup the ceiling
 *       exists to bound (attention is quadratic in sequence length). {@link #anOversizedSingleWordIsSkippedNotScored()}
 *       pins that it is skipped instead.</li>
 * </ol>
 *
 * <p>{@link GlinerLongSequenceDegradationTest} is the companion real-weight test: it confirms the
 * actual failure this mechanism was built for -- silent, complete loss of detections around a
 * pathological run of text, with no exception -- no longer occurs.
 */
class GlinerSequenceLengthSafetyTest {

    private static Path fixtureDir() throws Exception {
        final URL url = GlinerSequenceLengthSafetyTest.class.getResource("/gliner-fixture/gliner_config.json");
        assertNotNull(url, "the gliner-fixture is missing from the test resources");
        return Path.of(url.toURI()).getParent();
    }

    @Test
    @DisplayName("A ceiling far below normal forces many splits, and each leaf still finds its span")
    @Timeout(10)
    void recursionTerminatesQuicklyEvenWhenForced() throws Exception {

        // A ceiling this low is never reached by a real window (see DEFAULT_MAX_SEQUENCE_TOKENS'
        // own numbers); it is set here purely to force bisection deep into the word range on
        // ordinary short text, which the real-model case cannot exercise quickly or deterministically.
        final LocalPhEyeOptions options = LocalPhEyeOptions.withThreshold(0.5).withMaxSequenceTokens(12);

        try (LocalPhEyeDetector detector = new LocalPhEyeDetector(fixtureDir(), options)) {

            final String text = "alfa beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
            final List<PhEyeSpan> spans = assertDoesNotThrow(
                    () -> detector.detect(text, List.of("label"), "", 0));

            // The fixture's hit is at LOCAL word 0 of whatever window is actually fed to the model,
            // mapped back to global offsets via the window's own starting index. A cap low enough to
            // force bisection into several leaves therefore finds several non-overlapping spans, one
            // per leaf that survived intact -- more than the single span one whole window would give,
            // which is exactly the evidence that bisection is happening rather than being skipped.
            assertTrue(spans.size() > 1, () -> "expected bisection into more than one leaf: " + spans);

            for (final PhEyeSpan span : spans) {
                assertEquals(text.substring(span.getStart(), span.getEnd()), span.getText());
            }

            // Non-overlapping and in text order: exactly what the top-level greedy decode guarantees
            // once every leaf's candidates are merged back together.
            for (int i = 1; i < spans.size(); i++) {
                assertTrue(spans.get(i - 1).getEnd() <= spans.get(i).getStart(),
                        () -> "spans overlap or are out of order: " + spans);
            }

        }

    }

    @Test
    @DisplayName("A single word whose own encoding exceeds the ceiling is skipped, not scored")
    @Timeout(10)
    void anOversizedSingleWordIsSkippedNotScored() throws Exception {

        // The ceiling here is below what even one ordinary word plus the prompt encodes to, so the
        // recursion bottoms out at a single word it still cannot fit under budget. That word must be
        // skipped rather than run through the encoder regardless -- there is nothing smaller to
        // bisect it into.
        final LocalPhEyeOptions options = LocalPhEyeOptions.withThreshold(0.5).withMaxSequenceTokens(2);

        try (LocalPhEyeDetector detector = new LocalPhEyeDetector(fixtureDir(), options)) {
            final List<PhEyeSpan> spans = assertDoesNotThrow(
                    () -> detector.detect("alfa beta gamma", List.of("label"), "", 0));
            assertTrue(spans.isEmpty(), spans.toString());
        }

    }

    @Test
    @DisplayName("A ceiling comfortably above normal changes nothing")
    void aGenerousCeilingIsANoOp() throws Exception {

        try (LocalPhEyeDetector withDefault = new LocalPhEyeDetector(fixtureDir(), LocalPhEyeOptions.withThreshold(0.5));
             LocalPhEyeDetector withExplicit = new LocalPhEyeDetector(fixtureDir(),
                     LocalPhEyeOptions.withThreshold(0.5).withMaxSequenceTokens(100_000))) {

            final String text = "alfa beta gamma delta epsilon";
            assertEquals(withDefault.detect(text, List.of("label"), "", 0).size(),
                    withExplicit.detect(text, List.of("label"), "", 0).size());

        }

    }

    @Test
    @DisplayName("maxSequenceTokens must be positive")
    void nonPositiveCeilingIsRejected() {
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> LocalPhEyeOptions.defaults().withMaxSequenceTokens(0));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> LocalPhEyeOptions.defaults().withMaxSequenceTokens(-1));
    }

}
