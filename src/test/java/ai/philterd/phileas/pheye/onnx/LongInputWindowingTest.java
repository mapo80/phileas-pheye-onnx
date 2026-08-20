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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.BitSet;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The windowing contract for input longer than the model's {@code max_len}.
 *
 * <p>This matters more than a normal unit test: if a window plan leaves a gap, the words in that gap
 * are never shown to the model, and any personal data there survives redaction. The coverage
 * assertion below is the safety property.
 */
class LongInputWindowingTest {

    @Test
    @DisplayName("Input within max_len is a single window covering everything")
    void shortInputIsOneWindow() {
        final List<int[]> windows = LocalPhEyeDetector.planWindows(50, 384, 32, LongTextMode.CHUNK);
        assertEquals(1, windows.size());
        assertEquals(0, windows.get(0)[0]);
        assertEquals(50, windows.get(0)[1]);
    }

    @ParameterizedTest(name = "every word covered: total={0}, maxLen={1}, overlap={2}")
    @CsvSource({
            "385, 384, 32",
            "500, 384, 32",
            "1000, 384, 48",
            "5000, 384, 32",
            "770, 384, 11",
            "97, 32, 8",
            "100, 10, 9"
    })
    @DisplayName("CHUNK covers every single word index, with no gap anywhere")
    void chunkCoversEveryWord(final int total, final int maxLen, final int overlap) {

        final List<int[]> windows = LocalPhEyeDetector.planWindows(total, maxLen, overlap, LongTextMode.CHUNK);

        final BitSet covered = new BitSet(total);
        for (final int[] window : windows) {
            assertTrue(window[1] - window[0] <= maxLen,
                    "window [" + window[0] + "," + window[1] + ") exceeds maxLen " + maxLen);
            covered.set(window[0], window[1]);
        }

        assertEquals(total, covered.cardinality(),
                "words not covered by any window: " + gaps(covered, total) + " windows=" + describe(windows));
        assertEquals(total, windows.get(windows.size() - 1)[1], "the last window must reach the end of the input");

    }

    @ParameterizedTest(name = "consecutive windows overlap: total={0}, maxLen={1}, overlap={2}")
    @CsvSource({
            "500, 384, 32",
            "1000, 384, 48",
            "97, 32, 8"
    })
    @DisplayName("Consecutive windows share the requested overlap, so boundary spans stay whole somewhere")
    void consecutiveWindowsOverlap(final int total, final int maxLen, final int overlap) {

        final List<int[]> windows = LocalPhEyeDetector.planWindows(total, maxLen, overlap, LongTextMode.CHUNK);

        for (int i = 1; i < windows.size(); i++) {
            final int previousEnd = windows.get(i - 1)[1];
            final int currentStart = windows.get(i)[0];
            assertTrue(currentStart < previousEnd,
                    "window " + i + " starts at " + currentStart + " but the previous ended at " + previousEnd
                            + ": no overlap means a span across the boundary is lost");
            // The final window may be shorter, so only require the full overlap for interior joins.
            if (i < windows.size() - 1) {
                assertEquals(overlap, previousEnd - currentStart, "unexpected overlap between windows");
            }
        }

    }

    @Test
    @DisplayName("FAIL refuses long input rather than examining only part of it")
    void failModeRefusesLongInput() {

        final IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> LocalPhEyeDetector.planWindows(385, 384, 32, LongTextMode.FAIL));

        assertTrue(exception.getMessage().contains("385"));
        assertTrue(exception.getMessage().contains("384"));
    }

    @Test
    @DisplayName("TRUNCATE reproduces the upstream behaviour: only the first max_len words")
    void truncateModeMatchesUpstream() {
        final List<int[]> windows = LocalPhEyeDetector.planWindows(1000, 384, 32, LongTextMode.TRUNCATE);
        assertEquals(1, windows.size());
        assertEquals(0, windows.get(0)[0]);
        assertEquals(384, windows.get(0)[1]);
    }

    @Test
    @DisplayName("A degenerate overlap that would stall the scan still terminates and covers the input")
    void overlapEqualToMaxLenDoesNotStall() {
        // stride would be 0; the implementation clamps it to 1 rather than looping forever.
        final List<int[]> windows = LocalPhEyeDetector.planWindows(40, 10, 10, LongTextMode.CHUNK);
        final BitSet covered = new BitSet(40);
        windows.forEach(w -> covered.set(w[0], w[1]));
        assertEquals(40, covered.cardinality());
    }

    private static String gaps(final BitSet covered, final int total) {
        final StringBuilder builder = new StringBuilder();
        for (int i = 0; i < total; i++) {
            if (!covered.get(i)) {
                builder.append(i).append(' ');
            }
        }
        return builder.isEmpty() ? "(none)" : builder.toString();
    }

    private static String describe(final List<int[]> windows) {
        final StringBuilder builder = new StringBuilder();
        for (final int[] window : windows) {
            builder.append('[').append(window[0]).append(',').append(window[1]).append(") ");
        }
        return builder.toString();
    }

}
