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

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junitpioneer.jupiter.ClearSystemProperty;
import org.junitpioneer.jupiter.SetSystemProperty;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LocalPhEyeOptionsTest {

    @Test
    @DisplayName("Defaults keep the upstream 0.5 threshold, with safe long-input handling")
    void defaultsMatchUpstreamThreshold() {
        final LocalPhEyeOptions options = LocalPhEyeOptions.defaults();
        assertEquals(0.5, options.detectionThreshold());
        assertEquals(LocalPhEyeOptions.LongTextMode.CHUNK, options.longTextMode());
        assertNull(options.chunkOverlapWords());
    }

    @Test
    @DisplayName("An out-of-range threshold is rejected at construction")
    void thresholdIsValidated() {
        assertThrows(IllegalArgumentException.class, () -> LocalPhEyeOptions.withThreshold(-0.1));
        assertThrows(IllegalArgumentException.class, () -> LocalPhEyeOptions.withThreshold(1.5));
        assertThrows(IllegalArgumentException.class, () -> LocalPhEyeOptions.withThreshold(Double.NaN));
    }

    @Test
    @ClearSystemProperty(key = LocalPhEyeOptions.THRESHOLD_PROPERTY)
    @ClearSystemProperty(key = LocalPhEyeOptions.LONG_TEXT_MODE_PROPERTY)
    @ClearSystemProperty(key = LocalPhEyeOptions.CHUNK_OVERLAP_PROPERTY)
    @DisplayName("With nothing configured, fromEnvironment yields the defaults")
    void environmentDefaults() {
        final LocalPhEyeOptions options = LocalPhEyeOptions.fromEnvironment();
        assertEquals(LocalPhEyeOptions.DEFAULT_DETECTION_THRESHOLD, options.detectionThreshold());
    }

    @Test
    @SetSystemProperty(key = LocalPhEyeOptions.THRESHOLD_PROPERTY, value = "0.20")
    @SetSystemProperty(key = LocalPhEyeOptions.LONG_TEXT_MODE_PROPERTY, value = "fail")
    @SetSystemProperty(key = LocalPhEyeOptions.CHUNK_OVERLAP_PROPERTY, value = "40")
    @DisplayName("System properties override the defaults, case-insensitively for the mode")
    void systemPropertiesAreRead() {
        final LocalPhEyeOptions options = LocalPhEyeOptions.fromEnvironment();
        assertEquals(0.20, options.detectionThreshold());
        assertEquals(LocalPhEyeOptions.LongTextMode.FAIL, options.longTextMode());
        assertEquals(40, options.chunkOverlapWords());
    }

    @Test
    @SetSystemProperty(key = LocalPhEyeOptions.THRESHOLD_PROPERTY, value = "not-a-number")
    @DisplayName("A misspelled threshold fails loudly instead of silently reverting to 0.5")
    void badThresholdIsFatal() {
        final IllegalArgumentException exception =
                assertThrows(IllegalArgumentException.class, LocalPhEyeOptions::fromEnvironment);
        assertTrue(exception.getMessage().contains(LocalPhEyeOptions.THRESHOLD_PROPERTY));
    }

    @Test
    @SetSystemProperty(key = LocalPhEyeOptions.LONG_TEXT_MODE_PROPERTY, value = "sometimes")
    @DisplayName("An unknown long-text mode fails loudly")
    void badModeIsFatal() {
        final IllegalArgumentException exception =
                assertThrows(IllegalArgumentException.class, LocalPhEyeOptions::fromEnvironment);
        assertTrue(exception.getMessage().contains("CHUNK"));
    }

}
