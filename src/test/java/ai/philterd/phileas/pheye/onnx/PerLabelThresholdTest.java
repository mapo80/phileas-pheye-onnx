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

import ai.philterd.phileas.pheye.onnx.LocalPhEyeOptions.DecodeStrategy;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junitpioneer.jupiter.ClearSystemProperty;
import org.junitpioneer.jupiter.SetSystemProperty;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PerLabelThresholdTest {

    @Test
    @DisplayName("A label with no entry falls back to the default threshold")
    void fallbackToDefault() {
        final LocalPhEyeOptions options = LocalPhEyeOptions.of(0.50,
                Map.of("person", 0.20), DecodeStrategy.PER_LABEL_GREEDY);
        assertEquals(0.20, options.thresholdFor("person"));
        assertEquals(0.50, options.thresholdFor("organization"));
        assertEquals(0.50, options.thresholdFor("anything else"));
    }

    @Test
    @DisplayName("Label lookup is case-insensitive, since GLiNER labels are free text")
    void lookupIsCaseInsensitive() {
        final LocalPhEyeOptions options = LocalPhEyeOptions.of(0.50,
                Map.of("Nome Completo", 0.15), DecodeStrategy.PER_LABEL_GREEDY);
        assertEquals(0.15, options.thresholdFor("nome completo"));
        assertEquals(0.15, options.thresholdFor("NOME COMPLETO"));
    }

    @Test
    @DisplayName("An out-of-range per-label threshold is rejected at construction")
    void perLabelThresholdIsValidated() {
        assertThrows(IllegalArgumentException.class,
                () -> LocalPhEyeOptions.of(0.5, Map.of("person", 1.4), DecodeStrategy.FLAT_GREEDY));
        assertThrows(IllegalArgumentException.class,
                () -> LocalPhEyeOptions.of(0.5, Map.of("person", -0.2), DecodeStrategy.FLAT_GREEDY));
    }

    @Test
    @DisplayName("Defaults keep upstream behaviour: 0.5 everywhere, flat greedy")
    void defaultsAreBackwardCompatible() {
        final LocalPhEyeOptions options = LocalPhEyeOptions.defaults();
        assertEquals(0.5, options.thresholdFor("person"));
        assertEquals(0.5, options.thresholdFor("organization"));
        assertEquals(DecodeStrategy.FLAT_GREEDY, options.decodeStrategy());
        assertEquals(Map.of(), options.labelThresholds());
    }

    @Test
    @SetSystemProperty(key = "phileas.pheye.onnx.detectionThreshold", value = "0.40")
    @SetSystemProperty(key = "phileas.pheye.onnx.threshold.person", value = "0.20")
    @SetSystemProperty(key = "phileas.pheye.onnx.threshold.address", value = "0.35")
    @SetSystemProperty(key = "phileas.pheye.onnx.decodeStrategy", value = "per_label_greedy")
    @DisplayName("Per-label thresholds and the strategy are read from system properties")
    void readFromSystemProperties() {
        final LocalPhEyeOptions options = LocalPhEyeOptions.fromEnvironment();
        assertEquals(0.40, options.detectionThreshold());
        assertEquals(0.20, options.thresholdFor("person"));
        assertEquals(0.35, options.thresholdFor("address"));
        assertEquals(0.40, options.thresholdFor("organization"));
        assertEquals(DecodeStrategy.PER_LABEL_GREEDY, options.decodeStrategy());
    }

    @Test
    @ClearSystemProperty(key = "phileas.pheye.onnx.decodeStrategy")
    @SetSystemProperty(key = "phileas.pheye.onnx.threshold.person", value = "nope")
    @DisplayName("A misspelled per-label threshold fails loudly")
    void badPerLabelValueIsFatal() {
        assertThrows(IllegalArgumentException.class, LocalPhEyeOptions::fromEnvironment);
    }

}
