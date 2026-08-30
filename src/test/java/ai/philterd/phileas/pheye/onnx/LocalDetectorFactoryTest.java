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
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Which detector a model directory yields, and what threshold it starts with.
 *
 * <p>The dispatch is deliberately a fact about the layout rather than a configuration switch: a
 * caller that has to name the model family gets to name it wrongly, and a GLiNER detector pointed at
 * a token classifier fails at the ONNX signature check -- late, and with a confusing message.
 */
class LocalDetectorFactoryTest {

    private static final String GLINER = """
            { "words_splitter_type": "whitespace", "max_width": 12, "max_len": 384 }
            """;

    private static final String TOKEN_MODEL = """
            { "id2label": { "0": "B-CITY", "1": "I-CITY", "2": "O" } }
            """;

    private static String window(final String extra) {
        return "{ \"max_words\": 120, \"overlap_words\": 20, \"max_tokens\": 512" + extra + " }";
    }

    @Test
    @DisplayName("A gliner_config.json means a GLiNER model")
    void glinerDirectoryIsRecognized(@TempDir final Path dir) throws Exception {
        Files.writeString(dir.resolve("gliner_config.json"), GLINER);
        assertEquals(LocalDetectorFactory.Kind.GLINER, LocalDetectorFactory.kindOf(dir));
    }

    @Test
    @DisplayName("A token_classification_config.json plus config.json means a token classifier")
    void tokenClassificationDirectoryIsRecognized(@TempDir final Path dir) throws Exception {
        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), TOKEN_MODEL);
        Files.writeString(dir.resolve(TokenClassifierConfig.WINDOW_FILE), window(""));
        assertEquals(LocalDetectorFactory.Kind.TOKEN_CLASSIFICATION, LocalDetectorFactory.kindOf(dir));
    }

    @Test
    @DisplayName("A directory that is neither is refused, naming both layouts")
    void unknownDirectoryIsRefused(@TempDir final Path dir) {
        final Exception e = assertThrows(IllegalArgumentException.class, () -> LocalDetectorFactory.kindOf(dir));
        assertTrue(e.getMessage().contains("gliner_config.json"), e.getMessage());
        assertTrue(e.getMessage().contains(TokenClassifierConfig.WINDOW_FILE), e.getMessage());
    }

    @Test
    @DisplayName("A directory claiming to be both is refused rather than picked by ordering")
    void ambiguousDirectoryIsRefused(@TempDir final Path dir) throws Exception {
        Files.writeString(dir.resolve("gliner_config.json"), GLINER);
        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), TOKEN_MODEL);
        Files.writeString(dir.resolve(TokenClassifierConfig.WINDOW_FILE), window(""));
        final Exception e = assertThrows(IllegalArgumentException.class, () -> LocalDetectorFactory.kindOf(dir));
        assertTrue(e.getMessage().contains("both"), e.getMessage());
    }

    @Test
    @DisplayName("A file, or a path that does not exist, is not a model directory")
    void nonDirectoriesAreRefused(@TempDir final Path dir) throws Exception {
        final Path file = Files.writeString(dir.resolve("model.txt"), "not a directory");
        assertThrows(IllegalArgumentException.class, () -> LocalDetectorFactory.kindOf(file));
        assertThrows(IllegalArgumentException.class, () -> LocalDetectorFactory.kindOf(dir.resolve("absent")));
        assertThrows(IllegalArgumentException.class, () -> LocalDetectorFactory.kindOf(null));
    }

    @Test
    @DisplayName("The directory's calibrated threshold replaces the library default")
    void calibratedThresholdReplacesTheLibraryDefault() {

        final LocalPhEyeOptions defaults = LocalPhEyeOptions.defaults();
        assertEquals(LocalPhEyeOptions.DEFAULT_DETECTION_THRESHOLD, defaults.detectionThreshold(), 1e-9);
        assertEquals(false, defaults.thresholdExplicit());

        final LocalPhEyeOptions calibrated = defaults.withDefaultThreshold(0.92);
        assertEquals(0.92, calibrated.detectionThreshold(), 1e-9);
        assertEquals(true, calibrated.thresholdExplicit());
    }

    @Test
    @DisplayName("A threshold the caller chose is never replaced by the model directory's")
    void explicitThresholdIsNeverOverridden() {
        final LocalPhEyeOptions chosen = LocalPhEyeOptions.withThreshold(0.30);
        assertTrue(chosen.thresholdExplicit());
        assertEquals(0.30, chosen.withDefaultThreshold(0.92).detectionThreshold(), 1e-9);
    }

    @Test
    @DisplayName("Opening the real model directory yields a token classifier at its calibrated threshold")
    void realDirectoryOpensAtItsCalibratedThreshold() throws Exception {

        final String dir = System.getenv("PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(),
                "Set PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR to a token-classification model directory.");

        try (final var detector = LocalDetectorFactory.open(Path.of(dir))) {
            assertTrue(detector instanceof LocalTokenClassifierDetector, detector.getClass().getName());
            final LocalTokenClassifierDetector local = (LocalTokenClassifierDetector) detector;
            assertEquals(TokenClassifierConfig.load(Path.of(dir)).calibratedThreshold,
                    local.options().detectionThreshold(), 1e-9);
        }

    }

}
