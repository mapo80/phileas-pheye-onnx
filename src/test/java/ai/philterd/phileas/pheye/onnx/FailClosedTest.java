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

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Fail-closed contract: a broken model directory must stop the detector from being created.
 *
 * <p>The failure mode being guarded against is the dangerous one for a privacy component: a
 * detector that constructs successfully but detects nothing, so text flows through unredacted while
 * the pipeline reports success. Every case here must raise, never return a usable-looking detector.
 */
class FailClosedTest {

    private static final String VALID_CONFIG = """
            { "words_splitter_type": "whitespace", "max_width": 12, "max_len": 384 }
            """;

    @Test
    @DisplayName("An empty model directory is refused")
    void emptyDirectoryIsRefused(@TempDir final Path dir) {
        final Exception exception = assertThrows(Exception.class, () -> new LocalPhEyeDetector(dir));
        assertTrue(exception.getMessage().contains("gliner_config.json"), exception.getMessage());
    }

    @Test
    @DisplayName("A missing tokenizer.json is refused")
    void missingTokenizerIsRefused(@TempDir final Path dir) throws Exception {
        Files.writeString(dir.resolve("gliner_config.json"), VALID_CONFIG);
        final Exception exception = assertThrows(Exception.class, () -> new LocalPhEyeDetector(dir));
        assertTrue(exception.getMessage().contains("tokenizer.json"), exception.getMessage());
    }

    @Test
    @DisplayName("A missing ONNX model is refused")
    void missingOnnxIsRefused(@TempDir final Path dir) throws Exception {

        Files.writeString(dir.resolve("gliner_config.json"), VALID_CONFIG);
        copyFixtureTokenizer(dir);

        final Exception exception = assertThrows(Exception.class, () -> new LocalPhEyeDetector(dir));
        assertTrue(exception.getMessage().toLowerCase().contains("onnx"), exception.getMessage());

    }

    @Test
    @DisplayName("A corrupt ONNX file is refused rather than yielding a silent no-op detector")
    void corruptOnnxIsRefused(@TempDir final Path dir) throws Exception {

        Files.writeString(dir.resolve("gliner_config.json"), VALID_CONFIG);
        copyFixtureTokenizer(dir);

        Files.createDirectories(dir.resolve("onnx"));
        Files.write(dir.resolve("onnx").resolve("model.onnx"),
                "this is not a protobuf".getBytes(StandardCharsets.UTF_8));

        // ONNX Runtime rejects the graph; the point is that construction fails loudly.
        assertThrows(Exception.class, () -> new LocalPhEyeDetector(dir));

    }

    @Test
    @DisplayName("A malformed gliner_config.json is refused")
    void malformedConfigIsRefused(@TempDir final Path dir) throws Exception {
        Files.writeString(dir.resolve("gliner_config.json"), "{ this is not json ");
        assertThrows(Exception.class, () -> new LocalPhEyeDetector(dir));
    }

    @Test
    @DisplayName("A non-positive max_len or max_width is refused")
    void degenerateConfigIsRefused(@TempDir final Path dir) throws Exception {

        Files.writeString(dir.resolve("gliner_config.json"),
                "{ \"words_splitter_type\": \"whitespace\", \"max_width\": 12, \"max_len\": 0 }");
        final IllegalArgumentException zeroLen = assertThrows(IllegalArgumentException.class,
                () -> new LocalPhEyeDetector(dir));
        assertTrue(zeroLen.getMessage().contains("max_len"), zeroLen.getMessage());

        Files.writeString(dir.resolve("gliner_config.json"),
                "{ \"words_splitter_type\": \"whitespace\", \"max_width\": 0, \"max_len\": 384 }");
        final IllegalArgumentException zeroWidth = assertThrows(IllegalArgumentException.class,
                () -> new LocalPhEyeDetector(dir));
        assertTrue(zeroWidth.getMessage().contains("max_width"), zeroWidth.getMessage());

    }

    @Test
    @DisplayName("The required GLiNER tensor names are pinned, so a signature change is caught")
    void requiredSignatureIsPinned() {
        assertTrue(LocalPhEyeDetector.REQUIRED_INPUTS.containsAll(java.util.List.of(
                "input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask")));
        assertTrue(LocalPhEyeDetector.REQUIRED_INPUTS.size() == 6);
        assertNotNull(LocalPhEyeDetector.REQUIRED_OUTPUT);
    }

    /** The fixture tokenizer is a real tokenizer.json, needed to get past tokenizer loading. */
    private static void copyFixtureTokenizer(final Path dir) throws Exception {
        final var url = FailClosedTest.class.getClassLoader().getResource("gliner-fixture/tokenizer.json");
        assertNotNull(url, "the fixture tokenizer.json must be on the test classpath");
        Files.copy(Path.of(url.toURI()), dir.resolve("tokenizer.json"));
    }

}
