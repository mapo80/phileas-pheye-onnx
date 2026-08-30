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
import java.nio.file.StandardCopyOption;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Fail-closed contract for the token-classification path.
 *
 * <p>Same principle as {@link FailClosedTest}, and the same reason: the dangerous failure for a
 * redaction component is a detector that builds successfully and then finds nothing, because a
 * document with no personal data looks exactly like one that was never examined. Every case here
 * must raise.
 */
class TokenClassifierFailClosedTest {

    private static final String MODEL = """
            { "id2label": { "0": "B-CITY", "1": "I-CITY", "2": "O" } }
            """;

    private static final String WINDOW = """
            { "max_words": 120, "overlap_words": 20, "max_tokens": 512 }
            """;

    @Test
    @DisplayName("An empty model directory is refused")
    void emptyDirectoryIsRefused(@TempDir final Path dir) {
        final Exception e = assertThrows(Exception.class, () -> new LocalTokenClassifierDetector(dir));
        assertTrue(e.getMessage().contains(TokenClassifierConfig.MODEL_FILE), e.getMessage());
    }

    @Test
    @DisplayName("A missing tokenizer.json is refused")
    void missingTokenizerIsRefused(@TempDir final Path dir) throws Exception {
        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), MODEL);
        Files.writeString(dir.resolve(TokenClassifierConfig.WINDOW_FILE), WINDOW);
        final Exception e = assertThrows(Exception.class, () -> new LocalTokenClassifierDetector(dir));
        assertTrue(e.getMessage().contains("tokenizer.json"), e.getMessage());
    }

    @Test
    @DisplayName("A missing ONNX model is refused")
    void missingOnnxIsRefused(@TempDir final Path dir) throws Exception {

        final String tokenizerDir = System.getenv("PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR");
        assumeTrue(tokenizerDir != null && !tokenizerDir.isBlank(),
                "Set PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR: this case needs a loadable tokenizer.json"
                        + " to get as far as the ONNX check.");

        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), MODEL);
        Files.writeString(dir.resolve(TokenClassifierConfig.WINDOW_FILE), WINDOW);
        Files.copy(Path.of(tokenizerDir).resolve("tokenizer.json"), dir.resolve("tokenizer.json"),
                StandardCopyOption.REPLACE_EXISTING);

        final Exception e = assertThrows(Exception.class, () -> new LocalTokenClassifierDetector(dir));
        assertTrue(e.getMessage().contains("ONNX"), e.getMessage());
    }

    @Test
    @DisplayName("A taxonomy that does not match the graph's class count is refused")
    void labelCountMismatchIsRefused(@TempDir final Path dir) throws Exception {

        // The failure this prevents is silent and total: with the wrong number of classes, every
        // argmax lands on a label belonging to a different index and the whole document is
        // mislabelled, while the detector reports no error at all.
        final Path model = realModelDirectory();

        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), MODEL);
        Files.writeString(dir.resolve(TokenClassifierConfig.WINDOW_FILE), WINDOW);
        Files.copy(model.resolve("tokenizer.json"), dir.resolve("tokenizer.json"));
        Files.createDirectory(dir.resolve("onnx"));
        Files.createSymbolicLink(dir.resolve("onnx").resolve("model.onnx"),
                model.resolve("onnx").resolve("model.onnx").toAbsolutePath());

        final Exception e = assertThrows(IllegalStateException.class, () -> new LocalTokenClassifierDetector(dir));
        assertTrue(e.getMessage().contains("classes"), e.getMessage());
        assertTrue(e.getMessage().contains("different model"), e.getMessage());
    }

    @Test
    @DisplayName("A graph needing an input we never feed is refused at construction")
    void unexpectedGraphInputsAreRefused(@TempDir final Path dir) throws Exception {

        // The GLiNER graph takes six inputs; this detector feeds two. Loading one as the other must
        // fail here, not at the first document, which for a filter built at startup would mean the
        // failure surfaces in production traffic.
        final Path fixture = Path.of(TokenClassifierFailClosedTest.class
                .getResource("/gliner-fixture/gliner_config.json").toURI()).getParent();
        assumeTrue(Files.isReadable(fixture.resolve("model.onnx")),
                "the synthetic GLiNER graph is generated on demand; see its README");

        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), MODEL);
        Files.writeString(dir.resolve(TokenClassifierConfig.WINDOW_FILE), WINDOW);
        Files.copy(fixture.resolve("tokenizer.json"), dir.resolve("tokenizer.json"));
        Files.copy(fixture.resolve("model.onnx"), dir.resolve("model.onnx"));

        final Exception e = assertThrows(IllegalStateException.class, () -> new LocalTokenClassifierDetector(dir));
        assertTrue(e.getMessage().contains("does not provide") || e.getMessage().contains("not a token-classification"),
                e.getMessage());
    }

    @Test
    @DisplayName("Asking only for labels the model cannot emit is an error, not an empty result")
    void unknownLabelsAreRefused() throws Exception {

        // A zero-shot model would simply be prompted with these. A token classifier cannot be, and
        // returning nothing here would read downstream as "this document is clean".
        try (final var detector = new LocalTokenClassifierDetector(realModelDirectory())) {
            final Exception e = assertThrows(IllegalArgumentException.class,
                    () -> detector.detect("Mario Rossi abita a Roma.", List.of("person", "address"), "", 0));
            assertTrue(e.getMessage().contains("taxonomy"), e.getMessage());
            assertTrue(e.getMessage().contains("FULLNAME"), e.getMessage());
        }
    }

    @Test
    @DisplayName("A partly-known label set is honoured for the labels the model does have")
    void knownLabelsAreStillHonoured() throws Exception {
        try (final var detector = new LocalTokenClassifierDetector(realModelDirectory())) {
            final var spans = detector.detect("Il sig. Mario Rossi abita a Roma.",
                    List.of("person", "FULLNAME"), "", 0);
            assertEquals(1, spans.size(), spans.toString());
            assertEquals("FULLNAME", spans.get(0).getLabel());
            assertEquals("Mario Rossi", spans.get(0).getText());
        }
    }

    @Test
    @DisplayName("No labels at all yields no spans, as on the GLiNER path")
    void noLabelsYieldsNoSpans() throws Exception {
        try (final var detector = new LocalTokenClassifierDetector(realModelDirectory())) {
            assertTrue(detector.detect("Mario Rossi", List.of(), "", 0).isEmpty());
            assertTrue(detector.detect(null, detector.entityTypes(), "", 0).isEmpty());
            assertTrue(detector.detect("   ", detector.entityTypes(), "", 0).isEmpty());
        }
    }

    private static Path realModelDirectory() {
        final String dir = System.getenv("PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(),
                "Set PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR to a token-classification model directory.");
        return Path.of(dir);
    }

}
