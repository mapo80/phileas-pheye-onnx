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
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The token-classification model directory is read strictly, because every field it carries changes
 * what the detector labels and none of the errors would otherwise be visible: a taxonomy read at
 * the wrong index mislabels everything, and a wrong window degrades detection quietly.
 */
class TokenClassifierConfigTest {

    private static final String WINDOW = """
            { "max_words": 120, "overlap_words": 20, "max_tokens": 512 }
            """;

    private static Path directory(final Path dir, final String modelConfig, final String window) throws Exception {
        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), modelConfig);
        Files.writeString(dir.resolve(TokenClassifierConfig.WINDOW_FILE), window);
        return dir;
    }

    private static String labels(final String... entries) {
        final StringBuilder json = new StringBuilder("{ \"id2label\": {");
        for (int i = 0; i < entries.length; i++) {
            json.append(i > 0 ? ", " : "").append('"').append(i).append("\": \"").append(entries[i]).append('"');
        }
        return json.append("} }").toString();
    }

    @Test
    @DisplayName("Labels are ordered by class index, not by the order they appear in the file")
    void labelsAreOrderedByIndex(@TempDir final Path dir) throws Exception {

        // JSON objects have no order, so a file listing 10 before 2 must still map 2 to its own label.
        final String out = """
                { "id2label": { "10": "I-CITY", "0": "B-CITY", "1": "I-CITY", "2": "O",
                                "3": "B-ORG", "4": "I-ORG", "5": "B-DATE", "6": "I-DATE",
                                "7": "B-CF", "8": "I-CF", "9": "B-IBAN" } }
                """;
        final TokenClassifierConfig config = TokenClassifierConfig.load(directory(dir, out, WINDOW));

        assertEquals("B-CITY", config.id2label.get(0));
        assertEquals("O", config.id2label.get(2));
        assertEquals("I-CITY", config.id2label.get(10));
        assertEquals(11, config.id2label.size());
        assertEquals(List.of("CITY", "ORG", "DATE", "CF", "IBAN"), config.entityTypes);
    }

    @Test
    @DisplayName("A gap in id2label is refused")
    void gapInLabelsIsRefused(@TempDir final Path dir) throws Exception {
        final String out = """
                { "id2label": { "0": "B-CITY", "2": "O" } }
                """;
        final Exception e = assertThrows(IllegalArgumentException.class,
                () -> TokenClassifierConfig.load(directory(dir, out, WINDOW)));
        assertTrue(e.getMessage().contains("gap in id2label"), e.getMessage());
    }

    @Test
    @DisplayName("A non-BIO taxonomy is refused rather than decoded as if it were BIO")
    void nonBioTaxonomyIsRefused(@TempDir final Path dir) throws Exception {
        final Exception e = assertThrows(IllegalArgumentException.class,
                () -> TokenClassifierConfig.load(directory(dir, labels("CITY", "ORG", "O"), WINDOW)));
        assertTrue(e.getMessage().contains("BIO"), e.getMessage());
    }

    @Test
    @DisplayName("A taxonomy of nothing but O is refused")
    void emptyTaxonomyIsRefused(@TempDir final Path dir) throws Exception {
        final Exception e = assertThrows(IllegalArgumentException.class,
                () -> TokenClassifierConfig.load(directory(dir, labels("O"), WINDOW)));
        assertTrue(e.getMessage().contains("no entity"), e.getMessage());
    }

    @Test
    @DisplayName("A missing id2label is refused")
    void missingLabelsAreRefused(@TempDir final Path dir) throws Exception {
        final Exception e = assertThrows(IllegalArgumentException.class,
                () -> TokenClassifierConfig.load(directory(dir, "{ \"model_type\": \"modernbert\" }", WINDOW)));
        assertTrue(e.getMessage().contains("id2label"), e.getMessage());
    }

    @Test
    @DisplayName("The window file is required, not defaulted")
    void windowFileIsRequired(@TempDir final Path dir) throws Exception {
        Files.writeString(dir.resolve(TokenClassifierConfig.MODEL_FILE), labels("B-CITY", "O"));
        final Exception e = assertThrows(IllegalArgumentException.class, () -> TokenClassifierConfig.load(dir));
        assertTrue(e.getMessage().contains(TokenClassifierConfig.WINDOW_FILE), e.getMessage());
    }

    @Test
    @DisplayName("Each window field is required individually")
    void windowFieldsAreRequired(@TempDir final Path dir) throws Exception {
        final Exception e = assertThrows(IllegalArgumentException.class, () -> TokenClassifierConfig.load(
                directory(dir, labels("B-CITY", "O"), "{ \"max_words\": 120, \"overlap_words\": 20 }")));
        assertTrue(e.getMessage().contains("max_tokens"), e.getMessage());
    }

    @Test
    @DisplayName("An overlap at least as wide as the window is refused: the stride would be zero")
    void overlapWiderThanWindowIsRefused(@TempDir final Path dir) throws Exception {
        final String window = "{ \"max_words\": 120, \"overlap_words\": 120, \"max_tokens\": 512 }";
        final Exception e = assertThrows(IllegalArgumentException.class,
                () -> TokenClassifierConfig.load(directory(dir, labels("B-CITY", "O"), window)));
        assertTrue(e.getMessage().contains("overlap_words"), e.getMessage());
    }

    @Test
    @DisplayName("A non-whitespace splitter is refused")
    void otherSplittersAreRefused(@TempDir final Path dir) throws Exception {
        final String window = """
                { "max_words": 120, "overlap_words": 20, "max_tokens": 512, "words_splitter_type": "spacy" }
                """;
        final Exception e = assertThrows(IllegalArgumentException.class,
                () -> TokenClassifierConfig.load(directory(dir, labels("B-CITY", "O"), window)));
        assertTrue(e.getMessage().contains("words_splitter_type"), e.getMessage());
    }

    @Test
    @DisplayName("A calibrated threshold is read when declared, and is optional")
    void calibratedThresholdIsOptional(@TempDir final Path dir) throws Exception {

        assertNull(TokenClassifierConfig.load(directory(dir, labels("B-CITY", "O"), WINDOW)).calibratedThreshold);

        final String window = """
                { "max_words": 120, "overlap_words": 20, "max_tokens": 512, "calibrated_threshold": 0.92 }
                """;
        assertEquals(0.92,
                TokenClassifierConfig.load(directory(dir, labels("B-CITY", "O"), window)).calibratedThreshold, 1e-9);
    }

    @Test
    @DisplayName("A calibrated threshold outside [0, 1] is refused")
    void outOfRangeCalibratedThresholdIsRefused(@TempDir final Path dir) throws Exception {
        final String window = """
                { "max_words": 120, "overlap_words": 20, "max_tokens": 512, "calibrated_threshold": 1.4 }
                """;
        final Exception e = assertThrows(IllegalArgumentException.class,
                () -> TokenClassifierConfig.load(directory(dir, labels("B-CITY", "O"), window)));
        assertTrue(e.getMessage().contains("calibrated_threshold"), e.getMessage());
    }

    @Test
    @DisplayName("Requested labels resolve against the taxonomy case-insensitively")
    void labelsResolveCaseInsensitively(@TempDir final Path dir) throws Exception {
        final TokenClassifierConfig config =
                TokenClassifierConfig.load(directory(dir, labels("B-FULLNAME", "I-FULLNAME", "O"), WINDOW));
        assertEquals("FULLNAME", config.resolveEntityType("fullname"));
        assertEquals("FULLNAME", config.resolveEntityType("  FullName "));
        assertNull(config.resolveEntityType("person"));
        assertNull(config.resolveEntityType(null));
    }

}
