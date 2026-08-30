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
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Real-weight parity: {@link LocalTokenClassifierDetector} must reproduce, span for span, what the
 * model's Python reference pipeline produces on the same text.
 *
 * <p>The reference is HuggingFace {@code token-classification} with
 * {@code aggregation_strategy="simple"}, windowed and reduced exactly as the model's own
 * application does. It is regenerated with
 * {@code scripts/generate_token_classification_fixture.py}, which is the only source of the
 * expected values in {@code reference.json} -- they are never edited by hand to match the Java.
 *
 * <p>This is the test that matters for adopting a new model: a threshold calibrated in Python is
 * only meaningful here if the two pipelines agree on which spans exist and what they score. It runs
 * when {@code PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR} points at the model directory the fixture was
 * generated from, and skips otherwise.
 */
class LocalTokenClassifierParityTest {

    /** ONNX and PyTorch differ in the last few bits; observed logit drift is under 1e-4. */
    private static final double SCORE_TOLERANCE = 1e-3;

    private record Expected(String label, int start, int end, double score) {}

    private static JsonObject reference() throws Exception {
        try (final InputStream stream = LocalTokenClassifierParityTest.class
                .getResourceAsStream("/token-classification/reference.json")) {
            assertNotNull(stream, "reference.json is missing from the test resources");
            return new Gson().fromJson(new InputStreamReader(stream, StandardCharsets.UTF_8), JsonObject.class);
        }
    }

    private static Path modelDirectory() {
        final String dir = System.getenv("PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(),
                "Set PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR to the model directory reference.json was"
                        + " generated from to run token-classification parity.");
        return Path.of(dir);
    }

    private static List<Expected> expected(final JsonObject document, final String threshold) {
        final JsonArray array = document.getAsJsonObject("by_threshold").getAsJsonArray(threshold);
        final List<Expected> spans = new ArrayList<>(array.size());
        for (final JsonElement element : array) {
            final JsonObject span = element.getAsJsonObject();
            spans.add(new Expected(span.get("label").getAsString(), span.get("start").getAsInt(),
                    span.get("end").getAsInt(), span.get("score").getAsDouble()));
        }
        return spans;
    }

    private void assertParityAt(final double threshold, final String key) throws Exception {

        final Path dir = modelDirectory();
        final JsonObject reference = reference();

        try (final LocalTokenClassifierDetector detector =
                     new LocalTokenClassifierDetector(dir, LocalPhEyeOptions.withThreshold(threshold))) {

            final List<String> labels = detector.entityTypes();

            for (final JsonElement element : reference.getAsJsonArray("documents")) {

                final JsonObject document = element.getAsJsonObject();
                final String id = document.get("id").getAsString();
                final String text = document.get("text").getAsString();
                final List<Expected> want = expected(document, key);

                final List<PhEyeSpan> got = detector.detect(text, labels, "", 0);

                assertEquals(want.size(), got.size(),
                        () -> "span count differs on document '" + id + "' at threshold " + threshold
                                + "\n  expected: " + want
                                + "\n  actual:   " + got.stream()
                                .map(s -> s.getLabel() + "[" + s.getStart() + "," + s.getEnd() + ")").toList());

                for (int i = 0; i < want.size(); i++) {
                    final Expected e = want.get(i);
                    final PhEyeSpan a = got.get(i);
                    final int index = i;
                    assertEquals(e.label(), a.getLabel(), () -> "label of span " + index + " in '" + id + "'");
                    assertEquals(e.start(), a.getStart(), () -> "start of span " + index + " in '" + id + "'");
                    assertEquals(e.end(), a.getEnd(), () -> "end of span " + index + " in '" + id + "'");
                    assertEquals(e.score(), a.getScore(), SCORE_TOLERANCE,
                            () -> "score of span " + index + " in '" + id + "'");
                    assertEquals(text.substring(e.start(), e.end()), a.getText(),
                            () -> "text of span " + index + " in '" + id + "'");
                }

            }

        }

    }

    @Test
    @DisplayName("Every span matches the Python reference with nothing filtered out")
    void matchesReferenceWithNoThreshold() throws Exception {
        assertParityAt(0.0, "0");
    }

    @Test
    @DisplayName("Every span matches the Python reference at the model's calibrated threshold")
    void matchesReferenceAtCalibratedThreshold() throws Exception {
        assertParityAt(0.92, "0.92");
    }

    @Test
    @DisplayName("Documents longer than one window are covered to the last word")
    void longDocumentsAreCoveredToTheEnd() throws Exception {

        final Path dir = modelDirectory();
        final JsonObject reference = reference();

        try (final LocalTokenClassifierDetector detector =
                     new LocalTokenClassifierDetector(dir, LocalPhEyeOptions.withThreshold(0.92))) {

            for (final JsonElement element : reference.getAsJsonArray("documents")) {

                final JsonObject document = element.getAsJsonObject();
                if (document.get("windows").getAsInt() < 2) {
                    continue;
                }
                final String id = document.get("id").getAsString();
                final String text = document.get("text").getAsString();
                final List<PhEyeSpan> got = detector.detect(text, detector.entityTypes(), "", 0);

                // The reference finds entities in the tail of every multi-window document in the
                // fixture, so a truncating implementation cannot pass this.
                final int lastWindowStart = text.length() - (text.length() / 4);
                assertTrue(got.stream().anyMatch(s -> s.getStart() >= lastWindowStart),
                        () -> "no span found in the last quarter of multi-window document '" + id
                                + "'; the tail was not examined");
            }

        }

    }

}
