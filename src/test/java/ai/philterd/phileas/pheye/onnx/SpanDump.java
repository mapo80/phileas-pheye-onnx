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

import ai.philterd.phileas.services.filters.ai.pheye.PhEyeDetector;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeSpan;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * Streams documents through a local detector and prints the spans it finds, one JSON object per
 * line. Used by {@code scripts/cross_check_against_reference.py} to compare this implementation
 * against the model's Python reference over a whole corpus rather than a handful of fixtures.
 *
 * <p>Input is one JSON object per line, {@code {"id": ..., "text_b64": ...}}, with the text
 * base64-encoded so a document containing newlines stays on one line. Output carries offsets,
 * labels and scores; it never echoes the text back, so a corpus of real documents can be checked
 * without its content ending up in a report.
 *
 * <pre>
 *   java -cp target/test-classes:target/classes:$(cat target/classpath.txt) \
 *        ai.philterd.phileas.pheye.onnx.SpanDump MODEL_DIR THRESHOLD &lt; documents.jsonl
 * </pre>
 */
public final class SpanDump {

    private record Out(String id, List<Span> spans) {}

    private record Span(int start, int end, String label, double score) {}

    private SpanDump() {
    }

    public static void main(final String[] args) throws Exception {

        if (args.length != 2) {
            throw new IllegalArgumentException("usage: SpanDump MODEL_DIR THRESHOLD");
        }

        final Gson gson = new Gson();
        final LocalPhEyeOptions options = LocalPhEyeOptions.withThreshold(Double.parseDouble(args[1]));

        try (final PhEyeDetector detector = LocalDetectorFactory.open(Path.of(args[0]), options);
             final BufferedReader input = new BufferedReader(
                     new InputStreamReader(System.in, StandardCharsets.UTF_8));
             final BufferedWriter output = new BufferedWriter(
                     new OutputStreamWriter(System.out, StandardCharsets.UTF_8))) {

            final List<String> labels = detector instanceof LocalTokenClassifierDetector local
                    ? local.entityTypes()
                    : List.of("person", "organization", "address");

            String line;
            while ((line = input.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                final JsonObject document = gson.fromJson(line, JsonObject.class);
                final String text = new String(
                        Base64.getDecoder().decode(document.get("text_b64").getAsString()),
                        StandardCharsets.UTF_8);

                final List<Span> spans = new ArrayList<>();
                for (final PhEyeSpan span : detector.detect(text, labels, "", 0)) {
                    spans.add(new Span(span.getStart(), span.getEnd(), span.getLabel(), span.getScore()));
                }
                output.write(gson.toJson(new Out(document.get("id").getAsString(), spans)));
                output.newLine();
            }

        }

    }

}
