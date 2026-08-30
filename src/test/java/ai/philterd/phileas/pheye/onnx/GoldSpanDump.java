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
import java.util.Map;

/**
 * Reproduces the README's "GLiNER vs mmBERT" quality comparison: the exact GLiNER configuration a
 * prior evaluation in this workspace froze for production (labels {@code person}, {@code company},
 * {@code postal address}; decode threshold 0.60 globally, 0.40 for {@code person}; {@code
 * FLAT_GREEDY}; {@code CHUNK}), run through the <b>current local build</b> of this module rather than
 * a previously published release.
 *
 * <p>Input is one JSON object per line on stdin, {@code {"id": ..., "text_b64": ...}}; output is one
 * line per input carrying offsets, labels and scores only, so the corpus text never appears in the
 * result. See {@code scripts/gliner_vs_mmbert_comparison.py}, which drives this against a gold
 * dataset and scores it, for the full reproduction of that README section.
 *
 * <pre>
 *   java -cp target/test-classes:target/classes:$(cat target/classpath.txt) \
 *        ai.philterd.phileas.pheye.onnx.GoldSpanDump GLINER_MODEL_DIR &lt; documents.jsonl
 * </pre>
 */
public final class GoldSpanDump {

    private record Out(String id, List<Span> spans) {}

    private record Span(int start, int end, String label, double score) {}

    private GoldSpanDump() {
    }

    public static void main(final String[] args) throws Exception {

        if (args.length != 1) {
            throw new IllegalArgumentException("usage: GoldSpanDump GLINER_MODEL_DIR");
        }

        final List<String> labels = List.of("person", "company", "postal address");
        final LocalPhEyeOptions options = LocalPhEyeOptions.of(
                0.60, Map.of("person", 0.40), LocalPhEyeOptions.DecodeStrategy.FLAT_GREEDY);
        final Gson gson = new Gson();

        try (LocalPhEyeDetector detector = new LocalPhEyeDetector(Path.of(args[0]), options);
             BufferedReader input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
             BufferedWriter output = new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8))) {

            String line;
            while ((line = input.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                final JsonObject document = gson.fromJson(line, JsonObject.class);
                final String text = new String(
                        Base64.getDecoder().decode(document.get("text_b64").getAsString()), StandardCharsets.UTF_8);

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
