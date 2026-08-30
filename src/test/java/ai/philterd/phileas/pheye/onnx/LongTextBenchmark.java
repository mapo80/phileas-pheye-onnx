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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * How the token-classification detector scales with document length. Not a test: it prints a table.
 *
 * <pre>
 *   java -cp target/test-classes:target/classes:$(cat target/classpath.txt) \
 *        ai.philterd.phileas.pheye.onnx.LongTextBenchmark MODEL_DIR
 * </pre>
 *
 * <p>Windowing makes cost linear in the document rather than quadratic, and the point of the table
 * is to show that it stays linear: a per-window cost that grows with the document would mean the
 * windows are not independent, which for this detector they must be.
 */
public final class LongTextBenchmark {

    private static final String PARAGRAPH =
            "Il ricorrente Giovanni Battista Lombardi, nato il 27/11/1972 a Napoli, residente in "
                    + "Via Chiaia 88, 80132 Napoli, ha depositato la memoria il 15/04/2024 presso la "
                    + "cancelleria, allegando la ricevuta di pagamento del contributo unificato e la "
                    + "documentazione tecnica prodotta dal perito Anna Chiara Esposito. ";

    private static final int[] SIZES = {100, 500, 1_000, 2_000, 5_000, 10_000, 20_000};

    private static final int REPEATS = 5;

    private LongTextBenchmark() {
    }

    public static void main(final String[] args) throws Exception {

        if (args.length != 1) {
            throw new IllegalArgumentException("usage: LongTextBenchmark MODEL_DIR");
        }

        try (final var detector = (LocalTokenClassifierDetector)
                LocalDetectorFactory.open(Path.of(args[0]))) {

            final List<String> labels = detector.entityTypes();
            System.out.printf(Locale.ROOT, "model window: %d words, overlap %d, cap %d sub-tokens,"
                            + " threshold %.2f%n%n",
                    detector.maxWords(), detector.chunkOverlapWords(), detector.maxTokens(),
                    detector.options().detectionThreshold());

            System.out.printf(Locale.ROOT, "%8s %9s %8s %10s %11s %11s %9s%n",
                    "words", "chars", "windows", "median ms", "ms/window", "words/s", "spans");

            // Warm up outside the measurement: the first inference pays for ONNX Runtime's graph
            // preparation and the JIT, which says nothing about steady-state cost.
            detector.detect(document(200), labels, "", 0);

            for (final int size : SIZES) {

                final String text = document(size);
                final int words = WordsSplitter.splitOnRuns(text).size();
                final int windows = LocalPhEyeDetector.planWindows(words, detector.maxWords(),
                        detector.chunkOverlapWords(), LocalPhEyeOptions.LongTextMode.CHUNK).size();

                final List<Long> timings = new ArrayList<>(REPEATS);
                int spans = 0;
                for (int i = 0; i < REPEATS; i++) {
                    final long started = System.nanoTime();
                    final List<PhEyeSpan> found = detector.detect(text, labels, "", 0);
                    timings.add(System.nanoTime() - started);
                    spans = found.size();
                    verifyOffsets(text, found);
                }
                timings.sort(Long::compare);
                final double medianMs = timings.get(timings.size() / 2) / 1_000_000.0;

                System.out.printf(Locale.ROOT, "%8d %9d %8d %10.1f %11.2f %11.0f %9d%n",
                        words, text.length(), windows, medianMs, medianMs / windows,
                        words / (medianMs / 1000.0), spans);

            }

        }

    }

    /** A document of roughly {@code words} words, built by repeating one paragraph. */
    private static String document(final int words) {
        final int perParagraph = PARAGRAPH.trim().split("\\s+").length;
        return PARAGRAPH.repeat(Math.max(1, Math.round((float) words / perParagraph)));
    }

    /** Every span must quote the text it points at: an offset bug would show up here, not in timings. */
    private static void verifyOffsets(final String text, final List<PhEyeSpan> spans) {
        for (final PhEyeSpan span : spans) {
            if (!text.substring(span.getStart(), span.getEnd()).equals(span.getText())) {
                throw new IllegalStateException("offset mismatch at " + span.getStart());
            }
        }
    }

}
