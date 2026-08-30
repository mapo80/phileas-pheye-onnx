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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Pins an empirical finding about why {@code LocalPhEyeDetector} declares no fixed sub-token
 * capacity for GLiNER (see {@link LocalPhEyeDetector#rejectionMessage}): a window at the model's
 * full {@code max_len} in words, combined with a long label list, produces a sequence well past the
 * usual 512-token neighbourhood that absolute-position encoders are built around -- and a real,
 * currently supported GLiNER encoder (mdeberta-v3-base, relative position buckets) runs it without
 * error regardless.
 *
 * <p>If a future model swap changes that -- moves to an encoder with a hard absolute-position limit
 * -- this test starts failing here rather than the assumption quietly going stale in a comment. It
 * runs against {@code PHILEAS_GLINER_MODEL_DIR} and skips otherwise.
 */
class GlinerLongPromptTest {

    @Test
    @DisplayName("max_len words plus many labels does not make the encoder reject the window")
    void fullLengthWindowWithManyLabelsDoesNotThrow() throws Exception {

        final String dirEnv = System.getenv("PHILEAS_GLINER_MODEL_DIR");
        assumeTrue(dirEnv != null && !dirEnv.isBlank(), "Set PHILEAS_GLINER_MODEL_DIR to run this.");
        final Path dir = Path.of(dirEnv);

        try (LocalPhEyeDetector detector = new LocalPhEyeDetector(dir)) {

            final List<String> labels = new ArrayList<>();
            for (int i = 0; i < 60; i++) {
                labels.add("label number " + i + " a fairly long entity type name here");
            }

            final StringBuilder text = new StringBuilder();
            for (int i = 0; i < detector.maxWords(); i++) {
                text.append("parola").append(i).append(' ');
            }

            assertDoesNotThrow(() -> detector.detect(text.toString(), labels, "", 0));
        }

    }

}
