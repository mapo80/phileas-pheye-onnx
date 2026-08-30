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

import ai.onnxruntime.OrtException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * When ONNX Runtime rejects a window, the caller sees an {@link IllegalStateException} that says
 * what was fed to it (a sub-token count) and what to do about it, with the original
 * {@link OrtException} preserved as the cause -- rather than a bare "non-zero status code" message
 * naming a graph node no caller of this library can act on.
 *
 * <p>There is deliberately no test asserting that ONNX Runtime <i>does</i> reject an over-long
 * window: it depends on the encoder's own position scheme, and at least one real, currently
 * supported model (mdeberta-v3-base, relative position buckets) does not reject at all -- confirmed
 * by running 384 words with 60 labels through it without error. A universal capacity check would be
 * wrong for that model in one direction or the other. What both {@code LocalPhEyeDetector} and
 * {@code LocalTokenClassifierDetector} guarantee is only that a rejection, when the encoder does
 * raise one, arrives with this context attached; that is what these tests pin.
 */
class OnnxRejectionMessageTest {

    @Test
    @DisplayName("GLiNER: the sub-token count is in the message, and the cause is preserved")
    void glinerRejectionCarriesContextAndCause() {

        final OrtException cause = new OrtException("Non-zero status code returned while running Gather node");
        final IllegalStateException wrapped = LocalPhEyeDetector.rejectionMessage(713, cause);

        assertSame(cause, wrapped.getCause());
        assertTrue(wrapped.getMessage().contains("713"), wrapped.getMessage());
        assertTrue(wrapped.getMessage().contains("sub-tokens"), wrapped.getMessage());
        assertTrue(wrapped.getMessage().contains("label"), wrapped.getMessage());
    }

    @Test
    @DisplayName("Token classification: the declared capacity is in the message, and the cause is preserved")
    void tokenClassifierRejectionCarriesContextAndCause() {

        final OrtException cause = new OrtException("Non-zero status code returned while running Gather node");
        final IllegalStateException wrapped =
                LocalTokenClassifierDetector.rejectionMessage(9000, 8192, cause);

        assertSame(cause, wrapped.getCause());
        assertTrue(wrapped.getMessage().contains("9000"), wrapped.getMessage());
        assertTrue(wrapped.getMessage().contains("8192"), wrapped.getMessage());
        assertTrue(wrapped.getMessage().contains(TokenClassifierConfig.WINDOW_FILE), wrapped.getMessage());
    }

}
