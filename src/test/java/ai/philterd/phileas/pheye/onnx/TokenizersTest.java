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

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.net.URL;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The tokenizer must not shorten the input behind the detector's back.
 *
 * <p>DJL's {@code HuggingFaceTokenizer.newInstance(path)} truncates at 512 sub-tokens by default,
 * and says nothing when it does: the encoding just ends. Both detectors decide what to examine by
 * counting <i>words</i>, and words become an unpredictable number of sub-tokens, so a window that
 * looks well within budget can still be cut. Whatever falls past the cut is never scored and never
 * reported -- a document that ends early looks exactly like a document with nothing in it.
 *
 * <p>This test pins the default off. It uses the committed fixture tokenizer, so it runs everywhere
 * and does not need a model.
 */
class TokenizersTest {

    /** Comfortably past DJL's 512-token default, and past any plausible future default. */
    private static final int WORDS = 4000;

    private static Path fixtureTokenizer() throws Exception {
        final URL url = TokenizersTest.class.getResource("/gliner-fixture/tokenizer.json");
        assertNotNull(url, "the fixture tokenizer is missing from the test resources");
        return Path.of(url.toURI());
    }

    @Test
    @DisplayName("A long input is encoded in full, not silently cut at the library's default")
    void longInputIsNotTruncated() throws Exception {

        final String text = "parola ".repeat(WORDS).trim();

        try (final HuggingFaceTokenizer tokenizer = Tokenizers.load(fixtureTokenizer())) {

            final Encoding encoding = tokenizer.encode(text);

            assertTrue(encoding.getIds().length > WORDS,
                    () -> "the encoding holds " + encoding.getIds().length + " sub-tokens for " + WORDS
                            + " words: the input was truncated");

            // The last token must still point at the end of the text, so nothing was dropped from
            // the tail rather than the middle.
            final var spans = encoding.getCharTokenSpans();
            int lastEnd = 0;
            for (final var span : spans) {
                if (span != null && span.getEnd() > lastEnd) {
                    lastEnd = span.getEnd();
                }
            }
            assertEquals(text.length(), lastEnd, "the encoding does not reach the end of the text");

        }

    }

    @Test
    @DisplayName("The default DJL loader does truncate, which is why this module does not use it")
    void theDefaultLoaderIsWhyThisExists() throws Exception {

        // Pinned deliberately. If a future DJL stops truncating by default, this test fails and
        // Tokenizers can be revisited; until then it documents that the workaround is still needed.
        final String text = "parola ".repeat(WORDS).trim();

        try (final HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(fixtureTokenizer())) {
            assertEquals(512, tokenizer.encode(text).getIds().length,
                    "DJL no longer truncates at 512 by default; revisit Tokenizers");
        }

    }

}
