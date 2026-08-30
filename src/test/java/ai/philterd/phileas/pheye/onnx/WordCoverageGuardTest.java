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

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.net.URL;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The assumption behind the GLiNER path's word-coverage guard.
 *
 * <p>{@code LocalPhEyeDetector} refuses a window when not every text word reached the encoding,
 * because a window scored with its tail missing emits no spans there and reports nothing — the
 * silent leak this fork exists to close. That guard is only correct if "a word produced no tokens"
 * really does mean the sequence was cut short, and never that the tokenizer normalised the word
 * away to nothing. A word that legitimately vanishes would turn a working document into a hard
 * failure.
 *
 * <p>So: sweep the code points the splitter is willing to call a word, and assert that none of them
 * disappears. The two components are independent — a splitter regex and a tokenizer's normaliser —
 * and nothing but this test stops them from drifting apart.
 */
class WordCoverageGuardTest {

    /** Through the supplementary planes; past this, everything is unassigned or private use. */
    private static final int LAST_CODE_POINT = 0x2FFFF;

    @Test
    @DisplayName("No character the splitter calls a word is dropped by the tokenizer")
    void everyWordTheSplitterEmitsSurvivesTokenization() throws Exception {

        final URL url = WordCoverageGuardTest.class.getResource("/gliner-fixture/tokenizer.json");
        assertNotNull(url, "the fixture tokenizer is missing from the test resources");

        final List<String> vanished = new ArrayList<>();

        try (final HuggingFaceTokenizer tokenizer = Tokenizers.load(Path.of(url.toURI()))) {

            for (int codePoint = 1; codePoint <= LAST_CODE_POINT; codePoint++) {

                if (Character.getType(codePoint) == Character.UNASSIGNED
                        || Character.isSurrogate((char) codePoint)) {
                    continue;
                }

                final String candidate = new String(Character.toChars(codePoint));
                if (WordsSplitter.split(candidate).isEmpty()) {
                    continue;   // whitespace to the splitter; never offered to the tokenizer as a word
                }

                // Encoded after a word that certainly survives, so word id 1 exists if and only if
                // the candidate produced at least one sub-token of its own.
                long highest = 0;
                for (final long wordId : tokenizer.encode(new String[]{"parola", candidate}).getWordIds()) {
                    highest = Math.max(highest, wordId);
                }
                if (highest < 1) {
                    vanished.add(String.format("U+%04X", codePoint));
                }

            }

        }

        assertTrue(vanished.isEmpty(),
                () -> "these code points are words to the splitter but encode to nothing, so the"
                        + " coverage guard would refuse a window that is in fact complete: " + vanished);

    }

}
