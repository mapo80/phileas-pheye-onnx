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

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Splits text into words with their character offsets, matching GLiNER's
 * {@code WhitespaceTokenSplitter} (the model's {@code words_splitter_type} is "whitespace").
 *
 * <p>GLiNER uses the Python regex {@code \w+(?:[-_]\w+)*|\S}. Python's {@code re} treats
 * {@code \w} as Unicode for {@code str}, so {@link Pattern#UNICODE_CHARACTER_CLASS} is set
 * here to match accented names and non-ASCII word characters identically.
 *
 * <p>{@link #splitOnRuns(String)} is the other splitter in use here: the sub-word
 * token-classification path windows on whitespace-delimited runs ({@code \S+}), which is what its
 * reference pipeline chunks on. The two are deliberately not unified -- each mirrors the Python
 * implementation of the model family it drives, and quietly swapping one for the other would move
 * window boundaries and therefore results.
 */
public final class WordsSplitter {

    private static final Pattern PATTERN =
            Pattern.compile("\\w+(?:[-_]\\w+)*|\\S", Pattern.UNICODE_CHARACTER_CLASS);

    // UNICODE_CHARACTER_CLASS, for the same reason PATTERN sets it: Python's \s for str is
    // Unicode-aware and Java's is not. Without the flag a non-breaking space is part of a word here
    // and a separator in the reference -- nineteen characters differ, U+00A0 among them, and text
    // extracted from HTML, PDF or Word is full of it. A different word count means different window
    // boundaries, which means different results.
    private static final Pattern RUNS = Pattern.compile("\\S+", Pattern.UNICODE_CHARACTER_CLASS);

    /** A word and its half-open character span [start, end) in the source text. */
    public record Word(String text, int start, int end) {}

    private WordsSplitter() {
    }

    public static List<Word> split(final String text) {
        return split(text, PATTERN);
    }

    /**
     * Splits on whitespace only ({@code \S+}), so punctuation stays attached to the word it
     * touches. This is what the token-classification reference pipeline chunks on.
     */
    public static List<Word> splitOnRuns(final String text) {
        return split(text, RUNS);
    }

    private static List<Word> split(final String text, final Pattern pattern) {

        final List<Word> words = new ArrayList<>();
        final Matcher matcher = pattern.matcher(text);

        while (matcher.find()) {
            words.add(new Word(matcher.group(), matcher.start(), matcher.end()));
        }

        return words;

    }

}
