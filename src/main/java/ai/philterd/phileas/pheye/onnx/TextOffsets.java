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

/**
 * Character predicates and index conversions that have to agree with Python, because the pipeline
 * ported here is specified by its Python implementation.
 *
 * <p>Three places where the obvious Java call is the wrong one:
 *
 * <ul>
 *   <li><b>Offsets.</b> A fast tokenizer's character offsets are indices into a sequence of <i>code
 *       points</i> — that is what a Python string is indexed by. A Java {@code String} is indexed by
 *       UTF-16 <i>code units</i>. The two agree until the text contains one character outside the
 *       Basic Multilingual Plane — an emoji, a rarer CJK ideograph — and then every later offset is
 *       one too small per surrogate pair. It never throws: {@code substring} quietly returns the
 *       wrong characters, and a redaction component masks the wrong ones.</li>
 *   <li><b>Whitespace.</b> {@link Character#isWhitespace} deliberately excludes the non-breaking
 *       spaces; Python's {@code str.isspace()} includes them. Text extracted from HTML, PDF or Word
 *       is full of U+00A0.</li>
 *   <li><b>Word characters.</b> {@link Character#isLetterOrDigit} counts only decimal digits;
 *       Python's {@code str.isalnum()} also counts letter-numbers and other-numbers.</li>
 * </ul>
 */
final class TextOffsets {

    /** U+0085 NEXT LINE: whitespace to Python, to neither Java predicate. */
    private static final char NEXT_LINE = 0x0085;

    /** Maps a code-point index to a code-unit index, or {@code null} when the two coincide. */
    private final int[] codeUnitAt;

    private TextOffsets(final int[] codeUnitAt) {
        this.codeUnitAt = codeUnitAt;
    }

    /**
     * Build the conversion for one string.
     *
     * <p>Text outside the Basic Multilingual Plane is rare, and the two index spaces coincide
     * without it, so the common case allocates nothing and converts by returning its argument.
     */
    static TextOffsets of(final String text) {

        final int codeUnits = text.length();
        final int codePoints = text.codePointCount(0, codeUnits);
        if (codePoints == codeUnits) {
            return new TextOffsets(null);
        }

        // One extra slot, so an end offset pointing just past the last code point still maps.
        final int[] map = new int[codePoints + 1];
        int codePoint = 0;
        for (int codeUnit = 0; codeUnit < codeUnits; ) {
            map[codePoint++] = codeUnit;
            codeUnit += Character.charCount(text.codePointAt(codeUnit));
        }
        map[codePoints] = codeUnits;
        return new TextOffsets(map);

    }

    /** The UTF-16 index of the code point at {@code codePointIndex}. */
    int codeUnit(final int codePointIndex) {
        if (codeUnitAt == null) {
            return codePointIndex;
        }
        // A tokenizer reporting an offset past the end of what it was given would be a bug in the
        // tokenizer, but clamping keeps it from surfacing as a StringIndexOutOfBounds far away.
        final int clamped = Math.min(Math.max(codePointIndex, 0), codeUnitAt.length - 1);
        return codeUnitAt[clamped];
    }

    /** True when the string is entirely in the Basic Multilingual Plane and needs no conversion. */
    boolean isIdentity() {
        return codeUnitAt == null;
    }

    /**
     * Python's {@code str.isspace()}, which {@link Character#isWhitespace} alone does not match: it
     * excludes the non-breaking spaces (U+00A0, U+2007, U+202F), and both predicates miss U+0085.
     */
    static boolean isSpace(final char c) {
        return Character.isWhitespace(c) || Character.isSpaceChar(c) || c == NEXT_LINE;
    }

    /**
     * Python's {@code str.isalnum()}: any letter, or any number — decimal, letter-number or
     * other-number. {@link Character#isLetterOrDigit} covers only the decimal ones.
     */
    static boolean isAlphanumeric(final int codePoint) {
        if (Character.isLetterOrDigit(codePoint)) {
            return true;
        }
        final int type = Character.getType(codePoint);
        return type == Character.LETTER_NUMBER || type == Character.OTHER_NUMBER;
    }

}
