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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The three places where the obvious Java call disagrees with the Python this pipeline is ported
 * from. Each divergence below was a real defect before it was a test.
 */
class TextOffsetsTest {

    private static final char NBSP = 0x00A0;
    private static final char FIGURE_SPACE = 0x2007;
    private static final char NARROW_NBSP = 0x202F;
    private static final char NEXT_LINE = 0x0085;
    private static final char EN_QUAD = 0x2000;
    private static final char IDEOGRAPHIC_SPACE = 0x3000;
    private static final char LINE_SEPARATOR = 0x2028;
    private static final char ZERO_WIDTH_SPACE = 0x200B;

    @Test
    @DisplayName("Without a supplementary character, code points and code units coincide")
    void plainTextNeedsNoConversion() {
        final TextOffsets offsets = TextOffsets.of("Mario Rossi, Roma");
        assertTrue(offsets.isIdentity());
        assertEquals(0, offsets.codeUnit(0));
        assertEquals(6, offsets.codeUnit(6));
        assertEquals(17, offsets.codeUnit(17));
    }

    @Test
    @DisplayName("One emoji shifts every later offset by one code unit")
    void supplementaryCharactersShiftEverythingAfterThem() {

        // The tokenizer reports " Mario" as code points [1,7) here, exactly as Python would index
        // it. Read as Java string indices, [1,7) is half a surrogate pair followed by " Mari".
        final String text = "😀 Mario Rossi";
        final TextOffsets offsets = TextOffsets.of(text);

        assertFalse(offsets.isIdentity());
        assertEquals(14, text.length());
        assertEquals(13, text.codePointCount(0, text.length()));

        // What the unconverted offsets would have handed back: a lone surrogate and a clipped name.
        assertNotEquals(" Mario", text.substring(1, 7));
        assertEquals(" Mario", text.substring(offsets.codeUnit(1), offsets.codeUnit(7)));
        assertEquals(" Rossi", text.substring(offsets.codeUnit(7), offsets.codeUnit(13)));
    }

    @Test
    @DisplayName("An offset past the end maps to the end rather than throwing somewhere else")
    void outOfRangeOffsetsAreClamped() {
        final TextOffsets offsets = TextOffsets.of("😀 ok");
        assertEquals(5, offsets.codeUnit(99));
        assertEquals(0, offsets.codeUnit(-1));
    }

    @Test
    @DisplayName("The non-breaking spaces count as whitespace, as they do to Python")
    void nonBreakingSpacesAreWhitespace() {

        // Character.isWhitespace deliberately says false for these; str.isspace() says True. Text
        // extracted from HTML, PDF or Word is full of U+00A0.
        for (final char c : new char[]{NBSP, FIGURE_SPACE, NARROW_NBSP, NEXT_LINE}) {
            assertFalse(Character.isWhitespace(c), () -> String.format("U+%04X", (int) c));
            assertTrue(TextOffsets.isSpace(c), () -> String.format("U+%04X", (int) c));
        }

        for (final char c : new char[]{' ', '\t', '\n', '\r', '\f', EN_QUAD, IDEOGRAPHIC_SPACE,
                LINE_SEPARATOR}) {
            assertTrue(TextOffsets.isSpace(c), () -> String.format("U+%04X", (int) c));
        }

        for (final char c : new char[]{'a', '0', '_', '-', '.', ZERO_WIDTH_SPACE}) {
            assertFalse(TextOffsets.isSpace(c), () -> String.format("U+%04X", (int) c));
        }
    }

    @Test
    @DisplayName("Letter-numbers and other-numbers are word characters, as they are to Python")
    void numbersBeyondDecimalDigitsAreAlphanumeric() {

        // Character.isLetterOrDigit covers only decimal digits; str.isalnum() also covers the
        // letter-numbers and other-numbers.
        for (final int codePoint : new int[]{0x00BD, 0x00B2, 0x2167, 0x2460}) {
            assertFalse(Character.isLetterOrDigit(codePoint), () -> String.format("U+%04X", codePoint));
            assertTrue(TextOffsets.isAlphanumeric(codePoint), () -> String.format("U+%04X", codePoint));
        }

        for (final int codePoint : new int[]{'a', 'Z', '7', 0x00E0, 0x1D400}) {
            assertTrue(TextOffsets.isAlphanumeric(codePoint), () -> String.format("U+%04X", codePoint));
        }

        for (final int codePoint : new int[]{' ', '.', ',', '-', '_', NBSP}) {
            assertFalse(TextOffsets.isAlphanumeric(codePoint), () -> String.format("U+%04X", codePoint));
        }
    }

}
