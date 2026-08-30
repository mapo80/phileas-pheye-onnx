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

import ai.philterd.phileas.pheye.onnx.LocalTokenClassifierDetector.Entity;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The reduction from per-window entities to a final span set, tested without a model.
 *
 * <p>Each step exists for a reason that shows up as a defect when it is missing, and each test here
 * names that defect rather than the mechanism.
 */
class TokenClassifierReduceTest {

    @Test
    @DisplayName("The same entity found in two overlapping windows is emitted once")
    void duplicatesFromOverlappingWindowsCollapse() {

        final String text = "Il sig. Mario Rossi risiede a Roma.";
        final List<Entity> reduced = LocalTokenClassifierDetector.reduce(List.of(
                new Entity("FULLNAME", 8, 19, 0.99),
                new Entity("FULLNAME", 8, 19, 0.98)), text);

        assertEquals(1, reduced.size());
        assertEquals("Mario Rossi", text.substring(reduced.get(0).start(), reduced.get(0).end()));
        assertEquals(0.99, reduced.get(0).score(), 1e-9);
    }

    @Test
    @DisplayName("When two windows disagree on the boundary, the more confident one wins")
    void disagreeingWindowsResolveByScore() {

        final String text = "Il sig. Mario Rossi risiede a Roma.";
        final List<Entity> reduced = LocalTokenClassifierDetector.reduce(List.of(
                new Entity("FULLNAME", 8, 13, 0.70),
                new Entity("FULLNAME", 8, 19, 0.95)), text);

        assertEquals(1, reduced.size());
        assertEquals("Mario Rossi", text.substring(reduced.get(0).start(), reduced.get(0).end()));
    }

    @Test
    @DisplayName("A leading space from the tokenizer's offsets is trimmed off the span")
    void leadingWhitespaceIsTrimmed() {

        // This tokenizer's offsets include the space before a word, so " Rossi" is what the model
        // hands back. Masking it would move the placeholder one character to the left.
        final String text = "Il sig. Mario Rossi risiede a Roma.";
        final List<Entity> reduced = LocalTokenClassifierDetector.reduce(
                List.of(new Entity("FULLNAME", 7, 19, 0.99)), text);

        assertEquals(1, reduced.size());
        assertEquals("Mario Rossi", text.substring(reduced.get(0).start(), reduced.get(0).end()));
    }

    @Test
    @DisplayName("A span covering part of a word is widened to the whole word")
    void partialWordsAreWidened() {

        // "No" of "Novara": replacing only that leaves "[CITY_1]vara", which is still readable.
        final String text = "Residente a Novara da tre anni.";
        final List<Entity> reduced = LocalTokenClassifierDetector.reduce(
                List.of(new Entity("CITY", 12, 14, 0.88)), text);

        assertEquals(1, reduced.size());
        assertEquals("Novara", text.substring(reduced.get(0).start(), reduced.get(0).end()));
    }

    @Test
    @DisplayName("Widening never leaves two spans masking the same word twice")
    void wideningCannotProduceOverlaps() {

        final String text = "Foglio 12, particella 345.";
        final List<Entity> reduced = LocalTokenClassifierDetector.reduce(List.of(
                new Entity("CATASTO", 7, 8, 0.90),
                new Entity("CATASTO", 8, 9, 0.90)), text);

        assertEquals(1, reduced.size());
        assertEquals("12", text.substring(reduced.get(0).start(), reduced.get(0).end()));
    }

    @Test
    @DisplayName("Adjacent spans of different labels stay separate")
    void adjacentDifferentLabelsAreNotFused() {

        final String text = "Via Garibaldi 24";
        final List<Entity> reduced = LocalTokenClassifierDetector.reduce(List.of(
                new Entity("STREET", 0, 13, 0.99),
                new Entity("BUILDINGNUM", 14, 16, 0.99)), text);

        assertEquals(2, reduced.size());
        assertEquals("STREET", reduced.get(0).label());
        assertEquals("BUILDINGNUM", reduced.get(1).label());
    }

    @Test
    @DisplayName("A whitespace-only span is dropped rather than emitted as an empty mask")
    void whitespaceOnlySpansAreDropped() {
        final String text = "Roma   Milano";
        assertTrue(LocalTokenClassifierDetector.reduce(
                List.of(new Entity("CITY", 4, 7, 0.99)), text).isEmpty());
    }

    @Test
    @DisplayName("The result is ordered by position, whatever order the windows produced")
    void resultIsOrderedByPosition() {

        final String text = "Mario Rossi, Roma, 12/06/1985";
        final List<Entity> reduced = LocalTokenClassifierDetector.reduce(List.of(
                new Entity("DATE", 19, 29, 0.97),
                new Entity("FULLNAME", 0, 11, 0.99),
                new Entity("CITY", 13, 17, 0.98)), text);

        assertEquals(List.of("FULLNAME", "CITY", "DATE"), reduced.stream().map(Entity::label).toList());
    }

    @Test
    @DisplayName("Reduction of nothing is nothing")
    void emptyInputGivesEmptyOutput() {
        assertTrue(LocalTokenClassifierDetector.reduce(List.of(), "anything").isEmpty());
    }

}
