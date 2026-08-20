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

import ai.philterd.phileas.pheye.onnx.LocalPhEyeDetector.Candidate;
import ai.philterd.phileas.pheye.onnx.LocalPhEyeOptions.DecodeStrategy;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The cross-label suppression problem, and what PER_LABEL_GREEDY does about it.
 *
 * <p>GLiNER scores every (span, label) pair independently, so the same words routinely come back as
 * both a confident ORGANIZATION and a slightly less confident PERSON. Under one flat greedy pass the
 * ORGANIZATION wins and the PERSON is gone — the name is then never masked, which is a leak. These
 * tests pin both behaviours so the choice is explicit rather than accidental.
 */
class DecodeStrategyTest {

    /** class 0 = person, class 1 = organization. */
    private static final int PERSON = 0;
    private static final int ORGANIZATION = 1;

    @Test
    @DisplayName("FLAT_GREEDY lets a stronger ORGANIZATION delete a correct PERSON")
    void flatGreedyLosesThePerson() {

        // Same two words scored as both labels; the wrong label scores higher.
        final List<Candidate> candidates = new ArrayList<>(List.of(
                new Candidate(2, 3, ORGANIZATION, 0.90),
                new Candidate(2, 3, PERSON, 0.85)));

        final List<Candidate> kept = LocalPhEyeDetector.decode(candidates, DecodeStrategy.FLAT_GREEDY, 2);

        assertEquals(1, kept.size(), "flat greedy should keep only one span");
        assertEquals(ORGANIZATION, kept.get(0).classIndex());
        assertTrue(kept.stream().noneMatch(c -> c.classIndex() == PERSON),
                "this is the documented failure mode: the PERSON is gone");

    }

    @Test
    @DisplayName("PER_LABEL_GREEDY keeps both, so the PERSON survives to the resolver")
    void perLabelGreedyKeepsThePerson() {

        final List<Candidate> candidates = new ArrayList<>(List.of(
                new Candidate(2, 3, ORGANIZATION, 0.90),
                new Candidate(2, 3, PERSON, 0.85)));

        final List<Candidate> kept = LocalPhEyeDetector.decode(candidates, DecodeStrategy.PER_LABEL_GREEDY, 2);

        assertEquals(2, kept.size(), "both labels should survive, got " + kept);
        assertTrue(kept.stream().anyMatch(c -> c.classIndex() == PERSON), "the PERSON must survive");
        assertTrue(kept.stream().anyMatch(c -> c.classIndex() == ORGANIZATION));

    }

    @Test
    @DisplayName("PER_LABEL_GREEDY still suppresses overlaps WITHIN a label")
    void perLabelGreedyStillDeduplicatesWithinALabel() {

        // "Mario", "Rossi" and "Mario Rossi" all as PERSON. The two single-word spans do not overlap
        // each other, so both are kept and the wider span that covers both is dropped. Note this is
        // the opposite of what a masking layer wants, which is why the caller's resolver prefers the
        // widest span: see RecognizerPipeline in the consumer.
        final List<Candidate> candidates = new ArrayList<>(List.of(
                new Candidate(2, 2, PERSON, 0.95),
                new Candidate(3, 3, PERSON, 0.93),
                new Candidate(2, 3, PERSON, 0.90)));

        final List<Candidate> kept = LocalPhEyeDetector.decode(candidates, DecodeStrategy.PER_LABEL_GREEDY, 2);

        assertEquals(2, kept.size(), "the two disjoint single-word spans should survive, got " + kept);
        assertTrue(kept.stream().noneMatch(c -> c.startWord() == 2 && c.endWord() == 3),
                "the wider span overlaps both and must be dropped");

    }

    @Test
    @DisplayName("Genuinely overlapping same-label spans collapse to the highest score")
    void trulyOverlappingSameLabelSpansCollapse() {

        // Both cover word 3, so they really do overlap.
        final List<Candidate> candidates = new ArrayList<>(List.of(
                new Candidate(2, 3, PERSON, 0.95),
                new Candidate(3, 4, PERSON, 0.93)));

        final List<Candidate> kept = LocalPhEyeDetector.decode(candidates, DecodeStrategy.PER_LABEL_GREEDY, 2);

        assertEquals(1, kept.size(), "overlapping same-label spans must collapse, got " + kept);
        assertEquals(0.95, kept.get(0).score());

    }

    @Test
    @DisplayName("Non-overlapping spans are untouched by either strategy")
    void nonOverlappingSpansSurviveBoth() {

        final List<Candidate> candidates = List.of(
                new Candidate(0, 1, PERSON, 0.9),
                new Candidate(5, 6, ORGANIZATION, 0.8));

        assertEquals(2, LocalPhEyeDetector.decode(new ArrayList<>(candidates),
                DecodeStrategy.FLAT_GREEDY, 2).size());
        assertEquals(2, LocalPhEyeDetector.decode(new ArrayList<>(candidates),
                DecodeStrategy.PER_LABEL_GREEDY, 2).size());

    }

    @Test
    @DisplayName("Results are ordered by position for both strategies")
    void resultsAreOrdered() {

        final List<Candidate> candidates = new ArrayList<>(List.of(
                new Candidate(9, 9, PERSON, 0.7),
                new Candidate(1, 1, ORGANIZATION, 0.8),
                new Candidate(5, 5, PERSON, 0.9)));

        for (final DecodeStrategy strategy : DecodeStrategy.values()) {
            final List<Candidate> kept = LocalPhEyeDetector.decode(new ArrayList<>(candidates), strategy, 2);
            for (int i = 1; i < kept.size(); i++) {
                assertTrue(kept.get(i - 1).startWord() <= kept.get(i).startWord(),
                        strategy + " returned unordered spans: " + kept);
            }
        }

    }

}
