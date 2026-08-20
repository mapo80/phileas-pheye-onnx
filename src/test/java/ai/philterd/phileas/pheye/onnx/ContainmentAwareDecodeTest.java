package ai.philterd.phileas.pheye.onnx;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The one exception {@code CONTAINMENT_AWARE_GREEDY} makes to greedy decoding, and the three places
 * it deliberately does not.
 *
 * <p>The rule matters for masking rather than for extraction: greedy is highest-score-wins, and when
 * a model scores the head of a name above the whole name, highest-score-wins leaves the rest of the
 * name readable. Preferring the container can only mask more. Everything else stays greedy, because
 * a general "widest wins" would let a wide spurious span of one label delete a correct narrow span
 * of another.
 */
class ContainmentAwareDecodeTest {

    private static LocalPhEyeDetector.Candidate candidate(final int startWord, final int endWord,
                                                          final int classIndex, final double score) {
        return new LocalPhEyeDetector.Candidate(startWord, endWord, classIndex, score);
    }

    private static List<String> spans(final List<LocalPhEyeDetector.Candidate> kept) {
        final List<String> described = new ArrayList<>();
        for (final LocalPhEyeDetector.Candidate c : kept) {
            described.add(c.startWord() + "-" + c.endWord() + "/" + c.classIndex());
        }
        return described;
    }

    @Test
    void aContainingSpanOfTheSameLabelDisplacesTheHigherScoringSpanItContains() {
        // "Della Ratta Gianfilippo" (words 0-2) at 0.953 against "Gianfilippo" (word 2) at 0.969.
        final List<LocalPhEyeDetector.Candidate> kept = LocalPhEyeDetector.containmentAwareGreedy(
                new ArrayList<>(List.of(candidate(2, 2, 0, 0.969), candidate(0, 2, 0, 0.953))));
        assertEquals(List.of("0-2/0"), spans(kept),
                "the container must survive, so no part of the name is left readable");
    }

    @Test
    void plainGreedyKeepsTheShortOneSoTheDifferenceIsRealAndNotCosmetic() {
        final List<LocalPhEyeDetector.Candidate> kept = LocalPhEyeDetector.greedyNonOverlap(
                new ArrayList<>(List.of(candidate(2, 2, 0, 0.969), candidate(0, 2, 0, 0.953))));
        assertEquals(List.of("2-2/0"), spans(kept));
    }

    @Test
    void aContainingSpanOfADIFFERENTLabelDoesNotDisplaceAnything() {
        // A wide ORGANIZATION must never delete a correct narrow PERSON: that is how a name that the
        // model did find ends up unmasked, which is the failure the fork exists to avoid.
        final List<LocalPhEyeDetector.Candidate> kept = LocalPhEyeDetector.containmentAwareGreedy(
                new ArrayList<>(List.of(candidate(2, 2, 0, 0.969), candidate(0, 2, 1, 0.953))));
        assertEquals(List.of("2-2/0"), spans(kept));
    }

    @Test
    void aPartialOverlapIsStillResolvedByScoreRatherThanByLength() {
        final List<LocalPhEyeDetector.Candidate> kept = LocalPhEyeDetector.containmentAwareGreedy(
                new ArrayList<>(List.of(candidate(1, 2, 0, 0.90), candidate(0, 1, 0, 0.80))));
        assertEquals(List.of("1-2/0"), spans(kept),
                "containment is the exception; overlapping-but-not-containing stays greedy");
    }

    @Test
    void aSpanStraddlingTwoKeptSpansIsDroppedRatherThanDeletingBoth() {
        final List<LocalPhEyeDetector.Candidate> kept = LocalPhEyeDetector.containmentAwareGreedy(
                new ArrayList<>(List.of(candidate(0, 0, 0, 0.99), candidate(4, 4, 0, 0.98),
                        candidate(0, 4, 0, 0.97))));
        assertEquals(List.of("0-0/0", "4-4/0"), spans(kept),
                "promoting it would silently delete the second kept span");
    }

    @Test
    void identicalSpansAreNotTreatedAsContainmentSoTheHigherScoreWins() {
        final List<LocalPhEyeDetector.Candidate> kept = LocalPhEyeDetector.containmentAwareGreedy(
                new ArrayList<>(List.of(candidate(0, 2, 0, 0.90), candidate(0, 2, 0, 0.80))));
        assertEquals(1, kept.size());
        assertTrue(kept.get(0).score() > 0.85);
    }

    @Test
    void nonOverlappingSpansAreAllKeptExactlyAsBefore() {
        final List<LocalPhEyeDetector.Candidate> kept = LocalPhEyeDetector.containmentAwareGreedy(
                new ArrayList<>(List.of(candidate(0, 1, 0, 0.90), candidate(3, 4, 1, 0.80))));
        assertEquals(List.of("0-1/0", "3-4/1"), spans(kept));
    }
}
