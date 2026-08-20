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

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Tuning knobs for {@link LocalPhEyeDetector}.
 *
 * <p>These are deliberately confined to this module. Core phileas owns
 * {@code PhEyeConfiguration}, which has no field for either of these concerns, so the values are
 * read from system properties (or the equivalent environment variables) when the detector is built
 * through the {@link LocalPhEyeDetectorProvider} SPI. Code that constructs the detector directly can
 * pass an instance instead.
 *
 * <h2>Two different thresholds</h2>
 * <ul>
 *   <li><b>{@code detectionThreshold} (this class)</b> is the <i>ONNX local decode threshold</i>.
 *       It decides which candidate spans leave the model at all. A span discarded here is invisible
 *       to everything downstream.</li>
 *   <li><b>{@code PhEyeFilter}'s per-label thresholds</b> are a <i>policy</i> concern. They filter
 *       spans the detector already emitted, and can only ever be more restrictive.</li>
 * </ul>
 * The default is 0.5, which is what the upstream detector hardcodes, so behaviour is unchanged
 * unless a value is supplied. Different GLiNER models calibrate very differently, so no
 * model-specific value is baked in here.
 *
 * <h2>Long input</h2>
 * A GLiNER model has a hard word limit ({@code max_len} in {@code gliner_config.json}).
 * See {@link LongTextMode}.
 */
public final class LocalPhEyeOptions {

    /** Property/env names, so callers and docs agree on one spelling. */
    public static final String THRESHOLD_PROPERTY = "phileas.pheye.onnx.detectionThreshold";
    public static final String THRESHOLD_ENV = "PHILEAS_PHEYE_ONNX_DETECTION_THRESHOLD";
    public static final String LONG_TEXT_MODE_PROPERTY = "phileas.pheye.onnx.longTextMode";
    public static final String LONG_TEXT_MODE_ENV = "PHILEAS_PHEYE_ONNX_LONG_TEXT_MODE";
    public static final String CHUNK_OVERLAP_PROPERTY = "phileas.pheye.onnx.chunkOverlapWords";
    public static final String CHUNK_OVERLAP_ENV = "PHILEAS_PHEYE_ONNX_CHUNK_OVERLAP_WORDS";
    public static final String DECODE_STRATEGY_PROPERTY = "phileas.pheye.onnx.decodeStrategy";
    public static final String DECODE_STRATEGY_ENV = "PHILEAS_PHEYE_ONNX_DECODE_STRATEGY";
    /** Prefix for a per-label threshold, e.g. {@code phileas.pheye.onnx.threshold.person=0.30}. */
    public static final String LABEL_THRESHOLD_PREFIX = "phileas.pheye.onnx.threshold.";

    /** The upstream hardcoded value; kept as the default for backward compatibility. */
    public static final double DEFAULT_DETECTION_THRESHOLD = 0.5;

    /**
     * How overlapping candidate spans are reduced to a final set.
     *
     * <p>This matters more than it looks. GLiNER scores every (span, label) pair independently, so
     * the same words routinely come back as both a strong ORGANIZATION and a slightly weaker PERSON.
     */
    public enum DecodeStrategy {

        /**
         * One greedy pass over all labels: the highest-scoring span wins and suppresses every
         * overlapping span <i>including those of other labels</i>. This is upstream's behaviour and
         * remains the default so parity is preserved.
         *
         * <p>Its failure mode is specific and bad for redaction: a wrong ORGANIZATION at 0.90 deletes
         * a correct PERSON at 0.85 over the same words, and the name is then never masked.
         */
        FLAT_GREEDY,

        /**
         * Greedy per label, independently. Spans of different labels are allowed to overlap and are
         * left for the caller's resolver to reconcile.
         *
         * <p>For a privacy filter this is the safer trade: two overlapping classifications cost a
         * little precision, whereas a suppressed PERSON is a leak.
         */
        PER_LABEL_GREEDY,

        /**
         * Flat greedy, with one exception: a span that <b>strictly contains</b> an already-kept span
         * of the <b>same label</b> is kept in its place, and the contained one is dropped.
         *
         * <p>The exception exists because plain greedy is highest-score-wins, and for masking that is
         * the wrong tie-break in exactly one situation. When a model scores the head of a name higher
         * than the whole name -- "Gianfilippo" at 0.969 against "Della Ratta Gianfilippo" at 0.953 --
         * greedy keeps the short one, and the rest of the name is left readable. Preferring the
         * container there can only mask more, never less.
         *
         * <p>It is deliberately <b>not</b> a general "widest wins": widest-wins across labels would
         * let a spurious wide ORGANIZATION delete a correct narrow PERSON, and widest-wins for
         * non-containing overlaps would resolve partial overlaps by length rather than by
         * confidence. The rule fires only for same-label, strict containment, both above threshold.
         */
        CONTAINMENT_AWARE_GREEDY
    }

    /** What to do when the input has more words than the model can take. */
    public enum LongTextMode {

        /**
         * Split the input into overlapping windows, run every window, and merge the spans.
         * The whole input is examined. This is the default: for a redaction component, silently
         * ignoring the tail of a document is a data leak.
         */
        CHUNK,

        /**
         * Refuse the input with an exception rather than examine only part of it. Fail-closed for
         * callers who would rather reject a document than risk an unredacted tail.
         */
        FAIL,

        /**
         * Examine only the first {@code max_len} words and drop the rest. This is what the upstream
         * detector does. Available for strict parity, and unsafe for redaction.
         */
        TRUNCATE
    }

    private final double detectionThreshold;
    private final LongTextMode longTextMode;
    private final Integer chunkOverlapWords;
    private final Map<String, Double> labelThresholds;
    private final DecodeStrategy decodeStrategy;

    public LocalPhEyeOptions(final double detectionThreshold, final LongTextMode longTextMode,
                             final Integer chunkOverlapWords) {
        this(detectionThreshold, longTextMode, chunkOverlapWords, Map.of(), DecodeStrategy.FLAT_GREEDY);
    }

    /**
     * @param labelThresholds per-label decode thresholds, keyed by the label string exactly as it is
     *                        passed to {@code detect}; matched case-insensitively. A label with no
     *                        entry falls back to {@code detectionThreshold}.
     */
    public LocalPhEyeOptions(final double detectionThreshold, final LongTextMode longTextMode,
                             final Integer chunkOverlapWords, final Map<String, Double> labelThresholds,
                             final DecodeStrategy decodeStrategy) {

        if (detectionThreshold < 0.0 || detectionThreshold > 1.0 || Double.isNaN(detectionThreshold)) {
            throw new IllegalArgumentException("detectionThreshold must be within [0, 1]; got " + detectionThreshold);
        }
        if (chunkOverlapWords != null && chunkOverlapWords < 0) {
            throw new IllegalArgumentException("chunkOverlapWords must not be negative; got " + chunkOverlapWords);
        }

        final Map<String, Double> normalized = new LinkedHashMap<>();
        if (labelThresholds != null) {
            for (final Map.Entry<String, Double> entry : labelThresholds.entrySet()) {
                final Double value = entry.getValue();
                if (value == null || value < 0.0 || value > 1.0 || Double.isNaN(value)) {
                    throw new IllegalArgumentException("Threshold for label '" + entry.getKey()
                            + "' must be within [0, 1]; got " + value);
                }
                normalized.put(entry.getKey().toLowerCase(Locale.ROOT), value);
            }
        }

        this.detectionThreshold = detectionThreshold;
        this.longTextMode = longTextMode == null ? LongTextMode.CHUNK : longTextMode;
        this.chunkOverlapWords = chunkOverlapWords;
        this.labelThresholds = Map.copyOf(normalized);
        this.decodeStrategy = decodeStrategy == null ? DecodeStrategy.FLAT_GREEDY : decodeStrategy;
    }

    /** Upstream-equivalent defaults for the threshold, with safe long-input handling. */
    public static LocalPhEyeOptions defaults() {
        return new LocalPhEyeOptions(DEFAULT_DETECTION_THRESHOLD, LongTextMode.CHUNK, null);
    }

    /** Per-label thresholds and a decode strategy, with a fallback threshold for unlisted labels. */
    public static LocalPhEyeOptions of(final double defaultThreshold, final Map<String, Double> labelThresholds,
                                       final DecodeStrategy decodeStrategy) {
        return new LocalPhEyeOptions(defaultThreshold, LongTextMode.CHUNK, null,
                labelThresholds, decodeStrategy);
    }

    /** Defaults with a different decode threshold. */
    public static LocalPhEyeOptions withThreshold(final double detectionThreshold) {
        return new LocalPhEyeOptions(detectionThreshold, LongTextMode.CHUNK, null);
    }

    /**
     * Read the options from system properties, falling back to environment variables and then to
     * the defaults. An unparseable value is an error, never a silent fallback: a redaction
     * component that quietly ignores a misspelled threshold is worse than one that refuses to start.
     */
    public static LocalPhEyeOptions fromEnvironment() {

        final double threshold = parseDouble(
                value(THRESHOLD_PROPERTY, THRESHOLD_ENV), DEFAULT_DETECTION_THRESHOLD, THRESHOLD_PROPERTY);

        final String modeValue = value(LONG_TEXT_MODE_PROPERTY, LONG_TEXT_MODE_ENV);
        final LongTextMode mode;
        if (modeValue == null || modeValue.isBlank()) {
            mode = LongTextMode.CHUNK;
        } else {
            try {
                mode = LongTextMode.valueOf(modeValue.trim().toUpperCase(Locale.ROOT));
            } catch (final IllegalArgumentException e) {
                throw new IllegalArgumentException("Invalid " + LONG_TEXT_MODE_PROPERTY + " '" + modeValue
                        + "'. Expected one of CHUNK, FAIL, TRUNCATE.", e);
            }
        }

        final String overlapValue = value(CHUNK_OVERLAP_PROPERTY, CHUNK_OVERLAP_ENV);
        Integer overlap = null;
        if (overlapValue != null && !overlapValue.isBlank()) {
            try {
                overlap = Integer.valueOf(overlapValue.trim());
            } catch (final NumberFormatException e) {
                throw new IllegalArgumentException("Invalid " + CHUNK_OVERLAP_PROPERTY + " '" + overlapValue
                        + "'. Expected an integer number of words.", e);
            }
        }

        final String strategyValue = value(DECODE_STRATEGY_PROPERTY, DECODE_STRATEGY_ENV);
        final DecodeStrategy strategy;
        if (strategyValue == null || strategyValue.isBlank()) {
            strategy = DecodeStrategy.FLAT_GREEDY;
        } else {
            try {
                strategy = DecodeStrategy.valueOf(strategyValue.trim().toUpperCase(Locale.ROOT));
            } catch (final IllegalArgumentException e) {
                throw new IllegalArgumentException("Invalid " + DECODE_STRATEGY_PROPERTY + " '" + strategyValue
                        + "'. Expected FLAT_GREEDY or PER_LABEL_GREEDY.", e);
            }
        }

        // Per-label thresholds are discovered by prefix, because GLiNER labels are free text and
        // cannot be enumerated in advance.
        final Map<String, Double> perLabel = new LinkedHashMap<>();
        for (final String name : System.getProperties().stringPropertyNames()) {
            if (name.startsWith(LABEL_THRESHOLD_PREFIX) && name.length() > LABEL_THRESHOLD_PREFIX.length()) {
                final String label = name.substring(LABEL_THRESHOLD_PREFIX.length());
                perLabel.put(label, parseDouble(System.getProperty(name), threshold, name));
            }
        }

        return new LocalPhEyeOptions(threshold, mode, overlap, perLabel, strategy);

    }

    private static String value(final String property, final String environmentVariable) {
        final String fromProperty = System.getProperty(property);
        return fromProperty != null ? fromProperty : System.getenv(environmentVariable);
    }

    private static double parseDouble(final String raw, final double fallback, final String name) {
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Double.parseDouble(raw.trim());
        } catch (final NumberFormatException e) {
            throw new IllegalArgumentException("Invalid " + name + " '" + raw + "'. Expected a number in [0, 1].", e);
        }
    }

    public double detectionThreshold() {
        return detectionThreshold;
    }

    /** The threshold that applies to one label: its own if configured, otherwise the default. */
    public double thresholdFor(final String label) {
        final Double specific = labelThresholds.get(label.toLowerCase(Locale.ROOT));
        return specific != null ? specific : detectionThreshold;
    }

    public Map<String, Double> labelThresholds() {
        return labelThresholds;
    }

    public DecodeStrategy decodeStrategy() {
        return decodeStrategy;
    }

    public LongTextMode longTextMode() {
        return longTextMode;
    }

    /**
     * Words shared between consecutive windows. When unset, the detector derives a value from the
     * model's {@code max_width} so that any span the model could produce fits entirely inside at
     * least one window.
     */
    public Integer chunkOverlapWords() {
        return chunkOverlapWords;
    }

    @Override
    public String toString() {
        return "LocalPhEyeOptions[detectionThreshold=" + detectionThreshold
                + ", labelThresholds=" + labelThresholds
                + ", decodeStrategy=" + decodeStrategy
                + ", longTextMode=" + longTextMode
                + ", chunkOverlapWords=" + (chunkOverlapWords == null ? "auto" : chunkOverlapWords) + "]";
    }

}
