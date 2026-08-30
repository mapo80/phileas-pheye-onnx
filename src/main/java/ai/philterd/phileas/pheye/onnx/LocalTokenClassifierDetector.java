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
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeDetector;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeSpan;

import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * On-device inference for a <b>BIO token-classification</b> PII model, via ONNX Runtime.
 *
 * <p>This is the second model family the module drives, alongside {@link LocalPhEyeDetector}'s
 * zero-shot GLiNER span models. The two differ in every mechanical respect and share nothing but
 * the {@link PhEyeDetector} contract and {@link LocalPhEyeOptions}:
 *
 * <table border="1">
 *   <caption>The two model families</caption>
 *   <tr><th></th><th>GLiNER ({@link LocalPhEyeDetector})</th><th>token classifier (this)</th></tr>
 *   <tr><td>Labels</td><td>zero-shot: the caller's labels are the prompt</td>
 *       <td>fixed taxonomy, baked into the weights; the caller's labels select from it</td></tr>
 *   <tr><td>ONNX inputs</td><td>six tensors, including enumerated candidate spans</td>
 *       <td>{@code input_ids} and {@code attention_mask}</td></tr>
 *   <tr><td>Output</td><td>a score per (span, label) pair, sigmoid</td>
 *       <td>a distribution over BIO classes per sub-token, softmax</td></tr>
 *   <tr><td>Overlapping labels</td><td>routine, hence the decode strategies</td>
 *       <td>impossible: one argmax per token</td></tr>
 * </table>
 *
 * <p>Because a token classifier picks exactly one class per sub-token, the cross-label suppression
 * that {@link LocalPhEyeOptions.DecodeStrategy} exists to control cannot arise here, and that option
 * is not read on this path. {@code detectionThreshold}, the per-label thresholds, and
 * {@link LocalPhEyeOptions.LongTextMode} all apply as usual.
 *
 * <h2>Pipeline</h2>
 * A port of the reference Python pipeline, step for step, so a threshold calibrated there transfers
 * here unchanged:
 * <ol>
 *   <li>split the text into whitespace-delimited words and window them, {@code max_words} at a time
 *       with {@code overlap_words} shared;</li>
 *   <li>tokenize each window's exact substring, keeping character offsets, and run the model;</li>
 *   <li>softmax per sub-token, take the argmax class;</li>
 *   <li>group consecutive sub-tokens into entities as HuggingFace's
 *       {@code aggregation_strategy="simple"} does: a run continues while the entity type is
 *       unchanged and the tag is not a fresh {@code B-}; the entity's score is the mean of its
 *       sub-token scores;</li>
 *   <li>drop entities below the threshold for their label;</li>
 *   <li>reduce to a final set (see {@link #reduce}).</li>
 * </ol>
 *
 * <h2>Fail-closed</h2>
 * A missing file, an {@code id2label} that is not BIO, an ONNX graph with the wrong signature, or a
 * class count that disagrees with {@code id2label} is a constructor failure. Asking for labels that
 * are not in the model's taxonomy is a {@code detect} failure. Both would otherwise show up as
 * zero detections, which in a redaction component is indistinguishable from a clean document.
 */
public class LocalTokenClassifierDetector implements PhEyeDetector {

    /** The tensors this export must accept. */
    static final List<String> REQUIRED_INPUTS = List.of("input_ids", "attention_mask");

    /** The output the decode reads. */
    static final String REQUIRED_OUTPUT = "logits";

    private final TokenClassifierConfig config;
    private final HuggingFaceTokenizer tokenizer;
    private final OrtEnvironment ortEnvironment;
    private final OrtSession session;
    private final LocalPhEyeOptions options;
    private final int overlapWords;

    public LocalTokenClassifierDetector(final Path modelDir) throws Exception {
        this(modelDir, LocalPhEyeOptions.defaults());
    }

    public LocalTokenClassifierDetector(final Path modelDir, final double detectionThreshold) throws Exception {
        this(modelDir, LocalPhEyeOptions.withThreshold(detectionThreshold));
    }

    public LocalTokenClassifierDetector(final Path modelDir, final LocalPhEyeOptions options) throws Exception {

        this.config = TokenClassifierConfig.load(modelDir);

        // The model directory may declare the threshold its weights were calibrated at. It wins over
        // the library default -- which exists to reproduce upstream GLiNER and means nothing here --
        // but never over a threshold the caller actually chose. Applied in the constructor rather
        // than only in LocalDetectorFactory, so building the detector directly is not a quieter way
        // to end up at the wrong operating point.
        final LocalPhEyeOptions requestedOptions = options == null ? LocalPhEyeOptions.defaults() : options;
        this.options = config.calibratedThreshold == null
                ? requestedOptions
                : requestedOptions.withDefaultThreshold(config.calibratedThreshold);

        // The caller's overlap wins when set, but never so wide that the window stops advancing.
        final Integer requested = this.options.chunkOverlapWords();
        this.overlapWords = requested == null
                ? config.overlapWords
                : Math.min(requested, config.maxWords - 1);

        final Path tokenizerFile = modelDir.resolve("tokenizer.json");
        if (!Files.isReadable(tokenizerFile)) {
            throw new IllegalArgumentException("Missing or unreadable tokenizer.json at " + tokenizerFile + ".");
        }

        final Path onnx = resolveOnnxPath(modelDir);
        if (!Files.isReadable(onnx)) {
            throw new IllegalArgumentException("Missing or unreadable ONNX model at " + onnx
                    + ". Expected onnx/model.onnx or model.onnx under " + modelDir + ".");
        }

        // This constructor is meant to throw -- that is the fail-closed contract -- so everything
        // native it has opened has to be released on the way out. Both handles are off-heap and the
        // garbage collector will not reclaim them.
        HuggingFaceTokenizer openedTokenizer = null;
        OrtSession openedSession = null;
        try {
            openedTokenizer = Tokenizers.load(tokenizerFile);
            this.ortEnvironment = OrtEnvironment.getEnvironment();
            try (final OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions()) {
                openedSession = ortEnvironment.createSession(onnx.toString(), sessionOptions);
            }
            this.tokenizer = openedTokenizer;
            this.session = openedSession;
            validateSignature(onnx);
        } catch (final Throwable failure) {
            closeQuietly(openedSession, failure);
            closeQuietly(openedTokenizer, failure);
            throw failure;
        }

    }

    private static void closeQuietly(final AutoCloseable closeable, final Throwable failure) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (final Exception e) {
            // The construction failure is what the caller needs to see; losing this one would hide it.
            failure.addSuppressed(e);
        }
    }

    private static Path resolveOnnxPath(final Path modelDir) {
        final Path nested = modelDir.resolve("onnx").resolve("model.onnx");
        return Files.exists(nested) ? nested : modelDir.resolve("model.onnx");
    }

    /**
     * Reject a graph that is not a token-classification export, and one whose class count disagrees
     * with {@code id2label}: a mismatch there silently shifts every label by the difference.
     */
    private void validateSignature(final Path onnx) throws Exception {

        final Set<String> inputs = new LinkedHashSet<>(session.getInputNames());
        final List<String> missing = new ArrayList<>();
        for (final String required : REQUIRED_INPUTS) {
            if (!inputs.contains(required)) {
                missing.add(required);
            }
        }
        if (!missing.isEmpty()) {
            throw new IllegalStateException("The ONNX model at " + onnx + " is not a token-classification"
                    + " export: missing input tensors " + missing + ". Present inputs: " + inputs + ".");
        }

        // An input we do not feed is not a graph we can drive. ONNX Runtime would reject the call at
        // the first document instead, which for a filter built at startup means the failure surfaces
        // in production traffic rather than at construction.
        final List<String> unexpected = new ArrayList<>(inputs);
        unexpected.removeAll(REQUIRED_INPUTS);
        if (!unexpected.isEmpty()) {
            throw new IllegalStateException("The ONNX model at " + onnx + " requires input tensors "
                    + unexpected + ", which this detector does not provide. It drives graphs taking"
                    + " exactly " + REQUIRED_INPUTS + "; re-export the model without the extra inputs"
                    + " (token_type_ids is the usual one).");
        }

        if (!session.getOutputNames().contains(REQUIRED_OUTPUT)) {
            throw new IllegalStateException("The ONNX model at " + onnx + " does not expose a '"
                    + REQUIRED_OUTPUT + "' output. Present outputs: " + session.getOutputNames() + ".");
        }

        final var info = session.getOutputInfo().get(REQUIRED_OUTPUT).getInfo();
        if (info instanceof ai.onnxruntime.TensorInfo tensorInfo) {
            final long[] shape = tensorInfo.getShape();
            // [batch, sequence, classes]; the first two are dynamic, the last is fixed by the head.
            if (shape.length == 3 && shape[2] > 0 && shape[2] != config.id2label.size()) {
                throw new IllegalStateException("The ONNX model at " + onnx + " emits " + shape[2]
                        + " classes but " + TokenClassifierConfig.MODEL_FILE + " declares "
                        + config.id2label.size() + " labels. One of the two is from a different model.");
            }
        }

    }

    /**
     * {@inheritDoc}
     *
     * <p>{@code labels} selects from the model's fixed taxonomy and is matched case-insensitively,
     * so {@code "fullname"} and {@code "FULLNAME"} both work. Requesting labels of which
     * <i>none</i> is in the taxonomy is an error rather than an empty result: the two are
     * indistinguishable downstream, and the empty one hides a misconfiguration behind a clean
     * document. Use {@link #entityTypes()} to discover what this model emits.
     */
    @Override
    public List<PhEyeSpan> detect(final String text, final Collection<String> labels,
                                  final String context, final int piece) throws Exception {

        final List<PhEyeSpan> empty = new ArrayList<>();
        if (labels == null || labels.isEmpty() || text == null || text.isBlank()) {
            return empty;
        }

        final Set<String> wanted = new LinkedHashSet<>();
        final List<String> unknown = new ArrayList<>();
        for (final String label : labels) {
            final String resolved = config.resolveEntityType(label);
            if (resolved == null) {
                unknown.add(label);
            } else {
                wanted.add(resolved);
            }
        }
        if (wanted.isEmpty()) {
            throw new IllegalArgumentException("None of the requested labels " + unknown + " is in this"
                    + " model's taxonomy " + config.entityTypes + ". A token-classification model cannot"
                    + " be prompted with new labels; returning no spans here would look like a document"
                    + " with no personal data.");
        }

        final List<WordsSplitter.Word> words = WordsSplitter.splitOnRuns(text);
        if (words.isEmpty()) {
            return empty;
        }

        final List<int[]> windows = LocalPhEyeDetector.planWindows(
                words.size(), config.maxWords, overlapWords, options.longTextMode());

        final List<Entity> entities = new ArrayList<>();
        for (final int[] window : windows) {
            scoreWordRange(text, words, window[0], window[1], entities);
        }

        // Filter to the requested labels *before* reducing, not after. The reduce is greedy across
        // labels, so an unwanted high-scoring entity can suppress a wanted one over the same
        // characters -- and filtering afterwards would then leave nothing there at all. For a
        // redaction component that is a leak, and a silent one.
        entities.removeIf(entity -> !wanted.contains(entity.label));

        final List<Entity> reduced = reduce(entities, text);

        final List<PhEyeSpan> spans = new ArrayList<>(reduced.size());
        for (final Entity entity : reduced) {
            final PhEyeSpan span = new PhEyeSpan();
            span.setStart(entity.start);
            span.setEnd(entity.end);
            span.setLabel(entity.label);
            span.setText(text.substring(entity.start, entity.end));
            span.setScore(entity.score);
            spans.add(span);
        }

        return spans;

    }

    /**
     * Score one word range, appending its above-threshold entities in global character coordinates.
     *
     * <p>The range is normally one window and encodes to fewer sub-tokens than the encoder can take,
     * so it runs in a single pass -- which is what the reference pipeline does, and what the
     * calibrated threshold was measured against. Splitting earlier would be a different operating
     * point, not a safer one.
     *
     * <p>A range that will not fit at all -- a single multi-kilobyte "word" such as an embedded
     * base64 blob, which no amount of word-level windowing can break up -- is scored in sub-token
     * windows instead of being truncated, so every sub-token is still examined. Truncating here
     * would leave part of the document unlabelled and unreported, the failure mode
     * {@link LocalPhEyeOptions.LongTextMode} exists to prevent at the level above.
     */
    private void scoreWordRange(final String text, final List<WordsSplitter.Word> words,
                                final int from, final int to, final List<Entity> out) throws Exception {

        if (to <= from) {
            return;
        }

        final int offset = words.get(from).start();
        final String window = text.substring(offset, words.get(to - 1).end());
        final Encoding encoding = tokenizer.encode(window);
        final int length = encoding.getIds().length;

        // The tokenizer reports offsets into the window's code points; `offset` and everything
        // downstream are Java string indices. See TextOffsets.
        final TextOffsets offsets = TextOffsets.of(window);

        if (length <= config.maxTokens) {
            scoreEncoding(encoding, 0, length, offset, offsets, out);
            return;
        }

        final int stride = Math.max(config.maxTokens - Math.max(config.maxTokens / 8, 1), 1);
        for (int start = 0; start < length; start += stride) {
            final int end = Math.min(start + config.maxTokens, length);
            scoreEncoding(encoding, start, end, offset, offsets, out);
            if (end == length) {
                break;
            }
        }

    }

    /**
     * Run the model over {@code encoding[from, to)} and group the result into entities.
     *
     * <p>Grouping mirrors HuggingFace's {@code aggregation_strategy="simple"}: an entity run
     * continues while the type is unchanged and the tag is not a fresh {@code B-}, and its score is
     * the mean of its sub-tokens' probabilities. {@code O} runs are formed the same way and dropped
     * at the end, which is what keeps them acting as separators.
     */
    private void scoreEncoding(final Encoding encoding, final int from, final int to,
                               final int offset, final TextOffsets offsets,
                               final List<Entity> out) throws Exception {

        final long[] allIds = encoding.getIds();
        final long[] allMask = encoding.getAttentionMask();
        final long[] special = encoding.getSpecialTokenMask();
        final var charSpans = encoding.getCharTokenSpans();

        final int length = to - from;
        final long[] ids = new long[length];
        final long[] mask = new long[length];
        System.arraycopy(allIds, from, ids, 0, length);
        System.arraycopy(allMask, from, mask, 0, length);

        final float[][] logits = runModel(ids, mask, config.id2label.size());

        String openType = null;
        int openStart = 0;
        int openEnd = 0;
        double openScore = 0.0;
        int openCount = 0;

        for (int t = 0; t < length; t++) {

            final int absolute = from + t;
            if (special[absolute] == 1L) {
                continue;
            }
            final var charSpan = charSpans == null ? null : charSpans[absolute];
            if (charSpan == null || charSpan.getStart() < 0) {
                continue;
            }

            final float[] row = logits[t];
            int best = 0;
            for (int c = 1; c < row.length; c++) {
                if (row[c] > row[best]) {
                    best = c;
                }
            }
            final double probability = softmaxAt(row, best);
            final String label = config.id2label.get(best);
            final boolean beginning = label.startsWith("B-");
            final String type = "O".equals(label) ? "O" : label.substring(2);

            final boolean continues = type.equals(openType) && !beginning;
            if (openType != null && !continues) {
                emit(openType, openStart, openEnd, openScore / openCount, offset, out);
                openType = null;
            }

            if (openType == null) {
                openType = type;
                openStart = offsets.codeUnit(charSpan.getStart());
                openScore = 0.0;
                openCount = 0;
            }
            openEnd = offsets.codeUnit(charSpan.getEnd());
            openScore += probability;
            openCount++;

        }

        if (openType != null) {
            emit(openType, openStart, openEnd, openScore / openCount, offset, out);
        }

    }

    /** Keep an entity if it is not {@code O} and clears the threshold configured for its label. */
    private void emit(final String type, final int start, final int end, final double score,
                      final int offset, final List<Entity> out) {
        if ("O".equals(type) || end <= start) {
            return;
        }
        // The reference pipeline keeps a span whose score equals the threshold, so this is >= and not
        // >. The GLiNER path uses > because that is what upstream GLiNER does. Neither is "right";
        // they differ, and a calibrated value has to be applied by the same comparison it was chosen
        // with or the boundary case flips.
        if (score >= options.thresholdFor(type)) {
            out.add(new Entity(type, start + offset, end + offset, score));
        }
    }

    /**
     * {@code session.run}, with the encoder's own rejection turned into an actionable message. See
     * the equivalent method on {@link LocalPhEyeDetector} for why this module does not attempt to
     * declare a universal sub-token capacity and instead adds context to the encoder's own failure.
     *
     * <p>On this path the situation is narrower: {@link TokenClassifierConfig#maxTokens} already
     * bounds every call here to what the caller declared as the encoder's capacity, so a rejection
     * below that value means the declared capacity itself was wrong, not that a window ran long.
     */
    private OrtSession.Result runSession(final int seqLen, final Map<String, OnnxTensor> inputs) throws Exception {
        try {
            return session.run(inputs);
        } catch (final OrtException e) {
            throw rejectionMessage(seqLen, config.maxTokens, e);
        }
    }

    /** Package-private and static so the wording is unit-testable without a model. */
    static IllegalStateException rejectionMessage(final int seqLen, final int maxTokens, final OrtException cause) {
        return new IllegalStateException("ONNX Runtime rejected a window of " + seqLen + " sub-tokens,"
                + " within the max_tokens capacity (" + maxTokens + ") declared in "
                + TokenClassifierConfig.WINDOW_FILE + ". That declared value is likely too high for this"
                + " encoder; lower it and retry.", cause);
    }

    private static double softmaxAt(final float[] row, final int index) {
        float max = row[0];
        for (int i = 1; i < row.length; i++) {
            max = Math.max(max, row[i]);
        }
        double total = 0.0;
        for (final float value : row) {
            total += Math.exp(value - max);
        }
        return Math.exp(row[index] - max) / total;
    }

    /** Run the model and return logits shaped [sequence][classes]. */
    private float[][] runModel(final long[] ids, final long[] mask, final int classes) throws Exception {

        final Map<String, OnnxTensor> inputs = new HashMap<>();
        try {

            inputs.put("input_ids", OnnxTensor.createTensor(ortEnvironment, new long[][]{ids}));
            inputs.put("attention_mask", OnnxTensor.createTensor(ortEnvironment, new long[][]{mask}));

            try (final OrtSession.Result result = runSession(ids.length, inputs)) {

                final OnnxValue value = result.get(REQUIRED_OUTPUT).orElseThrow(
                        () -> new IllegalStateException("ONNX model did not return a 'logits' output."));

                final FloatBuffer buffer = ((OnnxTensor) value).getFloatBuffer();
                final float[] flat = new float[buffer.remaining()];
                buffer.get(flat);

                if (flat.length != ids.length * classes) {
                    throw new IllegalStateException("Unexpected logits length " + flat.length + " for "
                            + ids.length + " tokens and " + classes + " classes; the model's output shape"
                            + " is not [batch, sequence, classes].");
                }

                final float[][] logits = new float[ids.length][classes];
                int index = 0;
                for (int t = 0; t < ids.length; t++) {
                    for (int c = 0; c < classes; c++) {
                        logits[t][c] = flat[index++];
                    }
                }
                return logits;

            }

        } finally {
            for (final OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        }

    }

    /**
     * Reduce the entities of every window to one non-overlapping set, in the reference pipeline's
     * order. Package-private and static so it can be tested without a model.
     *
     * <ol>
     *   <li><b>Greedy by score, then by length.</b> Windows overlap, so the same entity is usually
     *       found twice; the copies collide here and only one survives. Where two windows disagree
     *       on an entity's boundaries, the more confident one wins, and length breaks the tie.</li>
     *   <li><b>Trim surrounding whitespace.</b> This tokenizer's offsets include the space before a
     *       word, so a span would otherwise start one character early.</li>
     *   <li><b>Widen to word boundaries.</b> A sub-word model can label part of a word -- "No" of
     *       "Novara" -- and replacing only that part leaves "[CITY_1]vara", which is still readable.
     *       Masking one character too many is the only acceptable error here.</li>
     *   <li><b>Coalesce.</b> Widening can make two spans touch or overlap; left alone, the same word
     *       would be masked twice.</li>
     * </ol>
     */
    static List<Entity> reduce(final List<Entity> entities, final String text) {

        // Stable sort: entities arrive in window order, and ties resolve to the earlier window.
        final List<Entity> ordered = new ArrayList<>(entities);
        ordered.sort((a, b) -> {
            final int byScore = Double.compare(b.score, a.score);
            return byScore != 0 ? byScore : Integer.compare(b.end - b.start, a.end - a.start);
        });

        final List<Entity> kept = new ArrayList<>();
        for (final Entity candidate : ordered) {
            boolean overlaps = false;
            for (final Entity k : kept) {
                if (candidate.start < k.end && k.start < candidate.end) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                kept.add(candidate);
            }
        }

        final List<Entity> trimmed = new ArrayList<>(kept.size());
        for (final Entity entity : kept) {
            int start = entity.start;
            int end = entity.end;
            while (start < end && TextOffsets.isSpace(text.charAt(start))) {
                start++;
            }
            while (end > start && TextOffsets.isSpace(text.charAt(end - 1))) {
                end--;
            }
            if (end <= start) {
                continue;
            }
            while (start > 0 && isWordCharacterBefore(text, start) && isWordCharacterAt(text, start)) {
                start -= Character.charCount(text.codePointBefore(start));
            }
            while (end < text.length() && isWordCharacterAt(text, end) && isWordCharacterBefore(text, end)) {
                end += Character.charCount(text.codePointAt(end));
            }
            trimmed.add(new Entity(entity.label, start, end, entity.score));
        }

        trimmed.sort((a, b) -> {
            final int byStart = Integer.compare(a.start, b.start);
            return byStart != 0 ? byStart : Integer.compare(b.end - b.start, a.end - a.start);
        });

        final List<Entity> merged = new ArrayList<>(trimmed.size());
        for (final Entity entity : trimmed) {
            if (!merged.isEmpty()) {
                final Entity last = merged.get(merged.size() - 1);
                if (entity.start < last.end) {
                    merged.set(merged.size() - 1,
                            new Entity(last.label, last.start, Math.max(last.end, entity.end), last.score));
                    continue;
                }
                if (entity.start == last.end && entity.label.equals(last.label)) {
                    merged.set(merged.size() - 1, new Entity(last.label, last.start, entity.end, last.score));
                    continue;
                }
            }
            merged.add(entity);
        }

        return merged;

    }

    /**
     * The reference {@code _is_word}, at a position rather than on a {@code char}: a character that
     * belongs inside a word. Taken by code point, so a supplementary-plane letter does not stop the
     * widening half way through its surrogate pair.
     */
    private static boolean isWordCharacterAt(final String text, final int index) {
        final int codePoint = text.codePointAt(index);
        return TextOffsets.isAlphanumeric(codePoint) || codePoint == '_';
    }

    /** As {@link #isWordCharacterAt}, for the code point ending at {@code index}. */
    private static boolean isWordCharacterBefore(final String text, final int index) {
        final int codePoint = text.codePointBefore(index);
        return TextOffsets.isAlphanumeric(codePoint) || codePoint == '_';
    }

    /** The entity types this model can emit. Nothing outside this set can be requested. */
    public List<String> entityTypes() {
        return config.entityTypes;
    }

    /** The effective options, after defaults and environment resolution. */
    public LocalPhEyeOptions options() {
        return options;
    }

    /** Words per inference window. */
    public int maxWords() {
        return config.maxWords;
    }

    /** Words shared between consecutive windows. */
    public int chunkOverlapWords() {
        return overlapWords;
    }

    /** The sub-token ceiling for one forward pass. */
    public int maxTokens() {
        return config.maxTokens;
    }

    @Override
    public void close() throws Exception {
        // Both handles are native. Closing the second only if the first succeeds would leak it
        // exactly when something has already gone wrong.
        try {
            if (session != null) {
                session.close();
            }
        } finally {
            if (tokenizer != null) {
                tokenizer.close();
            }
        }
    }

    /** A scored entity over character offsets in the original text. Package-private for testing. */
    record Entity(String label, int start, int end, double score) {

        @Override
        public String toString() {
            return label + "[" + start + "," + end + ")@" + String.format(Locale.ROOT, "%.4f", score);
        }

    }

}
