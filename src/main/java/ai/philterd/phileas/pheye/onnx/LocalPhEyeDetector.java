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
import java.util.Map;
import java.util.Set;

/**
 * On-device GLiNER inference via ONNX Runtime. This is a Java port of the GLiNER 0.2.25
 * uni-encoder span pipeline (the recipe PhEye uses): build an entity prompt, tokenize prompt+text,
 * enumerate candidate word spans, run the ONNX model, then sigmoid + threshold + greedy
 * (flat, non-overlapping) decode, mapping word spans back to character offsets.
 *
 * <p><b>Model directory layout</b>:
 * <ul>
 *   <li>{@code model.onnx} (or {@code onnx/model.onnx}) — the exported GLiNER model</li>
 *   <li>{@code tokenizer.json} — the HuggingFace fast tokenizer</li>
 *   <li>{@code gliner_config.json} — span width, max length, prompt tokens</li>
 * </ul>
 *
 * <p><b>Differences from upstream</b>, both driven by using this as a redaction component:
 * <ol>
 *   <li>The decode threshold is configurable through {@link LocalPhEyeOptions} instead of being
 *       hardcoded at 0.5. GLiNER models calibrate very differently from one another, and a floor
 *       that is too high silently discards correct detections. The default is still 0.5.</li>
 *   <li>Input longer than the model's {@code max_len} is processed in overlapping windows rather
 *       than truncated. Truncation means the tail of a document is never examined, which for a
 *       privacy filter is a data leak rather than a quality issue. See
 *       {@link LocalPhEyeOptions.LongTextMode}.</li>
 * </ol>
 *
 * <p><b>Fail-closed:</b> a missing file, an unreadable config, or an ONNX graph whose signature is
 * not the expected GLiNER one is a constructor failure. The detector never degrades to examining
 * part of the input, and never falls back to a remote service.
 *
 * <p><b>Parity:</b> a redaction model that decodes spans incorrectly leaks names, so this class is
 * parity-tested. {@code LocalPhEyeDetectorParityTest} verifies the pipeline mechanics against a
 * synthetic ONNX fixture, {@code LocalPhEyeDetectorRealModelParityTest} confirms its spans match
 * Python {@code gliner.predict_entities} on a real exported model, and
 * {@code LocalPhEyeDetectorThresholdTest} pins that at threshold 0.5 the output is identical to the
 * upstream hardcoded behaviour.
 */
public class LocalPhEyeDetector implements PhEyeDetector {

    /**
     * The threshold upstream hardcodes. Retained as the default so an unconfigured detector
     * behaves exactly as before.
     *
     * @deprecated prefer {@link LocalPhEyeOptions#DEFAULT_DETECTION_THRESHOLD}.
     */
    @Deprecated(since = "1.2.0")
    public static final double DEFAULT_THRESHOLD = LocalPhEyeOptions.DEFAULT_DETECTION_THRESHOLD;

    /** The tensors a GLiNER span export must accept. Anything else is not a model we can drive. */
    static final List<String> REQUIRED_INPUTS = List.of(
            "input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask");

    /** The output the decode reads. */
    static final String REQUIRED_OUTPUT = "logits";

    private final GlinerConfig config;
    private final HuggingFaceTokenizer tokenizer;
    private final OrtEnvironment ortEnvironment;
    private final OrtSession session;
    private final LocalPhEyeOptions options;
    private final int chunkOverlapWords;

    public LocalPhEyeDetector(final Path modelDir) throws Exception {
        this(modelDir, LocalPhEyeOptions.defaults());
    }

    /** Convenience for the common case of only changing the decode threshold. */
    public LocalPhEyeDetector(final Path modelDir, final double detectionThreshold) throws Exception {
        this(modelDir, LocalPhEyeOptions.withThreshold(detectionThreshold));
    }

    public LocalPhEyeDetector(final Path modelDir, final LocalPhEyeOptions options) throws Exception {

        this.options = options == null ? LocalPhEyeOptions.defaults() : options;

        // The config is read and validated first, so a malformed model directory fails on the
        // clearest possible error rather than on whichever file happens to be checked first.
        requireReadable(modelDir.resolve("gliner_config.json"), "gliner_config.json");

        this.config = GlinerConfig.load(modelDir);

        if (!"whitespace".equals(config.wordsSplitterType)) {
            throw new IllegalArgumentException("Unsupported words_splitter_type '" + config.wordsSplitterType
                    + "'. Only 'whitespace' is supported by this detector.");
        }
        if (config.maxLen <= 0) {
            throw new IllegalArgumentException("gliner_config.json declares a non-positive max_len: " + config.maxLen);
        }
        if (config.maxWidth <= 0) {
            throw new IllegalArgumentException("gliner_config.json declares a non-positive max_width: " + config.maxWidth);
        }

        // Overlap must be at least maxWidth - 1 words, otherwise a span that straddles a window
        // boundary is never wholly inside any window and would be missed entirely.
        final int requestedOverlap = this.options.chunkOverlapWords() != null
                ? this.options.chunkOverlapWords()
                : Math.min(Math.max(2 * config.maxWidth, 32), config.maxLen / 4);
        final int minimumOverlap = Math.max(config.maxWidth - 1, 0);
        final int maximumOverlap = config.maxLen - 1;
        this.chunkOverlapWords = Math.min(Math.max(requestedOverlap, minimumOverlap), Math.max(maximumOverlap, 0));

        requireReadable(modelDir.resolve("tokenizer.json"), "tokenizer.json");
        this.tokenizer = HuggingFaceTokenizer.newInstance(modelDir.resolve("tokenizer.json"));

        final Path onnx = resolveOnnxPath(modelDir);
        requireReadable(onnx, "the ONNX model");

        this.ortEnvironment = OrtEnvironment.getEnvironment();
        this.session = ortEnvironment.createSession(onnx.toString(), new OrtSession.SessionOptions());

        validateSignature(onnx);

    }

    private static void requireReadable(final Path path, final String what) {
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Missing " + what + " at " + path
                    + ". The model directory must contain gliner_config.json, tokenizer.json and"
                    + " onnx/model.onnx (or model.onnx).");
        }
        if (!Files.isReadable(path)) {
            throw new IllegalArgumentException("Cannot read " + what + " at " + path + ".");
        }
    }

    /**
     * Reject a graph that is not a GLiNER span export, rather than discovering it later as an
     * opaque ONNX Runtime error or, worse, as silently empty detections.
     */
    private void validateSignature(final Path onnx) {

        final Set<String> inputs = new LinkedHashSet<>(session.getInputNames());
        final List<String> missing = new ArrayList<>();
        for (final String required : REQUIRED_INPUTS) {
            if (!inputs.contains(required)) {
                missing.add(required);
            }
        }

        if (!missing.isEmpty()) {
            throw new IllegalStateException("The ONNX model at " + onnx + " is not a GLiNER span export:"
                    + " missing input tensors " + missing + ". Present inputs: " + inputs
                    + ". A GLiNER span model must accept " + REQUIRED_INPUTS + ".");
        }

        if (!session.getOutputNames().contains(REQUIRED_OUTPUT)) {
            throw new IllegalStateException("The ONNX model at " + onnx + " does not expose a '"
                    + REQUIRED_OUTPUT + "' output. Present outputs: " + session.getOutputNames() + ".");
        }

    }

    private static Path resolveOnnxPath(final Path modelDir) {
        final Path nested = modelDir.resolve("onnx").resolve("model.onnx");
        if (Files.exists(nested)) {
            return nested;
        }
        return modelDir.resolve("model.onnx");
    }

    @Override
    public List<PhEyeSpan> detect(final String text, final Collection<String> labels,
                                  final String context, final int piece) throws Exception {

        final List<String> labelList = new ArrayList<>(labels);
        if (labelList.isEmpty() || text == null || text.isBlank()) {
            return new ArrayList<>();
        }

        // 1. Split into words with char offsets. Every Word keeps its absolute offset in the
        //    original text, so windowing never needs to translate offsets afterwards.
        final List<WordsSplitter.Word> words = WordsSplitter.split(text);
        final int totalWords = words.size();
        if (totalWords == 0) {
            return new ArrayList<>();
        }

        // 2. Decide how much of the input to look at.
        final List<int[]> windows = planWindows(totalWords);

        // 3. Score every window, accumulating candidates in GLOBAL word coordinates.
        final List<Candidate> candidates = new ArrayList<>();
        for (final int[] window : windows) {
            collectCandidates(words, window[0], window[1], labelList, candidates);
        }

        // 4. Reduce overlapping candidates. Identical spans found in two overlapping windows collide
        //    here and only the highest-scoring copy survives, so this also deduplicates.
        final List<Candidate> selected = decode(candidates, options.decodeStrategy(), labelList.size());

        // 5. Map word spans -> char offsets -> PhEyeSpan.
        final List<PhEyeSpan> spans = new ArrayList<>();
        for (final Candidate cand : selected) {
            final int startChar = words.get(cand.startWord).start();
            final int endChar = words.get(cand.endWord).end();

            final PhEyeSpan span = new PhEyeSpan();
            span.setStart(startChar);
            span.setEnd(endChar);
            span.setLabel(labelList.get(cand.classIndex));
            span.setText(text.substring(startChar, endChar));
            span.setScore(cand.score);
            spans.add(span);
        }

        return spans;

    }

    /**
     * Work out which word ranges to score, as half-open {@code [from, to)} pairs.
     *
     * @throws IllegalArgumentException in {@code FAIL} mode when the input is too long: refusing the
     *                                 document is safer than examining part of it.
     */
    private List<int[]> planWindows(final int totalWords) {
        return planWindows(totalWords, config.maxLen, chunkOverlapWords, options.longTextMode());
    }

    /**
     * Pure windowing logic, package-private and static so it can be unit-tested without loading a
     * model. Guarantees, verified in {@code LongInputWindowingTest}: every word index appears in at
     * least one window, and consecutive windows share {@code overlapWords} words.
     */
    static List<int[]> planWindows(final int totalWords, final int maxLen, final int overlapWords,
                                   final LocalPhEyeOptions.LongTextMode mode) {

        final List<int[]> windows = new ArrayList<>();

        if (totalWords <= maxLen) {
            windows.add(new int[]{0, totalWords});
            return windows;
        }

        switch (mode) {

            case FAIL -> throw new IllegalArgumentException("Input has " + totalWords + " words, more than the"
                    + " model's max_len of " + maxLen + ", and longTextMode is FAIL. Chunk the input"
                    + " upstream or switch to CHUNK mode; examining only part of the text would leave the"
                    + " remainder unredacted.");

            case TRUNCATE -> windows.add(new int[]{0, maxLen});

            case CHUNK -> {
                final int stride = Math.max(maxLen - overlapWords, 1);
                for (int from = 0; from < totalWords; from += stride) {
                    final int to = Math.min(from + maxLen, totalWords);
                    windows.add(new int[]{from, to});
                    if (to == totalWords) {
                        break;
                    }
                }
            }

            default -> throw new IllegalStateException("Unhandled longTextMode " + mode);

        }

        return windows;

    }

    /** Score one word window and append its above-threshold candidates, in global coordinates. */
    private void collectCandidates(final List<WordsSplitter.Word> allWords, final int from, final int to,
                                   final List<String> labelList, final List<Candidate> out) throws Exception {

        final int numWords = to - from;
        if (numWords <= 0) {
            return;
        }

        // Build the prompt word list: [<<ENT>> label]* <<SEP>> <text words...>
        // (mirrors SpanProcessor.prepare_inputs; markerV0 prompt).
        final List<String> inputWords = new ArrayList<>();
        for (final String label : labelList) {
            inputWords.add(config.entToken);
            inputWords.add(label);
        }
        inputWords.add(config.sepToken);
        final int promptWordCount = inputWords.size();
        for (int i = from; i < to; i++) {
            inputWords.add(allWords.get(i).text());
        }

        // Tokenize pre-split (is_split_into_words=True). The tokenizer adds special tokens.
        final Encoding encoding = tokenizer.encode(inputWords.toArray(new String[0]));
        final long[] inputIds = encoding.getIds();
        final long[] attentionMask = encoding.getAttentionMask();
        final long[] wordIds = encoding.getWordIds(); // word index per token; -1 for special tokens
        final int seqLen = inputIds.length;

        // words_mask: first subtoken of each TEXT word gets its 1-based text-word index, else 0
        // (mirrors prepare_word_mask with skip_first_words=promptWordCount).
        final long[] wordsMask = new long[seqLen];
        long previousWordId = Long.MIN_VALUE;
        for (int t = 0; t < seqLen; t++) {
            final long wid = wordIds[t];
            if (wid >= promptWordCount && wid != previousWordId) {
                wordsMask[t] = wid - promptWordCount + 1;
            } else {
                wordsMask[t] = 0;
            }
            previousWordId = wid;
        }

        // Enumerate candidate spans [i, i+width] for width in 0..maxWidth-1; mask out-of-range.
        final int spansPerWord = config.maxWidth;
        final int numSpans = numWords * spansPerWord;
        final long[][] spanIdx = new long[numSpans][2];
        // span_mask is a boolean tensor in the GLiNER ONNX signature (not int64); feeding the
        // wrong dtype makes ONNX Runtime reject the input.
        final boolean[] spanMask = new boolean[numSpans];
        int s = 0;
        for (int i = 0; i < numWords; i++) {
            for (int k = 0; k < spansPerWord; k++) {
                final int end = i + k;
                spanIdx[s][0] = i;
                spanIdx[s][1] = end;
                spanMask[s] = end < numWords;
                s++;
            }
        }

        final float[][][] logits = runModel(inputIds, attentionMask, wordsMask, numWords, spanIdx, spanMask);

        // Resolved once per label rather than per candidate: thresholdFor() lowercases and looks up.
        final double[] thresholds = new double[labelList.size()];
        for (int c = 0; c < labelList.size(); c++) {
            thresholds[c] = options.thresholdFor(labelList.get(c));
        }

        for (int i = 0; i < numWords; i++) {
            for (int k = 0; k < spansPerWord; k++) {
                final int end = i + k;
                if (end >= numWords) {
                    continue;
                }
                for (int c = 0; c < labelList.size(); c++) {
                    final double prob = sigmoid(logits[i][k][c]);
                    if (prob > thresholds[c]) {
                        out.add(new Candidate(from + i, from + end, c, prob));
                    }
                }
            }
        }

    }

    /**
     * Run the span model and return logits shaped [numWords][maxWidth][numClasses].
     *
     * <p>GLiNER's span model emits logits of [batch, numWords, maxWidth, numClasses]; this method
     * reshapes to that.
     */
    private float[][][] runModel(final long[] inputIds, final long[] attentionMask, final long[] wordsMask,
                                 final int numWords, final long[][] spanIdx, final boolean[] spanMask)
            throws Exception {

        final Map<String, OnnxTensor> inputs = new HashMap<>();
        try {

            inputs.put("input_ids", OnnxTensor.createTensor(ortEnvironment, new long[][]{inputIds}));
            inputs.put("attention_mask", OnnxTensor.createTensor(ortEnvironment, new long[][]{attentionMask}));
            inputs.put("words_mask", OnnxTensor.createTensor(ortEnvironment, new long[][]{wordsMask}));
            inputs.put("text_lengths", OnnxTensor.createTensor(ortEnvironment, new long[][]{{numWords}}));
            inputs.put("span_idx", OnnxTensor.createTensor(ortEnvironment, new long[][][]{spanIdx}));
            inputs.put("span_mask", OnnxTensor.createTensor(ortEnvironment, new boolean[][]{spanMask}));

            try (final OrtSession.Result result = session.run(inputs)) {

                final OnnxValue value = result.get(REQUIRED_OUTPUT).orElseThrow(
                        () -> new IllegalStateException("ONNX model did not return a 'logits' output."));

                // Copy out of the buffer rather than calling array(): ONNX Runtime may hand back a
                // direct buffer, which has no backing array.
                final FloatBuffer buffer = ((OnnxTensor) value).getFloatBuffer();
                final float[] flat = new float[buffer.remaining()];
                buffer.get(flat);

                final int cells = numWords * config.maxWidth;
                if (cells == 0 || flat.length % cells != 0) {
                    throw new IllegalStateException("Unexpected logits length " + flat.length
                            + " for " + numWords + " words and max_width " + config.maxWidth
                            + "; the model's output shape is not the expected GLiNER span layout.");
                }

                final int numClasses = flat.length / cells;
                final float[][][] logits = new float[numWords][config.maxWidth][numClasses];
                int idx = 0;
                for (int i = 0; i < numWords; i++) {
                    for (int k = 0; k < config.maxWidth; k++) {
                        for (int c = 0; c < numClasses; c++) {
                            logits[i][k][c] = flat[idx++];
                        }
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
     * Reduce candidates according to the configured strategy.
     *
     * <p>{@code FLAT_GREEDY} is upstream's single pass across all labels. {@code PER_LABEL_GREEDY}
     * partitions by label first, so a high-scoring span of one label can no longer delete a
     * lower-scoring span of another over the same words -- the case where a spurious ORGANIZATION
     * silently removes a correct PERSON and the name goes unmasked.
     */
    static List<Candidate> decode(final List<Candidate> candidates, final LocalPhEyeOptions.DecodeStrategy strategy,
                                  final int numClasses) {

        if (strategy == LocalPhEyeOptions.DecodeStrategy.FLAT_GREEDY) {
            return greedyNonOverlap(candidates);
        }

        if (strategy == LocalPhEyeOptions.DecodeStrategy.CONTAINMENT_AWARE_GREEDY) {
            return containmentAwareGreedy(candidates);
        }

        final List<Candidate> kept = new ArrayList<>();
        for (int c = 0; c < numClasses; c++) {
            final int classIndex = c;
            final List<Candidate> ofClass = new ArrayList<>();
            for (final Candidate candidate : candidates) {
                if (candidate.classIndex == classIndex) {
                    ofClass.add(candidate);
                }
            }
            kept.addAll(greedyNonOverlap(ofClass));
        }

        kept.sort((a, b) -> Integer.compare(a.startWord, b.startWord));
        return kept;

    }

    /**
     * Greedy flat NER: keep highest-scoring spans that do not overlap already-kept ones.
     * Package-private so it can be unit-tested directly (see {@code GreedyNonOverlapTest}).
     */
    static List<Candidate> greedyNonOverlap(final List<Candidate> candidates) {
        candidates.sort((a, b) -> Double.compare(b.score, a.score));
        final List<Candidate> kept = new ArrayList<>();
        for (final Candidate c : candidates) {
            boolean overlaps = false;
            for (final Candidate k : kept) {
                if (c.startWord <= k.endWord && k.startWord <= c.endWord) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                kept.add(c);
            }
        }
        kept.sort((a, b) -> Integer.compare(a.startWord, b.startWord));
        return kept;
    }

    /**
     * Greedy, except that a strictly containing span of the same label displaces the span it
     * contains instead of being discarded by it.
     *
     * <p>Same descending-score pass as {@link #greedyNonOverlap}: the difference is what happens when
     * a candidate overlaps something already kept. Plain greedy always drops the candidate. Here, if
     * the candidate contains a kept span of the same label, and overlaps <i>nothing else</i>, the two
     * swap places. Requiring it to overlap nothing else is what keeps the pass deterministic and
     * order-independent -- a candidate straddling two kept spans has no single span to replace, and
     * promoting it would silently delete the second one.
     *
     * <p>Both spans are above the decode threshold by construction: candidates below it never reach
     * this method.
     */
    static List<Candidate> containmentAwareGreedy(final List<Candidate> candidates) {

        candidates.sort((a, b) -> Double.compare(b.score, a.score));
        final List<Candidate> kept = new ArrayList<>();

        for (final Candidate c : candidates) {

            Candidate contained = null;
            boolean blocked = false;

            for (final Candidate k : kept) {
                if (c.startWord > k.endWord || k.startWord > c.endWord) {
                    continue;
                }
                final boolean strictlyContains = c.startWord <= k.startWord && k.endWord <= c.endWord
                        && (c.startWord < k.startWord || k.endWord < c.endWord);
                if (strictlyContains && c.classIndex == k.classIndex && contained == null) {
                    contained = k;
                } else {
                    blocked = true;
                    break;
                }
            }

            if (blocked) {
                continue;
            }
            if (contained != null) {
                kept.remove(contained);
            }
            kept.add(c);

        }

        kept.sort((a, b) -> Integer.compare(a.startWord, b.startWord));
        return kept;

    }

    private static double sigmoid(final double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /** The effective options, after defaults and environment resolution. */
    public LocalPhEyeOptions options() {
        return options;
    }

    /** Words shared between consecutive windows in CHUNK mode. */
    public int chunkOverlapWords() {
        return chunkOverlapWords;
    }

    /** The model's word limit, from {@code gliner_config.json}. */
    public int maxWords() {
        return config.maxLen;
    }

    @Override
    public void close() throws Exception {
        if (session != null) {
            session.close();
        }
        if (tokenizer != null) {
            tokenizer.close();
        }
    }

    /** A scored candidate span over word indices (inclusive) for a given class. Package-private for testing. */
    record Candidate(int startWord, int endWord, int classIndex, double score) {}

}
