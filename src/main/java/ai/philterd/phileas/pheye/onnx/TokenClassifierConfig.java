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

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * What {@link LocalTokenClassifierDetector} needs to drive a BIO token-classification model.
 *
 * <p>It comes from two files, because the two answer different questions and only one of them is
 * written by the training framework:
 *
 * <ul>
 *   <li>{@code config.json} is the HuggingFace model config. {@code id2label} is read from it and
 *       is authoritative: it is the taxonomy the weights were trained on, in BIO notation.</li>
 *   <li>{@code token_classification_config.json} declares the <b>inference window</b>: how many
 *       words go into one forward pass, how many are shared with the next, and the sub-token
 *       ceiling the weights were trained at. None of this is in {@code config.json}: an encoder
 *       advertises its <i>positional</i> limit there ({@code max_position_embeddings}, 8192 for
 *       ModernBERT), which is not the length it was trained to label. Guessing that number wrong
 *       degrades a redaction model quietly, so this file is required rather than defaulted.</li>
 * </ul>
 *
 * <p>The file may also declare {@code calibrated_threshold}: the decode threshold this model was
 * calibrated at on a validation split. It is a property of the model, not of the library, which is
 * why it lives in the model directory. See {@link LocalDetectorFactory} for when it is applied.
 */
public final class TokenClassifierConfig {

    /** The file that declares the inference window. */
    public static final String WINDOW_FILE = "token_classification_config.json";

    /** The HuggingFace model config. */
    public static final String MODEL_FILE = "config.json";

    /** Label per class index, in BIO notation ({@code B-CITY}, {@code I-CITY}, {@code O}). */
    public final List<String> id2label;

    /** The entity types, i.e. the BIO suffixes without {@code O}, in taxonomy order. */
    public final List<String> entityTypes;

    /** Words per inference window. */
    public final int maxWords;

    /** Words shared between consecutive windows. */
    public final int overlapWords;

    /**
     * The encoder's hard sub-token capacity: the point past which the graph cannot run at all.
     *
     * <p>Not the length the weights were distilled at, and not a quality knob. A word window that
     * encodes to more sub-tokens than this is split; splitting any earlier than that would change
     * what the model sees relative to the reference pipeline, and so move the operating point the
     * threshold was calibrated on.
     */
    public final int maxTokens;

    /** Word-splitter strategy. Only {@code whitespace} is supported. */
    public final String wordsSplitterType;

    /** The threshold this model was calibrated at, or {@code null} when the directory declares none. */
    public final Double calibratedThreshold;

    private TokenClassifierConfig(final List<String> id2label, final List<String> entityTypes,
                                  final int maxWords, final int overlapWords, final int maxTokens,
                                  final String wordsSplitterType, final Double calibratedThreshold) {
        this.id2label = List.copyOf(id2label);
        this.entityTypes = List.copyOf(entityTypes);
        this.maxWords = maxWords;
        this.overlapWords = overlapWords;
        this.maxTokens = maxTokens;
        this.wordsSplitterType = wordsSplitterType;
        this.calibratedThreshold = calibratedThreshold;
    }

    /** True when the directory carries the files this detector reads. */
    public static boolean describes(final Path modelDir) {
        return Files.isReadable(modelDir.resolve(WINDOW_FILE)) && Files.isReadable(modelDir.resolve(MODEL_FILE));
    }

    /**
     * Load and validate both files.
     *
     * @throws IllegalArgumentException on anything that would make the detector label text wrongly:
     *                                  a missing file, an absent or non-BIO {@code id2label}, a
     *                                  non-positive or inconsistent window, an unsupported splitter.
     */
    public static TokenClassifierConfig load(final Path modelDir) throws IOException {

        final JsonObject model = read(modelDir.resolve(MODEL_FILE), MODEL_FILE);
        final JsonObject window = read(modelDir.resolve(WINDOW_FILE), WINDOW_FILE);

        if (!model.has("id2label") || !model.get("id2label").isJsonObject()) {
            throw new IllegalArgumentException(MODEL_FILE + " in " + modelDir + " declares no id2label."
                    + " A token-classification model directory must carry the label taxonomy the weights"
                    + " were trained on.");
        }

        // A JSON object has no order, and the keys are the class indices, so sort numerically rather
        // than trusting the file's key order: a label read at the wrong index mislabels everything.
        final Map<Integer, String> byIndex = new TreeMap<>();
        for (final Map.Entry<String, JsonElement> entry : model.getAsJsonObject("id2label").entrySet()) {
            final int index;
            try {
                index = Integer.parseInt(entry.getKey().trim());
            } catch (final NumberFormatException e) {
                throw new IllegalArgumentException(MODEL_FILE + " in " + modelDir + " has a non-numeric"
                        + " id2label key '" + entry.getKey() + "'.", e);
            }
            byIndex.put(index, entry.getValue().getAsString());
        }

        final List<String> labels = new ArrayList<>(byIndex.size());
        int expected = 0;
        for (final Map.Entry<Integer, String> entry : byIndex.entrySet()) {
            if (entry.getKey() != expected) {
                throw new IllegalArgumentException(MODEL_FILE + " in " + modelDir + " has a gap in id2label:"
                        + " expected index " + expected + ", found " + entry.getKey() + ".");
            }
            labels.add(entry.getValue());
            expected++;
        }

        final Set<String> types = new LinkedHashSet<>();
        for (final String label : labels) {
            if ("O".equals(label)) {
                continue;
            }
            if (label.length() < 3 || (!label.startsWith("B-") && !label.startsWith("I-"))) {
                throw new IllegalArgumentException(MODEL_FILE + " in " + modelDir + " declares label '"
                        + label + "', which is neither 'O' nor BIO-prefixed. This detector decodes BIO"
                        + " tagging; another scheme would produce wrong spans rather than no spans.");
            }
            types.add(label.substring(2));
        }
        if (types.isEmpty()) {
            throw new IllegalArgumentException(MODEL_FILE + " in " + modelDir + " declares no entity"
                    + " labels beyond 'O'.");
        }

        final int maxWords = required(window, "max_words", modelDir);
        final int overlapWords = required(window, "overlap_words", modelDir);
        final int maxTokens = required(window, "max_tokens", modelDir);

        if (maxWords <= 0) {
            throw new IllegalArgumentException(WINDOW_FILE + " in " + modelDir + " declares a non-positive"
                    + " max_words: " + maxWords + ".");
        }
        if (maxTokens <= 0) {
            throw new IllegalArgumentException(WINDOW_FILE + " in " + modelDir + " declares a non-positive"
                    + " max_tokens: " + maxTokens + ".");
        }
        if (overlapWords < 0 || overlapWords >= maxWords) {
            // An overlap at least as wide as the window makes the stride zero: windowing would never
            // advance and a long document would loop forever.
            throw new IllegalArgumentException(WINDOW_FILE + " in " + modelDir + " declares overlap_words "
                    + overlapWords + ", which must be within [0, max_words) = [0, " + maxWords + ").");
        }

        final String splitter = window.has("words_splitter_type")
                ? window.get("words_splitter_type").getAsString() : "whitespace";
        if (!"whitespace".equals(splitter)) {
            throw new IllegalArgumentException("Unsupported words_splitter_type '" + splitter + "' in "
                    + WINDOW_FILE + ". Only 'whitespace' is supported by this detector.");
        }

        Double calibrated = null;
        if (window.has("calibrated_threshold") && !window.get("calibrated_threshold").isJsonNull()) {
            calibrated = window.get("calibrated_threshold").getAsDouble();
            if (calibrated < 0.0 || calibrated > 1.0 || Double.isNaN(calibrated)) {
                throw new IllegalArgumentException(WINDOW_FILE + " in " + modelDir + " declares"
                        + " calibrated_threshold " + calibrated + ", which must be within [0, 1].");
            }
        }

        return new TokenClassifierConfig(labels, new ArrayList<>(types), maxWords, overlapWords,
                maxTokens, splitter, calibrated);

    }

    /** The taxonomy label matching a caller-supplied string, case-insensitively, or {@code null}. */
    public String resolveEntityType(final String requested) {
        if (requested == null) {
            return null;
        }
        final String needle = requested.trim().toLowerCase(Locale.ROOT);
        for (final String type : entityTypes) {
            if (type.toLowerCase(Locale.ROOT).equals(needle)) {
                return type;
            }
        }
        return null;
    }

    private static JsonObject read(final Path file, final String what) throws IOException {
        if (!Files.isReadable(file)) {
            throw new IllegalArgumentException("Missing or unreadable " + what + " at " + file
                    + ". A token-classification model directory must contain " + MODEL_FILE + ", "
                    + WINDOW_FILE + ", tokenizer.json and onnx/model.onnx (or model.onnx).");
        }
        try (final Reader reader = Files.newBufferedReader(file)) {
            final JsonObject json = new Gson().fromJson(reader, JsonObject.class);
            if (json == null) {
                throw new IllegalArgumentException(what + " at " + file + " is empty or not a JSON object.");
            }
            return json;
        } catch (final com.google.gson.JsonParseException e) {
            throw new IllegalArgumentException(what + " at " + file + " is not valid JSON.", e);
        }
    }

    private static int required(final JsonObject json, final String key, final Path modelDir) {
        if (!json.has(key) || json.get(key).isJsonNull()) {
            throw new IllegalArgumentException(WINDOW_FILE + " in " + modelDir + " does not declare '"
                    + key + "'. The inference window is model-specific and is not defaulted here:"
                    + " a wrong window degrades detection silently.");
        }
        return json.get(key).getAsInt();
    }

}
