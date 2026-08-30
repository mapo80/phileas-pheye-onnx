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

import ai.philterd.phileas.services.filters.ai.pheye.PhEyeDetector;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Opens whichever local detector a model directory describes.
 *
 * <p>The module drives two model families and the directory says which one it holds:
 * {@code gliner_config.json} means a GLiNER span model, {@code token_classification_config.json}
 * plus {@code config.json} mean a BIO token classifier. The two layouts are disjoint, so the choice
 * is a fact about the directory rather than a guess, and a directory that is neither is refused
 * here rather than half-loaded by one of the detectors.
 */
public final class LocalDetectorFactory {

    /** The model families this module can drive. */
    public enum Kind {

        /** Zero-shot GLiNER span model; see {@link LocalPhEyeDetector}. */
        GLINER,

        /** Fixed-taxonomy BIO token classifier; see {@link LocalTokenClassifierDetector}. */
        TOKEN_CLASSIFICATION
    }

    private LocalDetectorFactory() {
    }

    /**
     * Identify the model family from the directory's layout.
     *
     * @throws IllegalArgumentException when the directory describes neither, or somehow both.
     */
    public static Kind kindOf(final Path modelDir) {

        if (modelDir == null) {
            throw new IllegalArgumentException("modelDir must not be null.");
        }
        if (!Files.isDirectory(modelDir)) {
            throw new IllegalArgumentException("Model path " + modelDir + " is not a directory.");
        }

        final boolean gliner = Files.isReadable(modelDir.resolve("gliner_config.json"));
        final boolean tokenClassification = TokenClassifierConfig.describes(modelDir);

        if (gliner && tokenClassification) {
            throw new IllegalArgumentException("Model directory " + modelDir + " declares both a GLiNER"
                    + " config and a token-classification config. Exactly one model lives in a"
                    + " directory; which detector to build would be a coin toss.");
        }
        if (gliner) {
            return Kind.GLINER;
        }
        if (tokenClassification) {
            return Kind.TOKEN_CLASSIFICATION;
        }

        throw new IllegalArgumentException("Model directory " + modelDir + " is neither a GLiNER model"
                + " (gliner_config.json) nor a token-classification model ("
                + TokenClassifierConfig.MODEL_FILE + " and " + TokenClassifierConfig.WINDOW_FILE + ").");

    }

    /** Open the detector the directory describes, with library-default options. */
    public static PhEyeDetector open(final Path modelDir) throws Exception {
        return open(modelDir, LocalPhEyeOptions.defaults());
    }

    /**
     * Open the detector the directory describes.
     *
     * <p>When the caller has expressed no threshold of its own -- neither programmatically nor
     * through the system property or environment variable -- and the model directory declares a
     * {@code calibrated_threshold}, that value is used instead of the library default. The library
     * default exists to reproduce upstream GLiNER, and applying it to a model calibrated elsewhere
     * is how a redaction component ends up quietly under- or over-detecting. A threshold the caller
     * did set is never overridden. That substitution lives in the detector's own constructor, so
     * building one directly is not a quieter route to the wrong operating point.
     */
    public static PhEyeDetector open(final Path modelDir, final LocalPhEyeOptions options) throws Exception {

        final LocalPhEyeOptions effective = options == null ? LocalPhEyeOptions.defaults() : options;

        return switch (kindOf(modelDir)) {
            case GLINER -> new LocalPhEyeDetector(modelDir, effective);
            case TOKEN_CLASSIFICATION -> new LocalTokenClassifierDetector(modelDir, effective);
        };

    }

}
