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

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Loads the fast tokenizer with truncation switched off.
 *
 * <p>This is not a preference. {@code HuggingFaceTokenizer.newInstance(path)} enables truncation at
 * 512 sub-tokens by default, and truncation there is silent: the encoding simply ends, with no flag
 * and no exception. Whatever the caller passed beyond that point is never encoded, never scored, and
 * never reported -- for a redaction component the same data leak that
 * {@link LocalPhEyeOptions.LongTextMode} exists to prevent one level up, except invisible, because
 * windowing by words says nothing about how many sub-tokens those words become.
 *
 * <p>An input that is genuinely too long for the graph now reaches ONNX Runtime and fails loudly
 * there, or is windowed by the detector before it gets that far. Both are better than a quietly
 * shortened document.
 */
final class Tokenizers {

    private Tokenizers() {
    }

    static HuggingFaceTokenizer load(final Path tokenizerJson) throws IOException {
        return HuggingFaceTokenizer.builder()
                .optTokenizerPath(tokenizerJson)
                .optTruncation(false)
                .optPadding(false)
                .build();
    }

}
