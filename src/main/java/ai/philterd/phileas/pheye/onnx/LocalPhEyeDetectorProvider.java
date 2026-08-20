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

import ai.philterd.phileas.services.filters.ai.pheye.PhEyeConfiguration;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeDetector;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeDetectorProvider;

import java.nio.file.Path;

/**
 * {@link PhEyeDetectorProvider} that builds a local ONNX Runtime GLiNER detector. Registered
 * via {@code META-INF/services} so core phileas discovers it with {@link java.util.ServiceLoader}
 * when a {@code modelPath} is configured.
 *
 * <p>Tuning that core phileas cannot express -- the ONNX decode threshold and the long-input
 * policy -- is read from system properties or environment variables by
 * {@link LocalPhEyeOptions#fromEnvironment()}:
 * <pre>
 *   -Dphileas.pheye.onnx.detectionThreshold=0.20
 *   -Dphileas.pheye.onnx.longTextMode=CHUNK
 *   -Dphileas.pheye.onnx.chunkOverlapWords=32
 * </pre>
 */
public class LocalPhEyeDetectorProvider implements PhEyeDetectorProvider {

    @Override
    public PhEyeDetector create(final PhEyeConfiguration configuration) throws Exception {

        final String modelPath = configuration.getModelPath();

        if (modelPath == null || modelPath.isBlank()) {
            throw new IllegalArgumentException("PhEyeConfiguration.modelPath must be set to build a local detector.");
        }

        // Core phileas owns PhEyeConfiguration and it has no field for the decode threshold or the
        // long-input policy, so those are resolved from system properties / environment variables.
        // See LocalPhEyeOptions. Defaults reproduce upstream behaviour for the threshold.
        return new LocalPhEyeDetector(Path.of(modelPath), LocalPhEyeOptions.fromEnvironment());

    }

}
