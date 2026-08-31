/*
 *     Copyright 2026 Philterd, LLC @ https://www.philterd.ai
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
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.ServiceLoader;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Unit tests for {@link LocalPhEyeDetectorProvider}, the SPI entry point core phileas discovers via
 * {@link ServiceLoader}. Covers the modelPath validation and that the provider is actually
 * registered, since a broken {@code META-INF/services} file would silently disable local inference.
 */
class LocalPhEyeDetectorProviderTest {

    @Test
    void rejectsNullModelPath() {
        final PhEyeConfiguration configuration = new PhEyeConfiguration("http://localhost:18080");
        // modelPath is unset (null) by default.
        assertThrows(IllegalArgumentException.class,
                () -> new LocalPhEyeDetectorProvider().create(configuration));
    }

    @Test
    void rejectsBlankModelPath() {
        final PhEyeConfiguration configuration = new PhEyeConfiguration("http://localhost:18080");
        configuration.setModelPath("   ");
        assertThrows(IllegalArgumentException.class,
                () -> new LocalPhEyeDetectorProvider().create(configuration));
    }

    @Test
    void isDiscoverableViaServiceLoader() {
        // Core phileas only finds the local detector if this provider is registered for ServiceLoader.
        final boolean registered = ServiceLoader.load(PhEyeDetectorProvider.class).stream()
                .anyMatch(p -> p.type().equals(LocalPhEyeDetectorProvider.class));
        assertTrue(registered, "LocalPhEyeDetectorProvider must be registered in META-INF/services");
    }

    @Test
    void buildsADetectorForAValidModelPath() throws Exception {
        final Path dir = FixtureModel.dir();
        assumeTrue(dir != null, "GLiNER fixture not on the classpath; skipping detector-build test.");

        final PhEyeConfiguration configuration = new PhEyeConfiguration("http://localhost:18080");
        configuration.setModelPath(dir.toString());

        try (final PhEyeDetector detector = new LocalPhEyeDetectorProvider().create(configuration)) {
            assertNotNull(detector);
            assertTrue(detector instanceof LocalPhEyeDetector, detector.getClass().getName());
        }
    }

    @Test
    void buildsATokenClassifierWhenThatIsWhatTheDirectoryHolds() throws Exception {

        // The SPI is the only entry point core phileas uses, so the dispatch has to work through it
        // and not just through LocalDetectorFactory: a provider that always built the GLiNER
        // detector would fail on the ONNX signature check at startup, on this model.
        final String dir = System.getenv("PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR");
        assumeTrue(dir != null && !dir.isBlank(),
                "Set PHILEAS_TOKEN_CLASSIFIER_MODEL_DIR to a token-classification model directory.");

        final PhEyeConfiguration configuration = new PhEyeConfiguration("http://localhost:18080");
        configuration.setModelPath(dir);

        try (final PhEyeDetector detector = new LocalPhEyeDetectorProvider().create(configuration)) {
            assertTrue(detector instanceof LocalTokenClassifierDetector, detector.getClass().getName());
            final var spans = detector.detect(
                    "Il sig. Mario Rossi, residente a Roma, ha firmato il contratto.", List.of("FULLNAME"), "", 0);
            assertEquals(1, spans.size(), spans.toString());
            assertEquals("Mario Rossi", spans.get(0).getText());
        }
    }

}
