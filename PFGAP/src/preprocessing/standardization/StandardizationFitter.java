package preprocessing.standardization;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Fits reusable standardization statistics from an eager numeric dataset.
 *
 * <p>The fitter performs one value traversal. It trusts the configured
 * numeric and missing-value contracts and retains only cheap structural
 * checks. Primitive and boxed NaN values, plus boxed null values, are skipped.
 * {@link OnlineMoments} owns finite-value and accumulator-overflow checks.</p>
 *
 * <p>Supported instance representations are {@code double[]},
 * {@code Double[]}, {@code double[][]}, and {@code Double[][]}.
 * Multivariate arrays are dimension-major.</p>
 */
public final class StandardizationFitter {

    public static final double CONSTANT_SCALE = 1.0;

    private StandardizationFitter() {
        // Utility class.
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                StandardizationScope.PER_DIMENSION,
                VarianceConvention.POPULATION,
                Collections.emptyList()
        );
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationScope scope
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                scope,
                VarianceConvention.POPULATION,
                Collections.emptyList()
        );
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationScope scope,
            VarianceConvention varianceConvention
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                scope,
                varianceConvention,
                Collections.emptyList()
        );
    }

    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames
    ) {
        return fit(
                dataset,
                StandardizationMethod.Z_SCORE,
                scope,
                varianceConvention,
                featureNames
        );
    }

    /**
     * Fits reusable statistics in one pass over accepted numeric values.
     */
    public static StandardizationStats fit(
            ListObjectDataset dataset,
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(method, "StandardizationMethod cannot be null.");
        Objects.requireNonNull(scope, "StandardizationScope cannot be null.");
        Objects.requireNonNull(
                varianceConvention,
                "VarianceConvention cannot be null."
        );

        if (method != StandardizationMethod.Z_SCORE) {
            throw new UnsupportedOperationException(
                    "Standardization method " + method + " is not implemented."
            );
        }

        if (scope != StandardizationScope.GLOBAL
                && scope != StandardizationScope.PER_DIMENSION) {

            throw new UnsupportedOperationException(
                    "Standardization scope " + scope + " is not implemented."
            );
        }

        List<Object> data =
                Objects.requireNonNull(
                        dataset.getData(),
                        "Dataset data cannot be null."
                );

        if (data.isEmpty()) {
            throw new IllegalArgumentException(
                    "Cannot fit standardization statistics from an empty dataset."
            );
        }

        int dimensionCount =
                dimensionCountOf(firstRealizedInstance(data));

        List<String> normalizedFeatureNames =
                validateAndCopyFeatureNames(
                        featureNames,
                        dimensionCount,
                        scope
                );

        OnlineMoments[] moments =
                createAccumulators(scope, dimensionCount);

        for (Object instance : data) {
            accumulateInstance(
                    instance,
                    dimensionCount,
                    scope,
                    moments
            );
        }

        return buildStatistics(
                method,
                scope,
                varianceConvention,
                normalizedFeatureNames,
                moments
        );
    }

    private static Object firstRealizedInstance(
            List<Object> data
    ) {
        for (Object instance : data) {
            if (instance != null) {
                if (instance instanceof LazySeriesRef) {
                    throw new UnsupportedOperationException(
                            "Standardization fitting requires eager realized data."
                    );
                }
                return instance;
            }
        }

        throw new IllegalArgumentException(
                "Training dataset contains no realized instances."
        );
    }

    private static int dimensionCountOf(
            Object instance
    ) {
        if (instance instanceof double[] || instance instanceof Double[]) {
            return 1;
        }

        if (instance instanceof double[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            return matrix.length;
        }

        if (instance instanceof Double[][] matrix) {
            requirePositiveDimensionCount(matrix.length);
            return matrix.length;
        }

        throw unsupportedSeriesType(instance);
    }

    private static OnlineMoments[] createAccumulators(
            StandardizationScope scope,
            int dimensionCount
    ) {
        int groupCount =
                scope == StandardizationScope.GLOBAL
                        ? 1
                        : dimensionCount;

        OnlineMoments[] moments =
                new OnlineMoments[groupCount];

        for (int group = 0; group < groupCount; group++) {
            moments[group] = new OnlineMoments();
        }

        return moments;
    }

    private static void accumulateInstance(
            Object instance,
            int expectedDimensionCount,
            StandardizationScope scope,
            OnlineMoments[] moments
    ) {
        Objects.requireNonNull(instance, "Training instance cannot be null.");

        if (instance instanceof LazySeriesRef) {
            throw new UnsupportedOperationException(
                    "Standardization fitting requires eager realized data."
            );
        }

        if (instance instanceof double[] values) {
            requireUnivariateCompatibility(expectedDimensionCount);
            accumulatePrimitiveDimension(values, moments[0]);
            return;
        }

        if (instance instanceof Double[] values) {
            requireUnivariateCompatibility(expectedDimensionCount);
            accumulateBoxedDimension(values, moments[0]);
            return;
        }

        if (instance instanceof double[][] matrix) {
            requireExpectedDimensionCount(matrix.length, expectedDimensionCount);

            for (int dimension = 0; dimension < matrix.length; dimension++) {
                accumulatePrimitiveDimension(
                        Objects.requireNonNull(
                                matrix[dimension],
                                "Training series contains a null dimension."
                        ),
                        accumulator(scope, moments, dimension)
                );
            }
            return;
        }

        if (instance instanceof Double[][] matrix) {
            requireExpectedDimensionCount(matrix.length, expectedDimensionCount);

            for (int dimension = 0; dimension < matrix.length; dimension++) {
                accumulateBoxedDimension(
                        Objects.requireNonNull(
                                matrix[dimension],
                                "Training series contains a null dimension."
                        ),
                        accumulator(scope, moments, dimension)
                );
            }
            return;
        }

        throw unsupportedSeriesType(instance);
    }

    private static OnlineMoments accumulator(
            StandardizationScope scope,
            OnlineMoments[] moments,
            int dimension
    ) {
        return scope == StandardizationScope.GLOBAL
                ? moments[0]
                : moments[dimension];
    }

    private static void accumulatePrimitiveDimension(
            double[] values,
            OnlineMoments moments
    ) {
        for (double value : values) {
            if (!Double.isNaN(value)) {
                moments.add(value);
            }
        }
    }

    private static void accumulateBoxedDimension(
            Double[] values,
            OnlineMoments moments
    ) {
        for (Double value : values) {
            if (value != null && !Double.isNaN(value)) {
                moments.add(value);
            }
        }
    }

    private static StandardizationStats buildStatistics(
            StandardizationMethod method,
            StandardizationScope scope,
            VarianceConvention varianceConvention,
            List<String> featureNames,
            OnlineMoments[] moments
    ) {
        long[] counts = new long[moments.length];
        double[] centers = new double[moments.length];
        double[] scales = new double[moments.length];

        for (int group = 0; group < moments.length; group++) {
            OnlineMoments accumulator = moments[group];

            if (!accumulator.hasObservations()) {
                throw new IllegalArgumentException(
                        "Standardization statistic group "
                                + group
                                + " contains no observations."
                );
            }

            counts[group] = accumulator.getCount();
            centers[group] = accumulator.getMean();
            scales[group] = fittedScale(accumulator, varianceConvention);
        }

        return new StandardizationStats(
                method,
                scope,
                varianceConvention,
                featureNames,
                counts,
                centers,
                scales
        );
    }

    private static double fittedScale(
            OnlineMoments moments,
            VarianceConvention varianceConvention
    ) {
        if (!moments.canCalculateVariance(varianceConvention)) {
            return CONSTANT_SCALE;
        }

        double standardDeviation =
                moments.getStandardDeviation(varianceConvention);

        return standardDeviation == 0.0
                ? CONSTANT_SCALE
                : standardDeviation;
    }

    private static List<String> validateAndCopyFeatureNames(
            List<String> featureNames,
            int dimensionCount,
            StandardizationScope scope
    ) {
        if (featureNames == null || featureNames.isEmpty()) {
            return Collections.emptyList();
        }

        if (featureNames.size() != dimensionCount) {
            throw new IllegalArgumentException(
                    "Training data contains "
                            + dimensionCount
                            + " dimension(s), but "
                            + featureNames.size()
                            + " feature name(s) were supplied for "
                            + scope
                            + " standardization."
            );
        }

        List<String> copy = new ArrayList<>(featureNames.size());

        for (String featureName : featureNames) {
            if (featureName == null || featureName.isBlank()) {
                throw new IllegalArgumentException(
                        "Standardization feature names cannot be null or blank."
                );
            }
            copy.add(featureName.trim());
        }

        return Collections.unmodifiableList(copy);
    }

    private static void requirePositiveDimensionCount(
            int dimensionCount
    ) {
        if (dimensionCount < 1) {
            throw new IllegalArgumentException(
                    "Numeric series must contain at least one dimension."
            );
        }
    }

    private static void requireUnivariateCompatibility(
            int expectedDimensionCount
    ) {
        requireExpectedDimensionCount(1, expectedDimensionCount);
    }

    private static void requireExpectedDimensionCount(
            int actual,
            int expected
    ) {
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "Training series contains "
                            + actual
                            + " dimensions; expected "
                            + expected
                            + "."
            );
        }
    }

    private static IllegalArgumentException unsupportedSeriesType(
            Object instance
    ) {
        return new IllegalArgumentException(
                "Unsupported standardization training type: "
                        + instance.getClass().getTypeName()
                        + ". Expected double[], Double[], double[][], or "
                        + "Double[][]."
        );
    }
}