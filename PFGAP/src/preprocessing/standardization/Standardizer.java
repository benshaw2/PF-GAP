package preprocessing.standardization;

import datasets.ListObjectDataset;
import datasets.readers.lazy.LazySeriesRef;

import java.util.List;
import java.util.Objects;

/**
 * Applies prepared standardization statistics to realized numeric data.
 *
 * <p>Supported representations are {@code double[]}, {@code Double[]},
 * {@code double[][]}, and {@code Double[][]}. Multivariate arrays use
 * dimension-major orientation: {@code data[dimension][time]}.</p>
 *
 * <p>Transformation is performed in place and in one pass over the values.
 * This class intentionally does not perform a preliminary full-dataset
 * validation pass. Reader configuration and the configured numeric and
 * missing-value modes are treated as trusted input contracts. Cheap structural
 * checks are retained so configuration errors fail near their source.</p>
 *
 * <p>{@code null} boxed values and {@code NaN} values are preserved. No
 * standardized copy of the dataset is created.</p>
 */
public final class Standardizer {

    private Standardizer() {
        // Utility class.
    }

    /**
     * Standardizes every realized instance in a dataset in place.
     *
     * @param dataset eager numeric dataset
     * @param stats prepared standardization statistics
     * @param featureNames ordered dimension names, or null when unavailable
     * @return the same dataset instance
     */
    public static ListObjectDataset transformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats,
            List<String> featureNames
    ) {
        Objects.requireNonNull(dataset, "Dataset cannot be null.");
        Objects.requireNonNull(stats, "StandardizationStats cannot be null.");

        List<Object> data =
                Objects.requireNonNull(
                        dataset.getData(),
                        "Dataset data cannot be null."
                );

        stats.validateFeatureCompatibility(featureNames);

        for (Object instance : data) {
            transformInstanceInPlace(instance, stats);
        }

        return dataset;
    }

    /**
     * Standardizes every realized instance without named-feature validation.
     *
     * @param dataset eager numeric dataset
     * @param stats prepared standardization statistics
     * @return the same dataset instance
     */
    public static ListObjectDataset transformInPlace(
            ListObjectDataset dataset,
            StandardizationStats stats
    ) {
        Objects.requireNonNull(stats, "StandardizationStats cannot be null.");

        if (stats.hasFeatureNames()) {
            throw new IllegalArgumentException(
                    "Standardization statistics contain ordered feature "
                            + "names. Supply feature names to "
                            + "transformInPlace(dataset, stats, featureNames)."
            );
        }

        return transformInPlace(dataset, stats, null);
    }

    /**
     * Standardizes one realized numeric instance in place.
     *
     * <p>This method performs no preliminary value-validation traversal.
     * Representation and dimension compatibility are checked once, then each
     * value is transformed directly.</p>
     *
     * @param series realized numeric instance
     * @param stats prepared standardization statistics
     * @return the same instance object
     */
    public static Object transformInstanceInPlace(
            Object series,
            StandardizationStats stats
    ) {
        Objects.requireNonNull(series, "Series cannot be null.");
        Objects.requireNonNull(stats, "StandardizationStats cannot be null.");

        if (series instanceof LazySeriesRef) {
            throw new UnsupportedOperationException(
                    "Cannot standardize a LazySeriesRef directly. "
                            + "Standardize the realized series after "
                            + "materialization."
            );
        }

        if (series instanceof double[] values) {
            requireUnivariateStats(stats);
            transformPrimitiveDimension(values, center(stats, 0), scale(stats, 0));
            return values;
        }

        if (series instanceof Double[] values) {
            requireUnivariateStats(stats);
            transformBoxedDimension(values, center(stats, 0), scale(stats, 0));
            return values;
        }

        if (series instanceof double[][] values) {
            requireDimensionCompatibility(values.length, stats);

            for (int dimension = 0; dimension < values.length; dimension++) {
                double[] dimensionValues =
                        Objects.requireNonNull(
                                values[dimension],
                                "Numeric series contains a null dimension at "
                                        + dimension
                                        + "."
                        );

                transformPrimitiveDimension(
                        dimensionValues,
                        center(stats, dimension),
                        scale(stats, dimension)
                );
            }

            return values;
        }

        if (series instanceof Double[][] values) {
            requireDimensionCompatibility(values.length, stats);

            for (int dimension = 0; dimension < values.length; dimension++) {
                Double[] dimensionValues =
                        Objects.requireNonNull(
                                values[dimension],
                                "Numeric series contains a null dimension at "
                                        + dimension
                                        + "."
                        );

                transformBoxedDimension(
                        dimensionValues,
                        center(stats, dimension),
                        scale(stats, dimension)
                );
            }

            return values;
        }

        throw new IllegalArgumentException(
                "Unsupported standardization series type: "
                        + series.getClass().getTypeName()
                        + ". Expected double[], Double[], double[][], or "
                        + "Double[][]."
        );
    }

    private static void transformPrimitiveDimension(
            double[] values,
            double center,
            double scale
    ) {
        for (int index = 0; index < values.length; index++) {
            double value = values[index];

            if (!Double.isNaN(value)) {
                values[index] = (value - center) / scale;
            }
        }
    }

    private static void transformBoxedDimension(
            Double[] values,
            double center,
            double scale
    ) {
        for (int index = 0; index < values.length; index++) {
            Double value = values[index];

            if (value != null && !Double.isNaN(value)) {
                values[index] = (value - center) / scale;
            }
        }
    }

    private static double center(
            StandardizationStats stats,
            int dimension
    ) {
        return stats.getCenterForDimension(dimension);
    }

    private static double scale(
            StandardizationStats stats,
            int dimension
    ) {
        return stats.getScaleForDimension(dimension);
    }

    private static void requireUnivariateStats(
            StandardizationStats stats
    ) {
        requireDimensionCompatibility(1, stats);
    }

    private static void requireDimensionCompatibility(
            int dimensionCount,
            StandardizationStats stats
    ) {
        if (dimensionCount < 1) {
            throw new IllegalArgumentException(
                    "Numeric series must contain at least one dimension."
            );
        }

        if (stats.getScope() == StandardizationScope.PER_DIMENSION
                && dimensionCount != stats.getStatisticGroupCount()) {

            throw new IllegalArgumentException(
                    "Numeric series contains "
                            + dimensionCount
                            + " dimensions, but PER_DIMENSION statistics "
                            + "contain "
                            + stats.getStatisticGroupCount()
                            + " groups."
            );
        }
    }
}