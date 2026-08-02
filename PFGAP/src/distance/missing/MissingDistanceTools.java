package distance.missing;

/**
 * Generic utilities for missing-value-aware distance functions.
 *
 * This class is intentionally limited to reusable missing-value,
 * numeric conversion, and shape-validation helpers. It does not
 * implement any distance-specific logic such as DTW recurrences,
 * alignment rules, or imputation behavior.
 *
 * Supported missing-value conventions:
 * - null
 * - Double.NaN
 * - Float.NaN
 *
 * Supported numeric conventions:
 * - primitive double values
 * - boxed Number values, including Double, Float, Integer, etc.
 *
 * The public distance classes in this package should continue to
 * accept Object arguments, matching the rest of the distance package.
 */
public final class MissingDistanceTools {

    private MissingDistanceTools() {
        // Utility class; do not instantiate.
    }

    /**
     * Returns true if the supplied value should be treated as missing.
     *
     * For numeric values, Double.NaN and Float.NaN are treated as missing.
     * For object-valued arrays, null is treated as missing.
     * Other non-null objects are not considered missing by this method.
     */
    public static boolean isMissing(Object value) {

        if (value == null) {
            return true;
        }

        if (value instanceof Double) {
            return Double.isNaN((Double) value);
        }

        if (value instanceof Float) {
            return Float.isNaN((Float) value);
        }

        return false;
    }

    /**
     * Returns true if the primitive value is missing.
     */
    public static boolean isMissing(double value) {
        return Double.isNaN(value);
    }

    /**
     * Returns true if the supplied value is present and numeric.
     */
    public static boolean isPresentNumber(Object value) {
        return !isMissing(value) && value instanceof Number;
    }

    /**
     * Converts a present numeric object to double.
     *
     * Throws an IllegalArgumentException if the value is missing
     * or not numeric.
     */
    public static double toDouble(Object value) {

        if (isMissing(value)) {
            throw new IllegalArgumentException(
                    "Cannot convert missing value to double.");
        }

        if (!(value instanceof Number)) {
            throw new IllegalArgumentException(
                    "Expected a numeric value but received: "
                            + value.getClass().getName());
        }

        return ((Number) value).doubleValue();
    }

    /**
     * Returns the squared difference between two primitive values.
     *
     * Both values are assumed to be present.
     */
    public static double squaredDifference(double a, double b) {

        double diff = a - b;
        return diff * diff;
    }

    /**
     * Returns the squared difference between two numeric object values.
     *
     * Both values are assumed to be present and numeric.
     */
    public static double squaredDifference(Object a, Object b) {
        return squaredDifference(toDouble(a), toDouble(b));
    }

    /**
     * Returns the squared difference if both values are present;
     * otherwise returns 0.0.
     *
     * This is useful for missing-aware local costs where missing
     * numeric comparisons should contribute no squared error.
     */
    public static double squaredDifferenceIfPresent(Object a, Object b) {

        if (isMissing(a) || isMissing(b)) {
            return 0.0;
        }

        return squaredDifference(a, b);
    }

    /**
     * Returns the squared difference if both primitive values are present;
     * otherwise returns 0.0.
     */
    public static double squaredDifferenceIfPresent(double a, double b) {

        if (Double.isNaN(a) || Double.isNaN(b)) {
            return 0.0;
        }

        return squaredDifference(a, b);
    }

    /**
     * Validates that two primitive 1D series have equal length.
     */
    public static void validateSameLength(double[] series1, double[] series2) {

        if (series1.length != series2.length) {
            throw new IllegalArgumentException(
                    "Both series must have the same length.");
        }
    }

    /**
     * Validates that two object 1D series have equal length.
     */
    public static void validateSameLength(Object[] series1, Object[] series2) {

        if (series1.length != series2.length) {
            throw new IllegalArgumentException(
                    "Both series must have the same length.");
        }
    }

    /**
     * Validates that two primitive 2D series have the same number of rows.
     */
    public static void validateSameRows(double[][] series1, double[][] series2) {

        if (series1.length != series2.length) {
            throw new IllegalArgumentException(
                    "Both multivariate series must have the same number of rows.");
        }
    }

    /**
     * Validates that two object 2D series have the same number of rows.
     */
    public static void validateSameRows(Object[][] series1, Object[][] series2) {

        if (series1.length != series2.length) {
            throw new IllegalArgumentException(
                    "Both multivariate series must have the same number of rows.");
        }
    }

    /**
     * Validates that all corresponding rows in two primitive 2D series
     * have equal length.
     */
    public static void validateSameShape(double[][] series1, double[][] series2) {

        validateSameRows(series1, series2);

        for (int i = 0; i < series1.length; i++) {
            if (series1[i].length != series2[i].length) {
                throw new IllegalArgumentException(
                        "Corresponding rows must have the same length.");
            }
        }
    }

    /**
     * Validates that all corresponding rows in two object 2D series
     * have equal length.
     */
    public static void validateSameShape(Object[][] series1, Object[][] series2) {

        validateSameRows(series1, series2);

        for (int i = 0; i < series1.length; i++) {
            if (series1[i].length != series2[i].length) {
                throw new IllegalArgumentException(
                        "Corresponding rows must have the same length.");
            }
        }
    }

    /**
     * Counts present entries in a primitive 1D series.
     */
    public static int countAvailable(double[] series) {

        int count = 0;

        for (double value : series) {
            if (!Double.isNaN(value)) {
                count++;
            }
        }

        return count;
    }

    /**
     * Counts present entries in an object 1D series.
     */
    public static int countAvailable(Object[] series) {

        int count = 0;

        for (Object value : series) {
            if (!isMissing(value)) {
                count++;
            }
        }

        return count;
    }

    /**
     * Counts present entries in a primitive 2D series.
     */
    public static int countAvailable(double[][] series) {

        int count = 0;

        for (double[] row : series) {
            count += countAvailable(row);
        }

        return count;
    }

    /**
     * Counts present entries in an object 2D series.
     */
    public static int countAvailable(Object[][] series) {

        int count = 0;

        for (Object[] row : series) {
            count += countAvailable(row);
        }

        return count;
    }

    /**
     * Counts jointly observed positions in two primitive 1D series.
     */
    public static int countJointlyObserved(double[] series1, double[] series2) {

        validateSameLength(series1, series2);

        int count = 0;

        for (int i = 0; i < series1.length; i++) {
            if (!Double.isNaN(series1[i]) && !Double.isNaN(series2[i])) {
                count++;
            }
        }

        return count;
    }

    /**
     * Counts jointly observed positions in two object 1D series.
     */
    public static int countJointlyObserved(Object[] series1, Object[] series2) {

        validateSameLength(series1, series2);

        int count = 0;

        for (int i = 0; i < series1.length; i++) {
            if (!isMissing(series1[i]) && !isMissing(series2[i])) {
                count++;
            }
        }

        return count;
    }

    /**
     * Returns the NaN-Euclidean scale factor:
     *
     * fullLength / jointlyObservedLength
     *
     * If there are no jointly observed positions, returns
     * Double.POSITIVE_INFINITY.
     */
    public static double scaleFactor(
            int fullLength,
            int jointlyObservedLength) {

        if (jointlyObservedLength == 0) {
            return Double.POSITIVE_INFINITY;
        }

        return ((double) fullLength) / jointlyObservedLength;
    }
}