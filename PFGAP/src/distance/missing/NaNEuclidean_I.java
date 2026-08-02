package distance.missing;

import java.io.Serializable;

/**
 * Independent missing-value-aware Euclidean distance for multivariate
 * numeric time series.
 *
 * This class mirrors the convention used by distance.multiTS.Euclidean_I
 * and distance.multiTS.DTW_I:
 *
 * - each row/component of series1 is compared to the corresponding
 *   row/component of series2;
 * - component distances are averaged across rows.
 *
 * Supported input types:
 * - double[][]
 * - Double[][]
 * - Object[][] containing numeric values
 *
 * Missing-value conventions:
 * - null
 * - Double.NaN
 * - Float.NaN
 *
 * Notes:
 * - For file-loaded numeric data with missing values, PF-GAP currently
 *   uses Double[][] with null missing entries.
 * - Primitive double[][] support is included for NaN-backed numeric data,
 *   post-imputation data, and future Python/NumPy interop.
 */
public class NaNEuclidean_I implements Serializable {

    private static final long serialVersionUID = 1L;

    private final NaNEuclidean nanEuclidean;

    public NaNEuclidean_I() {
        this.nanEuclidean = new NaNEuclidean();
    }

    /**
     * Computes the independent NaN-Euclidean distance.
     *
     * This overload is useful for reflective calls that expect
     * distance(Object, Object).
     */
    public synchronized double distance(Object Series1, Object Series2) {
        return distance(Series1, Series2, Double.POSITIVE_INFINITY);
    }

    /**
     * Computes the average NaN-Euclidean distance across all corresponding
     * rows/components of the two multivariate series.
     *
     * The best-so-far threshold is applied after the final averaged distance
     * is computed. This avoids incorrectly abandoning a multivariate distance
     * merely because one component distance exceeds the final average threshold.
     *
     * @param Series1 Object expected to be double[][], Double[][], or numeric Object[][]
     * @param Series2 Object expected to be double[][], Double[][], or numeric Object[][]
     * @param bsf best-so-far threshold
     * @return average NaN-Euclidean distance across rows/components
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf) {

        double result;

        if (Series1 instanceof double[][] && Series2 instanceof double[][]) {
            result = distancePrimitive((double[][]) Series1, (double[][]) Series2);
        } else if (Series1 instanceof Object[][] && Series2 instanceof Object[][]) {
            result = distanceObject((Object[][]) Series1, (Object[][]) Series2);
        } else {
            throw new IllegalArgumentException(
                    "NaNEuclidean_I supports double[][], Double[][], or numeric Object[][] inputs."
            );
        }

        return result > bsf ? Double.POSITIVE_INFINITY : result;
    }

    private double distancePrimitive(double[][] series1, double[][] series2) {

        MissingDistanceTools.validateSameRows(series1, series2);

        if (series1.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;

        for (int i = 0; i < series1.length; i++) {
            double componentDistance = nanEuclidean.distance(
                    series1[i],
                    series2[i],
                    Double.POSITIVE_INFINITY
            );

            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }

            totalDistance += componentDistance;
        }

        return totalDistance / series1.length;
    }

    private double distanceObject(Object[][] series1, Object[][] series2) {

        MissingDistanceTools.validateSameRows(series1, series2);

        if (series1.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;

        for (int i = 0; i < series1.length; i++) {
            double componentDistance = nanEuclidean.distance(
                    series1[i],
                    series2[i],
                    Double.POSITIVE_INFINITY
            );

            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }

            totalDistance += componentDistance;
        }

        return totalDistance / series1.length;
    }
}