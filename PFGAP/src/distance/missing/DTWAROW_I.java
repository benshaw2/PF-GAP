package distance.missing;

import core.contracts.ObjectDataset;

import java.io.Serializable;
import java.util.Random;

/**
 * Independent DTW-AROW distance for multivariate numeric time series
 * with missing values.
 *
 * This class follows the independent multivariate convention used elsewhere
 * in PF-GAP:
 *
 * - each row/component of series1 is compared to the corresponding
 *   row/component of series2;
 * - each component distance is computed using univariate DTW-AROW;
 * - the final distance is the average component distance.
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
public class DTWAROW_I implements Serializable {

    private static final long serialVersionUID = 1L;

    private final DTWAROW dtwArow;

    public DTWAROW_I() {
        this.dtwArow = new DTWAROW();
    }

    /**
     * Convenience overload for reflective calls.
     */
    public synchronized double distance(Object Series1, Object Series2) {
        return distance(Series1, Series2, Double.POSITIVE_INFINITY, -1);
    }

    /**
     * Convenience overload using no window constraint.
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf) {
        return distance(Series1, Series2, bsf, -1);
    }

    /**
     * Computes the independent multivariate DTW-AROW distance.
     *
     * The bsf threshold is applied after averaging all component distances.
     * This avoids incorrectly abandoning when one component distance exceeds
     * bsf but the final averaged distance may still be below bsf.
     *
     * @param Series1 Object expected to be double[][], Double[][], or numeric Object[][]
     * @param Series2 Object expected to be double[][], Double[][], or numeric Object[][]
     * @param bsf best-so-far threshold
     * @param windowSize Sakoe-Chiba window size; -1 means unconstrained
     * @return average DTW-AROW distance across rows/components
     */
    public synchronized double distance(
            Object Series1,
            Object Series2,
            double bsf,
            int windowSize) {

        double result;

        if (Series1 instanceof double[][] && Series2 instanceof double[][]) {
            result = distancePrimitive(
                    (double[][]) Series1,
                    (double[][]) Series2,
                    windowSize
            );
        } else if (Series1 instanceof Object[][] && Series2 instanceof Object[][]) {
            result = distanceObject(
                    (Object[][]) Series1,
                    (Object[][]) Series2,
                    windowSize
            );
        } else {
            throw new IllegalArgumentException(
                    "DTWAROW_I supports double[][], Double[][], or numeric Object[][] inputs."
            );
        }

        return result > bsf ? Double.POSITIVE_INFINITY : result;
    }

    private double distancePrimitive(
            double[][] series1,
            double[][] series2,
            int windowSize) {

        MissingDistanceTools.validateSameRows(series1, series2);

        if (series1.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;

        for (int i = 0; i < series1.length; i++) {

            double componentDistance = dtwArow.distance(
                    series1[i],
                    series2[i],
                    Double.POSITIVE_INFINITY,
                    windowSize
            );

            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }

            totalDistance += componentDistance;
        }

        return totalDistance / series1.length;
    }

    private double distanceObject(
            Object[][] series1,
            Object[][] series2,
            int windowSize) {

        MissingDistanceTools.validateSameRows(series1, series2);

        if (series1.length == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double totalDistance = 0.0;

        for (int i = 0; i < series1.length; i++) {

            double componentDistance = dtwArow.distance(
                    series1[i],
                    series2[i],
                    Double.POSITIVE_INFINITY,
                    windowSize
            );

            if (Double.isInfinite(componentDistance)) {
                return Double.POSITIVE_INFINITY;
            }

            totalDistance += componentDistance;
        }

        return totalDistance / series1.length;
    }

    /**
     * Delegates random window selection to the univariate DTW-AROW distance.
     */
    public int get_random_window(ObjectDataset d, Random r) {
        return dtwArow.get_random_window(d, r);
    }
}