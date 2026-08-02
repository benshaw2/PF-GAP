package distance.missing;

import java.io.Serializable;

/**
 * Missing-value-aware Euclidean distance for univariate numeric series.
 *
 * Distance:
 *
 *     d(x, y) = sqrt( (M / M_obs) * sum_{i observed in both} (x_i - y_i)^2 )
 *
 * where:
 * - M is the full series length;
 * - M_obs is the number of positions where both series are observed.
 *
 * The corresponding squared distance is:
 *
 *     d^2(x, y) = (M / M_obs) * sum_{i observed in both} (x_i - y_i)^2
 *
 * Supported input types:
 * - double[]
 * - Double[]
 * - Object[] containing numeric values
 *
 * Missing-value conventions:
 * - null
 * - Double.NaN
 * - Float.NaN
 *
 * Notes:
 * - PF-GAP currently represents file-loaded numeric missing data as Double[]
 *   or Double[][] with null missing entries.
 * - Primitive double[] support is included for NaN-backed numeric data,
 *   post-imputation data, and future Python/NumPy interop.
 */
public class NaNEuclidean implements Serializable {

    private static final long serialVersionUID = 1L;

    public NaNEuclidean() {
    }

    /**
     * Computes NaN-Euclidean distance.
     *
     * This overload is useful for reflective calls that expect
     * distance(Object, Object).
     */
    public synchronized double distance(Object Series1, Object Series2) {
        return distance(Series1, Series2, Double.POSITIVE_INFINITY);
    }

    /**
     * Computes NaN-Euclidean distance with a best-so-far threshold.
     *
     * The threshold is applied after the final scaled distance is computed.
     * Exact early abandoning is not used here because the final scale factor
     * depends on the final number of jointly observed positions.
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf) {

        double squared = squaredDistance(Series1, Series2);

        if (Double.isInfinite(squared)) {
            return Double.POSITIVE_INFINITY;
        }

        double result = Math.sqrt(squared);

        return result > bsf ? Double.POSITIVE_INFINITY : result;
    }

    /**
     * Computes the scaled squared NaN-Euclidean distance:
     *
     *     (M / M_obs) * sum_{i observed in both} (x_i - y_i)^2
     *
     * This helper is useful when another distance needs the squared local
     * vector cost rather than the square-rooted distance.
     *
     * Returns Double.POSITIVE_INFINITY if there are no jointly observed
     * numeric positions.
     */
    public synchronized double squaredDistance(Object Series1, Object Series2) {

        if (Series1 instanceof double[] && Series2 instanceof double[]) {
            return squaredDistancePrimitive(
                    (double[]) Series1,
                    (double[]) Series2
            );
        }

        if (Series1 instanceof Object[] && Series2 instanceof Object[]) {
            return squaredDistanceObject(
                    (Object[]) Series1,
                    (Object[]) Series2
            );
        }

        throw new IllegalArgumentException(
                "NaNEuclidean supports double[], Double[], or numeric Object[] inputs."
        );
    }

    /**
     * Returns true if the NaN-Euclidean comparison is computable.
     *
     * A comparison is computable when the two series have at least one
     * jointly observed numeric position.
     */
    public synchronized boolean isComputable(Object Series1, Object Series2) {
        return !Double.isInfinite(squaredDistance(Series1, Series2));
    }

    private double squaredDistancePrimitive(double[] series1, double[] series2) {

        MissingDistanceTools.validateSameLength(series1, series2);

        double sum = 0.0;
        int jointlyObserved = 0;

        for (int i = 0; i < series1.length; i++) {

            double a = series1[i];
            double b = series2[i];

            if (MissingDistanceTools.isMissing(a)
                    || MissingDistanceTools.isMissing(b)) {
                continue;
            }

            sum += MissingDistanceTools.squaredDifference(a, b);
            jointlyObserved++;
        }

        if (jointlyObserved == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double scale = MissingDistanceTools.scaleFactor(
                series1.length,
                jointlyObserved
        );

        return scale * sum;
    }

    private double squaredDistanceObject(Object[] series1, Object[] series2) {

        MissingDistanceTools.validateSameLength(series1, series2);

        double sum = 0.0;
        int jointlyObserved = 0;

        for (int i = 0; i < series1.length; i++) {

            Object a = series1[i];
            Object b = series2[i];

            if (MissingDistanceTools.isMissing(a)
                    || MissingDistanceTools.isMissing(b)) {
                continue;
            }

            if (!(a instanceof Number) || !(b instanceof Number)) {
                throw new IllegalArgumentException(
                        "NaNEuclidean requires numeric values at all jointly observed positions. "
                                + "Found "
                                + a.getClass().getName()
                                + " and "
                                + b.getClass().getName()
                                + " at index "
                                + i
                                + "."
                );
            }

            double av = ((Number) a).doubleValue();
            double bv = ((Number) b).doubleValue();

            if (MissingDistanceTools.isMissing(av)
                    || MissingDistanceTools.isMissing(bv)) {
                continue;
            }

            sum += MissingDistanceTools.squaredDifference(av, bv);
            jointlyObserved++;
        }

        if (jointlyObserved == 0) {
            return Double.POSITIVE_INFINITY;
        }

        double scale = MissingDistanceTools.scaleFactor(
                series1.length,
                jointlyObserved
        );

        return scale * sum;
    }
}