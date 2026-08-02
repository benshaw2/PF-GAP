package distance.missing;

import core.AppContext;
import core.contracts.ObjectDataset;
import util.Pair;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * DTW-AROW distance for univariate numeric time series with missing values.
 *
 * DTW-AROW differs from ordinary DTW in two key ways:
 *
 * 1. The local cost is extended so that missing comparisons contribute 0:
 *
 *      delta_ext(x, y) = 0          if x or y is missing
 *                        (x-y)^2    otherwise
 *
 * 2. Horizontal and vertical moves are penalized with infinity when the
 *    corresponding adjacent samples involve missing values.
 *
 * Missing-value conventions:
 * - null
 * - Double.NaN
 * - Float.NaN
 *
 * Supported input types:
 * - double[]
 * - Double[]
 * - Object[] containing numeric values
 *
 * Notes:
 * - PF-GAP currently stores file-loaded numeric data with missing values as
 *   Double[] with null missing entries.
 * - Primitive double[] support is included for NaN-backed numeric data,
 *   post-imputation data, and future Python/NumPy interop.
 * - Unlike Euclidean distances, DTW-AROW does not require equal-length series.
 */
public class DTWAROW implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final double INF = Double.POSITIVE_INFINITY;

    public DTWAROW() {
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
     * Computes DTW-AROW distance.
     *
     * @param Series1 Object expected to be double[], Double[], or numeric Object[]
     * @param Series2 Object expected to be double[], Double[], or numeric Object[]
     * @param bsf best-so-far threshold
     * @param windowSize Sakoe-Chiba window size; -1 means unconstrained
     * @return DTW-AROW distance, or POSITIVE_INFINITY if unavailable or above bsf
     */
    public synchronized double distance(
            Object Series1,
            Object Series2,
            double bsf,
            int windowSize) {

        SeriesAccessor x = makeAccessor(Series1);
        SeriesAccessor y = makeAccessor(Series2);

        DTWResult result = compute(x, y, bsf, windowSize, false);
        return result.distance;
    }

    /**
     * Computes the optimal DTW-AROW alignment path.
     *
     * Path entries are zero-based index pairs into Series1 and Series2.
     */
    public synchronized List<Pair<Integer, Integer>> getAlignmentPath(
            Object Series1,
            Object Series2,
            int windowSize) {

        SeriesAccessor x = makeAccessor(Series1);
        SeriesAccessor y = makeAccessor(Series2);

        DTWResult result = compute(
                x,
                y,
                Double.POSITIVE_INFINITY,
                windowSize,
                true
        );

        return result.path;
    }

    private DTWResult compute(
            SeriesAccessor x,
            SeriesAccessor y,
            double bsf,
            int windowSize,
            boolean keepPath) {

        int m = x.length();
        int n = y.length();

        if (m == 0 || n == 0) {
            return new DTWResult(INF, Collections.emptyList());
        }

        int availableX = x.countAvailable();
        int availableY = y.countAvailable();

        if (availableX + availableY == 0) {
            return new DTWResult(INF, Collections.emptyList());
        }

        double gamma = ((double) (m + n)) / (availableX + availableY);

        int window = windowSize == -1 ? Math.max(m, n) : windowSize;
        if (window < 0) {
            throw new IllegalArgumentException(
                    "windowSize must be -1 or a non-negative integer."
            );
        }

        double[][] cost = new double[m + 1][n + 1];
        int[][] step = keepPath ? new int[m + 1][n + 1] : null;

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                cost[i][j] = INF;
            }
        }

        /*
         * Algorithm 1 initializes c_{j,0} = 0 and c_{0,j'} = 0.
         * The zeroth row and column are DP boundaries, not real samples.
         */
        for (int i = 0; i <= m; i++) {
            cost[i][0] = 0.0;
        }
        for (int j = 0; j <= n; j++) {
            cost[0][j] = 0.0;
        }

        for (int i = 1; i <= m; i++) {

            int jStart = Math.max(1, i - window);
            int jStop = Math.min(n, i + window);

            for (int j = jStart; j <= jStop; j++) {

                int xi = i - 1;
                int yj = j - 1;

                double localCost = extendedSquaredDifference(x, xi, y, yj);

                double diagonal = cost[i - 1][j - 1];

                double horizontalPenalty =
                        invalidHorizontalStep(x, xi, y, yj) ? INF : 0.0;

                double verticalPenalty =
                        invalidVerticalStep(x, xi, y, yj) ? INF : 0.0;

                double horizontal = safeAdd(cost[i][j - 1], horizontalPenalty);
                double vertical = safeAdd(cost[i - 1][j], verticalPenalty);

                double minPrev = diagonal;
                int bestStep = 1; // diagonal

                if (horizontal < minPrev) {
                    minPrev = horizontal;
                    bestStep = 2; // horizontal
                }

                if (vertical < minPrev) {
                    minPrev = vertical;
                    bestStep = 3; // vertical
                }

                cost[i][j] = safeAdd(localCost, minPrev);

                if (keepPath) {
                    step[i][j] = bestStep;
                }
            }
        }

        double finalCost = cost[m][n];

        if (Double.isInfinite(finalCost)) {
            return new DTWResult(INF, Collections.emptyList());
        }

        double distance = Math.sqrt(gamma * finalCost);

        if (distance > bsf) {
            return new DTWResult(INF, Collections.emptyList());
        }

        List<Pair<Integer, Integer>> path = keepPath
                ? backtrack(step, m, n)
                : Collections.emptyList();

        return new DTWResult(distance, path);
    }

    /**
     * delta_ext(x_i, y_j)
     */
    private double extendedSquaredDifference(
            SeriesAccessor x,
            int xi,
            SeriesAccessor y,
            int yj) {

        if (x.isMissing(xi) || y.isMissing(yj)) {
            return 0.0;
        }

        return MissingDistanceTools.squaredDifference(
                x.value(xi),
                y.value(yj)
        );
    }

    /**
     * Horizontal move penalty from Algorithm 1.
     *
     * In zero-based indexing, y_{j'-1} becomes y[j - 1].
     * If j - 1 is outside the real series, it refers to the DP boundary
     * and is not treated as missing.
     */
    private boolean invalidHorizontalStep(
            SeriesAccessor x,
            int xi,
            SeriesAccessor y,
            int yj) {

        return x.isMissing(xi)
                || y.isMissing(yj)
                || y.isMissingIfInRange(yj - 1);
    }

    /**
     * Vertical move penalty from Algorithm 1.
     *
     * In zero-based indexing, x_{j-1} becomes x[i - 1].
     * If i - 1 is outside the real series, it refers to the DP boundary
     * and is not treated as missing.
     */
    private boolean invalidVerticalStep(
            SeriesAccessor x,
            int xi,
            SeriesAccessor y,
            int yj) {

        return x.isMissing(xi)
                || x.isMissingIfInRange(xi - 1)
                || y.isMissing(yj);
    }

    private double safeAdd(double a, double b) {

        if (Double.isInfinite(a) || Double.isInfinite(b)) {
            return INF;
        }

        return a + b;
    }

    private List<Pair<Integer, Integer>> backtrack(
            int[][] step,
            int m,
            int n) {

        List<Pair<Integer, Integer>> path = new ArrayList<>();

        int i = m;
        int j = n;

        while (i > 0 && j > 0) {

            path.add(0, new Pair<>(i - 1, j - 1));

            int s = step[i][j];

            if (s == 1) {
                i--;
                j--;
            } else if (s == 2) {
                j--;
            } else if (s == 3) {
                i--;
            } else {
                break;
            }
        }

        return path;
    }

    private SeriesAccessor makeAccessor(Object series) {

        if (series instanceof double[]) {
            return new PrimitiveSeriesAccessor((double[]) series);
        }

        if (series instanceof Object[]) {
            return new ObjectSeriesAccessor((Object[]) series);
        }

        throw new IllegalArgumentException(
                "DTWAROW supports double[], Double[], or numeric Object[] inputs."
        );
    }

    public int get_random_window(ObjectDataset d, Random r) {
        int bound = Math.max(1, (AppContext.length + 1) / 4);
        return r.nextInt(bound);
    }

    private interface SeriesAccessor {

        int length();

        boolean isMissing(int index);

        double value(int index);

        int countAvailable();

        default boolean isMissingIfInRange(int index) {

            if (index < 0 || index >= length()) {
                return false;
            }

            return isMissing(index);
        }
    }

    private static class PrimitiveSeriesAccessor implements SeriesAccessor {

        private final double[] series;

        PrimitiveSeriesAccessor(double[] series) {
            this.series = series;
        }

        @Override
        public int length() {
            return series.length;
        }

        @Override
        public boolean isMissing(int index) {
            return MissingDistanceTools.isMissing(series[index]);
        }

        @Override
        public double value(int index) {
            return series[index];
        }

        @Override
        public int countAvailable() {
            return MissingDistanceTools.countAvailable(series);
        }
    }

    private static class ObjectSeriesAccessor implements SeriesAccessor {

        private final Object[] series;

        ObjectSeriesAccessor(Object[] series) {
            this.series = series;
        }

        @Override
        public int length() {
            return series.length;
        }

        @Override
        public boolean isMissing(int index) {

            Object value = series[index];

            if (MissingDistanceTools.isMissing(value)) {
                return true;
            }

            if (value instanceof Number) {
                return Double.isNaN(((Number) value).doubleValue());
            }

            return false;
        }

        @Override
        public double value(int index) {

            Object value = series[index];

            if (isMissing(index)) {
                throw new IllegalArgumentException(
                        "Cannot read a missing value as numeric."
                );
            }

            if (!(value instanceof Number)) {
                throw new IllegalArgumentException(
                        "DTWAROW requires numeric values at all observed positions. "
                                + "Found "
                                + value.getClass().getName()
                                + " at index "
                                + index
                                + "."
                );
            }

            return ((Number) value).doubleValue();
        }

        @Override
        public int countAvailable() {
            return MissingDistanceTools.countAvailable(series);
        }
    }

    private static class DTWResult {

        final double distance;
        final List<Pair<Integer, Integer>> path;

        DTWResult(double distance, List<Pair<Integer, Integer>> path) {
            this.distance = distance;
            this.path = path;
        }
    }
}