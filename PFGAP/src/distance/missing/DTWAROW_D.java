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
 * Dependent multivariate generalization of DTW-AROW.
 *
 * This class extends the univariate DTW-AROW idea to multivariate time series
 * using a single shared warping path across dimensions, consistent with the
 * dependent multivariate DTW convention used elsewhere in PF-GAP.
 *
 * Expected multivariate shape:
 *
 *     series[dimension][time]
 *
 * Supported input types:
 *
 *     double[][]
 *     Double[][]
 *     Object[][] containing numeric values
 *
 * Missing-value conventions:
 *
 *     null
 *     Double.NaN
 *     Float.NaN
 *
 * Multivariate local cost:
 *
 *     delta_ext_D(x_i, y_j) =
 *
 *         0
 *             if the vector comparison is not computable
 *
 *         squared NaN-Euclidean distance between x_i and y_j
 *             otherwise
 *
 * A vector comparison is computable when there is at least one dimension d
 * such that both x[d][i] and y[d][j] are present numeric values.
 *
 * Horizontal and vertical transition restrictions:
 *
 *     A horizontal move into (i,j), from (i,j-1), is forbidden if either
 *     endpoint comparison is not computable:
 *
 *         comparable(x_i, y_j)     must be true
 *         comparable(x_i, y_{j-1}) must be true
 *
 *     A vertical move into (i,j), from (i-1,j), is forbidden if either
 *     endpoint comparison is not computable:
 *
 *         comparable(x_i,   y_j) must be true
 *         comparable(x_{i-1}, y_j) must be true
 *
 * Diagonal moves remain allowed through incomputable comparisons, with local
 * cost 0, matching the spirit of scalar DTW-AROW.
 *
 * This is a principled dependent multivariate generalization of DTW-AROW,
 * not a verbatim scalar algorithm.
 */
public class DTWAROW_D implements Serializable {

    private static final long serialVersionUID = 1L;

    private static final double INF = Double.POSITIVE_INFINITY;

    public DTWAROW_D() {
    }

    /**
     * Convenience overload for reflective calls expecting distance(Object,Object).
     */
    public synchronized double distance(Object Series1, Object Series2) {
        return distance(Series1, Series2, Double.POSITIVE_INFINITY, -1);
    }

    /**
     * Convenience overload using no explicit window constraint.
     */
    public synchronized double distance(Object Series1, Object Series2, double bsf) {
        return distance(Series1, Series2, bsf, -1);
    }

    /**
     * Computes dependent multivariate DTW-AROW distance.
     *
     * @param Series1 Object expected to be double[][], Double[][], or numeric Object[][]
     * @param Series2 Object expected to be double[][], Double[][], or numeric Object[][]
     * @param bsf best-so-far threshold
     * @param windowSize Sakoe-Chiba window size; -1 means unconstrained
     * @return dependent multivariate DTW-AROW distance
     */
    public synchronized double distance(
            Object Series1,
            Object Series2,
            double bsf,
            int windowSize) {

        MultiSeriesAccessor x = makeAccessor(Series1);
        MultiSeriesAccessor y = makeAccessor(Series2);

        return compute(x, y, bsf, windowSize, false).distance;
    }

    /**
     * Computes the optimal dependent multivariate DTW-AROW alignment path.
     *
     * Path entries are zero-based time-index pairs into Series1 and Series2.
     */
    public synchronized List<Pair<Integer, Integer>> getAlignmentPath(
            Object Series1,
            Object Series2,
            int windowSize) {

        MultiSeriesAccessor x = makeAccessor(Series1);
        MultiSeriesAccessor y = makeAccessor(Series2);

        return compute(
                x,
                y,
                Double.POSITIVE_INFINITY,
                windowSize,
                true
        ).path;
    }

    private DTWResult compute(
            MultiSeriesAccessor x,
            MultiSeriesAccessor y,
            double bsf,
            int windowSize,
            boolean keepPath) {

        validateCompatibleSeries(x, y);

        int lenX = x.length();
        int lenY = y.length();

        if (lenX == 0 || lenY == 0 || x.dimensions() == 0) {
            return new DTWResult(INF, Collections.emptyList());
        }

        int availableX = x.countAvailableTimePoints();
        int availableY = y.countAvailableTimePoints();

        if (availableX + availableY == 0) {
            return new DTWResult(INF, Collections.emptyList());
        }

        double gamma = ((double) (lenX + lenY)) / (availableX + availableY);

        int window = normalizeWindow(windowSize, lenX, lenY);

        double[][] cost = new double[lenX + 1][lenY + 1];
        int[][] step = keepPath ? new int[lenX + 1][lenY + 1] : null;

        initializeCostMatrix(cost, lenX, lenY);

        for (int i = 1; i <= lenX; i++) {

            int jStart = Math.max(1, i - window);
            int jStop = Math.min(lenY, i + window);

            for (int j = jStart; j <= jStop; j++) {

                int xi = i - 1;
                int yj = j - 1;

                double localCost = extendedSquaredDistanceAt(x, xi, y, yj);

                double diagonal = cost[i - 1][j - 1];

                double horizontal = safeAdd(
                        cost[i][j - 1],
                        invalidHorizontalStep(x, xi, y, yj) ? INF : 0.0
                );

                double vertical = safeAdd(
                        cost[i - 1][j],
                        invalidVerticalStep(x, xi, y, yj) ? INF : 0.0
                );

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

        double finalCost = cost[lenX][lenY];

        if (Double.isInfinite(finalCost)) {
            return new DTWResult(INF, Collections.emptyList());
        }

        double distance = Math.sqrt(gamma * finalCost);

        if (distance > bsf) {
            return new DTWResult(INF, Collections.emptyList());
        }

        List<Pair<Integer, Integer>> path = keepPath
                ? backtrack(step, lenX, lenY)
                : Collections.emptyList();

        return new DTWResult(distance, path);
    }

    private void validateCompatibleSeries(
            MultiSeriesAccessor x,
            MultiSeriesAccessor y) {

        if (x.dimensions() != y.dimensions()) {
            throw new IllegalArgumentException(
                    "Both multivariate series must have the same number of dimensions."
            );
        }
    }

    private int normalizeWindow(int windowSize, int lenX, int lenY) {

        if (windowSize == -1) {
            return Math.max(lenX, lenY);
        }

        if (windowSize < 0) {
            throw new IllegalArgumentException(
                    "windowSize must be -1 or a non-negative integer."
            );
        }

        return windowSize;
    }

    private void initializeCostMatrix(
            double[][] cost,
            int lenX,
            int lenY) {

        for (int i = 0; i <= lenX; i++) {
            for (int j = 0; j <= lenY; j++) {
                cost[i][j] = INF;
            }
        }

        /*
         * DTW-AROW boundary convention from the algorithm:
         *
         *     c[i][0] = 0
         *     c[0][j] = 0
         *
         * The zeroth row and zeroth column are DP boundaries, not observed
         * time points.
         */
        for (int i = 0; i <= lenX; i++) {
            cost[i][0] = 0.0;
        }

        for (int j = 0; j <= lenY; j++) {
            cost[0][j] = 0.0;
        }
    }

    /**
     * Multivariate delta_ext.
     *
     * Returns 0.0 when the vector comparison is not computable. Otherwise
     * returns the scaled squared NaN-Euclidean distance across dimensions.
     */
    private double extendedSquaredDistanceAt(
            MultiSeriesAccessor x,
            int xi,
            MultiSeriesAccessor y,
            int yj) {

        double squared = squaredNaNEuclideanAt(x, xi, y, yj);

        if (Double.isInfinite(squared)) {
            return 0.0;
        }

        return squared;
    }

    /**
     * A horizontal move into (xi,yj) comes from (xi,yj-1).
     *
     * It is illegal if either endpoint vector comparison is not computable:
     *
     *     comparable(xi, yj)
     *     comparable(xi, yj - 1)
     */
    private boolean invalidHorizontalStep(
            MultiSeriesAccessor x,
            int xi,
            MultiSeriesAccessor y,
            int yj) {

        return !comparableIfInRange(x, xi, y, yj)
                || !comparableIfInRange(x, xi, y, yj - 1);
    }

    /**
     * A vertical move into (xi,yj) comes from (xi-1,yj).
     *
     * It is illegal if either endpoint vector comparison is not computable:
     *
     *     comparable(xi, yj)
     *     comparable(xi - 1, yj)
     */
    private boolean invalidVerticalStep(
            MultiSeriesAccessor x,
            int xi,
            MultiSeriesAccessor y,
            int yj) {

        return !comparableIfInRange(x, xi, y, yj)
                || !comparableIfInRange(x, xi - 1, y, yj);
    }

    /**
     * DP boundary indices are treated as valid boundaries. Real time-index
     * pairs must be computable.
     */
    private boolean comparableIfInRange(
            MultiSeriesAccessor x,
            int xi,
            MultiSeriesAccessor y,
            int yj) {

        if (xi < 0 || xi >= x.length()) {
            return true;
        }

        if (yj < 0 || yj >= y.length()) {
            return true;
        }

        return comparableAt(x, xi, y, yj);
    }

    /**
     * A vector comparison is computable if the scaled squared NaN-Euclidean
     * distance is finite, which occurs when at least one dimension is jointly
     * observed and numeric.
     */
    private boolean comparableAt(
            MultiSeriesAccessor x,
            int xi,
            MultiSeriesAccessor y,
            int yj) {

        return !Double.isInfinite(squaredNaNEuclideanAt(x, xi, y, yj));
    }

    /**
     * Computes scaled squared NaN-Euclidean distance between two multivariate
     * time-point vectors:
     *
     *     (D / D_obs) * sum_d (x_d - y_d)^2
     *
     * where the sum is over dimensions where both values are observed.
     *
     * Returns POSITIVE_INFINITY if no jointly observed numeric dimensions
     * exist.
     */
    private double squaredNaNEuclideanAt(
            MultiSeriesAccessor x,
            int xi,
            MultiSeriesAccessor y,
            int yj) {

        int dims = x.dimensions();

        double sum = 0.0;
        int observed = 0;

        for (int d = 0; d < dims; d++) {

            if (x.isMissing(d, xi) || y.isMissing(d, yj)) {
                continue;
            }

            double xv = x.value(d, xi);
            double yv = y.value(d, yj);

            double diff = xv - yv;
            sum += diff * diff;
            observed++;
        }

        if (observed == 0) {
            return INF;
        }

        return ((double) dims / observed) * sum;
    }

    private double safeAdd(double a, double b) {

        if (Double.isInfinite(a) || Double.isInfinite(b)) {
            return INF;
        }

        return a + b;
    }

    private List<Pair<Integer, Integer>> backtrack(
            int[][] step,
            int lenX,
            int lenY) {

        List<Pair<Integer, Integer>> path = new ArrayList<>();

        int i = lenX;
        int j = lenY;

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

    private MultiSeriesAccessor makeAccessor(Object series) {

        if (series instanceof double[][]) {
            return new PrimitiveMultiSeriesAccessor((double[][]) series);
        }

        if (series instanceof Object[][]) {
            return new ObjectMultiSeriesAccessor((Object[][]) series);
        }

        throw new IllegalArgumentException(
                "DTWAROW_D supports double[][], Double[][], or numeric Object[][] inputs."
        );
    }

    public int get_random_window(ObjectDataset d, Random r) {
        int bound = Math.max(1, (AppContext.length + 1) / 4);
        return r.nextInt(bound);
    }

    private interface MultiSeriesAccessor {

        int dimensions();

        int length();

        boolean isMissing(int dimension, int timeIndex);

        double value(int dimension, int timeIndex);

        int countAvailableTimePoints();
    }

    private static class PrimitiveMultiSeriesAccessor
            implements MultiSeriesAccessor {

        private final double[][] series;
        private final int dimensions;
        private final int length;

        PrimitiveMultiSeriesAccessor(double[][] series) {

            this.series = series;
            this.dimensions = series.length;
            this.length = dimensions == 0 ? 0 : series[0].length;

            validateRectangular();
        }

        private void validateRectangular() {

            for (int d = 0; d < dimensions; d++) {
                if (series[d].length != length) {
                    throw new IllegalArgumentException(
                            "All dimensions must have consistent time lengths."
                    );
                }
            }
        }

        @Override
        public int dimensions() {
            return dimensions;
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public boolean isMissing(int dimension, int timeIndex) {
            return Double.isNaN(series[dimension][timeIndex]);
        }

        @Override
        public double value(int dimension, int timeIndex) {
            return series[dimension][timeIndex];
        }

        @Override
        public int countAvailableTimePoints() {

            int count = 0;

            for (int t = 0; t < length; t++) {

                boolean available = false;

                for (int d = 0; d < dimensions; d++) {
                    if (!Double.isNaN(series[d][t])) {
                        available = true;
                        break;
                    }
                }

                if (available) {
                    count++;
                }
            }

            return count;
        }
    }

    private static class ObjectMultiSeriesAccessor
            implements MultiSeriesAccessor {

        private final Object[][] series;
        private final int dimensions;
        private final int length;

        ObjectMultiSeriesAccessor(Object[][] series) {

            this.series = series;
            this.dimensions = series.length;
            this.length = dimensions == 0 ? 0 : series[0].length;

            validateRectangular();
        }

        private void validateRectangular() {

            for (int d = 0; d < dimensions; d++) {
                if (series[d].length != length) {
                    throw new IllegalArgumentException(
                            "All dimensions must have consistent time lengths."
                    );
                }
            }
        }

        @Override
        public int dimensions() {
            return dimensions;
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public boolean isMissing(int dimension, int timeIndex) {

            Object v = series[dimension][timeIndex];

            if (v == null) {
                return true;
            }

            if (v instanceof Double) {
                return Double.isNaN((Double) v);
            }

            if (v instanceof Float) {
                return Float.isNaN((Float) v);
            }

            if (v instanceof Number) {
                return Double.isNaN(((Number) v).doubleValue());
            }

            throw new IllegalArgumentException(
                    "DTWAROW_D requires numeric values at observed positions. "
                            + "Found "
                            + v.getClass().getName()
                            + " at dimension "
                            + dimension
                            + ", time index "
                            + timeIndex
                            + "."
            );
        }

        @Override
        public double value(int dimension, int timeIndex) {

            Object v = series[dimension][timeIndex];

            if (isMissing(dimension, timeIndex)) {
                throw new IllegalArgumentException(
                        "Cannot read missing value as numeric at dimension "
                                + dimension
                                + ", time index "
                                + timeIndex
                                + "."
                );
            }

            if (!(v instanceof Number)) {
                throw new IllegalArgumentException(
                        "DTWAROW_D requires numeric values at observed positions. "
                                + "Found "
                                + v.getClass().getName()
                                + " at dimension "
                                + dimension
                                + ", time index "
                                + timeIndex
                                + "."
                );
            }

            return ((Number) v).doubleValue();
        }

        @Override
        public int countAvailableTimePoints() {

            int count = 0;

            for (int t = 0; t < length; t++) {

                boolean available = false;

                for (int d = 0; d < dimensions; d++) {
                    if (!isMissing(d, t)) {
                        available = true;
                        break;
                    }
                }

                if (available) {
                    count++;
                }
            }

            return count;
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
