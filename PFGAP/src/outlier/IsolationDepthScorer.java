package outlier;

import datasets.ListObjectDataset;
import trees.ProximityForest;
import trees.ProximityTree;

public class IsolationDepthScorer {

    private IsolationDepthScorer() {
        // Utility class
    }

    /**
     * Compute Isolation-Forest-style anomaly scores for all instances in a dataset.
     *
     * Higher scores indicate stronger outlier/anomaly behavior.
     *
     * Score formula:
     *
     *     score(x) = 2 ^ ( - averagePathLength(x) / c(n) )
     *
     * where c(n) is the standard expected path length correction.
     *
     * @param forest trained proximity forest in isolation mode
     * @param data dataset to score
     * @return anomaly scores, one per instance
     */
    public static double[] score(
            ProximityForest forest,
            ListObjectDataset data
    ) throws Exception {

        if (data == null) {
            return new double[]{};
        }

        return score(forest, data, data.size());
    }

    /**
     * Compute Isolation-Forest-style anomaly scores with an explicit
     * normalization sample size.
     *
     * For training scores, normalizationSampleSize is usually the training size.
     * For test scores, use the original training size when available.
     *
     * @param forest trained proximity forest in isolation mode
     * @param data dataset to score
     * @param normalizationSampleSize sample size used in c(n)
     * @return anomaly scores, one per instance
     */
    public static double[] score(
            ProximityForest forest,
            ListObjectDataset data,
            int normalizationSampleSize
    ) throws Exception {

        if (forest == null) {
            throw new IllegalArgumentException("forest cannot be null.");
        }

        if (data == null) {
            return new double[]{};
        }

        double[] scores = new double[data.size()];

        for (int i = 0; i < data.size(); i++) {
            scores[i] = scoreSeries(
                    forest,
                    data.get_series(i),
                    normalizationSampleSize
            );
        }

        return scores;
    }

    /**
     * Compute the anomaly score for a single series.
     *
     * @param forest trained proximity forest
     * @param series query series
     * @param normalizationSampleSize sample size used in c(n)
     * @return anomaly score
     */
    public static double scoreSeries(
            ProximityForest forest,
            Object series,
            int normalizationSampleSize
    ) throws Exception {

        double averagePathLength = averagePathLength(forest, series);
        double normalizer = correction(normalizationSampleSize);

        if (normalizer <= 0.0) {
            return averagePathLength <= 0.0 ? 1.0 : 0.0;
        }

        return Math.pow(2.0, -averagePathLength / normalizer);
    }

    /**
     * Compute average corrected path length across the trees.
     *
     * @param forest trained proximity forest
     * @param series query series
     * @return average corrected path length
     */
    public static double averagePathLength(
            ProximityForest forest,
            Object series
    ) throws Exception {

        ProximityTree[] trees = forest.getTrees();

        if (trees == null || trees.length == 0) {
            throw new IllegalStateException(
                    "Cannot compute isolation score: forest has no trees."
            );
        }

        double total = 0.0;
        int count = 0;

        for (ProximityTree tree : trees) {

            if (tree == null || tree.getRootNode() == null) {
                continue;
            }

            total += pathLength(tree, series);
            count++;
        }

        if (count == 0) {
            throw new IllegalStateException(
                    "Cannot compute isolation score: forest has no valid trees."
            );
        }

        return total / count;
    }

    /**
     * Compute corrected path length for one tree.
     *
     * The returned path length is:
     *
     *     depth(leaf) + c(leafSize)
     *
     * where leafSize is the number of in-bag training instances reaching
     * the terminal node.
     *
     * @param tree trained proximity tree
     * @param series query series
     * @return corrected path length
     */
    public static double pathLength(
            ProximityTree tree,
            Object series
    ) throws Exception {

        ProximityTree.Node node = tree.getRootNode();

        if (node == null) {
            return 0.0;
        }

        int depth = 0;

        while (!node.is_leaf()) {

            if (node.getSplitter() == null || node.get_children() == null) {
                break;
            }

            int branch = node.getSplitter().find_closest_branch(series);
            ProximityTree.Node[] children = node.get_children();

            if (branch < 0 || branch >= children.length || children[branch] == null) {
                break;
            }

            node = children[branch];
            depth++;
        }

        int leafSize = leafSize(node);

        return depth + correction(leafSize);
    }

    /**
     * Get the leaf size used for the path length correction.
     *
     * @param node terminal node
     * @return number of in-bag instances in the leaf, at least 1
     */
    private static int leafSize(ProximityTree.Node node) {

        if (node == null || node.getInBagIndices() == null) {
            return 1;
        }

        return Math.max(1, node.getInBagIndices().size());
    }

    /**
     * Standard Isolation Forest path length correction.
     *
     *     c(n) = 2 H(n - 1) - 2(n - 1)/n
     *
     * @param n sample size
     * @return expected path length correction
     */
    public static double correction(int n) {

        if (n <= 1) {
            return 0.0;
        }

        if (n == 2) {
            return 1.0;
        }

        return 2.0 * harmonic(n - 1) - (2.0 * (n - 1) / n);
    }

    private static double harmonic(int n) {

        double sum = 0.0;

        for (int i = 1; i <= n; i++) {
            sum += 1.0 / i;
        }

        return sum;
    }
}