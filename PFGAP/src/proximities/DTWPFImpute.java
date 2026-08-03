package proximities;

import core.AppContext;
import datasets.ListObjectDataset;
import distance.elastic.DTWWithPath;
import distance.missing.DTWAROW;
import distance.missing.DTWAROW_D;
import distance.multiTS.DTW_D;
import imputation.MissingIndices;
import util.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DTWPFImpute {

    private static final double EPSILON = 1e-6;

    public static Map<Integer, Map<Integer, List<Pair<Integer, Integer>>>> alignmentPaths;

    public static void trainNumericImpute(ListObjectDataset dataToUpdate) {

        List<Object> rawData = dataToUpdate.getData();
        MissingIndices missingIndices = dataToUpdate.getMissingIndices();

        if (rawData == null || rawData.isEmpty() || missingIndices == null) {
            return;
        }

        if (missingIndices.is2D()) {
            trainNumericImpute2D(dataToUpdate);
        } else {
            trainNumericImpute1D(dataToUpdate);
        }
    }

    public static void testNumericImpute(
            ListObjectDataset testData,
            ListObjectDataset trainData) {

        List<Object> testRaw = testData.getData();
        MissingIndices testMissing = testData.getMissingIndices();

        if (testRaw == null || testRaw.isEmpty() || testMissing == null) {
            return;
        }

        if (testMissing.is2D()) {
            testNumericImpute2D(testData, trainData);
        } else {
            testNumericImpute1D(testData, trainData);
        }
    }

    private static void trainNumericImpute1D(ListObjectDataset dataToUpdate) {

        List<Object> rawData = dataToUpdate.getData();
        List<List<Integer>> missingIndices1D =
                dataToUpdate.getMissingIndices().indices1D;

        double[] fallbackMeans = computeObservedMeans1D(
                rawData,
                missingIndices1D
        );

        List<Object> updated = IntStream.range(0, rawData.size())
                .parallel()
                .mapToObj(targetIndex -> {

                    Object targetSeries = rawData.get(targetIndex);
                    double[] updatedRow = copySeries1DToPrimitive(targetSeries);

                    List<Integer> missing = missingIndices1D.get(targetIndex);
                    Map<Integer, Double> neighbors =
                            getTrainingNeighborWeights(targetIndex);

                    for (int targetTime : missing) {

                        double weightedSum = 0.0;
                        double totalWeight = 0.0;

                        for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {

                            int neighborIndex = entry.getKey();

                            if (neighborIndex == targetIndex) {
                                continue;
                            }

                            double weight = entry.getValue();

                            if (weight <= EPSILON) {
                                continue;
                            }

                            Object neighborSeries = rawData.get(neighborIndex);
                            List<Pair<Integer, Integer>> path =
                                    getAlignmentPath(targetIndex, neighborIndex);

                            for (Pair<Integer, Integer> pair : path) {

                                if (pair.getKey() != targetTime) {
                                    continue;
                                }

                                int alignedTime = pair.getValue();

                                if (isMissing1D(
                                        missingIndices1D,
                                        neighborIndex,
                                        alignedTime
                                )) {
                                    continue;
                                }

                                if (!hasIndex1D(neighborSeries, alignedTime)) {
                                    continue;
                                }

                                double neighborValue =
                                        getNumericValue1D(
                                                neighborSeries,
                                                alignedTime
                                        );

                                if (Double.isNaN(neighborValue)) {
                                    continue;
                                }

                                weightedSum += weight * neighborValue;
                                totalWeight += weight;
                            }
                        }

                        updatedRow[targetTime] =
                                totalWeight > 0.0
                                        ? weightedSum / totalWeight
                                        : fallbackValue1D(
                                        updatedRow,
                                        targetTime,
                                        fallbackMeans
                                );
                    }

                    return (Object) updatedRow;
                })
                .collect(Collectors.toList());

        dataToUpdate.setData(updated);
    }

    private static void trainNumericImpute2D(ListObjectDataset dataToUpdate) {

        List<Object> rawData = dataToUpdate.getData();
        List<List<List<Integer>>> missingIndices2D =
                dataToUpdate.getMissingIndices().indices2D;

        double[][] fallbackMeans = computeObservedMeans2D(
                rawData,
                missingIndices2D
        );

        List<Object> updated = IntStream.range(0, rawData.size())
                .parallel()
                .mapToObj(targetIndex -> {

                    Object targetSeries = rawData.get(targetIndex);
                    double[][] updatedMatrix =
                            copySeries2DToPrimitive(targetSeries);

                    List<List<Integer>> missingRows =
                            missingIndices2D.get(targetIndex);

                    Map<Integer, Double> neighbors =
                            getTrainingNeighborWeights(targetIndex);

                    for (int dim = 0; dim < missingRows.size(); dim++) {

                        if (dim >= updatedMatrix.length) {
                            continue;
                        }

                        List<Integer> missing = missingRows.get(dim);

                        for (int targetTime : missing) {

                            double weightedSum = 0.0;
                            double totalWeight = 0.0;

                            for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {

                                int neighborIndex = entry.getKey();

                                if (neighborIndex == targetIndex) {
                                    continue;
                                }

                                double weight = entry.getValue();

                                if (weight <= EPSILON) {
                                    continue;
                                }

                                Object neighborSeries = rawData.get(neighborIndex);
                                List<Pair<Integer, Integer>> path =
                                        getAlignmentPath(targetIndex, neighborIndex);

                                for (Pair<Integer, Integer> pair : path) {

                                    if (pair.getKey() != targetTime) {
                                        continue;
                                    }

                                    int alignedTime = pair.getValue();

                                    if (isMissing2D(
                                            missingIndices2D,
                                            neighborIndex,
                                            dim,
                                            alignedTime
                                    )) {
                                        continue;
                                    }

                                    if (!hasIndex2D(
                                            neighborSeries,
                                            dim,
                                            alignedTime
                                    )) {
                                        continue;
                                    }

                                    double neighborValue =
                                            getNumericValue2D(
                                                    neighborSeries,
                                                    dim,
                                                    alignedTime
                                            );

                                    if (Double.isNaN(neighborValue)) {
                                        continue;
                                    }

                                    weightedSum += weight * neighborValue;
                                    totalWeight += weight;
                                }
                            }

                            updatedMatrix[dim][targetTime] =
                                    totalWeight > 0.0
                                            ? weightedSum / totalWeight
                                            : fallbackValue2D(
                                            updatedMatrix,
                                            dim,
                                            targetTime,
                                            fallbackMeans
                                    );
                        }
                    }

                    return (Object) updatedMatrix;
                })
                .collect(Collectors.toList());

        dataToUpdate.setData(updated);
    }

    private static void testNumericImpute1D(
            ListObjectDataset testData,
            ListObjectDataset trainData) {

        List<Object> testRaw = testData.getData();
        List<Object> trainRaw = trainData.getData();

        List<List<Integer>> testMissing1D =
                testData.getMissingIndices().indices1D;

        List<List<Integer>> trainMissing1D =
                trainData.getMissingIndices() == null
                        ? null
                        : trainData.getMissingIndices().indices1D;

        double[] fallbackMeans =
                computeObservedMeans1D(trainRaw, trainMissing1D);

        List<Object> updated = IntStream.range(0, testRaw.size())
                .parallel()
                .mapToObj(testIndex -> {

                    Object targetSeries = testRaw.get(testIndex);
                    double[] updatedRow =
                            copySeries1DToPrimitive(targetSeries);

                    List<Integer> missing =
                            testMissing1D.get(testIndex);

                    Map<Integer, Double> neighbors =
                            getTestingTrainingNeighborWeights(testIndex);

                    for (int targetTime : missing) {

                        double weightedSum = 0.0;
                        double totalWeight = 0.0;

                        for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {

                            int trainIndex = entry.getKey();
                            double weight = entry.getValue();

                            if (weight <= EPSILON) {
                                continue;
                            }

                            Object trainSeries = trainRaw.get(trainIndex);
                            List<Pair<Integer, Integer>> path =
                                    getAlignmentPath(testIndex, trainIndex);

                            for (Pair<Integer, Integer> pair : path) {

                                if (pair.getKey() != targetTime) {
                                    continue;
                                }

                                int alignedTime = pair.getValue();

                                if (isMissing1D(
                                        trainMissing1D,
                                        trainIndex,
                                        alignedTime
                                )) {
                                    continue;
                                }

                                if (!hasIndex1D(trainSeries, alignedTime)) {
                                    continue;
                                }

                                double trainValue =
                                        getNumericValue1D(
                                                trainSeries,
                                                alignedTime
                                        );

                                if (Double.isNaN(trainValue)) {
                                    continue;
                                }

                                weightedSum += weight * trainValue;
                                totalWeight += weight;
                            }
                        }

                        updatedRow[targetTime] =
                                totalWeight > 0.0
                                        ? weightedSum / totalWeight
                                        : fallbackValue1D(
                                        updatedRow,
                                        targetTime,
                                        fallbackMeans
                                );
                    }

                    return (Object) updatedRow;
                })
                .collect(Collectors.toList());

        testData.setData(updated);
    }

    private static void testNumericImpute2D(
            ListObjectDataset testData,
            ListObjectDataset trainData) {

        List<Object> testRaw = testData.getData();
        List<Object> trainRaw = trainData.getData();

        List<List<List<Integer>>> testMissing2D =
                testData.getMissingIndices().indices2D;

        List<List<List<Integer>>> trainMissing2D =
                trainData.getMissingIndices() == null
                        ? null
                        : trainData.getMissingIndices().indices2D;

        double[][] fallbackMeans =
                computeObservedMeans2D(trainRaw, trainMissing2D);

        List<Object> updated = IntStream.range(0, testRaw.size())
                .parallel()
                .mapToObj(testIndex -> {

                    Object targetSeries = testRaw.get(testIndex);
                    double[][] updatedMatrix =
                            copySeries2DToPrimitive(targetSeries);

                    List<List<Integer>> missingRows =
                            testMissing2D.get(testIndex);

                    Map<Integer, Double> neighbors =
                            getTestingTrainingNeighborWeights(testIndex);

                    for (int dim = 0; dim < missingRows.size(); dim++) {

                        if (dim >= updatedMatrix.length) {
                            continue;
                        }

                        List<Integer> missing = missingRows.get(dim);

                        for (int targetTime : missing) {

                            double weightedSum = 0.0;
                            double totalWeight = 0.0;

                            for (Map.Entry<Integer, Double> entry : neighbors.entrySet()) {

                                int trainIndex = entry.getKey();
                                double weight = entry.getValue();

                                if (weight <= EPSILON) {
                                    continue;
                                }

                                Object trainSeries = trainRaw.get(trainIndex);
                                List<Pair<Integer, Integer>> path =
                                        getAlignmentPath(testIndex, trainIndex);

                                for (Pair<Integer, Integer> pair : path) {

                                    if (pair.getKey() != targetTime) {
                                        continue;
                                    }

                                    int alignedTime = pair.getValue();

                                    if (isMissing2D(
                                            trainMissing2D,
                                            trainIndex,
                                            dim,
                                            alignedTime
                                    )) {
                                        continue;
                                    }

                                    if (!hasIndex2D(
                                            trainSeries,
                                            dim,
                                            alignedTime
                                    )) {
                                        continue;
                                    }

                                    double trainValue =
                                            getNumericValue2D(
                                                    trainSeries,
                                                    dim,
                                                    alignedTime
                                            );

                                    if (Double.isNaN(trainValue)) {
                                        continue;
                                    }

                                    weightedSum += weight * trainValue;
                                    totalWeight += weight;
                                }
                            }

                            updatedMatrix[dim][targetTime] =
                                    totalWeight > 0.0
                                            ? weightedSum / totalWeight
                                            : fallbackValue2D(
                                            updatedMatrix,
                                            dim,
                                            targetTime,
                                            fallbackMeans
                                    );
                        }
                    }

                    return (Object) updatedMatrix;
                })
                .collect(Collectors.toList());

        testData.setData(updated);
    }

    public static void buildAlignmentPathCache(
            ListObjectDataset dataA,
            ListObjectDataset dataB,
            Map<Integer, Map<Integer, Double>> sparseProximities,
            boolean is2D,
            int windowSize
    ) {

        DTWWithPath dtw1D = new DTWWithPath();
        DTW_D dtw2D = new DTW_D();

        DTWAROW dtwArow1D = new DTWAROW();
        DTWAROW_D dtwArow2D = new DTWAROW_D();

        alignmentPaths = new HashMap<>();

        if (dataA == null || dataB == null || sparseProximities == null) {
            return;
        }

        int sizeA = dataA.size();

        for (int i = 0; i < sizeA; i++) {

            Map<Integer, Double> row = sparseProximities.get(i);

            if (row == null || row.isEmpty()) {
                continue;
            }

            for (Map.Entry<Integer, Double> entry : row.entrySet()) {

                int j = entry.getKey();
                double proximity = entry.getValue();

                if (i == j && dataA == dataB) {
                    continue;
                }

                if (proximity <= EPSILON) {
                    continue;
                }

                if (j < 0 || j >= dataB.size()) {
                    continue;
                }

                Object s1 = dataA.get_series(i);
                Object s2 = dataB.get_series(j);

                List<Pair<Integer, Integer>> path;

                if (is2D) {
                    if (containsMissing(s1) || containsMissing(s2)) {
                        path = dtwArow2D.getAlignmentPath(
                                s1,
                                s2,
                                windowSize
                        );
                    } else {
                        path = dtw2D.getAlignmentPath(
                                copySeries2DToPrimitive(s1),
                                copySeries2DToPrimitive(s2),
                                windowSize
                        );
                    }
                } else {
                    if (containsMissing(s1) || containsMissing(s2)) {
                        path = dtwArow1D.getAlignmentPath(
                                s1,
                                s2,
                                windowSize
                        );
                    } else {
                        path = dtw1D.getAlignmentPath(
                                copySeries1DToPrimitive(s1),
                                copySeries1DToPrimitive(s2),
                                windowSize
                        );
                    }
                }

                alignmentPaths
                        .computeIfAbsent(i, key -> new HashMap<>())
                        .put(j, path);
            }
        }
    }

    private static List<Pair<Integer, Integer>> getAlignmentPath(
            int i,
            int j) {

        if (alignmentPaths == null) {
            return Collections.emptyList();
        }

        return alignmentPaths
                .getOrDefault(i, Collections.emptyMap())
                .getOrDefault(j, Collections.emptyList());
    }

    private static Map<Integer, Double> getTrainingNeighborWeights(
            int targetIndex) {

        if (AppContext.useSparseProximities) {
            if (AppContext.training_proximities_sparse == null) {
                return Collections.emptyMap();
            }

            return AppContext.training_proximities_sparse.getOrDefault(
                    targetIndex,
                    Collections.emptyMap()
            );
        }

        if (AppContext.training_proximities == null
                || targetIndex >= AppContext.training_proximities.length) {
            return Collections.emptyMap();
        }

        return convertDenseRowToMap(
                AppContext.training_proximities[targetIndex]
        );
    }

    private static Map<Integer, Double> getTestingTrainingNeighborWeights(
            int testIndex) {

        if (AppContext.useSparseProximities) {
            if (AppContext.testing_training_proximities_sparse == null) {
                return Collections.emptyMap();
            }

            return AppContext.testing_training_proximities_sparse.getOrDefault(
                    testIndex,
                    Collections.emptyMap()
            );
        }

        if (AppContext.testing_training_proximities == null
                || testIndex >= AppContext.testing_training_proximities.length) {
            return Collections.emptyMap();
        }

        return convertDenseRowToMap(
                AppContext.testing_training_proximities[testIndex]
        );
    }

    private static boolean isMissing1D(
            List<List<Integer>> missingIndices,
            int seriesIndex,
            int featureIndex) {

        if (missingIndices == null) {
            return false;
        }

        if (seriesIndex < 0 || seriesIndex >= missingIndices.size()) {
            return true;
        }

        return missingIndices.get(seriesIndex).contains(featureIndex);
    }

    private static boolean isMissing2D(
            List<List<List<Integer>>> missingIndices,
            int seriesIndex,
            int dim,
            int featureIndex) {

        if (missingIndices == null) {
            return false;
        }

        if (seriesIndex < 0 || seriesIndex >= missingIndices.size()) {
            return true;
        }

        List<List<Integer>> instanceMissing =
                missingIndices.get(seriesIndex);

        if (dim < 0 || dim >= instanceMissing.size()) {
            return true;
        }

        return instanceMissing.get(dim).contains(featureIndex);
    }

    private static boolean hasIndex1D(
            Object series,
            int featureIndex) {

        if (series instanceof double[]) {
            return featureIndex >= 0
                    && featureIndex < ((double[]) series).length;
        }

        if (series instanceof Object[]) {
            return featureIndex >= 0
                    && featureIndex < ((Object[]) series).length;
        }

        return false;
    }

    private static boolean hasIndex2D(
            Object series,
            int dim,
            int featureIndex) {

        if (series instanceof double[][]) {
            double[][] matrix = (double[][]) series;

            return dim >= 0
                    && dim < matrix.length
                    && featureIndex >= 0
                    && featureIndex < matrix[dim].length;
        }

        if (series instanceof Object[][]) {
            Object[][] matrix = (Object[][]) series;

            return dim >= 0
                    && dim < matrix.length
                    && featureIndex >= 0
                    && featureIndex < matrix[dim].length;
        }

        return false;
    }

    private static double getNumericValue1D(
            Object series,
            int featureIndex) {

        if (series instanceof double[]) {
            return ((double[]) series)[featureIndex];
        }

        if (series instanceof Object[]) {
            return objectToDouble(((Object[]) series)[featureIndex]);
        }

        throw new IllegalArgumentException(
                "Unsupported 1D numeric series type: "
                        + series.getClass().getName()
        );
    }

    private static double getNumericValue2D(
            Object series,
            int dim,
            int featureIndex) {

        if (series instanceof double[][]) {
            return ((double[][]) series)[dim][featureIndex];
        }

        if (series instanceof Object[][]) {
            return objectToDouble(((Object[][]) series)[dim][featureIndex]);
        }

        throw new IllegalArgumentException(
                "Unsupported 2D numeric series type: "
                        + series.getClass().getName()
        );
    }

    private static double objectToDouble(Object value) {

        if (value == null) {
            return Double.NaN;
        }

        if (!(value instanceof Number)) {
            throw new IllegalArgumentException(
                    "Expected numeric value but found "
                            + value.getClass().getName()
            );
        }

        double numericValue = ((Number) value).doubleValue();

        return Double.isNaN(numericValue)
                ? Double.NaN
                : numericValue;
    }

    private static double[] copySeries1DToPrimitive(Object series) {

        if (series instanceof double[]) {
            return Arrays.copyOf(
                    (double[]) series,
                    ((double[]) series).length
            );
        }

        if (series instanceof Object[]) {
            Object[] row = (Object[]) series;
            double[] copied = new double[row.length];

            for (int i = 0; i < row.length; i++) {
                copied[i] = objectToDouble(row[i]);
            }

            return copied;
        }

        throw new IllegalArgumentException(
                "Unsupported 1D numeric series type: "
                        + series.getClass().getName()
        );
    }

    private static double[][] copySeries2DToPrimitive(Object series) {

        if (series instanceof double[][]) {
            double[][] matrix = (double[][]) series;
            double[][] copied = new double[matrix.length][];

            for (int i = 0; i < matrix.length; i++) {
                copied[i] = Arrays.copyOf(matrix[i], matrix[i].length);
            }

            return copied;
        }

        if (series instanceof Object[][]) {
            Object[][] matrix = (Object[][]) series;
            double[][] copied = new double[matrix.length][];

            for (int i = 0; i < matrix.length; i++) {
                copied[i] = new double[matrix[i].length];

                for (int j = 0; j < matrix[i].length; j++) {
                    copied[i][j] = objectToDouble(matrix[i][j]);
                }
            }

            return copied;
        }

        throw new IllegalArgumentException(
                "Unsupported 2D numeric series type: "
                        + series.getClass().getName()
        );
    }

    private static boolean containsMissing(Object series) {

        if (series instanceof double[][]) {
            double[][] matrix = (double[][]) series;

            for (double[] row : matrix) {
                for (double value : row) {
                    if (Double.isNaN(value)) {
                        return true;
                    }
                }
            }

            return false;
        }

        if (series instanceof double[]) {
            double[] row = (double[]) series;

            for (double value : row) {
                if (Double.isNaN(value)) {
                    return true;
                }
            }

            return false;
        }

        if (series instanceof Object[][]) {
            Object[][] matrix = (Object[][]) series;

            for (Object[] row : matrix) {
                for (Object value : row) {
                    if (isMissingObject(value)) {
                        return true;
                    }
                }
            }

            return false;
        }

        if (series instanceof Object[]) {
            Object[] row = (Object[]) series;

            for (Object value : row) {
                if (isMissingObject(value)) {
                    return true;
                }
            }

            return false;
        }

        return false;
    }

    private static boolean isMissingObject(Object value) {

        if (value == null) {
            return true;
        }

        if (value instanceof Double) {
            return Double.isNaN((Double) value);
        }

        if (value instanceof Float) {
            return Float.isNaN((Float) value);
        }

        if (value instanceof Number) {
            return Double.isNaN(((Number) value).doubleValue());
        }

        return false;
    }

    private static double fallbackValue1D(
            double[] row,
            int featureIndex,
            double[] fallbackMeans) {

        if (featureIndex >= 0
                && featureIndex < row.length
                && !Double.isNaN(row[featureIndex])) {
            return row[featureIndex];
        }

        if (featureIndex >= 0
                && featureIndex < fallbackMeans.length
                && !Double.isNaN(fallbackMeans[featureIndex])) {
            return fallbackMeans[featureIndex];
        }

        return 0.0;
    }

    private static double fallbackValue2D(
            double[][] matrix,
            int dim,
            int featureIndex,
            double[][] fallbackMeans) {

        if (dim >= 0
                && dim < matrix.length
                && featureIndex >= 0
                && featureIndex < matrix[dim].length
                && !Double.isNaN(matrix[dim][featureIndex])) {
            return matrix[dim][featureIndex];
        }

        if (dim >= 0
                && dim < fallbackMeans.length
                && featureIndex >= 0
                && featureIndex < fallbackMeans[dim].length
                && !Double.isNaN(fallbackMeans[dim][featureIndex])) {
            return fallbackMeans[dim][featureIndex];
        }

        return 0.0;
    }

    private static double[] computeObservedMeans1D(
            List<Object> data,
            List<List<Integer>> missingIndices) {

        int maxLength = 0;

        for (Object series : data) {
            maxLength = Math.max(maxLength, length1D(series));
        }

        double[] sums = new double[maxLength];
        int[] counts = new int[maxLength];

        for (int i = 0; i < data.size(); i++) {

            Object series = data.get(i);
            int length = length1D(series);

            for (int k = 0; k < length; k++) {

                if (isMissing1D(missingIndices, i, k)) {
                    continue;
                }

                double value = getNumericValue1D(series, k);

                if (Double.isNaN(value)) {
                    continue;
                }

                sums[k] += value;
                counts[k]++;
            }
        }

        double[] means = new double[maxLength];

        for (int k = 0; k < maxLength; k++) {
            means[k] = counts[k] > 0
                    ? sums[k] / counts[k]
                    : Double.NaN;
        }

        return means;
    }

    private static double[][] computeObservedMeans2D(
            List<Object> data,
            List<List<List<Integer>>> missingIndices) {

        int maxDims = 0;

        for (Object series : data) {
            maxDims = Math.max(maxDims, dimensions2D(series));
        }

        int[] maxLengths = new int[maxDims];

        for (Object series : data) {
            for (int dim = 0; dim < dimensions2D(series); dim++) {
                maxLengths[dim] = Math.max(
                        maxLengths[dim],
                        length2D(series, dim)
                );
            }
        }

        double[][] sums = new double[maxDims][];
        int[][] counts = new int[maxDims][];

        for (int dim = 0; dim < maxDims; dim++) {
            sums[dim] = new double[maxLengths[dim]];
            counts[dim] = new int[maxLengths[dim]];
        }

        for (int i = 0; i < data.size(); i++) {

            Object series = data.get(i);

            for (int dim = 0; dim < dimensions2D(series); dim++) {
                for (int k = 0; k < length2D(series, dim); k++) {

                    if (isMissing2D(missingIndices, i, dim, k)) {
                        continue;
                    }

                    double value = getNumericValue2D(series, dim, k);

                    if (Double.isNaN(value)) {
                        continue;
                    }

                    sums[dim][k] += value;
                    counts[dim][k]++;
                }
            }
        }

        double[][] means = new double[maxDims][];

        for (int dim = 0; dim < maxDims; dim++) {
            means[dim] = new double[maxLengths[dim]];

            for (int k = 0; k < maxLengths[dim]; k++) {
                means[dim][k] =
                        counts[dim][k] > 0
                                ? sums[dim][k] / counts[dim][k]
                                : Double.NaN;
            }
        }

        return means;
    }

    private static int length1D(Object series) {

        if (series instanceof double[]) {
            return ((double[]) series).length;
        }

        if (series instanceof Object[]) {
            return ((Object[]) series).length;
        }

        return 0;
    }

    private static int dimensions2D(Object series) {

        if (series instanceof double[][]) {
            return ((double[][]) series).length;
        }

        if (series instanceof Object[][]) {
            return ((Object[][]) series).length;
        }

        return 0;
    }

    private static int length2D(Object series, int dim) {

        if (series instanceof double[][]) {
            double[][] matrix = (double[][]) series;

            if (dim < 0 || dim >= matrix.length) {
                return 0;
            }

            return matrix[dim].length;
        }

        if (series instanceof Object[][]) {
            Object[][] matrix = (Object[][]) series;

            if (dim < 0 || dim >= matrix.length) {
                return 0;
            }

            return matrix[dim].length;
        }

        return 0;
    }

    private static Map<Integer, Double> convertDenseRowToMap(double[] row) {

        Map<Integer, Double> map = new HashMap<>();

        if (row == null) {
            return map;
        }

        for (int i = 0; i < row.length; i++) {
            if (row[i] > EPSILON) {
                map.put(i, row[i]);
            }
        }

        return map;
    }
}