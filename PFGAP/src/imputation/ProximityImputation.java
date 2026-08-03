package imputation;

import core.AppContext;
import datasets.ListObjectDataset;
import distance.MEASURE;
import proximities.DTWPFImpute;
import proximities.PFImpute;
import trees.ProximityForest;

public final class ProximityImputation {

    public static final String IMPUTE_FIRST = "impute_first";
    public static final String PROXIMITY_FIRST = "proximity_first";

    public static final String GAP_STANDARD = "standard";
    public static final String GAP_DTW_ALIGNMENT = "dtw_alignment";

    private static final double EPSILON = 1e-6;

    private ProximityImputation() {
    }

    @FunctionalInterface
    public interface TrainProximityComputer {
        void compute(ProximityForest forest, ListObjectDataset trainData) throws Exception;
    }

    @FunctionalInterface
    public interface TestTrainProximityComputer {
        void compute(
                ProximityForest forest,
                ListObjectDataset testData,
                ListObjectDataset trainData
        ) throws Exception;
    }

    public static void imputeTraining(
            ListObjectDataset trainData,
            int repetition,
            TrainProximityComputer trainProximityComputer
    ) throws Exception {

        if (!AppContext.perform_train_imputation) {
            return;
        }

        if (!shouldRunNumericImputation(trainData)) {
            return;
        }

        String strategy = getInitializationStrategy();

        if (IMPUTE_FIRST.equals(strategy)) {
            imputeTrainingImputeFirst(
                    trainData,
                    repetition,
                    trainProximityComputer
            );
            return;
        }

        if (PROXIMITY_FIRST.equals(strategy)) {
            imputeTrainingProximityFirst(
                    trainData,
                    repetition,
                    trainProximityComputer
            );
            return;
        }

        throw new IllegalArgumentException(
                "Unknown imputation initialization strategy: " + strategy
        );
    }

    public static void imputeTesting(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            ProximityForest trainedForest,
            TestTrainProximityComputer testTrainProximityComputer
    ) throws Exception {

        if (!AppContext.perform_test_imputation) {
            return;
        }

        if (!shouldRunNumericImputation(testData)) {
            return;
        }

        String strategy = getInitializationStrategy();

        if (IMPUTE_FIRST.equals(strategy)) {
            imputeTestingImputeFirst(
                    testData,
                    trainData,
                    trainedForest,
                    testTrainProximityComputer
            );
            return;
        }

        if (PROXIMITY_FIRST.equals(strategy)) {
            imputeTestingProximityFirst(
                    testData,
                    trainData,
                    trainedForest,
                    testTrainProximityComputer
            );
            return;
        }

        throw new IllegalArgumentException(
                "Unknown imputation initialization strategy: " + strategy
        );
    }

    private static void imputeTrainingImputeFirst(
            ListObjectDataset trainData,
            int repetition,
            TrainProximityComputer trainProximityComputer
    ) throws Exception {

        if (AppContext.verbosity > 0) {
            System.out.println("Imputing the training set using impute-first strategy...");
            System.out.println("Performing initial imputation...");
        }

        AppContext.initial_imputer.Impute(trainData);

        for (int iteration = 0; iteration < AppContext.numImputes; iteration++) {

            if (AppContext.verbosity > 0) {
                System.out.println(
                        "Training imputation iteration "
                                + (iteration + 1)
                                + " of "
                                + AppContext.numImputes
                                + " using normal model distances..."
                );
            }

            ProximityForest forest = new ProximityForest(
                    repetition,
                    AppContext.userdistances
            );

            forest.train(trainData);
            trainProximityComputer.compute(forest, trainData);
            updateTrainingValues(trainData);
        }

        if (AppContext.verbosity > 0) {
            System.out.println("Done imputing the training set.");
        }
    }

    private static void imputeTrainingProximityFirst(
            ListObjectDataset trainData,
            int repetition,
            TrainProximityComputer trainProximityComputer
    ) throws Exception {

        requirePositiveIterationsForProximityFirst();

        if (AppContext.verbosity > 0) {
            System.out.println("Imputing the training set using proximity-first strategy...");
        }

        for (int iteration = 0; iteration < AppContext.numImputes; iteration++) {

            boolean firstPass = iteration == 0;

            MEASURE[] distances = firstPass
                    ? getMissingProximityDistances()
                    : AppContext.userdistances;

            if (AppContext.verbosity > 0) {
                String distanceMessage = firstPass
                        ? "using missing-compatible proximity distances..."
                        : "using normal model distances...";

                System.out.println(
                        "Training imputation iteration "
                                + (iteration + 1)
                                + " of "
                                + AppContext.numImputes
                                + " "
                                + distanceMessage
                );
            }

            ProximityForest forest = new ProximityForest(
                    repetition,
                    distances
            );

            forest.train(trainData);
            trainProximityComputer.compute(forest, trainData);
            updateTrainingValues(trainData);
        }

        if (AppContext.verbosity > 0) {
            System.out.println("Done imputing the training set.");
        }
    }

    private static void imputeTestingImputeFirst(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            ProximityForest trainedForest,
            TestTrainProximityComputer testTrainProximityComputer
    ) throws Exception {

        if (AppContext.verbosity > 0) {
            System.out.println("Imputing the testing set using impute-first strategy...");
            System.out.println("Performing initial imputation...");
        }

        AppContext.initial_imputer.Impute(testData);

        for (int iteration = 0; iteration < AppContext.numImputes; iteration++) {

            if (AppContext.verbosity > 0) {
                System.out.println(
                        "Testing imputation iteration "
                                + (iteration + 1)
                                + " of "
                                + AppContext.numImputes
                                + " using normal trained forest..."
                );
            }

            testTrainProximityComputer.compute(
                    trainedForest,
                    testData,
                    trainData
            );

            updateTestingValues(testData, trainData);
        }

        if (AppContext.verbosity > 0) {
            System.out.println("Done imputing the testing set.");
        }
    }

    private static void imputeTestingProximityFirst(
            ListObjectDataset testData,
            ListObjectDataset trainData,
            ProximityForest trainedForest,
            TestTrainProximityComputer testTrainProximityComputer
    ) throws Exception {

        requirePositiveIterationsForProximityFirst();

        if (AppContext.verbosity > 0) {
            System.out.println("Imputing the testing set using proximity-first strategy...");
        }

        for (int iteration = 0; iteration < AppContext.numImputes; iteration++) {

            boolean firstPass = iteration == 0;

            ProximityForest forestForProximities;

            if (firstPass) {

                if (AppContext.verbosity > 0) {
                    System.out.println(
                            "Testing imputation iteration "
                                    + (iteration + 1)
                                    + " of "
                                    + AppContext.numImputes
                                    + " using missing-compatible proximity distances..."
                    );
                }

                forestForProximities = new ProximityForest(
                        0,
                        getMissingProximityDistances()
                );

                forestForProximities.train(trainData);

            } else {

                if (AppContext.verbosity > 0) {
                    System.out.println(
                            "Testing imputation iteration "
                                    + (iteration + 1)
                                    + " of "
                                    + AppContext.numImputes
                                    + " using normal trained forest..."
                    );
                }

                forestForProximities = trainedForest;
            }

            testTrainProximityComputer.compute(
                    forestForProximities,
                    testData,
                    trainData
            );

            updateTestingValues(testData, trainData);
        }

        if (AppContext.verbosity > 0) {
            System.out.println("Done imputing the testing set.");
        }
    }

    private static void updateTrainingValues(
            ListObjectDataset trainData
    ) {

        if (usesDTWAlignmentUpdate()) {

            refreshTrainingSparseProximitiesForAlignment();

            DTWPFImpute.buildAlignmentPathCache(
                    trainData,
                    trainData,
                    AppContext.training_proximities_sparse,
                    AppContext.is2D,
                    -1
            );

            DTWPFImpute.trainNumericImpute(trainData);

        } else {

            PFImpute.trainNumericImpute(trainData);
        }
    }

    private static void updateTestingValues(
            ListObjectDataset testData,
            ListObjectDataset trainData
    ) {

        if (usesDTWAlignmentUpdate()) {

            refreshTestingTrainingSparseProximitiesForAlignment();

            DTWPFImpute.buildAlignmentPathCache(
                    testData,
                    trainData,
                    AppContext.testing_training_proximities_sparse,
                    AppContext.is2D,
                    -1
            );

            DTWPFImpute.testNumericImpute(testData, trainData);

        } else {

            PFImpute.testNumericImpute(testData, trainData);
        }
    }

    private static boolean shouldRunNumericImputation(
            ListObjectDataset data
    ) {

        return data != null
                && AppContext.hasMissingValues
                && AppContext.isNumeric
                && data.getMissingIndices() != null;
    }

    private static String getInitializationStrategy() {

        String strategy = AppContext.imputation_initialization_strategy;

        if (strategy == null || strategy.trim().isEmpty()) {
            return IMPUTE_FIRST;
        }

        return strategy.trim().toLowerCase();
    }

    private static boolean usesDTWAlignmentUpdate() {

        String strategy = AppContext.gap_update_strategy;

        if (strategy == null || strategy.trim().isEmpty()) {
            return AppContext.DTWImpute;
        }

        return GAP_DTW_ALIGNMENT.equals(strategy.trim().toLowerCase());
    }

    private static MEASURE[] getMissingProximityDistances() {

        if (AppContext.missing_proximity_distances != null
                && AppContext.missing_proximity_distances.length > 0) {
            return AppContext.missing_proximity_distances;
        }

        if (AppContext.is2D) {
            return new MEASURE[]{MEASURE.nan_euclidean_i};
        }

        return new MEASURE[]{MEASURE.nan_euclidean};
    }

    private static void requirePositiveIterationsForProximityFirst() {

        if (AppContext.numImputes <= 0) {
            throw new IllegalArgumentException(
                    "proximity_first imputation requires AppContext.numImputes > 0."
            );
        }
    }

    private static void refreshTrainingSparseProximitiesForAlignment() {

        if (AppContext.useSparseProximities) {

            if (AppContext.training_proximities_sparse == null) {
                throw new IllegalStateException(
                        "Sparse training proximities are not available."
                );
            }

            return;
        }

        if (AppContext.training_proximities == null) {
            throw new IllegalStateException(
                    "Dense training proximities are not available."
            );
        }

        AppContext.training_proximities_sparse =
                PFImpute.buildSparseProximityMap(
                        AppContext.training_proximities,
                        EPSILON
                );
    }

    private static void refreshTestingTrainingSparseProximitiesForAlignment() {

        if (AppContext.useSparseProximities) {

            if (AppContext.testing_training_proximities_sparse == null) {
                throw new IllegalStateException(
                        "Sparse testing-training proximities are not available."
                );
            }

            return;
        }

        if (AppContext.testing_training_proximities == null) {
            throw new IllegalStateException(
                    "Dense testing-training proximities are not available."
            );
        }

        AppContext.testing_training_proximities_sparse =
                PFImpute.buildSparseProximityMap(
                        AppContext.testing_training_proximities,
                        EPSILON
                );
    }

    public static String missingProximityDistancesToString() {

        MEASURE[] distances = getMissingProximityDistances();

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < distances.length; i++) {

            if (i > 0) {
                sb.append(",");
            }

            sb.append(distances[i]);
        }

        return sb.toString();
    }
}