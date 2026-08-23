package core;

import datasets.readers.lazy.LazySeriesRef;
import datasets.ListObjectDataset;
import datasets.readers.*;
import imputation.util.MissingIndicesBuilder;
import imputation.ProximityImputation;
import org.apache.commons.lang3.ArrayUtils;
import outlier.IsolationDepthScorer;
import outlier.OutlierScorer;
import preprocessing.standardization.*;
import trees.ProximityForest;
import util.GeneralUtilities;
import util.PrintUtilities;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static proximities.PFGAP.computeTestTrainProximities;
import static proximities.PFGAP.computeTrainProximities;

public class ExperimentRunner {

	ListObjectDataset train_data;
	ListObjectDataset test_data;

	public ExperimentRunner() {
	}

	public void run(boolean eval) throws Exception {

		if (!eval) {
			runTrainingMode();
		} else {
			runEvaluationMode();
		}
	}

	private void runTrainingMode() throws Exception {

		prepareSuppliedStandardization();

		ListObjectDataset trainDataOriginal = readTrainingData();

		prepareTrainingStatisticsBeforeTestRead(trainDataOriginal);

		ListObjectDataset testDataOriginal = AppContext.testing_file != null
				? readTestData()
				: null;

		// currently we don't support imputation on lazy datasets.
		validateDatasetCapabilities(trainDataOriginal, testDataOriginal);

		/*
		* Fit statistics only from training data, then apply the same fitted
		* transformation to both training and validation/test data.
		*/
		/*applyTrainingStandardization(
				trainDataOriginal,
				testDataOriginal
		);*/

		applyPreparedStandardization(trainDataOriginal, testDataOriginal);

		train_data = prepareTrainingData(trainDataOriginal);

		if (testDataOriginal != null) {
			test_data = prepareTestingData(
					testDataOriginal,
					train_data._get_initial_class_labels()
			);
		} else {
			test_data = null;
		}

		AppContext.setTraining_data(train_data);
		AppContext.setTesting_data(test_data);

		trainDataOriginal = null;
		testDataOriginal = null;
		System.gc();

		String datasetName = inferDatasetName(AppContext.training_file);
		AppContext.setDatasetName(datasetName);

		System.out.println();

		if (AppContext.shuffle_dataset) {
			System.out.println("Shuffling the training set...");
			train_data.shuffle();
		}

		for (int repetition = 0; repetition < AppContext.num_repeats; repetition++) {

			printRepetitionHeader(repetition, datasetName);

			if (AppContext.perform_train_imputation) {
				ProximityImputation.imputeTraining(
						train_data,
						repetition,
						(forest, data) -> computeTrainProximities(forest, data)
				);
			}

			ProximityForest forest = new ProximityForest(
					repetition,
					AppContext.userdistances
			);

			forest.train(train_data);

			if (AppContext.savemodel) {
				saveModel(forest);
			}

			if (AppContext.impute_train) {
				writeTrainingData();
			}

			if (test_data != null) {
				runValidationTesting(forest, repetition, datasetName);
			}

			handleTrainingOutlierScores(forest);
			handleRequestedTrainingProximities(forest);

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}
		}
	}

	private void runEvaluationMode() throws Exception {

		ModelIO.LoadedModel loaded = ModelIO.loadModel(AppContext.modelname + ".ser");
		ModelIO.applySnapshot(loaded.snapshot);

		ProximityForest forest = loaded.forest;
		ListObjectDataset trainData = loaded.trainData;

		this.train_data = trainData;
		AppContext.setTraining_data(trainData);

		ListObjectDataset testDataOriginal = readTestData();
		// no imputation on lazy datasets, currently
		validateDatasetCapabilities(trainData, testDataOriginal);

		applyEvaluationStandardization(testDataOriginal);

		test_data = prepareTestingData(
				testDataOriginal,
				trainData._get_initial_class_labels()
		);

		AppContext.setTesting_data(test_data);

		testDataOriginal = null;
		System.gc();

		String datasetName = inferDatasetName(AppContext.training_file);
		AppContext.setDatasetName(datasetName);

		for (int repetition = 0; repetition < AppContext.num_repeats; repetition++) {

			if (AppContext.perform_test_imputation) {
				ProximityImputation.imputeTesting(
						test_data,
						trainData,
						forest,
						(forestForProx, test, train) ->
								computeTestTrainProximities(forestForProx, test, train)
				);
			}

			if (AppContext.isIsolationMode()) {
				handleIsolationScores(forest, test_data, trainData.size(), "IsolationScores_saved.txt");
				continue;
			}

			ProximityForestResult result = forest.test(test_data);

			if (!AppContext.perform_test_imputation) {
				result.printResults(datasetName, repetition, "");
			}

			if (AppContext.impute_test) {
				writeTestingData();
			}

			//if (AppContext.isIsolationMode()) {
			//	handleIsolationScores(forest, test_data, trainData.size(), "IsolationScores_saved.txt");
			//	continue;
			//}

			if (AppContext.getprox) {
				computeAndWriteTestTrainProximities(forest, trainData);
			}

			if (AppContext.get_predictions && !AppContext.isIsolationMode()) {
				writePredictions(
						result,
						test_data,
						"Predictions_saved.txt"
				);
			}

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}
		}
	}

	private ReaderOptions buildBaseReaderOptions(ReaderType readerType, String filePattern) { //(String dataPath) {

		if (readerType == null) {
			throw new IllegalArgumentException(
					"A reader type must be specified."
			);
		}

		return new ReaderOptions()
				.setReaderType(readerType)
				.setEntrySeparator(AppContext.entry_separator)
				.setArraySeparator(AppContext.array_separator)
				.setHasHeader(AppContext.csv_has_header)
				.set2D(AppContext.is2D)
				.setNumeric(AppContext.isNumeric)
				.setHasMissingValues(AppContext.hasMissingValues)
				.setTargetColumnIsFirst(AppContext.target_column_is_first)
				.setRegression(AppContext.isRegressionMode())
				.setIdColumn(AppContext.id_column)
				.setTimeColumn(AppContext.time_column)
				.setFeatureColumns(AppContext.feature_columns)
				.setLabelColumns(AppContext.label_columns)
				.setHdf5DatasetPath(AppContext.hdf5_dataset_path)
				.setHdf5LabelDatasetPath(AppContext.hdf5_label_dataset_path)
				.setFilePattern(filePattern)
				.setStandardizationStats(AppContext.standardizationStats);
	}


	private ListObjectDataset readTrainingData() throws IOException {

		ReaderOptions trainOptions =
				buildBaseReaderOptions(
						AppContext.getTrainingReaderType(),
						AppContext.getTrainingFilePattern()
				) //(AppContext.training_file)
						.setDataPath(AppContext.training_file)
						.setLabelPath(AppContext.training_labels)
						.setTest(false);

		DatasetReader trainReader =
				DatasetReaderFactory.create(trainOptions);

		return trainReader.read();
	}

	private ListObjectDataset readTestData() throws IOException {

		ReaderOptions testOptions =
				buildBaseReaderOptions(
						AppContext.getTestingReaderType(),
						AppContext.getTestingFilePattern()
				) //(AppContext.testing_file)
						.setDataPath(AppContext.testing_file)
						.setLabelPath(AppContext.testing_labels)
						.setTest(true);

		DatasetReader testReader =
				DatasetReaderFactory.create(testOptions);

		return testReader.read();
	}

	private ListObjectDataset prepareTrainingData(
			ListObjectDataset original) {

		ListObjectDataset prepared;

		if (AppContext.isClassificationMode()) {
			prepared = original.reorder_class_labels(null);
		} else {
			prepared = original;
		}

		if (AppContext.hasMissingValues && !isLazyDataset(prepared)) {
			prepared.setMissingIndices(
					MissingIndicesBuilder.buildFromDataset(prepared.getData())
			);
		}

		return prepared;
	}

	private ListObjectDataset prepareTestingData(
			ListObjectDataset original,
			Map<Object, Integer> labelMap) {

		ListObjectDataset prepared;

		if (AppContext.isClassificationMode()) {
			prepared = original.reorder_class_labels(labelMap);
		} else {
			prepared = original;
		}

		if (AppContext.hasMissingValues && !isLazyDataset(prepared)) {
			prepared.setMissingIndices(
					MissingIndicesBuilder.buildFromDataset(prepared.getData())
			);
		}

		return prepared;
	}

	private void runValidationTesting(
			ProximityForest forest,
			int repetition,
			String datasetName) throws Exception {

		if (AppContext.perform_test_imputation) {
			ProximityImputation.imputeTesting(
					test_data,
					train_data,
					forest,
					(forestForProx, test, train) ->
							computeTestTrainProximities(forestForProx, test, train)
			);
		}

		if (AppContext.isIsolationMode()) {
			handleIsolationScores(forest, test_data, train_data.size(), "IsolationScores.txt");
			return;
		}

		if (AppContext.impute_test) {
			writeTestingData();
		}

		ProximityForestResult result = forest.test(test_data);

		if (AppContext.get_predictions && !AppContext.isIsolationMode()) {
			writePredictions(
					result,
					test_data,
					"Validation_Predictions.txt"
			);
		}

		result.printResults(datasetName, repetition, "");
	}

	private void saveModel(ProximityForest forest) {

		try {
			AppContextSnapshot snapshot = AppContextUtils.captureSnapshot();

			ModelIO.saveModel(
					AppContext.output_dir + "/" + AppContext.modelname + ".ser",
					forest,
					train_data,
					snapshot
			);
		} catch (IOException e) {
			PrintUtilities.abort(e);
		}
	}

	private void writeTrainingData() throws IOException {

		GeneralUtilities.writeDelimitedData(
				train_data.getData(),
				AppContext.output_dir + AppContext.training_file,
				AppContext.array_separator,
				AppContext.entry_separator
		);
	}

	private void writeTestingData() throws IOException {

		GeneralUtilities.writeDelimitedData(
				test_data.getData(),
				AppContext.output_dir + AppContext.testing_file,
				AppContext.array_separator,
				AppContext.entry_separator
		);
	}

	private void writePredictions(
			ProximityForestResult result,
			ListObjectDataset data,
			String fileName) throws IOException {

		if (AppContext.isIsolationMode()) {
			return;
		}

		PrintWriter writer = new PrintWriter(
				AppContext.output_dir + fileName,
				StandardCharsets.UTF_8
		);

		if (!AppContext.isRegressionMode()) {
			List<Object> predictedLabels = result.Predictions;
			Map<Integer, Object> newToOriginal =
					data.invertLabelMap(data._get_initial_class_labels());

			List<Object> originalPredictions = predictedLabels.stream()
					.map(newToOriginal::get)
					.collect(Collectors.toList());

			writer.print(ArrayUtils.toString(originalPredictions));
		} else {
			writer.print(ArrayUtils.toString(result.Predictions));
		}

		writer.close();
	}

	private void handleTrainingOutlierScores(
			ProximityForest forest) throws Exception {

		if (!AppContext.get_training_outlier_scores) {
			return;
		}

		if (AppContext.isIsolationMode()) {
			handleIsolationScores(
					forest,
					train_data,
					train_data.size(),
					"outlier_scores.txt"
			);
			return;
		}

		if (AppContext.isRegressionMode()) {
			return;
		}

		boolean originalSparseSetting = AppContext.useSparseProximities;

		if (AppContext.getprox) {
			AppContext.useSparseProximities = false;
		} else {
			AppContext.useSparseProximities = true;
		}

		System.out.println("Computing Training Proximities...");
		computeTrainProximities(forest, train_data);

		System.out.println("Computing Training Outlier Scores...");
		double[] scores = OutlierScorer.getOutlierScores(
				AppContext.useSparseProximities,
				false,
				true,
				train_data._internal_class_array(),
				AppContext.training_proximities,
				AppContext.training_proximities_sparse
		);

		PrintWriter writer = new PrintWriter(
				AppContext.output_dir + "outlier_scores.txt",
				StandardCharsets.UTF_8
		);

		writer.print(ArrayUtils.toString(scores));
		writer.close();

		AppContext.useSparseProximities = originalSparseSetting;
	}

	private void handleRequestedTrainingProximities(
			ProximityForest forest) throws IOException, ExecutionException, InterruptedException {

		if (!AppContext.getprox) {
			return;
		}

		if (!AppContext.get_training_outlier_scores) {
			boolean originalSparseSetting = AppContext.useSparseProximities;
			AppContext.useSparseProximities = false;

			System.out.println("Computing Training Proximities...");
			double start = System.currentTimeMillis();

			computeTrainProximities(forest, train_data);

			double end = System.currentTimeMillis();

			System.out.print("Done Computing Training Proximities. ");
			System.out.print("Computation time: ");
			System.out.println(end - start + "ms");

			AppContext.useSparseProximities = originalSparseSetting;
		}

		PrintWriter writer = new PrintWriter(
				AppContext.output_dir + "TrainingProximities.txt",
				StandardCharsets.UTF_8
		);

		writer.print(ArrayUtils.toString(AppContext.training_proximities));
		writer.close();

		if (test_data != null) {
			computeAndWriteTestTrainProximities(
					forest,
					train_data
			);
		}
	}

	private void handleIsolationScores(
			ProximityForest forest,
			ListObjectDataset data,
			int normalizationSampleSize,
			String fileName
	) throws Exception {

		double[] scores = IsolationDepthScorer.score(
				forest,
				data,
				normalizationSampleSize
		);

		PrintWriter writer = new PrintWriter(
				AppContext.output_dir + fileName,
				StandardCharsets.UTF_8
		);

		writer.print(ArrayUtils.toString(scores));
		writer.close();
	}

	private void computeAndWriteTestTrainProximities(
			ProximityForest forest,
			ListObjectDataset trainData) throws IOException, ExecutionException, InterruptedException {

		boolean originalSparseSetting = AppContext.useSparseProximities;
		AppContext.useSparseProximities = false;

		System.out.println("Computing Test/Train Proximities...");
		double start = System.currentTimeMillis();

		computeTestTrainProximities(forest, test_data, trainData);

		double end = System.currentTimeMillis();

		System.out.print("Done Computing Test/Train Proximities. ");
		System.out.print("Computation time: ");
		System.out.println(end - start + "ms");

		PrintWriter writer = new PrintWriter(
				AppContext.output_dir + "TestTrainProximities.txt",
				StandardCharsets.UTF_8
		);

		writer.print(ArrayUtils.toString(AppContext.testing_training_proximities));
		writer.close();

		AppContext.useSparseProximities = originalSparseSetting;
	}

	private void printRepetitionHeader(
			int repetition,
			String datasetName) {

		if (AppContext.verbosity > 0) {
			System.out.println(
					"-----------------Repetition No: "
							+ (repetition + 1)
							+ " ("
							+ datasetName
							+ ")   -----------------"
			);
			PrintUtilities.printMemoryUsage();
		} else if (AppContext.verbosity == 0 && repetition == 0) {
			System.out.println(
					"Repetition, Dataset, Score, TrainingTime(ms), TestingTime(ms), MeanDepthPerTree"
			);
		}
	}

	private String inferDatasetName(String trainingFilePath) {

		File trainingFile = new File(trainingFilePath);

		return trainingFile
				.getName()
				.replaceAll("_TRAIN.txt", "");
	}

	private void validateDatasetCapabilities(
			ListObjectDataset trainData,
			ListObjectDataset testData
	) {
		boolean lazyTraining = isLazyDataset(trainData);
		boolean lazyTesting = isLazyDataset(testData);

		if (!lazyTraining && !lazyTesting) {
			return;
		}

		if (AppContext.perform_train_imputation
				|| AppContext.perform_test_imputation
				|| AppContext.impute_train
				|| AppContext.impute_test) {

			throw new UnsupportedOperationException(
					"Imputation is not currently supported for lazy datasets. "
							+ "Disable perform_train_imputation, perform_test_imputation, "
							+ "impute_train, and impute_test when using a lazy reader."
			);
		}

		StandardizationConfig standardizationConfig =
				AppContext.standardizationConfig;

		if (lazyTraining
				&& standardizationConfig != null
				&& standardizationConfig.isEnabled()
				&& AppContext.standardizationStats == null) {

			throw new UnsupportedOperationException(
					"Lazy training with standardization requires precomputed "
							+ "statistics supplied through -standardization_stats. "
							+ "Automatic lazy statistics fitting is not implemented."
			);
		}

		if (lazyTraining != lazyTesting && trainData != null && testData != null) {
			System.out.println(
					"Warning: training and testing datasets use different storage modes. "
							+ "One dataset is lazy and the other is materialized."
			);
		}
	}

	private boolean isLazyDataset(ListObjectDataset dataset) {
		if (dataset == null) {
			return false;
		}

		for (Object value : dataset.getData()) {
			if (value != null) {
				return value instanceof LazySeriesRef;
			}
		}

		return false;
	}

	// here is a stricter check (not needed if no mixing can happen)

	/*private boolean isLazyDataset(ListObjectDataset dataset) {
		if (dataset == null) {
			return false;
		}

		boolean foundLazy = false;
		boolean foundMaterialized = false;

		for (Object value : dataset.getData()) {
			if (value == null) {
				continue;
			}

			if (value instanceof LazySeriesRef) {
				foundLazy = true;
			} else {
				foundMaterialized = true;
			}

			if (foundLazy && foundMaterialized) {
				throw new IllegalStateException(
						"A ListObjectDataset cannot mix lazy references "
								+ "and materialized instances."
				);
			}
		}

		return foundLazy;
	}*/

	private void applyTrainingStandardization(
			ListObjectDataset trainingData,
			ListObjectDataset testingData
	) throws IOException {

		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			AppContext.standardizationStats =
					null;

			return;
		}

		config.requireImplemented();

		boolean lazyTraining =
				isLazyDataset(trainingData);

		boolean lazyTesting =
				isLazyDataset(testingData);

		List<String> featureNames =
				getStandardizationFeatureNames();

		StandardizationStats stats =
				AppContext.standardizationStats;

		if (config.shouldLoadStatistics()) {
			/*
			 * prepareSuppliedStandardization() already loaded and validated the
			 * statistics before the lazy or eager readers were constructed.
			 */
			if (stats == null) {
				throw new IllegalStateException(
						"Supplied standardization statistics were not loaded "
								+ "before the dataset readers were created."
				);
			}
		} else {
			if (lazyTraining) {
				throw new UnsupportedOperationException(
						"Lazy training with standardization currently requires "
								+ "precomputed statistics supplied through "
								+ "-standardization_stats. Automatic streaming fit "
								+ "for lazy training is reserved for Phase 4."
				);
			}

			stats =
					StandardizationFitter.fit(
							trainingData,
							config.getMethod(),
							config.getScope(),
							config.getVarianceConvention(),
							featureNames
					);

			config.validateStatistics(stats);

			AppContext.standardizationStats =
					stats;

			if (AppContext.verbosity > 0) {
				System.out.println(
						"Fitted standardization statistics from "
								+ "the eager training dataset."
				);
			}
		}

		/*
		 * Eager datasets must be transformed here.
		 *
		 * Lazy datasets are transformed by their PerFile*SeriesReader each time
		 * a reference is materialized.
		 */
		if (!lazyTraining) {
			Standardizer.transformInPlace(
					trainingData,
					stats,
					featureNames
			);
		}

		if (testingData != null && !lazyTesting) {
			Standardizer.transformInPlace(
					testingData,
					stats,
					featureNames
			);
		}

		if (config.shouldSaveFittedStatistics()) {
			Path outputPath =
					resolveStandardizationOutputPath(config);

			StandardizationJson.write(
					outputPath,
					stats
			);

			if (AppContext.verbosity > 0) {
				System.out.println(
						"Saved standardization statistics to: "
								+ outputPath
				);
			}
		}

		if (AppContext.verbosity > 0
				&& !config.shouldLoadStatistics()) {

			printStandardizationSummary(stats);
		}
	}

	private Path resolveStandardizationOutputPath(
			StandardizationConfig config
	) {
		String configuredPath =
				config.getStatisticsOutputPath();

		if (configuredPath != null
				&& !configuredPath.isBlank()) {

			return Paths.get(
					configuredPath
			);
		}

		return Paths.get(
				AppContext.output_dir,
				"standardization_stats.json"
		);
	}

	private List<String> getStandardizationFeatureNames() {
		if (AppContext.feature_columns == null
				|| AppContext.feature_columns.isEmpty()) {

			return List.of();
		}

		return new java.util.ArrayList<>(
				AppContext.feature_columns
		);
	}

	private void printStandardizationSummary(
			StandardizationStats stats
	) {
		System.out.println(
				"Applied "
						+ stats.getMethod()
						+ " standardization with scope "
						+ stats.getScope()
						+ " using "
						+ stats.getStatisticGroupCount()
						+ " fitted statistic group(s)."
		);

		if (AppContext.verbosity > 1) {
			System.out.println(
					"Standardization centers: "
							+ java.util.Arrays.toString(
							stats.getCenters()
					)
			);

			System.out.println(
					"Standardization scales: "
							+ java.util.Arrays.toString(
							stats.getScales()
					)
			);

			System.out.println(
					"Standardization counts: "
							+ java.util.Arrays.toString(
							stats.getCounts()
					)
			);
		}
	}

	private void applyEvaluationStandardization(
			ListObjectDataset testingData
	) {
		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			return;
		}

		config.requireImplemented();

		StandardizationStats stats =
				AppContext.standardizationStats;

		if (stats == null) {
			throw new IllegalStateException(
					"The loaded model enables standardization but does not "
							+ "contain fitted or supplied statistics."
			);
		}

		config.validateStatistics(stats);

		if (isLazyDataset(testingData)) {
			/*
			 * readTestData() constructed the lazy test reader using the statistics
			 * restored by ModelIO.applySnapshot(). Each series will be transformed
			 * immediately after materialization.
			 */
			if (AppContext.verbosity > 0) {
				System.out.println(
						"Lazy testing data will use saved "
								+ stats.getMethod()
								+ " standardization during materialization."
				);
			}

			return;
		}

		List<String> featureNames =
				getStandardizationFeatureNames();

		Standardizer.transformInPlace(
				testingData,
				stats,
				featureNames
		);

		if (AppContext.verbosity > 0) {
			System.out.println(
					"Applied saved "
							+ stats.getMethod()
							+ " standardization to the eager testing dataset."
			);
		}
	}

	private void prepareSuppliedStandardization()
			throws IOException {

		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			AppContext.standardizationStats =
					null;

			return;
		}

		config.requireImplemented();

		if (!config.shouldLoadStatistics()) {
			/*
			 * Eager training will fit statistics after the dataset is read.
			 * Lazy training without supplied statistics is rejected later.
			 */
			AppContext.standardizationStats =
					null;

			return;
		}

		List<String> featureNames =
				getStandardizationFeatureNames();

		StandardizationStats stats =
				StandardizationJson.read(
						config.getStatisticsPath(),
						config,
						featureNames
				);

		config.validateStatistics(stats);

		AppContext.standardizationStats =
				stats;

		if (AppContext.verbosity > 0) {
			System.out.println(
					"Loaded standardization statistics from: "
							+ config.getStatisticsPath()
			);

			printStandardizationSummary(stats);
		}
	}

	private void prepareTrainingStatisticsBeforeTestRead(
			ListObjectDataset trainingData
	) throws IOException {

		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			AppContext.standardizationStats =
					null;

			return;
		}

		config.requireImplemented();

		if (config.shouldLoadStatistics()) {
			/*
			 * Already loaded by prepareSuppliedStandardization().
			 */
			if (AppContext.standardizationStats == null) {
				throw new IllegalStateException(
						"Configured standardization statistics were not loaded."
				);
			}

			return;
		}

		if (isLazyDataset(trainingData)) {
			throw new UnsupportedOperationException(
					"Lazy training with standardization currently requires "
							+ "precomputed statistics supplied through "
							+ "-standardization_stats."
			);
		}

		List<String> featureNames =
				getStandardizationFeatureNames();

		StandardizationStats stats =
				StandardizationFitter.fit(
						trainingData,
						config.getMethod(),
						config.getScope(),
						config.getVarianceConvention(),
						featureNames
				);

		config.validateStatistics(stats);

		AppContext.standardizationStats =
				stats;

		if (config.shouldSaveFittedStatistics()) {
			Path outputPath =
					resolveStandardizationOutputPath(config);

			StandardizationJson.write(
					outputPath,
					stats
			);

			if (AppContext.verbosity > 0) {
				System.out.println(
						"Saved fitted standardization statistics to: "
								+ outputPath
				);
			}
		}

		if (AppContext.verbosity > 0) {
			System.out.println(
					"Fitted standardization statistics from "
							+ "the eager training dataset."
			);

			printStandardizationSummary(stats);
		}
	}

	private void applyPreparedStandardization(
			ListObjectDataset trainingData,
			ListObjectDataset testingData
	) {
		StandardizationConfig config =
				AppContext.standardizationConfig;

		if (config == null || config.isDisabled()) {
			return;
		}

		StandardizationStats stats =
				AppContext.standardizationStats;

		if (stats == null) {
			throw new IllegalStateException(
					"Standardization is enabled but no prepared statistics "
							+ "are available."
			);
		}

		config.validateStatistics(stats);

		List<String> featureNames =
				getStandardizationFeatureNames();

		if (!isLazyDataset(trainingData)) {
			Standardizer.transformInPlace(
					trainingData,
					stats,
					featureNames
			);
		}

		if (testingData != null
				&& !isLazyDataset(testingData)) {

			Standardizer.transformInPlace(
					testingData,
					stats,
					featureNames
			);
		}
	}
}