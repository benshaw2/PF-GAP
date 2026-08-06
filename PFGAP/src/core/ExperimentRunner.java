package core;

import datasets.ListObjectDataset;
import imputation.util.MissingIndicesBuilder;
import imputation.ProximityImputation;
import org.apache.commons.lang3.ArrayUtils;
import outlier.OutlierScorer;
import trees.ProximityForest;
import util.GeneralUtilities;
import util.PrintUtilities;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
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

		ListObjectDataset trainDataOriginal = readTrainingData();
		ListObjectDataset testDataOriginal = AppContext.testing_file != null
				? readTestData()
				: null;

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

			ProximityForestResult result = forest.test(test_data);

			if (!AppContext.perform_test_imputation) {
				result.printResults(datasetName, repetition, "");
			}

			if (AppContext.impute_test) {
				writeTestingData();
			}

			if (AppContext.getprox) {
				computeAndWriteTestTrainProximities(forest, trainData);
			}

			if (AppContext.get_predictions) {
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

	private ListObjectDataset readTrainingData() {

		return DelimitedFileReader.readToListObjectDataset(
				AppContext.training_file,
				AppContext.training_labels,
				AppContext.entry_separator,
				AppContext.array_separator,
				AppContext.csv_has_header,
				AppContext.is2D,
				AppContext.isNumeric,
				AppContext.hasMissingValues,
				AppContext.target_column_is_first,
				false,
				AppContext.isRegression
		);
	}

	private ListObjectDataset readTestData() {

		return DelimitedFileReader.readToListObjectDataset(
				AppContext.testing_file,
				AppContext.testing_labels,
				AppContext.entry_separator,
				AppContext.array_separator,
				AppContext.csv_has_header,
				AppContext.is2D,
				AppContext.isNumeric,
				AppContext.hasMissingValues,
				AppContext.target_column_is_first,
				true,
				AppContext.isRegression
		);
	}

	private ListObjectDataset prepareTrainingData(
			ListObjectDataset original) {

		ListObjectDataset prepared;

		if (!AppContext.isRegression) {
			prepared = original.reorder_class_labels(null);
		} else {
			prepared = original;
		}

		if (AppContext.hasMissingValues) {
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

		if (!AppContext.isRegression) {
			prepared = original.reorder_class_labels(labelMap);
		} else {
			prepared = original;
		}

		if (AppContext.hasMissingValues) {
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

		if (AppContext.impute_test) {
			writeTestingData();
		}

		ProximityForestResult result = forest.test(test_data);

		if (AppContext.get_predictions) {
			writePredictions(
					result,
					test_data,
					"Validation_Predictions.txt"
			);
		}

		if (!AppContext.perform_test_imputation || true) {
			result.printResults(datasetName, repetition, "");
		}
	}

	private void saveModel(ProximityForest forest) {

		try {
			AppContextSnapshot snapshot = AppContextUtils.captureSnapshot();

			ModelIO.saveModel(
					AppContext.output_dir + AppContext.modelname + ".ser",
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

		PrintWriter writer = new PrintWriter(
				AppContext.output_dir + fileName,
				StandardCharsets.UTF_8
		);

		if (!AppContext.isRegression) {
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
			ProximityForest forest) throws IOException, ExecutionException, InterruptedException {

		if (!AppContext.get_training_outlier_scores || AppContext.isRegression) {
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
}