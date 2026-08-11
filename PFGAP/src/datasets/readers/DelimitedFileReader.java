package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Reader for delimited files such as CSV, TSV, TXT, and row-encoded
 * multivariate time series files.
 *
 * This reader converts each row into one ListObjectDataset instance.
 *
 * Supported instance shapes:
 *
 * 1. Numeric 1D, no missing values:
 *      double[]
 *
 * 2. Numeric 1D, with missing values:
 *      Double[]
 *
 * 3. Numeric 2D, no missing values:
 *      double[][]
 *
 * 4. Numeric 2D, with missing values:
 *      Double[][]
 *
 * 5. Generic 1D:
 *      Object[]
 *
 * 6. Generic 2D:
 *      Object[][]
 *
 * Labels may be supplied by a separate label file, or embedded in the data file
 * for the supported 1D numeric cases.
 */
public class DelimitedFileReader implements DatasetReader {

    private final String dataFileName;
    private final String labelFileName;
    private final String entrySeparator;
    private final String arraySeparator;
    private final boolean hasHeader;
    private final boolean is2D;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean targetColumnIsFirst;
    private final boolean isTest;
    private final boolean isRegression;

    public DelimitedFileReader(
            String dataFileName,
            String labelFileName,
            String entrySeparator,
            String arraySeparator,
            boolean hasHeader,
            boolean is2D,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean targetColumnIsFirst,
            boolean isTest,
            boolean isRegression
    ) {
        this.dataFileName = dataFileName;
        this.labelFileName = labelFileName;
        this.entrySeparator = entrySeparator;
        this.arraySeparator = arraySeparator;
        this.hasHeader = hasHeader;
        this.is2D = is2D;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.targetColumnIsFirst = targetColumnIsFirst;
        this.isTest = isTest;
        this.isRegression = isRegression;
    }

    @Override
    public ListObjectDataset read() throws IOException {

        ListObjectDataset dataset = new ListObjectDataset();

        int i = 0;
        long start = System.nanoTime();

        List<Object> labels = new ArrayList<>();

        if (labelFileName != null) {
            labels = readGenericLabels(labelFileName, hasHeader, isRegression);
        }

        try (BufferedReader br = new BufferedReader(new FileReader(dataFileName))) {

            if (hasHeader) {
                br.readLine();
            }

            String line;

            while ((line = br.readLine()) != null) {

                if (isNumeric) {

                    if (hasMissingValues) {

                        if (is2D) {
                            Double[][] data = RowParser.parseBoxedDoubleMatrix(
                                    line,
                                    arraySeparator,
                                    entrySeparator
                            );

                            Object label = getSeparateLabel(labels, i);
                            dataset.add(label, data, i);

                            AppContext.length = data[0].length;

                        } else {
                            Object label;
                            Double[] data;

                            boolean parseEmbeddedLabel =
                                    labelFileName == null
                                            && !AppContext.isIsolationMode()
                                            && (!isTest || AppContext.exists_testlabels);

                            if (parseEmbeddedLabel) {
                                String[] lineArray = line.split(entrySeparator, -1);

                                ParsedBoxedDoubleRow parsed =
                                        RowParser.parseBoxedDoubleRow(
                                                lineArray,
                                                targetColumnIsFirst,
                                                isRegression
                                        );

                                label = parsed.label;
                                data = parsed.features;

                            } else {
                                data = RowParser.parseBoxedDoubleArray(line, entrySeparator);
                                label = getSeparateLabel(labels, i);
                            }

                            dataset.add(label, data, i);
                            AppContext.length = data.length;
                        }

                    } else {

                        if (is2D) {
                            double[][] data = RowParser.parseDoubleMatrix(
                                    line,
                                    arraySeparator,
                                    entrySeparator
                            );

                            Object label = getSeparateLabel(labels, i);
                            dataset.add(label, data, i);

                            AppContext.length = data[0].length;

                        } else {

                            if (labelFileName == null && !isTest && !AppContext.isIsolationMode()) {
                                String[] lineArray = line.split(entrySeparator, -1);

                                ParsedDoubleRow parsed =
                                        RowParser.parseDoubleRow(
                                                lineArray,
                                                targetColumnIsFirst,
                                                isRegression
                                        );

                                dataset.add(parsed.label, parsed.features, i);
                                AppContext.length = parsed.features.length;

                            } else if (labelFileName == null && AppContext.isIsolationMode()) {
                                double[] data = RowParser.parseDoubleArray(line, entrySeparator);

                                dataset.add(null, data, i);
                                AppContext.length = data.length;

                            } else if (labelFileName == null && isTest) {

                                if (AppContext.exists_testlabels) {
                                    String[] lineArray = line.split(entrySeparator, -1);

                                    ParsedDoubleRow parsed =
                                            RowParser.parseDoubleRow(
                                                    lineArray,
                                                    targetColumnIsFirst,
                                                    isRegression
                                            );

                                    dataset.add(parsed.label, parsed.features, i);
                                    AppContext.length = parsed.features.length;

                                } else {
                                    double[] data = RowParser.parseDoubleArray(line, entrySeparator);

                                    dataset.add(null, data, i);
                                    AppContext.length = data.length;
                                }

                            } else {
                                double[] data = RowParser.parseDoubleArray(line, entrySeparator);

                                Object label = getSeparateLabel(labels, i);
                                dataset.add(label, data, i);

                                AppContext.length = data.length;
                            }
                        }
                    }

                } else {

                    if (is2D) {
                        Object[][] data = RowParser.parse2DRow(
                                line,
                                arraySeparator,
                                entrySeparator
                        );

                        Object label = getSeparateLabel(labels, i);
                        dataset.add(label, data, i);

                        AppContext.length = data[0].length;

                    } else {
                        Object[] data = RowParser.parse1DRow(line, entrySeparator);

                        Object label = getSeparateLabel(labels, i);
                        dataset.add(label, data, i);

                        AppContext.length = data.length;
                    }
                }

                ProgressLogger.logProgress(i);
                i++;
            }
        }

        long end = System.nanoTime();
        ProgressLogger.logDuration(start, end);

        return dataset;
    }

    private Object getSeparateLabel(List<Object> labels, int index) {
        return labelFileName != null ? labels.get(index) : null;
    }

    public static List<Integer> readLabels(
            String labelFileName,
            boolean hasHeader
    ) throws IOException {

        List<Integer> labels = new ArrayList<>();

        if (Objects.equals(labelFileName, null)) {
            return labels;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(labelFileName))) {

            if (hasHeader) {
                br.readLine();
            }

            String line;

            while ((line = br.readLine()) != null) {
                labels.add(Integer.parseInt(line.trim()));
            }
        }

        return labels;
    }

    public static List<Object> readGenericLabels(
            String labelFileName,
            boolean hasHeader,
            boolean isRegression
    ) throws IOException {

        List<Object> labels = new ArrayList<>();

        if (Objects.equals(labelFileName, null)) {
            return labels;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(labelFileName))) {

            if (hasHeader) {
                br.readLine();
            }

            String line;

            while ((line = br.readLine()) != null) {

                if (isRegression) {
                    labels.add(Double.parseDouble(line.trim()));
                } else {
                    try {
                        labels.add(Integer.parseInt(line.trim()));
                    } catch (NumberFormatException e) {
                        labels.add(line.trim());
                    }
                }
            }
        }

        return labels;
    }

    public static class ParsedDoubleRow {

        public final Object label;
        public final double[] features;

        public ParsedDoubleRow(Object label, double[] features) {
            this.label = label;
            this.features = features;
        }
    }

    public static class ParsedBoxedDoubleRow {

        public final Object label;
        public final Double[] features;

        public ParsedBoxedDoubleRow(Object label, Double[] features) {
            this.label = label;
            this.features = features;
        }
    }

    public static class ProgressLogger {

        public static void logProgress(int i) {

            if (i % 1000 == 0) {

                if (i % 100000 == 0) {
                    System.out.print("\n");

                    if (i % 1000000 == 0) {
                        long usedMem =
                                AppContext.runtime.totalMemory()
                                        - AppContext.runtime.freeMemory();

                        System.out.print(i + ":" + usedMem / 1024 / 1024 + "mb\n");
                    }

                } else {
                    System.out.print(".");
                }
            }
        }

        public static void logDuration(long start, long end) {

            long elapsed = end - start;

            String timeDuration =
                    DurationFormatUtils.formatDuration(
                            (long) (elapsed / 1e6),
                            "H:m:s.SSS"
                    );

            System.out.println("finished in " + timeDuration);
        }
    }

    public static class FileInfoExtractor {

        public static int[] getFileInformation(
                String fileName,
                boolean hasHeader,
                String separator
        ) throws IOException {

            String line;
            String[] lineArray = null;
            int[] fileInfo = new int[2];

            try (FileReader input = new FileReader(fileName);
                 LineNumberReader lineNumberReader = new LineNumberReader(input)) {

                boolean lengthCheck = true;

                while ((line = lineNumberReader.readLine()) != null) {

                    if (lengthCheck) {
                        lengthCheck = false;
                        lineArray = line.split(separator);
                    }
                }

                fileInfo[0] =
                        hasHeader
                                ? lineNumberReader.getLineNumber() - 1
                                : lineNumberReader.getLineNumber();

                fileInfo[1] = lineArray == null ? 0 : lineArray.length;
            }

            return fileInfo;
        }
    }

    public static class RowParser {

        private static Set<String> missingIndicators = AppContext.MissingStrings;

        public static void setMissingIndicators(Set<String> indicators) {

            missingIndicators =
                    indicators.stream()
                            .map(String::toUpperCase)
                            .collect(Collectors.toSet());
        }

        public static Set<String> getMissingIndicators() {
            return missingIndicators;
        }

        public static ParsedDoubleRow parseDoubleRow(
                String[] lineArray,
                boolean targetColumnIsFirst,
                boolean isRegression
        ) {

            int dataLength = lineArray.length - 1;
            double[] features = new double[dataLength];
            Object label;

            if (targetColumnIsFirst) {

                label =
                        isRegression
                                ? Double.parseDouble(lineArray[0].trim())
                                : tryParseLabel(lineArray[0]);

                for (int j = 1; j <= dataLength; j++) {
                    features[j - 1] = Double.parseDouble(lineArray[j].trim());
                }

            } else {

                label =
                        isRegression
                                ? Double.parseDouble(lineArray[dataLength].trim())
                                : tryParseLabel(lineArray[dataLength]);

                for (int j = 0; j < dataLength; j++) {
                    features[j] = Double.parseDouble(lineArray[j].trim());
                }
            }

            return new ParsedDoubleRow(label, features);
        }

        public static ParsedBoxedDoubleRow parseBoxedDoubleRow(
                String[] lineArray,
                boolean targetColumnIsFirst,
                boolean isRegression
        ) {

            int dataLength = lineArray.length - 1;
            Double[] features = new Double[dataLength];
            Object label;

            if (targetColumnIsFirst) {

                label =
                        isRegression
                                ? Double.parseDouble(lineArray[0].trim())
                                : tryParseLabel(lineArray[0]);

                for (int j = 1; j <= dataLength; j++) {
                    features[j - 1] = parseBoxedDoubleToken(lineArray[j]);
                }

            } else {

                label =
                        isRegression
                                ? Double.parseDouble(lineArray[dataLength].trim())
                                : tryParseLabel(lineArray[dataLength]);

                for (int j = 0; j < dataLength; j++) {
                    features[j] = parseBoxedDoubleToken(lineArray[j]);
                }
            }

            return new ParsedBoxedDoubleRow(label, features);
        }

        private static Double parseBoxedDoubleToken(String token) {

            if (token == null) {
                return null;
            }

            String trimmed = token.trim();

            if (missingIndicators.contains(trimmed.toUpperCase())) {
                return null;
            }

            return Double.valueOf(trimmed);
        }

        private static Object tryParseLabel(String token) {

            String trimmed = token.trim();

            try {
                return Integer.parseInt(trimmed);
            } catch (NumberFormatException ignored) {

                try {
                    double value = Double.parseDouble(trimmed);

                    if (value == Math.rint(value)) {
                        return (int) value;
                    }

                    return value;

                } catch (NumberFormatException ignoredAgain) {
                    return trimmed;
                }
            }
        }

        public static double[] parseDoubleArray(
                String row,
                String separator
        ) {

            String[] tokens = row.split(separator, -1);
            double[] parsed = new double[tokens.length];

            for (int i = 0; i < tokens.length; i++) {
                parsed[i] = Double.parseDouble(tokens[i].trim());
            }

            return parsed;
        }

        public static double[][] parseDoubleMatrix(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {

            String[] rowStrings = row.split(arraySeparator);
            double[][] matrix = new double[rowStrings.length][];

            for (int i = 0; i < rowStrings.length; i++) {
                matrix[i] = parseDoubleArray(rowStrings[i], entrySeparator);
            }

            return matrix;
        }

        public static Double[] parseBoxedDoubleArray(
                String row,
                String separator
        ) {

            String[] tokens = row.split(separator, -1);
            Double[] parsed = new Double[tokens.length];

            for (int i = 0; i < tokens.length; i++) {

                String token = tokens[i].trim();

                parsed[i] =
                        missingIndicators.contains(token.toUpperCase())
                                ? null
                                : Double.valueOf(token);
            }

            return parsed;
        }

        public static Double[][] parseBoxedDoubleMatrix(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {

            String[] rowStrings = row.split(arraySeparator);
            Double[][] matrix = new Double[rowStrings.length][];

            for (int i = 0; i < rowStrings.length; i++) {
                matrix[i] = parseBoxedDoubleArray(rowStrings[i], entrySeparator);
            }

            return matrix;
        }

        public static Object[] parse1DRow(
                String row,
                String separator
        ) {

            String[] tokens = row.split(separator, -1);
            Object[] parsed = new Object[tokens.length];

            for (int i = 0; i < tokens.length; i++) {
                parsed[i] = parseValue(tokens[i]);
            }

            return parsed;
        }

        public static Object[][] parse2DRow(
                String row,
                String arraySeparator,
                String entrySeparator
        ) {

            String[] rowStrings = row.split(arraySeparator);
            Object[][] matrix = new Object[rowStrings.length][];

            for (int i = 0; i < rowStrings.length; i++) {
                matrix[i] = parse1DRow(rowStrings[i], entrySeparator);
            }

            return matrix;
        }

        public static Object parseValue(String token) {

            if (token == null) {
                return null;
            }

            String trimmed = token.trim();

            if (missingIndicators.contains(trimmed.toUpperCase())) {
                return null;
            }

            try {
                return Double.parseDouble(trimmed);
            } catch (NumberFormatException e1) {

                if (trimmed.equalsIgnoreCase("true")
                        || trimmed.equalsIgnoreCase("false")) {
                    return Boolean.parseBoolean(trimmed);
                }

                return trimmed;
            }
        }
    }
}