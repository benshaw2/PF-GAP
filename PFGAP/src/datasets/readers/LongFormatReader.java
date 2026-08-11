package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Reader for long-format delimited time-series data.
 *
 * Long-format data has one row per time point, not one row per instance.
 *
 * Example:
 *
 *      id,time,temp,pressure,label
 *      A,0,10.1,100.0,class1
 *      A,1,10.4,101.2,class1
 *      A,2,10.2,100.8,class1
 *      B,0,5.2,80.1,class2
 *      B,1,5.4,81.0,class2
 *
 * Rows are grouped by idColumn. Within each group, rows are sorted by
 * timeColumn if supplied. Otherwise, input row order is preserved.
 *
 * Output shape:
 *
 * If featureColumns.size() == 1:
 *
 *      numeric, no missing:       double[]
 *      numeric, with missing:     Double[]
 *      generic:                   Object[]
 *
 * If featureColumns.size() > 1:
 *
 *      numeric, no missing:       double[][]
 *      numeric, with missing:     Double[][]
 *      generic:                   Object[][]
 *
 * For multivariate output, matrix orientation is:
 *
 *      feature x time
 *
 * Labels:
 *
 *      - If labelColumns is empty, label is null.
 *      - If labelColumns has one column, label is a scalar Object.
 *      - If labelColumns has multiple columns, label is List<Object>.
 *
 * This design supports future multi-label classification and multi-target
 * regression while preserving ListObjectDataset's existing Object label type.
 */
public class LongFormatReader implements DatasetReader {

    private final String dataFileName;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;

    private final String idColumn;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final List<String> labelColumns;

    public LongFormatReader(ReaderOptions options) {

        this(
                options.getDataPath(),
                options.getEntrySeparator(),
                options.hasHeader(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                options.getIdColumn(),
                options.getTimeColumn(),
                options.getFeatureColumns(),
                options.getLabelColumns()
        );
    }

    public LongFormatReader(
            String dataFileName,
            String entrySeparator,
            boolean hasHeader,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String idColumn,
            String timeColumn,
            List<String> featureColumns,
            List<String> labelColumns
    ) {
        this.dataFileName = dataFileName;
        this.entrySeparator = entrySeparator;
        this.hasHeader = hasHeader;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.idColumn = idColumn;
        this.timeColumn = timeColumn;
        this.featureColumns = featureColumns == null
                ? new ArrayList<>()
                : new ArrayList<>(featureColumns);
        this.labelColumns = labelColumns == null
                ? new ArrayList<>()
                : new ArrayList<>(labelColumns);
    }

    @Override
    public ListObjectDataset read() throws IOException {

        validateOptions();

        if (idColumn == null || idColumn.trim().isEmpty()) {
            return readRowWiseDataset();
        }

        return readGroupedLongFormatDataset();

    }

    public ListObjectDataset readGroupedLongFormatDataset() throws IOException {

        //validateOptions();

        long start = System.nanoTime();

        Map<Object, List<LongRow>> groupedRows = new LinkedHashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(dataFileName))) {

            String headerLine = br.readLine();

            if (headerLine == null) {
                throw new IOException("Long-format file is empty: " + dataFileName);
            }

            if (!hasHeader) {
                throw new IOException(
                        "LongFormatReader currently requires a header because "
                                + "idColumn, timeColumn, featureColumns, and labelColumns "
                                + "are specified by column name."
                );
            }

            String[] header = splitLine(headerLine);
            Map<String, Integer> columnIndex = buildColumnIndex(header);

            int idIndex = requireColumn(columnIndex, idColumn, "idColumn");

            Integer timeIndex = null;

            if (timeColumn != null && !timeColumn.trim().isEmpty()) {
                timeIndex = requireColumn(columnIndex, timeColumn, "timeColumn");
            }

            List<Integer> featureIndices = new ArrayList<>();

            for (String featureColumn : featureColumns) {
                featureIndices.add(
                        requireColumn(columnIndex, featureColumn, "featureColumns")
                );
            }

            List<Integer> labelIndices = new ArrayList<>();

            for (String labelColumn : labelColumns) {
                labelIndices.add(
                        requireColumn(columnIndex, labelColumn, "labelColumns")
                );
            }

            String line;
            int rowNumber = 0;

            while ((line = br.readLine()) != null) {

                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] tokens = splitLine(line);

                Object id = parseGenericValue(getToken(tokens, idIndex));

                Object timeValue = null;

                if (timeIndex != null) {
                    timeValue = parseGenericValue(getToken(tokens, timeIndex));
                }

                Object[] featureValues = new Object[featureIndices.size()];

                for (int j = 0; j < featureIndices.size(); j++) {
                    featureValues[j] =
                            parseFeatureValue(getToken(tokens, featureIndices.get(j)));
                }

                Object label = parseLabelValues(tokens, labelIndices);

                LongRow row =
                        new LongRow(
                                id,
                                timeValue,
                                featureValues,
                                label,
                                rowNumber
                        );

                groupedRows
                        .computeIfAbsent(id, ignored -> new ArrayList<>())
                        .add(row);

                ProgressLogger.logProgress(rowNumber);
                rowNumber++;
            }
        }

        ListObjectDataset dataset = buildDataset(groupedRows);

        long end = System.nanoTime();
        ProgressLogger.logDuration(start, end);

        return dataset;
    }

    private ListObjectDataset readRowWiseDataset() throws IOException {

        long start = System.nanoTime();

        ListObjectDataset dataset = new ListObjectDataset();

        try (BufferedReader br = new BufferedReader(new FileReader(dataFileName))) {

            String headerLine = br.readLine();

            if (headerLine == null) {
                throw new IOException("Long-format file is empty: " + dataFileName);
            }

            if (!hasHeader) {
                throw new IOException(
                        "LongFormatReader row-wise mode currently requires a header "
                                + "because featureColumns and labelColumns are specified "
                                + "by column name."
                );
            }

            String[] header = splitLine(headerLine);
            Map<String, Integer> columnIndex = buildColumnIndex(header);

            List<Integer> featureIndices = new ArrayList<>();

            for (String featureColumn : featureColumns) {
                featureIndices.add(
                        requireColumn(columnIndex, featureColumn, "featureColumns")
                );
            }

            List<Integer> labelIndices = new ArrayList<>();

            for (String labelColumn : labelColumns) {
                labelIndices.add(
                        requireColumn(columnIndex, labelColumn, "labelColumns")
                );
            }

            String line;
            int rowIndex = 0;

            while ((line = br.readLine()) != null) {

                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] tokens = splitLine(line);

                Object[] featureValues = new Object[featureIndices.size()];

                for (int j = 0; j < featureIndices.size(); j++) {
                    featureValues[j] =
                            parseFeatureValue(getToken(tokens, featureIndices.get(j)));
                }

                Object label = parseLabelValues(tokens, labelIndices);

                Object data = buildRowWiseData(featureValues);

                dataset.add(label, data, rowIndex);

                updateGlobalLength(data);

                ProgressLogger.logProgress(rowIndex);
                rowIndex++;
            }
        }

        long end = System.nanoTime();
        ProgressLogger.logDuration(start, end);

        return dataset;
    }

    private ListObjectDataset buildDataset(
            Map<Object, List<LongRow>> groupedRows
    ) {

        ListObjectDataset dataset = new ListObjectDataset();

        int instanceIndex = 0;

        for (Map.Entry<Object, List<LongRow>> entry : groupedRows.entrySet()) {

            List<LongRow> rows = entry.getValue();

            sortRows(rows);

            Object label = inferGroupLabel(rows);

            Object data = buildSeriesData(rows);

            dataset.add(label, data, instanceIndex);

            updateGlobalLength(data);

            instanceIndex++;
        }

        return dataset;
    }

    private void sortRows(List<LongRow> rows) {

        if (timeColumn == null || timeColumn.trim().isEmpty()) {
            rows.sort(Comparator.comparingInt(row -> row.inputOrder));
            return;
        }

        rows.sort((a, b) -> compareTimeValues(a.timeValue, b.timeValue));
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private int compareTimeValues(Object a, Object b) {

        if (a == null && b == null) {
            return 0;
        }

        if (a == null) {
            return -1;
        }

        if (b == null) {
            return 1;
        }

        if (a instanceof Number && b instanceof Number) {
            return Double.compare(
                    ((Number) a).doubleValue(),
                    ((Number) b).doubleValue()
            );
        }

        if (a instanceof Comparable && a.getClass().isInstance(b)) {
            return ((Comparable) a).compareTo(b);
        }

        return a.toString().compareTo(b.toString());
    }

    private Object inferGroupLabel(List<LongRow> rows) {

        if (rows.isEmpty()) {
            return null;
        }

        Object firstLabel = rows.get(0).label;

        for (LongRow row : rows) {
            if (!labelsEqual(firstLabel, row.label)) {
                throw new IllegalArgumentException(
                        "Inconsistent labels found within long-format group for id: "
                                + row.id
                );
            }
        }

        return firstLabel;
    }

    private boolean labelsEqual(Object a, Object b) {

        if (a == null && b == null) {
            return true;
        }

        if (a == null || b == null) {
            return false;
        }

        return a.equals(b);
    }

    private Object buildSeriesData(List<LongRow> rows) {

        int timeLength = rows.size();
        int dimensionCount = featureColumns.size();

        if (dimensionCount == 1) {
            return buildUnivariateSeries(rows, timeLength);
        }

        return buildMultivariateSeries(rows, dimensionCount, timeLength);
    }

    private Object buildUnivariateSeries(
            List<LongRow> rows,
            int timeLength
    ) {

        if (isNumeric) {

            if (hasMissingValues) {
                Double[] data = new Double[timeLength];

                for (int t = 0; t < timeLength; t++) {
                    data[t] = toBoxedDouble(rows.get(t).featureValues[0]);
                }

                return data;
            }

            double[] data = new double[timeLength];

            for (int t = 0; t < timeLength; t++) {
                data[t] = toPrimitiveDouble(rows.get(t).featureValues[0]);
            }

            return data;
        }

        Object[] data = new Object[timeLength];

        for (int t = 0; t < timeLength; t++) {
            data[t] = rows.get(t).featureValues[0];
        }

        return data;
    }

    private Object buildMultivariateSeries(
            List<LongRow> rows,
            int dimensionCount,
            int timeLength
    ) {

        if (isNumeric) {

            if (hasMissingValues) {
                Double[][] data = new Double[dimensionCount][timeLength];

                for (int t = 0; t < timeLength; t++) {
                    LongRow row = rows.get(t);

                    for (int d = 0; d < dimensionCount; d++) {
                        data[d][t] = toBoxedDouble(row.featureValues[d]);
                    }
                }

                return data;
            }

            double[][] data = new double[dimensionCount][timeLength];

            for (int t = 0; t < timeLength; t++) {
                LongRow row = rows.get(t);

                for (int d = 0; d < dimensionCount; d++) {
                    data[d][t] = toPrimitiveDouble(row.featureValues[d]);
                }
            }

            return data;
        }

        Object[][] data = new Object[dimensionCount][timeLength];

        for (int t = 0; t < timeLength; t++) {
            LongRow row = rows.get(t);

            for (int d = 0; d < dimensionCount; d++) {
                data[d][t] = row.featureValues[d];
            }
        }

        return data;
    }

    private Object buildRowWiseData(Object[] featureValues) {

        int length = featureValues.length;

        if (isNumeric) {

            if (hasMissingValues) {
                Double[] data = new Double[length];

                for (int i = 0; i < length; i++) {
                    data[i] = toBoxedDouble(featureValues[i]);
                }

                return data;
            }

            double[] data = new double[length];

            for (int i = 0; i < length; i++) {
                data[i] = toPrimitiveDouble(featureValues[i]);
            }

            return data;
        }

        Object[] data = new Object[length];

        System.arraycopy(featureValues, 0, data, 0, length);

        return data;
    }

    private Object parseLabelValues(
            String[] tokens,
            List<Integer> labelIndices
    ) {

        if (labelIndices.isEmpty()) {
            return null;
        }

        if (labelIndices.size() == 1) {
            return parseLabelValue(getToken(tokens, labelIndices.get(0)));
        }

        List<Object> labels = new ArrayList<>();

        for (Integer labelIndex : labelIndices) {
            labels.add(parseLabelValue(getToken(tokens, labelIndex)));
        }

        return labels;
    }

    private Object parseFeatureValue(String token) {

        if (isNumeric) {

            if (hasMissingValues) {
                return parseBoxedDoubleToken(token);
            }

            return Double.parseDouble(token.trim());
        }

        return parseGenericValue(token);
    }

    private Object parseLabelValue(String token) {

        if (token == null) {
            return null;
        }

        String trimmed = token.trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        if (isRegression) {
            return Double.parseDouble(trimmed);
        }

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

    private static Object parseGenericValue(String token) {

        if (token == null) {
            return null;
        }

        String trimmed = token.trim();

        if (MissingValueParser.isMissing(trimmed)) {
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

    private static Double parseBoxedDoubleToken(String token) {

        if (token == null) {
            return null;
        }

        String trimmed = token.trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        return Double.valueOf(trimmed);
    }

    private static Double toBoxedDouble(Object value) {

        if (value == null) {
            return null;
        }

        if (value instanceof Double) {
            return (Double) value;
        }

        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }

        return Double.valueOf(value.toString());
    }

    private static double toPrimitiveDouble(Object value) {

        if (value == null) {
            throw new IllegalArgumentException(
                    "Encountered null numeric value, but hasMissingValues=false."
            );
        }

        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }

        return Double.parseDouble(value.toString());
    }

    private String[] splitLine(String line) {
        return line.split(entrySeparator, -1);
    }

    private static String getToken(String[] tokens, int index) {

        if (index < 0 || index >= tokens.length) {
            throw new IllegalArgumentException(
                    "Column index out of bounds: " + index
            );
        }

        return tokens[index];
    }

    private static Map<String, Integer> buildColumnIndex(String[] header) {

        Map<String, Integer> columnIndex = new LinkedHashMap<>();

        for (int i = 0; i < header.length; i++) {
            String columnName = header[i].trim();

            if (columnIndex.containsKey(columnName)) {
                throw new IllegalArgumentException(
                        "Duplicate column name in long-format file: " + columnName
                );
            }

            columnIndex.put(columnName, i);
        }

        return columnIndex;
    }

    private static int requireColumn(
            Map<String, Integer> columnIndex,
            String columnName,
            String optionName
    ) {

        if (columnName == null || columnName.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires " + optionName + "."
            );
        }

        Integer index = columnIndex.get(columnName);

        if (index == null) {
            throw new IllegalArgumentException(
                    "Column not found for " + optionName + ": " + columnName
            );
        }

        return index;
    }

    private void validateOptions() {

        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires entrySeparator."
            );
        }

        if (idColumn == null || idColumn.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires idColumn."
            );
        }

        if (featureColumns == null || featureColumns.isEmpty()) {
            throw new IllegalArgumentException(
                    "LongFormatReader requires at least one feature column."
            );
        }

        if (!hasHeader) {
            throw new IllegalArgumentException(
                    "LongFormatReader currently requires hasHeader=true."
            );
        }
    }

    private static void updateGlobalLength(Object data) {

        if (data instanceof double[]) {
            AppContext.length = ((double[]) data).length;

        } else if (data instanceof Double[]) {
            AppContext.length = ((Double[]) data).length;

        } else if (data instanceof double[][]) {
            double[][] matrix = (double[][]) data;

            if (matrix.length > 0) {
                AppContext.length = matrix[0].length;
            }

        } else if (data instanceof Double[][]) {
            Double[][] matrix = (Double[][]) data;

            if (matrix.length > 0) {
                AppContext.length = matrix[0].length;
            }

        } else if (data instanceof Object[][]) {
            Object[][] matrix = (Object[][]) data;

            if (matrix.length > 0) {
                AppContext.length = matrix[0].length;
            }

        } else if (data instanceof Object[]) {
            AppContext.length = ((Object[]) data).length;
        }
    }

    private static class LongRow {

        private final Object id;
        private final Object timeValue;
        private final Object[] featureValues;
        private final Object label;
        private final int inputOrder;

        private LongRow(
                Object id,
                Object timeValue,
                Object[] featureValues,
                Object label,
                int inputOrder
        ) {
            this.id = id;
            this.timeValue = timeValue;
            this.featureValues = featureValues;
            this.label = label;
            this.inputOrder = inputOrder;
        }
    }

    private static class MissingValueParser {

        private static Set<String> missingIndicators =
                AppContext.MissingStrings;

        public static void setMissingIndicators(Set<String> indicators) {
            missingIndicators =
                    indicators.stream()
                            .map(String::toUpperCase)
                            .collect(Collectors.toSet());
        }

        public static boolean isMissing(String token) {

            if (token == null) {
                return true;
            }

            return missingIndicators.contains(token.trim().toUpperCase());
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
}