package datasets.readers;

import core.AppContext;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import dev.hardwood.InputFile;
import dev.hardwood.reader.ParquetFileReader;
import dev.hardwood.reader.RowReader;
import dev.hardwood.schema.ColumnProjection;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Resolves a LazySeriesRef into an actual time-series object by reading
 * exactly one Parquet file.
 *
 * This class intentionally mirrors LongFormatParquetReader's output format,
 * but does not group by id. The whole file is one instance.
 */
public class PerFileParquetSeriesReader implements LazySeriesReader {

    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this.timeColumn = timeColumn;

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.standardizationStats = standardizationStats;
    }

    public PerFileParquetSeriesReader(
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues
    ) {
        this.timeColumn = timeColumn;

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.standardizationStats = null;
    }

    @Override
    public Object read(
            LazySeriesRef ref
    ) {
        if (ref == null) {
            throw new IllegalArgumentException(
                    "Cannot read null LazySeriesRef."
            );
        }

        try {
            return readFile(ref.getFile());

        } catch (IOException e) {
            throw new RuntimeException(
                    "Failed to lazily read Parquet time series from: "
                            + ref.getFile(),
                    e
            );
        }
    }

    private Object readFile(
            Path file
    ) throws IOException {

        List<PerFileParquetRow> rows =
                new ArrayList<>();

        try (ParquetFileReader fileReader =
                     ParquetFileReader.open(InputFile.of(file));
             RowReader reader =
                     fileReader.buildRowReader()
                             .projection(buildColumnProjection())
                             .build()) {

            int rowNumber = 0;

            while (reader.hasNext()) {
                reader.next();

                Object timeValue = null;

                if (timeColumn != null &&
                        !timeColumn.trim().isEmpty()) {

                    timeValue =
                            normalizeValue(
                                    getValue(reader, timeColumn)
                            );
                }

                Object[] featureValues =
                        new Object[featureColumns.size()];

                for (int j = 0; j < featureColumns.size(); j++) {
                    String featureColumn =
                            featureColumns.get(j);

                    Object rawValue =
                            normalizeValue(
                                    getValue(reader, featureColumn)
                            );

                    featureValues[j] =
                            parseFeatureValue(
                                    rawValue,
                                    featureColumn
                            );
                }

                rows.add(
                        new PerFileParquetRow(
                                timeValue,
                                featureValues,
                                rowNumber
                        )
                );

                rowNumber++;
            }
        }

        sortRows(rows);

        Object series =
                buildSeriesData(rows);

        if (standardizationStats != null) {
            Standardizer.transformInstanceInPlace(
                    series,
                    standardizationStats
            );
        }

        return series;
    }

    private ColumnProjection buildColumnProjection() {
        List<String> columns =
                new ArrayList<>();

        if (timeColumn != null &&
                !timeColumn.trim().isEmpty()) {

            columns.add(timeColumn);
        }

        columns.addAll(featureColumns);

        return ColumnProjection.columns(
                columns.toArray(new String[0])
        );
    }

    private static Object getValue(
            RowReader reader,
            String columnName
    ) {
        if (reader.isNull(columnName)) {
            return null;
        }

        return reader.getValue(columnName);
    }

    private void sortRows(
            List<PerFileParquetRow> rows
    ) {
        if (timeColumn == null ||
                timeColumn.trim().isEmpty()) {

            rows.sort(
                    Comparator.comparingInt(
                            row -> row.inputOrder
                    )
            );

            return;
        }

        rows.sort(
                (a, b) ->
                        compareTimeValues(
                                a.timeValue,
                                b.timeValue
                        )
        );
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private int compareTimeValues(
            Object a,
            Object b
    ) {
        if (a == null && b == null) {
            return 0;
        }

        if (a == null) {
            return -1;
        }

        if (b == null) {
            return 1;
        }

        if (a instanceof Number &&
                b instanceof Number) {

            return Double.compare(
                    ((Number) a).doubleValue(),
                    ((Number) b).doubleValue()
            );
        }

        if (a instanceof Comparable &&
                a.getClass().isInstance(b)) {

            return ((Comparable) a).compareTo(b);
        }

        return a.toString()
                .compareTo(b.toString());
    }

    private Object buildSeriesData(
            List<PerFileParquetRow> rows
    ) {
        int timeLength =
                rows.size();

        int dimensionCount =
                featureColumns.size();

        if (dimensionCount == 1) {
            return buildUnivariateSeries(
                    rows,
                    timeLength
            );
        }

        return buildMultivariateSeries(
                rows,
                dimensionCount,
                timeLength
        );
    }

    private Object buildUnivariateSeries(
            List<PerFileParquetRow> rows,
            int timeLength
    ) {
        if (isNumeric) {

            if (hasMissingValues) {
                Double[] data =
                        new Double[timeLength];

                for (int t = 0; t < timeLength; t++) {
                    data[t] =
                            toBoxedDouble(
                                    rows.get(t).featureValues[0]
                            );
                }

                return data;
            }

            double[] data =
                    new double[timeLength];

            for (int t = 0; t < timeLength; t++) {
                data[t] =
                        toPrimitiveDouble(
                                rows.get(t).featureValues[0]
                        );
            }

            return data;
        }

        Object[] data =
                new Object[timeLength];

        for (int t = 0; t < timeLength; t++) {
            data[t] =
                    rows.get(t).featureValues[0];
        }

        return data;
    }

    private Object buildMultivariateSeries(
            List<PerFileParquetRow> rows,
            int dimensionCount,
            int timeLength
    ) {
        if (isNumeric) {

            if (hasMissingValues) {
                Double[][] data =
                        new Double[dimensionCount][timeLength];

                for (int t = 0; t < timeLength; t++) {
                    Object[] featureValues =
                            rows.get(t).featureValues;

                    for (int d = 0; d < dimensionCount; d++) {
                        data[d][t] =
                                toBoxedDouble(
                                        featureValues[d]
                                );
                    }
                }

                return data;
            }

            double[][] data =
                    new double[dimensionCount][timeLength];

            for (int t = 0; t < timeLength; t++) {
                Object[] featureValues =
                        rows.get(t).featureValues;

                for (int d = 0; d < dimensionCount; d++) {
                    data[d][t] =
                            toPrimitiveDouble(
                                    featureValues[d]
                            );
                }
            }

            return data;
        }

        Object[][] data =
                new Object[dimensionCount][timeLength];

        for (int t = 0; t < timeLength; t++) {
            Object[] featureValues =
                    rows.get(t).featureValues;

            for (int d = 0; d < dimensionCount; d++) {
                data[d][t] =
                        featureValues[d];
            }
        }

        return data;
    }

    private Object parseFeatureValue(
            Object value,
            String featureColumn
    ) {
        if (value == null) {

            if (hasMissingValues) {
                return null;
            }

            throw new IllegalArgumentException(
                    "Encountered null value in feature column '"
                            + featureColumn
                            + "', but hasMissingValues=false."
            );
        }

        if (isNumeric) {
            return toBoxedDouble(value);
        }

        return parseGenericValue(value);
    }

    private static Object parseGenericValue(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        Object normalized =
                normalizeValue(value);

        if (normalized == null) {
            return null;
        }

        if (normalized instanceof String) {
            String trimmed =
                    normalized.toString().trim();

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

        return normalized;
    }

    private static Object normalizeValue(
            Object value
    ) {
        return value;
    }

    private static Double toBoxedDouble(
            Object value
    ) {
        if (value == null) {
            return null;
        }

        Object normalized =
                normalizeValue(value);

        if (normalized == null) {
            return null;
        }

        if (normalized instanceof Number) {
            return ((Number) normalized).doubleValue();
        }

        String trimmed =
                normalized.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        return Double.valueOf(trimmed);
    }

    private static double toPrimitiveDouble(
            Object value
    ) {
        if (value == null) {
            throw new IllegalArgumentException(
                    "Encountered null numeric value, "
                            + "but hasMissingValues=false."
            );
        }

        Object normalized =
                normalizeValue(value);

        if (normalized instanceof Number) {
            return ((Number) normalized).doubleValue();
        }

        String trimmed =
                normalized.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {
            throw new IllegalArgumentException(
                    "Encountered missing numeric value, "
                            + "but hasMissingValues=false."
            );
        }

        return Double.parseDouble(trimmed);
    }

    private void validateStandardizationConfiguration() {
        if (standardizationStats == null) {
            return;
        }

        if (!isNumeric) {
            throw new IllegalArgumentException(
                    "Standardization statistics cannot be applied by "
                            + "PerFileParquetSeriesReader when isNumeric=false."
            );
        }

        standardizationStats.validateFeatureCompatibility(
                featureColumns
        );
    }

    private static class PerFileParquetRow {

        private final Object timeValue;
        private final Object[] featureValues;
        private final int inputOrder;

        private PerFileParquetRow(
                Object timeValue,
                Object[] featureValues,
                int inputOrder
        ) {
            this.timeValue = timeValue;
            this.featureValues = featureValues;
            this.inputOrder = inputOrder;
        }
    }

    private static class MissingValueParser {

        private static Set<String> missingIndicators =
                AppContext.MissingStrings;

        public static void setMissingIndicators(
                Set<String> indicators
        ) {
            missingIndicators =
                    indicators.stream()
                            .map(String::toUpperCase)
                            .collect(Collectors.toSet());
        }

        public static boolean isMissing(
                String token
        ) {
            if (token == null) {
                return true;
            }

            return missingIndicators.contains(
                    token.trim().toUpperCase()
            );
        }
    }
}