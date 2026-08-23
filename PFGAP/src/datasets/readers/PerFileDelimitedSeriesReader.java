package datasets.readers;

import core.AppContext;
import datasets.readers.lazy.LazySeriesReader;
import datasets.readers.lazy.LazySeriesRef;
import preprocessing.standardization.StandardizationStats;
import preprocessing.standardization.Standardizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Materializes one time series from one delimited file.
 *
 * This class is shared by:
 *
 *      PerFileDelimitedReader
 *      LazyPerFileDelimitedReader
 *
 * The class itself is not inherently eager or lazy. It reads one file when
 * read(...) is called. The higher-level dataset reader determines whether
 * that call occurs:
 *
 *      1. During initial dataset construction, for eager loading.
 *      2. During distance evaluation, for lazy loading.
 *
 * Expected file organization:
 *
 *      one row = one time point
 *      one selected column = one time-series dimension
 *
 *
 * Returned representation:
 *
 *      numeric data without missing values:
 *          double[dimension][time]
 *
 *      numeric data with missing values:
 *          Double[dimension][time]
 *
 *      nonnumeric data:
 *          Object[dimension][time]
 *
 * This dimension-major representation matches the expected organization of
 * multivariate time-series distances in PFGAP.
 *
 * Header behavior:
 *
 *      hasHeader = true
 *
 *          featureColumns and timeColumn are interpreted as column names.
 *
 *      hasHeader = false
 *
 *          featureColumns may contain zero-based integer column indices,
 *          such as:
 *
 *              ["0", "1", "2"]
 *
 *          If featureColumns is empty, every column except timeColumn is
 *          treated as a feature column.
 *
 *          When no header is present, timeColumn may also be a zero-based
 *          integer column index.
 *
 * Missing numeric values are represented as Double.NaN. Missing nonnumeric
 * values are retained as null.
 *
 * The delimiter is treated literally. This implementation does not currently
 * implement quoted CSV fields containing delimiters or embedded newlines.
 */
public class PerFileDelimitedSeriesReader
        implements LazySeriesReader {

    private final String entrySeparator;
    private final boolean hasHeader;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final StandardizationStats standardizationStats;

    public PerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedSeriesReader requires a non-empty "
                            + "entry separator."
            );
        }

        this.entrySeparator =
                normalizeSeparator(entrySeparator);

        this.hasHeader =
                hasHeader;

        this.timeColumn =
                normalizeNullableString(timeColumn);

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.standardizationStats = standardizationStats;

        validateStandardizationConfiguration();
    }

    public PerFileDelimitedSeriesReader(
            String entrySeparator,
            boolean hasHeader,
            String timeColumn,
            List<String> featureColumns,
            boolean isNumeric,
            boolean hasMissingValues
    ) {
        if (entrySeparator == null || entrySeparator.isEmpty()) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedSeriesReader requires a non-empty "
                            + "entry separator."
            );
        }

        this.entrySeparator =
                normalizeSeparator(entrySeparator);

        this.hasHeader =
                hasHeader;

        this.timeColumn =
                normalizeNullableString(timeColumn);

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.isNumeric =
                isNumeric;

        this.hasMissingValues =
                hasMissingValues;

        this.standardizationStats = null;
    }

    /**
     * Reads and materializes the file identified by a lazy series reference.
     *
     * The reader key and dataset index identify the reference within PFGAP,
     * while the file path identifies the actual delimited file to read.
     */
    @Override
    public Object read(
            LazySeriesRef reference
    ) {
        if (reference == null) {
            throw new IllegalArgumentException(
                    "Cannot read a null LazySeriesRef."
            );
        }

        Path file =
                reference.getFile();

        try {
            return readFile(file);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to read delimited time-series file: "
                            + file,
                    e
            );
        }
    }

    /**
     * Reads one delimited file directly.
     *
     * This method is useful to the future eager PerFileDelimitedReader,
     * which can materialize files without constructing temporary
     * LazySeriesRef objects.
     */
    public Object readFile(
            Path file
    ) throws IOException {
        validateFile(file);

        try (BufferedReader reader =
                     Files.newBufferedReader(
                             file,
                             StandardCharsets.UTF_8
                     )) {

            String firstLine =
                    readNextDataLine(reader);

            if (firstLine == null) {
                throw new IOException(
                        "Delimited time-series file is empty: "
                                + file
                );
            }

            String[] header = null;
            String[] firstDataRow = null;

            if (hasHeader) {
                header =
                        splitLine(firstLine);
            } else {
                firstDataRow =
                        splitLine(firstLine);
            }

            ColumnSelection selection =
                    resolveColumnSelection(
                            file,
                            header,
                            firstDataRow
                    );

            List<String[]> rows =
                    new ArrayList<>();

            if (firstDataRow != null) {
                validateRowWidth(
                        file,
                        1,
                        firstDataRow,
                        selection.columnCount
                );

                rows.add(firstDataRow);
            }

            String line;
            int physicalLineNumber =
                    hasHeader
                            ? 1
                            : 0;

            while ((line = reader.readLine()) != null) {
                physicalLineNumber++;

                /*
                 * Empty and whitespace-only lines are ignored. This is useful
                 * for files ending with extra line separators.
                 */
                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] row =
                        splitLine(line);

                validateRowWidth(
                        file,
                        physicalLineNumber,
                        row,
                        selection.columnCount
                );

                rows.add(row);
            }

            if (rows.isEmpty()) {
                throw new IOException(
                        "Delimited time-series file contains no data rows: "
                                + file
                );
            }

            Object series;

            if (isNumeric) {
                if (hasMissingValues) {
                    series =
                            materializeBoxedNumeric(
                                    file,
                                    rows,
                                    selection.featureIndices
                            );
                } else {
                    series =
                            materializePrimitiveNumeric(
                                    file,
                                    rows,
                                    selection.featureIndices
                            );
                }
            } else {
                series =
                        materializeNonnumeric(
                                rows,
                                selection.featureIndices
                        );
            }

            if (standardizationStats != null) {
                Standardizer.transformInstanceInPlace(
                        series,
                        standardizationStats
                );
            }

            return series;
        }
    }

    private void validateFile(
            Path file
    ) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException(
                    "PerFileDelimitedSeriesReader requires a non-null file."
            );
        }

        if (!Files.exists(file)) {
            throw new IOException(
                    "Delimited time-series file does not exist: "
                            + file
            );
        }

        if (!Files.isRegularFile(file)) {
            throw new IOException(
                    "Delimited time-series path is not a regular file: "
                            + file
            );
        }

        if (!Files.isReadable(file)) {
            throw new IOException(
                    "Delimited time-series file is not readable: "
                            + file
            );
        }
    }

    private String readNextDataLine(
            BufferedReader reader
    ) throws IOException {
        String line;

        while ((line = reader.readLine()) != null) {
            if (!line.trim().isEmpty()) {
                return line;
            }
        }

        return null;
    }

    private String[] splitLine(
            String line
    ) {
        return line.split(
                Pattern.quote(entrySeparator),
                -1
        );
    }

    private ColumnSelection resolveColumnSelection(
            Path file,
            String[] header,
            String[] firstDataRow
    ) {
        int columnCount =
                header != null
                        ? header.length
                        : firstDataRow.length;

        if (columnCount == 0) {
            throw new IllegalArgumentException(
                    "Delimited file has no columns: "
                            + file
            );
        }

        int timeColumnIndex =
                resolveTimeColumnIndex(
                        file,
                        header,
                        columnCount
                );

        int[] featureIndices =
                resolveFeatureIndices(
                        file,
                        header,
                        columnCount,
                        timeColumnIndex
                );

        if (featureIndices.length == 0) {
            throw new IllegalArgumentException(
                    "No feature columns were selected for file: "
                            + file
            );
        }

        return new ColumnSelection(
                columnCount,
                timeColumnIndex,
                featureIndices
        );
    }

    private int resolveTimeColumnIndex(
            Path file,
            String[] header,
            int columnCount
    ) {
        if (timeColumn == null) {
            return -1;
        }

        if (hasHeader) {
            return findNamedColumn(
                    file,
                    header,
                    timeColumn,
                    "time"
            );
        }

        int index =
                parseColumnIndex(
                        timeColumn,
                        "time_column",
                        file
                );

        validateColumnIndex(
                index,
                columnCount,
                "time_column",
                file
        );

        return index;
    }

    private int[] resolveFeatureIndices(
            Path file,
            String[] header,
            int columnCount,
            int timeColumnIndex
    ) {
        if (featureColumns.isEmpty()) {
            return allColumnsExcept(
                    columnCount,
                    timeColumnIndex
            );
        }

        int[] indices =
                new int[featureColumns.size()];

        boolean[] used =
                new boolean[columnCount];

        for (int i = 0; i < featureColumns.size(); i++) {
            String feature =
                    featureColumns.get(i);

            if (feature == null || feature.isBlank()) {
                throw new IllegalArgumentException(
                        "Feature-column names or indices cannot be "
                                + "null or blank."
                );
            }

            int index;

            if (hasHeader) {
                index =
                        findNamedColumn(
                                file,
                                header,
                                feature,
                                "feature"
                        );
            } else {
                index =
                        parseColumnIndex(
                                feature,
                                "feature column",
                                file
                        );

                validateColumnIndex(
                        index,
                        columnCount,
                        "feature column",
                        file
                );
            }

            if (index == timeColumnIndex) {
                throw new IllegalArgumentException(
                        "Column "
                                + feature
                                + " is configured as both the time column "
                                + "and a feature column in file: "
                                + file
                );
            }

            if (used[index]) {
                throw new IllegalArgumentException(
                        "Feature column was selected more than once: "
                                + feature
                                + " in file "
                                + file
                );
            }

            used[index] =
                    true;

            indices[i] =
                    index;
        }

        return indices;
    }

    private int findNamedColumn(
            Path file,
            String[] header,
            String requestedColumn,
            String role
    ) {
        Map<String, Integer> headerIndices =
                buildHeaderIndex(
                        file,
                        header
                );

        Integer index =
                headerIndices.get(
                        requestedColumn
                );

        if (index == null) {
            throw new IllegalArgumentException(
                    "Could not find "
                            + role
                            + " column '"
                            + requestedColumn
                            + "' in file "
                            + file
                            + ". Available columns: "
                            + Arrays.toString(header)
            );
        }

        return index;
    }

    private Map<String, Integer> buildHeaderIndex(
            Path file,
            String[] header
    ) {
        Map<String, Integer> indices =
                new HashMap<>();

        for (int i = 0; i < header.length; i++) {
            String name =
                    header[i].trim();

            if (name.isEmpty()) {
                throw new IllegalArgumentException(
                        "Delimited file contains a blank header at "
                                + "column "
                                + i
                                + ": "
                                + file
                );
            }

            Integer previous =
                    indices.put(
                            name,
                            i
                    );

            if (previous != null) {
                throw new IllegalArgumentException(
                        "Delimited file contains duplicate header '"
                                + name
                                + "': "
                                + file
                );
            }
        }

        return indices;
    }

    private int[] allColumnsExcept(
            int columnCount,
            int excludedIndex
    ) {
        int resultLength =
                excludedIndex >= 0
                        ? columnCount - 1
                        : columnCount;

        int[] indices =
                new int[resultLength];

        int outputIndex =
                0;

        for (int columnIndex = 0;
             columnIndex < columnCount;
             columnIndex++) {

            if (columnIndex == excludedIndex) {
                continue;
            }

            indices[outputIndex++] =
                    columnIndex;
        }

        return indices;
    }

    private double[][] materializePrimitiveNumeric(
            Path file,
            List<String[]> rows,
            int[] featureIndices
    ) {
        double[][] result =
                new double[featureIndices.length][rows.size()];

        for (int timeIndex = 0;
             timeIndex < rows.size();
             timeIndex++) {

            String[] row =
                    rows.get(timeIndex);

            for (int dimension = 0;
                 dimension < featureIndices.length;
                 dimension++) {

                int columnIndex =
                        featureIndices[dimension];

                String token =
                        row[columnIndex].trim();

                if (isMissingValue(token)) {
                    throw new IllegalArgumentException(
                            "Encountered a missing value in file "
                                    + file
                                    + " at data row "
                                    + timeIndex
                                    + ", column "
                                    + columnIndex
                                    + ", but hasMissingValues is false."
                    );
                }

                try {
                    result[dimension][timeIndex] =
                            Double.parseDouble(token);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "Could not parse numeric value '"
                                    + token
                                    + "' in file "
                                    + file
                                    + " at data row "
                                    + timeIndex
                                    + ", column "
                                    + columnIndex
                                    + ".",
                            e
                    );
                }
            }
        }

        return result;
    }

    private Double[][] materializeBoxedNumeric(
            Path file,
            List<String[]> rows,
            int[] featureIndices
    ) {
        Double[][] result =
                new Double[featureIndices.length][rows.size()];

        for (int timeIndex = 0;
             timeIndex < rows.size();
             timeIndex++) {

            String[] row =
                    rows.get(timeIndex);

            for (int dimension = 0;
                 dimension < featureIndices.length;
                 dimension++) {

                int columnIndex =
                        featureIndices[dimension];

                String token =
                        row[columnIndex].trim();

                if (isMissingValue(token)) {
                    result[dimension][timeIndex] =
                            null;

                    continue;
                }

                try {
                    result[dimension][timeIndex] =
                            Double.valueOf(token);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "Could not parse numeric value '"
                                    + token
                                    + "' in file "
                                    + file
                                    + " at data row "
                                    + timeIndex
                                    + ", column "
                                    + columnIndex
                                    + ".",
                            e
                    );
                }
            }
        }

        return result;
    }

    /*private Double[][] materializeNumeric(
            Path file,
            List<String[]> rows,
            int[] featureIndices
    ) {
        Double[][] result =
                new Double[featureIndices.length][rows.size()];

        for (int timeIndex = 0;
             timeIndex < rows.size();
             timeIndex++) {

            String[] row =
                    rows.get(timeIndex);

            for (int dimension = 0;
                 dimension < featureIndices.length;
                 dimension++) {

                int columnIndex =
                        featureIndices[dimension];

                String value =
                        row[columnIndex].trim();

                result[dimension][timeIndex] =
                        parseNumericValue(
                                file,
                                timeIndex,
                                columnIndex,
                                value
                        );
            }
        }

        return result;
    }*/

    /*private Double parseNumericValue(
            Path file,
            int timeIndex,
            int columnIndex,
            String value
    ) {
        if (isMissingValue(value)) {
            if (!hasMissingValues) {
                throw new IllegalArgumentException(
                        "Encountered missing value in file "
                                + file
                                + " at data row "
                                + timeIndex
                                + ", column "
                                + columnIndex
                                + ", but hasMissingValues is false."
                );
            }

            return Double.NaN;
        }

        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Could not parse numeric value '"
                            + value
                            + "' in file "
                            + file
                            + " at data row "
                            + timeIndex
                            + ", column "
                            + columnIndex
                            + ".",
                    e
            );
        }
    }*/

    private Object[][] materializeNonnumeric(
            List<String[]> rows,
            int[] featureIndices
    ) {
        Object[][] result =
                new Object[featureIndices.length][rows.size()];

        for (int timeIndex = 0;
             timeIndex < rows.size();
             timeIndex++) {

            String[] row =
                    rows.get(timeIndex);

            for (int dimension = 0;
                 dimension < featureIndices.length;
                 dimension++) {

                String value =
                        row[featureIndices[dimension]];

                /*if (isMissingValue(value)) {
                    result[dimension][timeIndex] =
                            null;
                } else {
                    result[dimension][timeIndex] =
                            value;
                }*/
                result[dimension][timeIndex] = DelimitedFileReader.RowParser.parseValue(value);
            }
        }

        return result;
    }

    private boolean isMissingValue(
            String value
    ) {
        if (value == null) {
            return true;
        }

        String trimmed =
                value.trim();

        if (trimmed.isEmpty()) {
            return true;
        }

        if (AppContext.MissingStrings == null
                || AppContext.MissingStrings.isEmpty()) {

            return false;
        }

        return AppContext.MissingStrings.contains(
                trimmed
        );
    }

    private void validateRowWidth(
            Path file,
            int lineNumber,
            String[] row,
            int expectedColumnCount
    ) {
        if (row.length != expectedColumnCount) {
            throw new IllegalArgumentException(
                    "Inconsistent column count in file "
                            + file
                            + " at line "
                            + lineNumber
                            + ". Expected "
                            + expectedColumnCount
                            + " columns but found "
                            + row.length
                            + "."
            );
        }
    }

    private int parseColumnIndex(
            String value,
            String argumentName,
            Path file
    ) {
        try {
            return Integer.parseInt(
                    value.trim()
            );
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    argumentName
                            + " must be a zero-based integer column "
                            + "index when hasHeader is false. Received '"
                            + value
                            + "' for file "
                            + file
                            + ".",
                    e
            );
        }
    }

    private void validateColumnIndex(
            int index,
            int columnCount,
            String argumentName,
            Path file
    ) {
        if (index < 0 || index >= columnCount) {
            throw new IllegalArgumentException(
                    argumentName
                            + " index "
                            + index
                            + " is outside the valid range [0, "
                            + (columnCount - 1)
                            + "] for file "
                            + file
                            + "."
            );
        }
    }

    private static String normalizeSeparator(
            String separator
    ) {
        return switch (separator) {
            case "\\t" -> "\t";
            case "\\n" -> "\n";
            case "\\r" -> "\r";
            default -> separator;
        };
    }

    private static String normalizeNullableString(
            String value
    ) {
        if (value == null) {
            return null;
        }

        String trimmed =
                value.trim();

        if (trimmed.isEmpty()
                || trimmed.equalsIgnoreCase("None")) {

            return null;
        }

        return trimmed;
    }

    private void validateStandardizationConfiguration() {
        if (standardizationStats == null) {
            return;
        }

        if (!isNumeric) {
            throw new IllegalArgumentException(
                    "Standardization statistics cannot be applied by "
                            + "PerFileDelimitedSeriesReader when isNumeric=false."
            );
        }

        /*
         * Empty feature columns are valid for delimited readers. In that case,
         * dimensions are aligned positionally and the final dimension count is
         * validated when the realized series is transformed.
         */
        if (!featureColumns.isEmpty()) {
            standardizationStats.validateFeatureCompatibility(
                    featureColumns
            );
        }
    }

    private static final class ColumnSelection {

        private final int columnCount;
        private final int timeColumnIndex;
        private final int[] featureIndices;

        private ColumnSelection(
                int columnCount,
                int timeColumnIndex,
                int[] featureIndices
        ) {
            this.columnCount =
                    columnCount;

            this.timeColumnIndex =
                    timeColumnIndex;

            this.featureIndices =
                    featureIndices.clone();
        }
    }
}