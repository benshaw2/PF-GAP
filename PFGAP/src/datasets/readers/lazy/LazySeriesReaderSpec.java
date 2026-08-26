package datasets.readers.lazy;

import datasets.readers.ReaderType;
import preprocessing.standardization.StandardizationStats;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Serializable configuration needed to reconstruct a LazySeriesReader.
 *
 * <p>Runtime reader objects are not serialized. A saved model stores these
 * specifications and uses LazySeriesReaderFactory to recreate the readers
 * after model loading.</p>
 *
 * <p>The {@code initialTimeCapacity} field was added after the original
 * serialized form was established. Older serialized specifications do not
 * contain this field, so Java deserialization initializes it to zero. The
 * public getter interprets a nonpositive stored value as the backward-
 * compatible default capacity.</p>
 */
public final class LazySeriesReaderSpec
        implements Serializable {

    @Serial
    private static final long serialVersionUID =
            1L;

    /**
     * Default initial capacity used by per-file delimited series readers.
     *
     * <p>This is an allocation hint only. It does not constrain the final
     * number of time points in a materialized series.</p>
     */
    public static final int DEFAULT_INITIAL_TIME_CAPACITY =
            256;

    private final String readerKey;
    private final ReaderType readerType;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean numeric;
    private final boolean hasMissingValues;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final StandardizationStats standardizationStats;

    /**
     * Initial allocation capacity for each materialized time-series
     * dimension.
     *
     * <p>For specifications serialized before this field was introduced,
     * Java deserialization assigns zero. Callers must access this value
     * through {@link #getInitialTimeCapacity()}, which converts a nonpositive
     * stored value to {@link #DEFAULT_INITIAL_TIME_CAPACITY}.</p>
     */
    private final int initialTimeCapacity;

    /**
     * Full constructor including the initial time capacity.
     *
     * <p>New specialized numeric lazy readers should use this constructor.</p>
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            String entrySeparator,
            boolean hasHeader,
            StandardizationStats standardizationStats,
            int initialTimeCapacity
    ) {
        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec requires a non-empty readerKey."
            );
        }

        this.readerType =
                Objects.requireNonNull(
                        readerType,
                        "LazySeriesReaderSpec requires readerType."
                );

        if (initialTimeCapacity < 1) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec initialTimeCapacity must be "
                            + "at least 1. Received: "
                            + initialTimeCapacity
                            + "."
            );
        }

        this.readerKey =
                readerKey.trim();

        this.timeColumn =
                normalizeNullableString(
                        timeColumn
                );

        this.featureColumns =
                copyFeatureColumns(
                        featureColumns
                );

        this.numeric =
                numeric;

        this.hasMissingValues =
                hasMissingValues;

        this.entrySeparator =
                entrySeparator;

        this.hasHeader =
                hasHeader;

        this.standardizationStats =
                standardizationStats;

        this.initialTimeCapacity =
                initialTimeCapacity;
    }

    /**
     * Backward-compatible constructor retaining the original full signature.
     *
     * <p>Existing general lazy-reader call sites can continue using this
     * constructor. It delegates to the new full constructor with the default
     * initial capacity.</p>
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            String entrySeparator,
            boolean hasHeader,
            StandardizationStats standardizationStats
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                entrySeparator,
                hasHeader,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    /**
     * Backward-compatible constructor for reader specifications that do not
     * require delimited-reader configuration.
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                null,
                false,
                standardizationStats,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    /**
     * Backward-compatible constructor for reader specifications without
     * delimited-reader or standardization configuration.
     */
    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                null,
                false,
                null,
                DEFAULT_INITIAL_TIME_CAPACITY
        );
    }

    public String getReaderKey() {
        return readerKey;
    }

    public ReaderType getReaderType() {
        return readerType;
    }

    public String getTimeColumn() {
        return timeColumn;
    }

    public List<String> getFeatureColumns() {
        return Collections.unmodifiableList(
                featureColumns
        );
    }

    public boolean isNumeric() {
        return numeric;
    }

    public boolean hasMissingValues() {
        return hasMissingValues;
    }

    public String getEntrySeparator() {
        return entrySeparator;
    }

    public boolean hasHeader() {
        return hasHeader;
    }

    public StandardizationStats getStandardizationStats() {
        return standardizationStats;
    }

    public boolean hasStandardizationStats() {
        return standardizationStats != null;
    }

    /**
     * Returns the initial time capacity used when reconstructing a compatible
     * per-file series reader.
     *
     * <p>A value of zero can occur when an older serialized specification is
     * loaded because the older serialized form did not contain this field.
     * In that case, the original default of 256 is returned.</p>
     *
     * @return a positive initial capacity
     */
    public int getInitialTimeCapacity() {
        return initialTimeCapacity > 0
                ? initialTimeCapacity
                : DEFAULT_INITIAL_TIME_CAPACITY;
    }

    /**
     * Indicates whether the serialized specification contains an explicit
     * positive initial time capacity.
     *
     * <p>This is mainly useful for diagnostics. Reader reconstruction should
     * normally call {@link #getInitialTimeCapacity()} directly.</p>
     */
    public boolean hasExplicitInitialTimeCapacity() {
        return initialTimeCapacity > 0;
    }

    private static List<String> copyFeatureColumns(
            List<String> featureColumns
    ) {
        if (featureColumns == null || featureColumns.isEmpty()) {
            return new ArrayList<>();
        }

        return new ArrayList<>(
                featureColumns
        );
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

    @Override
    public String toString() {
        return "LazySeriesReaderSpec{"
                + "readerKey='"
                + readerKey
                + '\''
                + ", readerType="
                + readerType
                + ", timeColumn='"
                + timeColumn
                + '\''
                + ", featureColumns="
                + featureColumns
                + ", numeric="
                + numeric
                + ", hasMissingValues="
                + hasMissingValues
                + ", entrySeparator='"
                + printableSeparator(entrySeparator)
                + '\''
                + ", hasHeader="
                + hasHeader
                + ", hasStandardizationStats="
                + hasStandardizationStats()
                + ", initialTimeCapacity="
                + getInitialTimeCapacity()
                + ", explicitInitialTimeCapacity="
                + hasExplicitInitialTimeCapacity()
                + '}';
    }

    private static String printableSeparator(
            String separator
    ) {
        if (separator == null) {
            return null;
        }

        return switch (separator) {
            case "\t" -> "\\t";
            case "\n" -> "\\n";
            case "\r" -> "\\r";
            default -> separator;
        };
    }
}

/*package datasets.readers.lazy;

import datasets.readers.ReaderType;
import preprocessing.standardization.StandardizationStats;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

//Serializable configuration needed to reconstruct a LazySeriesReader.
//
//Runtime reader objects are not serialized. A saved model stores these
//specifications and uses LazySeriesReaderFactory to recreate the readers
//after model loading.
public final class LazySeriesReaderSpec implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final String readerKey;
    private final ReaderType readerType;
    private final String timeColumn;
    private final List<String> featureColumns;
    private final boolean numeric;
    private final boolean hasMissingValues;
    private final String entrySeparator;
    private final boolean hasHeader;
    private final StandardizationStats standardizationStats;
    private final int initialTimeCapacity;

    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            String entrySeparator,
            boolean hasHeader,
            StandardizationStats standardizationStats
    ) {
        if (readerKey == null || readerKey.isBlank()) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec requires a non-empty readerKey."
            );
        }

        this.readerType = Objects.requireNonNull(
                readerType,
                "LazySeriesReaderSpec requires readerType."
        );

        this.readerKey = readerKey;
        this.timeColumn = timeColumn;

        this.featureColumns =
                featureColumns == null
                        ? new ArrayList<>()
                        : new ArrayList<>(featureColumns);

        this.numeric = numeric;
        this.hasMissingValues = hasMissingValues;
        this.entrySeparator = entrySeparator;
        this.hasHeader = hasHeader;
        this.standardizationStats = standardizationStats;
    }

    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            StandardizationStats standardizationStats
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                null,
                false,
                standardizationStats
        );
    }

    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues
    ) {
        this(
                readerKey,
                readerType,
                timeColumn,
                featureColumns,
                numeric,
                hasMissingValues,
                null,
                false,
                null
        );
    }

    public String getReaderKey() {
        return readerKey;
    }

    public ReaderType getReaderType() {
        return readerType;
    }

    public String getTimeColumn() {
        return timeColumn;
    }

    public List<String> getFeatureColumns() {
        return Collections.unmodifiableList(featureColumns);
    }

    public boolean isNumeric() {
        return numeric;
    }

    public boolean hasMissingValues() {
        return hasMissingValues;
    }

    public String getEntrySeparator() {
        return entrySeparator;
    }

    public boolean hasHeader() {
        return hasHeader;
    }

    public StandardizationStats getStandardizationStats() {
        return standardizationStats;
    }

    public boolean hasStandardizationStats() {
        return standardizationStats != null;
    }

    @Override
    public String toString() {
        return "LazySeriesReaderSpec{"
                + "readerKey='" + readerKey + '\''
                + ", readerType=" + readerType
                + ", timeColumn='" + timeColumn + '\''
                + ", featureColumns=" + featureColumns
                + ", numeric=" + numeric
                + ", hasMissingValues=" + hasMissingValues
                + '}';
    }
}*/