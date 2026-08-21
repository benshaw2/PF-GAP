package datasets.readers.lazy;

import datasets.readers.ReaderType;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Serializable configuration needed to reconstruct a LazySeriesReader.
 *
 * Runtime reader objects are not serialized. A saved model stores these
 * specifications and uses LazySeriesReaderFactory to recreate the readers
 * after model loading.
 */
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

    public LazySeriesReaderSpec(
            String readerKey,
            ReaderType readerType,
            String timeColumn,
            List<String> featureColumns,
            boolean numeric,
            boolean hasMissingValues,
            String entrySeparator,
            boolean hasHeader
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
                false
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
}