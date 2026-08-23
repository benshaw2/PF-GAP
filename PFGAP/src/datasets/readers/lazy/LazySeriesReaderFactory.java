package datasets.readers.lazy;

import datasets.readers.PerFileDelimitedSeriesReader;
import datasets.readers.PerFileParquetSeriesReader;
import datasets.readers.ReaderType;

public final class LazySeriesReaderFactory {

    private LazySeriesReaderFactory() {
    }

    public static LazySeriesReader create(
            LazySeriesReaderSpec spec
    ) {
        if (spec == null) {
            throw new IllegalArgumentException(
                    "LazySeriesReaderSpec cannot be null."
            );
        }

        ReaderType readerType =
                spec.getReaderType();

        return switch (readerType) {
            case LAZY_PER_FILE_PARQUET ->
                    new PerFileParquetSeriesReader(
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.isNumeric(),
                            spec.hasMissingValues(),
                            spec.getStandardizationStats()
                    );

            case LAZY_PER_FILE_DELIMITED ->
                    new PerFileDelimitedSeriesReader(
                            spec.getEntrySeparator(),
                            spec.hasHeader(),
                            spec.getTimeColumn(),
                            spec.getFeatureColumns(),
                            spec.isNumeric(),
                            spec.hasMissingValues(),
                            spec.getStandardizationStats()
                    );

            default ->
                    throw new UnsupportedOperationException(
                            "Reader type "
                                    + readerType
                                    + " cannot currently be restored "
                                    + "as a LazySeriesReader."
                    );
        };
    }
}