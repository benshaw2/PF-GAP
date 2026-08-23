package datasets.readers;

import preprocessing.standardization.StandardizationStats;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Configuration object used by DatasetReaderFactory to construct
 * concrete DatasetReader implementations.
 *
 * ReaderOptions intentionally contains both common reader options and
 * some format-specific options. Not every field is meaningful for every
 * ReaderType.
 *
 * Examples:
 *
 * DELIMITED uses:
 *
 *      readerType
 *      dataPath
 *      labelPath
 *      entrySeparator
 *      arraySeparator
 *      hasHeader
 *      is2D
 *      isNumeric
 *      hasMissingValues
 *      targetColumnIsFirst
 *      isTest
 *      isRegression
 *
 * TS uses:
 *
 *      readerType
 *      dataPath
 *      labelPath
 *      isNumeric
 *      hasMissingValues
 *      isRegression
 *
 * LONG_FORMAT_DELIMITED and LONG_FORMAT_PARQUET will use:
 *
 *      readerType
 *      dataPath
 *      idColumn
 *      timeColumn
 *      featureColumns
 *      labelColumns
 *      entrySeparator
 *      hasHeader
 *      isNumeric
 *      hasMissingValues
 *      isRegression
 */
public class ReaderOptions {

    private ReaderType readerType;

    private String dataPath;
    private String labelPath;

    private String entrySeparator = ",";
    private String arraySeparator = ":";

    private boolean hasHeader = false;
    private boolean is2D = false;
    private boolean isNumeric = true;
    private boolean hasMissingValues = false;
    private boolean targetColumnIsFirst = false;
    private boolean isTest = false;
    private boolean isRegression = false;

    /*
     * Long-format options.
     *
     * idColumn:
     *      Column identifying which rows belong to the same instance.
     *
     * timeColumn:
     *      Optional column used to sort rows within each instance.
     *      If null, input row order should be preserved.
     *
     * featureColumns:
     *      Columns to treat as time-varying features.
     *
     * labelColumns:
     *      Columns to use as labels. Multiple label columns support
     *      future multi-label or multi-target learning.
     */
    private String idColumn;
    private String timeColumn;
    private List<String> featureColumns = new ArrayList<>();
    private List<String> labelColumns = new ArrayList<>();

    /*
     * Directory-reader options.
     *
     * These are placeholders for the upcoming DirectoryReader.
     */
    private String filePattern;
    private boolean recursive = false;
    private ReaderType innerReaderType;

    /*
     * HDF5-reader options.
     *
     * These are placeholders for the upcoming HDF5Reader.
     */
    private String hdf5DatasetPath;
    private String hdf5LabelDatasetPath;
    private StandardizationStats standardizationStats;

    public ReaderOptions() {
    }

    public ReaderOptions(ReaderType readerType, String dataPath) {
        this.readerType = readerType;
        this.dataPath = dataPath;
    }

    public ReaderType getReaderType() {
        return readerType;
    }

    public ReaderOptions setReaderType(ReaderType readerType) {
        this.readerType = readerType;
        return this;
    }

    public String getDataPath() {
        return dataPath;
    }

    public ReaderOptions setDataPath(String dataPath) {
        this.dataPath = dataPath;
        return this;
    }

    public String getLabelPath() {
        return labelPath;
    }

    public ReaderOptions setLabelPath(String labelPath) {
        this.labelPath = labelPath;
        return this;
    }

    public String getEntrySeparator() {
        return entrySeparator;
    }

    public ReaderOptions setEntrySeparator(String entrySeparator) {
        this.entrySeparator = entrySeparator;
        return this;
    }

    public String getArraySeparator() {
        return arraySeparator;
    }

    public ReaderOptions setArraySeparator(String arraySeparator) {
        this.arraySeparator = arraySeparator;
        return this;
    }

    public boolean hasHeader() {
        return hasHeader;
    }

    public ReaderOptions setHasHeader(boolean hasHeader) {
        this.hasHeader = hasHeader;
        return this;
    }

    public boolean is2D() {
        return is2D;
    }

    public ReaderOptions set2D(boolean is2D) {
        this.is2D = is2D;
        return this;
    }

    public boolean isNumeric() {
        return isNumeric;
    }

    public ReaderOptions setNumeric(boolean numeric) {
        isNumeric = numeric;
        return this;
    }

    public boolean hasMissingValues() {
        return hasMissingValues;
    }

    public ReaderOptions setHasMissingValues(boolean hasMissingValues) {
        this.hasMissingValues = hasMissingValues;
        return this;
    }

    public boolean targetColumnIsFirst() {
        return targetColumnIsFirst;
    }

    public ReaderOptions setTargetColumnIsFirst(boolean targetColumnIsFirst) {
        this.targetColumnIsFirst = targetColumnIsFirst;
        return this;
    }

    public boolean isTest() {
        return isTest;
    }

    public ReaderOptions setTest(boolean test) {
        isTest = test;
        return this;
    }

    public boolean isRegression() {
        return isRegression;
    }

    public ReaderOptions setRegression(boolean regression) {
        isRegression = regression;
        return this;
    }

    public String getIdColumn() {
        return idColumn;
    }

    public ReaderOptions setIdColumn(String idColumn) {
        this.idColumn = idColumn;
        return this;
    }

    public String getTimeColumn() {
        return timeColumn;
    }

    public ReaderOptions setTimeColumn(String timeColumn) {
        this.timeColumn = timeColumn;
        return this;
    }

    public List<String> getFeatureColumns() {
        return Collections.unmodifiableList(featureColumns);
    }

    public ReaderOptions setFeatureColumns(List<String> featureColumns) {
        this.featureColumns = copyStringList(featureColumns);
        return this;
    }

    public ReaderOptions addFeatureColumn(String featureColumn) {
        Objects.requireNonNull(featureColumn, "featureColumn cannot be null.");
        this.featureColumns.add(featureColumn);
        return this;
    }

    public List<String> getLabelColumns() {
        return Collections.unmodifiableList(labelColumns);
    }

    public ReaderOptions setLabelColumns(List<String> labelColumns) {
        this.labelColumns = copyStringList(labelColumns);
        return this;
    }

    public ReaderOptions addLabelColumn(String labelColumn) {
        Objects.requireNonNull(labelColumn, "labelColumn cannot be null.");
        this.labelColumns.add(labelColumn);
        return this;
    }

    public String getFilePattern() {
        return filePattern;
    }

    public ReaderOptions setFilePattern(String filePattern) {
        this.filePattern = filePattern;
        return this;
    }

    public boolean isRecursive() {
        return recursive;
    }

    public ReaderOptions setRecursive(boolean recursive) {
        this.recursive = recursive;
        return this;
    }

    public ReaderType getInnerReaderType() {
        return innerReaderType;
    }

    public ReaderOptions setInnerReaderType(ReaderType innerReaderType) {
        this.innerReaderType = innerReaderType;
        return this;
    }

    public String getHdf5DatasetPath() {
        return hdf5DatasetPath;
    }

    public ReaderOptions setHdf5DatasetPath(String hdf5DatasetPath) {
        this.hdf5DatasetPath = hdf5DatasetPath;
        return this;
    }

    public String getHdf5LabelDatasetPath() {
        return hdf5LabelDatasetPath;
    }

    public ReaderOptions setHdf5LabelDatasetPath(String hdf5LabelDatasetPath) {
        this.hdf5LabelDatasetPath = hdf5LabelDatasetPath;
        return this;
    }

    public StandardizationStats getStandardizationStats() {
        return standardizationStats;
    }

    public ReaderOptions setStandardizationStats(
            StandardizationStats standardizationStats
    ) {
        this.standardizationStats =
                standardizationStats;

        return this;
    }

    /**
     * Returns a shallow copy of these options.
     *
     * This is useful for train/test settings where most reader options are the
     * same but paths and isTest differ.
     */
    public ReaderOptions copy() {
        return new ReaderOptions()
                .setReaderType(readerType)
                .setDataPath(dataPath)
                .setLabelPath(labelPath)
                .setEntrySeparator(entrySeparator)
                .setArraySeparator(arraySeparator)
                .setHasHeader(hasHeader)
                .set2D(is2D)
                .setNumeric(isNumeric)
                .setHasMissingValues(hasMissingValues)
                .setTargetColumnIsFirst(targetColumnIsFirst)
                .setTest(isTest)
                .setRegression(isRegression)
                .setIdColumn(idColumn)
                .setTimeColumn(timeColumn)
                .setFeatureColumns(featureColumns)
                .setLabelColumns(labelColumns)
                .setFilePattern(filePattern)
                .setRecursive(recursive)
                .setInnerReaderType(innerReaderType)
                .setHdf5DatasetPath(hdf5DatasetPath)
                .setHdf5LabelDatasetPath(hdf5LabelDatasetPath);
    }

    private static List<String> copyStringList(List<String> values) {
        if (values == null) {
            return new ArrayList<>();
        }

        return new ArrayList<>(values);
    }
}