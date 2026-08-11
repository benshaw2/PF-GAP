package datasets.readers;

import core.AppContext;
import datasets.ListObjectDataset;
import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import org.apache.commons.lang3.time.DurationFormatUtils;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Reader for HDF5 / .h5 files.
 *
 * This reader assumes the HDF5 file contains a fixed-size array dataset,
 * usually something like:
 *
 *      /X
 *      /y
 *
 * Supported data ranks:
 *
 * Rank 2:
 *      [N, T] or [N, D]
 *
 *      Each instance becomes:
 *          double[]
 *          Double[]
 *          Object[]
 *
 * Rank 3:
 *      [N, D, T] by default
 *      [N, T, D] if hdf5Layout = "NTD"
 *
 *      Each instance becomes:
 *          double[][]
 *          Double[][]
 *          Object[][]
 *
 *      Output orientation is always:
 *
 *          dimension x time
 *
 * Rank 4:
 *      [N, H, W, C] by default
 *      [N, C, H, W] if hdf5Layout = "NCHW"
 *
 *      Each instance becomes:
 *
 *          double[pixels][channels]
 *          Double[pixels][channels]
 *          Object[pixels][channels]
 *
 *      where pixels = H * W.
 *
 *      This intentionally treats each pixel as a channel vector rather than
 *      forcing image data into a time-series representation.
 *
 * Labels:
 *
 *      No label dataset:
 *          label = null
 *
 *      Label dataset shape *          scalar Object label
 *
 *      Label dataset shape [N, K]:
 *          List<Object> label
 *
 * Numeric hints:
 *
 *      isNumeric = true, hasMissingValues = false:
 *          try to produce primitive double arrays.
 *
 *      isNumeric = true, hasMissingValues = true:
 *          produce boxed Double arrays.
 *
 *      isNumeric = false:
 *          produce Object arrays.
 */
public class HDF5Reader implements DatasetReader {

    private static final String DEFAULT_LAYOUT = "AUTO";
    private static final boolean DEFAULT_COLLAPSE_SINGLETON_DIMENSION = false;

    private final String dataFileName;
    private final String hdf5DatasetPath;
    private final String hdf5LabelDatasetPath;

    private final boolean isNumeric;
    private final boolean hasMissingValues;
    private final boolean isRegression;

    /*
     * Supported values:
     *
     *      AUTO
     *      NT
     *      NDT
     *      NTD
     *      NHWC
     *      NCHW
     *
     * ReaderOptions does not expose this yet, so the ReaderOptions constructor
     * currently defaults to AUTO. A direct constructor is provided for future
     * use.
     */
    private final String hdf5Layout;

    /*
     * If true:
     *
     *      [N, 1, T] -> double[] / Double[] / Object[]
     *
     * If false:
     *
     *      [N, 1, T] -> double[][] / Double[][] / Object[][]
     *
     * Default is false to preserve declared HDF5 rank.
     */
    private final boolean collapseSingletonDimension;

    public HDF5Reader(ReaderOptions options) {
        this(
                options.getDataPath(),
                options.getHdf5DatasetPath(),
                options.getHdf5LabelDatasetPath(),
                options.isNumeric(),
                options.hasMissingValues(),
                options.isRegression(),
                DEFAULT_LAYOUT,
                DEFAULT_COLLAPSE_SINGLETON_DIMENSION
        );
    }

    public HDF5Reader(
            String dataFileName,
            String hdf5DatasetPath,
            String hdf5LabelDatasetPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression
    ) {
        this(
                dataFileName,
                hdf5DatasetPath,
                hdf5LabelDatasetPath,
                isNumeric,
                hasMissingValues,
                isRegression,
                DEFAULT_LAYOUT,
                DEFAULT_COLLAPSE_SINGLETON_DIMENSION
        );
    }

    public HDF5Reader(
            String dataFileName,
            String hdf5DatasetPath,
            String hdf5LabelDatasetPath,
            boolean isNumeric,
            boolean hasMissingValues,
            boolean isRegression,
            String hdf5Layout,
            boolean collapseSingletonDimension
    ) {
        this.dataFileName = dataFileName;
        this.hdf5DatasetPath = hdf5DatasetPath;
        this.hdf5LabelDatasetPath = hdf5LabelDatasetPath;
        this.isNumeric = isNumeric;
        this.hasMissingValues = hasMissingValues;
        this.isRegression = isRegression;
        this.hdf5Layout =
                hdf5Layout == null || hdf5Layout.trim().isEmpty()
                        ? DEFAULT_LAYOUT
                        : hdf5Layout.trim().toUpperCase();
        this.collapseSingletonDimension = collapseSingletonDimension;
    }

    @Override
    public ListObjectDataset read() throws IOException {

        validateOptions();

        long start = System.nanoTime();

        ListObjectDataset dataset = new ListObjectDataset();

        try (HdfFile hdfFile = new HdfFile(Paths.get(dataFileName))) {

            Dataset xDataset = hdfFile.getDatasetByPath(hdf5DatasetPath);

            if (xDataset == null) {
                throw new IllegalArgumentException(
                        "HDF5 dataset not found: " + hdf5DatasetPath
                );
            }

            Object rawData = xDataset.getData();
            int[] dataShape = toIntShape(xDataset.getDimensions());

            if (dataShape.length < 2 || dataShape.length > 4) {
                throw new IllegalArgumentException(
                        "HDF5Reader supports data ranks 2, 3, and 4. Found rank "
                                + dataShape.length
                                + " for dataset "
                                + hdf5DatasetPath
                );
            }

            int nInstances = dataShape[0];

            Object rawLabels = null;
            int[] labelShape = null;

            if (hdf5LabelDatasetPath != null
                    && !hdf5LabelDatasetPath.trim().isEmpty()) {

                Dataset yDataset =
                        hdfFile.getDatasetByPath(hdf5LabelDatasetPath);

                if (yDataset == null) {
                    throw new IllegalArgumentException(
                            "HDF5 label dataset not found: "
                                    + hdf5LabelDatasetPath
                    );
                }

                rawLabels = yDataset.getData();
                labelShape = toIntShape(yDataset.getDimensions());

                validateLabelShape(labelShape, nInstances);
            }

            for (int i = 0; i < nInstances; i++) {

                Object data = buildInstance(rawData, dataShape, i);
                Object label = buildLabel(rawLabels, labelShape, i);

                dataset.add(label, data, i);

                updateGlobalLength(data);

                ProgressLogger.logProgress(i);
            }
        }

        long end = System.nanoTime();
        ProgressLogger.logDuration(start, end);

        return dataset;
    }

    private Object buildInstance(
            Object rawData,
            int[] shape,
            int instanceIndex
    ) {

        switch (shape.length) {
            case 2:
                return buildRank2Instance(rawData, shape, instanceIndex);

            case 3:
                return buildRank3Instance(rawData, shape, instanceIndex);

            case 4:
                return buildRank4Instance(rawData, shape, instanceIndex);

            default:
                throw new IllegalArgumentException(
                        "Unsupported HDF5 data rank: " + shape.length
                );
        }
    }

    /**
     * Rank 2:
     *
     *      [N, T] or [N, D]
     *
     * In either case, PFGAP sees one 1D array per instance.
     */
    private Object buildRank2Instance(
            Object rawData,
            int[] shape,
            int instanceIndex
    ) {

        int length = shape[1];

        if (isNumeric) {

            if (hasMissingValues) {
                Double[] data = new Double[length];

                for (int t = 0; t < length; t++) {
                    data[t] = toBoxedDouble(
                            getValue(rawData, instanceIndex, t)
                    );
                }

                return data;
            }

            double[] data = new double[length];

            for (int t = 0; t < length; t++) {
                data[t] = toPrimitiveDouble(
                        getValue(rawData, instanceIndex, t)
                );
            }

            return data;
        }

        Object[] data = new Object[length];

        for (int t = 0; t < length; t++) {
            data[t] = parseGenericValue(
                    getValue(rawData, instanceIndex, t)
            );
        }

        return data;
    }

    /**
     * Rank 3:
     *
     * Default/AUTO/NDT:
     *
     *      [N, D, T]
     *
     * NTD:
     *
     *      [N, T, D]
     *
     * Output orientation:
     *
     *      dimension x time
     */
    private Object buildRank3Instance(
            Object rawData,
            int[] shape,
            int instanceIndex
    ) {

        boolean isNTD = hdf5Layout.equals("NTD");

        int dimensionCount;
        int timeLength;

        if (isNTD) {
            timeLength = shape[1];
            dimensionCount = shape[2];
        } else {
            dimensionCount = shape[1];
            timeLength = shape[2];
        }

        if (collapseSingletonDimension && dimensionCount == 1) {
            return buildCollapsedRank3Instance(
                    rawData,
                    shape,
                    instanceIndex,
                    isNTD,
                    timeLength
            );
        }

        if (isNumeric) {

            if (hasMissingValues) {
                Double[][] data = new Double[dimensionCount][timeLength];

                for (int d = 0; d < dimensionCount; d++) {
                    for (int t = 0; t < timeLength; t++) {
                        Object value =
                                isNTD
                                        ? getValue(rawData, instanceIndex, t, d)
                                        : getValue(rawData, instanceIndex, d, t);

                        data[d][t] = toBoxedDouble(value);
                    }
                }

                return data;
            }

            double[][] data = new double[dimensionCount][timeLength];

            for (int d = 0; d < dimensionCount; d++) {
                for (int t = 0; t < timeLength; t++) {
                    Object value =
                            isNTD
                                    ? getValue(rawData, instanceIndex, t, d)
                                    : getValue(rawData, instanceIndex, d, t);

                    data[d][t] = toPrimitiveDouble(value);
                }
            }

            return data;
        }

        Object[][] data = new Object[dimensionCount][timeLength];

        for (int d = 0; d < dimensionCount; d++) {
            for (int t = 0; t < timeLength; t++) {
                Object value =
                        isNTD
                                ? getValue(rawData, instanceIndex, t, d)
                                : getValue(rawData, instanceIndex, d, t);

                data[d][t] = parseGenericValue(value);
            }
        }

        return data;
    }

    private Object buildCollapsedRank3Instance(
            Object rawData,
            int[] shape,
            int instanceIndex,
            boolean isNTD,
            int timeLength
    ) {

        if (isNumeric) {

            if (hasMissingValues) {
                Double[] data = new Double[timeLength];

                for (int t = 0; t < timeLength; t++) {
                    Object value =
                            isNTD
                                    ? getValue(rawData, instanceIndex, t, 0)
                                    : getValue(rawData, instanceIndex, 0, t);

                    data[t] = toBoxedDouble(value);
                }

                return data;
            }

            double[] data = new double[timeLength];

            for (int t = 0; t < timeLength; t++) {
                Object value =
                        isNTD
                                ? getValue(rawData, instanceIndex, t, 0)
                                : getValue(rawData, instanceIndex, 0, t);

                data[t] = toPrimitiveDouble(value);
            }

            return data;
        }

        Object[] data = new Object[timeLength];

        for (int t = 0; t < timeLength; t++) {
            Object value =
                    isNTD
                            ? getValue(rawData, instanceIndex, t, 0)
                            : getValue(rawData, instanceIndex, 0, t);

            data[t] = parseGenericValue(value);
        }

        return data;
    }

    /**
     * Rank 4:
     *
     * Default/AUTO/NHWC:
     *
     *      [N, H, W, C]
     *
     * NCHW:
     *
     *      [N, C, H, W]
     *
     * Output:
     *
     *      pixels x channels
     *
     * where:
     *
     *      pixels = H * W
     *
     * This intentionally represents each pixel as a channel vector.
     */
    private Object buildRank4Instance(
            Object rawData,
            int[] shape,
            int instanceIndex
    ) {

        boolean isNCHW = hdf5Layout.equals("NCHW");

        int height;
        int width;
        int channels;

        if (isNCHW) {
            channels = shape[1];
            height = shape[2];
            width = shape[3];
        } else {
            height = shape[1];
            width = shape[2];
            channels = shape[3];
        }

        int pixels = height * width;

        if (isNumeric) {

            if (hasMissingValues) {
                Double[][] data = new Double[pixels][channels];

                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int pixelIndex = h * width + w;

                        for (int c = 0; c < channels; c++) {
                            Object value =
                                    isNCHW
                                            ? getValue(rawData, instanceIndex, c, h, w)
                                            : getValue(rawData, instanceIndex, h, w, c);

                            data[pixelIndex][c] = toBoxedDouble(value);
                        }
                    }
                }

                return data;
            }

            double[][] data = new double[pixels][channels];

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int pixelIndex = h * width + w;

                    for (int c = 0; c < channels; c++) {
                        Object value =
                                isNCHW
                                        ? getValue(rawData, instanceIndex, c, h, w)
                                        : getValue(rawData, instanceIndex, h, w, c);

                        data[pixelIndex][c] = toPrimitiveDouble(value);
                    }
                }
            }

            return data;
        }

        Object[][] data = new Object[pixels][channels];

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int pixelIndex = h * width + w;

                for (int c = 0; c < channels; c++) {
                    Object value =
                            isNCHW
                                    ? getValue(rawData, instanceIndex, c, h, w)
                                    : getValue(rawData, instanceIndex, h, w, c);

                    data[pixelIndex][c] = parseGenericValue(value);
                }
            }
        }

        return data;
    }

    private Object buildLabel(
            Object rawLabels,
            int[] labelShape,
            int instanceIndex
    ) {

        if (rawLabels == null || labelShape == null) {
            return null;
        }

        if (labelShape.length == 1) {
            return parseLabelValue(
                    getValue(rawLabels, instanceIndex)
            );
        }

        if (labelShape.length == 2) {
            int labelCount = labelShape[1];

            List<Object> labels = new ArrayList<>();

            for (int k = 0; k < labelCount; k++) {
                labels.add(
                        parseLabelValue(
                                getValue(rawLabels, instanceIndex, k)
                        )
                );
            }

            return labels;
        }

        throw new IllegalArgumentException(
                "Unsupported HDF5 label rank: " + labelShape.length
        );
    }

    private Object parseLabelValue(Object value) {

        Object normalized = normalizeValue(value);

        if (normalized == null) {
            return null;
        }

        if (isRegression) {
            return toPrimitiveDouble(normalized);
        }

        if (normalized instanceof Integer) {
            return normalized;
        }

        if (normalized instanceof Long) {
            long longValue = (Long) normalized;

            if (longValue >= Integer.MIN_VALUE
                    && longValue <= Integer.MAX_VALUE) {
                return (int) longValue;
            }

            return longValue;
        }

        if (normalized instanceof Number) {
            double doubleValue = ((Number) normalized).doubleValue();

            if (doubleValue == Math.rint(doubleValue)
                    && doubleValue >= Integer.MIN_VALUE
                    && doubleValue <= Integer.MAX_VALUE) {
                return (int) doubleValue;
            }

            return doubleValue;
        }

        String trimmed = normalized.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        try {
            return Integer.parseInt(trimmed);
        } catch (NumberFormatException ignored) {
            try {
                double parsed = Double.parseDouble(trimmed);

                if (parsed == Math.rint(parsed)
                        && parsed >= Integer.MIN_VALUE
                        && parsed <= Integer.MAX_VALUE) {
                    return (int) parsed;
                }

                return parsed;
            } catch (NumberFormatException ignoredAgain) {
                return trimmed;
            }
        }
    }

    private static Object parseGenericValue(Object value) {

        Object normalized = normalizeValue(value);

        if (normalized == null) {
            return null;
        }

        if (normalized instanceof String) {
            String trimmed = normalized.toString().trim();

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

    private static Double toBoxedDouble(Object value) {

        Object normalized = normalizeValue(value);

        if (normalized == null) {
            return null;
        }

        if (normalized instanceof Number) {
            return ((Number) normalized).doubleValue();
        }

        if (normalized instanceof Boolean) {
            return ((Boolean) normalized) ? 1.0 : 0.0;
        }

        String trimmed = normalized.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {
            return null;
        }

        return Double.valueOf(trimmed);
    }

    private static double toPrimitiveDouble(Object value) {

        Object normalized = normalizeValue(value);

        if (normalized == null) {
            throw new IllegalArgumentException(
                    "Encountered null numeric value, but hasMissingValues=false."
            );
        }

        if (normalized instanceof Number) {
            return ((Number) normalized).doubleValue();
        }

        if (normalized instanceof Boolean) {
            return ((Boolean) normalized) ? 1.0 : 0.0;
        }

        String trimmed = normalized.toString().trim();

        if (MissingValueParser.isMissing(trimmed)) {
            throw new IllegalArgumentException(
                    "Encountered missing numeric value, but hasMissingValues=false."
            );
        }

        return Double.parseDouble(trimmed);
    }

    /**
     * Converts HDF5/JVM-specific scalar representations into ordinary
     * Java objects where possible.
     */
    private static Object normalizeValue(Object value) {

        if (value == null) {
            return null;
        }

        if (value instanceof byte[]) {
            byte[] bytes = (byte[]) value;

            return new String(bytes, StandardCharsets.UTF_8).trim();
        }

        if (value instanceof Character) {
            return value.toString();
        }

        return value;
    }

    /**
     * Reflectively indexes into primitive or Object multi-dimensional arrays.
     *
     * This avoids having to special-case every possible primitive array type,
     * such as double[][], float[][], int[][], etc.
     */
    private static Object getValue(Object array, int... indices) {

        Object current = array;

        for (int index : indices) {
            current = Array.get(current, index);
        }

        return current;
    }

    private void validateOptions() {

        if (dataFileName == null || dataFileName.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    "HDF5Reader requires dataFileName."
            );
        }

        if (hdf5DatasetPath == null || hdf5DatasetPath.trim().isEmpty()) {
            throw new IllegalArgumentException(
                    "HDF5Reader requires hdf5DatasetPath."
            );
        }

        validateLayout();
    }

    private void validateLayout() {

        String layout = hdf5Layout;

        if (layout.equals("AUTO")
                || layout.equals("NT")
                || layout.equals("NDT")
                || layout.equals("NTD")
                || layout.equals("NHWC")
                || layout.equals("NCHW")) {
            return;
        }

        throw new IllegalArgumentException(
                "Unsupported hdf5Layout: "
                        + hdf5Layout
                        + ". Use AUTO, NT, NDT, NTD, NHWC, or NCHW."
        );
    }

    private static void validateLabelShape(
            int[] labelShape,
            int nInstances
    ) {

        if (labelShape.length != 1 && labelShape.length != 2) {
            throw new IllegalArgumentException(
                    "HDF5Reader supports label ranks 1 and 2. Found rank "
                            + labelShape.length
            );
        }

        if (labelShape[0] != nInstances) {
            throw new IllegalArgumentException(
                    "Label dataset first dimension does not match data. "
                            + "Data instances: "
                            + nInstances
                            + ", label instances: "
                            + labelShape[0]
            );
        }
    }

    private static int[] toIntShape(long[] dimensions) {

        int[] shape = new int[dimensions.length];

        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "HDF5 dimension too large for Java int indexing: "
                                + dimensions[i]
                );
            }

            shape[i] = (int) dimensions[i];
        }

        return shape;
    }

    private static int[] toIntShape(int[] dimensions) {

        int[] shape = new int[dimensions.length];

        System.arraycopy(dimensions, 0, shape, 0, dimensions.length);

        return shape;
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