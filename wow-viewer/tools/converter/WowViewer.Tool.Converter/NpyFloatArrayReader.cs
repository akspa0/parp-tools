using System.Buffers.Binary;
using System.Text;
using System.Text.RegularExpressions;

namespace WowViewer.Tool.Converter;

internal static class NpyFloatArrayReader
{
    private static readonly byte[] Magic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];
    private static readonly Regex ShapeRegex = new(@"\((?<rows>\d+)\s*,\s*(?<cols>\d+)\s*,?\)", RegexOptions.Compiled);

    public static float[] ReadMatrix(string path, out int rows, out int cols)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        byte[] data = File.ReadAllBytes(path);
        return ReadMatrix(data, out rows, out cols);
    }

    public static float[] ReadMatrix(byte[] data, out int rows, out int cols)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length < 10 || !data.AsSpan(0, Magic.Length).SequenceEqual(Magic))
            throw new InvalidDataException("The supplied file is not a valid NumPy .npy payload.");

        byte major = data[6];
        byte minor = data[7];
        int headerLength;
        int headerOffset;
        switch (major)
        {
            case 1:
                headerLength = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(8, 2));
                headerOffset = 10;
                break;

            case 2:
            case 3:
                headerLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(8, 4)));
                headerOffset = 12;
                break;

            default:
                throw new InvalidDataException($"Unsupported NumPy header version {major}.{minor}.");
        }

        if (headerOffset + headerLength > data.Length)
            throw new InvalidDataException("NumPy header overruns the available payload.");

        string header = Encoding.ASCII.GetString(data, headerOffset, headerLength);
        bool fortranOrder = header.Contains("'fortran_order': True", StringComparison.Ordinal)
            || header.Contains("\"fortran_order\": True", StringComparison.Ordinal);
        if (fortranOrder)
            throw new InvalidDataException("Fortran-order NumPy arrays are not supported by this terrain patch command.");

        Match shapeMatch = ShapeRegex.Match(header);
        if (!shapeMatch.Success)
            throw new InvalidDataException("NumPy array shape is missing or unsupported.");

        rows = int.Parse(shapeMatch.Groups["rows"].Value, System.Globalization.CultureInfo.InvariantCulture);
        cols = int.Parse(shapeMatch.Groups["cols"].Value, System.Globalization.CultureInfo.InvariantCulture);
        if (rows <= 0 || cols <= 0)
            throw new InvalidDataException("NumPy array shape must be positive.");

        string descr = ReadDescr(header);
        int dataOffset = headerOffset + headerLength;
        int elementCount = checked(rows * cols);
        float[] values = new float[elementCount];

        switch (descr)
        {
            case "<f4":
                if (dataOffset + (elementCount * sizeof(float)) > data.Length)
                    throw new InvalidDataException("NumPy float32 payload is truncated.");

                for (int index = 0; index < elementCount; index++)
                    values[index] = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(dataOffset + (index * sizeof(float)), sizeof(float)));
                break;

            case "<f8":
                if (dataOffset + (elementCount * sizeof(double)) > data.Length)
                    throw new InvalidDataException("NumPy float64 payload is truncated.");

                for (int index = 0; index < elementCount; index++)
                    values[index] = (float)BinaryPrimitives.ReadDoubleLittleEndian(data.AsSpan(dataOffset + (index * sizeof(double)), sizeof(double)));
                break;

            default:
                throw new InvalidDataException($"Unsupported NumPy dtype '{descr}'. Expected '<f4' or '<f8'.");
        }

        return values;
    }

    private static string ReadDescr(string header)
    {
        const string singleQuoteToken = "'descr': '";
        int start = header.IndexOf(singleQuoteToken, StringComparison.Ordinal);
        if (start >= 0)
        {
            start += singleQuoteToken.Length;
            int end = header.IndexOf('\'', start);
            if (end > start)
                return header[start..end];
        }

        const string doubleQuoteToken = "\"descr\": \"";
        start = header.IndexOf(doubleQuoteToken, StringComparison.Ordinal);
        if (start >= 0)
        {
            start += doubleQuoteToken.Length;
            int end = header.IndexOf('"', start);
            if (end > start)
                return header[start..end];
        }

        throw new InvalidDataException("NumPy dtype descriptor is missing from the header.");
    }
}