using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;

namespace WowViewer.Tools.WdlRead;

/// <summary>
/// Minimal NPY/NPZ reader and writer for the WdlRead shim. Shim-local by design:
/// the core libraries stay untouched, and the shim only needs float32/int32
/// C-order arrays of rank 1-3 (NumPy format v1.0/v2.0, little-endian).
/// </summary>
internal static class Npy
{
    private static ReadOnlySpan<byte> Magic => [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    internal sealed record NpyArray(float[] Data, int[] Shape)
    {
        public int Rank => Shape.Length;
        public long ElementCount => Shape.Aggregate(1L, static (acc, d) => acc * d);
    }

    public static Dictionary<string, NpyArray> ReadNpz(string path)
    {
        using ZipArchive zip = ZipFile.OpenRead(path);
        Dictionary<string, NpyArray> result = new(StringComparer.OrdinalIgnoreCase);
        foreach (ZipArchiveEntry entry in zip.Entries)
        {
            if (!entry.Name.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                continue;

            using Stream stream = entry.Open();
            using MemoryStream memory = new();
            stream.CopyTo(memory);
            string key = entry.Name[..^4];
            result[key] = ReadNpy(memory.ToArray(), $"{path}::{entry.Name}");
        }

        return result;
    }

    public static NpyArray ReadNpy(byte[] bytes, string source)
    {
        if (bytes.Length < 10 || !bytes.AsSpan(0, 6).SequenceEqual(Magic))
            throw new InvalidDataException($"'{source}' is not an NPY payload.");

        byte major = bytes[6];
        int headerLength;
        int headerStart;
        if (major == 1)
        {
            headerLength = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8, 2));
            headerStart = 10;
        }
        else if (major == 2)
        {
            headerLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8, 4)));
            headerStart = 12;
        }
        else
        {
            throw new NotSupportedException($"'{source}': NPY format version {major}.x is not supported.");
        }

        string header = Encoding.ASCII.GetString(bytes, headerStart, headerLength);
        string descr = ExtractHeaderValue(header, "descr", source);
        bool fortranOrder = ExtractHeaderValue(header, "fortran_order", source).StartsWith("True", StringComparison.Ordinal);
        if (fortranOrder)
            throw new NotSupportedException($"'{source}': fortran_order arrays are not supported.");

        int[] shape = ParseShape(header, source);
        long count = shape.Aggregate(1L, static (acc, d) => acc * d);
        int dataStart = headerStart + headerLength;

        float[] data = new float[count];
        ReadOnlySpan<byte> payload = bytes.AsSpan(dataStart);
        switch (descr)
        {
            case "<f4":
                RequireBytes(payload, count * 4, source);
                for (long i = 0; i < count; i++)
                    data[i] = BinaryPrimitives.ReadSingleLittleEndian(payload.Slice(checked((int)(i * 4)), 4));
                break;
            case "<f8":
                RequireBytes(payload, count * 8, source);
                for (long i = 0; i < count; i++)
                    data[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(payload.Slice(checked((int)(i * 8)), 8));
                break;
            case "<i4":
                RequireBytes(payload, count * 4, source);
                for (long i = 0; i < count; i++)
                    data[i] = BinaryPrimitives.ReadInt32LittleEndian(payload.Slice(checked((int)(i * 4)), 4));
                break;
            case "<i8":
                RequireBytes(payload, count * 8, source);
                for (long i = 0; i < count; i++)
                    data[i] = BinaryPrimitives.ReadInt64LittleEndian(payload.Slice(checked((int)(i * 8)), 8));
                break;
            case "|b1":
            case "|u1":
                RequireBytes(payload, count, source);
                for (long i = 0; i < count; i++)
                    data[i] = payload[checked((int)i)];
                break;
            case "<i2":
                RequireBytes(payload, count * 2, source);
                for (long i = 0; i < count; i++)
                    data[i] = BinaryPrimitives.ReadInt16LittleEndian(payload.Slice(checked((int)(i * 2)), 2));
                break;
            default:
                throw new NotSupportedException($"'{source}': NPY dtype '{descr}' is not supported by the shim.");
        }

        return new NpyArray(data, shape);
    }

    public static void WriteNpz(string path, IReadOnlyList<(string Name, Array Data, int[] Shape)> arrays)
    {
        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using FileStream fs = File.Create(path);
        using ZipArchive zip = new(fs, ZipArchiveMode.Create);
        foreach ((string name, Array data, int[] shape) in arrays)
        {
            ZipArchiveEntry entry = zip.CreateEntry($"{name}.npy", CompressionLevel.Fastest);
            using Stream stream = entry.Open();
            WriteNpy(stream, data, shape);
        }
    }

    private static void WriteNpy(Stream stream, Array data, int[] shape)
    {
        (string descr, int elementSize) = data switch
        {
            float[] => ("<f4", 4),
            int[] => ("<i4", 4),
            _ => throw new NotSupportedException($"NPY write does not support {data.GetType().Name}."),
        };

        long count = shape.Aggregate(1L, static (acc, d) => acc * d);
        if (count != data.Length)
            throw new ArgumentException($"Shape ({string.Join(",", shape)}) does not match data length {data.Length}.");

        string shapeText = shape.Length == 1
            ? $"({shape[0]},)"
            : $"({string.Join(", ", shape)})";
        string header = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shapeText}, }}";
        int prefix = Magic.Length + 2 + 2;
        int unpadded = prefix + header.Length + 1;
        int padding = (64 - (unpadded % 64)) % 64;
        header += new string(' ', padding) + "\n";

        byte[] preamble = new byte[prefix];
        Magic.CopyTo(preamble);
        preamble[6] = 1;
        preamble[7] = 0;
        BinaryPrimitives.WriteUInt16LittleEndian(preamble.AsSpan(8, 2), (ushort)header.Length);
        stream.Write(preamble);
        stream.Write(Encoding.ASCII.GetBytes(header));

        byte[] buffer = new byte[data.Length * elementSize];
        switch (data)
        {
            case float[] floats:
                for (int i = 0; i < floats.Length; i++)
                    BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(i * 4, 4), floats[i]);
                break;
            case int[] ints:
                for (int i = 0; i < ints.Length; i++)
                    BinaryPrimitives.WriteInt32LittleEndian(buffer.AsSpan(i * 4, 4), ints[i]);
                break;
        }

        stream.Write(buffer);
    }

    private static void RequireBytes(ReadOnlySpan<byte> payload, long required, string source)
    {
        if (payload.Length < required)
            throw new InvalidDataException($"'{source}': NPY payload truncated (needed {required} bytes, found {payload.Length}).");
    }

    private static string ExtractHeaderValue(string header, string key, string source)
    {
        string marker = $"'{key}':";
        int index = header.IndexOf(marker, StringComparison.Ordinal);
        if (index < 0)
            throw new InvalidDataException($"'{source}': NPY header is missing '{key}'.");

        int start = index + marker.Length;
        while (start < header.Length && header[start] == ' ')
            start++;

        if (header[start] == '\'')
        {
            int end = header.IndexOf('\'', start + 1);
            return header.Substring(start + 1, end - start - 1);
        }

        int stop = start;
        while (stop < header.Length && header[stop] != ',' && header[stop] != '}')
            stop++;
        return header[start..stop].Trim();
    }

    private static int[] ParseShape(string header, string source)
    {
        int open = header.IndexOf("'shape':", StringComparison.Ordinal);
        if (open < 0)
            throw new InvalidDataException($"'{source}': NPY header is missing 'shape'.");

        int parenStart = header.IndexOf('(', open);
        int parenEnd = header.IndexOf(')', parenStart);
        string inner = header.Substring(parenStart + 1, parenEnd - parenStart - 1);
        string[] parts = inner.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
            return [1];

        return parts.Select(int.Parse).ToArray();
    }
}
