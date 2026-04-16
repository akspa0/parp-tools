using System.Buffers.Binary;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2SkinReader
{
    private const int SignatureSizeBytes = 4;
    private const int MinimumHeaderSizeBytes = 44;
    private const int ExtendedHeaderSizeBytes = 60;
    private const int BoneEntryStride = 0x04;
    private const int SubmeshStride = 0x30;
    private const int BatchStride = 0x18;

    public static M2SkinDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static M2SkinDocument Read(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("M2 skin reading requires a seekable stream.", nameof(stream));

        if (!Path.GetExtension(sourcePath).Equals(".skin", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"M2 skin reading requires a .skin path. Found '{Path.GetExtension(sourcePath)}'.", nameof(sourcePath));

        byte[] data = ReadAllBytes(stream);
        if (data.Length < MinimumHeaderSizeBytes)
            throw new InvalidDataException($"Skin file '{sourcePath}' is too small to contain a strict SKIN header.");

        string signature = System.Text.Encoding.ASCII.GetString(data, 0, SignatureSizeBytes);
        if (!string.Equals(signature, "SKIN", StringComparison.Ordinal))
            throw new InvalidDataException($"Skin file '{sourcePath}' does not contain a strict SKIN root. Found '{signature}'.");

        int vertexLookupCount = checked((int)ReadUInt32At(data, 0x04));
        uint vertexLookupOffset = ReadUInt32At(data, 0x08);
        int triangleIndexCount = checked((int)ReadUInt32At(data, 0x0C));
        uint triangleIndexOffset = ReadUInt32At(data, 0x10);
        int boneEntryCount = checked((int)ReadUInt32At(data, 0x14));
        uint boneEntryOffset = ReadUInt32At(data, 0x18);
        int submeshCount = checked((int)ReadUInt32At(data, 0x1C));
        uint submeshOffset = ReadUInt32At(data, 0x20);
        int batchCount = checked((int)ReadUInt32At(data, 0x24));
        uint batchOffset = ReadUInt32At(data, 0x28);
        uint globalVertexOffset = ReadUInt32At(data, 0x2C);

        uint shadowBatchCount = 0;
        uint shadowBatchOffset = 0;
        if (data.Length >= ExtendedHeaderSizeBytes)
        {
            uint candidateShadowBatchCount = ReadUInt32At(data, 0x30);
            uint candidateShadowBatchOffset = ReadUInt32At(data, 0x34);
            if (HasValidOptionalSpan(data.Length, candidateShadowBatchCount, candidateShadowBatchOffset, BatchStride))
            {
                shadowBatchCount = candidateShadowBatchCount;
                shadowBatchOffset = candidateShadowBatchOffset;
            }
        }

        List<ushort> vertexLookup = ReadUInt16Table(data, sourcePath, "vertexLookup", vertexLookupCount, vertexLookupOffset);
        List<ushort> triangleIndices = ReadUInt16Table(data, sourcePath, "triangleIndices", triangleIndexCount, triangleIndexOffset);
        List<M2SkinBoneEntry> boneEntries = ReadBoneEntries(data, sourcePath, boneEntryCount, boneEntryOffset);
        List<M2SkinSubmesh> submeshes = ReadSubmeshes(data, sourcePath, submeshCount, submeshOffset);
        List<M2SkinBatch> batches = ReadBatches(data, sourcePath, batchCount, batchOffset);

        return new M2SkinDocument(
            M2ModelIdentity.NormalizePath(sourcePath),
            signature,
            vertexLookup,
            vertexLookupOffset,
            triangleIndices,
            triangleIndexOffset,
            boneEntries,
            boneEntryOffset,
            submeshes,
            submeshOffset,
            batches,
            batchOffset,
            globalVertexOffset,
            shadowBatchCount,
            shadowBatchOffset);
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] data = new byte[checked((int)stream.Length)];
            stream.ReadExactly(data);
            return data;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint ReadUInt32At(byte[] data, int offset)
    {
        return BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
    }

    private static ushort ReadUInt16At(byte[] data, int offset)
    {
        return BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(offset, sizeof(ushort)));
    }

    private static short ReadInt16At(byte[] data, int offset)
    {
        return BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(offset, sizeof(short)));
    }

    private static List<ushort> ReadUInt16Table(byte[] data, string sourcePath, string label, int count, uint offset)
    {
        ValidateSpan(count, offset, sizeof(ushort), data.Length, sourcePath, label);
        List<ushort> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(ReadUInt16At(data, checked((int)offset + (index * sizeof(ushort)))));

        return values;
    }

    private static List<M2SkinBoneEntry> ReadBoneEntries(byte[] data, string sourcePath, int count, uint offset)
    {
        ValidateSpan(count, offset, BoneEntryStride, data.Length, sourcePath, "boneEntries");
        List<M2SkinBoneEntry> values = new(count);
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * BoneEntryStride));
            values.Add(new M2SkinBoneEntry(
                data[entryOffset + 0x00],
                data[entryOffset + 0x01],
                data[entryOffset + 0x02],
                data[entryOffset + 0x03]));
        }

        return values;
    }

    private static List<M2SkinSubmesh> ReadSubmeshes(byte[] data, string sourcePath, int count, uint offset)
    {
        ValidateSpan(count, offset, SubmeshStride, data.Length, sourcePath, "submeshes");
        List<M2SkinSubmesh> values = new(count);
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * SubmeshStride));
            values.Add(new M2SkinSubmesh(
                ReadUInt16At(data, entryOffset + 0x00),
                ReadUInt16At(data, entryOffset + 0x02),
                ReadUInt16At(data, entryOffset + 0x04),
                ReadUInt16At(data, entryOffset + 0x06),
                ReadUInt16At(data, entryOffset + 0x08),
                ReadUInt16At(data, entryOffset + 0x0A),
                ReadUInt16At(data, entryOffset + 0x0C),
                ReadUInt16At(data, entryOffset + 0x0E),
                ReadUInt16At(data, entryOffset + 0x10),
                ReadUInt16At(data, entryOffset + 0x12)));
        }

        return values;
    }

    private static List<M2SkinBatch> ReadBatches(byte[] data, string sourcePath, int count, uint offset)
    {
        ValidateSpan(count, offset, BatchStride, data.Length, sourcePath, "batches");
        List<M2SkinBatch> values = new(count);
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * BatchStride));
            values.Add(new M2SkinBatch(
                data[entryOffset + 0x00],
                data[entryOffset + 0x01],
                ReadUInt16At(data, entryOffset + 0x02),
                ReadUInt16At(data, entryOffset + 0x04),
                ReadUInt16At(data, entryOffset + 0x06),
                ReadInt16At(data, entryOffset + 0x08),
                ReadUInt16At(data, entryOffset + 0x0A),
                ReadUInt16At(data, entryOffset + 0x0C),
                ReadUInt16At(data, entryOffset + 0x0E),
                ReadUInt16At(data, entryOffset + 0x10),
                ReadUInt16At(data, entryOffset + 0x12),
                ReadUInt16At(data, entryOffset + 0x14),
                ReadUInt16At(data, entryOffset + 0x16)));
        }

        return values;
    }

    private static void ValidateSpan(int count, uint offset, int stride, int length, string sourcePath, string label)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        if (count == 0)
            return;

        if (offset == 0)
            throw new InvalidDataException($"Skin file '{sourcePath}' has a zero offset for non-empty span '{label}'.");

        ulong total = checked((ulong)count * (ulong)stride);
        ulong end = (ulong)offset + total;
        if ((ulong)offset >= (ulong)length || end > (ulong)length || end < offset)
        {
            throw new InvalidDataException(
                $"Skin file '{sourcePath}' has an out-of-range span for '{label}': count={count}, offset=0x{offset:X}, stride=0x{stride:X}, length=0x{length:X}.");
        }
    }

    private static bool HasValidOptionalSpan(int length, uint count, uint offset, int stride)
    {
        if (count == 0)
            return true;

        if (offset == 0)
            return false;

        ulong total = (ulong)count * (ulong)stride;
        ulong end = (ulong)offset + total;
        return (ulong)offset < (ulong)length && end <= (ulong)length && end >= offset;
    }
}
