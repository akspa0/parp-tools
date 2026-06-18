using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Services;

public static class Pd4ResearchReader
{
    public static Pd4ResearchDocument ReadFile(string path)
    {
        return Read(File.ReadAllBytes(path), path);
    }

    public static Pd4ResearchDocument Read(byte[] bytes, string? sourcePath = null)
    {
        var chunks = new List<Pm4ChunkRecord>();
        var diagnostics = new List<string>();

        var mslk = new List<Pd4MslkEntry>();
        var mspv = new List<Vector3>();
        var mspi = new List<uint>();
        var msvt = new List<Vector3>();
        var msvi = new List<uint>();
        var msur = new List<Pd4MsurEntry>();
        var mscn = new List<Vector3>();

        uint version = 0;
        uint mcrc = 0;
        int offset = 0;

        while (offset + 8 <= bytes.Length)
        {
            int headerOffset = offset;
            string signature = ReadSignature(bytes.AsSpan(offset, 4));
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            int dataOffset = offset + 8;
            long endOffsetLong = (long)dataOffset + size;
            if (endOffsetLong > bytes.Length)
            {
                diagnostics.Add($"Chunk '{signature}' at 0x{headerOffset:X} overruns file: size={size}, end=0x{endOffsetLong:X}, file=0x{bytes.Length:X}.");
                break;
            }

            int endOffset = (int)endOffsetLong;
            byte[] payload = bytes.AsSpan(dataOffset, (int)size).ToArray();
            chunks.Add(new Pm4ChunkRecord(signature, headerOffset, dataOffset, size, payload));

            try
            {
                switch (signature)
                {
                    case "MVER":
                        if (payload.Length >= 4)
                            version = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0, 4));
                        else
                            diagnostics.Add("MVER payload is smaller than 4 bytes.");
                        break;

                    case "MCRC":
                        if (payload.Length >= 4)
                            mcrc = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0, 4));
                        break;

                    case "MSPV":
                        ParseVectors(payload, mspv, diagnostics, signature);
                        break;

                    case "MSPI":
                        ParseUInt32List(payload, mspi, diagnostics, signature);
                        break;

                    case "MSVT":
                        ParsePd4Msvt(payload, msvt, diagnostics);
                        break;

                    case "MSVI":
                        ParseUInt32List(payload, msvi, diagnostics, signature);
                        break;

                    case "MSUR":
                        ParsePd4Msur(payload, msur, diagnostics);
                        break;

                    case "MSCN":
                        ParseVectors(payload, mscn, diagnostics, signature);
                        break;

                    case "MSLK":
                        ParsePd4Mslk(payload, mslk, diagnostics);
                        break;
                }
            }
            catch (Exception exception)
            {
                diagnostics.Add($"Failed to parse chunk '{signature}' at 0x{headerOffset:X}: {exception.Message}");
            }

            offset = endOffset;
        }

        if (offset < bytes.Length)
            diagnostics.Add($"Trailing {bytes.Length - offset} bytes remain after chunk walk at 0x{offset:X}.");

        return new Pd4ResearchDocument(
            sourcePath, version, mcrc, chunks,
            new Pd4KnownChunkSet(mslk, mspv, mspi, msvt, msvi, msur, mscn),
            diagnostics);
    }

    private static string ReadSignature(ReadOnlySpan<byte> bytes)
    {
        Span<byte> signature = stackalloc byte[4];
        bytes.CopyTo(signature);
        signature.Reverse();
        return Encoding.ASCII.GetString(signature);
    }

    private static void ParseVectors(ReadOnlySpan<byte> payload, List<Vector3> target, List<string> diagnostics, string signature)
    {
        const int stride = 12;
        ValidateStride(payload, stride, signature, diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
            target.Add(ReadVector3(payload[offset..(offset + stride)]));
    }

    private static void ParsePd4Msvt(ReadOnlySpan<byte> payload, List<Vector3> target, List<string> diagnostics)
    {
        const int stride = 12;
        ValidateStride(payload, stride, "MSVT", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            float y = ReadSingle(payload[offset..(offset + 4)]);
            float x = ReadSingle(payload[(offset + 4)..(offset + 8)]);
            float z = ReadSingle(payload[(offset + 8)..(offset + 12)]);
            target.Add(new Vector3(x, y, z));
        }
    }

    private static void ParsePd4Msur(ReadOnlySpan<byte> payload, List<Pd4MsurEntry> target, List<string> diagnostics)
    {
        const int stride = 32;
        ValidateStride(payload, stride, "MSUR", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pd4MsurEntry(
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
                ReadVector3(payload[(offset + 4)..(offset + 16)]),
                ReadSingle(payload[(offset + 16)..(offset + 20)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 20)..(offset + 24)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 24)..(offset + 28)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 28)..(offset + 32)])));
        }
    }

    private static void ParsePd4Mslk(ReadOnlySpan<byte> payload, List<Pd4MslkEntry> target, List<string> diagnostics)
    {
        const int stride = 24;
        ValidateStride(payload, stride, "MSLK", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pd4MslkEntry(
                payload[offset],
                payload[offset + 1],
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 2)..(offset + 4)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 4)..(offset + 8)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 8)..(offset + 12)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 12)..(offset + 16)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 16)..(offset + 20)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 20)..(offset + 24)])));
        }
    }

    private static void ParseUInt32List(ReadOnlySpan<byte> payload, List<uint> target, List<string> diagnostics, string signature)
    {
        const int stride = 4;
        ValidateStride(payload, stride, signature, diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
            target.Add(BinaryPrimitives.ReadUInt32LittleEndian(payload[offset..(offset + stride)]));
    }

    private static void ValidateStride(ReadOnlySpan<byte> payload, int stride, string signature, List<string> diagnostics)
    {
        int remainder = payload.Length % stride;
        if (remainder != 0)
            diagnostics.Add($"{signature} payload length {payload.Length} leaves remainder {remainder} with stride {stride}.");
    }

    private static Vector3 ReadVector3(ReadOnlySpan<byte> payload)
    {
        return new Vector3(
            ReadSingle(payload[0..4]),
            ReadSingle(payload[4..8]),
            ReadSingle(payload[8..12]));
    }

    private static float ReadSingle(ReadOnlySpan<byte> payload)
    {
        int raw = BinaryPrimitives.ReadInt32LittleEndian(payload);
        return BitConverter.Int32BitsToSingle(raw);
    }
}
