using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Services;

public static class Pm4ResearchReader
{
    public static Pm4ResearchDocument ReadFile(string path)
    {
        return Read(File.ReadAllBytes(path), path);
    }

    public static Pm4ResearchDocument Read(byte[] bytes, string? sourcePath = null)
    {
        var chunks = new List<Pm4ChunkRecord>();
        var diagnostics = new List<string>();

        var mslk = new List<Pm4MslkEntry>();
        var mspv = new List<Vector3>();
        var mspi = new List<uint>();
        var msvt = new List<Vector3>();
        var msvi = new List<uint>();
        var msur = new List<Pm4MsurEntry>();
        var mscn = new List<Vector3>();
        var mprl = new List<Pm4MprlEntry>();
        var mprr = new List<Pm4MprrEntry>();
        var mdbi = new List<Pm4MdbiEntry>();
        var mdbf = new List<Pm4MdbfEntry>();
        var mdos = new List<Pm4MdosEntry>();
        var mdsf = new List<Pm4MdsfEntry>();

        Pm4MshdHeader? mshd = null;
        Pm4MdbhEntry? mdbh = null;
        uint version = 0;
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

                    case "MSHD":
                        mshd = ParseMshd(payload, diagnostics);
                        break;

                    case "MSLK":
                        ParseMslk(payload, mslk, diagnostics);
                        break;

                    case "MSPV":
                        ParseVectors(payload, mspv, diagnostics, signature);
                        break;

                    case "MSPI":
                        ParseUInt32List(payload, mspi, diagnostics, signature);
                        break;

                    case "MSVT":
                        ParseVectors(payload, msvt, diagnostics, signature);
                        break;

                    case "MSVI":
                        ParseUInt32List(payload, msvi, diagnostics, signature);
                        break;

                    case "MSUR":
                        ParseMsur(payload, msur, diagnostics);
                        break;

                    case "MSCN":
                        ParseVectors(payload, mscn, diagnostics, signature);
                        break;

                    case "MPRL":
                        ParseMprl(payload, mprl, diagnostics);
                        break;

                    case "MPRR":
                        ParseMprr(payload, mprr, diagnostics);
                        break;

                    case "MDBH":
                        mdbh = ParseMdbh(payload, diagnostics);
                        break;

                    case "MDBI":
                        ParseMdbi(payload, mdbi, diagnostics);
                        break;

                    case "MDBF":
                        mdbf.Add(ParseMdbf(payload));
                        break;

                    case "MDOS":
                        ParseMdos(payload, mdos, diagnostics);
                        break;

                    case "MDSF":
                        ParseMdsf(payload, mdsf, diagnostics);
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

        return new Pm4ResearchDocument(
            sourcePath,
            version,
            chunks,
            new Pm4KnownChunkSet(mshd, mslk, mspv, mspi, msvt, msvi, msur, mscn, mprl, mprr, mdbh, mdbi, mdbf, mdos, mdsf),
            diagnostics);
    }

    private static string ReadSignature(ReadOnlySpan<byte> bytes)
    {
        Span<byte> signature = stackalloc byte[4];
        bytes.CopyTo(signature);
        signature.Reverse();
        return Encoding.ASCII.GetString(signature);
    }

    private static Pm4MshdHeader ParseMshd(ReadOnlySpan<byte> payload, List<string> diagnostics)
    {
        if (payload.Length < 32)
        {
            diagnostics.Add($"MSHD payload too small: expected 32 bytes, got {payload.Length}.");
            return new Pm4MshdHeader(0, 0, 0, 0, 0, 0, 0, 0);
        }

        return new Pm4MshdHeader(
            BinaryPrimitives.ReadUInt32LittleEndian(payload[0..4]),
            BinaryPrimitives.ReadUInt32LittleEndian(payload[4..8]),
            BinaryPrimitives.ReadUInt32LittleEndian(payload[8..12]),
            BinaryPrimitives.ReadUInt32LittleEndian(payload[12..16]),
            BinaryPrimitives.ReadUInt32LittleEndian(payload[16..20]),
            BinaryPrimitives.ReadUInt32LittleEndian(payload[20..24]),
            BinaryPrimitives.ReadUInt32LittleEndian(payload[24..28]),
            BinaryPrimitives.ReadUInt32LittleEndian(payload[28..32]));
    }

    private static void ParseMslk(ReadOnlySpan<byte> payload, List<Pm4MslkEntry> target, List<string> diagnostics)
    {
        const int stride = 20;
        ValidateStride(payload, stride, "MSLK", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pm4MslkEntry(
                payload[offset],
                payload[offset + 1],
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 2)..(offset + 4)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 4)..(offset + 8)]),
                ReadSignedInt24(payload[(offset + 8)..(offset + 11)]),
                payload[offset + 11],
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 12)..(offset + 16)]),
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 16)..(offset + 18)]),
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 18)..(offset + 20)])));
        }
    }

    private static void ParseMsur(ReadOnlySpan<byte> payload, List<Pm4MsurEntry> target, List<string> diagnostics)
    {
        const int stride = 32;
        ValidateStride(payload, stride, "MSUR", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pm4MsurEntry(
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

    private static void ParseMprl(ReadOnlySpan<byte> payload, List<Pm4MprlEntry> target, List<string> diagnostics)
    {
        const int stride = 24;
        ValidateStride(payload, stride, "MPRL", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pm4MprlEntry(
                BinaryPrimitives.ReadUInt16LittleEndian(payload[offset..(offset + 2)]),
                BinaryPrimitives.ReadInt16LittleEndian(payload[(offset + 2)..(offset + 4)]),
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 4)..(offset + 6)]),
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 6)..(offset + 8)]),
                ReadVector3(payload[(offset + 8)..(offset + 20)]),
                BinaryPrimitives.ReadInt16LittleEndian(payload[(offset + 20)..(offset + 22)]),
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 22)..(offset + 24)])));
        }
    }

    private static void ParseMprr(ReadOnlySpan<byte> payload, List<Pm4MprrEntry> target, List<string> diagnostics)
    {
        const int stride = 4;
        ValidateStride(payload, stride, "MPRR", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pm4MprrEntry(
                BinaryPrimitives.ReadUInt16LittleEndian(payload[offset..(offset + 2)]),
                BinaryPrimitives.ReadUInt16LittleEndian(payload[(offset + 2)..(offset + 4)])));
        }
    }

    private static Pm4MdbhEntry ParseMdbh(ReadOnlySpan<byte> payload, List<string> diagnostics)
    {
        if (payload.Length < 4)
        {
            diagnostics.Add($"MDBH payload too small: expected 4 bytes, got {payload.Length}.");
            return new Pm4MdbhEntry(0);
        }

        if (payload.Length != 4)
            diagnostics.Add($"MDBH payload length {payload.Length} is larger than the documented 4-byte count field.");

        return new Pm4MdbhEntry(BinaryPrimitives.ReadUInt32LittleEndian(payload[0..4]));
    }

    private static void ParseMdbi(ReadOnlySpan<byte> payload, List<Pm4MdbiEntry> target, List<string> diagnostics)
    {
        const int stride = 4;
        ValidateStride(payload, stride, "MDBI", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
            target.Add(new Pm4MdbiEntry(BinaryPrimitives.ReadUInt32LittleEndian(payload[offset..(offset + stride)])));
    }

    private static Pm4MdbfEntry ParseMdbf(ReadOnlySpan<byte> payload)
    {
        int nullIndex = payload.IndexOf((byte)0);
        ReadOnlySpan<byte> content = nullIndex >= 0 ? payload[..nullIndex] : payload;
        string filename = Encoding.ASCII.GetString(content);
        return new Pm4MdbfEntry(filename, payload.Length);
    }

    private static void ParseMdos(ReadOnlySpan<byte> payload, List<Pm4MdosEntry> target, List<string> diagnostics)
    {
        const int stride = 8;
        ValidateStride(payload, stride, "MDOS", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pm4MdosEntry(
                BinaryPrimitives.ReadUInt32LittleEndian(payload[offset..(offset + 4)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 4)..(offset + 8)])));
        }
    }

    private static void ParseMdsf(ReadOnlySpan<byte> payload, List<Pm4MdsfEntry> target, List<string> diagnostics)
    {
        const int stride = 8;
        ValidateStride(payload, stride, "MDSF", diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
        {
            target.Add(new Pm4MdsfEntry(
                BinaryPrimitives.ReadUInt32LittleEndian(payload[offset..(offset + 4)]),
                BinaryPrimitives.ReadUInt32LittleEndian(payload[(offset + 4)..(offset + 8)])));
        }
    }

    private static void ParseVectors(ReadOnlySpan<byte> payload, List<Vector3> target, List<string> diagnostics, string signature)
    {
        const int stride = 12;
        ValidateStride(payload, stride, signature, diagnostics);
        for (int offset = 0; offset + stride <= payload.Length; offset += stride)
            target.Add(ReadVector3(payload[offset..(offset + stride)]));
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

    private static int ReadSignedInt24(ReadOnlySpan<byte> payload)
    {
        int value = payload[0] | (payload[1] << 8) | (payload[2] << 16);
        if ((value & 0x0080_0000) != 0)
            value |= unchecked((int)0xFF00_0000);

        return value;
    }
}