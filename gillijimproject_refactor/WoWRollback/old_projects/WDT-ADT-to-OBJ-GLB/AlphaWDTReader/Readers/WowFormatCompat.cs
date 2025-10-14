using System.Text;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace AlphaWDTReader.Readers;

// Compatibility adapter façade for subchunk decoding.
// For now, uses in-house implementations. Later, we can redirect to WoWFormatLib readers
// once the library is present, keeping the API stable.
public static class WowFormatCompat
{
    private const int VertexCount = 145;

    // Read 145 float32 heights from an already-positioned BinaryReader.
    // Caller must ensure stream position is at the start of MCVT payload.
    public static float[]? ReadMCVT145(BinaryReader br)
    {
        const int bytes = VertexCount * 4; // 580
        var data = br.ReadBytes(bytes);
        if (data.Length != bytes) return null;
        var heights = new float[VertexCount];
        for (int i = 0; i < VertexCount; i++) heights[i] = BitConverter.ToSingle(data, i * 4);
        return heights;
    }

    // Read 145x3 signed-byte normals from an already-positioned BinaryReader.
    // Returns 145*3 floats in [-1,1] (nx,ny,nz) triplets.
    public static float[]? ReadMCNR145(BinaryReader br)
    {
        const int components = 3;
        int bytes = VertexCount * components; // 435
        var data = br.ReadBytes(bytes);
        if (data.Length != bytes) return null;
        var normals = new float[bytes];
        for (int i = 0; i < VertexCount; i++)
        {
            int b = i * components;
            normals[b + 0] = (sbyte)data[b + 0] / 127f;
            normals[b + 1] = (sbyte)data[b + 1] / 127f;
            normals[b + 2] = (sbyte)data[b + 2] / 127f;
        }
        return normals;
    }

    // Conservative MCCV summary. Detects stride by size%145 (3 or 4) and returns
    // minimal stats and a small preview. Does not assume layout beyond byte stride.
    public static MccvSummary TrySummarizeMCCV(BinaryReader br, int payloadSize, int previewCount = 8)
    {
        var summary = new MccvSummary();
        if (payloadSize <= 0) { summary.Status = "empty"; return summary; }
        int stride = payloadSize % VertexCount == 0 ? payloadSize / VertexCount : 0;
        if (stride != 3 && stride != 4)
        {
            summary.Status = "unknown_stride";
            // capture small hex for forensics
            int toRead = Math.Min(payloadSize, 32);
            var bytes = br.ReadBytes(toRead);
            summary.SampleHex = BitConverter.ToString(bytes).Replace("-", "");
            summary.TotalBytes = payloadSize;
            return summary;
        }

        summary.Status = "ok";
        summary.Stride = stride;
        summary.Tuples = VertexCount;
        summary.TotalBytes = payloadSize;

        int toReadAll = VertexCount * stride;
        var data = br.ReadBytes(toReadAll);
        if (data.Length != toReadAll)
        {
            summary.Status = "truncated";
            summary.SampleHex = BitConverter.ToString(data.Take(Math.Min(32, data.Length)).ToArray()).Replace("-", "");
            return summary;
        }

        byte minR = 255, minG = 255, minB = 255, minA = 255;
        byte maxR = 0, maxG = 0, maxB = 0, maxA = 0;
        int preview = Math.Min(previewCount, VertexCount);
        summary.First = new List<byte[]>(preview);
        for (int i = 0; i < VertexCount; i++)
        {
            int b = i * stride;
            byte r = data[b + 0];
            byte g = data[b + 1];
            byte ch2 = data[b + 2]; // B in 3/4 stride
            byte a = stride == 4 ? data[b + 3] : (byte)0;
            // Track as RGB(A)
            if (r < minR) minR = r; if (r > maxR) maxR = r;
            if (g < minG) minG = g; if (g > maxG) maxG = g;
            if (ch2 < minB) minB = ch2; if (ch2 > maxB) maxB = ch2;
            if (stride == 4) { if (a < minA) minA = a; if (a > maxA) maxA = a; }
            if (i < preview)
            {
                if (stride == 3)
                    summary.First.Add(new byte[] { r, g, ch2 });
                else
                    summary.First.Add(new byte[] { r, g, ch2, a });
            }
        }
        summary.Min = stride == 3 ? new byte[] { minR, minG, minB } : new byte[] { minR, minG, minB, minA };
        summary.Max = stride == 3 ? new byte[] { maxR, maxG, maxB } : new byte[] { maxR, maxG, maxB, maxA };
        return summary;
    }

    // Minimal MCLY summary: report total bytes and inferred entry count by common stride guesses.
    public static MclySummary TrySummarizeMCLY(BinaryReader br, int payloadSize)
    {
        var s = new MclySummary { TotalBytes = payloadSize };
        if (payloadSize <= 0)
        {
            s.Status = "empty";
            return s;
        }
        // Common guesses: 16 or 12 bytes per entry depending on variant; report both
        s.Status = "ok";
        s.EntryCount16 = payloadSize / 16;
        s.EntryCount12 = payloadSize / 12;
        // Do not read the stream further (caller may continue elsewhere)
        return s;
    }

    // Generic presence summary (e.g., for MCSH), returns only size and a tiny hex sample.
    public static PresenceSummary SummarizePresence(BinaryReader br, int payloadSize, int hexBytes = 16)
    {
        var s = new PresenceSummary { TotalBytes = payloadSize };
        if (payloadSize <= 0) { s.Status = "empty"; return s; }
        s.Status = "ok";
        int toRead = Math.Min(payloadSize, hexBytes);
        var bytes = br.ReadBytes(toRead);
        s.SampleHex = BitConverter.ToString(bytes).Replace("-", "");
        return s;
    }

    // Top-level: MVER is a single uint version.
    public static UIntSummary TrySummarizeMVER(BinaryReader br, int payloadSize)
    {
        var s = new UIntSummary();
        if (payloadSize < 4) { s.Status = "truncated"; return s; }
        try
        {
            s.Value = br.ReadUInt32();
            s.Status = "ok";
        }
        catch { s.Status = "error"; }
        return s;
    }

    // Top-level: MCIN table entry count is payloadSize/16 for classic layouts.
    public static CountSummary TrySummarizeMCIN(int payloadSize)
    {
        var s = new CountSummary { TotalBytes = payloadSize };
        if (payloadSize <= 0) { s.Status = "empty"; return s; }
        if (payloadSize % 16 != 0) { s.Status = "unknown_stride"; return s; }
        s.Status = "ok";
        s.Count = payloadSize / 16;
        return s;
    }

    // Selective WoWFormatLib compatibility adapter
    private const int McvtBytes = VertexCount * 4;      // 580
    private const int McnrBytes = VertexCount * 3;      // 435

    // Reads 145 float32 heights directly from a file at the given absolute offset.
    // Validates size and file bounds. Returns null on any mismatch.
    public static float[]? TryReadMcvt145(string filePath, long offset, int size)
    {
        if (string.IsNullOrWhiteSpace(filePath)) return null;
        if (offset < 0 || size < McvtBytes) return null;
        try
        {
            using var fs = File.OpenRead(filePath);
            if (offset + McvtBytes > fs.Length) return null;
            using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
            fs.Position = offset;
            var heights = new float[VertexCount];
            for (int i = 0; i < VertexCount; i++) heights[i] = br.ReadSingle();
            return heights;
        }
        catch { return null; }
    }

    // Reads 145*3 signed-byte normals from a file at the given absolute offset
    // and converts to [-1,1] floats in (nx,ny,nz) order. Returns null on mismatch.
    public static float[]? TryReadMcnr145(string filePath, long offset, int size)
    {
        if (string.IsNullOrWhiteSpace(filePath)) return null;
        if (offset < 0 || size < McnrBytes) return null;
        try
        {
            using var fs = File.OpenRead(filePath);
            if (offset + McnrBytes > fs.Length) return null;
            using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
            fs.Position = offset;
            var normals = new float[McnrBytes];
            for (int i = 0; i < McnrBytes; i++)
            {
                sbyte v = (sbyte)br.ReadByte();
                normals[i] = Math.Clamp(v / 127.0f, -1.0f, 1.0f);
            }
            return normals;
        }
        catch { return null; }
    }
}

public class MccvSummary
{
    public string Status { get; set; } = ""; // ok, unknown_stride, truncated, empty
    public int Stride { get; set; }
    public int Tuples { get; set; }
    public int TotalBytes { get; set; }
    public List<byte[]>? First { get; set; }
    public byte[]? Min { get; set; }
    public byte[]? Max { get; set; }
    public string? SampleHex { get; set; }
}

public class MclySummary
{
    public string Status { get; set; } = ""; // ok, empty
    public int TotalBytes { get; set; }
    public int EntryCount16 { get; set; }
    public int EntryCount12 { get; set; }
}

public class PresenceSummary
{
    public string Status { get; set; } = ""; // ok, empty
    public int TotalBytes { get; set; }
    public string? SampleHex { get; set; }
}

public class UIntSummary
{
    public string Status { get; set; } = ""; // ok, truncated, error
    public uint Value { get; set; }
}

public class CountSummary
{
    public string Status { get; set; } = ""; // ok, empty, unknown_stride
    public int TotalBytes { get; set; }
    public int Count { get; set; }
}
