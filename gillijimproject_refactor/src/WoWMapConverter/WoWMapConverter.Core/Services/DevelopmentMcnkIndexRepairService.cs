using System.Text;

namespace WoWMapConverter.Core.Services;

public sealed record DevelopmentMcnkIndexRepairReport(
    string SourceLabel,
    int ChunkCount,
    int MismatchCount,
    bool UsedMcin,
    bool SkippedEmptyFile,
    byte[]? RepairedBytes);

public static class DevelopmentMcnkIndexRepairService
{
    public static DevelopmentMcnkIndexRepairReport RepairBytes(byte[] bytes, string sourceLabel, bool writeChanges)
    {
        if (bytes.Length == 0)
        {
            return new DevelopmentMcnkIndexRepairReport(
                SourceLabel: sourceLabel,
                ChunkCount: 0,
                MismatchCount: 0,
                UsedMcin: false,
                SkippedEmptyFile: true,
                RepairedBytes: writeChanges ? bytes : null);
        }

        List<McnkOffsetInfo> offsets = ResolveMcnkOffsets(bytes, out bool usedMcin);
        if (offsets.Count == 0)
        {
            return new DevelopmentMcnkIndexRepairReport(
                SourceLabel: sourceLabel,
                ChunkCount: 0,
                MismatchCount: 0,
                UsedMcin: usedMcin,
                SkippedEmptyFile: false,
                RepairedBytes: writeChanges ? bytes : null);
        }

        byte[] workingBytes = writeChanges ? (byte[])bytes.Clone() : bytes;
        int mismatchCount = 0;

        foreach (McnkOffsetInfo offset in offsets)
        {
            int headerStart = offset.Offset + 8;
            if (headerStart + 12 > workingBytes.Length)
                continue;

            int actualX = BitConverter.ToInt32(workingBytes, headerStart + 4);
            int actualY = BitConverter.ToInt32(workingBytes, headerStart + 8);
            if (actualX == offset.ExpectedX && actualY == offset.ExpectedY)
                continue;

            mismatchCount++;
            if (!writeChanges)
                continue;

            WriteInt32(workingBytes, headerStart + 4, offset.ExpectedX);
            WriteInt32(workingBytes, headerStart + 8, offset.ExpectedY);
        }

        return new DevelopmentMcnkIndexRepairReport(
            SourceLabel: sourceLabel,
            ChunkCount: offsets.Count,
            MismatchCount: mismatchCount,
            UsedMcin: usedMcin,
            SkippedEmptyFile: false,
            RepairedBytes: writeChanges ? workingBytes : null);
    }

    private static List<McnkOffsetInfo> ResolveMcnkOffsets(byte[] bytes, out bool usedMcin)
    {
        List<McnkOffsetInfo> fromMcin = TryReadMcnkOffsetsFromMcin(bytes);
        if (fromMcin.Count > 0)
        {
            usedMcin = true;
            return fromMcin;
        }

        usedMcin = false;
        return ReadMcnkOffsetsByChunkScan(bytes);
    }

    private static List<McnkOffsetInfo> TryReadMcnkOffsetsFromMcin(byte[] bytes)
    {
        int? mcinOffset = FindTopLevelChunkOffset(bytes, "MCIN");
        if (mcinOffset == null)
            return new List<McnkOffsetInfo>();

        int chunkOffset = mcinOffset.Value;
        int chunkSize = BitConverter.ToInt32(bytes, chunkOffset + 4);
        int dataStart = chunkOffset + 8;
        if (chunkSize < 256 * 16 || dataStart + chunkSize > bytes.Length)
            return new List<McnkOffsetInfo>();

        var offsets = new List<McnkOffsetInfo>(256);
        for (int i = 0; i < 256; i++)
        {
            int entryOffset = dataStart + (i * 16);
            int mcnkOffset = BitConverter.ToInt32(bytes, entryOffset);
            if (mcnkOffset <= 0 || mcnkOffset + 8 > bytes.Length)
                continue;

            if (!string.Equals(ReadNormalizedFourCc(bytes, mcnkOffset), "MCNK", StringComparison.Ordinal))
                continue;

            offsets.Add(new McnkOffsetInfo(mcnkOffset, i % 16, i / 16));
        }

        return offsets;
    }

    private static List<McnkOffsetInfo> ReadMcnkOffsetsByChunkScan(byte[] bytes)
    {
        var offsets = new List<McnkOffsetInfo>();
        int position = 0;
        int ordinal = 0;

        while (position + 8 <= bytes.Length)
        {
            string fourCc = ReadNormalizedFourCc(bytes, position);
            int chunkSize = BitConverter.ToInt32(bytes, position + 4);
            if (chunkSize < 0 || position + 8 + chunkSize > bytes.Length)
                break;

            if (string.Equals(fourCc, "MCNK", StringComparison.Ordinal))
            {
                offsets.Add(new McnkOffsetInfo(position, ordinal % 16, ordinal / 16));
                ordinal++;
            }

            position += 8 + chunkSize;
        }

        return offsets;
    }

    private static int? FindTopLevelChunkOffset(byte[] bytes, string chunkName)
    {
        int position = 0;
        while (position + 8 <= bytes.Length)
        {
            string fourCc = ReadNormalizedFourCc(bytes, position);
            int chunkSize = BitConverter.ToInt32(bytes, position + 4);
            if (chunkSize < 0 || position + 8 + chunkSize > bytes.Length)
                return null;

            if (string.Equals(fourCc, chunkName, StringComparison.Ordinal))
                return position;

            position += 8 + chunkSize;
        }

        return null;
    }

    private static string ReadNormalizedFourCc(byte[] bytes, int offset)
    {
        string sig = Encoding.ASCII.GetString(bytes, offset, 4);
        string reversedSig = new(sig.Reverse().ToArray());

        return reversedSig switch
        {
            "MVER" or "MHDR" or "MCIN" or "MCNK" or "MTEX" or "MMDX" or "MMID" or "MWMO" or "MWID" or "MDDF" or "MODF" or "MH2O" => reversedSig,
            _ => sig
        };
    }

    private static void WriteInt32(byte[] bytes, int offset, int value)
    {
        byte[] valueBytes = BitConverter.GetBytes(value);
        Buffer.BlockCopy(valueBytes, 0, bytes, offset, sizeof(int));
    }

    private sealed record McnkOffsetInfo(int Offset, int ExpectedX, int ExpectedY);
}
