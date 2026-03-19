using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWRollback.Cli.Commands;

internal sealed record McnkIndexRepairFileReport(
    string Path,
    int ChunkCount,
    int MismatchCount,
    bool UsedMcin,
    bool SkippedEmptyFile,
    bool WroteOutput,
    byte[]? RepairedBytes);

internal static class McnkIndexRepairService
{
    public static McnkIndexRepairFileReport RepairFile(string path, bool writeChanges, string? outputPath = null)
    {
        byte[] bytes = File.ReadAllBytes(path);
        McnkIndexRepairFileReport report = RepairBytes(bytes, path, writeChanges, outputPath);

        if (writeChanges && report.WroteOutput && report.RepairedBytes != null)
        {
            string destinationPath = outputPath ?? path;
            string? destinationDir = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrEmpty(destinationDir))
            {
                Directory.CreateDirectory(destinationDir);
            }

            File.WriteAllBytes(destinationPath, report.RepairedBytes);
        }

        return report;
    }

    public static McnkIndexRepairFileReport RepairBytes(byte[] bytes, string sourceLabel, bool writeChanges, string? outputPath = null)
    {
        if (bytes.Length == 0)
        {
            return new McnkIndexRepairFileReport(
                Path: outputPath ?? sourceLabel,
                ChunkCount: 0,
                MismatchCount: 0,
                UsedMcin: false,
                SkippedEmptyFile: true,
                WroteOutput: false,
                RepairedBytes: writeChanges ? bytes : null);
        }

        var mcnkOffsets = ResolveMcnkOffsets(bytes, out bool usedMcin);
        if (mcnkOffsets.Count == 0)
        {
            return new McnkIndexRepairFileReport(
                Path: outputPath ?? sourceLabel,
                ChunkCount: 0,
                MismatchCount: 0,
                UsedMcin: usedMcin,
                SkippedEmptyFile: false,
                WroteOutput: false,
                RepairedBytes: writeChanges ? bytes : null);
        }

        byte[] workingBytes = writeChanges ? (byte[])bytes.Clone() : bytes;
        int mismatchCount = 0;

        foreach (var mcnkOffset in mcnkOffsets)
        {
            int headerStart = mcnkOffset.Offset + 8;
            if (headerStart + 12 > workingBytes.Length)
            {
                continue;
            }

            int actualIndexX = BitConverter.ToInt32(workingBytes, headerStart + 4);
            int actualIndexY = BitConverter.ToInt32(workingBytes, headerStart + 8);
            if (actualIndexX == mcnkOffset.ExpectedIndexX && actualIndexY == mcnkOffset.ExpectedIndexY)
            {
                continue;
            }

            mismatchCount++;
            if (!writeChanges)
            {
                continue;
            }

            WriteInt32(workingBytes, headerStart + 4, mcnkOffset.ExpectedIndexX);
            WriteInt32(workingBytes, headerStart + 8, mcnkOffset.ExpectedIndexY);
        }

        return new McnkIndexRepairFileReport(
            Path: outputPath ?? sourceLabel,
            ChunkCount: mcnkOffsets.Count,
            MismatchCount: mismatchCount,
            UsedMcin: usedMcin,
            SkippedEmptyFile: false,
            WroteOutput: writeChanges && mismatchCount > 0,
            RepairedBytes: writeChanges ? workingBytes : null);
    }

    private static List<McnkOffsetInfo> ResolveMcnkOffsets(byte[] bytes, out bool usedMcin)
    {
        var viaMcin = TryReadMcnkOffsetsFromMcin(bytes);
        if (viaMcin.Count > 0)
        {
            usedMcin = true;
            return viaMcin;
        }

        usedMcin = false;
        return ReadMcnkOffsetsByChunkScan(bytes);
    }

    private static List<McnkOffsetInfo> TryReadMcnkOffsetsFromMcin(byte[] bytes)
    {
        int? mcinChunkOffset = FindTopLevelChunkOffset(bytes, "MCIN");
        if (mcinChunkOffset == null)
        {
            return new List<McnkOffsetInfo>();
        }

        int mcinOffset = mcinChunkOffset.Value;
        int mcinSize = BitConverter.ToInt32(bytes, mcinOffset + 4);
        int mcinDataStart = mcinOffset + 8;
        if (mcinSize < 256 * 16 || mcinDataStart + mcinSize > bytes.Length)
        {
            return new List<McnkOffsetInfo>();
        }

        var offsets = new List<McnkOffsetInfo>(256);
        for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
        {
            int entryOffset = mcinDataStart + (chunkIndex * 16);
            int mcnkOffset = BitConverter.ToInt32(bytes, entryOffset);
            if (mcnkOffset <= 0 || mcnkOffset + 8 > bytes.Length)
            {
                continue;
            }

            if (!string.Equals(ReadNormalizedFourCc(bytes, mcnkOffset), "MCNK", StringComparison.Ordinal))
            {
                continue;
            }

            offsets.Add(new McnkOffsetInfo(mcnkOffset, chunkIndex % 16, chunkIndex / 16));
        }

        return offsets;
    }

    private static List<McnkOffsetInfo> ReadMcnkOffsetsByChunkScan(byte[] bytes)
    {
        var offsets = new List<McnkOffsetInfo>();
        int position = 0;
        int mcnkOrdinal = 0;

        while (position + 8 <= bytes.Length)
        {
            string fourCc = ReadNormalizedFourCc(bytes, position);
            int chunkSize = BitConverter.ToInt32(bytes, position + 4);
            if (chunkSize < 0 || position + 8 + chunkSize > bytes.Length)
            {
                break;
            }

            if (string.Equals(fourCc, "MCNK", StringComparison.Ordinal))
            {
                offsets.Add(new McnkOffsetInfo(position, mcnkOrdinal % 16, mcnkOrdinal / 16));
                mcnkOrdinal++;
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
            {
                return null;
            }

            if (string.Equals(fourCc, chunkName, StringComparison.Ordinal))
            {
                return position;
            }

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

    private sealed record McnkOffsetInfo(int Offset, int ExpectedIndexX, int ExpectedIndexY);
}

public static class RepairMcnkIndicesCommand
{
    public static int Execute(Dictionary<string, string> opts)
    {
        string? inputPath = opts.GetValueOrDefault("in") ?? opts.GetValueOrDefault("input");
        string? outputPath = opts.GetValueOrDefault("out");
        bool auditOnly = opts.ContainsKey("audit-only");
        bool overwrite = opts.ContainsKey("overwrite");

        if (string.IsNullOrWhiteSpace(inputPath))
        {
            Console.WriteLine("Usage: repair-mcnk-indices --in <file-or-dir> [--out <file-or-dir>] [--audit-only] [--overwrite]");
            return 1;
        }

        if (!File.Exists(inputPath) && !Directory.Exists(inputPath))
        {
            Console.WriteLine($"[ERROR] Input path not found: {inputPath}");
            return 1;
        }

        if (!auditOnly && !overwrite && string.IsNullOrWhiteSpace(outputPath))
        {
            Console.WriteLine("[ERROR] Specify --out for repaired output or use --overwrite / --audit-only.");
            return 1;
        }

        List<McnkIndexRepairFileReport> reports;
        if (File.Exists(inputPath))
        {
            string? destination = auditOnly || overwrite ? inputPath : outputPath;
            reports = new List<McnkIndexRepairFileReport>
            {
                McnkIndexRepairService.RepairFile(inputPath, !auditOnly, overwrite ? inputPath : destination)
            };
        }
        else
        {
            reports = ProcessDirectory(inputPath, outputPath, auditOnly, overwrite);
        }

        PrintSummary(reports, auditOnly);
        return 0;
    }

    private static List<McnkIndexRepairFileReport> ProcessDirectory(string inputDir, string? outputDir, bool auditOnly, bool overwrite)
    {
        var reports = new List<McnkIndexRepairFileReport>();
        foreach (string adtPath in Directory.EnumerateFiles(inputDir, "*.adt", SearchOption.TopDirectoryOnly)
            .Where(path => !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
                && !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase))
            .OrderBy(Path.GetFileName))
        {
            string targetPath = adtPath;
            if (!auditOnly && !overwrite)
            {
                string fileName = Path.GetFileName(adtPath);
                targetPath = Path.Combine(outputDir!, fileName);
            }

            reports.Add(McnkIndexRepairService.RepairFile(adtPath, !auditOnly, overwrite ? adtPath : targetPath));
        }

        return reports;
    }

    private static void PrintSummary(List<McnkIndexRepairFileReport> reports, bool auditOnly)
    {
        int totalFiles = reports.Count;
        int emptyFiles = reports.Count(report => report.SkippedEmptyFile);
        int filesWithChunks = reports.Count(report => report.ChunkCount > 0);
        int filesWithMismatches = reports.Count(report => report.MismatchCount > 0);
        int totalMismatches = reports.Sum(report => report.MismatchCount);

        Console.WriteLine($"[MCNK] Files scanned: {totalFiles}");
        Console.WriteLine($"[MCNK] Empty files skipped: {emptyFiles}");
        Console.WriteLine($"[MCNK] Files with MCNK chunks: {filesWithChunks}");
        Console.WriteLine($"[MCNK] Files with index mismatches: {filesWithMismatches}");
        Console.WriteLine($"[MCNK] Total chunk index mismatches: {totalMismatches}");

        foreach (var report in reports.Where(report => report.MismatchCount > 0))
        {
            string mode = auditOnly ? "audit" : report.WroteOutput ? "repaired" : "unchanged";
            Console.WriteLine($"  [{mode}] {Path.GetFileName(report.Path)} mismatches={report.MismatchCount} chunks={report.ChunkCount} source={(report.UsedMcin ? "MCIN" : "scan")}");
        }
    }
}