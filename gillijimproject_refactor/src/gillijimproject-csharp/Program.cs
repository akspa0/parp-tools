using System;
using System.IO;
using System.Linq;
using U = GillijimProject.Utilities.Utilities;
using GillijimProject.WowFiles.Alpha;
using System.Security.Cryptography;
using GillijimProject.WowFiles.LichKing;

namespace GillijimProject;

/// <summary>
/// [PORT] Minimal CLI stub. Mirrors lib/gillijimproject/main.cpp flow incrementally.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: gillijimproject-csharp <WDT_ALPHA_PATH> [-o|--out <OUTPUT_DIR>] | --adt-lk <ADT_LK_PATH> [--out <OUTPUT_DIR>] [--compare] [--report-mclq]");
            return 1;
        }

        // LK ADT roundtrip mode
        if (args.Contains("--adt-lk"))
        {
            return HandleAdtLkRoundtrip(args);
        }

        string? path = null;
        string? outputDir = null;
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a == "-o" || a == "--out")
            {
                if (i + 1 >= args.Length)
                {
                    Console.Error.WriteLine("Missing value for -o|--out");
                    return 1;
                }
                outputDir = args[++i];
            }
            else if (!a.StartsWith("-"))
            {
                if (path == null) path = a;
            }
        }

        if (path == null || !File.Exists(path))
        {
            Console.Error.WriteLine($"File not found: {path}");
            return 2;
        }

        if (string.IsNullOrWhiteSpace(outputDir))
        {
            var baseName = Path.GetFileNameWithoutExtension(path);
            var dirName = baseName + "_out";
            var parent = Path.GetDirectoryName(path) ?? string.Empty;
            outputDir = Path.Combine(parent, dirName);
        }
        Directory.CreateDirectory(outputDir);
        Console.WriteLine($"[INFO] Output directory: {Path.GetFullPath(outputDir)}");

        try
        {
            var bytes = U.GetWholeFile(path);
            Console.WriteLine($"[INFO] Read {bytes.Length} bytes from {path}");
            var wdtVersion = U.GetWdtVersion(path);
            Console.WriteLine($"[INFO] WDT version guess: {wdtVersion}");

            var wdtAlpha = new WdtAlpha(path);
            var existing = wdtAlpha.GetExistingAdtsNumbers();
            Console.WriteLine($"[INFO] ADT tiles present: {existing.Count}");
            Console.WriteLine("[INFO] First 16 tile indices: " + string.Join(", ", existing.Take(16)));

            var mdnm = wdtAlpha.GetMdnmFileNames();
            var monm = wdtAlpha.GetMonmFileNames();
            Console.WriteLine($"[INFO] MDNM count: {mdnm.Count}, MONM count: {monm.Count}");
            Console.WriteLine("[INFO] MDNM sample: " + string.Join(" | ", mdnm.Take(3)));
            Console.WriteLine("[INFO] MONM sample: " + string.Join(" | ", monm.Take(3)));

            // Write LK WDT
            var wdtLk = wdtAlpha.ToWdt();
            wdtLk.ToFile(outputDir);
            Console.WriteLine($"[INFO] Wrote WDT: {Path.Combine(outputDir, Path.GetFileName(path) + "_new")}");

            // Write LK ADTs for existing tiles
            var adtOffsets = wdtAlpha.GetAdtOffsetsInMain();
            int written = 0;
            foreach (var idx in existing)
            {
                var adtAlphaTile = new AdtAlpha(path, adtOffsets[idx], idx);
                var lkAdt = adtAlphaTile.ToAdtLk(mdnm, monm);
                lkAdt.ToFile(outputDir);
                written++;
            }
            Console.WriteLine($"[INFO] Wrote {written} ADT files (Alpha → LK, with MCNK content).");

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 3;
        }
    }

    private static int HandleAdtLkRoundtrip(string[] args)
    {
        try
        {
            string? adtPath = null;
            string? outputDir = null;
            bool compare = false;
            bool reportMclq = false;

            for (int i = 0; i < args.Length; i++)
            {
                var a = args[i];
                if (a == "--adt-lk")
                {
                    if (i + 1 >= args.Length)
                    {
                        Console.Error.WriteLine("Missing value for --adt-lk");
                        return 1;
                    }
                    adtPath = args[++i];
                }
                else if (a == "-o" || a == "--out")
                {
                    if (i + 1 >= args.Length)
                    {
                        Console.Error.WriteLine("Missing value for -o|--out");
                        return 1;
                    }
                    outputDir = args[++i];
                }
                else if (a == "--compare")
                {
                    compare = true;
                }
                else if (a == "--report-mclq")
                {
                    reportMclq = true;
                }
            }

            if (adtPath == null || !File.Exists(adtPath))
            {
                Console.Error.WriteLine($"File not found: {adtPath}");
                return 2;
            }

            if (string.IsNullOrWhiteSpace(outputDir))
            {
                var baseName = Path.GetFileNameWithoutExtension(adtPath);
                var dirName = baseName + "_out";
                var parent = Path.GetDirectoryName(adtPath) ?? string.Empty;
                outputDir = Path.Combine(parent, dirName);
            }
            Directory.CreateDirectory(outputDir);
            Console.WriteLine($"[INFO] Output directory: {Path.GetFullPath(outputDir)}");

            var inBytes = U.GetWholeFile(adtPath);
            var fileNameOnly = Path.GetFileName(adtPath);
            var adt = new AdtLk(inBytes, fileNameOnly);

            // Write out
            adt.ToFile(outputDir);
            var outPath = Path.Combine(outputDir, fileNameOnly);
            Console.WriteLine($"[INFO] Wrote ADT: {outPath}");

            // Reload for validation
            var outBytes = U.GetWholeFile(outPath);
            var reloaded = new AdtLk(outBytes, fileNameOnly);

            // Optional integrity validation
            var integrityOk = reloaded.ValidateIntegrity();
            Console.WriteLine($"[INFO] Integrity after write: {(integrityOk ? "OK" : "FAIL")}");

            if (reportMclq)
            {
                Console.WriteLine($"[INFO] MCNKs: input={adt.GetMcnkCount()}, output={reloaded.GetMcnkCount()}");
                Console.WriteLine($"[INFO] MCLQ count: input={adt.CountMclqChunks()}, output={reloaded.CountMclqChunks()}");
            }

            if (compare)
            {
                var inHash = ComputeSha256Hex(adtPath);
                var outHash = ComputeSha256Hex(outPath);
                Console.WriteLine($"[COMPARE] Input size:  {inBytes.Length:N0} bytes");
                Console.WriteLine($"[COMPARE] Output size: {outBytes.Length:N0} bytes");
                Console.WriteLine($"[COMPARE] Input SHA256:  {inHash}");
                Console.WriteLine($"[COMPARE] Output SHA256: {outHash}");
                Console.WriteLine($"[COMPARE] Byte parity: {(inHash == outHash ? "MATCH" : "DIFF")}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 3;
        }
    }

    private static string ComputeSha256Hex(string filePath)
    {
        using var sha = SHA256.Create();
        using var fs = File.OpenRead(filePath);
        var hash = sha.ComputeHash(fs);
        return BitConverter.ToString(hash).Replace("-", string.Empty).ToLowerInvariant();
    }
}
