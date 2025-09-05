using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.IO;
using GillijimProject.Next.Core.Analysis;

namespace GillijimProject.Next.Cli.Commands;

public static class WdlDumpCommand
{
    public static int Run(string[] args)
    {
        var opts = ParseOptions(args);
        if (!opts.TryGetValue("--in", out var inPath) || string.IsNullOrWhiteSpace(inPath) || !File.Exists(inPath))
        {
            Console.Error.WriteLine("[wdl-dump] Missing or invalid --in <path-to.wdl>.");
            Console.Error.WriteLine("Usage: wdl-dump --in <path-to.wdl> [--out-root <dir>] [--whiteplate-threshold <0..1>]");
            return 2;
        }

        try
        {
            // Resolve output root and per-run directory: <out-root>/<filename_MMDDYY_HHMMSS>
            string outRoot = opts.TryGetValue("--out-root", out var oroot) && !string.IsNullOrWhiteSpace(oroot) ? oroot : "out";
            string baseName = Path.GetFileNameWithoutExtension(inPath);
            string stamp = DateTime.Now.ToString("MMddyy_HHmmss");
            string runDir = Path.Combine(outRoot, $"{baseName}_{stamp}");
            Directory.CreateDirectory(runDir);

            var originalOut = Console.Out;
            var originalErr = Console.Error;
            var logPath = Path.Combine(runDir, "wdl-dump.log");
            using var logWriter = new StreamWriter(File.Open(logPath, FileMode.Create, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
            using var outTee = new TeeWriter(originalOut, logWriter);
            using var errTee = new TeeWriter(originalErr, logWriter);
            Console.SetOut(outTee);
            Console.SetError(errTee);
            try
            {
                var wdl = AlphaReader.ParseWdl(inPath);
                int present = 0;
                int tilesWithHoles = 0;
                WdlTile? sample = null;
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        var t = wdl.Tiles[y, x];
                        if (t is null) continue;
                        present++;
                        if (sample is null) sample = t;
                        bool anyHole = false;
                        var rows = t.HoleMask16;
                        for (int r = 0; r < rows.Length; r++)
                        {
                            if (rows[r] != 0) { anyHole = true; break; }
                        }
                        if (anyHole) tilesWithHoles++;
                    }
                }

                Console.WriteLine($"[wdl-dump] file={Path.GetFileName(inPath)} present_tiles={present} of 4096 tiles_with_holes={tilesWithHoles}");
                if (sample is not null)
                {
                    Console.WriteLine($"[wdl-dump] sample height17[0,0]={sample.Height17[0,0]} height16[0,0]={sample.Height16[0,0]}");
                }

                // Whiteplate analysis (metadata only)
                double threshold = 0.66; // default dominance threshold
                if (opts.TryGetValue("--whiteplate-threshold", out var thrRaw) && double.TryParse(thrRaw, out var thrVal))
                {
                    if (thrVal >= 0 && thrVal <= 1) threshold = thrVal;
                }
                var analysis = WdlAnalyzer.Analyze(wdl, threshold);
                bool qualifies = analysis.BaselineShare >= threshold;
                var baseHex = analysis.BaselineHash.ToString("X16");
                Console.WriteLine($"[wdl-dump] whiteplate present={analysis.PresentTiles} zero={analysis.ZeroTiles} baselineCount={analysis.BaselineCount} baselineShare={analysis.BaselineShare:P1} baselineIsZero={analysis.BaselineIsZero} baselineHash=0x{baseHex} qualifies={(qualifies ? "yes" : "no")}");
                if (sample is not null)
                {
                    var sigSample = WdlAnalyzer.ComputeSignature(sample).ToString("X16");
                    bool sampleIsWhiteplate = qualifies && string.Equals(sigSample, baseHex, StringComparison.OrdinalIgnoreCase);
                    Console.WriteLine($"[wdl-dump] sample_signature=0x{sigSample} is_whiteplate={sampleIsWhiteplate}");
                }

                // Persist summary.json for scientific analysis/debugging
                var summary = new DumpSummary(
                    InputFile: Path.GetFullPath(inPath),
                    OutputRunDir: Path.GetFullPath(runDir),
                    PresentTiles: present,
                    TilesWithHoles: tilesWithHoles,
                    SampleHeight17_0_0: sample?.Height17[0, 0] ?? 0,
                    SampleHeight16_0_0: sample?.Height16[0, 0] ?? 0,
                    WhiteplateBaselineHashHex: baseHex,
                    WhiteplateBaselineCount: analysis.BaselineCount,
                    WhiteplateBaselineShare: analysis.BaselineShare,
                    WhiteplateBaselineIsZero: analysis.BaselineIsZero,
                    WhiteplateThreshold: threshold
                );
                var summaryJson = JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(Path.Combine(runDir, "summary.json"), summaryJson);

                // Also emit baseline.txt for quick scripting
                File.WriteAllText(Path.Combine(runDir, "baseline.txt"), $"hash=0x{baseHex}\ncount={analysis.BaselineCount}\nshare={analysis.BaselineShare:F6}\nthreshold={threshold:F2}\nisZero={analysis.BaselineIsZero}\n");

                // Emit tiles.csv with per-tile info for spreadsheet analysis
                var csvPath = Path.Combine(runDir, "tiles.csv");
                var sb = new StringBuilder(4096 * 64);
                sb.AppendLine("x,y,height17_0_0,height16_0_0,holeMaskRow0,signatureHex,isWhiteplate");
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        var t = wdl.Tiles[y, x];
                        if (t is null) continue;
                        var sig = WdlAnalyzer.ComputeSignature(t);
                        bool isWp = qualifies && sig == analysis.BaselineHash;
                        string sigHex = sig.ToString("X16");
                        short h17 = t.Height17[0, 0];
                        short h16 = t.Height16[0, 0];
                        ushort hm0 = t.HoleMask16.Length > 0 ? t.HoleMask16[0] : (ushort)0;
                        sb.Append(x).Append(',').Append(y).Append(',')
                          .Append(h17).Append(',').Append(h16).Append(',')
                          .Append(hm0).Append(',')
                          .Append(sigHex).Append(',')
                          .Append(isWp ? "true" : "false").AppendLine();
                    }
                }
                File.WriteAllText(csvPath, sb.ToString());
                Console.WriteLine($"[wdl-dump] wrote tiles.csv: {csvPath}");
            }
            finally
            {
                // restore console streams
                Console.SetOut(originalOut);
                Console.SetError(originalErr);
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[wdl-dump] Failed: {ex.Message}");
            return 3;
        }
    }

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (key.StartsWith("--", StringComparison.Ordinal))
            {
                if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    dict[key] = args[++i];
                }
                else
                {
                    dict[key] = string.Empty;
                }
            }
        }
        return dict;
    }
}

internal sealed record DumpSummary(
    string InputFile,
    string OutputRunDir,
    int PresentTiles,
    int TilesWithHoles,
    short SampleHeight17_0_0,
    short SampleHeight16_0_0,
    string WhiteplateBaselineHashHex,
    int WhiteplateBaselineCount,
    double WhiteplateBaselineShare,
    bool WhiteplateBaselineIsZero,
    double WhiteplateThreshold
);

internal sealed class TeeWriter : TextWriter
{
    private readonly TextWriter _a;
    private readonly TextWriter _b;
    private readonly object _lock = new object();
    public TeeWriter(TextWriter a, TextWriter b) { _a = a; _b = b; }
    public override Encoding Encoding => Encoding.UTF8;
    public override void Write(char value) { lock (_lock) { _a.Write(value); _b.Write(value); } }
    public override void Write(string? value) { lock (_lock) { _a.Write(value); _b.Write(value); } }
    public override void WriteLine(string? value) { lock (_lock) { _a.WriteLine(value); _b.WriteLine(value); } }
    public override void Flush() { lock (_lock) { _a.Flush(); _b.Flush(); } }
}
