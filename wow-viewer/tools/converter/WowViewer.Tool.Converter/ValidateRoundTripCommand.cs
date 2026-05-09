using System.Diagnostics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.Converter;

/// <summary>
/// Roundtrip validation: supports two modes.
/// Mode 1 (--mode lk): LK ADT → Alpha → LK, comparing LK output vs LK input.
/// Mode 2 (--mode alpha): Alpha WDT → LK → Alpha, comparing Alpha output vs Alpha input.
/// </summary>
internal static class ValidateRoundTripCommand
{
    public static void Run(string[] args)
    {
        string? clientRoot = GetOpt(args, "--client-root", "-c");
        string mapName = GetOpt(args, "--map", "-m") ?? "Azeroth";
        string mode = GetOpt(args, "--mode", "-M") ?? "lk";
        int? limit = int.TryParse(GetOpt(args, "--limit", "-n"), out int n) ? n : null;
        bool verbose = args.Any(a => a is "--verbose" or "-v");
        float hEps = float.TryParse(GetOpt(args, "--height-epsilon", "--he"), out float he) ? he : 0.5f;
        float aEps = float.TryParse(GetOpt(args, "--alpha-epsilon", "--ae"), out float ae) ? ae : 0.05f;

        if (string.IsNullOrEmpty(clientRoot) || !Directory.Exists(Path.GetFullPath(clientRoot)))
        {
            Console.Error.WriteLine("Error: --client-root <dir> required and must exist.");
            Environment.ExitCode = 1;
            return;
        }
        clientRoot = Path.GetFullPath(clientRoot);

        Console.WriteLine($"validate-roundtrip  mode={mode}  map={mapName}  hε={hEps}  aε={aEps}");
        Console.WriteLine($"  client: {clientRoot}");
        var sw = Stopwatch.StartNew();

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);

        if (mode.Equals("alpha", StringComparison.OrdinalIgnoreCase))
            RunAlphaRoundTrip(catalog, mapName, limit, verbose, hEps, aEps);
        else
            RunLkRoundTrip(catalog, mapName, limit, verbose, hEps, aEps);

        Console.WriteLine($"\n  Elapsed: {sw.ElapsedMilliseconds}ms");
    }

    // Alpha WDT → LK ADTs → Alpha WDT, compare Alpha vs Alpha
    static void RunAlphaRoundTrip(NativeMpqService catalog, string mapName, int? limit, bool verbose, float hEps, float aEps)
    {
        byte[]? wdtBytes = catalog.ReadFile($"World\\Maps\\{mapName}\\{mapName}.wdt");
        if (wdtBytes is null || !AlphaWdtReader.IsAlphaWdt(wdtBytes))
        { Console.Error.WriteLine("Cannot read Alpha WDT."); Environment.ExitCode = 1; return; }

        var tiles = AlphaWdtReader.ReadExistingTiles(wdtBytes);
        Console.WriteLine($"  Alpha tiles: {tiles.Count}");

        int ok = 0, fail = 0, skip = 0, done = 0;
        float gMaxH = 0, gMaxA = 0;

        foreach (var (tx, ty) in tiles.OrderBy(t => t.Y * 64 + t.X))
        {
            if (limit.HasValue && done >= limit.Value) break;
            if (!AlphaWdtReader.TryReadTile(wdtBytes, tx, ty, out var orig) || orig == null) { skip++; continue; }
            done++;

            try
            {
                // Alpha → LK
                var lk = AlphaToLkConverter.ConvertTile(orig, tx, ty);
                byte[] lkBytes = LkAdtWriter.Build(lk);
                // LK → Alpha: use the PROVEN reader from LkToAlphaCommand
                var lkRead = LkToAlphaCommand.ReadLkAdtPublic(lkBytes, tx, ty);
                var rt = LkToAlphaConverter.ConvertTile(lkRead, tx, ty);

                if (lkRead.Chunks.Count == 0)
                {
                    fail++;
                    continue;
                }

                var (pass, maxH, maxA, msgs) = CompareAlpha(orig, rt, hEps, aEps);
                gMaxH = MathF.Max(gMaxH, maxH); gMaxA = MathF.Max(gMaxA, maxA);
                if (pass) { ok++; if (verbose) Console.WriteLine($"  PASS ({tx},{ty}) Δh={maxH:F3} Δα={maxA:F3}"); }
                else { fail++; Console.WriteLine($"  FAIL ({tx},{ty}): {string.Join("; ", msgs)}"); }
            }
            catch (Exception ex) { fail++; Console.WriteLine($"  ERR ({tx},{ty}): {ex.Message}"); }
        }
        PrintSummary(done, ok, fail, skip, gMaxH, gMaxA);
    }

    // LK ADTs (from MPQ) → Alpha WDT → LK ADTs, compare LK vs LK
    static void RunLkRoundTrip(NativeMpqService catalog, string mapName, int? limit, bool verbose, float hEps, float aEps)
    {
        int ok = 0, fail = 0, skip = 0, done = 0;
        float gMaxH = 0, gMaxA = 0;
        int found = 0;

        for (int ty = 0; ty < 64; ty++)
        {
            for (int tx = 0; tx < 64; tx++)
            {
                if (limit.HasValue && done >= limit.Value) break;
                byte[]? adtBytes = catalog.ReadFile($"World\\Maps\\{mapName}\\{mapName}_{tx}_{ty}.adt");
                if (adtBytes is null) continue;
                found++;

                try
                {
                    var origLk = LkToAlphaCommand.ReadLkAdtPublic(adtBytes, tx, ty);
                    // LK → Alpha
                    var alpha = LkToAlphaConverter.ConvertTile(origLk, tx, ty);
                    // Alpha → LK
                    var rtLk = AlphaToLkConverter.ConvertTile(alpha, tx, ty);

                    var (pass, maxH, maxA, msgs) = CompareLk(origLk, rtLk, hEps, aEps);
                    gMaxH = MathF.Max(gMaxH, maxH); gMaxA = MathF.Max(gMaxA, maxA);
                    done++;
                    if (pass) { ok++; if (verbose) Console.WriteLine($"  PASS ({tx},{ty}) Δh={maxH:F3} Δα={maxA:F3}"); }
                    else { fail++; Console.WriteLine($"  FAIL ({tx},{ty}): {string.Join("; ", msgs)}"); }
                }
                catch (Exception ex) { fail++; done++; Console.WriteLine($"  ERR ({tx},{ty}): {ex.Message}"); }
            }
            if (limit.HasValue && done >= limit.Value) break;
        }
        Console.WriteLine($"  LK tiles found: {found}");
        PrintSummary(done, ok, fail, skip, gMaxH, gMaxA);
    }

    static void PrintSummary(int done, int ok, int fail, int skip, float maxH, float maxA)
    {
        Console.WriteLine($"\n--- Summary ---");
        Console.WriteLine($"  Processed: {done}  Pass: {ok}  Fail: {fail}  Skip: {skip}");
        Console.WriteLine($"  Global max Δheight: {maxH:F6}");
        Console.WriteLine($"  Global max Δalpha:  {maxA:F6}");
    }

    static (bool pass, float maxH, float maxA, List<string> msgs) CompareAlpha(
        AlphaTileData orig, AlphaTileData rt, float hEps, float aEps)
    {
        var msgs = new List<string>();
        float maxH = 0, maxA = 0;
        int hBad = 0;
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
            {
                float d = MathF.Abs(orig.Heightmap[y, x] - rt.Heightmap[y, x]);
                if (d > maxH) maxH = d;
                if (d > hEps) hBad++;
            }
        if (hBad > 0)
        {
            float origVal = 0, rtVal = 0;
            int badX = -1, badY = -1;
            for (int y = 0; y < 257; y++)
                for (int x = 0; x < 257; x++)
                    if (MathF.Abs(orig.Heightmap[y, x] - rt.Heightmap[y, x]) > hEps)
                    {
                        origVal = orig.Heightmap[y, x];
                        rtVal = rt.Heightmap[y, x];
                        badX = x; badY = y;
                        y = 257; break;
                    }
            msgs.Add($"height:{hBad} exceed ε (max={maxH:F3}) [orig={origVal:F3} rt={rtVal:F3} at {badX},{badY}]");
        }

        if (orig.McalAlphaPack != null && rt.McalAlphaPack != null)
        {
            float origVal = 0, rtVal = 0;
            int badX = -1, badY = -1, badL = -1;
            int aBad = 0;
            for (int y = 0; y < 256; y++)
                for (int x = 0; x < 256; x++)
                    for (int l = 0; l < 4; l++)
                    {
                        float d = MathF.Abs(orig.McalAlphaPack[y, x, l] - rt.McalAlphaPack[y, x, l]);
                        if (d > maxA) maxA = d;
                        if (d > aEps)
                        {
                            if (badX == -1)
                            {
                                origVal = orig.McalAlphaPack[y, x, l];
                                rtVal = rt.McalAlphaPack[y, x, l];
                                badX = x; badY = y; badL = l;
                            }
                            aBad++;
                        }
                    }
            if (aBad > 0)
                msgs.Add($"alpha:{aBad} exceed ε (max={maxA:F3}) [orig={origVal:F3} rt={rtVal:F3} at {badX},{badY} l={badL}]");
        }
        else if (orig.McalAlphaPack != null) msgs.Add("alpha lost");

        if (orig.LiquidChunks.Count != rt.LiquidChunks.Count)
            msgs.Add($"liquid:{orig.LiquidChunks.Count}→{rt.LiquidChunks.Count}");

        return (msgs.Count == 0, maxH, maxA, msgs);
    }

    static (bool pass, float maxH, float maxA, List<string> msgs) CompareLk(
        LkAdtData orig, LkAdtData rt, float hEps, float aEps)
    {
        var msgs = new List<string>();
        float maxH = 0, maxA = 0;
        int hBad = 0;

        int chunkCount = Math.Min(orig.Chunks.Count, rt.Chunks.Count);
        for (int ci = 0; ci < chunkCount; ci++)
        {
            var oc = orig.Chunks[ci]; var rc = rt.Chunks[ci];
            if (oc.Heights != null && rc.Heights != null)
            {
                int len = Math.Min(oc.Heights.Length, rc.Heights.Length);
                for (int i = 0; i < len; i++)
                {
                    float d = MathF.Abs((oc.Heights[i] + oc.BaseHeight) - (rc.Heights[i] + rc.BaseHeight));
                    if (d > maxH) maxH = d;
                    if (d > hEps) hBad++;
                }
            }
        }
        if (hBad > 0) msgs.Add($"height:{hBad} exceed ε (max={maxH:F3})");

        if (orig.TextureNames.Count != rt.TextureNames.Count)
            msgs.Add($"tex:{orig.TextureNames.Count}→{rt.TextureNames.Count}");

        return (msgs.Count == 0, maxH, maxA, msgs);
    }

    static string? GetOpt(string[] a, string l, string s)
    {
        for (int i = 0; i < a.Length - 1; i++)
            if (a[i].Equals(l, StringComparison.OrdinalIgnoreCase) || a[i].Equals(s, StringComparison.OrdinalIgnoreCase))
                return a[i + 1];
        return null;
    }
}
