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
                // LK → Alpha: use the PROVEN reader from LkAdtReader
                var lkRead = LkAdtReader.Read(lkBytes, null, null, tx, ty);
                var rt = LkToAlphaConverter.ConvertTile(lkRead, tx, ty);

                if (verbose)
                {
                    // Byte-level probe: does the file's MCAL region match what the writer meant?
                    // Divergence here indicts LkAdtWriter/LkAdtReader, not the pack math.
                    for (int probe = 0; probe < 4; probe++)
                    {
                        var modelChunk = lk.Chunks[probe];
                        // LK ADT: MVER(12) + MHDR header(8) + payload(64) + MCIN header(8) = 92.
                        int mcinEntry = BitConverter.ToInt32(lkBytes, 12 + 8 + 64 + 8 + (probe * 16));
                        int hdr = mcinEntry + 8;
                        int ofsMcal = BitConverter.ToInt32(lkBytes, hdr + 0x24);
                        int sizeMcal = BitConverter.ToInt32(lkBytes, hdr + 0x28);
                        int ofsMcsh = BitConverter.ToInt32(lkBytes, hdr + 0x2C);
                        int sizeMcsh = BitConverter.ToInt32(lkBytes, hdr + 0x30);
                        string mcSig = ofsMcal >= 8
                            ? new string(new[] { (char)lkBytes[hdr + ofsMcal - 8 + 3], (char)lkBytes[hdr + ofsMcal - 8 + 2], (char)lkBytes[hdr + ofsMcal - 8 + 1], (char)lkBytes[hdr + ofsMcal - 8] })
                            : "n/a";
                        Console.WriteLine($"    [{tx},{ty}] model chunk {probe}: layers={modelChunk.NLayers} alphaBytes={modelChunk.AlphaMapSize} fileOfsMcal={ofsMcal} fileSizeMcal={sizeMcal} sigAtOfs='{mcSig}' ofsMcsh={ofsMcsh} sizeMcsh={sizeMcsh}");
                    }
                }

                if (lkRead.Chunks.Count == 0)
                {
                    fail++;
                    continue;
                }

                var (pass, maxH, maxA, msgs) = CompareAlpha(orig, rt, hEps, aEps, verbose, tx, ty);
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
                    var origLk = LkAdtReader.Read(adtBytes, null, null, tx, ty);
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
        AlphaTileData orig, AlphaTileData rt, float hEps, float aEps, bool verbose = false, int tx = 0, int ty = 0)
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
            // AlphaTileData.McalAlphaPack is the reader's 4x box-downsampled 256 signal while the
            // return leg decodes full-resolution 1024. Comparing them index-for-index compared
            // orig256[y,x] against upsample(orig256)[y,x] = orig256[y/4,x/4] and reported hard
            // alpha edges as 1.000 flips. Compare at the ORIGINAL pack's resolution by box-
            // downsampling the round-trip pack to match. (Known conversion-quality debt: the
            // Alpha->LK leg itself consumes the lossy 256 pack; see Spec 221 evidence.)
            float[,,] rtPack = rt.McalAlphaPack;
            if (rtPack.GetLength(0) != orig.McalAlphaPack.GetLength(0)
                || rtPack.GetLength(1) != orig.McalAlphaPack.GetLength(1))
            {
                int oH = orig.McalAlphaPack.GetLength(0), oW = orig.McalAlphaPack.GetLength(1);
                int oL = Math.Min(orig.McalAlphaPack.GetLength(2), rtPack.GetLength(2));
                int ratio = rtPack.GetLength(0) / oH;
                var downsampled = new float[oH, oW, oL];
                for (int y = 0; y < oH; y++)
                    for (int x = 0; x < oW; x++)
                        for (int l = 0; l < oL; l++)
                        {
                            float sum = 0f;
                            for (int dy = 0; dy < ratio; dy++)
                                for (int dx = 0; dx < ratio; dx++)
                                    sum += rtPack[y * ratio + dy, x * ratio + dx, l];
                            downsampled[y, x, l] = sum / (ratio * ratio);
                        }
                rtPack = downsampled;
            }

            float origVal = 0, rtVal = 0;
            int badX = -1, badY = -1, badL = -1;
            int aBad = 0;
            for (int y = 0; y < orig.McalAlphaPack.GetLength(0); y++)
                for (int x = 0; x < orig.McalAlphaPack.GetLength(1); x++)
                    for (int l = 0; l < Math.Min(orig.McalAlphaPack.GetLength(2), rtPack.GetLength(2)); l++)
                    {
                        float d = MathF.Abs(orig.McalAlphaPack[y, x, l] - rtPack[y, x, l]);
                        if (d > maxA) maxA = d;
                        if (d > aEps)
                        {
                            if (badX == -1)
                            {
                                origVal = orig.McalAlphaPack[y, x, l];
                                rtVal = rtPack[y, x, l];
                                badX = x; badY = y; badL = l;
                            }
                            aBad++;
                        }
                    }
            if (aBad > 0)
                msgs.Add($"alpha:{aBad} exceed ε (max={maxA:F3}) [orig={origVal:F3} rt={rtVal:F3} at {badX},{badY} l={badL}]");

            if (verbose)
            {
                // Per-chunk/per-layer drift: localizes whether the fault is chunk-wide (index
                // confusion), row/column bands (span or nibble order), or scattered (decode).
                Console.WriteLine($"    [{tx},{ty}] pack dims orig=({orig.McalAlphaPack!.GetLength(0)},{orig.McalAlphaPack.GetLength(1)},{orig.McalAlphaPack.GetLength(2)}) rt=({rt.McalAlphaPack!.GetLength(0)},{rt.McalAlphaPack.GetLength(1)},{rt.McalAlphaPack.GetLength(2)})");
                int layerCount = Math.Min(orig.McalAlphaPack.GetLength(2), rt.McalAlphaPack.GetLength(2));
                int rows = Math.Min(orig.McalAlphaPack.GetLength(0), rt.McalAlphaPack.GetLength(0));
                int cols = Math.Min(orig.McalAlphaPack.GetLength(1), rt.McalAlphaPack.GetLength(1));
                for (int cy = 0; cy < 16 && cy * 64 < rows; cy++)
                    for (int cx = 0; cx < 16 && cx * 64 < cols; cx++)
                    {
                        float chunkMax = 0f;
                        int chunkBadLayer = -1;
                        for (int l = 1; l < layerCount; l++)
                        {
                            float layerMax = 0f;
                            for (int yy = 0; yy < 64 && cy * 64 + yy < rows; yy++)
                                for (int xx = 0; xx < 64 && cx * 64 + xx < cols; xx++)
                                {
                                    float d = MathF.Abs(orig.McalAlphaPack[cy * 64 + yy, cx * 64 + xx, l]
                                                        - rt.McalAlphaPack[cy * 64 + yy, cx * 64 + xx, l]);
                                    if (d > layerMax) layerMax = d;
                                }
                            if (layerMax > chunkMax) { chunkMax = layerMax; chunkBadLayer = l; }
                        }
                        if (chunkMax > aEps)
                            Console.WriteLine($"    [{tx},{ty}] chunk(c={cx},r={cy}) maxΔα={chunkMax:F3} (worst l={chunkBadLayer})");
                    }
            }
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
