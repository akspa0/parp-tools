using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Security.Cryptography;
using WoWRollback.LkToAlphaModule;
using WoWRollback.Core.Services.Assets;
using WoWRollback.LkToAlphaModule.Readers;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using WoWRollback.LkToAlphaModule.Builders;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.LkToAlphaModule.Writers;

public sealed class AlphaWdtMonolithicWriter
{
    private const int GridTiles = 64 * 64; // 4096
    private const long OutSizeLimit = (long)int.MaxValue - (1L << 16);

    public static void WriteMonolithic(string lkWdtPath, string lkMapDir, string outWdtPath, LkToAlphaOptions? opts = null, bool skipWmos = false)
    {
        bool verbose = opts?.Verbose == true;
        if (string.IsNullOrWhiteSpace(lkWdtPath)) throw new ArgumentException("lkWdtPath required");
        if (string.IsNullOrWhiteSpace(lkMapDir)) throw new ArgumentException("lkMapDir required");
        if (string.IsNullOrWhiteSpace(outWdtPath)) throw new ArgumentException("outWdtPath required");
        Directory.CreateDirectory(Path.GetDirectoryName(outWdtPath) ?? ".");

        // Derive map name from WDT filename
        string mapName = Path.GetFileNameWithoutExtension(lkWdtPath);

        // Discover existing root ADTs under lkMapDir
        var rootAdts = Directory.EnumerateFiles(lkMapDir, mapName + "_*.adt", SearchOption.TopDirectoryOnly)
            .Where(p => !p.Contains("_obj", StringComparison.OrdinalIgnoreCase) && !p.Contains("_tex", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();
        Console.WriteLine($"[pack] root ADTs: {rootAdts.Count}");
        if (verbose && rootAdts.Count > 0)
        {
            Console.WriteLine($"[pack] first: {Path.GetFileName(rootAdts.First())}, last: {Path.GetFileName(rootAdts.Last())}");
        }


        // Collect WMO and M2 names from WDT and all tile ADTs
        var wdtReader = new LkWdtReader();
        var adtReader = new LkAdtReader();
        var allWmoNames = new HashSet<string>();
        var allM2Names = new HashSet<string>();
        
        // Read from top-level WDT first
        var wdtWmos = wdtReader.ReadWmoNames(lkWdtPath);
        foreach (var name in wdtWmos)
        {
            allWmoNames.Add(name);
        }
        
        foreach (var rootAdt in rootAdts)
        {
            try
            {
                var bytesScan = File.ReadAllBytes(rootAdt);
                foreach (var n in ReadM2NamesFromBytes(bytesScan)) allM2Names.Add(n);
                foreach (var n in ReadWmoNamesFromBytes(bytesScan)) allWmoNames.Add(n);
                var baseNameScan = Path.GetFileNameWithoutExtension(rootAdt);
                var dirScan = Path.GetDirectoryName(rootAdt) ?? ".";
                var objScan = Path.Combine(dirScan, baseNameScan + "_obj.adt");
                if (File.Exists(objScan))
                {
                    var objBytesScan = File.ReadAllBytes(objScan);
                    foreach (var n in ReadM2NamesFromBytes(objBytesScan)) allM2Names.Add(n);
                    foreach (var n in ReadWmoNamesFromBytes(objBytesScan)) allWmoNames.Add(n);
                }
            }
            catch { /* best-effort name harvest */ }
        }
        
        int mccvExported = 0;
        int mccvHeaders = 0;

        // Build WDT scaffolding (Alpha format): MVER -> MPHD(16) -> MAIN -> MDNM -> MONM
        using var ms = new MemoryStream();
        // MVER
        ms.Write(new Chunk("MVER", 4, BitConverter.GetBytes(18)).GetWholeChunk());
        // MPHD (128 bytes) placeholder; will patch absolute offsets and counts later
        long mphdStart = ms.Position; var mphd = new Chunk("MPHD", 128, new byte[128]); ms.Write(mphd.GetWholeChunk()); long mphdDataStart = mphdStart + 8;
        // MAIN placeholder (4096 * 16)
        var mainData = new byte[GridTiles * 16]; var main = new Chunk("MAIN", mainData.Length, mainData); long mainStart = ms.Position; ms.Write(main.GetWholeChunk());
        // Build name lists (optional); currently use collected sets (may be empty)
        var wmoNames = allWmoNames.ToList();
        var m2Names = allM2Names.ToList();
        if (!string.IsNullOrWhiteSpace(opts?.TargetListfilePath) && File.Exists(opts.TargetListfilePath))
        {
            try
            {
                var idx = ListfileIndex.Load(opts.TargetListfilePath!);
                var gate = new AssetGate(idx);
                var keptM2 = gate.FilterNames(m2Names, out var droppedM2);
                var keptWmo = gate.FilterNames(wmoNames, out var droppedWmo);
                if (opts!.StrictTargetAssets)
                {
                    m2Names = keptM2.ToList();
                    wmoNames = keptWmo.ToList();
                    var dropCsv = Path.Combine(Path.GetDirectoryName(outWdtPath) ?? ".", "dropped_assets.csv");
                    AssetGate.WriteDropReport(dropCsv, droppedM2, droppedWmo);
                }
            }
            catch { }
        }
        // Global name indices for MDNM/MONM
        var mdnmIndexFs = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < m2Names.Count; i++) mdnmIndexFs[NormalizeAssetName(m2Names[i])] = i;
        var monmIndexFs = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < wmoNames.Count; i++) monmIndexFs[NormalizeAssetName(wmoNames[i])] = i;
        // MDNM/MONM chunks
        long mdnmStart = ms.Position; var mdnm = new Chunk("MDNM", BuildMdnmData(m2Names).Length, BuildMdnmData(m2Names)); ms.Write(mdnm.GetWholeChunk());
        long monmStart = ms.Position; var monm = new Chunk("MONM", BuildMonmData(wmoNames).Length, BuildMonmData(wmoNames)); ms.Write(monm.GetWholeChunk());
        // Patch MPHD with absolute offsets and counts
        long savePos = ms.Position; ms.Position = mphdDataStart; Span<byte> mphdData = stackalloc byte[128]; mphdData.Clear();
        // MPHD layout: [0..3]=nTextures (M2), [4..7]=MDNM abs, [8..11]=nMapObjNames (WMO, +1 when any), [12..15]=MONM abs
        BitConverter.GetBytes(m2Names.Count > 0 ? m2Names.Count + 1 : 0).CopyTo(mphdData);
        BitConverter.GetBytes(checked((int)mdnmStart)).CopyTo(mphdData.Slice(4));
        BitConverter.GetBytes(wmoNames.Count > 0 ? wmoNames.Count + 1 : 0).CopyTo(mphdData.Slice(8));
        BitConverter.GetBytes(checked((int)monmStart)).CopyTo(mphdData.Slice(12));
        ms.Write(mphdData); ms.Position = savePos;

        // MAIN patch arrays
        var mhdrAbs = Enumerable.Repeat(0, GridTiles).ToArray();
        var mhdrToFirstMcnkSizes = Enumerable.Repeat(0, GridTiles).ToArray();
        // Provide alias for compatibility with downstream code segments
        var outMs = ms;
        bool limitLogged = false;
        // RAW EMBED MODE disabled in monolithic packer (use conversion path below)

        // Read from each tile ADT (root and obj variants)
        foreach (var rootAdt in rootAdts)
        {
            int tileIndex = -1;
            long tileStart = 0;
            if (outMs.Position >= OutSizeLimit)
            {
                if (!limitLogged) { Console.WriteLine("[pack][limit] WDT near 2GiB; stopping further tiles."); limitLogged = true; }
                break;
            }
            try
            {
                // Parse tile indices from file name map_yy_xx.adt
                var file = Path.GetFileNameWithoutExtension(rootAdt);
                // Expected: <map>_YY_XX
                var parts = file.Split('_');
                if (parts.Length < 3) continue;
                if (!int.TryParse(parts[^2], out int yy)) continue;
                if (!int.TryParse(parts[^1], out int xx)) continue;
                // Alpha MAIN grid stores tiles with X-major ordering (xx as rows).
                tileIndex = xx * 64 + yy;

                tileStart = outMs.Position;

                var bytes = File.ReadAllBytes(rootAdt);
            // Locate LK MHDR → MCIN to get MCNK offsets to know which exist
            int mhdrOffset = FindFourCC(bytes, "MHDR");
            if (mhdrOffset < 0) continue;
            var lkMhdr = new Mhdr(bytes, mhdrOffset);
            int lkMhdrDataStart = mhdrOffset + 8;
            int lkMcinOff = lkMhdr.GetOffset(Mhdr.McinOffset);
            if (lkMcinOff == 0) continue;
            var lkMcin = new Mcin(bytes, lkMhdrDataStart + lkMcinOff);
            var lkMcnkOffsets = lkMcin.GetMcnkOffsets(); // absolute LK file offsets or 0

            // Optional: read _tex.adt for this tile and get its MCNK offsets as well
            byte[]? texBytesForChunks = null;
            System.Collections.Generic.List<int>? texMcnkOffsets = null;
            try
            {
                var rootDirLocal = Path.GetDirectoryName(rootAdt)!;
                var baseNameLocal = Path.GetFileNameWithoutExtension(rootAdt);
                var texAdtLocal = Path.Combine(rootDirLocal, baseNameLocal + "_tex.adt");
                if (File.Exists(texAdtLocal))
                {
                    texBytesForChunks = File.ReadAllBytes(texAdtLocal);
                    int tMhOff = FindFourCC(texBytesForChunks, "MHDR");
                    if (tMhOff >= 0)
                    {
                        var tMhdr = new Mhdr(texBytesForChunks, tMhOff);
                        int tData = tMhOff + 8;
                        int tMcinRel = tMhdr.GetOffset(Mhdr.McinOffset);
                        if (tMcinRel > 0)
                        {
                            var tMcin = new Mcin(texBytesForChunks, tData + tMcinRel);
                            texMcnkOffsets = tMcin.GetMcnkOffsets();
                        }
                    }
                }
            }
            catch { /* optional */ }

            // Prebuild Alpha MCNK bytes for present entries
            var alphaMcnkBytes = new byte[256][];
            var presentIndices = new List<int>(256);
            for (int i = 0; i < 256; i++)
            {
                int off = (i < lkMcnkOffsets.Count) ? lkMcnkOffsets[i] : 0;
                if (off > 0)
                {
                    int texOff = (texMcnkOffsets != null && i < texMcnkOffsets.Count) ? texMcnkOffsets[i] : 0;
                    // Root-first: AlphaMcnkBuilder should prefer root MCLY/MCAL and only fallback to _tex if root is missing those per-chunk
                    alphaMcnkBytes[i] = AlphaMcnkBuilder.BuildFromLk(bytes, off, opts, texBytesForChunks, texOff);
                    presentIndices.Add(i);
                    if (off + 8 + 0x80 <= bytes.Length)
                    {
                        int hdrStart = off + 8;
                        int mRel = BitConverter.ToInt32(bytes, hdrStart + 0x64);
                        if (mRel > 0) mccvHeaders++;
                    }
                    if (!string.IsNullOrWhiteSpace(opts?.ExportMccvDir))
                    {
                        try { if (ExportMccvIfPresent(bytes, off, Path.Combine(opts!.ExportMccvDir!, mapName), yy, xx, i)) mccvExported++; } catch { }
                    }
                }
                else alphaMcnkBytes[i] = Array.Empty<byte>();
            }


            tileIndex = xx * 64 + yy;
            long mhdrAbsolute = outMs.Position;
            var aMhdr = AlphaMhdrBuilder.BuildMhdrForTerrain();
            outMs.Write(aMhdr.GetWholeChunk());

            // Emit MCIN placeholder; we'll rewrite with absolute offsets and sizes after computing MCNK layout
            long mcinPosition = outMs.Position; var mcinPh = AlphaMcinBuilder.BuildMcin(new int[256], new int[256]); var mcinWhole = mcinPh.GetWholeChunk(); outMs.Write(mcinWhole);

            // MTEX (prefer tile _tex.adt, fallback to root ADT MTEX, then BaseTexture)
            byte[] mtexData = Array.Empty<byte>();
            var rootDir = Path.GetDirectoryName(rootAdt)!;
            var baseName = Path.GetFileNameWithoutExtension(rootAdt);
            var texAdt = Path.Combine(rootDir, baseName + "_tex.adt");
            if (File.Exists(texAdt))
            {
                var texBytes = File.ReadAllBytes(texAdt);
                mtexData = ExtractLkMtexData(texBytes);
            }
            if (mtexData.Length == 0) mtexData = ExtractLkMtexData(bytes);
            if (mtexData.Length == 0)
            {
                var baseTexturePath = string.IsNullOrWhiteSpace(opts?.BaseTexture) ? "Tileset\\Generic\\Checkers.blp" : opts!.BaseTexture!;
                mtexData = Encoding.ASCII.GetBytes(baseTexturePath + "\0");
            }
            long mtexPosition = outMs.Position; outMs.Write(new Chunk("MTEX", mtexData.Length, mtexData).GetWholeChunk()); long mtexEnd = outMs.Position;

            // Build MDDF/MODF from _obj.adt if present (fallback: root)
            byte[] objBytesFs = Array.Empty<byte>();
            var objAdt = Path.Combine(rootDir, baseName + "_obj.adt");
            if (File.Exists(objAdt)) objBytesFs = File.ReadAllBytes(objAdt);
            if (objBytesFs.Length == 0) objBytesFs = bytes;

            var mmdxOrderedFs = BuildMmdxOrdered(objBytesFs);
            if (mmdxOrderedFs.Count == 0) mmdxOrderedFs = BuildMmdxOrdered(bytes);
            var mwmoOrderedFs = BuildMwmoOrdered(objBytesFs);
            if (mwmoOrderedFs.Count == 0) mwmoOrderedFs = BuildMwmoOrdered(bytes);

            var mddfDataFs = BuildMddfFromLk(objBytesFs, mmdxOrderedFs, mdnmIndexFs);
            var modfDataFs = BuildModfFromLk(objBytesFs, mwmoOrderedFs, monmIndexFs);
            var (doodadRefsByChunkFs, wmoRefsByChunkFs) = BuildRefsByChunk(objBytesFs, mmdxOrderedFs, mwmoOrderedFs, mdnmIndexFs, monmIndexFs, yy, xx);

            long mddfPosition = outMs.Position; outMs.Write(new Chunk("MDDF", mddfDataFs.Length, mddfDataFs).GetWholeChunk()); long mddfEnd = outMs.Position;
            long modfPosition = outMs.Position; outMs.Write(new Chunk("MODF", modfDataFs.Length, modfDataFs).GetWholeChunk()); long modfEnd = outMs.Position;

            // Rebuild MCNKs with MCRF refs
            alphaMcnkBytes = new byte[256][];
            for (int ci = 0; ci < 256; ci++)
            {
                int off = (ci < lkMcnkOffsets.Count) ? lkMcnkOffsets[ci] : 0;
                if (off > 0)
                {
                    int texOff = (texMcnkOffsets != null && ci < texMcnkOffsets.Count) ? texMcnkOffsets[ci] : 0;
                    var drefs = doodadRefsByChunkFs[ci];
                    var wrefs = wmoRefsByChunkFs[ci];
                    alphaMcnkBytes[ci] = AlphaMcnkBuilder.BuildFromLk(bytes, off, opts, texBytesForChunks, texOff, drefs, wrefs);
                }
                else alphaMcnkBytes[ci] = Array.Empty<byte>();
            }

            long firstMcnkAbsolute = outMs.Position;
            if (firstMcnkAbsolute > OutSizeLimit) throw new InvalidOperationException("2GiB limit");
            long predictedEnd = firstMcnkAbsolute;
            for (int i = 0; i < 256; i++) { var buf = alphaMcnkBytes[i]; if (buf is { Length: > 0 }) predictedEnd += buf.Length; }
            if (predictedEnd > OutSizeLimit) throw new InvalidOperationException($"tile {yy:D2}_{xx:D2} would exceed 2GiB");
            int[] mcnkAbs = new int[256]; int[] mcnkSizes = new int[256]; long cursor = firstMcnkAbsolute;
            for (int i = 0; i < 256; i++)
            {
                var buf = alphaMcnkBytes[i];
                if (buf is { Length: > 0 }) { mcnkAbs[i] = checked((int)cursor); mcnkSizes[i] = buf.Length; cursor += buf.Length; }
                else { mcnkAbs[i] = 0; mcnkSizes[i] = 0; }
            }

            // Patch MHDR.data with relative offsets
            long mhdrDataStart = mhdrAbsolute + 8;
            // offsInfo (0x00) points to MCIN chunk; we place MCIN immediately after MHDR.data => 64
            outMs.Position = mhdrDataStart + 0; outMs.Write(BitConverter.GetBytes(64));
            outMs.Position = mhdrDataStart + 4; outMs.Write(BitConverter.GetBytes(checked((int)(mtexPosition - mhdrDataStart))));
            outMs.Position = mhdrDataStart + 8; outMs.Write(BitConverter.GetBytes(mtexData.Length));
            outMs.Position = mhdrDataStart + 12; outMs.Write(BitConverter.GetBytes(checked((int)(mddfPosition - mhdrDataStart))));
            outMs.Position = mhdrDataStart + 16; outMs.Write(BitConverter.GetBytes(mddfDataFs.Length));
            outMs.Position = mhdrDataStart + 20; outMs.Write(BitConverter.GetBytes(checked((int)(modfPosition - mhdrDataStart))));
            outMs.Position = mhdrDataStart + 24; outMs.Write(BitConverter.GetBytes(modfDataFs.Length));
            // Rewrite MCIN with ABSOLUTE positions (letters positions) and full chunk lengths
            outMs.Position = mcinPosition; var mcinReal = AlphaMcinBuilder.BuildMcin(mcnkAbs, mcnkSizes); outMs.Write(mcinReal.GetWholeChunk());
            // Move to the start of the MCNK region and write MCNKs in place
            outMs.Position = firstMcnkAbsolute;

            int mhdrToFirstVal = (alphaMcnkBytes.Any(b => b != null && b.Length > 0)) ? checked((int)(firstMcnkAbsolute - mhdrAbsolute)) : 0;

            // Write MCNKs
            for (int i = 0; i < 256; i++) { var buf = alphaMcnkBytes[i]; if (buf is { Length: > 0 }) outMs.Write(buf); }
            // Sanity: verify token at first present MCNK absolute position and at client-computed (mhdrAbsolute + rel) position
            try
            {
                int firstIdxPresent = -1;
                for (int i = 0; i < 256; i++) { if (alphaMcnkBytes[i] is { Length: > 0 }) { firstIdxPresent = i; break; } }
                if (firstIdxPresent >= 0)
                {
                    long absPos = mcnkAbs[firstIdxPresent];
                    long cur = outMs.Position;
                    outMs.Position = absPos;
                    Span<byte> sig = stackalloc byte[4];
                    int read = outMs.Read(sig);
                    outMs.Position = cur;
                    var tokenStr = Encoding.ASCII.GetString(sig);
                    Console.WriteLine($"[pack][check] tile {yy:D2}_{xx:D2} MCNK[{firstIdxPresent}] abs={absPos} len={outMs.Length} read={read} token='{tokenStr}'");
                    // Validate all MCIN entries for this tile (absolute)
                    for (int i = 0; i < 256; i++)
                    {
                        if (mcnkAbs[i] > 0)
                        {
                            long pos = mcnkAbs[i];
                            long cur2 = outMs.Position;
                            outMs.Position = pos;
                            Span<byte> t = stackalloc byte[4];
                            int r = outMs.Read(t);
                            outMs.Position = cur2;
                            var tok = Encoding.ASCII.GetString(t);
                            if (r != 4 || tok != "MCNK")
                            {
                                Console.WriteLine($"[pack][BAD] tile {yy:D2}_{xx:D2} MCIN[{i}] abs={pos} read={r} token='{tok}'");
                                break;
                            }
                        }
                    }
                }
            }
            catch { }
            // Commit MAIN entries after successful tile write
            mhdrAbs[tileIndex] = checked((int)mhdrAbsolute);
            mhdrToFirstMcnkSizes[tileIndex] = mhdrToFirstVal;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[pack] tile failed {Path.GetFileName(rootAdt)}: {ex.Message}");
            try
            {
                if (tileIndex >= 0)
                {
                    outMs.SetLength(tileStart);
                    outMs.Position = tileStart;
                    mhdrAbs[tileIndex] = 0;
                    mhdrToFirstMcnkSizes[tileIndex] = 0;
                }
            }
            catch { /* best-effort rollback */ }
            continue;
        }
        }

        // Patch MAIN data
        outMs.Position = mainStart + 8;
        var patchedMain = AlphaMainBuilder.BuildMain(mhdrAbs, mhdrToFirstMcnkSizes, false);
        outMs.Write(patchedMain.Data, 0, patchedMain.Data.Length);
        using (var fs = File.Create(outWdtPath))
        {
            outMs.Position = 0;
            outMs.WriteTo(fs);
        }
    if (!string.IsNullOrWhiteSpace(opts?.ExportMccvDir))
    {
        Console.WriteLine($"[pack] MCCV images exported: {mccvExported}");
        Console.WriteLine($"[pack] MCCV headers present: {mccvHeaders}");
    }

    }

    public static void WriteMonolithicFromArchive(IArchiveSource src, string mapName, string outWdtPath, LkToAlphaOptions? opts = null)
    {
        if (src is null) throw new ArgumentNullException(nameof(src));
        if (string.IsNullOrWhiteSpace(mapName)) throw new ArgumentException("mapName required");
        if (string.IsNullOrWhiteSpace(outWdtPath)) throw new ArgumentException("outWdtPath required");
        Directory.CreateDirectory(Path.GetDirectoryName(outWdtPath) ?? ".");

        bool verbose = opts?.Verbose == true;

        // Discover tiles from archive
        var tiles = new List<(int YY, int XX, string RootVPath)>();
        var pattern = $"world/maps/{mapName}/{mapName}_*_*.adt";
        var allCandidates = src.EnumerateFiles(pattern)
            .Where(p => !p.Contains("_obj", StringComparison.OrdinalIgnoreCase) && !p.Contains("_tex", StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (allCandidates.Count > 0)
        {
            foreach (var vp in allCandidates)
            {
                var stem = Path.GetFileNameWithoutExtension(vp);
                var body = stem.Substring(mapName.Length + 1);
                var parts = body.Split('_');
                if (parts.Length >= 2 && int.TryParse(parts[0], out var yy) && int.TryParse(parts[1], out var xx))
                {
                    tiles.Add((yy, xx, vp));
                }
            }
        }
        else
        {
            for (int yy = 0; yy < 64; yy++)
            {
                for (int xx = 0; xx < 64; xx++)
                {
                    var vp = $"world/maps/{mapName}/{mapName}_{yy}_{xx}.adt";
                    if (src.FileExists(vp)) tiles.Add((yy, xx, vp));
                }
            }
        }
        if (verbose) Console.WriteLine($"[pack] archive tiles detected: {tiles.Count}");

        // Build name tables by scanning tiles
        var m2Names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var wmoNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (yy, xx, rootVPath) in tiles)
        {
            try
            {
                using (var s = src.OpenFile(rootVPath)) using (var msScan = new MemoryStream())
                {
                    s.CopyTo(msScan);
                    var bytes = msScan.ToArray();
                    foreach (var n in ReadM2NamesFromBytes(bytes)) m2Names.Add(n);
                    foreach (var n in ReadWmoNamesFromBytes(bytes)) wmoNames.Add(n);
                }
                var objVPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}_obj.adt";
                if (src.FileExists(objVPath))
                {
                    using var so = src.OpenFile(objVPath);
                    using var mso = new MemoryStream();
                    so.CopyTo(mso);
                    var objBytes = mso.ToArray();
                    foreach (var n in ReadM2NamesFromBytes(objBytes)) m2Names.Add(n);
                    foreach (var n in ReadWmoNamesFromBytes(objBytes)) wmoNames.Add(n);
                }
            }
            catch { /* best-effort */ }
        }

        var m2List = m2Names.ToList();
        var wmoList = wmoNames.ToList();

        // Optional asset gating to target listfile
        if (!string.IsNullOrWhiteSpace(opts?.TargetListfilePath) && File.Exists(opts!.TargetListfilePath))
        {
            try
            {
                var idx = ListfileIndex.Load(opts.TargetListfilePath!);
                var gate = new AssetGate(idx);
                var keptM2 = gate.FilterNames(m2List, out var droppedM2);
                var keptWmo = gate.FilterNames(wmoList, out var droppedWmo);
                if (opts.StrictTargetAssets)
                {
                    m2List = keptM2.ToList();
                    wmoList = keptWmo.ToList();
                    var dropCsv = Path.Combine(Path.GetDirectoryName(outWdtPath) ?? ".", "dropped_assets.csv");
                    AssetGate.WriteDropReport(dropCsv, droppedM2, droppedWmo);
                }
            }
            catch (Exception ex) { Console.WriteLine($"[warn] Asset gating failed: {ex.Message}"); }
        }
        // Build global name indices (normalized)
        var mdnmIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < m2List.Count; i++) mdnmIndex[NormalizeAssetName(m2List[i])] = i;
        var monmIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < wmoList.Count; i++) monmIndex[NormalizeAssetName(wmoList[i])] = i;

        // Build WDT scaffolding
        using var outMs = new MemoryStream();
        outMs.Write(new Chunk("MVER", 4, BitConverter.GetBytes(18)).GetWholeChunk());
        long mphdStart = outMs.Position; var mphd = new Chunk("MPHD", 128, new byte[128]); outMs.Write(mphd.GetWholeChunk()); long mphdDataStart = mphdStart + 8;
        var mainData = new byte[GridTiles * 16]; var main = new Chunk("MAIN", mainData.Length, mainData); long mainStart = outMs.Position; outMs.Write(main.GetWholeChunk());
        long mdnmStart = outMs.Position; var mdnm = new Chunk("MDNM", BuildMdnmData(m2List).Length, BuildMdnmData(m2List)); outMs.Write(mdnm.GetWholeChunk());
        long monmStart = outMs.Position; var monm = new Chunk("MONM", BuildMonmData(wmoList).Length, BuildMonmData(wmoList)); outMs.Write(monm.GetWholeChunk());
        long savePos = outMs.Position; outMs.Position = mphdDataStart; Span<byte> mphdData = stackalloc byte[128]; mphdData.Clear();
        BitConverter.GetBytes(m2List.Count > 0 ? m2List.Count + 1 : 0).CopyTo(mphdData);
        BitConverter.GetBytes(checked((int)mdnmStart)).CopyTo(mphdData.Slice(4));
        // MONM count must include a trailing empty string when any names exist
        BitConverter.GetBytes(wmoList.Count > 0 ? wmoList.Count + 1 : 0).CopyTo(mphdData.Slice(8));
        BitConverter.GetBytes(checked((int)monmStart)).CopyTo(mphdData.Slice(12));
        outMs.Write(mphdData); outMs.Position = savePos;

        var mhdrAbs = Enumerable.Repeat(0, GridTiles).ToArray();
        var mhdrToFirst = Enumerable.Repeat(0, GridTiles).ToArray();
        int mccvExported2 = 0;
        bool limitLogged2 = false;

        // Build tiles from archive
        foreach (var (yy, xx, rootVPath) in tiles)
        {
            // Prepare per-tile bookkeeping
            int tileIndex = xx * 64 + yy;
            long tileStart = outMs.Position;
            if (outMs.Position >= OutSizeLimit)
            {
                if (!limitLogged2) { Console.WriteLine("[pack][limit] WDT near 2GiB; stopping further tiles."); limitLogged2 = true; }
                break;
            }
            try
            {
                using var s = src.OpenFile(rootVPath);
                using var msTile = new MemoryStream(); s.CopyTo(msTile); var bytes = msTile.ToArray();

                int mhdrOffset = FindFourCC(bytes, "MHDR"); if (mhdrOffset < 0) continue;
                var lkMhdr = new Mhdr(bytes, mhdrOffset);
                int lkMhdrDataStart = mhdrOffset + 8;
                int lkMcinOff = lkMhdr.GetOffset(Mhdr.McinOffset); if (lkMcinOff == 0) continue;
                var lkMcin = new Mcin(bytes, lkMhdrDataStart + lkMcinOff);
                var lkMcnkOffsets = lkMcin.GetMcnkOffsets();

                var alphaMcnkBytes = new byte[256][];
                var present = new List<int>(256);
                for (int i = 0; i < 256; i++)
                {
                    int off = (i < lkMcnkOffsets.Count) ? lkMcnkOffsets[i] : 0;
                    if (off > 0)
                    {
                        alphaMcnkBytes[i] = AlphaMcnkBuilder.BuildFromLk(bytes, off, opts);
                        present.Add(i);
                        if (!string.IsNullOrWhiteSpace(opts?.ExportMccvDir))
                        {
                            try { if (ExportMccvIfPresent(bytes, off, Path.Combine(opts!.ExportMccvDir!, mapName), yy, xx, i)) mccvExported2++; } catch { }
                        }
                    }
                    else alphaMcnkBytes[i] = Array.Empty<byte>();
                }

                long mhdrAbsolute = outMs.Position; // Defer assigning mhdrAbs until success
                var aMhdr = AlphaMhdrBuilder.BuildMhdrForTerrain();
                outMs.Write(aMhdr.GetWholeChunk());

                long mcinPosition = outMs.Position; var mcinPh = AlphaMcinBuilder.BuildMcin(new int[256], new int[256]); var mcinWhole = mcinPh.GetWholeChunk(); outMs.Write(mcinWhole);

                // MTEX from _tex.adt, fallback to root bytes, else BaseTexture
                byte[] mtexData = Array.Empty<byte>();
                var texVPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}_tex.adt";
                if (src.FileExists(texVPath))
                {
                    using var ts = src.OpenFile(texVPath); using var tms = new MemoryStream(); ts.CopyTo(tms); mtexData = ExtractLkMtexData(tms.ToArray());
                }
                if (mtexData.Length == 0) mtexData = ExtractLkMtexData(bytes);
                if (mtexData.Length == 0)
                {
                    var baseTexturePath = string.IsNullOrWhiteSpace(opts?.BaseTexture) ? "Tileset\\Generic\\Checkers.blp" : opts!.BaseTexture!;
                    mtexData = Encoding.ASCII.GetBytes(baseTexturePath + "\0");
                }
                long mtexPosition = outMs.Position; outMs.Write(new Chunk("MTEX", mtexData.Length, mtexData).GetWholeChunk());

                // Build MDDF/MODF from _obj.adt if present (fallback: root)
                byte[] objBytes = Array.Empty<byte>();
                var objVPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}_obj.adt";
                if (src.FileExists(objVPath)) { using var os = src.OpenFile(objVPath); using var oms = new MemoryStream(); os.CopyTo(oms); objBytes = oms.ToArray(); }
                if (objBytes.Length == 0) objBytes = bytes;

                var mmdxOrdered = BuildMmdxOrdered(objBytes);
                if (mmdxOrdered.Count == 0) mmdxOrdered = BuildMmdxOrdered(bytes);
                var mwmoOrdered = BuildMwmoOrdered(objBytes);
                if (mwmoOrdered.Count == 0) mwmoOrdered = BuildMwmoOrdered(bytes);

                var mddfData = BuildMddfFromLk(objBytes, mmdxOrdered, mdnmIndex);
                var modfData = BuildModfFromLk(objBytes, mwmoOrdered, monmIndex);
                (System.Collections.Generic.List<int>[] doodadRefsByChunk, System.Collections.Generic.List<int>[] wmoRefsByChunk) =
                    BuildRefsByChunk(objBytes, mmdxOrdered, mwmoOrdered, mdnmIndex, monmIndex, yy, xx);

                long mddfPosition = outMs.Position; outMs.Write(new Chunk("MDDF", mddfData.Length, mddfData).GetWholeChunk());
                long modfPosition = outMs.Position; outMs.Write(new Chunk("MODF", modfData.Length, modfData).GetWholeChunk());
                if (opts?.Verbose == true)
                {
                    int mddfCount = mddfData.Length / 36;
                    int modfCount = modfData.Length / 64;
                    Console.WriteLine($"[pack][objects] tile {yy:D2}_{xx:D2} mddf={mddfCount} modf={modfCount}");
                    try
                    {
                        int mddfOff2 = FindFourCC(objBytes, "MDDF");
                        if (mddfOff2 >= 0)
                        {
                            int size2 = BitConverter.ToInt32(objBytes, mddfOff2 + 4);
                            int data2 = mddfOff2 + 8;
                            const int entry2 = 36;
                            int count2 = Math.Max(0, size2 / entry2);
                            int samples = 0;
                            for (int k = 0; k < count2 && samples < 3; k++)
                            {
                                int p2 = data2 + k * entry2; if (p2 + entry2 > objBytes.Length) break;
                                int local = BitConverter.ToInt32(objBytes, p2 + 0);
                                if (local < 0 || local >= mmdxOrdered.Count) continue;
                                string n = NormalizeAssetName(mmdxOrdered[local]);
                                mdnmIndex.TryGetValue(n, out int gidx);
                                float wx = BitConverter.ToSingle(objBytes, p2 + 8);
                                float wy = BitConverter.ToSingle(objBytes, p2 + 16);
                                var (yyW, xxW) = WorldToTileFromCentered(wx, wy);
                                Console.WriteLine($"[pack][sample][mddf] {yy:D2}_{xx:D2} world=({wx:F2},{wy:F2}) -> tile={yyW:D2}_{xxW:D2} name='{n}' mdnmIdx={gidx}");
                                if (yyW != yy || xxW != xx)
                                {
                                    Console.WriteLine($"[pack][warn][tile-check] mddf worldTile {yyW:D2}_{xxW:D2} != current {yy:D2}_{xx:D2}");
                                }
                                samples++;
                            }
                        }

                        int modfOff2 = FindFourCC(objBytes, "MODF");
                        if (modfOff2 >= 0)
                        {
                            int size2 = BitConverter.ToInt32(objBytes, modfOff2 + 4);
                            int data2 = modfOff2 + 8;
                            const int entry2 = 64;
                            int count2 = Math.Max(0, size2 / entry2);
                            int samples = 0;
                            for (int k = 0; k < count2 && samples < 3; k++)
                            {
                                int p2 = data2 + k * entry2; if (p2 + entry2 > objBytes.Length) break;
                                int local = BitConverter.ToInt32(objBytes, p2 + 0);
                                if (local < 0 || local >= mwmoOrdered.Count) continue;
                                string n = NormalizeAssetName(mwmoOrdered[local]);
                                monmIndex.TryGetValue(n, out int gidx);
                                float wx = BitConverter.ToSingle(objBytes, p2 + 8);
                                float wy = BitConverter.ToSingle(objBytes, p2 + 16);
                                var (yyW, xxW) = WorldToTileFromCentered(wx, wy);
                                Console.WriteLine($"[pack][sample][modf] {yy:D2}_{xx:D2} world=({wx:F2},{wy:F2}) -> tile={yyW:D2}_{xxW:D2} name='{n}' monmIdx={gidx}");
                                if (yyW != yy || xxW != xx)
                                {
                                    Console.WriteLine($"[pack][warn][tile-check] modf worldTile {yyW:D2}_{xxW:D2} != current {yy:D2}_{xx:D2}");
                                }
                                samples++;
                            }
                        }
                    }
                    catch { }
                }

                // Rebuild MCNKs with MCRF refs
                alphaMcnkBytes = new byte[256][];
                for (int ci = 0; ci < 256; ci++)
                {
                    int off = (ci < lkMcnkOffsets.Count) ? lkMcnkOffsets[ci] : 0;
                    if (off > 0)
                    {
                        var drefs = doodadRefsByChunk[ci];
                        var wrefs = wmoRefsByChunk[ci];
                        alphaMcnkBytes[ci] = AlphaMcnkBuilder.BuildFromLk(bytes, off, opts, null, -1, drefs, wrefs);
                    }
                    else alphaMcnkBytes[ci] = Array.Empty<byte>();
                }

                long firstMcnkAbsolute = outMs.Position;
                if (firstMcnkAbsolute > OutSizeLimit) throw new InvalidOperationException("2GiB limit");
                long predictedEnd = firstMcnkAbsolute;
                for (int i = 0; i < 256; i++) { var buf = alphaMcnkBytes[i]; if (buf is { Length: > 0 }) predictedEnd += buf.Length; }
                if (predictedEnd > OutSizeLimit) throw new InvalidOperationException($"tile {yy:D2}_{xx:D2} would exceed 2GiB");
                int[] mcnkAbs = new int[256]; int[] mcnkSizes = new int[256]; long cursor = firstMcnkAbsolute;
                for (int i = 0; i < 256; i++)
                {
                    var buf = alphaMcnkBytes[i];
                    if (buf is { Length: > 0 }) { mcnkAbs[i] = checked((int)cursor); mcnkSizes[i] = buf.Length; cursor += buf.Length; }
                    else { mcnkAbs[i] = 0; mcnkSizes[i] = 0; }
                }

                // Patch MHDR.data with relative offsets
                long mhdrDataStartRel = mhdrAbsolute + 8;
                outMs.Position = mhdrDataStartRel + 0; outMs.Write(BitConverter.GetBytes(64));
                outMs.Position = mhdrDataStartRel + 4; outMs.Write(BitConverter.GetBytes(checked((int)(mtexPosition - mhdrDataStartRel))));
                outMs.Position = mhdrDataStartRel + 8; outMs.Write(BitConverter.GetBytes(mtexData.Length));
                outMs.Position = mhdrDataStartRel + 12; outMs.Write(BitConverter.GetBytes(checked((int)(mddfPosition - mhdrDataStartRel))));
                outMs.Position = mhdrDataStartRel + 16; outMs.Write(BitConverter.GetBytes(mddfData.Length));
                outMs.Position = mhdrDataStartRel + 20; outMs.Write(BitConverter.GetBytes(checked((int)(modfPosition - mhdrDataStartRel))));
                outMs.Position = mhdrDataStartRel + 24; outMs.Write(BitConverter.GetBytes(modfData.Length));

                outMs.Position = mcinPosition; var mcinReal = AlphaMcinBuilder.BuildMcin(mcnkAbs, mcnkSizes); outMs.Write(mcinReal.GetWholeChunk());
                outMs.Position = firstMcnkAbsolute;

                // MAIN.size distance (defer array assign until success)
                int mhdrToFirstVal2 = (alphaMcnkBytes.Any(b => b != null && b.Length > 0)) ? checked((int)(firstMcnkAbsolute - mhdrAbsolute)) : 0;

                // Write MCNKs
                for (int i = 0; i < 256; i++) { var buf = alphaMcnkBytes[i]; if (buf is { Length: > 0 }) outMs.Write(buf); }
                // Sanity: verify MCIN[0] absolute points to 'KNCM'
                try
                {
                    int firstIdx = -1; for (int i = 0; i < 256; i++) if (mcnkAbs[i] > 0) { firstIdx = i; break; }
                    if (firstIdx >= 0)
                    {
                        long abs = mcnkAbs[firstIdx];
                        long cur = outMs.Position;
                        outMs.Position = abs;
                        Span<byte> sig = stackalloc byte[4];
                        outMs.Read(sig);
                        outMs.Position = cur;
                        var tokenStr2 = Encoding.ASCII.GetString(sig);
                        Console.WriteLine($"[pack][check] tile {yy:D2}_{xx:D2} MCIN[{firstIdx}] abs={abs} token='{tokenStr2}'");
                    }
                }
                catch { }
                // Sanity: when verbose, inspect first MCNK with nonzero MCRF counts and verify letters at offsRefs
                if (opts?.Verbose == true)
                {
                    try
                    {
                        for (int i = 0; i < 256; i++)
                        {
                            if (mcnkAbs[i] <= 0) continue;
                            long baseAbs = mcnkAbs[i];
                            long cur = outMs.Position;
                            outMs.Position = baseAbs + 8; // start of MCNK header
                            Span<byte> hdr = stackalloc byte[0x80];
                            outMs.Read(hdr);
                            int nDood = BitConverter.ToInt32(hdr.Slice(0x14, 4));
                            int offsRefs = BitConverter.ToInt32(hdr.Slice(0x24, 4));
                            int nWmo = BitConverter.ToInt32(hdr.Slice(0x3C, 4));
                            if ((nDood + nWmo) > 0 && offsRefs > 0)
                            {
                                outMs.Position = baseAbs + offsRefs;
                                Span<byte> tag = stackalloc byte[4]; outMs.Read(tag);
                                string token = Encoding.ASCII.GetString(tag);
                                Console.WriteLine($"[pack][mcrf] chunk {i:D3} nDood={nDood} nWmo={nWmo} offsRefs={offsRefs} token='{token}'");
                                outMs.Position = cur;
                                break;
                            }
                            outMs.Position = cur;
                        }
                    }
                    catch { }
                }
                // Commit MAIN entries after successful tile write
                mhdrAbs[tileIndex] = checked((int)mhdrAbsolute);
                mhdrToFirst[tileIndex] = mhdrToFirstVal2;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[pack] tile {yy}_{xx} failed: {ex.Message}");
                try
                {
                    // Roll back partial writes for this tile and leave MAIN entries zeroed
                    outMs.SetLength(tileStart);
                    outMs.Position = tileStart;
                    mhdrAbs[tileIndex] = 0;
                    mhdrToFirst[tileIndex] = 0;
                }
                catch { /* best-effort rollback */ }
                continue;
            }
        }

        // Patch MAIN data and save
        outMs.Position = mainStart + 8;
        var patched = AlphaMainBuilder.BuildMain(mhdrAbs, mhdrToFirst, false);
        outMs.Write(patched.Data, 0, patched.Data.Length);
        using (var fs = File.Create(outWdtPath))
        {
            outMs.Position = 0;
            outMs.WriteTo(fs);
        }
        try
        {
            var wdlPath = Path.ChangeExtension(outWdtPath, ".wdl");
            WdlWriterV18.WriteFromArchive(src, mapName, wdlPath);
        }
        catch { }
        if (!string.IsNullOrWhiteSpace(opts?.ExportMccvDir)) Console.WriteLine($"[pack] MCCV images exported: {mccvExported2}");
    }

    private static IEnumerable<string> ReadM2NamesFromBytes(byte[] bytes)
    {
        var result = new List<string>();
        if (bytes == null || bytes.Length < 8) return result;
        int i = 0;
        while (i + 8 <= bytes.Length)
        {
            string four = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > bytes.Length) break;
            if (four == "XDMM")
            {
                int pos = dataStart; int end = dataStart + size;
                while (pos < end)
                {
                    int nul = Array.IndexOf(bytes, (byte)0, pos, end - pos);
                    if (nul == -1) nul = end;
                    int len = nul - pos;
                    if (len > 0) result.Add(Encoding.UTF8.GetString(bytes, pos, len));
                    pos = nul + 1;
                }
                break;
            }
            i = next;
        }
        return result;
    }

    private static IEnumerable<string> ReadWmoNamesFromBytes(byte[] bytes)
    {
        var result = new List<string>();
        if (bytes == null || bytes.Length < 8) return result;
        int i = 0;
        while (i + 8 <= bytes.Length)
        {
            string four = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8; int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > bytes.Length) break;
            if (four == "OMWM") // MWMO reversed
            {
                int pos = dataStart; int end = dataStart + size;
                while (pos < end)
                {
                    int nul = Array.IndexOf(bytes, (byte)0, pos, end - pos); if (nul == -1) nul = end; int len = nul - pos; if (len > 0) result.Add(Encoding.UTF8.GetString(bytes, pos, len)); pos = nul + 1;
                }
                break;
            }
            i = next;
        }
        return result;
    }

    private static byte[] BuildMdnmData(List<string> m2Names)
    {
        if (m2Names == null || m2Names.Count == 0)
            return Array.Empty<byte>();

        using var ms = new MemoryStream();
        foreach (var name in m2Names)
        {
            var nameBytes = Encoding.ASCII.GetBytes(NormalizeAssetName(name));
            ms.Write(nameBytes, 0, nameBytes.Length);
            ms.WriteByte(0);
        }
        ms.WriteByte(0);
        return ms.ToArray();
    }

    private static byte[] ExtractLkMtexData(byte[] adtBytes)
    {
        if (adtBytes == null || adtBytes.Length < 8) return Array.Empty<byte>();
        int i = 0;
        while (i + 8 <= adtBytes.Length)
        {
            string fcc = Encoding.ASCII.GetString(adtBytes, i, 4);
            int size = BitConverter.ToInt32(adtBytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > adtBytes.Length) break;
            if (fcc == "XETM") // MTEX reversed
            {
                // Return the raw string table as-is
                var data = new byte[size];
                Buffer.BlockCopy(adtBytes, dataStart, data, 0, size);
                return data;
            }
            i = next;
        }
        return Array.Empty<byte>();
    }

    private static byte[] BuildMonmData(List<string> wmoNames)
    {
        if (wmoNames == null || wmoNames.Count == 0)
            return Array.Empty<byte>();

        using var ms = new MemoryStream();
        foreach (var name in wmoNames)
        {
            // Use ASCII encoding to match Alpha WoW format
            var nameBytes = Encoding.ASCII.GetBytes(NormalizeAssetName(name));
            ms.Write(nameBytes, 0, nameBytes.Length);
            ms.WriteByte(0); // null terminator per entry
        }
        // Append trailing empty string (extra null) to satisfy Alpha MONM iteration rules
        ms.WriteByte(0);
        return ms.ToArray();
    }

    private static string NormalizeAssetName(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return string.Empty;
        var s = name.Replace('/', '\\').Trim();
        // Alpha uses .mdx for model files
        if (s.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            s = s.Substring(0, s.Length - 3) + ".mdx";
        }
        return s;
    }

    private static List<string> BuildMmdxOrdered(byte[] bytes)
    {
        var result = new List<string>();
        if (bytes == null || bytes.Length < 8) return result;
        int mmdx = FindFourCC(bytes, "MMDX");
        int mmid = FindFourCC(bytes, "MMID");
        if (mmdx < 0 || mmid < 0) return result;
        int mmdxSize = BitConverter.ToInt32(bytes, mmdx + 4);
        int mmdxData = mmdx + 8;
        int mmidSize = BitConverter.ToInt32(bytes, mmid + 4);
        int mmidData = mmid + 8;
        int entries = Math.Max(0, mmidSize / 4);
        for (int i = 0; i < entries; i++)
        {
            int rel = BitConverter.ToInt32(bytes, mmidData + i * 4);
            if (rel < 0 || rel >= mmdxSize) { result.Add(string.Empty); continue; }
            int sp = mmdxData + rel;
            int end = Array.IndexOf(bytes, (byte)0, sp);
            if (end < 0 || end > mmdxData + mmdxSize) end = Math.Min(bytes.Length, mmdxData + mmdxSize);
            int len = Math.Max(0, end - sp);
            string s = len > 0 ? Encoding.ASCII.GetString(bytes, sp, len) : string.Empty;
            result.Add(s);
        }
        return result;
    }

    private static List<string> BuildMwmoOrdered(byte[] bytes)
    {
        var result = new List<string>();
        if (bytes == null || bytes.Length < 8) return result;
        int mwmo = FindFourCC(bytes, "MWMO");
        int mwid = FindFourCC(bytes, "MWID");
        if (mwmo < 0 || mwid < 0) return result;
        int mwmoSize = BitConverter.ToInt32(bytes, mwmo + 4);
        int mwmoData = mwmo + 8;
        int mwidSize = BitConverter.ToInt32(bytes, mwid + 4);
        int mwidData = mwid + 8;
        int entries = Math.Max(0, mwidSize / 4);
        for (int i = 0; i < entries; i++)
        {
            int rel = BitConverter.ToInt32(bytes, mwidData + i * 4);
            if (rel < 0 || rel >= mwmoSize) { result.Add(string.Empty); continue; }
            int sp = mwmoData + rel;
            int end = Array.IndexOf(bytes, (byte)0, sp);
            if (end < 0 || end > mwmoData + mwmoSize) end = Math.Min(bytes.Length, mwmoData + mwmoSize);
            int len = Math.Max(0, end - sp);
            string s = len > 0 ? Encoding.ASCII.GetString(bytes, sp, len) : string.Empty;
            result.Add(s);
        }
        return result;
    }

    private static (List<int>[] doodadRefs, List<int>[] wmoRefs) BuildRefsByChunk(
        byte[] bytes,
        List<string> mmdxOrdered,
        List<string> mwmoOrdered,
        Dictionary<string, int> mdnmIndex,
        Dictionary<string, int> monmIndex,
        int tileYY,
        int tileXX)
    {
        var doodadSets = new HashSet<int>[256];
        var wmoSets = new HashSet<int>[256];
        for (int i = 0; i < 256; i++) { doodadSets[i] = new HashSet<int>(); wmoSets[i] = new HashSet<int>(); }

        const float TILESIZE = 533.33333f;
        const float CHUNK = TILESIZE / 16f;
        float tileMinX = 32f * TILESIZE - (tileXX + 1) * TILESIZE;
        float tileMinY = 32f * TILESIZE - (tileYY + 1) * TILESIZE;

        // MDDF
        int mddf = FindFourCC(bytes, "MDDF");
        if (mddf >= 0)
        {
            int size = BitConverter.ToInt32(bytes, mddf + 4);
            int data = mddf + 8;
            const int entry = 36;
            int count = Math.Max(0, size / entry);
            for (int i = 0; i < count; i++)
            {
                int p = data + i * entry; if (p + entry > bytes.Length) break;
                int localIdx = BitConverter.ToInt32(bytes, p + 0);
                if (localIdx >= 0 && localIdx < mmdxOrdered.Count)
                {
                    string s = NormalizeAssetName(mmdxOrdered[localIdx]);
                    if (mdnmIndex.TryGetValue(s, out int g))
                    {
                        float wx = BitConverter.ToSingle(bytes, p + 8);
                        float wz = BitConverter.ToSingle(bytes, p + 16);
                        int cx = (int)Math.Floor((wx - tileMinX) / CHUNK);
                        int cy = (int)Math.Floor((wz - tileMinY) / CHUNK);
                        if (cx >= 0 && cx < 16 && cy >= 0 && cy < 16)
                        {
                            int idx = cy * 16 + cx;
                            doodadSets[idx].Add(g);
                        }
                    }
                }
            }
        }

        // MODF
        int modf = FindFourCC(bytes, "MODF");
        if (modf >= 0)
        {
            int size = BitConverter.ToInt32(bytes, modf + 4);
            int data = modf + 8;
            const int entry = 64;
            int count = Math.Max(0, size / entry);
            for (int i = 0; i < count; i++)
            {
                int p = data + i * entry; if (p + entry > bytes.Length) break;
                int localIdx = BitConverter.ToInt32(bytes, p + 0);
                if (localIdx >= 0 && localIdx < mwmoOrdered.Count)
                {
                    string s = NormalizeAssetName(mwmoOrdered[localIdx]);
                    if (monmIndex.TryGetValue(s, out int g))
                    {
                        float wx = BitConverter.ToSingle(bytes, p + 8);
                        float wz = BitConverter.ToSingle(bytes, p + 16);
                        int cx = (int)Math.Floor((wx - tileMinX) / CHUNK);
                        int cy = (int)Math.Floor((wz - tileMinY) / CHUNK);
                        if (cx >= 0 && cx < 16 && cy >= 0 && cy < 16)
                        {
                            int idx = cy * 16 + cx;
                            wmoSets[idx].Add(g);
                        }
                    }
                }
            }
        }

        var doodadRefs = new List<int>[256];
        var wmoRefs = new List<int>[256];
        for (int i = 0; i < 256; i++)
        {
            var dl = new List<int>(doodadSets[i]); dl.Sort(); doodadRefs[i] = dl;
            var wl = new List<int>(wmoSets[i]); wl.Sort(); wmoRefs[i] = wl;
        }
        return (doodadRefs, wmoRefs);
    }

    private static byte[] BuildMddfFromLk(byte[] bytes, List<string> mmdxOrdered, Dictionary<string, int> mdnmIndex)
    {
        if (bytes == null || bytes.Length < 8) return Array.Empty<byte>();
        int mddf = FindFourCC(bytes, "MDDF");
        if (mddf < 0) return Array.Empty<byte>();
        int size = BitConverter.ToInt32(bytes, mddf + 4);
        int data = mddf + 8;
        const int entry = 36;
        int count = Math.Max(0, Math.Min(size / entry, 1_000_000));
        using var ms = new MemoryStream();
        for (int i = 0; i < count; i++)
        {
            int p = data + i * entry;
            if (p + entry > bytes.Length) break;
            int localIdx = BitConverter.ToInt32(bytes, p + 0);
            if (localIdx < 0 || localIdx >= mmdxOrdered.Count) continue;
            string name = NormalizeAssetName(mmdxOrdered[localIdx]);
            if (string.IsNullOrEmpty(name) || !mdnmIndex.TryGetValue(name, out int globalIdx)) continue;
            int uniqueId = BitConverter.ToInt32(bytes, p + 4);
            float posX = BitConverter.ToSingle(bytes, p + 8);
            float posY = BitConverter.ToSingle(bytes, p + 12);
            float posZ = BitConverter.ToSingle(bytes, p + 16);
            float rotX = BitConverter.ToSingle(bytes, p + 20);
            float rotY = BitConverter.ToSingle(bytes, p + 24);
            float rotZ = BitConverter.ToSingle(bytes, p + 28);
            ushort scale = BitConverter.ToUInt16(bytes, p + 32);
            ushort flags = BitConverter.ToUInt16(bytes, p + 34);
            ms.Write(BitConverter.GetBytes(globalIdx));
            ms.Write(BitConverter.GetBytes(uniqueId));
            // Alpha MDDF stores position as X, Z(height), Y in centered world coordinates
            ms.Write(BitConverter.GetBytes(posX));
            ms.Write(BitConverter.GetBytes(posZ));
            ms.Write(BitConverter.GetBytes(posY));
            ms.Write(BitConverter.GetBytes(rotX));
            ms.Write(BitConverter.GetBytes(rotY));
            ms.Write(BitConverter.GetBytes(rotZ));
            ms.Write(BitConverter.GetBytes(scale));
            ms.Write(BitConverter.GetBytes(flags));
        }
        return ms.ToArray();
    }

    private static byte[] BuildModfFromLk(byte[] bytes, List<string> mwmoOrdered, Dictionary<string, int> monmIndex)
    {
        if (bytes == null || bytes.Length < 8) return Array.Empty<byte>();
        int modf = FindFourCC(bytes, "MODF");
        if (modf < 0) return Array.Empty<byte>();
        int size = BitConverter.ToInt32(bytes, modf + 4);
        int data = modf + 8;
        const int entry = 64;
        int count = Math.Max(0, Math.Min(size / entry, 1_000_000));
        using var ms = new MemoryStream();
        for (int i = 0; i < count; i++)
        {
            int p = data + i * entry;
            if (p + entry > bytes.Length) break;
            int localIdx = BitConverter.ToInt32(bytes, p + 0);
            if (localIdx < 0 || localIdx >= mwmoOrdered.Count) continue;
            string name = NormalizeAssetName(mwmoOrdered[localIdx]);
            if (string.IsNullOrEmpty(name) || !monmIndex.TryGetValue(name, out int globalIdx)) continue;
            int uniqueId = BitConverter.ToInt32(bytes, p + 4);
            float posX = BitConverter.ToSingle(bytes, p + 8);
            float posY = BitConverter.ToSingle(bytes, p + 12);
            float posZ = BitConverter.ToSingle(bytes, p + 16);
            float rotX = BitConverter.ToSingle(bytes, p + 20);
            float rotY = BitConverter.ToSingle(bytes, p + 24);
            float rotZ = BitConverter.ToSingle(bytes, p + 28);
            // extents 0x20..0x3F
            ms.Write(BitConverter.GetBytes(globalIdx));
            ms.Write(BitConverter.GetBytes(uniqueId));
            // Alpha MODF stores position as X, Z(height), Y in centered world coordinates
            ms.Write(BitConverter.GetBytes(posX));
            ms.Write(BitConverter.GetBytes(posZ));
            ms.Write(BitConverter.GetBytes(posY));
            ms.Write(BitConverter.GetBytes(rotX));
            ms.Write(BitConverter.GetBytes(rotY));
            ms.Write(BitConverter.GetBytes(rotZ));
            // extents (copy-through, centered world coordinate basis)
            ms.Write(bytes, p + 32, 24);
            ushort flags = BitConverter.ToUInt16(bytes, p + 56);
            ushort doodadSet = BitConverter.ToUInt16(bytes, p + 58);
            ushort nameSet = BitConverter.ToUInt16(bytes, p + 60);
            ushort scale = BitConverter.ToUInt16(bytes, p + 62);
            ms.Write(BitConverter.GetBytes(flags));
            ms.Write(BitConverter.GetBytes(doodadSet));
            ms.Write(BitConverter.GetBytes(nameSet));
            ms.Write(BitConverter.GetBytes(scale));
        }
        return ms.ToArray();
    }

    private static int FindFourCC(byte[] buf, string forwardFourCC)
    {
        if (buf == null || buf.Length < 8) return -1;
        if (string.IsNullOrEmpty(forwardFourCC) || forwardFourCC.Length != 4) return -1;
        
        // On-disk bytes are reversed in our Chunk reader logic; here we scan for reversed letters
        string reversed = new string(new[] { forwardFourCC[3], forwardFourCC[2], forwardFourCC[1], forwardFourCC[0] });
        for (int i = 0; i + 8 <= buf.Length;)
        {
            // Ensure we have enough bytes for FourCC
            if (i < 0 || i + 4 > buf.Length) break;
            
            string fcc = Encoding.ASCII.GetString(buf, i, 4);
            int size = BitConverter.ToInt32(buf, i + 4);
            
            // Validate size to prevent infinite loops or negative values
            if (size < 0 || size > buf.Length) break;
            
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            
            if (fcc == reversed) return i;
            if (dataStart + size > buf.Length) break;
            
            // Ensure we're making forward progress
            if (next <= i) break;
            
            i = next;
        }
        return -1;
    }

    private static (int YY, int XX) WorldToTileFromCentered(float worldX, float worldY)
    {
        const float BLOCK_SIZE = 533.33333f;
        int xx = (int)Math.Floor(32 - (worldX / BLOCK_SIZE));
        int yy = (int)Math.Floor(32 - (worldY / BLOCK_SIZE));
        if (xx < 0) xx = 0; else if (xx > 63) xx = 63;
        if (yy < 0) yy = 0; else if (yy > 63) yy = 63;
        return (yy, xx);
    }

    // --- MCCV Export (optional debug) ---
    private static bool ExportMccvIfPresent(byte[] adtBytes, int mcnkOffset, string baseOutDir, int tileY, int tileX, int chunkIdx)
    {
        const int ChunkLettersAndSize = 8;
        const int McnkHeaderSize = 0x80;
        if (mcnkOffset < 0 || mcnkOffset + ChunkLettersAndSize + McnkHeaderSize > adtBytes.Length) return false;

        // MCCV offset is at 0x64 (100) within the LK MCNK header
        const int mccvOffsetInHeader = 0x64;
        int headerStart = mcnkOffset + ChunkLettersAndSize;
        int mccvRel = BitConverter.ToInt32(adtBytes, headerStart + mccvOffsetInHeader);
        if (mccvRel <= 0) return false;

        // Offsets in header point to chunk DATA (after 8-byte header). Move back 8 to land on FourCC.
        int mccvChunkOffset = headerStart + mccvRel - ChunkLettersAndSize;
        if (mccvChunkOffset < 0 || mccvChunkOffset + 8 > adtBytes.Length) return false;

        Chunk mccvChunk;
        try { mccvChunk = new Chunk(adtBytes, mccvChunkOffset); }
        catch { return false; }
        if (mccvChunk.GivenSize <= 0 || mccvChunk.Data.Length <= 0) return false;

        // Build a 64x64 BGR image from MCCV vertex shading (use 9x9 outer grid, bilinear upsample)
        var bgr = BuildMccvBgrImage(mccvChunk.Data, 64, 64);
        if (bgr.Length == 0) return false;

        var outDir = Path.Combine(baseOutDir, $"{tileY:D2}_{tileX:D2}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, $"{tileY:D2}_{tileX:D2}_mcnk_{chunkIdx:D3}_mccv.bmp");
        try { SaveBmp24(outPath, 64, 64, bgr); } catch { return false; }
        return true;
    }

    private static byte[] BuildMccvBgrImage(byte[] mccvData, int width, int height)
    {
        // Expect 145 RGBA entries interleaved as rows: 9,8,9,8,... (like MCNR/MCVT)
        if (mccvData == null || mccvData.Length < 9 * 9 * 4) return Array.Empty<byte>();

        var outer = new (byte r, byte g, byte b)[9,9];
        int src = 0;
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (src + 4 > mccvData.Length) return Array.Empty<byte>();
                byte r = mccvData[src + 0]; byte g = mccvData[src + 1]; byte b = mccvData[src + 2];
                outer[i, j] = (r, g, b);
                src += 4;
            }
            if (i < 8)
            {
                // skip inner row (8 vertices) to keep outer-only for simple bilinear
                src += 8 * 4;
                if (src > mccvData.Length) return Array.Empty<byte>();
            }
        }

        var bgr = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            double v = (double)y * 8.0 / (height - 1);
            int v0 = (int)Math.Floor(v);
            int v1 = Math.Min(8, v0 + 1);
            double ty = v - v0;
            for (int x = 0; x < width; x++)
            {
                double u = (double)x * 8.0 / (width - 1);
                int u0 = (int)Math.Floor(u);
                int u1 = Math.Min(8, u0 + 1);
                double tx = u - u0;

                var c00 = outer[v0, u0];
                var c10 = outer[v0, u1];
                var c01 = outer[v1, u0];
                var c11 = outer[v1, u1];

                double w00 = (1 - tx) * (1 - ty);
                double w10 = tx * (1 - ty);
                double w01 = (1 - tx) * ty;
                double w11 = tx * ty;

                int r = (int)(c00.r * w00 + c10.r * w10 + c01.r * w01 + c11.r * w11 + 0.5);
                int g = (int)(c00.g * w00 + c10.g * w10 + c01.g * w01 + c11.g * w11 + 0.5);
                int b = (int)(c00.b * w00 + c10.b * w10 + c01.b * w01 + c11.b * w11 + 0.5);

                int idx = (y * width + x) * 3;
                bgr[idx + 0] = (byte)Math.Clamp(b, 0, 255);
                bgr[idx + 1] = (byte)Math.Clamp(g, 0, 255);
                bgr[idx + 2] = (byte)Math.Clamp(r, 0, 255);
            }
        }
        return bgr;
    }

    private static void SaveBmp24(string path, int width, int height, byte[] bgrTopDown)
    {
        // 24bpp BMP with BITMAPINFOHEADER; use negative height for top-down rows
        int stride = width * 3;
        int imageSize = stride * height;
        int fileSize = 54 + imageSize;
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> header = stackalloc byte[54];
        // BITMAPFILEHEADER
        header[0] = (byte)'B'; header[1] = (byte)'M';
        BitConverter.GetBytes(fileSize).CopyTo(header.Slice(2));
        // reserved 4 bytes at 6..9 are zero
        BitConverter.GetBytes(54).CopyTo(header.Slice(10)); // pixel data offset
        // BITMAPINFOHEADER
        BitConverter.GetBytes(40).CopyTo(header.Slice(14)); // info header size
        BitConverter.GetBytes(width).CopyTo(header.Slice(18));
        BitConverter.GetBytes(-height).CopyTo(header.Slice(22)); // negative for top-down
        BitConverter.GetBytes((ushort)1).CopyTo(header.Slice(26)); // planes
        BitConverter.GetBytes((ushort)24).CopyTo(header.Slice(28)); // bpp
        // compression 0 at 30..33
        BitConverter.GetBytes(imageSize).CopyTo(header.Slice(34));
        // xppm/yppm and palette fields left zero
        fs.Write(header);
        fs.Write(bgrTopDown, 0, bgrTopDown.Length);
    }
}
