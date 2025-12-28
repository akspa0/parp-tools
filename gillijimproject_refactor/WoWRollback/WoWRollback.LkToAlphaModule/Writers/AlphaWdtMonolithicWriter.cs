using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Globalization;
using System.Threading.Tasks;
using WoWRollback.LkToAlphaModule;
using WoWRollback.Core.Services.Assets;
using WoWRollback.LkToAlphaModule.Readers;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using WoWRollback.LkToAlphaModule.Builders;
using WoWRollback.Core.Services.Archive;
using WoWRollback.LkToAlphaModule.AssetConversion;

namespace WoWRollback.LkToAlphaModule.Writers;

public sealed class AlphaWdtMonolithicWriter
{
    private const int GridTiles = 64 * 64; // 4096
    private const long OutSizeLimit = (long)int.MaxValue - (1L << 16);

    public static void WriteMonolithic(string lkWdtPath, string lkMapDir, string outWdtPath, LkToAlphaOptions? opts = null, bool skipWmos = false)
    {
        bool verbose = opts?.Verbose == true;
        var usedM2ForExtract = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var usedWmoForExtract = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var texturesCsv = new List<string> { "tile_yy,tile_xx,chunk_idx,tex_path,src" };
        var texturesUsed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
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
                foreach (var n in BuildMmdxOrdered(bytesScan)) allM2Names.Add(n);
                foreach (var n in BuildMwmoOrdered(bytesScan)) allWmoNames.Add(n);
                var baseNameScan = Path.GetFileNameWithoutExtension(rootAdt);
                var dirScan = Path.GetDirectoryName(rootAdt) ?? ".";
                var objScan = Path.Combine(dirScan, baseNameScan + "_obj.adt");
                var obj0Scan = Path.Combine(dirScan, baseNameScan + "_obj0.adt");
                var obj1Scan = Path.Combine(dirScan, baseNameScan + "_obj1.adt");
                if (File.Exists(objScan))
                {
                    var objBytesScan = File.ReadAllBytes(objScan);
                    foreach (var n in BuildMmdxOrdered(objBytesScan)) allM2Names.Add(n);
                    foreach (var n in BuildMwmoOrdered(objBytesScan)) allWmoNames.Add(n);
                }
            // Clear diagnostics hook for safety before moving to next tile
            AlphaMcnkBuilder.OnChunkBuilt = null;
                if (File.Exists(obj0Scan))
                {
                    var obj0BytesScan = File.ReadAllBytes(obj0Scan);
                    foreach (var n in BuildMmdxOrdered(obj0BytesScan)) allM2Names.Add(n);
                    foreach (var n in BuildMwmoOrdered(obj0BytesScan)) allWmoNames.Add(n);
                }
                if (File.Exists(obj1Scan))
                {
                    var obj1BytesScan = File.ReadAllBytes(obj1Scan);
                    foreach (var n in BuildMmdxOrdered(obj1BytesScan)) allM2Names.Add(n);
                    foreach (var n in BuildMwmoOrdered(obj1BytesScan)) allWmoNames.Add(n);
                }
            }
            catch { /* best-effort name harvest */ }
        }
        
        int mccvExported = 0;
        int mccvHeaders = 0;
        var objectsCsvLines = new List<string> { "tile_yy,tile_xx,mddf_count,modf_count,first_mddf,first_modf" };
        var mcnkSizesCsv = new List<string> { "tile_yy,tile_xx,chunk_idx,nLayers,mcly_bytes,mcal_bytes,mcsh_bytes,mcse_bytes,mclq_bytes,nDoodadRefs,nMapObjRefs,mcrf_bytes" };
        var rawObjectsLines = new List<string> { "tile_yy,tile_xx,source,mddf_count,modf_count,first_mddf,first_modf" };
        var placementsSkippedLines = new List<string> { "tile_yy,tile_xx,source,type,local_index,name,reason" };
        var placementsMddfLines = new List<string> { "tile_yy,tile_xx,source,chunk_idx,local_index,global_index,name,uid,x,z,y,rx,ry,rz,scale,flags" };
        var placementsModfLines = new List<string> { "tile_yy,tile_xx,source,chunk_idx,local_index,global_index,name,uid,x,z,y,rx,ry,rz,bb_minx,bb_miny,bb_minz,bb_maxx,bb_maxy,bb_maxz,flags,doodad_set,name_set,scale" };
        var packTimingRows = new List<string>();
        long totalMddfWritten = 0, totalModfWritten = 0;

        // Build WDT scaffolding (Alpha format): MVER -> MPHD(16) -> MAIN -> MDNM -> MONM
        using var ms = new MemoryStream();
        // MVER
        ms.Write(new Chunk("MVER", 4, BitConverter.GetBytes(18)).GetWholeChunk());
        // MPHD (128 bytes) placeholder; will patch absolute offsets and counts later
        long mphdStart = ms.Position; var mphd = new Chunk("MPHD", 128, new byte[128]); ms.Write(mphd.GetWholeChunk()); long mphdDataStart = mphdStart + 8;
        // MAIN placeholder (4096 * 16)
        var mainData = new byte[GridTiles * 16]; var main = new Chunk("MAIN", mainData.Length, mainData); long mainStart = ms.Position; ms.Write(main.GetWholeChunk());
        // Build name lists (normalize + distinct); currently use collected sets (may be empty)
        var m2Names = allM2Names
            .Select(NormalizeAssetName)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        var wmoNames = allWmoNames
            .Select(NormalizeAssetName)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (!string.IsNullOrWhiteSpace(opts?.TargetListfilePath) && File.Exists(opts.TargetListfilePath))
        {
            try
            {
                var idx = ListfileIndex.Load(opts.TargetListfilePath!);
                var gate = new AssetGate(idx);
                var keptM2 = gate.FilterNames(m2Names, out var droppedM2);
                var keptWmo = gate.FilterNames(wmoNames, out var droppedWmo);
                var outDir = Path.GetDirectoryName(outWdtPath) ?? ".";
                var dropCsv = Path.Combine(outDir, "dropped_assets.csv");
                AssetGate.WriteDropReport(dropCsv, droppedM2, droppedWmo);
                try
                {
                    var keptCsv = Path.Combine(outDir, "kept_assets.csv");
                    using var sw = new StreamWriter(keptCsv, false, Encoding.UTF8);
                    sw.WriteLine("type,path");
                    foreach (var p in keptM2) sw.WriteLine($"m2,{p}");
                    foreach (var p in keptWmo) sw.WriteLine($"wmo,{p}");
                }
                catch { }
                // Apply gating results
                m2Names = keptM2.ToList();
                wmoNames = keptWmo.ToList();
            }
            catch { }
        }
        // Respect skipWmos flag
        if (skipWmos) wmoNames.Clear();
        // Global name indices for MDNM/MONM
        var mdnmIndexFs = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < m2Names.Count; i++) mdnmIndexFs[NormalizeAssetName(m2Names[i])] = i;
        var monmIndexFs = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < wmoNames.Count; i++) monmIndexFs[NormalizeAssetName(wmoNames[i])] = i;
        // MDNM/MONM chunks
        long mdnmStart = ms.Position; var mdnmData = BuildMdnmData(m2Names); var mdnm = new Chunk("MDNM", mdnmData.Length, mdnmData); ms.Write(mdnm.GetWholeChunk());
        long monmStart = ms.Position; var monmData = BuildMonmData(wmoNames); var monm = new Chunk("MONM", monmData.Length, monmData); ms.Write(monm.GetWholeChunk());
        // MODF chunk (top-level WMO definitions) - must exist even if empty per Alpha spec
        ms.Write(new Chunk("MODF", 0, Array.Empty<byte>()).GetWholeChunk());
        // Patch MPHD with absolute offsets and counts
        long savePos = ms.Position; ms.Position = mphdDataStart; Span<byte> mphdData = new byte[128]; mphdData.Clear();
        // MPHD layout: [0..3]=nTextures (M2), [4..7]=MDNM abs, [8..11]=nMapObjNames (WMO, +1 when any), [12..15]=MONM abs
        BitConverter.GetBytes(m2Names.Count > 0 ? m2Names.Count + 1 : 0).CopyTo(mphdData);
        BitConverter.GetBytes(checked((int)mdnmStart)).CopyTo(mphdData.Slice(4));
        BitConverter.GetBytes(wmoNames.Count > 0 ? wmoNames.Count + 1 : 0).CopyTo(mphdData.Slice(8));
        BitConverter.GetBytes(checked((int)monmStart)).CopyTo(mphdData.Slice(12));
        ms.Write(mphdData); ms.Position = savePos;
        try
        {
            var outDirIdx = Path.GetDirectoryName(outWdtPath) ?? ".";
            WriteIndexCsv(outDirIdx, mdnmIndexFs, monmIndexFs);
        }
        catch { }

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
                Stopwatch swTile = Stopwatch.StartNew();
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

            // Inventory textures referenced by present chunks for diagnostics (textures.csv)
            try
            {
                var mtexNames = ParseMtexNamesLocal(mtexData);
                var presentTexScan = new List<int>(presentIndices);
                if (texMcnkOffsets != null)
                {
                    for (int i = 0; i < texMcnkOffsets.Count; i++)
                    {
                        if (texMcnkOffsets[i] > 0 && !presentTexScan.Contains(i)) presentTexScan.Add(i);
                    }
                }
                presentTexScan.Sort();
                bool preferTexForMtex = (texBytesForChunks != null && texBytesForChunks.Length > 0);
                foreach (var ci in presentTexScan)
                {
                    int rootOff = (ci < lkMcnkOffsets.Count) ? lkMcnkOffsets[ci] : 0;
                    int texOff = (texMcnkOffsets != null && ci < texMcnkOffsets.Count) ? texMcnkOffsets[ci] : 0;
                    List<int> ids = new List<int>();
                    string srcTag = string.Empty;
                    if (preferTexForMtex)
                    {
                        if (texOff > 0) { ids = ReadMCLYTextureIdsInChunk(texBytesForChunks!, texOff); srcTag = "tex"; }
                        else if (rootOff > 0) { ids = ReadMCLYTextureIdsInChunk(bytes, rootOff); srcTag = "root"; }
                    }
                    else
                    {
                        if (rootOff > 0) { ids = ReadMCLYTextureIdsInChunk(bytes, rootOff); srcTag = "root"; }
                        else if (texOff > 0) { ids = ReadMCLYTextureIdsInChunk(texBytesForChunks!, texOff); srcTag = "tex"; }
                    }
                    foreach (var tid in ids)
                    {
                        if (tid >= 0 && tid < mtexNames.Count)
                        {
                            var texPath = NormalizeTexturePath(mtexNames[tid]);
                            if (!string.IsNullOrWhiteSpace(texPath))
                            {
                                if (texturesUsed.Add(texPath)) { }
                                texturesCsv.Add($"{yy},{xx},{ci},{texPath},{srcTag}");
                            }
                        }
                    }
                }
            }
            catch { }

            // Build MDDF/MODF from _obj0/_obj1 if present (fallback: _obj, then root)
            var objAdt = Path.Combine(rootDir, baseName + "_obj.adt");
            var obj0Adt = Path.Combine(rootDir, baseName + "_obj0.adt");
            var obj1Adt = Path.Combine(rootDir, baseName + "_obj1.adt");

            byte[] objBytesFs = File.Exists(objAdt) ? File.ReadAllBytes(objAdt) : Array.Empty<byte>();
            byte[] obj0BytesFs = File.Exists(obj0Adt) ? File.ReadAllBytes(obj0Adt) : Array.Empty<byte>();
            byte[] obj1BytesFs = File.Exists(obj1Adt) ? File.ReadAllBytes(obj1Adt) : Array.Empty<byte>();

            // Local name tables per source
            var mmdxRoot = BuildMmdxOrdered(bytes);
            var mwmoRoot = BuildMwmoOrdered(bytes);
            var mmdxObj = BuildMmdxOrdered(objBytesFs);
            var mwmoObj = BuildMwmoOrdered(objBytesFs);
            var mmdxObj0 = BuildMmdxOrdered(obj0BytesFs);
            var mwmoObj0 = BuildMwmoOrdered(obj0BytesFs);
            var mmdxObj1 = BuildMmdxOrdered(obj1BytesFs);
            var mwmoObj1 = BuildMwmoOrdered(obj1BytesFs);

            // MDDF/MODF refs by chunk must point to actual record indices within this tile's MDDF/MODF
            var doodadRefsByChunkFs = new System.Collections.Generic.List<int>[256];
            var wmoRefsByChunkFs = new System.Collections.Generic.List<int>[256];
            for (int i = 0; i < 256; i++) { doodadRefsByChunkFs[i] = new System.Collections.Generic.List<int>(); wmoRefsByChunkFs[i] = new System.Collections.Generic.List<int>(); }

            byte[] mddfDataFs;
            if (opts?.SkipM2 == true)
            {
                mddfDataFs = Array.Empty<byte>();
            }
            else
            {
                using var msMddf = new MemoryStream();
                int mddfBaseFs = 0;
                if (obj0BytesFs.Length > 0)
                {
                    var part = BuildMddfFromLk(obj0BytesFs, mmdxObj0, mdnmIndexFs, placementsSkippedLines, "obj0", yy, xx, doodadRefsByChunkFs, mddfBaseFs, usedM2ForExtract, placementsMddfLines);
                    if (part.Length > 0) { msMddf.Write(part, 0, part.Length); mddfBaseFs += part.Length / 36; }
                }
                if (objBytesFs.Length > 0)
                {
                    var part = BuildMddfFromLk(objBytesFs, mmdxObj, mdnmIndexFs, placementsSkippedLines, "obj", yy, xx, doodadRefsByChunkFs, mddfBaseFs, usedM2ForExtract, placementsMddfLines);
                    if (part.Length > 0) { msMddf.Write(part, 0, part.Length); mddfBaseFs += part.Length / 36; }
                }
                {
                    var part = BuildMddfFromLk(bytes, mmdxRoot, mdnmIndexFs, placementsSkippedLines, "root", yy, xx, doodadRefsByChunkFs, mddfBaseFs, usedM2ForExtract, placementsMddfLines);
                    if (part.Length > 0) { msMddf.Write(part, 0, part.Length); mddfBaseFs += part.Length / 36; }
                }
                mddfDataFs = msMddf.ToArray();
            }

            // MODF: prefer obj1, then obj, then root (or skip entirely if requested)
            byte[] modfDataFs;
            if (!skipWmos)
            {
                using var msModf = new MemoryStream();
                int modfBaseFs = 0;
                if (obj1BytesFs.Length > 0)
                {
                    var part = BuildModfFromLk(obj1BytesFs, mwmoObj1, monmIndexFs, placementsSkippedLines, "obj1", yy, xx, wmoRefsByChunkFs, modfBaseFs, usedWmoForExtract, placementsModfLines);
                    if (part.Length > 0) { msModf.Write(part, 0, part.Length); modfBaseFs += part.Length / 64; }
                }
                if (objBytesFs.Length > 0)
                {
                    var part = BuildModfFromLk(objBytesFs, mwmoObj, monmIndexFs, placementsSkippedLines, "obj", yy, xx, wmoRefsByChunkFs, modfBaseFs, usedWmoForExtract, placementsModfLines);
                    if (part.Length > 0) { msModf.Write(part, 0, part.Length); modfBaseFs += part.Length / 64; }
                }
                {
                    var part = BuildModfFromLk(bytes, mwmoRoot, monmIndexFs, placementsSkippedLines, "root", yy, xx, wmoRefsByChunkFs, modfBaseFs, usedWmoForExtract, placementsModfLines);
                    if (part.Length > 0) { msModf.Write(part, 0, part.Length); modfBaseFs += part.Length / 64; }
                }
                modfDataFs = msModf.ToArray();
            }
            else
            {
                // Clear WMO refs per chunk when skipping WMOs
                for (int i = 0; i < 256; i++) wmoRefsByChunkFs[i].Clear();
                modfDataFs = Array.Empty<byte>();
            }

            // Raw source counts and first names
            try
            {
                void AddRaw(string sourceLbl, byte[] b, List<string> mmdxOrd, List<string> mwmoOrd)
                {
                    int cm = GetMddfCount(b);
                    int cw = GetModfCount(b);
                    string fm = TryGetFirstMddfName(b, mmdxOrd);
                    string fw = TryGetFirstModfName(b, mwmoOrd);
                    rawObjectsLines.Add($"{yy},{xx},{sourceLbl},{cm},{cw},{fm},{fw}");
                }
                AddRaw("obj0", obj0BytesFs, mmdxObj0, mwmoObj0);
                AddRaw("obj1", obj1BytesFs, mmdxObj1, mwmoObj1);
                AddRaw("obj", objBytesFs, mmdxObj, mwmoObj);
                AddRaw("root", bytes, mmdxRoot, mwmoRoot);
            }
            catch { }

            // doodadRefsByChunkFs and wmoRefsByChunkFs were populated during MDDF/MODF writes above

            long mddfPosition = outMs.Position; outMs.Write(new Chunk("MDDF", mddfDataFs.Length, mddfDataFs).GetWholeChunk()); long mddfEnd = outMs.Position;
            long modfPosition = outMs.Position; outMs.Write(new Chunk("MODF", modfDataFs.Length, modfDataFs).GetWholeChunk()); long modfEnd = outMs.Position;
            int mddfCountFs = mddfDataFs.Length / 36;
            int modfCountFs = modfDataFs.Length / 64;
            totalMddfWritten += mddfCountFs; totalModfWritten += modfCountFs;
            if (verbose)
            {
                Console.WriteLine($"[pack][objects] tile {yy:D2}_{xx:D2} mddf={mddfCountFs} modf={modfCountFs}");
            }
            string firstMddfName = string.Empty;
            string firstModfName = string.Empty;
            try
            {
                int mddfOff = FindFourCC(objBytesFs, "MDDF");
                if (mddfOff >= 0)
                {
                    int size = BitConverter.ToInt32(objBytesFs, mddfOff + 4);
                    int data = mddfOff + 8;
                    int count = Math.Max(0, size / 36);
                    if (count > 0)
                    {
                        int p = data;
                        int local = BitConverter.ToInt32(objBytesFs, p + 0);
                        if (local >= 0 && local < mmdxObj.Count)
                        {
                            firstMddfName = NormalizeAssetName(mmdxObj[local]);
                        }
                    }
                }
            }
            catch { }
            try
            {
                int modfOff = FindFourCC(objBytesFs, "MODF");
                if (modfOff >= 0)
                {
                    int size = BitConverter.ToInt32(objBytesFs, modfOff + 4);
                    int data = modfOff + 8;
                    int count = Math.Max(0, size / 64);
                    if (count > 0)
                    {
                        int p = data;
                        int local = BitConverter.ToInt32(objBytesFs, p + 0);
                        if (local >= 0 && local < mwmoObj.Count)
                        {
                            firstModfName = NormalizeAssetName(mwmoObj[local]);
                        }
                    }
                }
            }
            catch { }
            objectsCsvLines.Add($"{yy},{xx},{mddfCountFs},{modfCountFs},{firstMddfName},{firstModfName}");

            // Emit per-tile MCRF debug CSV for quick inspection
            try
            {
                var outDirLocal = Path.GetDirectoryName(outWdtPath) ?? ".";
                Directory.CreateDirectory(Path.Combine(outDirLocal, "mcrf_debug"));
                var dbgPath = Path.Combine(outDirLocal, "mcrf_debug", $"{yy:D2}_{xx:D2}_mcrf_debug.csv");
                using var swDbg = new StreamWriter(dbgPath, false, Encoding.UTF8);
                swDbg.WriteLine("chunk_idx,nDoodadRefs,nMapObjRefs,d_min,d_max,w_min,w_max,d_samples,w_samples,violations");
                for (int ci = 0; ci < 256; ci++)
                {
                    var drefs = doodadRefsByChunkFs[ci];
                    var wrefs = wmoRefsByChunkFs[ci];
                    int dn = drefs?.Count ?? 0;
                    int wn = wrefs?.Count ?? 0;
                    int dmin = int.MaxValue, dmax = int.MinValue, wmin = int.MaxValue, wmax = int.MinValue;
                    bool viol = false;
                    if (dn > 0)
                    {
                        for (int k = 0; k < dn; k++) { int v = drefs[k]; if (v < dmin) dmin = v; if (v > dmax) dmax = v; if (v < 0 || v >= m2Names.Count) viol = true; }
                    }
                    else { dmin = dmax = -1; }
                    if (wn > 0)
                    {
                        for (int k = 0; k < wn; k++) { int v = wrefs[k]; if (v < wmin) wmin = v; if (v > wmax) wmax = v; if (v < 0 || v >= wmoNames.Count) viol = true; }
                    }
                    else { wmin = wmax = -1; }

                    // Sample up to 4 refs of each type
                    string ds = string.Empty;
                    if (dn > 0)
                    {
                        int take = Math.Min(4, dn);
                        var arr = new string[take];
                        for (int t = 0; t < take; t++) arr[t] = drefs[t].ToString(CultureInfo.InvariantCulture);
                        ds = string.Join('|', arr);
                    }
                    string ws = string.Empty;
                    if (wn > 0)
                    {
                        int take = Math.Min(4, wn);
                        var arr = new string[take];
                        for (int t = 0; t < take; t++) arr[t] = wrefs[t].ToString(CultureInfo.InvariantCulture);
                        ws = string.Join('|', arr);
                    }
                    swDbg.WriteLine($"{ci},{dn},{wn},{dmin},{dmax},{wmin},{wmax},{ds},{ws},{(viol ? "viol" : "")}");
                }
            }
            catch { }

            // Rebuild MCNKs with MCRF refs
            // Hook diagnostics to collect per-chunk sizes for this tile
            AlphaMcnkBuilder.OnChunkBuilt = (ciDiag, nLayersDiag, mclyLen, mcalLen, mcshLen, mcseLen, mclqLen, nDRefs, nWRefs, mcrfLen) =>
            {
                mcnkSizesCsv.Add(string.Join(',', new[]
                {
                    yy.ToString(CultureInfo.InvariantCulture),
                    xx.ToString(CultureInfo.InvariantCulture),
                    ciDiag.ToString(CultureInfo.InvariantCulture),
                    nLayersDiag.ToString(CultureInfo.InvariantCulture),
                    mclyLen.ToString(CultureInfo.InvariantCulture),
                    mcalLen.ToString(CultureInfo.InvariantCulture),
                    mcshLen.ToString(CultureInfo.InvariantCulture),
                    mcseLen.ToString(CultureInfo.InvariantCulture),
                    mclqLen.ToString(CultureInfo.InvariantCulture),
                    nDRefs.ToString(CultureInfo.InvariantCulture),
                    nWRefs.ToString(CultureInfo.InvariantCulture),
                    mcrfLen.ToString(CultureInfo.InvariantCulture),
                }));
            };
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
                    Span<byte> sig = new byte[4];
                    int read = outMs.Read(sig);
                    outMs.Position = cur;
                    var tokenRaw = Encoding.ASCII.GetString(sig);
                    var tokenFwd = new string(tokenRaw.Reverse().ToArray());
                    Console.WriteLine($"[pack][check] tile {yy:D2}_{xx:D2} MCNK[{firstIdxPresent}] abs={absPos} len={outMs.Length} read={read} token='{tokenFwd}'");
                    // Validate all MCIN entries for this tile (absolute)
                    for (int i = 0; i < 256; i++)
                    {
                        if (mcnkAbs[i] > 0)
                        {
                            long pos = mcnkAbs[i];
                            long cur2 = outMs.Position;
                            outMs.Position = pos;
                            Span<byte> t = new byte[4];
                            int r = outMs.Read(t);
                            outMs.Position = cur2;
                            var tokRaw = Encoding.ASCII.GetString(t);
                            var tokFwd = new string(tokRaw.Reverse().ToArray());
                            if (r != 4 || tokFwd != "MCNK")
                            {
                                Console.WriteLine($"[pack][BAD] tile {yy:D2}_{xx:D2} MCIN[{i}] abs={pos} read={r} token='{tokFwd}'");
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
            try
            {
                swTile.Stop();
                packTimingRows.Add($"{yy},{xx},{tileIndex},{swTile.Elapsed.TotalMilliseconds.ToString(CultureInfo.InvariantCulture)},");
            }
            catch { }
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
            try
            {
                packTimingRows.Add($"-1,-1,{tileIndex},,{'"'}{ex.Message.Replace("\n", " ").Replace("\r", " ")}{'"'}");
            }
            catch { }
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
        try
        {
            var outDir = Path.GetDirectoryName(outWdtPath) ?? ".";
            File.WriteAllLines(Path.Combine(outDir, "objects_written.csv"), objectsCsvLines, Encoding.UTF8);
            File.WriteAllLines(Path.Combine(outDir, "adt_objects_raw.csv"), rawObjectsLines, Encoding.UTF8);
            if (placementsMddfLines.Count > 1)
                File.WriteAllLines(Path.Combine(outDir, "placements_mddf.csv"), placementsMddfLines, Encoding.UTF8);
            if (placementsModfLines.Count > 1)
                File.WriteAllLines(Path.Combine(outDir, "placements_modf.csv"), placementsModfLines, Encoding.UTF8);
            if (placementsSkippedLines.Count > 1)
                File.WriteAllLines(Path.Combine(outDir, "placements_skipped.csv"), placementsSkippedLines, Encoding.UTF8);
            if (texturesCsv.Count > 1)
                File.WriteAllLines(Path.Combine(outDir, "textures.csv"), texturesCsv, Encoding.UTF8);
            if (usedM2ForExtract.Count > 0)
                File.WriteAllLines(Path.Combine(outDir, "m2_used.csv"), new [] { "name" }.Concat(usedM2ForExtract.OrderBy(s => s, StringComparer.OrdinalIgnoreCase)), Encoding.UTF8);
            if (usedWmoForExtract.Count > 0)
                File.WriteAllLines(Path.Combine(outDir, "wmo_used.csv"), new [] { "name" }.Concat(usedWmoForExtract.OrderBy(s => s, StringComparer.OrdinalIgnoreCase)), Encoding.UTF8);
            if (packTimingRows.Count > 0)
                File.WriteAllLines(Path.Combine(outDir, "pack_prep_timing.csv"), new [] { "tile_yy,tile_xx,index,ms_total,exception" }.Concat(packTimingRows), Encoding.UTF8);
            if (mcnkSizesCsv.Count > 1)
            {
                File.WriteAllLines(Path.Combine(outDir, "alpha_pack_mcnk_sizes.csv"), mcnkSizesCsv, Encoding.UTF8);
            }
        }
        catch { }

        try
        {
            var outDir = Path.GetDirectoryName(outWdtPath) ?? ".";
            if (opts?.ExtractAssets == true && !string.IsNullOrWhiteSpace(opts.AssetsSourceRoot) && texturesUsed.Count > 0)
            {
                var tilesetsRoot = Path.Combine(outDir, "assets", "tilesets");
                Directory.CreateDirectory(tilesetsRoot);
                var missing = new List<string>();
                foreach (var t in texturesUsed.OrderBy(s => s, StringComparer.OrdinalIgnoreCase))
                {
                    var rel = t.Replace('/', Path.DirectorySeparatorChar);
                    var srcPath = Path.Combine(opts.AssetsSourceRoot!, rel);
                    try
                    {
                        if (!File.Exists(srcPath)) { missing.Add(t); continue; }
                        var subRel = IsTilesetTexturePath(t) ? TilesetRelativeSubpath(t).Replace('/', Path.DirectorySeparatorChar) : rel;
                        var outPath = Path.Combine(tilesetsRoot, subRel);
                        Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? tilesetsRoot);
                        File.Copy(srcPath, outPath, true);
                    }
                    catch { missing.Add(t); }
                }
                if (missing.Count > 0)
                {
                    File.WriteAllLines(Path.Combine(outDir, "textures_missing.csv"), new [] { "texture" }.Concat(missing), Encoding.UTF8);
                }
            }
            if (mcnkSizesCsv.Count > 1)
            {
                File.WriteAllLines(Path.Combine(outDir, "alpha_pack_mcnk_sizes.csv"), mcnkSizesCsv, Encoding.UTF8);
            }
        }
        catch { }

    if (!string.IsNullOrWhiteSpace(opts?.ExportMccvDir))
    {
        Console.WriteLine($"[pack] MCCV images exported: {mccvExported}");
        Console.WriteLine($"[pack] MCCV headers present: {mccvHeaders}");
    }
    try
    {
        var objCsvPath = Path.Combine(Path.GetDirectoryName(outWdtPath) ?? ".", "objects_written.csv");
        File.WriteAllLines(objCsvPath, objectsCsvLines, Encoding.UTF8);
    }
    catch { }

    }

    public static void WriteMonolithicFromArchive(IArchiveSource src, string mapName, string outWdtPath, LkToAlphaOptions? opts = null)
    {
        if (src is null) throw new ArgumentNullException(nameof(src));
        if (string.IsNullOrWhiteSpace(mapName)) throw new ArgumentException("mapName required");
        if (string.IsNullOrWhiteSpace(outWdtPath)) throw new ArgumentException("outWdtPath required");
        Directory.CreateDirectory(Path.GetDirectoryName(outWdtPath) ?? ".");

        bool verbose = opts?.Verbose == true;
        var usedM2ForExtract = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var usedWmoForExtract = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

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
        var wdtMissingTiles = new List<string>();
        if (tiles.Count == 0)
        {
            try
            {
                var wdtVPath = $"world/maps/{mapName}/{mapName}.wdt";
                if (src.FileExists(wdtVPath))
                {
                    using var ws = src.OpenFile(wdtVPath);
                    using var wms = new MemoryStream(); ws.CopyTo(wms); var wdtBytes = wms.ToArray();
                    int iMain = FindFourCC(wdtBytes, "MAIN");
                    if (iMain >= 0)
                    {
                        int size = BitConverter.ToInt32(wdtBytes, iMain + 4);
                        int dataStart = iMain + 8;
                        int len = Math.Max(0, Math.Min(size, wdtBytes.Length - dataStart));
                        int entrySizeApprox = GridTiles > 0 ? (len / GridTiles) : 0;
                        int step = entrySizeApprox >= 16 ? 16 : entrySizeApprox >= 8 ? 8 : entrySizeApprox;
                        if (step >= 8)
                        {
                            for (int idx = 0; idx < GridTiles; idx++)
                            {
                                int off = dataStart + idx * step;
                                bool present = false;
                                for (int b = 0; b < step; b++) { if (wdtBytes[off + b] != 0) { present = true; break; } }
                                if (present)
                                {
                                    int yy = idx / 64; int xx = idx % 64;
                                    var vp = $"world/maps/{mapName}/{mapName}_{yy}_{xx}.adt";
                                    if (src.FileExists(vp)) tiles.Add((yy, xx, vp)); else wdtMissingTiles.Add(vp);
                                }
                            }
                            if (verbose) Console.WriteLine($"[pack] wdt-present tiles found: {tiles.Count}, missing: {wdtMissingTiles.Count}");
                        }
                    }
                }
            }
            catch { }
        }

        // Texture harvesting across tiles (archive mode only)
        var texturesUsed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var texturesCsv = new List<string> { "tile_yy,tile_xx,chunk_index,texture,source" };

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
                var keptM2 = gate.FilterNames(m2List.Select(NormalizeAssetName), out var droppedM2);
                var keptWmo = gate.FilterNames(wmoList.Select(NormalizeAssetName), out var droppedWmo);
                var outDir2 = Path.GetDirectoryName(outWdtPath) ?? ".";
                var dropCsv2 = Path.Combine(outDir2, "dropped_assets.csv");
                AssetGate.WriteDropReport(dropCsv2, droppedM2, droppedWmo);
                try
                {
                    var keptCsv2 = Path.Combine(outDir2, "kept_assets.csv");
                    using var sw2 = new StreamWriter(keptCsv2, false, Encoding.UTF8);
                    sw2.WriteLine("type,path");
                    foreach (var p in keptM2) sw2.WriteLine($"m2,{p}");
                    foreach (var p in keptWmo) sw2.WriteLine($"wmo,{p}");
                }
                catch { }
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
        long mdnmStart = outMs.Position; var mdnmData2 = BuildMdnmData(m2List); var mdnm = new Chunk("MDNM", mdnmData2.Length, mdnmData2); outMs.Write(mdnm.GetWholeChunk());
        long monmStart = outMs.Position; var monmData2 = BuildMonmData(wmoList); var monm = new Chunk("MONM", monmData2.Length, monmData2); outMs.Write(monm.GetWholeChunk());
        // MODF chunk (top-level WMO definitions) - must exist even if empty per Alpha spec
        outMs.Write(new Chunk("MODF", 0, Array.Empty<byte>()).GetWholeChunk());
        long savePos = outMs.Position; outMs.Position = mphdDataStart; Span<byte> mphdData = new byte[128]; mphdData.Clear();
        BitConverter.GetBytes(m2List.Count > 0 ? m2List.Count + 1 : 0).CopyTo(mphdData);
        BitConverter.GetBytes(checked((int)mdnmStart)).CopyTo(mphdData.Slice(4));
        // MONM count must include a trailing empty string when any names exist
        BitConverter.GetBytes(wmoList.Count > 0 ? wmoList.Count + 1 : 0).CopyTo(mphdData.Slice(8));
        BitConverter.GetBytes(checked((int)monmStart)).CopyTo(mphdData.Slice(12));
        outMs.Write(mphdData); outMs.Position = savePos;
        try
        {
            var outDirIdx2 = Path.GetDirectoryName(outWdtPath) ?? ".";
            WriteIndexCsv(outDirIdx2, mdnmIndex, monmIndex);
        }
        catch { }

        var mhdrAbs = Enumerable.Repeat(0, GridTiles).ToArray();
        var mhdrToFirst = Enumerable.Repeat(0, GridTiles).ToArray();
        var objectsCsvLines2 = new List<string> { "tile_yy,tile_xx,mddf_count,modf_count,first_mddf,first_modf" };
        var mcnkSizesCsv2 = new List<string> { "tile_yy,tile_xx,chunk_idx,nLayers,mcly_bytes,mcal_bytes,mcsh_bytes,mcse_bytes,mclq_bytes,nDoodadRefs,nMapObjRefs,mcrf_bytes" };
        var rawObjectsLines2 = new List<string> { "tile_yy,tile_xx,source,mddf_count,modf_count,first_mddf,first_modf" };
        var placementsMddfLines2 = new List<string> { "tile_yy,tile_xx,source,chunk_idx,local_index,global_index,name,uid,x,z,y,rx,ry,rz,scale,flags" };
        var placementsModfLines2 = new List<string> { "tile_yy,tile_xx,source,chunk_idx,local_index,global_index,name,uid,x,z,y,rx,ry,rz,bb_minx,bb_miny,bb_minz,bb_maxx,bb_maxy,bb_maxz,flags,doodad_set,name_set,scale" };
        var placementsSkippedLines2 = new List<string> { "tile_yy,tile_xx,source,type,local_index,name,reason" };
        long totalMddf2 = 0, totalModf2 = 0;
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
                byte[] texBytesFull = Array.Empty<byte>();
                System.Collections.Generic.List<int>? texMcnkOffsets = null;
                var texVPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}_tex.adt";
                if (src.FileExists(texVPath))
                {
                    using var ts = src.OpenFile(texVPath);
                    using var tms = new MemoryStream();
                    ts.CopyTo(tms);
                    texBytesFull = tms.ToArray();
                    mtexData = ExtractLkMtexData(texBytesFull);
                    int tMhOff = FindFourCC(texBytesFull, "MHDR");
                    if (tMhOff >= 0)
                    {
                        var tMhdr = new Mhdr(texBytesFull, tMhOff);
                        int tData = tMhOff + 8;
                        int tMcinRel = tMhdr.GetOffset(Mhdr.McinOffset);
                        if (tMcinRel > 0)
                        {
                            var tMcin = new Mcin(texBytesFull, tData + tMcinRel);
                            texMcnkOffsets = tMcin.GetMcnkOffsets();
                        }
                    }
                }
                if (mtexData.Length == 0) mtexData = ExtractLkMtexData(bytes);
                if (mtexData.Length == 0)
                {
                    var baseTexturePath = string.IsNullOrWhiteSpace(opts?.BaseTexture) ? "Tileset\\Generic\\Checkers.blp" : opts!.BaseTexture!;
                    mtexData = Encoding.ASCII.GetBytes(baseTexturePath + "\0");
                }
                long mtexPosition = outMs.Position; outMs.Write(new Chunk("MTEX", mtexData.Length, mtexData).GetWholeChunk());

                // Chunk-driven texture harvesting: parse MTEX names and scan MCLY per present chunk
                var mtexNames = ParseMtexNamesLocal(mtexData);
                // Build union of present chunks between root and _tex for inventory
                var presentTexScan = new List<int>(present);
                if (texMcnkOffsets != null)
                {
                    for (int i = 0; i < texMcnkOffsets.Count; i++)
                    {
                        if (texMcnkOffsets[i] > 0 && !presentTexScan.Contains(i)) presentTexScan.Add(i);
                    }
                }
                presentTexScan.Sort();
                bool preferTexForMtex = texBytesFull.Length > 0; // we used _tex MTEX when available
                foreach (var ci in presentTexScan)
                {
                    int rootOff = (ci < lkMcnkOffsets.Count) ? lkMcnkOffsets[ci] : 0;
                    int texOff = (texMcnkOffsets != null && ci < texMcnkOffsets.Count) ? texMcnkOffsets[ci] : 0;
                    List<int> ids = new List<int>();
                    string srcTag = string.Empty;
                    if (preferTexForMtex)
                    {
                        if (texOff > 0) { ids = ReadMCLYTextureIdsInChunk(texBytesFull, texOff); srcTag = "tex"; }
                        else if (rootOff > 0) { ids = ReadMCLYTextureIdsInChunk(bytes, rootOff); srcTag = "root"; }
                    }
                    else
                    {
                        if (rootOff > 0) { ids = ReadMCLYTextureIdsInChunk(bytes, rootOff); srcTag = "root"; }
                        else if (texOff > 0) { ids = ReadMCLYTextureIdsInChunk(texBytesFull, texOff); srcTag = "tex"; }
                    }
                    foreach (var tid in ids)
                    {
                        if (tid >= 0 && tid < mtexNames.Count)
                        {
                            var tex = NormalizeTexturePath(mtexNames[tid]);
                            if (!string.IsNullOrWhiteSpace(tex))
                            {
                                texturesUsed.Add(tex);
                                texturesCsv.Add($"{yy},{xx},{ci},{tex},{srcTag}");
                            }
                        }
                    }
                }

                // Build MDDF/MODF from _obj0/_obj1 if present (fallbacks: _obj then root)
                byte[] objBytes = Array.Empty<byte>();
                var objVPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}_obj.adt";
                if (src.FileExists(objVPath)) { using var os = src.OpenFile(objVPath); using var oms = new MemoryStream(); os.CopyTo(oms); objBytes = oms.ToArray(); }
                byte[] obj0Bytes = Array.Empty<byte>();
                var obj0VPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}_obj0.adt";
                if (src.FileExists(obj0VPath)) { using var os0 = src.OpenFile(obj0VPath); using var oms0 = new MemoryStream(); os0.CopyTo(oms0); obj0Bytes = oms0.ToArray(); }
                byte[] obj1Bytes = Array.Empty<byte>();
                var obj1VPath = $"world/maps/{mapName}/{mapName}_{yy}_{xx}_obj1.adt";
                if (src.FileExists(obj1VPath)) { using var os1 = src.OpenFile(obj1VPath); using var oms1 = new MemoryStream(); os1.CopyTo(oms1); obj1Bytes = oms1.ToArray(); }

                var mmdxRoot2 = BuildMmdxOrdered(bytes);
                var mwmoRoot2 = BuildMwmoOrdered(bytes);
                var mmdxObj2 = BuildMmdxOrdered(objBytes);
                var mwmoObj2 = BuildMwmoOrdered(objBytes);
                var mmdxObj0_2 = BuildMmdxOrdered(obj0Bytes);
                var mwmoObj0_2 = BuildMwmoOrdered(obj0Bytes);
                var mmdxObj1_2 = BuildMmdxOrdered(obj1Bytes);
                var mwmoObj1_2 = BuildMwmoOrdered(obj1Bytes);

                // Prepare per-chunk refs arrays to be filled during writes
                var doodadRefsByChunk = new System.Collections.Generic.List<int>[256];
                var wmoRefsByChunk = new System.Collections.Generic.List<int>[256];
                for (int i2 = 0; i2 < 256; i2++) { doodadRefsByChunk[i2] = new System.Collections.Generic.List<int>(); wmoRefsByChunk[i2] = new System.Collections.Generic.List<int>(); }

                byte[] mddfData;
                if (opts?.SkipM2 == true)
                {
                    mddfData = Array.Empty<byte>();
                }
                else
                {
                    using var msMddf2 = new MemoryStream();
                    int mddfBase = 0;
                    if (obj0Bytes.Length > 0) { var part = BuildMddfFromLk(obj0Bytes, mmdxObj0_2, mdnmIndex, placementsSkippedLines2, "obj0", yy, xx, doodadRefsByChunk, mddfBase, usedM2ForExtract, placementsMddfLines2); if (part.Length > 0) { msMddf2.Write(part, 0, part.Length); mddfBase += part.Length / 36; } }
                    if (objBytes.Length > 0) { var part = BuildMddfFromLk(objBytes, mmdxObj2, mdnmIndex, placementsSkippedLines2, "obj", yy, xx, doodadRefsByChunk, mddfBase, usedM2ForExtract, placementsMddfLines2); if (part.Length > 0) { msMddf2.Write(part, 0, part.Length); mddfBase += part.Length / 36; } }
                    { var part = BuildMddfFromLk(bytes, mmdxRoot2, mdnmIndex, placementsSkippedLines2, "root", yy, xx, doodadRefsByChunk, mddfBase, usedM2ForExtract, placementsMddfLines2); if (part.Length > 0) { msMddf2.Write(part, 0, part.Length); mddfBase += part.Length / 36; } }
                    mddfData = msMddf2.ToArray();
                }

                byte[] modfData;
                if (opts?.SkipWmos == true)
                {
                    modfData = Array.Empty<byte>();
                }
                else
                {
                    using var msModf2 = new MemoryStream();
                    int modfBase = 0;
                    if (obj1Bytes.Length > 0) { var part = BuildModfFromLk(obj1Bytes, mwmoObj1_2, monmIndex, placementsSkippedLines2, "obj1", yy, xx, wmoRefsByChunk, modfBase, usedWmoForExtract, placementsModfLines2); if (part.Length > 0) { msModf2.Write(part, 0, part.Length); modfBase += part.Length / 64; } }
                    if (objBytes.Length > 0) { var part = BuildModfFromLk(objBytes, mwmoObj2, monmIndex, placementsSkippedLines2, "obj", yy, xx, wmoRefsByChunk, modfBase, usedWmoForExtract, placementsModfLines2); if (part.Length > 0) { msModf2.Write(part, 0, part.Length); modfBase += part.Length / 64; } }
                    { var part = BuildModfFromLk(bytes, mwmoRoot2, monmIndex, placementsSkippedLines2, "root", yy, xx, wmoRefsByChunk, modfBase, usedWmoForExtract, placementsModfLines2); if (part.Length > 0) { msModf2.Write(part, 0, part.Length); modfBase += part.Length / 64; } }
                    modfData = msModf2.ToArray();
                }

                // Raw source counts
                try
                {
                    void AddRaw2(string sourceLbl, byte[] b, List<string> mmdxOrd, List<string> mwmoOrd)
                    {
                        int cm = GetMddfCount(b);
                        int cw = GetModfCount(b);
                        string fm = TryGetFirstMddfName(b, mmdxOrd);
                        string fw = TryGetFirstModfName(b, mwmoOrd);
                        rawObjectsLines2.Add($"{yy},{xx},{sourceLbl},{cm},{cw},{fm},{fw}");
                    }
                    AddRaw2("obj0", obj0Bytes, mmdxObj0_2, mwmoObj0_2);
                    AddRaw2("obj1", obj1Bytes, mmdxObj1_2, mwmoObj1_2);
                    AddRaw2("obj", objBytes, mmdxObj2, mwmoObj2);
                    AddRaw2("root", bytes, mmdxRoot2, mwmoRoot2);
                }
                catch { }

                // per-chunk refs were filled during MDDF/MODF writes above

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
                                if (local < 0 || local >= mmdxObj2.Count) continue;
                                string n = NormalizeAssetName(mmdxObj2[local]);
                                mdnmIndex.TryGetValue(n, out int gidx);
                                float wx = BitConverter.ToSingle(objBytes, p2 + 8);
                                float wy = BitConverter.ToSingle(objBytes, p2 + 12);
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
                                if (local < 0 || local >= mwmoObj2.Count) continue;
                                string n = NormalizeAssetName(mwmoObj2[local]);
                                monmIndex.TryGetValue(n, out int gidx);
                                float wx = BitConverter.ToSingle(objBytes, p2 + 8);
                                float wy = BitConverter.ToSingle(objBytes, p2 + 12);
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
                try
                {
                    int mddfCountW = mddfData.Length / 36;
                    int modfCountW = modfData.Length / 64;
                    totalMddf2 += mddfCountW; totalModf2 += modfCountW;
                    string firstM = string.Empty;
                    string firstW = string.Empty;
                    int mddfOff3 = FindFourCC(objBytes, "MDDF");
                    if (mddfOff3 >= 0)
                    {
                        int size3 = BitConverter.ToInt32(objBytes, mddfOff3 + 4);
                        int data3 = mddfOff3 + 8;
                        if (size3 >= 36)
                        {
                            int local3 = BitConverter.ToInt32(objBytes, data3 + 0);
                            if (local3 >= 0 && local3 < mmdxObj2.Count) firstM = NormalizeAssetName(mmdxObj2[local3]);
                        }
                    }
                    int modfOff3 = FindFourCC(objBytes, "MODF");
                    if (modfOff3 >= 0)
                    {
                        int size3 = BitConverter.ToInt32(objBytes, modfOff3 + 4);
                        int data3 = modfOff3 + 8;
                        if (size3 >= 64)
                        {
                            int local3 = BitConverter.ToInt32(objBytes, data3 + 0);
                            if (local3 >= 0 && local3 < mwmoObj2.Count) firstW = NormalizeAssetName(mwmoObj2[local3]);
                        }
                    }
                    objectsCsvLines2.Add($"{yy},{xx},{mddfCountW},{modfCountW},{firstM},{firstW}");
                }
                catch { }

                // Rebuild MCNKs with MCRF refs
                AlphaMcnkBuilder.OnChunkBuilt = (ciDiag, nLayersDiag, mclyLen, mcalLen, mcshLen, mcseLen, mclqLen, nDRefs, nWRefs, mcrfLen) =>
                {
                    mcnkSizesCsv2.Add(string.Join(',', new[]
                    {
                        yy.ToString(CultureInfo.InvariantCulture),
                        xx.ToString(CultureInfo.InvariantCulture),
                        ciDiag.ToString(CultureInfo.InvariantCulture),
                        nLayersDiag.ToString(CultureInfo.InvariantCulture),
                        mclyLen.ToString(CultureInfo.InvariantCulture),
                        mcalLen.ToString(CultureInfo.InvariantCulture),
                        mcshLen.ToString(CultureInfo.InvariantCulture),
                        mcseLen.ToString(CultureInfo.InvariantCulture),
                        mclqLen.ToString(CultureInfo.InvariantCulture),
                        nDRefs.ToString(CultureInfo.InvariantCulture),
                        nWRefs.ToString(CultureInfo.InvariantCulture),
                        mcrfLen.ToString(CultureInfo.InvariantCulture),
                    }));
                };
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
                        Span<byte> sig = new byte[4];
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
                            Span<byte> hdr = new byte[0x80];
                            outMs.Read(hdr);
                            int nDood = BitConverter.ToInt32(hdr.Slice(0x14, 4));
                            int offsRefs = BitConverter.ToInt32(hdr.Slice(0x24, 4));
                            int nWmo = BitConverter.ToInt32(hdr.Slice(0x3C, 4));
                            if ((nDood + nWmo) > 0 && offsRefs > 0)
                            {
                                outMs.Position = baseAbs + offsRefs;
                                Span<byte> tag = new byte[4]; outMs.Read(tag);
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
        try
        {
            var outDir = Path.GetDirectoryName(outWdtPath) ?? ".";
            File.WriteAllLines(Path.Combine(outDir, "objects_written.csv"), objectsCsvLines2, Encoding.UTF8);
            File.WriteAllLines(Path.Combine(outDir, "adt_objects_raw.csv"), rawObjectsLines2, Encoding.UTF8);
            if (placementsMddfLines2.Count > 1)
                File.WriteAllLines(Path.Combine(outDir, "placements_mddf.csv"), placementsMddfLines2, Encoding.UTF8);
            if (placementsModfLines2.Count > 1)
                File.WriteAllLines(Path.Combine(outDir, "placements_modf.csv"), placementsModfLines2, Encoding.UTF8);
            if (placementsSkippedLines2.Count > 1)
                File.WriteAllLines(Path.Combine(outDir, "placements_skipped.csv"), placementsSkippedLines2, Encoding.UTF8);
            if (texturesCsv.Count > 1)
            {
                File.WriteAllLines(Path.Combine(outDir, "textures.csv"), texturesCsv, Encoding.UTF8);
            }
            if (wdtMissingTiles.Count > 0)
            {
                File.WriteAllLines(Path.Combine(outDir, "tiles_missing.csv"), new [] { "tile" }.Concat(wdtMissingTiles), Encoding.UTF8);
            }
            if (opts?.ExtractAssets == true && texturesUsed.Count > 0)
            {
                var assetsRoot = string.IsNullOrWhiteSpace(opts.AssetsOut) ? Path.Combine(outDir, "assets") : opts.AssetsOut!;
                var tilesetsRoot = Path.Combine(assetsRoot, "tilesets");
                Directory.CreateDirectory(tilesetsRoot);
                var missing = new List<string>();
                foreach (var t in texturesUsed.OrderBy(s => s, StringComparer.OrdinalIgnoreCase))
                {
                    var vp = t.Replace('\\', '/');
                    try
                    {
                        if (!src.FileExists(vp)) { missing.Add(t); continue; }
                        using var sTex = src.OpenFile(vp);
                        var rel = IsTilesetTexturePath(t) ? TilesetRelativeSubpath(t) : vp; 
                        var outPath = Path.Combine(tilesetsRoot, rel.Replace('/', Path.DirectorySeparatorChar));
                        Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? tilesetsRoot);
                        using var fsTex = File.Create(outPath);
                        sTex.CopyTo(fsTex);
                    }
                    catch { missing.Add(t); }
                }
                if (missing.Count > 0)
                {
                    File.WriteAllLines(Path.Combine(outDir, "textures_missing.csv"), new [] { "texture" }.Concat(missing), Encoding.UTF8);
                }
            }
            // Extract models when requested and scope includes models; write missing list
            if (opts?.ExtractAssets == true && (string.Equals(opts.AssetScope, "textures+models", StringComparison.OrdinalIgnoreCase) || string.Equals(opts.AssetScope, "models", StringComparison.OrdinalIgnoreCase)))
            {
                var assetsRoot = string.IsNullOrWhiteSpace(opts.AssetsOut) ? Path.Combine(outDir, "assets") : opts.AssetsOut!;
                var modelsRoot = Path.Combine(assetsRoot, "models");
                Directory.CreateDirectory(modelsRoot);
                var missingModels = new List<string> { "type,path" };
                // copy M2/MDX
                foreach (var m in usedM2ForExtract.OrderBy(s => s, StringComparer.OrdinalIgnoreCase))
                {
                    var mdxPath = m.Replace('\\', '/');
                    var m2Path = (mdxPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)) ? mdxPath.Substring(0, mdxPath.Length - 3) + "m2" : mdxPath;
                    bool ok = false;
                    try
                    {
                        string attempt = src.FileExists(mdxPath) ? mdxPath : (src.FileExists(m2Path) ? m2Path : string.Empty);
                        if (!string.IsNullOrEmpty(attempt))
                        {
                            using var sModel = src.OpenFile(attempt);
                            var outRel = attempt.Replace('/', Path.DirectorySeparatorChar);
                            var outPath = Path.Combine(modelsRoot, outRel);
                            Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? modelsRoot);
                            using var fsOut = File.Create(outPath);
                            sModel.CopyTo(fsOut);
                            ok = true;
                        }
                    }
                    catch { ok = false; }
                    if (!ok) missingModels.Add($"m2,{mdxPath}");
                }
                // copy WMO
                foreach (var w in usedWmoForExtract.OrderBy(s => s, StringComparer.OrdinalIgnoreCase))
                {
                    var wmoPath = w.Replace('\\', '/');
                    bool ok = false;
                    try
                    {
                        if (src.FileExists(wmoPath))
                        {
                            using var sWmo = src.OpenFile(wmoPath);
                            var outRel = wmoPath.Replace('/', Path.DirectorySeparatorChar);
                            var outPath = Path.Combine(modelsRoot, outRel);
                            Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? modelsRoot);
                            using var fsOut = File.Create(outPath);
                            sWmo.CopyTo(fsOut);
                            ok = true;
                        }
                    }
                    catch { ok = false; }
                    if (!ok) missingModels.Add($"wmo,{wmoPath}");
                }
                if (missingModels.Count > 1)
                {
                    File.WriteAllLines(Path.Combine(outDir, "models_missing.csv"), missingModels, Encoding.UTF8);
                }

                // Optional legacy conversion pass (stubbed via Warcraft.NET wrappers)
                var convManifest = new List<string> { "type,source,destination,status,note" };
                if (opts?.ConvertModelsToLegacy == true)
                {
                    var legacyModelsRoot = Path.Combine(assetsRoot, "models_legacy");
                    Directory.CreateDirectory(legacyModelsRoot);
                    var modelConv = new WarcraftNetModelConverter();
                    foreach (var m in usedM2ForExtract.OrderBy(s => s, StringComparer.OrdinalIgnoreCase))
                    {
                        var mdxPath = m.Replace('\\', '/');
                        var m2Path = (mdxPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)) ? mdxPath.Substring(0, mdxPath.Length - 3) + "m2" : mdxPath;
                        string attempt = src.FileExists(m2Path) ? m2Path : (src.FileExists(mdxPath) ? mdxPath : string.Empty);
                        if (string.IsNullOrEmpty(attempt)) { convManifest.Add($"m2,{mdxPath},,missing,source not found"); continue; }
                        using var sIn = src.OpenFile(attempt);
                        var relNoExt = Path.ChangeExtension(attempt, null).Replace('/', Path.DirectorySeparatorChar);
                        var outRel = relNoExt + ".mdx";
                        var outPath = Path.Combine(legacyModelsRoot, outRel);
                        Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? legacyModelsRoot);
                        using var sOut = File.Create(outPath);
                        if (modelConv.TryConvertM2ToMdx(sIn, sOut, out var note)) convManifest.Add($"m2,{attempt},{outRel},converted,{note}");
                        else { sOut.Flush(); convManifest.Add($"m2,{attempt},{outRel},passthrough,{note}"); }
                    }
                }
                if (opts?.ConvertWmosToLegacy == true)
                {
                    var legacyWmosRoot = Path.Combine(assetsRoot, "wmos_legacy");
                    Directory.CreateDirectory(legacyWmosRoot);
                    var wmoConv = new WarcraftNetWmoConverter();
                    foreach (var w in usedWmoForExtract.OrderBy(s => s, StringComparer.OrdinalIgnoreCase))
                    {
                        var wmoPath = w.Replace('\\', '/');
                        if (!src.FileExists(wmoPath)) { convManifest.Add($"wmo,{wmoPath},,missing,source not found"); continue; }
                        using var sIn = src.OpenFile(wmoPath);
                        var outRel = wmoPath.Replace('/', Path.DirectorySeparatorChar); // keep .wmo
                        var outPath = Path.Combine(legacyWmosRoot, outRel);
                        Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? legacyWmosRoot);
                        using var sOut = File.Create(outPath);
                        if (wmoConv.TryConvertWmoToV14(sIn, sOut, out var note)) convManifest.Add($"wmo,{wmoPath},{outRel},converted,{note}");
                        else { sOut.Flush(); convManifest.Add($"wmo,{wmoPath},{outRel},passthrough,{note}"); }
                    }
                }
                if (convManifest.Count > 1)
                {
                    File.WriteAllLines(Path.Combine(outDir, "conversion_manifest.csv"), convManifest, Encoding.UTF8);
                }
            }

            if ((m2List.Count + wmoList.Count) > 0 && (totalMddf2 + totalModf2) == 0)
            {
                Console.WriteLine("[warn] MDNM/MONM non-empty but wrote zero placements (MDDF/MODF=0). Check name-index mapping and gating.");
            }
        }
        catch { }
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
        // CRITICAL: Alpha client reads MDNM chunk at the offset in MPHD.
        // Even if no M2 names exist, we must write a valid chunk with at least a trailing null.
        if (m2Names == null || m2Names.Count == 0)
            return new byte[] { 0 }; // Single null byte = empty string list terminator

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
        // CRITICAL: Alpha client ALWAYS reads MONM chunk at the offset in MPHD.
        // Even if no WMO names exist, we must write a valid chunk with at least a trailing null.
        if (wmoNames == null || wmoNames.Count == 0)
            return new byte[] { 0 }; // Single null byte = empty string list terminator

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

    // Helpers for texture harvesting (archive path)
    private static List<string> ParseMtexNamesLocal(byte[] data)
    {
        var list = new List<string>();
        if (data == null || data.Length == 0) return list;
        int i = 0;
        while (i < data.Length)
        {
            int j = i;
            while (j < data.Length && data[j] != 0) j++;
            if (j > i)
            {
                var s = Encoding.ASCII.GetString(data, i, j - i);
                list.Add(s);
            }
            i = j + 1;
        }
        return list;
    }

    private static string NormalizeTexturePath(string p)
    {
        if (string.IsNullOrWhiteSpace(p)) return string.Empty;
        return p.Replace('\\', '/');
    }

    private static bool IsTilesetTexturePath(string p)
    {
        if (string.IsNullOrWhiteSpace(p)) return false;
        var s = p.Replace('\\', '/');
        return s.IndexOf("tileset/", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static string TilesetRelativeSubpath(string p)
    {
        var s = p.Replace('\\', '/');
        int idx = s.IndexOf("tileset/", StringComparison.OrdinalIgnoreCase);
        if (idx >= 0)
        {
            return s.Substring(idx + "tileset/".Length);
        }
        return Path.GetFileName(s);
    }

    private static List<int> ReadMCLYTextureIdsInChunk(byte[] adtBytes, int mcNkOffset)
    {
        var ids = new List<int>();
        if (adtBytes == null || mcNkOffset < 0 || mcNkOffset + 8 > adtBytes.Length) return ids;
        int mcnkSize = BitConverter.ToInt32(adtBytes, mcNkOffset + 4);
        int subStart = mcNkOffset + 8 + 0x80; // after MCNK header
        int subEnd = Math.Min(adtBytes.Length, mcNkOffset + 8 + Math.Max(0, mcnkSize));
        for (int p = subStart; p + 8 <= subEnd;)
        {
            string fcc = Encoding.ASCII.GetString(adtBytes, p, 4);
            int size = BitConverter.ToInt32(adtBytes, p + 4);
            int dataStart = p + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > subEnd || next <= p) break;
            if (fcc == "YLCM") // MCLY reversed
            {
                int layers = size / 16; // LK MCLY entry size
                for (int i = 0; i < layers; i++)
                {
                    int baseOff = dataStart + i * 16;
                    if (baseOff + 4 <= adtBytes.Length)
                    {
                        int texId = unchecked((int)BitConverter.ToUInt32(adtBytes, baseOff + 0));
                        if (texId >= 0) ids.Add(texId);
                    }
                }
                break;
            }
            p = next;
        }
        return ids;
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
                        float wy = BitConverter.ToSingle(bytes, p + 12);
                        int cx = (int)Math.Floor((wx - tileMinX) / CHUNK);
                        int cy = (int)Math.Floor((wy - tileMinY) / CHUNK);
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
                        float wy = BitConverter.ToSingle(bytes, p + 12);
                        int cx = (int)Math.Floor((wx - tileMinX) / CHUNK);
                        int cy = (int)Math.Floor((wy - tileMinY) / CHUNK);
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

    private static byte[] BuildMddfFromLk(byte[] bytes, List<string> mmdxOrdered, Dictionary<string, int> mdnmIndex, List<string>? skipped = null, string? source = null, int tileYY = -1, int tileXX = -1, System.Collections.Generic.List<int>[]? perChunkRefs = null, int baseIndex = 0, HashSet<string>? usedM2Out = null, List<string>? placementsOut = null)
    {
        if (bytes == null || bytes.Length < 8) return Array.Empty<byte>();
        int mddf = FindFourCC(bytes, "MDDF");
        if (mddf < 0) return Array.Empty<byte>();
        int size = BitConverter.ToInt32(bytes, mddf + 4);
        int data = mddf + 8;
        const int entry = 36;
        int count = Math.Max(0, Math.Min(size / entry, 1_000_000));
        using var ms = new MemoryStream();
        int written = 0; // number of MDDF records written by this function
        for (int i = 0; i < count; i++)
        {
            int p = data + i * entry;
            if (p + entry > bytes.Length) break;
            int localIdx = BitConverter.ToInt32(bytes, p + 0);
            if (localIdx < 0 || localIdx >= mmdxOrdered.Count)
            {
                skipped?.Add($"{tileYY},{tileXX},{source},mddf,{localIdx},,out_of_range");
                continue;
            }
            string name = NormalizeAssetName(mmdxOrdered[localIdx]);
            if (string.IsNullOrEmpty(name)) { skipped?.Add($"{tileYY},{tileXX},{source},mddf,{localIdx},,empty_name"); continue; }
            if (!mdnmIndex.TryGetValue(name, out int globalIdx)) { skipped?.Add($"{tileYY},{tileXX},{source},mddf,{localIdx},{name},not_in_global_names"); continue; }
            usedM2Out?.Add(name);
            int uniqueId = BitConverter.ToInt32(bytes, p + 4);
            float posX = BitConverter.ToSingle(bytes, p + 8);
            float posZ = BitConverter.ToSingle(bytes, p + 12);
            float posY = BitConverter.ToSingle(bytes, p + 16);
            // Convert LK global (server) coords to Alpha centered coords
            // LK & Alpha: plane X/Y, height Z. Alpha is centered; LK is absolute.
            const float TILESIZE = 533.33333f; const float WORLD_BASE = 32f * TILESIZE;
            float rotX = BitConverter.ToSingle(bytes, p + 20);
            float rotY = BitConverter.ToSingle(bytes, p + 24);
            float rotZ = BitConverter.ToSingle(bytes, p + 28);
            ushort scale = BitConverter.ToUInt16(bytes, p + 32);
            ushort flags = BitConverter.ToUInt16(bytes, p + 34);
            ms.Write(BitConverter.GetBytes(globalIdx));
            ms.Write(BitConverter.GetBytes(uniqueId));
            // Alpha MDDF stores position as X, Z(height), Y in centered world coordinates
            ms.Write(BitConverter.GetBytes(posX));
            ms.Write(BitConverter.GetBytes(posZ)); // Z is height
            ms.Write(BitConverter.GetBytes(posY));
            ms.Write(BitConverter.GetBytes(rotX));
            ms.Write(BitConverter.GetBytes(rotY));
            ms.Write(BitConverter.GetBytes(rotZ));
            ms.Write(BitConverter.GetBytes(scale));
            ms.Write(BitConverter.GetBytes(flags));

            // Populate per-chunk refs using planar world coords; store MDDF record index (baseIndex + written)
            if (perChunkRefs != null && tileYY >= 0 && tileXX >= 0)
            {
                const float CHUNK = TILESIZE / 16f;
                float tileMinX = 32f * TILESIZE - (tileXX + 1) * TILESIZE;
                float tileMinY = 32f * TILESIZE - (tileYY + 1) * TILESIZE;
                float localX = posX - tileMinX;
                float localY = posY - tileMinY;
                int cx = (int)Math.Floor(localX / CHUNK);
                int cy = (int)Math.Floor(localY / CHUNK);
                if (cx >= 0 && cx < 16 && cy >= 0 && cy < 16)
                {
                    int idx = cy * 16 + cx;
                    perChunkRefs[idx].Add(baseIndex + written);
                }
            }

            // Append placement row
            if (placementsOut != null)
            {
                int chunkIdx = -1;
                if (tileYY >= 0 && tileXX >= 0)
                {
                    const float CHUNK2 = TILESIZE / 16f;
                    float tMinX = 32f * TILESIZE - (tileXX + 1) * TILESIZE;
                    float tMinY = 32f * TILESIZE - (tileYY + 1) * TILESIZE;
                    float localX2 = posX - tMinX;
                    float localY2 = posY - tMinY;
                    int cx2 = (int)Math.Floor(localX2 / CHUNK2);
                    int cy2 = (int)Math.Floor(localY2 / CHUNK2);
                    if (cx2 >= 0 && cx2 < 16 && cy2 >= 0 && cy2 < 16) chunkIdx = cy2 * 16 + cx2;
                }
                placementsOut.Add(string.Join(',', new[]
                {
                    tileYY.ToString(CultureInfo.InvariantCulture),
                    tileXX.ToString(CultureInfo.InvariantCulture),
                    source ?? string.Empty,
                    chunkIdx.ToString(CultureInfo.InvariantCulture),
                    localIdx.ToString(CultureInfo.InvariantCulture),
                    globalIdx.ToString(CultureInfo.InvariantCulture),
                    name,
                    uniqueId.ToString(CultureInfo.InvariantCulture),
                    posX.ToString(CultureInfo.InvariantCulture),
                    posZ.ToString(CultureInfo.InvariantCulture),
                    posY.ToString(CultureInfo.InvariantCulture),
                    rotX.ToString(CultureInfo.InvariantCulture),
                    rotY.ToString(CultureInfo.InvariantCulture),
                    rotZ.ToString(CultureInfo.InvariantCulture),
                    scale.ToString(CultureInfo.InvariantCulture),
                    flags.ToString(CultureInfo.InvariantCulture)
                }));
            }
            written++;
        }
        return ms.ToArray();
    }

    private static byte[] BuildModfFromLk(byte[] bytes, List<string> mwmoOrdered, Dictionary<string, int> monmIndex, List<string>? skipped = null, string? source = null, int tileYY = -1, int tileXX = -1, System.Collections.Generic.List<int>[]? perChunkRefs = null, int baseIndex = 0, HashSet<string>? usedWmoOut = null, List<string>? placementsOut = null)
    {
        if (bytes == null || bytes.Length < 8) return Array.Empty<byte>();
        int modf = FindFourCC(bytes, "MODF");
        if (modf < 0) return Array.Empty<byte>();
        int size = BitConverter.ToInt32(bytes, modf + 4);
        int data = modf + 8;
        const int entry = 64;
        int count = Math.Max(0, Math.Min(size / entry, 1_000_000));
        using var ms = new MemoryStream();
        int written = 0; // number of MODF records written by this function
        for (int i = 0; i < count; i++)
        {
            int p = data + i * entry;
            if (p + entry > bytes.Length) break;
            int localIdx = BitConverter.ToInt32(bytes, p + 0);
            if (localIdx < 0 || localIdx >= mwmoOrdered.Count)
            {
                skipped?.Add($"{tileYY},{tileXX},{source},modf,{localIdx},,out_of_range");
                continue;
            }
            string name = NormalizeAssetName(mwmoOrdered[localIdx]);
            if (string.IsNullOrEmpty(name)) { skipped?.Add($"{tileYY},{tileXX},{source},modf,{localIdx},,empty_name"); continue; }
            if (!monmIndex.TryGetValue(name, out int globalIdx)) { skipped?.Add($"{tileYY},{tileXX},{source},modf,{localIdx},{name},not_in_global_names"); continue; }
            usedWmoOut?.Add(name);
            int uniqueId = BitConverter.ToInt32(bytes, p + 4);
            float posX = BitConverter.ToSingle(bytes, p + 8);
            float posZ = BitConverter.ToSingle(bytes, p + 12);
            float posY = BitConverter.ToSingle(bytes, p + 16);
            // Convert LK global (server) coords to Alpha centered coords (plane X/Y, height Z)
            const float TILESIZE_M = 533.33333f;
            float rotX = BitConverter.ToSingle(bytes, p + 20);
            float rotY = BitConverter.ToSingle(bytes, p + 24);
            float rotZ = BitConverter.ToSingle(bytes, p + 28);
            // extents 0x20..0x3F
            ms.Write(BitConverter.GetBytes(globalIdx));
            ms.Write(BitConverter.GetBytes(uniqueId));
            // Write position as X, Z, Y in centered world coordinates
            ms.Write(BitConverter.GetBytes(posX));
            ms.Write(BitConverter.GetBytes(posZ));
            ms.Write(BitConverter.GetBytes(posY));
            ms.Write(BitConverter.GetBytes(rotX));
            ms.Write(BitConverter.GetBytes(rotY));
            ms.Write(BitConverter.GetBytes(rotZ));
            // extents: center X/Y only; Z unchanged
            float lkMinX = BitConverter.ToSingle(bytes, p + 32);
            float lkMinZ = BitConverter.ToSingle(bytes, p + 36);
            float lkMinY = BitConverter.ToSingle(bytes, p + 40);
            float lkMaxX = BitConverter.ToSingle(bytes, p + 44);
            float lkMaxZ = BitConverter.ToSingle(bytes, p + 48);
            float lkMaxY = BitConverter.ToSingle(bytes, p + 52);
            float cMinX = lkMinX;
            float cMinY = lkMinY;
            float cMinZ = lkMinZ;
            float cMaxX = lkMaxX;
            float cMaxY = lkMaxY;
            float cMaxZ = lkMaxZ;
            ms.Write(BitConverter.GetBytes(cMinX));
            ms.Write(BitConverter.GetBytes(cMinY));
            ms.Write(BitConverter.GetBytes(cMinZ));
            ms.Write(BitConverter.GetBytes(cMaxX));
            ms.Write(BitConverter.GetBytes(cMaxY));
            ms.Write(BitConverter.GetBytes(cMaxZ));
            ushort flags = BitConverter.ToUInt16(bytes, p + 56);
            ushort doodadSet = BitConverter.ToUInt16(bytes, p + 58);
            ushort nameSet = BitConverter.ToUInt16(bytes, p + 60);
            ushort scale = BitConverter.ToUInt16(bytes, p + 62);
            ms.Write(BitConverter.GetBytes(flags));
            ms.Write(BitConverter.GetBytes(doodadSet));
            ms.Write(BitConverter.GetBytes(nameSet));
            ms.Write(BitConverter.GetBytes(scale));

            // Populate per-chunk refs using planar world coords; store MODF record index (baseIndex + written)
            if (perChunkRefs != null && tileYY >= 0 && tileXX >= 0)
            {
                const float CHUNK_ABS = TILESIZE_M / 16f;
                float tileMinX = 32f * TILESIZE_M - (tileXX + 1) * TILESIZE_M;
                float tileMinY = 32f * TILESIZE_M - (tileYY + 1) * TILESIZE_M;
                float localX = posX - tileMinX;
                float localY = posY - tileMinY;
                int cx = (int)Math.Floor(localX / CHUNK_ABS);
                int cy = (int)Math.Floor(localY / CHUNK_ABS);
                if (cx >= 0 && cx < 16 && cy >= 0 && cy < 16)
                {
                    int idx = cy * 16 + cx;
                    perChunkRefs[idx].Add(baseIndex + written);
                }
            }

            if (placementsOut != null)
            {
                int chunkIdx = -1;
                if (tileYY >= 0 && tileXX >= 0)
                {
                    const float CHUNK_ABS2 = TILESIZE_M / 16f;
                    float tMinX = 32f * TILESIZE_M - (tileXX + 1) * TILESIZE_M;
                    float tMinY = 32f * TILESIZE_M - (tileYY + 1) * TILESIZE_M;
                    float localX2 = posX - tMinX;
                    float localY2 = posY - tMinY;
                    int cx2 = (int)Math.Floor(localX2 / CHUNK_ABS2);
                    int cy2 = (int)Math.Floor(localY2 / CHUNK_ABS2);
                    if (cx2 >= 0 && cx2 < 16 && cy2 >= 0 && cy2 < 16) chunkIdx = cy2 * 16 + cx2;
                }
                float bbMinXc = cMinX;
                float bbMinYc = cMinY;
                float bbMinZc = cMinZ;
                float bbMaxXc = cMaxX;
                float bbMaxYc = cMaxY;
                float bbMaxZc = cMaxZ;
                placementsOut.Add(string.Join(',', new[]
                {
                    tileYY.ToString(CultureInfo.InvariantCulture),
                    tileXX.ToString(CultureInfo.InvariantCulture),
                    source ?? string.Empty,
                    chunkIdx.ToString(CultureInfo.InvariantCulture),
                    localIdx.ToString(CultureInfo.InvariantCulture),
                    globalIdx.ToString(CultureInfo.InvariantCulture),
                    name,
                    uniqueId.ToString(CultureInfo.InvariantCulture),
                    posX.ToString(CultureInfo.InvariantCulture),
                    posZ.ToString(CultureInfo.InvariantCulture),
                    posY.ToString(CultureInfo.InvariantCulture),
                    rotX.ToString(CultureInfo.InvariantCulture),
                    rotY.ToString(CultureInfo.InvariantCulture),
                    rotZ.ToString(CultureInfo.InvariantCulture),
                    bbMinXc.ToString(CultureInfo.InvariantCulture),
                    bbMinYc.ToString(CultureInfo.InvariantCulture),
                    bbMinZc.ToString(CultureInfo.InvariantCulture),
                    bbMaxXc.ToString(CultureInfo.InvariantCulture),
                    bbMaxYc.ToString(CultureInfo.InvariantCulture),
                    bbMaxZc.ToString(CultureInfo.InvariantCulture),
                    flags.ToString(CultureInfo.InvariantCulture),
                    doodadSet.ToString(CultureInfo.InvariantCulture),
                    nameSet.ToString(CultureInfo.InvariantCulture),
                    scale.ToString(CultureInfo.InvariantCulture)
                }));
            }
            written++;
        }
        return ms.ToArray();
    }

    private static void WriteIndexCsv(string outDir, Dictionary<string, int> mdnmIndex, Dictionary<string, int> monmIndex)
    {
        try
        {
            var path = Path.Combine(outDir, "mdnm_monm_index.csv");
            using var sw = new StreamWriter(path, false, Encoding.UTF8);
            sw.WriteLine("type,index,name");
            foreach (var kv in mdnmIndex.OrderBy(k => k.Value)) sw.WriteLine($"m2,{kv.Value},{kv.Key}");
            foreach (var kv in monmIndex.OrderBy(k => k.Value)) sw.WriteLine($"wmo,{kv.Value},{kv.Key}");
        }
        catch { }
    }

    private static int GetMddfCount(byte[] bytes)
    {
        if (bytes == null || bytes.Length < 12) return 0;
        int off = FindFourCC(bytes, "MDDF"); if (off < 0) return 0;
        int size = BitConverter.ToInt32(bytes, off + 4); return Math.Max(0, size / 36);
    }

    private static int GetModfCount(byte[] bytes)
    {
        if (bytes == null || bytes.Length < 12) return 0;
        int off = FindFourCC(bytes, "MODF"); if (off < 0) return 0;
        int size = BitConverter.ToInt32(bytes, off + 4); return Math.Max(0, size / 64);
    }

    private static string TryGetFirstMddfName(byte[] bytes, List<string> mmdxOrdered)
    {
        try
        {
            int off = FindFourCC(bytes, "MDDF"); if (off < 0) return string.Empty;
            int size = BitConverter.ToInt32(bytes, off + 4); if (size < 36) return string.Empty;
            int data = off + 8; int local = BitConverter.ToInt32(bytes, data + 0);
            if (local >= 0 && local < mmdxOrdered.Count) return NormalizeAssetName(mmdxOrdered[local]);
        }
        catch { }
        return string.Empty;
    }

    private static string TryGetFirstModfName(byte[] bytes, List<string> mwmoOrdered)
    {
        try
        {
            int off = FindFourCC(bytes, "MODF"); if (off < 0) return string.Empty;
            int size = BitConverter.ToInt32(bytes, off + 4); if (size < 64) return string.Empty;
            int data = off + 8; int local = BitConverter.ToInt32(bytes, data + 0);
            if (local >= 0 && local < mwmoOrdered.Count) return NormalizeAssetName(mwmoOrdered[local]);
        }
        catch { }
        return string.Empty;
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
        Span<byte> header = new byte[54];
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
        BitConverter.GetBytes((ushort)1).CopyTo(header.Slice(26)); // planes
        BitConverter.GetBytes((ushort)24).CopyTo(header.Slice(28)); // bpp
        // compression 0 at 30..33
        BitConverter.GetBytes(imageSize).CopyTo(header.Slice(34));
        // xppm/yppm and palette fields left zero
        fs.Write(header);
        fs.Write(bgrTopDown, 0, bgrTopDown.Length);
    }
}
