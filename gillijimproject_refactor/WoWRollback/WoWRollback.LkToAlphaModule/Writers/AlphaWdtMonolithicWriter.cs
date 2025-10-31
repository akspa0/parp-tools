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

namespace WoWRollback.LkToAlphaModule.Writers;

public sealed class AlphaWdtMonolithicWriter
{
    private const int GridTiles = 64 * 64; // 4096

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
        if (verbose)
        {
            Console.WriteLine($"[pack] root ADTs: {rootAdts.Count}");
            if (rootAdts.Count > 0)
            {
                Console.WriteLine($"[pack] first: {Path.GetFileName(rootAdts.First())}, last: {Path.GetFileName(rootAdts.Last())}");
            }
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
        
        // Read from each tile ADT (root and obj variants)
        foreach (var rootAdt in rootAdts)
        {
            var tileWmos = adtReader.ReadWmoNames(rootAdt);
            foreach (var name in tileWmos)
            {
                allWmoNames.Add(name);
            }
            
            var tileM2s = adtReader.ReadM2Names(rootAdt);
            foreach (var name in tileM2s)
            {
                allM2Names.Add(name);
            }

            // Also scan OBJ ADT if present (often carries additional names)
            var objAdt = Path.Combine(Path.GetDirectoryName(rootAdt)!,
                Path.GetFileNameWithoutExtension(rootAdt) + "_obj.adt");
            if (File.Exists(objAdt))
            {
                var objWmos = adtReader.ReadWmoNames(objAdt);
                foreach (var name in objWmos)
                {
                    allWmoNames.Add(name);
                }

                var objM2s = adtReader.ReadM2Names(objAdt);
                foreach (var name in objM2s)
                {
                    allM2Names.Add(name);
                }
            }
        }
        
        var wmoNames = allWmoNames.ToList();
        var m2Names = allM2Names.ToList();
        
        Console.WriteLine($"[pack] Collected {wmoNames.Count} unique WMO names:");
        foreach (var name in wmoNames)
        {
            Console.WriteLine($"  - {name}");
        }
        
        if (verbose)
        {
            Console.WriteLine($"[pack] Collected {m2Names.Count} unique M2 names from {rootAdts.Count} tiles");
        }

        // Asset gating against target listfile (e.g., 3.3.5)
        if (!string.IsNullOrWhiteSpace(opts?.TargetListfilePath) && File.Exists(opts!.TargetListfilePath))
        {
            try
            {
                var idx = ListfileIndex.Load(opts.TargetListfilePath!);
                var gate = new AssetGate(idx);
                var keptM2 = gate.FilterNames(m2Names, out var droppedM2);
                var keptWmo = gate.FilterNames(wmoNames, out var droppedWmo);

                if (opts.StrictTargetAssets)
                {
                    m2Names = keptM2.ToList();
                    wmoNames = keptWmo.ToList();
                    var dropCsv = Path.Combine(Path.GetDirectoryName(outWdtPath) ?? ".", "dropped_assets.csv");
                    AssetGate.WriteDropReport(dropCsv, droppedM2, droppedWmo);
                    Console.WriteLine($"[gate] Target listfile: kept M2={m2Names.Count}, dropped={droppedM2.Count}; kept WMO={wmoNames.Count}, dropped={droppedWmo.Count}");
                }
                else
                {
                    Console.WriteLine("[gate] StrictTargetAssets=false; gating report only (no drops)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[warn] Asset gating failed: {ex.Message}");
            }
        }

        using var ms = new MemoryStream();
        // Write MVER
        var mver = new Chunk("MVER", 4, BitConverter.GetBytes(18));
        ms.Write(mver.GetWholeChunk());

        // Write MPHD (Alpha expects 128 bytes). We'll patch its data after writing MDNM/MONM.
        long mphdStart = ms.Position; // letters position
        var mphd = new Chunk("MPHD", 128, new byte[128]);
        long mphdDataStart = mphdStart + 8; // letters+size then data
        var mphdWhole = mphd.GetWholeChunk();
        ms.Write(mphdWhole, 0, mphdWhole.Length);

        // Prepare MAIN placeholder (4096 * 16)
        var mainData = new byte[GridTiles * 16]; // zeros now
        var main = new Chunk("MAIN", mainData.Length, mainData);
        long mainStart = ms.Position; // we will patch its data later
        var mainWhole = main.GetWholeChunk();
        ms.Write(mainWhole, 0, mainWhole.Length);

        // MDNM then MONM must follow MAIN in Alpha order
        // Build from collected LK names (MMDX → MDNM, MWMO → MONM)
        var mdnmDataBytes = BuildMdnmData(m2Names);
        var monmDataBytes = BuildMonmData(wmoNames);

        long mdnmStart = ms.Position;
        var mdnm = new Chunk("MDNM", mdnmDataBytes.Length, mdnmDataBytes);
        ms.Write(mdnm.GetWholeChunk());

        long monmStart = ms.Position;
        var monm = new Chunk("MONM", monmDataBytes.Length, monmDataBytes);
        ms.Write(monm.GetWholeChunk());

        if (verbose)
        {
            Console.WriteLine($"[pack] MDNM bytes: {mdnmDataBytes.Length}, names: {m2Names.Count}");
            Console.WriteLine($"[pack] MONM bytes: {monmDataBytes.Length}, names: {wmoNames.Count}");
        }

        // Patch MPHD with counts and offsets (absolute file offsets to chunk FourCC)
        // struct SMMapHeader { uint32 nDoodadNames; uint32 offsDoodadNames; uint32 nMapObjNames; uint32 offsMapObjNames; uint8 pad[112]; };
        long savePos = ms.Position;
        ms.Position = mphdDataStart;
        Span<byte> mphdData = stackalloc byte[128];
        mphdData.Clear();
        BitConverter.GetBytes(m2Names.Count).CopyTo(mphdData);                  // nDoodadNames
        BitConverter.GetBytes(checked((int)mdnmStart)).CopyTo(mphdData.Slice(4));   // offsDoodadNames (absolute)
        BitConverter.GetBytes(wmoNames.Count).CopyTo(mphdData.Slice(8));            // nMapObjNames
        BitConverter.GetBytes(checked((int)monmStart)).CopyTo(mphdData.Slice(12));  // offsMapObjNames (absolute)
        // write patched data
        ms.Write(mphdData);
        // restore
        ms.Position = savePos;

        // Build tile segments and collect MHDR absolute offsets for MAIN
        var mhdrAbsoluteOffsets = Enumerable.Repeat(0, GridTiles).ToArray();
        var mhdrToFirstMcnkSizes = Enumerable.Repeat(0, GridTiles).ToArray();

        foreach (var rootAdt in rootAdts)
        {
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
                int tileIndex = xx * 64 + yy;

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

            // Prebuild Alpha MCNK bytes for present entries
            var alphaMcnkBytes = new byte[256][];
            var presentIndices = new List<int>(256);
            for (int i = 0; i < 256; i++)
            {
                int off = (i < lkMcnkOffsets.Count) ? lkMcnkOffsets[i] : 0;
                if (off > 0)
                {
                    alphaMcnkBytes[i] = AlphaMcnkBuilder.BuildFromLk(bytes, off, opts);
                    presentIndices.Add(i);

                    if (!string.IsNullOrWhiteSpace(opts?.ExportMccvDir))
                    {
                        try
                        {
                            ExportMccvIfPresent(bytes, off, Path.Combine(opts!.ExportMccvDir!, mapName), yy, xx, i);
                        }
                        catch { }
                    }
                }
                else
                {
                    alphaMcnkBytes[i] = Array.Empty<byte>();
                }
            }
            if (verbose)
            {
                Console.WriteLine($"[pack] tile {yy:D2}_{xx:D2}: mcnk present {presentIndices.Count}");
            }

            // Now we can compute absolute offsets and write MHDR + MCIN + MCNKs
            long mhdrAbsolute = ms.Position;
            // MAIN.offset points to MHDR start (letters)
            mhdrAbsoluteOffsets[tileIndex] = checked((int)(mhdrAbsolute));

            var mhdr = AlphaMhdrBuilder.BuildMhdrForTerrain();
            var mhdrWhole = mhdr.GetWholeChunk();
            ms.Write(mhdrWhole, 0, mhdrWhole.Length);

            // MCIN absolute offset comes after MHDR
            long mcinAbsolute = ms.Position;
            long mhdrDataStart = mhdrAbsolute + 8;
            
            // Prepare MTEX data: prefer LK tile TEX ADT MTEX; then root ADT; fallback to BaseTexture
            byte[] mtexData = Array.Empty<byte>();
            var rootDir = Path.GetDirectoryName(rootAdt)!;
            var baseName = Path.GetFileNameWithoutExtension(rootAdt);
            var texAdt = Path.Combine(rootDir, baseName + "_tex.adt");
            if (File.Exists(texAdt))
            {
                var texBytes = File.ReadAllBytes(texAdt);
                mtexData = ExtractLkMtexData(texBytes);
            }
            if (mtexData.Length == 0)
            {
                mtexData = ExtractLkMtexData(bytes);
            }
            if (mtexData.Length == 0)
            {
                var baseTexturePath = string.IsNullOrWhiteSpace(opts?.BaseTexture) ? "Tileset\\Generic\\Checkers.blp" : opts!.BaseTexture!;
                var mtexString = baseTexturePath + "\0";
                try { mtexData = Encoding.ASCII.GetBytes(mtexString); }
                catch (Exception ex) { throw new InvalidOperationException($"Failed to encode MTEX texture path '{baseTexturePath}': {ex.Message}", ex); }
            }

            // STEP 1: Write all chunks FIRST and track their actual positions
            // This ensures offsets always match reality
            
            // Write MCIN (placeholder, will rebuild later with correct MCNK positions)
            long mcinPosition = ms.Position;
            var mcinPlaceholder = AlphaMcinBuilder.BuildMcin(new int[256], new int[256]);
            var mcinWhole = mcinPlaceholder.GetWholeChunk();
            ms.Write(mcinWhole, 0, mcinWhole.Length);
            long mcinEndPosition = ms.Position;
            
            // Write MTEX
            long mtexPosition = ms.Position;
            var mtex = new Chunk("MTEX", mtexData.Length, mtexData);
            var mtexWhole = mtex.GetWholeChunk();
            ms.Write(mtexWhole, 0, mtexWhole.Length);
            long mtexEndPosition = ms.Position;
            
            // Write empty MDDF
            long mddfPosition = ms.Position;
            var mddf = new Chunk("MDDF", 0, Array.Empty<byte>());
            var mddfWhole = mddf.GetWholeChunk();
            ms.Write(mddfWhole, 0, mddfWhole.Length);
            long mddfEndPosition = ms.Position;
            
            // Write empty MODF
            long modfPosition = ms.Position;
            var modf = new Chunk("MODF", 0, Array.Empty<byte>());
            var modfWhole = modf.GetWholeChunk();
            ms.Write(modfWhole, 0, modfWhole.Length);
            long modfEndPosition = ms.Position;
            
            // First MCNK starts here
            long firstMcnkAbsolute = ms.Position;

            // STEP 2: Calculate MCNK positions now that we know where they start
            int[] mcnkAbs = new int[256];
            int[] mcnkSizes = new int[256];
            long cursor = firstMcnkAbsolute;
            for (int i = 0; i < 256; i++)
            {
                if (alphaMcnkBytes[i] is { Length: > 0 })
                {
                    mcnkAbs[i] = checked((int)cursor);
                    mcnkSizes[i] = alphaMcnkBytes[i].Length;
                    cursor += alphaMcnkBytes[i].Length;
                }
                else
                {
                    mcnkAbs[i] = 0;
                    mcnkSizes[i] = 0;
                }
            }

            // STEP 3: Now go back and patch MHDR with ACTUAL positions
            long savedPosition = ms.Position;
            
            // Calculate offsets relative to MHDR.data start
            // CRITICAL: ALL offsets point to chunk start (FourCC), not data!
            // This is the Alpha convention - verified against real 0.5.3 Kalidar WDT
            int offsTexRel = checked((int)(mtexPosition - mhdrDataStart)); // Point to MTEX FourCC
            int offsDooRel = checked((int)(mddfPosition - mhdrDataStart)); // Point to MDDF FourCC
            int offsMobRel = checked((int)(modfPosition - mhdrDataStart)); // Point to MODF FourCC
            
            // Debug logging (always on for now)
            Console.WriteLine($"[OFFSET-DEBUG] Tile {tileIndex}:");
            Console.WriteLine($"  mhdrDataStart: 0x{mhdrDataStart:X}");
            Console.WriteLine($"  MCIN: 0x{mcinPosition:X} (length: {mcinEndPosition - mcinPosition})");
            Console.WriteLine($"  MTEX: 0x{mtexPosition:X} (length: {mtexEndPosition - mtexPosition})");
            Console.WriteLine($"  MDDF: 0x{mddfPosition:X} (length: {mddfEndPosition - mddfPosition})");
            Console.WriteLine($"  MODF: 0x{modfPosition:X} (length: {modfEndPosition - modfPosition})");
            Console.WriteLine($"  First MCNK: 0x{firstMcnkAbsolute:X}");
            Console.WriteLine($"  Calculated offsTex: {offsTexRel}");
            Console.WriteLine($"  Calculated offsDoo: {offsDooRel}");
            Console.WriteLine($"  Calculated offsMob: {offsMobRel}");
            
            // Write Alpha MHDR structure (NO sizeInfo field!)
            // struct SMAreaHeader {
            //     uint32_t offsInfo;  // MCIN  [offset 0]
            //     uint32_t offsTex;   // MTEX  [offset 4]
            //     uint32_t sizeTex;           [offset 8]
            //     uint32_t offsDoo;   // MDDF  [offset 12]
            //     uint32_t sizeDoo;           [offset 16]
            //     uint32_t offsMob;   // MODF  [offset 20]
            //     uint32_t sizeMob;           [offset 24]
            //     uint8_t pad[36];            [offset 28-63]
            // };
            
            ms.Position = mhdrDataStart + 0;
            ms.Write(BitConverter.GetBytes(64)); // offsInfo - MCIN immediately follows 64-byte MHDR.data
            
            ms.Position = mhdrDataStart + 4;
            ms.Write(BitConverter.GetBytes(offsTexRel)); // offsTex
            
            ms.Position = mhdrDataStart + 8;
            ms.Write(BitConverter.GetBytes(mtexData.Length)); // sizeTex
            
            ms.Position = mhdrDataStart + 12;
            ms.Write(BitConverter.GetBytes(offsDooRel)); // offsDoo
            
            ms.Position = mhdrDataStart + 16;
            ms.Write(BitConverter.GetBytes(0)); // sizeDoo
            
            ms.Position = mhdrDataStart + 20;
            ms.Write(BitConverter.GetBytes(offsMobRel)); // offsMob
            
            ms.Position = mhdrDataStart + 24;
            ms.Write(BitConverter.GetBytes(0)); // sizeMob
            
            // STEP 4: Go back and rewrite MCIN with correct MCNK positions
            ms.Position = mcinPosition;
            var mcin = AlphaMcinBuilder.BuildMcin(mcnkAbs, mcnkSizes);
            mcinWhole = mcin.GetWholeChunk();
            ms.Write(mcinWhole, 0, mcinWhole.Length);
            
            // STEP 5: Restore position to write MCNKs
            ms.Position = savedPosition;
            
            // MAIN.size = (first MCNK absolute - MHDR start), or 0 if none
            mhdrToFirstMcnkSizes[tileIndex] = presentIndices.Count > 0 ? checked((int)(firstMcnkAbsolute - mhdrAbsolute)) : 0;

            // Write MCNK bytes in index order
            for (int i = 0; i < 256; i++)
            {
                var buf = alphaMcnkBytes[i];
                if (buf is { Length: > 0 })
                {
                    ms.Write(buf, 0, buf.Length);
                }
            }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to process tile '{Path.GetFileName(rootAdt)}': {ex.Message}", ex);
            }
        }

        // Patch MAIN data with MHDR absolute offsets
        // CRITICAL: mainStart points to MAIN chunk start (FourCC), header is already written
        // We only need to write the DATA portion, so seek to mainStart + 8
        ms.Position = mainStart + 8; // Skip the 8-byte header (FourCC + size)
        // Rebuild MAIN with collected offsets
        bool pointToData = opts?.MainPointToMhdrData ?? false;
        var patchedMain = AlphaMainBuilder.BuildMain(mhdrAbsoluteOffsets, mhdrToFirstMcnkSizes, pointToData);
        // Write only the data, not the whole chunk (which would include header again)
        ms.Write(patchedMain.Data, 0, patchedMain.Data.Length);
        if (verbose)
        {
            Console.WriteLine($"[pack] MAIN offset mode: {(pointToData ? "MHDR.data (+8)" : "MHDR letters")}");
        }

        // Flush to file
        File.WriteAllBytes(outWdtPath, ms.ToArray());
    }

    private static byte[] BuildMdnmData(List<string> m2Names)
    {
        if (m2Names == null || m2Names.Count == 0)
            return Array.Empty<byte>();

        using var ms = new MemoryStream();
        foreach (var name in m2Names)
        {
            var nameBytes = Encoding.ASCII.GetBytes(name);
            ms.Write(nameBytes, 0, nameBytes.Length);
            ms.WriteByte(0);
        }
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
            var nameBytes = Encoding.ASCII.GetBytes(name);
            ms.Write(nameBytes, 0, nameBytes.Length);
            ms.WriteByte(0); // null terminator
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

    // --- MCCV Export (optional debug) ---
    private static void ExportMccvIfPresent(byte[] adtBytes, int mcnkOffset, string baseOutDir, int tileY, int tileX, int chunkIdx)
    {
        const int ChunkLettersAndSize = 8;
        const int McnkHeaderSize = 0x80;
        if (mcnkOffset < 0 || mcnkOffset + ChunkLettersAndSize + McnkHeaderSize > adtBytes.Length) return;

        // MccvOffset is at byte 116 within the LK MCNK header
        int mccvOffsetInHeader = 116;
        int headerStart = mcnkOffset + ChunkLettersAndSize;
        int mccvRel = BitConverter.ToInt32(adtBytes, headerStart + mccvOffsetInHeader);
        if (mccvRel <= 0) return;

        int mccvChunkOffset = mcnkOffset + mccvRel;
        if (mccvChunkOffset < 0 || mccvChunkOffset + 8 > adtBytes.Length) return;

        Chunk mccvChunk;
        try { mccvChunk = new Chunk(adtBytes, mccvChunkOffset); }
        catch { return; }
        if (mccvChunk.GivenSize <= 0 || mccvChunk.Data.Length <= 0) return;

        // Build a 64x64 BGR image from MCCV vertex shading (use 9x9 outer grid, bilinear upsample)
        var bgr = BuildMccvBgrImage(mccvChunk.Data, 64, 64);
        if (bgr.Length == 0) return;

        var outDir = Path.Combine(baseOutDir, $"{tileY:D2}_{tileX:D2}");
        Directory.CreateDirectory(outDir);
        var outPath = Path.Combine(outDir, $"{tileY:D2}_{tileX:D2}_mcnk_{chunkIdx:D3}_mccv.bmp");
        try { SaveBmp24(outPath, 64, 64, bgr); } catch { }
    }

    private static byte[] BuildMccvBgrImage(byte[] mccvData, int width, int height)
    {
        // Expect 145 RGBA entries interleaved as rows: 9,8,9,8,... (like MCNR/MCVT)
        int expected = 145 * 4;
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
            }
            src += 9 * 4;
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
