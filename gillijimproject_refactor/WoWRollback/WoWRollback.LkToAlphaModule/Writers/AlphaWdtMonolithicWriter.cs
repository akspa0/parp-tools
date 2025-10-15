using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Security.Cryptography;
using WoWRollback.LkToAlphaModule;
using WoWRollback.LkToAlphaModule.Readers;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using WoWRollback.LkToAlphaModule.Builders;

namespace WoWRollback.LkToAlphaModule.Writers;

public sealed class AlphaWdtMonolithicWriter
{
    private const int GridTiles = 64 * 64; // 4096

    public void Pack(string lkWdtPath, string lkMapDir, string outWdtPath, string mapName, LkToAlphaOptions? opts = null)
    {
        bool verbose = opts?.Verbose == true;
        if (string.IsNullOrWhiteSpace(lkWdtPath)) throw new ArgumentException("lkWdtPath required");
        if (string.IsNullOrWhiteSpace(lkMapDir)) throw new ArgumentException("lkMapDir required");
        if (string.IsNullOrWhiteSpace(outWdtPath)) throw new ArgumentException("outWdtPath required");
        Directory.CreateDirectory(Path.GetDirectoryName(outWdtPath) ?? ".");

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

        // Read WMO names from source LK WDT
        var wdtReader = new LkWdtReader();
        var wmoNames = wdtReader.ReadWmoNames(lkWdtPath);
        
        // CRITICAL: Alpha client crashes if MONM is empty but nMapObjNames > 0
        // Add dummy WMO name if source has none to prevent client from reading garbage
        if (wmoNames.Count == 0)
        {
            wmoNames.Add(@"world\wmo\cameron.wmo");  // Test WMO from earliest rendering code
            if (verbose)
            {
                Console.WriteLine("[pack] No WMO names in source, adding dummy cameron.wmo");
            }
        }
        else if (verbose)
        {
            Console.WriteLine($"[pack] Found {wmoNames.Count} WMO names in source WDT");
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
        long mdnmStart = ms.Position;
        var mdnm = new Chunk("MDNM", 0, Array.Empty<byte>());
        ms.Write(mdnm.GetWholeChunk());
        
        // Write MONM with actual WMO names from source
        long monmStart = ms.Position;
        byte[] monmData = BuildMonmData(wmoNames);
        var monm = new Chunk("MONM", monmData.Length, monmData);
        ms.Write(monm.GetWholeChunk());
        // Some Alpha WDTs (e.g., RazorfenDowns) include a top-level empty MODF after MONM
        long topModfStart = ms.Position;
        var topModf = new Chunk("MODF", 0, Array.Empty<byte>());
        ms.Write(topModf.GetWholeChunk());

        // Patch MPHD with offsets to MDNM/MONM
        // struct SMMapHeader { uint32 nDoodadNames; uint32 offsDoodadNames; uint32 nMapObjNames; uint32 offsMapObjNames; uint8 pad[112]; };
        long savePos = ms.Position;
        ms.Position = mphdDataStart;
        Span<byte> mphdData = stackalloc byte[128];
        // nDoodadNames = 0
        BitConverter.GetBytes(0).CopyTo(mphdData);
        // offsDoodadNames = absolute offset to MDNM letters
        BitConverter.GetBytes(checked((int)mdnmStart)).CopyTo(mphdData.Slice(4));
        // nMapObjNames = actual count from source
        // Note: Alpha client counts by splitting on nulls, so "name\0" = 2 parts (name + empty)
        // We need to add 1 to match this behavior if we have any names
        int wmoCount = wmoNames.Count > 0 ? wmoNames.Count + 1 : 0;
        BitConverter.GetBytes(wmoCount).CopyTo(mphdData.Slice(8));
        // offsMapObjNames = absolute offset to MONM letters
        BitConverter.GetBytes(checked((int)monmStart)).CopyTo(mphdData.Slice(12));
        // write patched data
        ms.Write(mphdData);
        // restore
        ms.Position = savePos;

        // Build tile segments and collect MHDR absolute offsets for MAIN
        var mhdrAbsoluteOffsets = Enumerable.Repeat(0, GridTiles).ToArray();
        var mhdrToFirstMcnkSizes = Enumerable.Repeat(0, GridTiles).ToArray();

        foreach (var rootAdt in rootAdts)
        {
            // Parse tile indices from file name map_yy_xx.adt
            var file = Path.GetFileNameWithoutExtension(rootAdt);
            // Expected: <map>_YY_XX
            var parts = file.Split('_');
            if (parts.Length < 3) continue;
            if (!int.TryParse(parts[^2], out int yy)) continue;
            if (!int.TryParse(parts[^1], out int xx)) continue;
            int tileIndex = yy * 64 + xx;

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
            // Precompute chunk lengths for MCIN and MTEX to position first MCNK
            int mcinChunkLen = new Chunk("MCIN", 256 * 16, new byte[256 * 16]).GetWholeChunk().Length;
            // Minimal MTEX with one base texture entry
            // Default to a known Alpha-era texture the client should have
            var baseTexturePath = string.IsNullOrWhiteSpace(opts?.BaseTexture) ? "Tileset\\Generic\\Checkers.blp" : opts!.BaseTexture!;
            var mtexData = Encoding.ASCII.GetBytes(baseTexturePath + "\0");
            int mtexChunkLen = new Chunk("MTEX", mtexData.Length, mtexData).GetWholeChunk().Length;
            int mddfChunkLen = new Chunk("MDDF", 0, Array.Empty<byte>()).GetWholeChunk().Length; // typically 8
            int modfChunkLen = new Chunk("MODF", 0, Array.Empty<byte>()).GetWholeChunk().Length; // typically 8
            long firstMcnkAbsolute = mcinAbsolute + mcinChunkLen + mtexChunkLen + mddfChunkLen + modfChunkLen;

            // Compute MCIN entry absolute offsets (to MCNK letters) and sizes
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
            // Patch MHDR offsTex/sizeTex and offsDoo/sizeDoo and offsMob/sizeMob relative to MHDR.data
            if (presentIndices.Count >= 0)
            {
                long mhdrDataStart = mhdrAbsolute + 8;
                int offsTexRel = 64 + mcinChunkLen;
                long save = ms.Position;
                
                // CRITICAL: Write offsInfo field (offset to MCIN)
                // This field was missing and caused client crash!
                ms.Position = mhdrDataStart + 0; // offsInfo
                ms.Write(BitConverter.GetBytes(64)); // MCIN immediately follows 64-byte MHDR.data
                
                ms.Position = mhdrDataStart + 4; // offsTex
                ms.Write(BitConverter.GetBytes(offsTexRel));
                ms.Position = mhdrDataStart + 8; // sizeTex
                ms.Write(BitConverter.GetBytes(mtexData.Length));
                // offsDoo (MDDF)
                int offsDooRel = offsTexRel + mtexChunkLen;
                ms.Position = mhdrDataStart + 0x0C; // offsDoo
                ms.Write(BitConverter.GetBytes(offsDooRel));
                ms.Position = mhdrDataStart + 0x10; // sizeDoo
                ms.Write(BitConverter.GetBytes(0));
                // offsMob (MODF)
                int offsMobRel = offsDooRel + mddfChunkLen;
                ms.Position = mhdrDataStart + 0x14; // offsMob
                ms.Write(BitConverter.GetBytes(offsMobRel));
                ms.Position = mhdrDataStart + 0x18; // sizeMob
                ms.Write(BitConverter.GetBytes(0));
                ms.Position = save;
            }

            // MAIN.size = (first MCNK absolute - MHDR start), or 0 if none
            mhdrToFirstMcnkSizes[tileIndex] = presentIndices.Count > 0 ? checked((int)(firstMcnkAbsolute - mhdrAbsolute)) : 0;

            var mcin = AlphaMcinBuilder.BuildMcin(mcnkAbs, mcnkSizes);
            var mcinWhole = mcin.GetWholeChunk();
            ms.Write(mcinWhole, 0, mcinWhole.Length);

            // Write MTEX with one texture string
            var mtex = new Chunk("MTEX", mtexData.Length, mtexData);
            var mtexWhole = mtex.GetWholeChunk();
            ms.Write(mtexWhole, 0, mtexWhole.Length);

            // Write empty MDDF and MODF chunks
            var mddf = new Chunk("MDDF", 0, Array.Empty<byte>());
            ms.Write(mddf.GetWholeChunk());
            var modf = new Chunk("MODF", 0, Array.Empty<byte>());
            ms.Write(modf.GetWholeChunk());

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

        // Patch MAIN data with MHDR absolute offsets
        ms.Position = mainStart;
        // Rebuild MAIN with collected offsets
        bool pointToData = opts?.MainPointToMhdrData ?? false;
        var patchedMain = AlphaMainBuilder.BuildMain(mhdrAbsoluteOffsets, mhdrToFirstMcnkSizes, pointToData);
        var patchedMainWhole = patchedMain.GetWholeChunk();
        ms.Write(patchedMainWhole, 0, patchedMainWhole.Length);
        if (verbose)
        {
            Console.WriteLine($"[pack] MAIN offset mode: {(pointToData ? "MHDR.data (+8)" : "MHDR letters")}");
        }

        // Flush to file
        File.WriteAllBytes(outWdtPath, ms.ToArray());
    }

    private static byte[] BuildMonmData(List<string> wmoNames)
    {
        if (wmoNames == null || wmoNames.Count == 0)
            return Array.Empty<byte>();

        using var ms = new MemoryStream();
        foreach (var name in wmoNames)
        {
            var nameBytes = Encoding.UTF8.GetBytes(name);
            ms.Write(nameBytes, 0, nameBytes.Length);
            ms.WriteByte(0); // null terminator
        }
        return ms.ToArray();
    }

    private static int FindFourCC(byte[] buf, string forwardFourCC)
    {
        // On-disk bytes are reversed in our Chunk reader logic; here we scan for reversed letters
        string reversed = new string(new[] { forwardFourCC[3], forwardFourCC[2], forwardFourCC[1], forwardFourCC[0] });
        for (int i = 0; i + 8 <= buf.Length;)
        {
            string fcc = Encoding.ASCII.GetString(buf, i, 4);
            int size = BitConverter.ToInt32(buf, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (fcc == reversed) return i;
            if (dataStart + size > buf.Length) break;
            i = next;
        }
        return -1;
    }
}
