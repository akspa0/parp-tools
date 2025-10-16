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
        
        // Read from each tile ADT
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
        // NOTE: Alpha client doesn't support M2 names in WDT, keep MDNM empty
        // CRITICAL: Client EXPECTS these chunks to exist, even if empty
        // Write them with size=0 for terrain-only mode
        long mdnmStart = ms.Position;
        var mdnm = new Chunk("MDNM", 0, Array.Empty<byte>());
        ms.Write(mdnm.GetWholeChunk());
        
        long monmStart = ms.Position;
        var monm = new Chunk("MONM", 0, Array.Empty<byte>());
        ms.Write(monm.GetWholeChunk());
        
        // Some Alpha WDTs include empty MODF after MONM
        long topModfStart = ms.Position;
        var topModf = new Chunk("MODF", 0, Array.Empty<byte>());
        ms.Write(topModf.GetWholeChunk());
        
        if (verbose)
        {
            Console.WriteLine("[pack] Written empty MDNM/MONM/MODF for terrain-only mode");
        }

        // Patch MPHD to point to empty chunks
        // struct SMMapHeader { uint32 nDoodadNames; uint32 offsDoodadNames; uint32 nMapObjNames; uint32 offsMapObjNames; uint8 pad[112]; };
        long savePos = ms.Position;
        ms.Position = mphdDataStart;
        Span<byte> mphdData = stackalloc byte[128];
        mphdData.Clear();
        // Point to chunks but with count=0 since they're empty
        BitConverter.GetBytes(0).CopyTo(mphdData);      // nDoodadNames = 0
        BitConverter.GetBytes(checked((int)mdnmStart)).CopyTo(mphdData.Slice(4));  // offsDoodadNames = offset to MDNM
        BitConverter.GetBytes(0).CopyTo(mphdData.Slice(8));  // nMapObjNames = 0  
        BitConverter.GetBytes(checked((int)monmStart)).CopyTo(mphdData.Slice(12)); // offsMapObjNames = offset to MONM
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
            long mhdrDataStart = mhdrAbsolute + 8;
            
            // Prepare MTEX data
            var baseTexturePath = string.IsNullOrWhiteSpace(opts?.BaseTexture) ? "Tileset\\Generic\\Checkers.blp" : opts!.BaseTexture!;
            var mtexData = Encoding.ASCII.GetBytes(baseTexturePath + "\0");

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
            
            // Write offsInfo (offset to MCIN)
            ms.Position = mhdrDataStart + 0;
            ms.Write(BitConverter.GetBytes(64)); // MCIN immediately follows 64-byte MHDR.data
            
            // Write offsTex and sizeTex
            ms.Position = mhdrDataStart + 4;
            ms.Write(BitConverter.GetBytes(offsTexRel));
            ms.Position = mhdrDataStart + 8;
            ms.Write(BitConverter.GetBytes(mtexData.Length));
            
            // Write offsDoo and sizeDoo (MDDF)
            ms.Position = mhdrDataStart + 0x0C;
            ms.Write(BitConverter.GetBytes(offsDooRel));
            ms.Position = mhdrDataStart + 0x10;
            ms.Write(BitConverter.GetBytes(0));
            
            // Write offsMob and sizeMob (MODF)
            ms.Position = mhdrDataStart + 0x14;
            ms.Write(BitConverter.GetBytes(offsMobRel));
            ms.Position = mhdrDataStart + 0x18;
            ms.Write(BitConverter.GetBytes(0));
            
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
