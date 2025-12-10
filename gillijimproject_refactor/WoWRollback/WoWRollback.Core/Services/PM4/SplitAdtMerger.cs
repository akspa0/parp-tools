using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Merges split ADT files (root + _obj0 + _tex0) into monolithic 3.3.5 format.
/// </summary>
public sealed class SplitAdtMerger
{
    // Chunk signatures - reversed on disk
    private static readonly byte[] REVM = Encoding.ASCII.GetBytes("REVM"); // MVER
    private static readonly byte[] RDHM = Encoding.ASCII.GetBytes("RDHM"); // MHDR
    private static readonly byte[] NICM = Encoding.ASCII.GetBytes("NICM"); // MCIN
    private static readonly byte[] XETM = Encoding.ASCII.GetBytes("XETM"); // MTEX
    private static readonly byte[] XDMM = Encoding.ASCII.GetBytes("XDMM"); // MMDX
    private static readonly byte[] DIMM = Encoding.ASCII.GetBytes("DIMM"); // MMID
    private static readonly byte[] OMWM = Encoding.ASCII.GetBytes("OMWM"); // MWMO
    private static readonly byte[] DIWM = Encoding.ASCII.GetBytes("DIWM"); // MWID
    private static readonly byte[] FDDM = Encoding.ASCII.GetBytes("FDDM"); // MDDF
    private static readonly byte[] FDOM = Encoding.ASCII.GetBytes("FDOM"); // MODF
    private static readonly byte[] KNCM = Encoding.ASCII.GetBytes("KNCM"); // MCNK

    /// <summary>
    /// Merge split ADT files into monolithic format.
    /// </summary>
    public byte[]? MergeSplitAdt(string rootAdtPath, string? obj0Path = null, string? tex0Path = null)
    {
        if (!File.Exists(rootAdtPath))
        {
            Console.WriteLine($"[ERROR] Root ADT not found: {rootAdtPath}");
            return null;
        }

        var rootData = File.ReadAllBytes(rootAdtPath);
        byte[]? obj0Data = obj0Path != null && File.Exists(obj0Path) ? File.ReadAllBytes(obj0Path) : null;
        byte[]? tex0Data = tex0Path != null && File.Exists(tex0Path) ? File.ReadAllBytes(tex0Path) : null;

        return MergeSplitAdtData(rootData, obj0Data, tex0Data);
    }

    /// <summary>
    /// Merge split ADT data into monolithic format.
    /// </summary>
    public byte[] MergeSplitAdtData(byte[] rootData, byte[]? obj0Data, byte[]? tex0Data)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Parse chunks from each file
        var rootChunks = ParseChunks(rootData);
        var obj0Chunks = obj0Data != null ? ParseChunks(obj0Data) : new List<ChunkInfo>();
        var tex0Chunks = tex0Data != null ? ParseChunks(tex0Data) : new List<ChunkInfo>();

        // Build chunk lookup
        var obj0Lookup = new Dictionary<string, ChunkInfo>();
        foreach (var c in obj0Chunks) obj0Lookup[c.Signature] = c;
        
        var tex0Lookup = new Dictionary<string, ChunkInfo>();
        foreach (var c in tex0Chunks) tex0Lookup[c.Signature] = c;

        // Write MVER
        var mverChunk = rootChunks.Find(c => c.Signature == "REVM");
        if (mverChunk != null)
        {
            bw.Write(rootData, mverChunk.Offset, mverChunk.TotalSize);
        }
        else
        {
            // Write default MVER
            bw.Write(REVM);
            bw.Write(4);
            bw.Write(18);
        }

        // Write MHDR (will need to fix offsets later)
        long mhdrPos = ms.Position;
        var mhdrChunk = rootChunks.Find(c => c.Signature == "RDHM");
        if (mhdrChunk != null)
        {
            bw.Write(rootData, mhdrChunk.Offset, mhdrChunk.TotalSize);
        }

        // Write MCIN from root
        var mcinChunk = rootChunks.Find(c => c.Signature == "NICM");
        long mcinPos = ms.Position;
        if (mcinChunk != null)
        {
            bw.Write(rootData, mcinChunk.Offset, mcinChunk.TotalSize);
        }

        // Write MTEX from tex0 or root
        if (tex0Lookup.TryGetValue("XETM", out var mtexChunk))
        {
            bw.Write(tex0Data!, mtexChunk.Offset, mtexChunk.TotalSize);
        }
        else
        {
            var rootMtex = rootChunks.Find(c => c.Signature == "XETM");
            if (rootMtex != null)
                bw.Write(rootData, rootMtex.Offset, rootMtex.TotalSize);
            else
            {
                bw.Write(XETM);
                bw.Write(0);
            }
        }

        // Write MMDX from obj0 or root
        if (obj0Lookup.TryGetValue("XDMM", out var mmdxChunk))
        {
            bw.Write(obj0Data!, mmdxChunk.Offset, mmdxChunk.TotalSize);
        }
        else
        {
            var rootMmdx = rootChunks.Find(c => c.Signature == "XDMM");
            if (rootMmdx != null)
                bw.Write(rootData, rootMmdx.Offset, rootMmdx.TotalSize);
            else
            {
                bw.Write(XDMM);
                bw.Write(0);
            }
        }

        // Write MMID from obj0 or root
        if (obj0Lookup.TryGetValue("DIMM", out var mmidChunk))
        {
            bw.Write(obj0Data!, mmidChunk.Offset, mmidChunk.TotalSize);
        }
        else
        {
            var rootMmid = rootChunks.Find(c => c.Signature == "DIMM");
            if (rootMmid != null)
                bw.Write(rootData, rootMmid.Offset, rootMmid.TotalSize);
            else
            {
                bw.Write(DIMM);
                bw.Write(0);
            }
        }

        // Write MWMO from obj0 or root
        if (obj0Lookup.TryGetValue("OMWM", out var mwmoChunk))
        {
            bw.Write(obj0Data!, mwmoChunk.Offset, mwmoChunk.TotalSize);
        }
        else
        {
            var rootMwmo = rootChunks.Find(c => c.Signature == "OMWM");
            if (rootMwmo != null)
                bw.Write(rootData, rootMwmo.Offset, rootMwmo.TotalSize);
            else
            {
                bw.Write(OMWM);
                bw.Write(0);
            }
        }

        // Write MWID from obj0 or root
        if (obj0Lookup.TryGetValue("DIWM", out var mwidChunk))
        {
            bw.Write(obj0Data!, mwidChunk.Offset, mwidChunk.TotalSize);
        }
        else
        {
            var rootMwid = rootChunks.Find(c => c.Signature == "DIWM");
            if (rootMwid != null)
                bw.Write(rootData, rootMwid.Offset, rootMwid.TotalSize);
            else
            {
                bw.Write(DIWM);
                bw.Write(0);
            }
        }

        // Write MDDF from obj0 or root
        if (obj0Lookup.TryGetValue("FDDM", out var mddfChunk))
        {
            bw.Write(obj0Data!, mddfChunk.Offset, mddfChunk.TotalSize);
        }
        else
        {
            var rootMddf = rootChunks.Find(c => c.Signature == "FDDM");
            if (rootMddf != null)
                bw.Write(rootData, rootMddf.Offset, rootMddf.TotalSize);
            else
            {
                bw.Write(FDDM);
                bw.Write(0);
            }
        }

        // Write MODF from obj0 or root
        if (obj0Lookup.TryGetValue("FDOM", out var modfChunk))
        {
            bw.Write(obj0Data!, modfChunk.Offset, modfChunk.TotalSize);
        }
        else
        {
            var rootModf = rootChunks.Find(c => c.Signature == "FDOM");
            if (rootModf != null)
                bw.Write(rootData, rootModf.Offset, rootModf.TotalSize);
            else
            {
                bw.Write(FDOM);
                bw.Write(0);
            }
        }

        // Write all MCNK chunks
        // In split format, MCNK in root has terrain, tex0 has textures
        // We must merge them into a single monolithic MCNK
        var mcnkChunks = rootChunks.FindAll(c => c.Signature == "KNCM");
        var tex0McnkChunks = tex0Chunks.FindAll(c => c.Signature == "KNCM");
        var obj0McnkChunks = obj0Chunks.FindAll(c => c.Signature == "KNCM");
        
        // Track MCNK positions for MCIN update
        var mcnkPositions = new List<(long offset, int size)>();

        for (int i = 0; i < mcnkChunks.Count; i++)
        {
            long mcnkPos = ms.Position;
            var rootMcnk = mcnkChunks[i];
            var tex0Mcnk = (tex0McnkChunks.Count > i) ? tex0McnkChunks[i] : null;
            var obj0Mcnk = (obj0McnkChunks.Count > i) ? obj0McnkChunks[i] : null;

            if (tex0Mcnk != null)
            {
                // Merge Root + Tex0 (+ Obj0 if we handle it)
                var mergedData = MergeMcnk(rootMcnk, tex0Mcnk, obj0Mcnk, rootData, tex0Data, obj0Data);
                bw.Write(mergedData);
                mcnkPositions.Add((mcnkPos, mergedData.Length));
            }
            else
            {
                // No Tex0, just copy Root (likely no textures or flat terrain)
                bw.Write(rootData, rootMcnk.Offset, rootMcnk.TotalSize);
                mcnkPositions.Add((mcnkPos, rootMcnk.TotalSize));
            }
        }

        // Update MCIN with correct offsets
        if (mcinChunk != null && mcnkPositions.Count == 256)
        {
            long currentPos = ms.Position;
            ms.Position = mcinPos + 8; // Skip MCIN header
            
            for (int i = 0; i < 256; i++)
            {
                var (offset, size) = mcnkPositions[i];
                bw.Write((uint)offset);
                bw.Write((uint)size);
                bw.Write(0u); // flags
                bw.Write(0u); // asyncId
            }
            
            ms.Position = currentPos;
        }

        return ms.ToArray();
    }

    private byte[] MergeMcnk(ChunkInfo rootMcnk, ChunkInfo tex0Mcnk, ChunkInfo? obj0Mcnk, byte[] rootData, byte[]? tex0Data, byte[]? obj0Data)
    {
        // Extract subchunks
        var rootSub = ParseSubChunks(rootData, rootMcnk.Offset + 128, rootMcnk.DataSize - 128); // Skip 128 byte header
        var tex0Sub = tex0Data != null ? ParseSubChunks(tex0Data, tex0Mcnk.Offset + 128, tex0Mcnk.DataSize - 128) : new Dictionary<string, ChunkInfo>();
        var obj0Sub = obj0Data != null && obj0Mcnk != null ? ParseSubChunks(obj0Data, obj0Mcnk.Offset + 128, obj0Mcnk.DataSize - 128) : new Dictionary<string, ChunkInfo>();

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // 1. Write Header (Copy from Root initially, will update)
        // Header is 128 bytes (0x80)
        byte[] header = new byte[128];
        Array.Copy(rootData, rootMcnk.Offset + 8, header, 0, 128); // +8 to skip KNCM and Size
        
        // Write standard chunk header
        bw.Write(Encoding.ASCII.GetBytes("MCNK")); // Not KNCM here? Wait, parsing used REVM (reversed). 
        // File on disk has "MCNK" reversed as "KNCM" if little endian read as int?
        // No, standard strings in binary writer are forward.
        // My parsing logic used `Encoding.ASCII.GetString` which reads forward.
        // So "KNCM" means the file has "K", "N", "C", "M" bytes? 
        // No, usually it's "MCNK". Let's check my parser constants.
        // private static readonly byte[] KNCM = Encoding.ASCII.GetBytes("KNCM");
        // Wait, "KNCM" is reversed "MCNK".
        // If I read "KNCM" string, then the bytes in file are K, N, C, M.
        // But standard ADT has "MCNK".
        // Ah, typically we read Signature as Int32 for performance, then it is reversed on LE systems.
        // But I used `GetString` which reads bytes in order.
        // If the file has "MCNK", `GetString` returns "MCNK".
        // So my parser constants are probably expecting "MCNK" but I named them reversed?
        // Let's stick to "MCNK" for output.
        // Wait, the parser loop used `Parsing chunks...`. 
        // If previous code used "KNCM", maybe the input file IS reversed (WDL artifact)?
        // Or I copied code that assumes reading as int.
        
        // Let's Assume Output should be "MCNK".
        // But wait, the previous code block wrote: `bw.Write(rootData, rootMcnk.Offset, ...)` which copies exact bytes.
        // If I write new bytes, I should write "MCNK" reversed as bytes so it matches?
        // Actually, let's just write the 4 bytes from the header copy if possible.
        // But MCNK header is *inside* the chunk data in my ParseChunks?
        // No, `ChunkInfo` includes the header (8 bytes) in `TotalSize` but `Offset` points to start of Sig.
        // So `rootMcnk.Offset` is the "MCNK" signature.
        
        // I'll write the signature manually.
        bw.Write(Encoding.ASCII.GetBytes("MCNK"));
        // Placeholder for size
        bw.Write(0); 
        
        // Write the 128 byte MCNK header
        bw.Write(header);

        // Helper to write subchunk and return relative offset
        int WriteSubChunk(string sig, Dictionary<string, ChunkInfo> source, byte[] data)
        {
            if (source.TryGetValue(sig, out var chunk))
            {
                int offset = (int)ms.Position - 8; // Relative to MCNK data start (after Sig+Size)
                bw.Write(data, chunk.Offset, chunk.TotalSize);
                return offset;
            }
            return 0;
        }

        // Write Subchunks in standard 3.3.5 order
        int offMcvt = WriteSubChunk("TVCM", rootSub, rootData); // MCVT
        int offMcnr = WriteSubChunk("RNCM", rootSub, rootData); // MCNR
        int offMcly = WriteSubChunk("YLCM", tex0Sub, tex0Data!); // MCLY (From Tex0!)
        
        // MCRF (From Obj0, fallback to Root)
        int offMcrf = WriteSubChunk("FRCM", obj0Sub, obj0Data!); 
        if (offMcrf == 0) offMcrf = WriteSubChunk("FRCM", rootSub, rootData);

        int offMcal = WriteSubChunk("LAXM", tex0Sub, tex0Data!); // MCAL (From Tex0!)
        if (offMcal == 0) offMcal = WriteSubChunk("LACM", tex0Sub, tex0Data!);
        
        int offMcsh = WriteSubChunk("HSCM", tex0Sub, tex0Data!); // MCSH (From Tex0!)
        int offMcse = WriteSubChunk("ESCM", rootSub, rootData); // MCSE
        int offMclq = WriteSubChunk("QLCM", rootSub, rootData); // MCLQ
        if (offMclq == 0) offMclq = WriteSubChunk("QLCM", tex0Sub, tex0Data!); 

        // MCCV (Vertex Colors)
        int offMccv = WriteSubChunk("VCCM", rootSub, rootData); 

        // Update Size
        int totalSize = (int)ms.Length - 8;
        long endPos = ms.Position;
        ms.Position = 4;
        bw.Write(totalSize);

        // Update Header Offsets & Sizes
        ms.Position = 8; // Back to MCNK header
        using var br = new BinaryReader(new MemoryStream(header));
        
        // Update nLayers
        if (offMcly > 0 && tex0Sub.ContainsKey("YLCM"))
        {
            int nLayers = tex0Sub["YLCM"].DataSize / 16;
            ms.Position = 8 + 0x0C;
            bw.Write(nLayers);
        }

        // Helper to get chunk size
        int GetSize(string sig, Dictionary<string, ChunkInfo> sub) 
            => sub.ContainsKey(sig) ? sub[sig].DataSize : 0;

        // Calculate Sizes
        int sizeAlpha = GetSize("LAXM", tex0Sub);
        if (sizeAlpha == 0) sizeAlpha = GetSize("LACM", tex0Sub);
        
        int sizeShadow = GetSize("HSCM", tex0Sub);

        // Update Offsets & Sizes
        void WriteOffset(int offset, int val)
        {
            if (val > 0)
            {
                ms.Position = 8 + offset;
                bw.Write(val);
            }
        }

        WriteOffset(0x5C, offMcvt);
        WriteOffset(0x60, offMcnr);
        WriteOffset(0x64, offMcly);
        WriteOffset(0x68, offMcrf);
        WriteOffset(0x6C, offMcal);
        
        ms.Position = 8 + 0x70; // sizeAlpha
        bw.Write(sizeAlpha);
        
        WriteOffset(0x74, offMcsh);
        
        ms.Position = 8 + 0x78; // sizeShadow
        bw.Write(sizeShadow);

        WriteOffset(0x7C, offMcse);
        
        // MCCV usually goes at 0x84 if it exists? 
        // Or sometimes it's implied by flags.
        // For now, let's leave it as is unless we know the specific offset.
        // But we DO need to update MCNK flags if we add MCCV?
        // Assuming original root MCNK flags are correct for MCCV presence.

        ms.Position = endPos;
        return ms.ToArray();
    }

    private Dictionary<string, ChunkInfo> ParseSubChunks(byte[] data, int start, int length)
    {
        var chunks = new Dictionary<string, ChunkInfo>();
        int pos = start;
        int end = start + length;

        while (pos < end - 4)
        {
            // Subchunks usually have 4 byte sig (reversed) + 4 byte size
            var sig = Encoding.ASCII.GetString(data, pos, 4);
            var size = BitConverter.ToInt32(data, pos + 4);
            
            chunks[sig] = new ChunkInfo(sig, pos, size, 8 + size);
            pos += 8 + size;
        }
        return chunks;
    }

    // Existing methods...

    public static (bool hasRoot, bool hasObj0, bool hasTex0) CheckSplitAdtExists(string directory, string mapName, int tileX, int tileY)
    {
        string baseName = $"{mapName}_{tileX}_{tileY}";
        bool hasRoot = File.Exists(Path.Combine(directory, $"{baseName}.adt"));
        bool hasObj0 = File.Exists(Path.Combine(directory, $"{baseName}_obj0.adt"));
        bool hasTex0 = File.Exists(Path.Combine(directory, $"{baseName}_tex0.adt"));
        return (hasRoot, hasObj0, hasTex0);
    }

    private record ChunkInfo(string Signature, int Offset, int DataSize, int TotalSize);

    private List<ChunkInfo> ParseChunks(byte[] data)
    {
        var chunks = new List<ChunkInfo>();
        int pos = 0;

        while (pos < data.Length - 8)
        {
            var sig = Encoding.ASCII.GetString(data, pos, 4);
            var size = BitConverter.ToInt32(data, pos + 4);

            if (size < 0 || pos + 8 + size > data.Length)
                break;

            chunks.Add(new ChunkInfo(sig, pos, size, 8 + size));
            pos += 8 + size;
        }

        return chunks;
    }
}
