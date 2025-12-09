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

        // Write all MCNK chunks from root (they contain terrain data)
        // In split format, MCNK in root has terrain, tex0 has textures, obj0 has refs
        // For 3.3.5 monolithic, we need to merge them
        var mcnkChunks = rootChunks.FindAll(c => c.Signature == "KNCM");
        var tex0McnkChunks = tex0Chunks.FindAll(c => c.Signature == "KNCM");
        var obj0McnkChunks = obj0Chunks.FindAll(c => c.Signature == "KNCM");

        // Track MCNK positions for MCIN update
        var mcnkPositions = new List<(long offset, int size)>();

        for (int i = 0; i < mcnkChunks.Count; i++)
        {
            long mcnkPos = ms.Position;
            
            // For now, just copy root MCNK - a full implementation would merge subchunks
            var rootMcnk = mcnkChunks[i];
            bw.Write(rootData, rootMcnk.Offset, rootMcnk.TotalSize);
            
            mcnkPositions.Add((mcnkPos, rootMcnk.TotalSize));
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

    /// <summary>
    /// Check if split ADT files exist for a tile.
    /// </summary>
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
