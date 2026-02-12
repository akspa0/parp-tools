using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Injects MWMO and MODF chunks into 3.3.5 monolithic ADT files.
/// </summary>
public sealed class AdtModfInjector
{
    // ADT chunk signatures - REVERSED on disk (e.g., "MVER" stored as "REVM")
    private static readonly byte[] MVER = Encoding.ASCII.GetBytes("REVM");
    private static readonly byte[] MHDR = Encoding.ASCII.GetBytes("RDHM");
    private static readonly byte[] MCIN = Encoding.ASCII.GetBytes("NICM");
    private static readonly byte[] MTEX = Encoding.ASCII.GetBytes("XETM");
    private static readonly byte[] MMDX = Encoding.ASCII.GetBytes("XDMM");
    private static readonly byte[] MMID = Encoding.ASCII.GetBytes("DIMM");
    private static readonly byte[] MWMO = Encoding.ASCII.GetBytes("OMWM");
    private static readonly byte[] MWID = Encoding.ASCII.GetBytes("DIWM");
    private static readonly byte[] MDDF = Encoding.ASCII.GetBytes("FDDM");
    private static readonly byte[] MODF = Encoding.ASCII.GetBytes("FDOM");
    private static readonly byte[] MCNK = Encoding.ASCII.GetBytes("KNCM");
    private static readonly byte[] MCVT = Encoding.ASCII.GetBytes("TVCM");
    private static readonly byte[] MCNR = Encoding.ASCII.GetBytes("RNCM");
    private static readonly byte[] MCLY = Encoding.ASCII.GetBytes("YLCM");
    private static readonly byte[] MCRF = Encoding.ASCII.GetBytes("FRCM");
    private static readonly byte[] MCSH = Encoding.ASCII.GetBytes("HSCM");
    private static readonly byte[] MCAL = Encoding.ASCII.GetBytes("LACM");

    /// <summary>
    /// MODF entry for 3.3.5 ADT (64 bytes).
    /// </summary>
    public struct ModfEntry335
    {
        public uint NameId;           // Index into MWID (offset into MWMO)
        public uint UniqueId;         // Unique identifier
        public Vector3 Position;      // World position
        public Vector3 Rotation;      // Rotation in degrees
        public Vector3 BoundsMin;     // Bounding box min
        public Vector3 BoundsMax;     // Bounding box max
        public ushort Flags;          // Placement flags
        public ushort DoodadSet;      // Doodad set index
        public ushort NameSet;        // Name set index
        public ushort Scale;          // Scale (unused for WMO, always 1024)
    }

    /// <summary>
    /// Inject MODF data into an existing 3.3.5 ADT file.
    /// </summary>
    public void InjectModf(string adtPath, string outputPath, List<string> wmoNames, List<ModfEntry335> modfEntries)
    {
        if (!File.Exists(adtPath))
        {
            Console.WriteLine($"[ERROR] ADT file not found: {adtPath}");
            return;
        }

        var adtData = File.ReadAllBytes(adtPath);
        var result = InjectModfIntoAdt(adtData, wmoNames, modfEntries);
        
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, result);
        
        Console.WriteLine($"[INFO] Wrote ADT with MODF to: {outputPath}");
    }

    /// <summary>
    /// Inject MODF data into ADT byte array.
    /// </summary>
    public byte[] InjectModfIntoAdt(byte[] adtData, List<string> wmoNames, List<ModfEntry335> modfEntries)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Parse existing ADT to find chunk positions
        var chunks = ParseChunks(adtData);
        
        // Build new MWMO chunk
        var (mwmoData, mwidData) = BuildMwmoAndMwid(wmoNames);
        
        // Build new MODF chunk
        var modfData = BuildModf(modfEntries);

        // Reconstruct ADT with new/updated chunks
        int pos = 0;
        bool wroteNewMwmo = false;
        bool wroteNewModf = false;

        foreach (var chunk in chunks)
        {
            // Write any data before this chunk
            if (chunk.Offset > pos)
            {
                bw.Write(adtData, pos, chunk.Offset - pos);
            }

            if (chunk.Signature == "MWMO")
            {
                // Replace MWMO with our data
                WriteMwmoChunk(bw, mwmoData);
                wroteNewMwmo = true;
            }
            else if (chunk.Signature == "MWID")
            {
                // Replace MWID with our data
                WriteMwidChunk(bw, mwidData);
            }
            else if (chunk.Signature == "MODF")
            {
                // Replace MODF with our data
                WriteModfChunk(bw, modfData);
                wroteNewModf = true;
            }
            else if (chunk.Signature == "MDDF" && !wroteNewMwmo)
            {
                // Insert MWMO/MWID before MDDF if they don't exist
                WriteMwmoChunk(bw, mwmoData);
                WriteMwidChunk(bw, mwidData);
                wroteNewMwmo = true;
                
                // Write original MDDF
                bw.Write(adtData, chunk.Offset, chunk.TotalSize);
            }
            else if (chunk.Signature == "MCNK" && !wroteNewModf)
            {
                // Insert MODF before first MCNK if it doesn't exist
                WriteModfChunk(bw, modfData);
                wroteNewModf = true;
                
                // Write original MCNK
                bw.Write(adtData, chunk.Offset, chunk.TotalSize);
            }
            else
            {
                // Copy chunk as-is
                bw.Write(adtData, chunk.Offset, chunk.TotalSize);
            }

            pos = chunk.Offset + chunk.TotalSize;
        }

        // Write any remaining data
        if (pos < adtData.Length)
        {
            bw.Write(adtData, pos, adtData.Length - pos);
        }

        // If we never wrote MWMO/MODF, append them
        if (!wroteNewMwmo)
        {
            Console.WriteLine("[WARN] MWMO chunk not found in ADT, appending at end");
            WriteMwmoChunk(bw, mwmoData);
            WriteMwidChunk(bw, mwidData);
        }
        if (!wroteNewModf)
        {
            Console.WriteLine("[WARN] MODF chunk not found in ADT, appending at end");
            WriteModfChunk(bw, modfData);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Create a minimal 3.3.5 ADT file with just MODF data.
    /// </summary>
    public byte[] CreateMinimalAdt(int tileX, int tileY, List<string> wmoNames, List<ModfEntry335> modfEntries)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // MVER - Version (always 18 for 3.3.5)
        bw.Write(MVER);
        bw.Write(4); // size
        bw.Write(18); // version

        // MHDR - Header (64 bytes, offsets will be fixed later)
        long mhdrPos = ms.Position;
        bw.Write(MHDR);
        bw.Write(64); // size
        var mhdrDataPos = ms.Position;
        for (int i = 0; i < 16; i++) bw.Write(0); // 16 uint32 offsets, fill with 0 for now

        // MCIN - Chunk index (256 entries * 16 bytes = 4096 bytes)
        long mcinPos = ms.Position;
        bw.Write(MCIN);
        bw.Write(4096); // size
        for (int i = 0; i < 256; i++)
        {
            bw.Write(0); // offset (will be filled later)
            bw.Write(0); // size
            bw.Write(0); // flags
            bw.Write(0); // async_id
        }

        // MTEX - Empty texture list
        bw.Write(MTEX);
        bw.Write(0);

        // MMDX - Empty M2 model list
        bw.Write(MMDX);
        bw.Write(0);

        // MMID - Empty M2 ID list
        bw.Write(MMID);
        bw.Write(0);

        // MWMO - WMO names
        var (mwmoData, mwidData) = BuildMwmoAndMwid(wmoNames);
        long mwmoPos = ms.Position;
        WriteMwmoChunk(bw, mwmoData);

        // MWID - WMO name offsets
        WriteMwidChunk(bw, mwidData);

        // MDDF - Empty M2 placements
        bw.Write(MDDF);
        bw.Write(0);

        // MODF - WMO placements
        long modfPos = ms.Position;
        var modfData = BuildModf(modfEntries);
        WriteModfChunk(bw, modfData);

        // MCNK - 256 terrain chunks (minimal/empty)
        long mcnkStartPos = ms.Position;
        var mcnkOffsets = new List<(long offset, int size)>();
        
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                long chunkPos = ms.Position;
                WriteMcnkChunk(bw, tileX, tileY, x, y);
                mcnkOffsets.Add((chunkPos, (int)(ms.Position - chunkPos)));
            }
        }

        // Go back and fix MHDR offsets
        long endPos = ms.Position;
        ms.Position = mhdrDataPos;
        bw.Write((uint)(mcinPos - 8 - mhdrDataPos + 8)); // mcin offset (relative to MHDR data start)
        bw.Write(0u); // mtex
        bw.Write(0u); // mmdx
        bw.Write(0u); // mmid
        bw.Write((uint)(mwmoPos - 8 - mhdrDataPos + 8)); // mwmo
        bw.Write(0u); // mwid
        bw.Write(0u); // mddf
        bw.Write((uint)(modfPos - 8 - mhdrDataPos + 8)); // modf
        // Rest are 0

        // Fix MCIN entries
        ms.Position = mcinPos + 8; // Skip chunk header
        for (int i = 0; i < 256; i++)
        {
            var (offset, size) = mcnkOffsets[i];
            bw.Write((uint)offset);
            bw.Write((uint)size);
            bw.Write(0u); // flags
            bw.Write(0u); // async_id
        }

        ms.Position = endPos;
        return ms.ToArray();
    }

    private void WriteMcnkChunk(BinaryWriter bw, int tileX, int tileY, int chunkX, int chunkY)
    {
        // Full MCNK chunk with all required subchunks for 3.3.5
        bw.Write(MCNK);
        
        // We'll write the size after we know it
        long sizePos = bw.BaseStream.Position;
        bw.Write(0); // placeholder for size

        long dataStart = bw.BaseStream.Position;

        // Pre-calculate subchunk sizes and offsets (relative to MCNK data start)
        // Header: 128 bytes
        // MCVT: 8 + 580 = 588 bytes (heights)
        // MCNR: 8 + 435 + 13 = 456 bytes (normals + padding)
        // MCLY: 8 + 16 = 24 bytes (1 layer entry)
        // MCRF: 8 + 0 = 8 bytes (empty refs)
        // MCSH: 8 + 0 = 8 bytes (empty shadow) - optional but include for completeness
        // MCAL: 8 + 0 = 8 bytes (empty alpha)
        
        const int headerSize = 128;
        const int mcvtSize = 580;
        const int mcnrSize = 435;
        const int mcnrPadding = 13;
        const int mclyEntrySize = 16; // 1 layer
        const int mcrfSize = 0; // empty
        const int mcshSize = 0; // empty
        const int mcalSize = 0; // empty
        
        // Offsets relative to MCNK data start (after header)
        uint ofsMcly = headerSize + 8 + mcvtSize + 8 + mcnrSize + mcnrPadding;
        uint ofsMcrf = ofsMcly + 8 + mclyEntrySize;
        uint ofsMcal = ofsMcrf + 8 + mcrfSize;
        uint ofsMcsh = ofsMcal + 8 + mcalSize;

        // MCNK header (128 bytes) - 3.3.5 format
        bw.Write(0u); // 0x00: flags (MCNKFlags)
        bw.Write((uint)chunkX); // 0x04: indexX
        bw.Write((uint)chunkY); // 0x08: indexY
        bw.Write(1u); // 0x0C: nLayers (1 base layer)
        bw.Write(0u); // 0x10: nDoodadRefs
        bw.Write(0uL); // 0x14: holesHighRes (8 bytes)
        bw.Write(ofsMcly); // 0x1C: ofsMCLY
        bw.Write(ofsMcrf); // 0x20: ofsMCRF
        bw.Write(ofsMcal); // 0x24: ofsMCAL
        bw.Write((uint)mcalSize); // 0x28: sizeAlpha
        bw.Write(ofsMcsh); // 0x2C: ofsMCSH
        bw.Write((uint)mcshSize); // 0x30: sizeShadows
        bw.Write(0u); // 0x34: areaID
        bw.Write(0u); // 0x38: nMapObjRefs
        bw.Write((ushort)0); // 0x3C: holesLowRes
        bw.Write((ushort)0); // 0x3E: unknownPad
        // lowQualityTexturingMap (8 shorts = 16 bytes)
        for (int i = 0; i < 8; i++) bw.Write((short)0);
        bw.Write(0L); // 0x50: noEffectDoodad (8 bytes)
        bw.Write(0u); // 0x58: ofsMCSE - no sound
        bw.Write(0u); // 0x5C: numMCSE
        bw.Write(0u); // 0x60: ofsMCLQ - no liquid
        bw.Write(0u); // 0x64: sizeMCLQ
        
        // Position - chunk corner in world coordinates
        float posX = (32 - tileX) * 533.33333f - chunkX * 33.33333f;
        float posY = (32 - tileY) * 533.33333f - chunkY * 33.33333f;
        bw.Write(posX); // 0x68: position.x
        bw.Write(posY); // 0x6C: position.y
        bw.Write(0f);   // 0x70: position.z (base height)
        
        bw.Write(0u); // 0x74: ofsMCCV - no vertex colors
        bw.Write(0u); // 0x78: ofsMCLV - no vertex lighting
        bw.Write(0u); // 0x7C: unused

        // Verify we wrote exactly 128 bytes for header
        long headerEnd = bw.BaseStream.Position;
        if (headerEnd - dataStart != 128)
        {
            throw new InvalidOperationException($"MCNK header size mismatch: expected 128, got {headerEnd - dataStart}");
        }

        // MCVT - Height map (145 floats = 580 bytes)
        bw.Write(MCVT);
        bw.Write(mcvtSize);
        for (int i = 0; i < 145; i++)
            bw.Write(0f); // flat terrain at height 0

        // MCNR - Normals (145 * 3 bytes = 435 bytes + 13 padding)
        bw.Write(MCNR);
        bw.Write(mcnrSize);
        for (int i = 0; i < 145; i++)
        {
            bw.Write((sbyte)0);   // X normal
            bw.Write((sbyte)0);   // Y normal
            bw.Write((sbyte)127); // Z normal (pointing up)
        }
        // MCNR has 13 bytes of padding after the normals
        for (int i = 0; i < mcnrPadding; i++)
            bw.Write((byte)0);

        // MCLY - Texture layers (1 base layer = 16 bytes)
        bw.Write(MCLY);
        bw.Write(mclyEntrySize);
        bw.Write(0u); // textureId (index into MTEX)
        bw.Write(0u); // flags
        bw.Write(0u); // offsetInMCAL
        bw.Write(0u); // effectId + padding

        // MCRF - Doodad/WMO references (empty)
        bw.Write(MCRF);
        bw.Write(mcrfSize);

        // MCAL - Alpha maps (empty for base layer only)
        bw.Write(MCAL);
        bw.Write(mcalSize);

        // MCSH - Shadow map (empty)
        bw.Write(MCSH);
        bw.Write(mcshSize);

        // Calculate and write MCNK chunk size
        long dataEnd = bw.BaseStream.Position;
        int chunkSize = (int)(dataEnd - dataStart);
        bw.BaseStream.Position = sizePos;
        bw.Write(chunkSize);
        bw.BaseStream.Position = dataEnd;
    }

    private (byte[] mwmoData, uint[] mwidData) BuildMwmoAndMwid(List<string> wmoNames)
    {
        using var ms = new MemoryStream();
        var offsets = new List<uint>();

        foreach (var name in wmoNames)
        {
            offsets.Add((uint)ms.Position);
            var bytes = Encoding.ASCII.GetBytes(name);
            ms.Write(bytes, 0, bytes.Length);
            ms.WriteByte(0); // null terminator
        }

        return (ms.ToArray(), offsets.ToArray());
    }

    private byte[] BuildModf(List<ModfEntry335> entries)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        foreach (var entry in entries)
        {
            bw.Write(entry.NameId);
            bw.Write(entry.UniqueId);
            bw.Write(entry.Position.X);
            bw.Write(entry.Position.Y);
            bw.Write(entry.Position.Z);
            bw.Write(entry.Rotation.X);
            bw.Write(entry.Rotation.Y);
            bw.Write(entry.Rotation.Z);
            bw.Write(entry.BoundsMin.X);
            bw.Write(entry.BoundsMin.Y);
            bw.Write(entry.BoundsMin.Z);
            bw.Write(entry.BoundsMax.X);
            bw.Write(entry.BoundsMax.Y);
            bw.Write(entry.BoundsMax.Z);
            bw.Write(entry.Flags);
            bw.Write(entry.DoodadSet);
            bw.Write(entry.NameSet);
            bw.Write(entry.Scale);
        }

        return ms.ToArray();
    }

    private void WriteMwmoChunk(BinaryWriter bw, byte[] data)
    {
        bw.Write(MWMO);
        bw.Write(data.Length);
        bw.Write(data);
    }

    private void WriteMwidChunk(BinaryWriter bw, uint[] offsets)
    {
        bw.Write(MWID);
        bw.Write(offsets.Length * 4);
        foreach (var offset in offsets)
            bw.Write(offset);
    }

    private void WriteModfChunk(BinaryWriter bw, byte[] data)
    {
        bw.Write(MODF);
        bw.Write(data.Length);
        bw.Write(data);
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

    /// <summary>
    /// Convert PM4 server coordinates to ADT world coordinates for MODF.
    /// </summary>
    public static Vector3 ServerToAdtPosition(Vector3 serverPos)
    {
        // Transform: Identity (WoW Server Coords == ADT World Coords for 3.3.5)
        // MODF chunks use standard world coordinates.
        return serverPos;
    }
}
