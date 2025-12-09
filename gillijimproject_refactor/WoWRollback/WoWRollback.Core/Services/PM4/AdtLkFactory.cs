using System;
using System.Collections.Generic;
using System.Numerics;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Factory for creating valid LK (3.3.5) ADT files using the gillijimproject-csharp infrastructure.
/// </summary>
public static class AdtLkFactory
{
    /// <summary>
    /// Create a minimal but valid LK ADT with MODF placements.
    /// </summary>
    public static AdtLk CreateMinimalAdt(
        string adtName,
        int tileX,
        int tileY,
        List<string> wmoNames,
        List<ModfEntry> modfEntries)
    {
        // MVER - Version 18
        var mver = new Chunk("MVER", 4, BitConverter.GetBytes(18));

        // MH2O - Empty water
        var mh2o = new Mh2o();

        // MTEX - Empty texture list
        var mtex = new Chunk("MTEX", 0, Array.Empty<byte>());

        // MMDX - Empty M2 model list
        var mmdx = new Mmdx("MMDX", 0, Array.Empty<byte>());

        // MMID - Empty M2 ID list
        var mmid = new Mmid("MMID", 0, Array.Empty<byte>());

        // MWMO - WMO names
        var mwmo = CreateMwmo(wmoNames);

        // MWID - WMO name offsets
        var mwid = CreateMwid(wmoNames);

        // MDDF - Empty M2 placements
        var mddf = new Mddf("MDDF", 0, Array.Empty<byte>());

        // MODF - WMO placements
        var modf = CreateModf(modfEntries);

        // MCNK - 256 terrain chunks
        var mcnks = new List<McnkLk>(256);
        int adtNumber = tileY * 64 + tileX;
        
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                var mcnk = CreateMinimalMcnk(adtNumber, x, y);
                mcnks.Add(mcnk);
            }
        }

        // MFBO - Empty flight bounds
        var mfbo = new Chunk("MFBO", 0, Array.Empty<byte>());

        // MTXF - Empty texture flags
        var mtxf = new Chunk("MTXF", 0, Array.Empty<byte>());

        // Create the ADT
        return new AdtLk(
            adtName,
            mver,
            0, // mhdrFlags
            mh2o,
            mtex,
            mmdx,
            mmid,
            mwmo,
            mwid,
            mddf,
            modf,
            mcnks,
            mfbo,
            mtxf);
    }

    /// <summary>
    /// Create MWMO chunk from WMO names.
    /// </summary>
    private static Mwmo CreateMwmo(List<string> wmoNames)
    {
        using var ms = new System.IO.MemoryStream();
        foreach (var name in wmoNames)
        {
            var bytes = System.Text.Encoding.ASCII.GetBytes(name);
            ms.Write(bytes, 0, bytes.Length);
            ms.WriteByte(0); // null terminator
        }
        return new Mwmo("MWMO", (int)ms.Length, ms.ToArray());
    }

    /// <summary>
    /// Create MWID chunk from WMO names.
    /// </summary>
    private static Mwid CreateMwid(List<string> wmoNames)
    {
        using var ms = new System.IO.MemoryStream();
        using var bw = new System.IO.BinaryWriter(ms);
        
        int offset = 0;
        foreach (var name in wmoNames)
        {
            bw.Write(offset);
            offset += name.Length + 1; // +1 for null terminator
        }
        
        return new Mwid("MWID", (int)ms.Length, ms.ToArray());
    }

    /// <summary>
    /// Create MODF chunk from placement entries.
    /// </summary>
    private static Modf CreateModf(List<ModfEntry> entries)
    {
        using var ms = new System.IO.MemoryStream();
        using var bw = new System.IO.BinaryWriter(ms);

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

        return new Modf("MODF", (int)ms.Length, ms.ToArray());
    }

    /// <summary>
    /// Create a minimal but complete MCNK chunk.
    /// </summary>
    private static McnkLk CreateMinimalMcnk(int adtNumber, int indexX, int indexY)
    {
        // Compute world position
        var (posX, posY, posZ) = McnkLk.ComputePositionFromAdt(adtNumber, indexX, indexY);

        // Create MCNK header
        var header = new McnkHeader
        {
            Flags = 0,
            IndexX = indexX,
            IndexY = indexY,
            NLayers = 0,
            M2Number = 0,      // NDoodadRefs
            WmoNumber = 0,     // NMapObjRefs
            Holes = 0,
            AreaId = 0,
            PosX = posX,
            PosY = posY,
            PosZ = posZ,
            // Offsets will be computed by McnkLk.GetWholeChunk()
        };

        // MCVT - Heights (145 floats = 580 bytes)
        var mcvtData = new byte[145 * 4];
        // All zeros = flat terrain at height 0
        var mcvt = new Chunk("MCVT", mcvtData.Length, mcvtData);

        // MCNR - Normals (145 * 3 bytes = 435 bytes)
        // Note: MCNR has 13 bytes of padding AFTER the chunk that's not counted in size
        // McnrLk should handle this - let's use just the normal data
        var mcnrData = new byte[145 * 3]; // 435 bytes
        for (int i = 0; i < 145; i++)
        {
            mcnrData[i * 3 + 0] = 0;   // X normal
            mcnrData[i * 3 + 1] = 0;   // Y normal
            mcnrData[i * 3 + 2] = 127; // Z normal (pointing up)
        }
        var mcnr = new McnrLk("MCNR", 435, mcnrData);

        // MCLY - Empty texture layers
        var mcly = new Chunk("MCLY", 0, Array.Empty<byte>());

        // MCRF - Empty references
        var mcrf = new Mcrf("MCRF", 0, Array.Empty<byte>());

        // MCSH - No shadow
        Chunk? mcsh = null;

        // MCAL - No alpha
        var mcal = new Mcal("MCAL", 0, Array.Empty<byte>());

        // MCLQ - No liquid
        Chunk? mclq = null;

        // MCSE - No sound emitters
        Chunk? mcse = null;

        // MCCV - No vertex colors
        Chunk? mccv = null;

        return new McnkLk(header, mcvt, mccv, mcnr, mcly, mcrf, mcsh, mcal, mclq, mcse);
    }

    /// <summary>
    /// MODF entry structure.
    /// </summary>
    public struct ModfEntry
    {
        public uint NameId;           // Index into MWID
        public uint UniqueId;         // Unique identifier
        public Vector3 Position;      // World position
        public Vector3 Rotation;      // Rotation in degrees
        public Vector3 BoundsMin;     // Bounding box min
        public Vector3 BoundsMax;     // Bounding box max
        public ushort Flags;          // Placement flags
        public ushort DoodadSet;      // Doodad set index
        public ushort NameSet;        // Name set index
        public ushort Scale;          // Scale (1024 = 1.0)
    }
}
