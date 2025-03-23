using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk containing terrain data and subchunks.
/// </summary>
public class McnkChunk : ChunkBase
{
    public override string ChunkId => "MCNK";

    [Flags]
    public enum McnkFlags : uint
    {
        HasMcsh = 0x1,              // Shadow map present
        Impass = 0x2,              // Impassable terrain
        LiquidRiver = 0x4,         // River liquid type
        LiquidOcean = 0x8,         // Ocean liquid type
        LiquidMagma = 0x10,        // Magma liquid type
        LiquidSlime = 0x20,        // Slime liquid type
        HasMccv = 0x40,            // Has vertex shading
        HasMclv = 0x80,            // Has vertex lighting
        HasMcse = 0x100,           // Has sound emitters
        HasMclq = 0x200,           // Has old-style liquid (MCLQ)
        HasMcal = 0x400,           // Has alpha map
        HasMcsh2 = 0x800,          // Has new-style shadow map
        HasMcmt = 0x1000,          // Has multiple texture maps
        HasMcrd = 0x2000,          // Has render data
        HasMcbb = 0x4000,          // Has bounding box
        HasMcbn = 0x8000,          // Has BSP tree
        HasMcrf = 0x10000,         // Has render flags
        HasMcsh3 = 0x20000,        // Has another shadow map
        HasMcsh4 = 0x40000,        // Has yet another shadow map
        HasMccv2 = 0x80000,        // Has another vertex shading
        HasMclv2 = 0x100000,       // Has another vertex lighting
        HasMcse2 = 0x200000,       // Has another sound emitter
        HasMclq2 = 0x400000,       // Has another liquid
        HasMcal2 = 0x800000,       // Has another alpha map
        HasMcmt2 = 0x1000000,      // Has another texture map
        HasMcrd2 = 0x2000000,      // Has another render data
        HasMcbb2 = 0x4000000,      // Has another bounding box
        HasMcbn2 = 0x8000000,      // Has another BSP tree
        HasMcrf2 = 0x10000000,     // Has another render flags
        HasMcsh5 = 0x20000000,     // Has yet another shadow map
        HasMcsh6 = 0x40000000,     // Has yet another shadow map
        HasMcsh7 = 0x80000000      // Has yet another shadow map
    }

    /// <summary>
    /// MCNK header structure.
    /// </summary>
    public struct McnkHeader
    {
        public McnkFlags Flags;             // Flags
        public uint IndexX;                 // X index in the map (0-15)
        public uint IndexY;                 // Y index in the map (0-15)
        public uint LayerCount;             // Number of texture layers
        public uint DoodadRefs;             // Number of doodad references
        public uint McvtOffset;             // Offset to MCVT subchunk (vertices)
        public uint McnrOffset;             // Offset to MCNR subchunk (normals)
        public uint MclyOffset;             // Offset to MCLY subchunk (texture layers)
        public uint McrfOffset;             // Offset to MCRF subchunk (render flags)
        public uint McalOffset;             // Offset to MCAL subchunk (alpha maps)
        public uint SizeAlpha;              // Size of alpha map
        public uint McshOffset;             // Offset to MCSH subchunk (shadow map)
        public uint SizeShadow;             // Size of shadow map
        public uint AreaId;                 // Area ID (zones)
        public uint MapObjectRefsCount;     // Number of WMO references
        public uint HolesLowRes;            // Low resolution holes (4x4)
        public ushort PredTexture;          // Low resolution texture map
        public ushort NoEffectDoodad;       // Number of doodads without MDX effects
        public uint McseOffset;             // Offset to MCSE subchunk (sound emitters)
        public uint NumSoundEmitters;       // Number of sound emitters
        public uint MclqOffset;             // Offset to MCLQ subchunk (liquid)
        public uint SizeLiquid;             // Size of liquid map
        public Vector3F Position;           // Position of the chunk
        public uint MccvOffset;             // Offset to MCCV subchunk (vertex colors)
        public uint McbbOffset;             // Offset to MCBB subchunk (bounding box)
        public uint MclvOffset;             // Offset to MCLV subchunk (light refs)
        public uint Unused;                 // Padding
    }

    /// <summary>
    /// Gets the chunk header.
    /// </summary>
    public McnkHeader Header { get; private set; }

    /// <summary>
    /// Gets the MCVT subchunk (vertices).
    /// </summary>
    public McvtChunk? Mcvt { get; private set; }

    /// <summary>
    /// Gets the MCNR subchunk (normals).
    /// </summary>
    public McnrChunk? Mcnr { get; private set; }

    /// <summary>
    /// Gets the MCLY subchunk (texture layers).
    /// </summary>
    public MclyChunk? Mcly { get; private set; }

    /// <summary>
    /// Gets the MCRF subchunk (render flags).
    /// </summary>
    public McrfChunk? Mcrf { get; private set; }

    /// <summary>
    /// Gets the MCSH subchunk (shadow map).
    /// </summary>
    public McshChunk? Mcsh { get; private set; }

    /// <summary>
    /// Gets the MCAL subchunk (alpha maps).
    /// </summary>
    public McalChunk? Mcal { get; private set; }

    public override void Parse(BinaryReader reader, uint size)
    {
        var startPosition = reader.BaseStream.Position;

        // Read header
        Header = new McnkHeader
        {
            Flags = (McnkFlags)reader.ReadUInt32(),
            IndexX = reader.ReadUInt32(),
            IndexY = reader.ReadUInt32(),
            LayerCount = reader.ReadUInt32(),
            DoodadRefs = reader.ReadUInt32(),
            McvtOffset = reader.ReadUInt32(),
            McnrOffset = reader.ReadUInt32(),
            MclyOffset = reader.ReadUInt32(),
            McrfOffset = reader.ReadUInt32(),
            McalOffset = reader.ReadUInt32(),
            SizeAlpha = reader.ReadUInt32(),
            McshOffset = reader.ReadUInt32(),
            SizeShadow = reader.ReadUInt32(),
            AreaId = reader.ReadUInt32(),
            MapObjectRefsCount = reader.ReadUInt32(),
            HolesLowRes = reader.ReadUInt32(),
            PredTexture = reader.ReadUInt16(),
            NoEffectDoodad = reader.ReadUInt16(),
            McseOffset = reader.ReadUInt32(),
            NumSoundEmitters = reader.ReadUInt32(),
            MclqOffset = reader.ReadUInt32(),
            SizeLiquid = reader.ReadUInt32(),
            Position = reader.ReadVector3F(),
            McbbOffset = reader.ReadUInt32(),
            MccvOffset = reader.ReadUInt32(),
            MclvOffset = reader.ReadUInt32(),
            Unused = reader.ReadUInt32()
        };

        // Read subchunks based on header offsets
        if (Header.McvtOffset > 0)
        {
            reader.BaseStream.Position = startPosition + Header.McvtOffset;
            Mcvt = new McvtChunk();
            Mcvt.Parse(reader, 0); // Size will be determined by the subchunk
        }

        if (Header.McnrOffset > 0)
        {
            reader.BaseStream.Position = startPosition + Header.McnrOffset;
            Mcnr = new McnrChunk();
            Mcnr.Parse(reader, 0);
        }

        if (Header.MclyOffset > 0)
        {
            reader.BaseStream.Position = startPosition + Header.MclyOffset;
            Mcly = new MclyChunk();
            Mcly.Parse(reader, Header.LayerCount * 16); // Each layer entry is 16 bytes
        }

        if (Header.McrfOffset > 0)
        {
            reader.BaseStream.Position = startPosition + Header.McrfOffset;
            Mcrf = new McrfChunk();
            Mcrf.Parse(reader, 64); // 64 render flags
        }

        if (Header.McshOffset > 0 && Header.SizeShadow > 0)
        {
            reader.BaseStream.Position = startPosition + Header.McshOffset;
            Mcsh = new McshChunk();
            Mcsh.Parse(reader, Header.SizeShadow);
        }

        if (Header.McalOffset > 0 && Header.SizeAlpha > 0)
        {
            reader.BaseStream.Position = startPosition + Header.McalOffset;
            Mcal = new McalChunk();
            Mcal.Parse(reader, Header.SizeAlpha);
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        var startPosition = writer.BaseStream.Position;

        // Write header
        writer.Write((uint)Header.Flags);
        writer.Write(Header.IndexX);
        writer.Write(Header.IndexY);
        writer.Write(Header.LayerCount);
        writer.Write(Header.DoodadRefs);
        writer.Write(Header.McvtOffset);
        writer.Write(Header.McnrOffset);
        writer.Write(Header.MclyOffset);
        writer.Write(Header.McrfOffset);
        writer.Write(Header.McalOffset);
        writer.Write(Header.SizeAlpha);
        writer.Write(Header.McshOffset);
        writer.Write(Header.SizeShadow);
        writer.Write(Header.AreaId);
        writer.Write(Header.MapObjectRefsCount);
        writer.Write(Header.HolesLowRes);
        writer.Write(Header.PredTexture);
        writer.Write(Header.NoEffectDoodad);
        writer.Write(Header.McseOffset);
        writer.Write(Header.NumSoundEmitters);
        writer.Write(Header.MclqOffset);
        writer.Write(Header.SizeLiquid);
        writer.WriteVector3F(Header.Position);
        writer.Write(Header.McbbOffset);
        writer.Write(Header.MccvOffset);
        writer.Write(Header.MclvOffset);
        writer.Write(Header.Unused);

        // Write subchunks
        if (Mcvt != null)
        {
            writer.BaseStream.Position = startPosition + Header.McvtOffset;
            Mcvt.Write(writer);
        }

        if (Mcnr != null)
        {
            writer.BaseStream.Position = startPosition + Header.McnrOffset;
            Mcnr.Write(writer);
        }

        if (Mcly != null)
        {
            writer.BaseStream.Position = startPosition + Header.MclyOffset;
            Mcly.Write(writer);
        }

        if (Mcrf != null)
        {
            writer.BaseStream.Position = startPosition + Header.McrfOffset;
            Mcrf.Write(writer);
        }

        if (Mcsh != null)
        {
            writer.BaseStream.Position = startPosition + Header.McshOffset;
            Mcsh.Write(writer);
        }

        if (Mcal != null)
        {
            writer.BaseStream.Position = startPosition + Header.McalOffset;
            Mcal.Write(writer);
        }
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Map Chunk ({Header.IndexX}, {Header.IndexY})");
        builder.AppendLine($"Position: {Header.Position}");
        builder.AppendLine($"Flags: {Header.Flags}");
        builder.AppendLine($"Area ID: {Header.AreaId}");
        builder.AppendLine($"Layer Count: {Header.LayerCount}");
        builder.AppendLine($"Doodad References: {Header.DoodadRefs}");
        builder.AppendLine($"WMO References: {Header.MapObjectRefsCount}");
        builder.AppendLine($"Sound Emitters: {Header.NumSoundEmitters}");
        builder.AppendLine($"Has Holes: {(Header.HolesLowRes != 0 ? "Yes" : "No")}");

        if (Mcvt != null) builder.AppendLine("\n" + Mcvt.ToHumanReadable());
        if (Mcnr != null) builder.AppendLine("\n" + Mcnr.ToHumanReadable());
        if (Mcly != null) builder.AppendLine("\n" + Mcly.ToHumanReadable());
        if (Mcrf != null) builder.AppendLine("\n" + Mcrf.ToHumanReadable());
        if (Mcsh != null) builder.AppendLine("\n" + Mcsh.ToHumanReadable());
        if (Mcal != null) builder.AppendLine("\n" + Mcal.ToHumanReadable());

        return builder.ToString();
    }
} 