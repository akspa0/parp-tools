// DTOs for ADT _obj0 Structure (TerrainObjectZero)
using System.Collections.Generic;
using System.Numerics;
using Warcraft.NET.Files.ADT.Chunks;
// using Warcraft.NET.Files.ADT.Common; // REMOVED - Namespace likely doesn't exist or isn't needed here

namespace WoWToolbox.FileDumper.DTOs
{
    // Top-level DTO for the TerrainObjectZero (_obj0.adt)
    public class AdtObj0Dto
    {
        public MverChunkDto? MVER { get; set; } // Reusing MVER DTO from Pm4FileDto if identical
        public MddfChunkDto? ModelPlacementInfo { get; set; }
        public ModfChunkDto? WorldModelObjectPlacementInfo { get; set; }
        // Add other potential chunks from TerrainObjectZero if needed
    }

    // --- DTOs for relevant _obj0 chunks ---

    // DTO for MDDF Chunk Data (Model Placement)
    public class MddfEntryDto
    {
        // Based on Warcraft.NET.Files.ADT.Chunks.MDDFEntry
        public uint NameId { get; set; }
        public uint UniqueID { get; set; }
        public Vector3 Position { get; set; }
        public Vector3 Rotation { get; set; }
        public ushort Scale { get; set; }
        public ushort Flags { get; set; }
        // Potentially add parsed Flags interpretation (e.g., IsEntryTrivialCost)
    }
    public class MddfChunkDto
    {
        public List<MddfEntryDto> MDDFEntries { get; set; } = new();
    }

    // DTO for MODF Chunk Data (WMO Placement)
    public class ModfEntryDto
    {
        // Based on Warcraft.NET.Files.ADT.Chunks.MODFEntry
        public uint NameId { get; set; }
        public uint UniqueId { get; set; }
        public Vector3 Position { get; set; }
        public Vector3 Rotation { get; set; }
        public Vector3 ExtentsMin { get; set; } // UpperCorner in Warcraft.NET struct?
        public Vector3 ExtentsMax { get; set; } // LowerCorner in Warcraft.NET struct?
        public ushort Flags { get; set; }
        public ushort DoodadSet { get; set; }
        public ushort NameSet { get; set; }
        public ushort Scale { get; set; }
        // Potentially add parsed Flags interpretation
    }
    public class ModfChunkDto
    {
        public List<ModfEntryDto> MODFEntries { get; set; } = new();
    }

    // Note: We might need to define a Vector3Dto if System.Numerics.Vector3 doesn't serialize well
    // or if we want specific formatting, but YamlDotNet usually handles it.
} 