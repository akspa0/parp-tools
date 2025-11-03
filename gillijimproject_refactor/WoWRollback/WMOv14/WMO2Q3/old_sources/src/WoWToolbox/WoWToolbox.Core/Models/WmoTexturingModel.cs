using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.Models
{
    /// <summary>
    /// Represents a block of texture file paths (WmoMotxChunk, formerly MOTX).
    /// </summary>
    public class WmoTextureBlock
    {
        /// <summary>
        /// List of texture file paths, in order as stored in WmoMotxChunk (MOTX).
        /// </summary>
        public List<string> TexturePaths { get; set; } = new();
    }

    /// <summary>
    /// Represents a single material entry (MOMT).
    /// </summary>
    public class WmoMaterial
    {
        public uint Flags { get; set; }
        public uint Shader { get; set; }
        public uint BlendMode { get; set; }
        public int Texture1Index { get; set; }
        public int Texture2Index { get; set; }
        public int Texture3Index { get; set; } // Optional/unused in v14
        public uint Color1 { get; set; }
        public uint Color1b { get; set; }
        public uint Color2 { get; set; }
        public uint Color3 { get; set; }
        public uint GroupType { get; set; }
        public uint Flags3 { get; set; }
        public uint[] RuntimeData { get; set; } = new uint[4];
        public List<uint> UnknownFields { get; set; } = new();
    }

    /// <summary>
    /// Represents the full set of materials for a WMO.
    /// </summary>
    public class WmoMaterialBlock
    {
        public List<WmoMaterial> Materials { get; set; } = new();
    }

    /// <summary>
    /// Represents a block of group names (MOGN).
    /// </summary>
    public class WmoGroupNameBlock
    {
        public List<string> GroupNames { get; set; } = new();
    }

    /// <summary>
    /// Represents a single group info entry (MOGI).
    /// </summary>
    public class WmoGroupInfo
    {
        public uint Flags { get; set; }
        public Vector3 BoundingBoxMin { get; set; }
        public Vector3 BoundingBoxMax { get; set; }
        public int NameIndex { get; set; } // Index into MOGN, or -1 if unused
    }

    /// <summary>
    /// Represents the full set of group info entries.
    /// </summary>
    public class WmoGroupInfoBlock
    {
        public List<WmoGroupInfo> Groups { get; set; } = new();
    }
} 