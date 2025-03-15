using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace WCAnalyzer.Core.Models
{
    /// <summary>
    /// Represents information about an ADT file.
    /// </summary>
    public class AdtInfo
    {
        /// <summary>
        /// Gets or sets the full path to the ADT file.
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the file name of the ADT file.
        /// </summary>
        public string? FileName { get; set; }

        /// <summary>
        /// Gets or sets the version of the ADT file.
        /// </summary>
        public int Version { get; set; }

        /// <summary>
        /// Gets or sets the flags of the ADT file.
        /// </summary>
        public uint Flags { get; set; }

        /// <summary>
        /// Gets or sets the number of terrain chunks in the ADT file.
        /// </summary>
        public int TerrainChunks { get; set; }

        /// <summary>
        /// Gets or sets the list of texture names referenced in the ADT file.
        /// </summary>
        public List<string> TextureNames { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the list of model names referenced in the ADT file.
        /// </summary>
        public List<string> ModelNames { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the list of WMO names referenced in the ADT file.
        /// </summary>
        public List<string> WmoNames { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the number of model placements in the ADT file.
        /// </summary>
        public int ModelPlacements { get; set; }

        /// <summary>
        /// Gets or sets the number of WMO placements in the ADT file.
        /// </summary>
        public int WmoPlacements { get; set; }

        /// <summary>
        /// Gets or sets the list of area IDs found in the ADT file.
        /// </summary>
        public List<int> AreaIds { get; set; } = new List<int>();

        /// <summary>
        /// Gets or sets the list of unique IDs found in the ADT file.
        /// </summary>
        public List<int> UniqueIds { get; set; } = new List<int>();

        /// <summary>
        /// Gets or sets the list of model FileDataIDs referenced in the ADT file.
        /// </summary>
        public HashSet<uint> ReferencedModels { get; set; } = new HashSet<uint>();

        /// <summary>
        /// Gets or sets the list of WMO FileDataIDs referenced in the ADT file.
        /// </summary>
        public HashSet<uint> ReferencedWmos { get; set; } = new HashSet<uint>();

        /// <summary>
        /// Gets or sets detailed information about terrain chunks.
        /// </summary>
        public List<TerrainChunkInfo> TerrainChunkDetails { get; set; } = new List<TerrainChunkInfo>();

        /// <summary>
        /// Gets or sets detailed information about model placements.
        /// </summary>
        public List<ModelPlacementInfo> ModelPlacementDetails { get; set; } = new List<ModelPlacementInfo>();

        /// <summary>
        /// Gets or sets detailed information about WMO placements.
        /// </summary>
        public List<WmoPlacementInfo> WmoPlacementDetails { get; set; } = new List<WmoPlacementInfo>();
        
        /// <summary>
        /// Gets or sets a dictionary of additional properties for the ADT file.
        /// </summary>
        /// <remarks>
        /// Used to store additional information that doesn't fit into the standard properties,
        /// such as whether the file uses FileDataIDs.
        /// </remarks>
        public Dictionary<string, object> Properties { get; set; } = new Dictionary<string, object>();
        
        /// <summary>
        /// Gets whether the ADT file uses FileDataIDs.
        /// </summary>
        public bool UsesFileDataId 
        {
            get
            {
                if (Properties.TryGetValue("UsesFileDataId", out var value) && value is bool boolValue)
                {
                    return boolValue;
                }
                return false;
            }
            set
            {
                Properties["UsesFileDataId"] = value;
            }
        }

        /// <summary>
        /// Terrain Level of Detail data from ML chunks
        /// </summary>
        public TerrainLod? TerrainLod { get; set; }
    }

    /// <summary>
    /// Represents detailed information about a terrain chunk.
    /// </summary>
    public class TerrainChunkInfo
    {
        /// <summary>
        /// Gets or sets the area ID of the terrain chunk.
        /// </summary>
        public int AreaId { get; set; }

        /// <summary>
        /// Gets or sets the flags of the terrain chunk.
        /// </summary>
        public int Flags { get; set; }

        /// <summary>
        /// Gets or sets the X position of the terrain chunk.
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Gets or sets the Y position of the terrain chunk.
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// Gets or sets the Z position of the terrain chunk.
        /// </summary>
        public float Z { get; set; }

        /// <summary>
        /// Gets the position of the terrain chunk as a Vector3.
        /// </summary>
        public Vector3 WorldPosition => new Vector3(X, Y, Z);
        
        /// <summary>
        /// Gets or sets the height values for the terrain chunk (MCVT).
        /// </summary>
        public float[] Heights { get; set; } = Array.Empty<float>();
        
        /// <summary>
        /// Gets or sets the normal vectors for the terrain chunk (MCNR).
        /// </summary>
        public Vector3[] Normals { get; set; } = Array.Empty<Vector3>();
        
        /// <summary>
        /// Gets or sets the texture layers for the terrain chunk (MCLY).
        /// </summary>
        public List<TextureLayerInfo> TextureLayers { get; set; } = new List<TextureLayerInfo>();
        
        /// <summary>
        /// Gets or sets the alpha map data for the terrain chunk (MCAL).
        /// </summary>
        public byte[] AlphaMapData { get; set; } = Array.Empty<byte>();
        
        /// <summary>
        /// Gets or sets the shadow map data for the terrain chunk (MCSH).
        /// </summary>
        public byte[] ShadowMapData { get; set; } = Array.Empty<byte>();
    }

    /// <summary>
    /// Represents information about a texture layer in a terrain chunk.
    /// </summary>
    public class TextureLayerInfo
    {
        /// <summary>
        /// Gets or sets the texture ID (index into the texture name list).
        /// </summary>
        public uint TextureId { get; set; }
        
        /// <summary>
        /// Gets or sets the flags for the texture layer.
        /// </summary>
        public uint Flags { get; set; }
        
        /// <summary>
        /// Gets or sets the offset in the MCAL chunk for this layer's alpha map.
        /// </summary>
        public uint OffsetInMCAL { get; set; }
        
        /// <summary>
        /// Gets or sets the effect ID for the texture layer.
        /// </summary>
        public int EffectId { get; set; }
    }

    /// <summary>
    /// Represents detailed information about a model placement.
    /// </summary>
    public class ModelPlacementInfo
    {
        /// <summary>
        /// Gets or sets the index into the model name list.
        /// </summary>
        /// <remarks>
        /// If NameIdIsFileDataId is true, this value is a direct FileDataID instead of an index.
        /// </remarks>
        public uint NameId { get; set; }

        /// <summary>
        /// Gets or sets the unique ID of the model placement.
        /// </summary>
        public int UniqueId { get; set; }

        /// <summary>
        /// Gets or sets the position of the model placement.
        /// </summary>
        public Vector3 Position { get; set; }

        /// <summary>
        /// Gets or sets the rotation of the model placement.
        /// </summary>
        public Vector3 Rotation { get; set; }

        /// <summary>
        /// Gets or sets the scale of the model placement.
        /// </summary>
        /// <remarks>
        /// This is calculated from the ushort ScalingFactor where 1024 = 1.0f
        /// </remarks>
        public float Scale { get; set; }

        /// <summary>
        /// Gets or sets the flags of the model placement.
        /// </summary>
        /// <remarks>
        /// In Warcraft.NET, this is MDDFFlags, which is a ushort enum.
        /// </remarks>
        public ushort Flags { get; set; }
        
        /// <summary>
        /// Gets or sets whether the NameId is a FileDataID.
        /// </summary>
        /// <remarks>
        /// In newer versions of the ADT format, NameId can be a direct FileDataID
        /// instead of an index into the model name list.
        /// This is determined by checking if the 0x40 bit is set in Flags (MDDFFlags.NameIdIsFiledataId).
        /// </remarks>
        public bool NameIdIsFileDataId { get; set; }
    }

    /// <summary>
    /// Represents detailed information about a WMO placement.
    /// </summary>
    public class WmoPlacementInfo
    {
        /// <summary>
        /// Gets or sets the index into the WMO name list.
        /// </summary>
        /// <remarks>
        /// If NameIdIsFileDataId is true, this value is a direct FileDataID instead of an index.
        /// </remarks>
        public uint NameId { get; set; }

        /// <summary>
        /// Gets or sets the unique ID of the WMO placement.
        /// </summary>
        public int UniqueId { get; set; }

        /// <summary>
        /// Gets or sets the position of the WMO placement.
        /// </summary>
        public Vector3 Position { get; set; }

        /// <summary>
        /// Gets or sets the rotation of the WMO placement.
        /// </summary>
        public Vector3 Rotation { get; set; }

        /// <summary>
        /// Gets or sets the first bounding box corner of the WMO placement.
        /// </summary>
        public Vector3 BoundingBox1 { get; set; }

        /// <summary>
        /// Gets or sets the second bounding box corner of the WMO placement.
        /// </summary>
        public Vector3 BoundingBox2 { get; set; }

        /// <summary>
        /// Gets or sets the flags of the WMO placement.
        /// </summary>
        public ushort Flags { get; set; }

        /// <summary>
        /// Gets or sets the doodad set of the WMO placement.
        /// </summary>
        public ushort DoodadSet { get; set; }

        /// <summary>
        /// Gets or sets the name set of the WMO placement.
        /// </summary>
        public ushort NameSet { get; set; }

        /// <summary>
        /// Gets or sets the scale of the WMO placement.
        /// </summary>
        public ushort Scale { get; set; }
        
        /// <summary>
        /// Gets or sets whether the NameId is a FileDataID.
        /// </summary>
        /// <remarks>
        /// In newer versions of the ADT format, NameId can be a direct FileDataID
        /// instead of an index into the WMO name list.
        /// </remarks>
        public bool NameIdIsFileDataId { get; set; }
    }
} 