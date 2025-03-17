using System;
using System.Collections.Generic;
using System.Numerics;

namespace WCAnalyzer.Core.Models
{
    /// <summary>
    /// Terrain Level of Detail information.
    /// </summary>
    public class TerrainLod
    {
        /// <summary>
        /// Gets or sets the terrain LOD header.
        /// </summary>
        public TerrainLodHeader Header { get; set; } = new TerrainLodHeader();
        
        /// <summary>
        /// Gets or sets the height data.
        /// </summary>
        public float[] HeightData { get; set; } = Array.Empty<float>();
        
        /// <summary>
        /// Gets or sets the LOD levels.
        /// </summary>
        public List<TerrainLodLevel> Levels { get; set; } = new List<TerrainLodLevel>();
        
        /// <summary>
        /// Gets or sets the LOD nodes.
        /// </summary>
        public List<TerrainLodNode> Nodes { get; set; } = new List<TerrainLodNode>();
        
        /// <summary>
        /// Gets or sets the vertex indices.
        /// </summary>
        public ushort[] VertexIndices { get; set; } = Array.Empty<ushort>();
        
        /// <summary>
        /// Gets or sets the skirt indices.
        /// </summary>
        public ushort[] SkirtIndices { get; set; } = Array.Empty<ushort>();
        
        /// <summary>
        /// Gets or sets the liquid data.
        /// </summary>
        public TerrainLodLiquidData LiquidData { get; set; } = new TerrainLodLiquidData();
        
        /// <summary>
        /// Gets or sets the liquid node.
        /// </summary>
        public TerrainLodLiquidNode LiquidNode { get; set; } = new TerrainLodLiquidNode();
        
        /// <summary>
        /// Gets or sets the liquid indices.
        /// </summary>
        public Vector3[] LiquidIndices { get; set; } = Array.Empty<Vector3>();
        
        /// <summary>
        /// Gets or sets the liquid vertices.
        /// </summary>
        public Vector3[] LiquidVertices { get; set; } = Array.Empty<Vector3>();
    }
    
    /// <summary>
    /// Terrain LOD header information.
    /// </summary>
    public class TerrainLodHeader
    {
        /// <summary>
        /// Gets or sets the flags.
        /// </summary>
        public uint Flags { get; set; }
        
        /// <summary>
        /// Gets or sets the bounding box.
        /// </summary>
        public BoundingBox BoundingBox { get; set; } = new BoundingBox();
    }
    
    /// <summary>
    /// Terrain LOD level information.
    /// </summary>
    public class TerrainLodLevel
    {
        /// <summary>
        /// Gets or sets the LOD bands.
        /// </summary>
        public float LodBands { get; set; }
        
        /// <summary>
        /// Gets or sets the height length.
        /// </summary>
        public uint HeightLength { get; set; }
        
        /// <summary>
        /// Gets or sets the height index.
        /// </summary>
        public uint HeightIndex { get; set; }
        
        /// <summary>
        /// Gets or sets the map area low length.
        /// </summary>
        public uint MapAreaLowLength { get; set; }
        
        /// <summary>
        /// Gets or sets the map area low index.
        /// </summary>
        public uint MapAreaLowIndex { get; set; }
    }
    
    /// <summary>
    /// Terrain LOD node information.
    /// </summary>
    public class TerrainLodNode
    {
        /// <summary>
        /// Gets or sets the vertex indices offset.
        /// </summary>
        public uint VertexIndicesOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the vertex indices length.
        /// </summary>
        public uint VertexIndicesLength { get; set; }
        
        /// <summary>
        /// Gets or sets the unknown value 1.
        /// </summary>
        public uint Unknown1 { get; set; }
        
        /// <summary>
        /// Gets or sets the unknown value 2.
        /// </summary>
        public uint Unknown2 { get; set; }
        
        /// <summary>
        /// Gets or sets the child indices.
        /// </summary>
        public ushort[] ChildIndices { get; set; } = new ushort[4];
    }
    
    /// <summary>
    /// Terrain LOD liquid data.
    /// </summary>
    public class TerrainLodLiquidData
    {
        /// <summary>
        /// Gets or sets the flags.
        /// </summary>
        public uint Flags { get; set; }
        
        /// <summary>
        /// Gets or sets the depth chunk size.
        /// </summary>
        public ushort DepthChunkSize { get; set; }
        
        /// <summary>
        /// Gets or sets the alpha chunk size.
        /// </summary>
        public ushort AlphaChunkSize { get; set; }
        
        /// <summary>
        /// Gets or sets the depth chunk data.
        /// </summary>
        public byte[] DepthChunkData { get; set; } = Array.Empty<byte>();
        
        /// <summary>
        /// Gets or sets the alpha chunk data.
        /// </summary>
        public byte[] AlphaChunkData { get; set; } = Array.Empty<byte>();
    }
    
    /// <summary>
    /// Terrain LOD liquid node.
    /// </summary>
    public class TerrainLodLiquidNode
    {
        /// <summary>
        /// Unknown value 1.
        /// </summary>
        public uint Unknown1 { get; set; }
        
        /// <summary>
        /// MLLI length.
        /// </summary>
        public uint MlliLength { get; set; }
        
        /// <summary>
        /// Unknown value 3.
        /// </summary>
        public uint Unknown3 { get; set; }
        
        /// <summary>
        /// Unknown value 4a.
        /// </summary>
        public ushort Unknown4a { get; set; }
        
        /// <summary>
        /// Unknown value 4b.
        /// </summary>
        public ushort Unknown4b { get; set; }
        
        /// <summary>
        /// Unknown value 5.
        /// </summary>
        public uint Unknown5 { get; set; }
        
        /// <summary>
        /// Unknown value 6.
        /// </summary>
        public uint Unknown6 { get; set; }
    }
} 