using System;
using System.Collections.Generic;
using System.Numerics;

namespace WCAnalyzer.Core.Models
{
    /// <summary>
    /// Represents the Level of Detail (LOD) data for terrain in ADT files.
    /// These chunks start with ML prefix and are used for terrain rendering at different distances.
    /// </summary>
    public class TerrainLod
    {
        /// <summary>
        /// MLHD: Header for the Level of Detail data containing basic information.
        /// </summary>
        public TerrainLodHeader Header { get; set; } = new TerrainLodHeader();
        
        /// <summary>
        /// MLVH: Heightmap data for LOD terrain.
        /// </summary>
        public float[] HeightData { get; set; } = Array.Empty<float>();
        
        /// <summary>
        /// MLLL: Level data defining distances at which different LOD levels are used.
        /// </summary>
        public List<TerrainLodLevel> Levels { get; set; } = new List<TerrainLodLevel>();
        
        /// <summary>
        /// MLND: Quadtree node data for LOD terrain rendering.
        /// </summary>
        public List<TerrainLodNode> Nodes { get; set; } = new List<TerrainLodNode>();
        
        /// <summary>
        /// MLVI: Vertex indices for LOD terrain geometry.
        /// </summary>
        public ushort[] VertexIndices { get; set; } = Array.Empty<ushort>();
        
        /// <summary>
        /// MLSI: Skirt indices for LOD terrain edges.
        /// </summary>
        public ushort[] SkirtIndices { get; set; } = Array.Empty<ushort>();
        
        /// <summary>
        /// MLLD: LOD liquid data.
        /// </summary>
        public TerrainLodLiquidData LiquidData { get; set; } = new TerrainLodLiquidData();
        
        /// <summary>
        /// MLLN: Data for LOD liquid meshes.
        /// </summary>
        public TerrainLodLiquidNode LiquidNode { get; set; } = new TerrainLodLiquidNode();
        
        /// <summary>
        /// MLLI: Indices for LOD liquid meshes.
        /// </summary>
        public Vector3[] LiquidIndices { get; set; } = Array.Empty<Vector3>();
        
        /// <summary>
        /// MLLV: Vertices for LOD liquid meshes.
        /// </summary>
        public Vector3[] LiquidVertices { get; set; } = Array.Empty<Vector3>();
    }
    
    /// <summary>
    /// MLHD: Header data for terrain LOD.
    /// </summary>
    public class TerrainLodHeader
    {
        /// <summary>
        /// Version or flags.
        /// </summary>
        public uint Flags { get; set; }
        
        /// <summary>
        /// Bounding box for the terrain.
        /// </summary>
        public BoundingBox BoundingBox { get; set; } = new BoundingBox();
    }
    
    /// <summary>
    /// MLLL: Level data for terrain LOD.
    /// </summary>
    public class TerrainLodLevel
    {
        /// <summary>
        /// LOD distance bands.
        /// </summary>
        public float LodBands { get; set; }
        
        /// <summary>
        /// Height data length.
        /// </summary>
        public uint HeightLength { get; set; }
        
        /// <summary>
        /// Height data index.
        /// </summary>
        public uint HeightIndex { get; set; }
        
        /// <summary>
        /// Map area low data length.
        /// </summary>
        public uint MapAreaLowLength { get; set; }
        
        /// <summary>
        /// Map area low data index.
        /// </summary>
        public uint MapAreaLowIndex { get; set; }
    }
    
    /// <summary>
    /// MLND: Node data for terrain LOD quadtree.
    /// </summary>
    public class TerrainLodNode
    {
        /// <summary>
        /// Vertex indices offset.
        /// </summary>
        public uint VertexIndicesOffset { get; set; }
        
        /// <summary>
        /// Vertex indices length.
        /// </summary>
        public uint VertexIndicesLength { get; set; }
        
        /// <summary>
        /// Unknown value 1.
        /// </summary>
        public uint Unknown1 { get; set; }
        
        /// <summary>
        /// Unknown value 2.
        /// </summary>
        public uint Unknown2 { get; set; }
        
        /// <summary>
        /// Child node indices (4 indices for quadtree children).
        /// </summary>
        public ushort[] ChildIndices { get; set; } = new ushort[4];
    }
    
    /// <summary>
    /// MLLD: Liquid data for terrain LOD.
    /// </summary>
    public class TerrainLodLiquidData
    {
        /// <summary>
        /// Flags for the liquid data.
        /// </summary>
        public uint Flags { get; set; }
        
        /// <summary>
        /// Size of the depth chunk.
        /// </summary>
        public ushort DepthChunkSize { get; set; }
        
        /// <summary>
        /// Approximate size of the alpha chunk.
        /// </summary>
        public ushort AlphaChunkSize { get; set; }
        
        /// <summary>
        /// Depth chunk data.
        /// </summary>
        public byte[] DepthChunkData { get; set; } = Array.Empty<byte>();
        
        /// <summary>
        /// Alpha chunk data.
        /// </summary>
        public byte[] AlphaChunkData { get; set; } = Array.Empty<byte>();
    }
    
    /// <summary>
    /// MLLN: Liquid node data for terrain LOD.
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
    
    /// <summary>
    /// Bounding box for 3D objects.
    /// </summary>
    public class BoundingBox
    {
        /// <summary>
        /// Minimum corner of the bounding box.
        /// </summary>
        public Vector3 Min { get; set; }
        
        /// <summary>
        /// Maximum corner of the bounding box.
        /// </summary>
        public Vector3 Max { get; set; }
    }
} 