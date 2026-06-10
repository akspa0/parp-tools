using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MCNK chunk in an ADT file, containing map chunk data.
    /// </summary>
    public class McnkChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MCNK";
        
        /// <summary>
        /// Flags for MCNK chunks.
        /// </summary>
        [Flags]
        public enum MCNKFlags : uint
        {
            /// <summary>
            /// No flags.
            /// </summary>
            None = 0x0,
            
            /// <summary>
            /// Has shadow map.
            /// </summary>
            HasShadowMap = 0x1,
            
            /// <summary>
            /// Impassible terrain.
            /// </summary>
            Impassible = 0x2,
            
            /// <summary>
            /// Has terrain in river.
            /// </summary>
            RiverTerrain = 0x4,
            
            /// <summary>
            /// Has terrain in ocean.
            /// </summary>
            OceanTerrain = 0x8,
            
            /// <summary>
            /// Has unspecified magma.
            /// </summary>
            Magma1 = 0x10,
            
            /// <summary>
            /// Has unspecified slime.
            /// </summary>
            Slime1 = 0x20,
            
            /// <summary>
            /// Has invisible terrain (0x1 or not).
            /// </summary>
            HasInvisibleTerrain = 0x40,
            
            /// <summary>
            /// Has quad tree set.
            /// </summary>
            HasQuadTree = 0x80,
            
            /// <summary>
            /// Has unspecified magma.
            /// </summary>
            Magma2 = 0x100,
            
            /// <summary>
            /// Has unspecified slime.
            /// </summary>
            Slime2 = 0x200,
            
            /// <summary>
            /// Has vertex normals.
            /// </summary>
            HasVertexNormals = 0x400,
            
            /// <summary>
            /// Has "low detail" height map.
            /// </summary>
            HasLowDetailHeightMap = 0x800,
            
            /// <summary>
            /// Has high-resolution hole data.
            /// </summary>
            HasHighResHoles = 0x1000,
            
            /// <summary>
            /// Bordering a chunk with low precision liquid.
            /// </summary>
            LowPrecisionOceanBorder = 0x2000,
            
            /// <summary>
            /// Uses 2 bits per hole.
            /// </summary>
            Fatigue1 = 0x4000,
            
            /// <summary>
            /// Uses 4 bits per hole.
            /// </summary>
            Fatigue2 = 0x8000,
            
            /// <summary>
            /// Has vertex colors in MCCV.
            /// </summary>
            HasVertexColors = 0x10000,
            
            /// <summary>
            /// Unused.
            /// </summary>
            Unused1 = 0x20000,
            
            /// <summary>
            /// Unused.
            /// </summary>
            Unused2 = 0x40000,
            
            /// <summary>
            /// Do not fix alpha map offset (ADT hack).
            /// </summary>
            DoNotFixAlphaMap = 0x80000,
            
            /// <summary>
            /// High resolution liquid and terrain blend.
            /// </summary>
            HighResLiquidAndBlend = 0x100000,
            
            /// <summary>
            /// MH2O is used or has game-driven liquid.
            /// </summary>
            HasMH2OOrGameLiquid = 0x200000,
            
            /// <summary>
            /// Unused.
            /// </summary>
            Unused3 = 0x400000,
            
            /// <summary>
            /// Unused.
            /// </summary>
            Unused4 = 0x800000,
            
            /// <summary>
            /// Has animated MCAL layers.
            /// </summary>
            HasMCALAnim = 0x1000000
        }

        /// <summary>
        /// Gets or sets the flags.
        /// </summary>
        public MCNKFlags Flags { get; set; }
        
        /// <summary>
        /// Gets or sets the index X of the chunk in the tile.
        /// </summary>
        public uint IndexX { get; set; }
        
        /// <summary>
        /// Gets or sets the index Y of the chunk in the tile.
        /// </summary>
        public uint IndexY { get; set; }
        
        /// <summary>
        /// Gets or sets the layer count.
        /// </summary>
        public uint LayerCount { get; set; }
        
        /// <summary>
        /// Gets or sets the doodad reference count.
        /// </summary>
        public uint DoodadRefCount { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCVT (height map) sub-chunk.
        /// </summary>
        public uint McvtOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCNR (normals) sub-chunk.
        /// </summary>
        public uint McnrOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCLY (texture layers) sub-chunk.
        /// </summary>
        public uint MclyOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCRF (doodad references) sub-chunk.
        /// </summary>
        public uint McrfOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCSH (shadow map) sub-chunk.
        /// </summary>
        public uint McshOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCAL (alpha maps) sub-chunk.
        /// </summary>
        public uint McalOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the size of the alpha maps.
        /// </summary>
        public uint SizeAlpha { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCLQ (liquid) sub-chunk.
        /// </summary>
        public uint MclqOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the size of the liquid data.
        /// </summary>
        public uint SizeLiquid { get; set; }
        
        /// <summary>
        /// Gets or sets the position of the chunk.
        /// </summary>
        public Vector3 Position { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCSE (sound emitters) sub-chunk.
        /// </summary>
        public uint McseOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the reference count for sound emitters.
        /// </summary>
        public uint McseCount { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCCV (vertex colors) sub-chunk.
        /// </summary>
        public uint MccvOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the offset to MCLV (light sub-chunk).
        /// </summary>
        public uint MclvOffset { get; set; }
        
        /// <summary>
        /// Gets or sets the 9x9 height map values.
        /// </summary>
        public float[] HeightMap { get; private set; } = new float[145]; // 9x9 + 8x8 grid
        
        /// <summary>
        /// Gets the texture layers.
        /// </summary>
        public List<TextureLayerDefinition> Layers { get; } = new List<TextureLayerDefinition>();
        
        /// <summary>
        /// Gets or sets the hole bitmap (low resolution).
        /// </summary>
        public ushort HoleBitmap { get; set; }
        
        /// <summary>
        /// Gets or sets the high resolution hole data.
        /// </summary>
        public byte[] HighResHoles { get; set; } = Array.Empty<byte>();
        
        /// <summary>
        /// Gets or sets the doodad references.
        /// </summary>
        public uint[] DoodadReferences { get; set; } = Array.Empty<uint>();
        
        /// <summary>
        /// Definition of a texture layer.
        /// </summary>
        public class TextureLayerDefinition
        {
            /// <summary>
            /// Gets or sets the texture ID.
            /// </summary>
            public uint TextureId { get; set; }
            
            /// <summary>
            /// Gets or sets the flags.
            /// </summary>
            public uint Flags { get; set; }
            
            /// <summary>
            /// Gets or sets the offset in MCAL data.
            /// </summary>
            public uint OffsetInMcal { get; set; }
            
            /// <summary>
            /// Gets or sets the effect ID.
            /// </summary>
            public uint EffectId { get; set; }
        }
        
        /// <summary>
        /// Creates a new MCNK chunk
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McnkChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }
        
        /// <summary>
        /// Parses the chunk data
        /// </summary>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public override bool Parse()
        {
            try
            {
                using (MemoryStream ms = new MemoryStream(Data))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    // Read the header
                    Flags = (MCNKFlags)reader.ReadUInt32();
                    IndexX = reader.ReadUInt32();
                    IndexY = reader.ReadUInt32();
                    LayerCount = reader.ReadUInt32();
                    DoodadRefCount = reader.ReadUInt32();
                    McvtOffset = reader.ReadUInt32();
                    McnrOffset = reader.ReadUInt32();
                    MclyOffset = reader.ReadUInt32();
                    McrfOffset = reader.ReadUInt32();
                    McalOffset = reader.ReadUInt32();
                    SizeAlpha = reader.ReadUInt32();
                    McshOffset = reader.ReadUInt32();
                    SizeLiquid = reader.ReadUInt32();
                    
                    // Read the position
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    Position = new Vector3(x, y, z);
                    
                    // Read the remaining fields
                    McseOffset = reader.ReadUInt32();
                    McseCount = reader.ReadUInt32();
                    MccvOffset = reader.ReadUInt32();
                    MclvOffset = reader.ReadUInt32();
                    reader.ReadUInt32(); // Unused
                    
                    // Read the hole bitmap
                    HoleBitmap = reader.ReadUInt16();
                    
                    // Skip unused bytes
                    reader.BaseStream.Position += 2;
                    
                    // Parse sub-chunks
                    if (McvtOffset > 0)
                    {
                        ParseMcvt(reader, McvtOffset);
                    }
                    
                    if (MclyOffset > 0 && LayerCount > 0)
                    {
                        ParseMcly(reader, MclyOffset, LayerCount);
                    }
                    
                    if (McrfOffset > 0 && DoodadRefCount > 0)
                    {
                        ParseMcrf(reader, McrfOffset, DoodadRefCount);
                    }
                    
                    IsParsed = true;
                    return true;
                }
            }
            catch (Exception ex)
            {
                LogError($"Error parsing MCNK chunk: {ex.Message}");
                return false;
            }
        }
        
        private void ParseMcvt(BinaryReader reader, uint offset)
        {
            // Save the current position
            long currentPosition = reader.BaseStream.Position;
            
            // Seek to the MCVT sub-chunk
            reader.BaseStream.Position = offset;
            
            // Read the height map (145 floats)
            for (int i = 0; i < 145; i++)
            {
                HeightMap[i] = reader.ReadSingle();
            }
            
            // Restore the position
            reader.BaseStream.Position = currentPosition;
        }
        
        private void ParseMcly(BinaryReader reader, uint offset, uint layerCount)
        {
            // Save the current position
            long currentPosition = reader.BaseStream.Position;
            
            // Seek to the MCLY sub-chunk
            reader.BaseStream.Position = offset;
            
            // Read the texture layers
            for (int i = 0; i < layerCount; i++)
            {
                TextureLayerDefinition layer = new TextureLayerDefinition
                {
                    TextureId = reader.ReadUInt32(),
                    Flags = reader.ReadUInt32(),
                    OffsetInMcal = reader.ReadUInt32(),
                    EffectId = reader.ReadUInt32()
                };
                
                Layers.Add(layer);
            }
            
            // Restore the position
            reader.BaseStream.Position = currentPosition;
        }
        
        private void ParseMcrf(BinaryReader reader, uint offset, uint count)
        {
            // Save the current position
            long currentPosition = reader.BaseStream.Position;
            
            // Seek to the MCRF sub-chunk
            reader.BaseStream.Position = offset;
            
            // Read the doodad references
            DoodadReferences = new uint[count];
            for (int i = 0; i < count; i++)
            {
                DoodadReferences[i] = reader.ReadUInt32();
            }
            
            // Restore the position
            reader.BaseStream.Position = currentPosition;
        }
        
        /// <summary>
        /// Writes the chunk data
        /// </summary>
        /// <returns>Byte array containing the chunk data</returns>
        public override byte[] Write()
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter writer = new BinaryWriter(ms))
            {
                // Write the header
                writer.Write((uint)Flags);
                writer.Write(IndexX);
                writer.Write(IndexY);
                writer.Write(LayerCount);
                writer.Write(DoodadRefCount);
                writer.Write(McvtOffset);
                writer.Write(McnrOffset);
                writer.Write(MclyOffset);
                writer.Write(McrfOffset);
                writer.Write(McalOffset);
                writer.Write(SizeAlpha);
                writer.Write(McshOffset);
                writer.Write(SizeLiquid);
                
                // Write the position
                writer.Write(Position.X);
                writer.Write(Position.Y);
                writer.Write(Position.Z);
                
                // Write the remaining fields
                writer.Write(McseOffset);
                writer.Write(McseCount);
                writer.Write(MccvOffset);
                writer.Write(MclvOffset);
                writer.Write(0U); // Unused
                
                // Write the hole bitmap
                writer.Write(HoleBitmap);
                
                // Write unused bytes
                writer.Write((ushort)0);
                
                // At this point, we'd need to write the sub-chunks as well,
                // but this is a simplified implementation for now
                
                // TODO: Write MCVT (HeightMap)
                // TODO: Write MCLY (Layers)
                // TODO: Write MCRF (DoodadReferences)
                
                return ms.ToArray();
            }
        }
        
        /// <summary>
        /// Returns a string representation of this chunk
        /// </summary>
        public override string ToString()
        {
            return $"{SIGNATURE} (Pos: {Position}, Flags: {Flags})";
        }
    }
} 