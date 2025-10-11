using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MH2O chunk in an ADT file, containing liquid (water) data.
    /// </summary>
    public class Mh2oChunk : ADTChunk
    {
        /// <summary>
        /// The MH2O chunk signature
        /// </summary>
        public const string SIGNATURE = "MH2O";

        /// <summary>
        /// Class representing liquid vertex data.
        /// </summary>
        public class LiquidVertexData
        {
            /// <summary>
            /// Gets or sets the height values.
            /// </summary>
            public float[] HeightValues { get; set; } = Array.Empty<float>();
            
            /// <summary>
            /// Gets or sets the depth values.
            /// </summary>
            public float[] DepthValues { get; set; } = Array.Empty<float>();
            
            /// <summary>
            /// Gets or sets the texture coordinate values.
            /// </summary>
            public float[] TextureCoords { get; set; } = Array.Empty<float>();
        }
        
        /// <summary>
        /// Class representing a liquid instance.
        /// </summary>
        public class LiquidInstance
        {
            /// <summary>
            /// Type of liquid.
            /// </summary>
            public enum LiquidType : ushort
            {
                /// <summary>
                /// No liquid.
                /// </summary>
                None = 0,
                
                /// <summary>
                /// Water.
                /// </summary>
                Water = 1,
                
                /// <summary>
                /// Ocean.
                /// </summary>
                Ocean = 2,
                
                /// <summary>
                /// Magma.
                /// </summary>
                Magma = 3,
                
                /// <summary>
                /// Slime.
                /// </summary>
                Slime = 4,
                
                /// <summary>
                /// Naxxramas slime.
                /// </summary>
                NaxxramasSlime = 21
            }
            
            /// <summary>
            /// Gets or sets the liquid type.
            /// </summary>
            public LiquidType Type { get; set; }
            
            /// <summary>
            /// Gets or sets the liquid flags.
            /// </summary>
            public ushort Flags { get; set; }
            
            /// <summary>
            /// Gets or sets the liquid vertex format.
            /// </summary>
            public byte LiquidVertexFormat { get; set; }
            
            /// <summary>
            /// Gets or sets the height level.
            /// </summary>
            public byte HeightLevel1 { get; set; }
            
            /// <summary>
            /// Gets or sets the X coordinate where the instance starts.
            /// </summary>
            public byte OffsetX { get; set; }
            
            /// <summary>
            /// Gets or sets the Y coordinate where the instance starts.
            /// </summary>
            public byte OffsetY { get; set; }
            
            /// <summary>
            /// Gets or sets the width of the instance (in vertices).
            /// </summary>
            public byte Width { get; set; }
            
            /// <summary>
            /// Gets or sets the height of the instance (in vertices).
            /// </summary>
            public byte Height { get; set; }
            
            /// <summary>
            /// Gets or sets the offset to vertex data.
            /// </summary>
            public uint VertexDataOffset { get; set; }
            
            /// <summary>
            /// Gets or sets the offset to the tile bitmap.
            /// </summary>
            public uint TileBitmapOffset { get; set; }
            
            /// <summary>
            /// Gets or sets the vertex data for this instance.
            /// </summary>
            public LiquidVertexData? VertexData { get; set; }
            
            /// <summary>
            /// Gets or sets the tile bitmap.
            /// </summary>
            public byte[]? TileBitmap { get; set; }
        }
        
        /// <summary>
        /// Class representing header data for one map chunk.
        /// </summary>
        public class ChunkHeader
        {
            /// <summary>
            /// Gets or sets the offsets to instance data.
            /// </summary>
            public uint[] InstanceOffsets { get; set; } = new uint[16 * 16];
            
            /// <summary>
            /// Gets or sets the instances for this chunk.
            /// </summary>
            public LiquidInstance[] Instances { get; set; } = Array.Empty<LiquidInstance>();
        }
        
        /// <summary>
        /// Gets the headers for each chunk.
        /// </summary>
        public ChunkHeader[] ChunkHeaders { get; } = new ChunkHeader[256];

        /// <summary>
        /// Initializes a new instance of the <see cref="Mh2oChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public Mh2oChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Initialize chunk headers
            for (int i = 0; i < 256; i++)
            {
                ChunkHeaders[i] = new ChunkHeader();
            }
        }

        /// <summary>
        /// Parses the chunk data
        /// </summary>
        public override void Parse()
        {
            if (Data == null || Data.Length == 0)
            {
                AddError("No data to parse for MH2O chunk");
                return;
            }

            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Read header data (256 entries, each with 16 offsets)
                    for (int i = 0; i < 256; i++)
                    {
                        // Each header has 16 offsets (one for each possible layer)
                        for (int j = 0; j < 16; j++)
                        {
                            ChunkHeaders[i].InstanceOffsets[j] = reader.ReadUInt32();
                        }
                    }
                    
                    // Parse each chunk's instances
                    for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
                    {
                        var header = ChunkHeaders[chunkIndex];
                        var validInstanceOffsets = new List<uint>();
                        
                        // Find all valid instance offsets
                        for (int i = 0; i < 16; i++)
                        {
                            if (header.InstanceOffsets[i] > 0)
                            {
                                validInstanceOffsets.Add(header.InstanceOffsets[i]);
                            }
                        }
                        
                        // Create instances array
                        header.Instances = new LiquidInstance[validInstanceOffsets.Count];
                        
                        // Parse each instance
                        for (int i = 0; i < validInstanceOffsets.Count; i++)
                        {
                            uint offset = validInstanceOffsets[i];
                            reader.BaseStream.Position = offset;
                            
                            var instance = new LiquidInstance
                            {
                                Type = (LiquidInstance.LiquidType)reader.ReadUInt16(),
                                Flags = reader.ReadUInt16(),
                                LiquidVertexFormat = reader.ReadByte(),
                                HeightLevel1 = reader.ReadByte(),
                                OffsetX = reader.ReadByte(),
                                OffsetY = reader.ReadByte(),
                                Width = reader.ReadByte(),
                                Height = reader.ReadByte()
                            };
                            
                            // Skip 2 bytes of padding
                            reader.ReadUInt16();
                            
                            instance.VertexDataOffset = reader.ReadUInt32();
                            instance.TileBitmapOffset = reader.ReadUInt32();
                            
                            // Parse vertex data if present
                            if (instance.VertexDataOffset > 0)
                            {
                                ParseVertexData(reader, instance);
                            }
                            
                            // Parse tile bitmap if present
                            if (instance.TileBitmapOffset > 0)
                            {
                                ParseTileBitmap(reader, instance);
                            }
                            
                            header.Instances[i] = instance;
                        }
                    }
                    
                    Logger?.LogDebug($"MH2O: Parsed liquid data for 256 map chunks");
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MH2O chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Parses vertex data for a liquid instance.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <param name="instance">The liquid instance.</param>
        private void ParseVertexData(BinaryReader reader, LiquidInstance instance)
        {
            if (instance.Width == 0 || instance.Height == 0)
            {
                Logger?.LogWarning($"MH2O: Invalid vertex dimensions: {instance.Width}x{instance.Height}");
                return;
            }
            
            // Save current position
            long currentPosition = reader.BaseStream.Position;
            
            try
            {
                // Go to vertex data position
                reader.BaseStream.Position = instance.VertexDataOffset;
                
                // Create vertex data
                instance.VertexData = new LiquidVertexData();
                
                // Calculate total vertices
                int totalVertices = (instance.Width + 1) * (instance.Height + 1);
                
                // Read height values
                instance.VertexData.HeightValues = new float[totalVertices];
                for (int i = 0; i < totalVertices; i++)
                {
                    instance.VertexData.HeightValues[i] = reader.ReadSingle();
                }
                
                // Check if there's depth data (format 2)
                if (instance.LiquidVertexFormat >= 2)
                {
                    // Read depth values
                    instance.VertexData.DepthValues = new float[totalVertices];
                    for (int i = 0; i < totalVertices; i++)
                    {
                        instance.VertexData.DepthValues[i] = reader.ReadSingle();
                    }
                }
                
                // Check if there's texture coordinate data (format 1 or 3)
                if (instance.LiquidVertexFormat == 1 || instance.LiquidVertexFormat == 3)
                {
                    // Read texture coordinates (2 floats per vertex)
                    instance.VertexData.TextureCoords = new float[totalVertices * 2];
                    for (int i = 0; i < totalVertices * 2; i++)
                    {
                        instance.VertexData.TextureCoords[i] = reader.ReadSingle();
                    }
                }
                
                Logger?.LogDebug($"MH2O: Parsed vertex data for liquid instance: {instance.Width}x{instance.Height}, format {instance.LiquidVertexFormat}");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MH2O vertex data: {ex.Message}");
                instance.VertexData = null;
            }
            finally
            {
                // Restore original position
                reader.BaseStream.Position = currentPosition;
            }
        }
        
        /// <summary>
        /// Parses tile bitmap for a liquid instance.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        /// <param name="instance">The liquid instance.</param>
        private void ParseTileBitmap(BinaryReader reader, LiquidInstance instance)
        {
            if (instance.Width == 0 || instance.Height == 0)
            {
                Logger?.LogWarning($"MH2O: Invalid tile bitmap dimensions: {instance.Width}x{instance.Height}");
                return;
            }
            
            // Save current position
            long currentPosition = reader.BaseStream.Position;
            
            try
            {
                // Go to tile bitmap position
                reader.BaseStream.Position = instance.TileBitmapOffset;
                
                // Calculate number of bytes needed for bitmap
                // Each bit in the bitmap represents one tile, so we need (width * height + 7) / 8 bytes
                int bitmapSize = (instance.Width * instance.Height + 7) / 8;
                
                // Read tile bitmap
                instance.TileBitmap = reader.ReadBytes(bitmapSize);
                
                Logger?.LogDebug($"MH2O: Parsed tile bitmap for liquid instance: {instance.Width}x{instance.Height}, {bitmapSize} bytes");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MH2O tile bitmap: {ex.Message}");
                instance.TileBitmap = null;
            }
            finally
            {
                // Restore original position
                reader.BaseStream.Position = currentPosition;
            }
        }

        /// <summary>
        /// Writes the chunk data to the specified writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                AddError("Cannot write to null writer");
                return;
            }

            try
            {
                // We'll maintain a list of offsets for each instance and vertex data block
                Dictionary<LiquidInstance, uint> instanceOffsets = new Dictionary<LiquidInstance, uint>();
                Dictionary<LiquidVertexData, uint> vertexDataOffsets = new Dictionary<LiquidVertexData, uint>();
                Dictionary<byte[], uint> bitmapOffsets = new Dictionary<byte[], uint>();
                
                // Write header data with 0 offsets initially
                // We'll come back and update these later
                long headerStart = writer.BaseStream.Position;
                
                for (int i = 0; i < 256; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        writer.Write((uint)0); // Placeholder offset
                    }
                }
                
                // Write instances, keeping track of their offsets
                for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
                {
                    var header = ChunkHeaders[chunkIndex];
                    
                    for (int i = 0; i < header.Instances.Length; i++)
                    {
                        var instance = header.Instances[i];
                        
                        // Store instance offset
                        instanceOffsets[instance] = (uint)writer.BaseStream.Position;
                        
                        // Write instance data
                        writer.Write((ushort)instance.Type);
                        writer.Write(instance.Flags);
                        writer.Write(instance.LiquidVertexFormat);
                        writer.Write(instance.HeightLevel1);
                        writer.Write(instance.OffsetX);
                        writer.Write(instance.OffsetY);
                        writer.Write(instance.Width);
                        writer.Write(instance.Height);
                        writer.Write((ushort)0); // 2 bytes padding
                        
                        // Placeholder offsets for vertex data and tile bitmap
                        writer.Write((uint)0);
                        writer.Write((uint)0);
                    }
                }
                
                // Write vertex data for each instance
                foreach (var instance in instanceOffsets.Keys)
                {
                    if (instance.VertexData != null)
                    {
                        // Record the offset where vertex data begins
                        uint vertexDataOffset = (uint)writer.BaseStream.Position;
                        
                        // Calculate total vertices
                        int totalVertices = (instance.Width + 1) * (instance.Height + 1);
                        
                        // Write height values
                        for (int i = 0; i < Math.Min(totalVertices, instance.VertexData.HeightValues.Length); i++)
                        {
                            writer.Write(instance.VertexData.HeightValues[i]);
                        }
                        
                        // Write depth values if present (format 2 or 3)
                        if (instance.LiquidVertexFormat >= 2 && instance.VertexData.DepthValues.Length > 0)
                        {
                            for (int i = 0; i < Math.Min(totalVertices, instance.VertexData.DepthValues.Length); i++)
                            {
                                writer.Write(instance.VertexData.DepthValues[i]);
                            }
                        }
                        
                        // Write texture coords if present (format 1 or 3)
                        if ((instance.LiquidVertexFormat == 1 || instance.LiquidVertexFormat == 3) && 
                            instance.VertexData.TextureCoords.Length > 0)
                        {
                            for (int i = 0; i < Math.Min(totalVertices * 2, instance.VertexData.TextureCoords.Length); i++)
                            {
                                writer.Write(instance.VertexData.TextureCoords[i]);
                            }
                        }
                        
                        // Store the offset for this vertex data
                        vertexDataOffsets[instance.VertexData] = vertexDataOffset;
                    }
                    
                    if (instance.TileBitmap != null && instance.TileBitmap.Length > 0)
                    {
                        // Record the offset where tile bitmap begins
                        uint bitmapOffset = (uint)writer.BaseStream.Position;
                        
                        // Write tile bitmap
                        writer.Write(instance.TileBitmap);
                        
                        // Store the offset for this bitmap
                        bitmapOffsets[instance.TileBitmap] = bitmapOffset;
                    }
                }
                
                // Go back and update instance offsets with the correct vertex data and bitmap offsets
                foreach (var instance in instanceOffsets.Keys)
                {
                    long currentPosition = writer.BaseStream.Position;
                    
                    // Go to the position where we need to write the offsets
                    writer.BaseStream.Position = instanceOffsets[instance] + 10; // 10 bytes to skip to offset fields
                    
                    // Write vertex data offset
                    if (instance.VertexData != null && vertexDataOffsets.ContainsKey(instance.VertexData))
                    {
                        writer.Write(vertexDataOffsets[instance.VertexData]);
                    }
                    else
                    {
                        writer.Write((uint)0);
                    }
                    
                    // Write tile bitmap offset
                    if (instance.TileBitmap != null && instance.TileBitmap.Length > 0 && bitmapOffsets.ContainsKey(instance.TileBitmap))
                    {
                        writer.Write(bitmapOffsets[instance.TileBitmap]);
                    }
                    else
                    {
                        writer.Write((uint)0);
                    }
                    
                    // Restore position
                    writer.BaseStream.Position = currentPosition;
                }
                
                // Go back and update header with instance offsets
                long endPosition = writer.BaseStream.Position;
                writer.BaseStream.Position = headerStart;
                
                for (int chunkIndex = 0; chunkIndex < 256; chunkIndex++)
                {
                    var header = ChunkHeaders[chunkIndex];
                    
                    // Write offsets for each layer
                    for (int j = 0; j < 16; j++)
                    {
                        if (j < header.Instances.Length && header.Instances[j] != null && instanceOffsets.ContainsKey(header.Instances[j]))
                        {
                            writer.Write(instanceOffsets[header.Instances[j]]);
                        }
                        else
                        {
                            writer.Write((uint)0);
                        }
                    }
                }
                
                // Restore end position
                writer.BaseStream.Position = endPosition;
            }
            catch (Exception ex)
            {
                AddError($"Error writing MH2O chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Gets the liquid instance for a specific map chunk and layer.
        /// </summary>
        /// <param name="chunkIndex">The map chunk index (0-255).</param>
        /// <param name="layer">The liquid layer (0-15).</param>
        /// <returns>The liquid instance, or null if not found.</returns>
        public LiquidInstance? GetLiquidInstance(int chunkIndex, int layer)
        {
            if (chunkIndex < 0 || chunkIndex >= 256)
            {
                AddError($"MH2O: Invalid chunk index: {chunkIndex}");
                return null;
            }
            
            if (layer < 0 || layer >= 16)
            {
                AddError($"MH2O: Invalid layer index: {layer}");
                return null;
            }
            
            var header = ChunkHeaders[chunkIndex];
            
            if (layer >= header.Instances.Length)
            {
                return null;
            }
            
            return header.Instances[layer];
        }
    }
} 