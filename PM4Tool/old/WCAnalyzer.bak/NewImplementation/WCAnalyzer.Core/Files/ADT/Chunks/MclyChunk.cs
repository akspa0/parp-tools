using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents a MclyChunk (MCLY) chunk that contains texture layer definitions for terrain.
    /// </summary>
    public class MclyChunk : ADTChunk
    {
        /// <summary>
        /// The signature for this chunk type.
        /// </summary>
        public const string SIGNATURE = "MCLY";

        /// <summary>
        /// Flags for texture layers.
        /// </summary>
        [Flags]
        public enum MclyFlags : uint
        {
            /// <summary>
            /// No flags set.
            /// </summary>
            None = 0,

            /// <summary>
            /// Animation: none
            /// </summary>
            AnimNone = 0,

            /// <summary>
            /// Animation: Wave (45-degree rotation)
            /// </summary>
            AnimWave45 = 1,

            /// <summary>
            /// Animation: Wave (90-degree rotation)
            /// </summary>
            AnimWave90 = 2,

            /// <summary>
            /// Animation: Wave (180-degree rotation)
            /// </summary>
            AnimWave180 = 3,

            /// <summary>
            /// Animation mask (first two bits)
            /// </summary>
            AnimMask = 3,

            /// <summary>
            /// Texture should be oriented toward the camera.
            /// </summary>
            Billboard = 4,

            /// <summary>
            /// Texture has higher detail, drawn farther.
            /// </summary>
            HigherDetail = 8,

            /// <summary>
            /// Texture is projected onto the terrain with a 0-degree rotation.
            /// </summary>
            Projected0 = 0x10,

            /// <summary>
            /// Texture is projected onto the terrain with a 90-degree rotation.
            /// </summary>
            Projected90 = 0x20,

            /// <summary>
            /// Texture is projected onto the terrain with a 180-degree rotation.
            /// </summary>
            Projected180 = 0x40,

            /// <summary>
            /// Texture is projected onto the terrain with a 270-degree rotation.
            /// </summary>
            Projected270 = 0x80,

            /// <summary>
            /// Mask for projected texture orientation.
            /// </summary>
            ProjectedMask = 0xF0
        }

        /// <summary>
        /// Represents a single texture layer definition.
        /// </summary>
        public class TextureLayerDefinition
        {
            /// <summary>
            /// Gets or sets the texture ID (index into the MTEX chunk).
            /// </summary>
            public uint TextureId { get; set; }

            /// <summary>
            /// Gets or sets the flags for this texture layer.
            /// </summary>
            public MclyFlags Flags { get; set; }

            /// <summary>
            /// Gets or sets the offset into the MCAL chunk for this layer's alpha map.
            /// </summary>
            public uint OffsetInMcal { get; set; }

            /// <summary>
            /// Gets or sets the effect ID for this texture layer.
            /// </summary>
            public int EffectId { get; set; }

            /// <summary>
            /// Initializes a new instance of the <see cref="TextureLayerDefinition"/> class.
            /// </summary>
            public TextureLayerDefinition()
            {
                TextureId = 0;
                Flags = MclyFlags.None;
                OffsetInMcal = 0;
                EffectId = 0;
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="TextureLayerDefinition"/> class with specified values.
            /// </summary>
            /// <param name="textureId">The texture ID (index into the MTEX chunk).</param>
            /// <param name="flags">The flags for this texture layer.</param>
            /// <param name="offsetInMcal">The offset into the MCAL chunk for this layer's alpha map.</param>
            /// <param name="effectId">The effect ID for this texture layer.</param>
            public TextureLayerDefinition(uint textureId, MclyFlags flags, uint offsetInMcal, int effectId)
            {
                TextureId = textureId;
                Flags = flags;
                OffsetInMcal = offsetInMcal;
                EffectId = effectId;
            }
        }

        /// <summary>
        /// Gets the list of texture layer definitions.
        /// </summary>
        public List<TextureLayerDefinition> TextureLayers { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MclyChunk"/> class.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        /// <param name="logger">The logger instance.</param>
        public MclyChunk(byte[] data, ILogger logger) 
            : base(SIGNATURE, data, logger)
        {
            TextureLayers = new List<TextureLayerDefinition>();
            Parse(data);
        }

        /// <summary>
        /// Parses the raw chunk data.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        protected override void Parse(byte[] data)
        {
            if (data == null || data.Length == 0)
            {
                Logger.LogWarning("MclyChunk: Empty data provided to Parse method");
                return;
            }

            try
            {
                using (MemoryStream ms = new MemoryStream(data))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    // Each texture layer definition is 16 bytes
                    int layerCount = data.Length / 16;
                    
                    for (int i = 0; i < layerCount; i++)
                    {
                        uint textureId = reader.ReadUInt32();
                        MclyFlags flags = (MclyFlags)reader.ReadUInt32();
                        uint offsetInMcal = reader.ReadUInt32();
                        int effectId = reader.ReadInt32();

                        TextureLayerDefinition layer = new TextureLayerDefinition(
                            textureId, flags, offsetInMcal, effectId);
                        
                        TextureLayers.Add(layer);
                    }

                    Logger.LogDebug($"MclyChunk: Successfully parsed {layerCount} texture layer definitions");
                }
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MclyChunk: Error parsing chunk data: {ex.Message}");
            }
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger.LogWarning("MclyChunk: Null writer provided to Write method");
                return;
            }

            try
            {
                // Write the chunk signature
                writer.Write(SignatureBytes);

                // Calculate the size of the data (16 bytes per layer)
                int dataSize = TextureLayers.Count * 16;
                writer.Write(dataSize);

                // Write each texture layer definition
                foreach (var layer in TextureLayers)
                {
                    writer.Write(layer.TextureId);
                    writer.Write((uint)layer.Flags);
                    writer.Write(layer.OffsetInMcal);
                    writer.Write(layer.EffectId);
                }

                Logger.LogDebug($"MclyChunk: Successfully wrote {TextureLayers.Count} texture layer definitions");
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MclyChunk: Error writing chunk data: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets a texture layer definition by index.
        /// </summary>
        /// <param name="index">The index of the texture layer to retrieve.</param>
        /// <returns>The texture layer definition at the specified index.</returns>
        public TextureLayerDefinition GetLayer(int index)
        {
            if (index < 0 || index >= TextureLayers.Count)
            {
                Logger.LogWarning($"MclyChunk: Layer index {index} is out of range (0-{TextureLayers.Count - 1})");
                return null;
            }

            return TextureLayers[index];
        }

        /// <summary>
        /// Gets the number of texture layers in this chunk.
        /// </summary>
        public int LayerCount => TextureLayers.Count;

        /// <summary>
        /// Adds a texture layer definition to the chunk.
        /// </summary>
        /// <param name="layer">The texture layer definition to add.</param>
        public void AddLayer(TextureLayerDefinition layer)
        {
            if (layer == null)
            {
                Logger.LogWarning("MclyChunk: Attempted to add null layer");
                return;
            }

            TextureLayers.Add(layer);
            Logger.LogDebug($"MclyChunk: Added layer (total: {TextureLayers.Count})");
        }
    }
} 