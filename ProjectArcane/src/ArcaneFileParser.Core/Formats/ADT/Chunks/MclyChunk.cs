using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Layer subchunk containing texture layer information.
/// </summary>
public class MclyChunk : ChunkBase
{
    public override string ChunkId => "MCLY";

    [Flags]
    public enum LayerFlags : uint
    {
        None = 0x0,
        AnimatedUV = 0x1,           // Use animated texture coordinates
        AnimatedUVFast = 0x2,       // Use fast animated texture coordinates
        Compressed = 0x4,           // Alpha map is compressed
        UseAlphaMap = 0x8,         // Use alpha map
        AlphaCompressed = 0x10,    // Alpha map is compressed (alternative flag)
        UseHeightTexture = 0x20,   // Use height-based texture blending
        Unknown40 = 0x40,          // Unknown flag
        Unknown80 = 0x80,          // Unknown flag
        Unknown100 = 0x100,        // Unknown flag
        UseAlpha = 0x200          // Use alpha blending
    }

    /// <summary>
    /// Texture layer information.
    /// </summary>
    public struct TextureLayer
    {
        public uint TextureId;      // Index into texture filename list
        public LayerFlags Flags;    // Layer flags
        public uint OffsetInMcal;   // Offset of alpha map in MCAL chunk
        public int EffectId;        // Index into EffectDoodad list
        public uint Unused1;        // Padding
        public uint Unused2;        // Padding

        public bool HasAlpha => (Flags & (LayerFlags.UseAlphaMap | LayerFlags.UseAlpha)) != 0;
        public bool IsCompressed => (Flags & (LayerFlags.Compressed | LayerFlags.AlphaCompressed)) != 0;
        public bool HasAnimation => (Flags & (LayerFlags.AnimatedUV | LayerFlags.AnimatedUVFast)) != 0;
    }

    /// <summary>
    /// Gets the list of texture layers.
    /// </summary>
    public List<TextureLayer> Layers { get; } = new();

    public override void Parse(BinaryReader reader, uint size)
    {
        // Clear existing data
        Layers.Clear();

        // Each layer entry is 16 bytes
        var layerCount = size / 16;

        // Read all layer entries
        for (int i = 0; i < layerCount; i++)
        {
            var layer = new TextureLayer
            {
                TextureId = reader.ReadUInt32(),
                Flags = (LayerFlags)reader.ReadUInt32(),
                OffsetInMcal = reader.ReadUInt32(),
                EffectId = reader.ReadInt32(),
                Unused1 = reader.ReadUInt32(),
                Unused2 = reader.ReadUInt32()
            };

            Layers.Add(layer);
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write each layer entry
        foreach (var layer in Layers)
        {
            writer.Write(layer.TextureId);
            writer.Write((uint)layer.Flags);
            writer.Write(layer.OffsetInMcal);
            writer.Write(layer.EffectId);
            writer.Write(layer.Unused1);
            writer.Write(layer.Unused2);
        }
    }

    /// <summary>
    /// Gets a texture layer by index.
    /// </summary>
    /// <param name="index">Index of the layer.</param>
    /// <returns>The texture layer if found, null otherwise.</returns>
    public TextureLayer? GetLayer(int index)
    {
        if (index < 0 || index >= Layers.Count)
            return null;

        return Layers[index];
    }

    /// <summary>
    /// Adds a new texture layer.
    /// </summary>
    /// <param name="textureId">Index into texture filename list.</param>
    /// <param name="flags">Layer flags.</param>
    /// <param name="offsetInMcal">Offset of alpha map in MCAL chunk.</param>
    /// <param name="effectId">Index into EffectDoodad list.</param>
    public void AddLayer(uint textureId, LayerFlags flags = LayerFlags.None, uint offsetInMcal = 0, int effectId = -1)
    {
        var layer = new TextureLayer
        {
            TextureId = textureId,
            Flags = flags,
            OffsetInMcal = offsetInMcal,
            EffectId = effectId,
            Unused1 = 0,
            Unused2 = 0
        };

        Layers.Add(layer);
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Texture Layers: {Layers.Count}");

        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            builder.AppendLine($"\nLayer {i}:");
            builder.AppendLine($"  Texture ID: {layer.TextureId}");
            builder.AppendLine($"  Flags: {layer.Flags}");
            if (layer.HasAlpha)
            {
                builder.AppendLine($"  Alpha Map Offset: 0x{layer.OffsetInMcal:X8}");
                builder.AppendLine($"  Alpha Compressed: {layer.IsCompressed}");
            }
            if (layer.HasAnimation)
            {
                builder.AppendLine($"  Has Animation: Yes");
                builder.AppendLine($"  Fast Animation: {(layer.Flags & LayerFlags.AnimatedUVFast) != 0}");
            }
            if (layer.EffectId >= 0)
            {
                builder.AppendLine($"  Effect ID: {layer.EffectId}");
            }
        }

        return builder.ToString();
    }
} 