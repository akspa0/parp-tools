using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Shadow subchunk containing shadow map data.
/// </summary>
public class McshChunk : ChunkBase
{
    public override string ChunkId => "MCSH";

    /// <summary>
    /// The width/height of the shadow map (64x64).
    /// </summary>
    public const int SHADOW_MAP_DIM = 64;

    /// <summary>
    /// The total size of the shadow map in bytes (64x64 = 4096).
    /// </summary>
    public const int SHADOW_MAP_SIZE = 4096;

    /// <summary>
    /// Gets the shadow map data (64x64 bytes).
    /// </summary>
    public byte[] ShadowMap { get; } = new byte[SHADOW_MAP_SIZE];

    public override void Parse(BinaryReader reader, uint size)
    {
        // Read shadow map data
        var bytesRead = reader.Read(ShadowMap, 0, Math.Min((int)size, SHADOW_MAP_SIZE));

        // Clear any remaining bytes if the size was smaller than expected
        if (bytesRead < SHADOW_MAP_SIZE)
        {
            Array.Clear(ShadowMap, bytesRead, SHADOW_MAP_SIZE - bytesRead);
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write shadow map data
        writer.Write(ShadowMap, 0, SHADOW_MAP_SIZE);
    }

    /// <summary>
    /// Gets a shadow value at the specified coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-63).</param>
    /// <param name="y">Y coordinate (0-63).</param>
    /// <returns>The shadow value if coordinates are valid, 0 otherwise.</returns>
    public byte GetShadow(int x, int y)
    {
        if (x < 0 || x >= SHADOW_MAP_DIM || y < 0 || y >= SHADOW_MAP_DIM)
            return 0;

        return ShadowMap[y * SHADOW_MAP_DIM + x];
    }

    /// <summary>
    /// Sets a shadow value at the specified coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-63).</param>
    /// <param name="y">Y coordinate (0-63).</param>
    /// <param name="value">The shadow value to set.</param>
    public void SetShadow(int x, int y, byte value)
    {
        if (x < 0 || x >= SHADOW_MAP_DIM || y < 0 || y >= SHADOW_MAP_DIM)
            return;

        ShadowMap[y * SHADOW_MAP_DIM + x] = value;
    }

    /// <summary>
    /// Fills the shadow map with a specific value.
    /// </summary>
    /// <param name="value">The value to fill with.</param>
    public void Fill(byte value)
    {
        Array.Fill(ShadowMap, value);
    }

    /// <summary>
    /// Clears the shadow map (sets all values to 0).
    /// </summary>
    public void Clear()
    {
        Array.Clear(ShadowMap, 0, SHADOW_MAP_SIZE);
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Shadow Map (64x64):");

        // Calculate shadow coverage statistics
        int totalShadowed = 0;
        int maxShadow = 0;
        int minShadow = 255;
        double avgShadow = 0;

        for (int i = 0; i < SHADOW_MAP_SIZE; i++)
        {
            if (ShadowMap[i] > 0)
            {
                totalShadowed++;
                maxShadow = Math.Max(maxShadow, ShadowMap[i]);
                minShadow = Math.Min(minShadow, ShadowMap[i]);
                avgShadow += ShadowMap[i];
            }
        }

        if (totalShadowed > 0)
        {
            avgShadow /= totalShadowed;
            double coverage = (double)totalShadowed / SHADOW_MAP_SIZE * 100;

            builder.AppendLine($"Shadow Coverage: {coverage:F1}%");
            builder.AppendLine($"Min Shadow: {minShadow}");
            builder.AppendLine($"Max Shadow: {maxShadow}");
            builder.AppendLine($"Average Shadow: {avgShadow:F2}");
        }
        else
        {
            builder.AppendLine("No shadows present");
        }

        return builder.ToString();
    }
} 