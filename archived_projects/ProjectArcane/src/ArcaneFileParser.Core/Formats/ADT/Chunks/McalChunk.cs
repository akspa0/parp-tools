using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Alpha Layer subchunk containing alpha maps for texture blending.
/// </summary>
public class McalChunk : ChunkBase
{
    public override string ChunkId => "MCAL";

    /// <summary>
    /// The size of each alpha map in bytes (64x64 = 4096).
    /// </summary>
    public const int ALPHA_MAP_SIZE = 4096;

    /// <summary>
    /// The width/height of each alpha map.
    /// </summary>
    public const int ALPHA_MAP_DIM = 64;

    /// <summary>
    /// Gets the raw alpha map data.
    /// </summary>
    public byte[] RawData { get; private set; } = Array.Empty<byte>();

    /// <summary>
    /// Gets the list of alpha maps extracted from the raw data.
    /// Each alpha map is 64x64 bytes.
    /// </summary>
    public List<byte[]> AlphaMaps { get; } = new();

    public override void Parse(BinaryReader reader, uint size)
    {
        // Read raw data
        RawData = reader.ReadBytes((int)size);

        // Clear existing alpha maps
        AlphaMaps.Clear();

        // Extract individual alpha maps
        int offset = 0;
        while (offset + ALPHA_MAP_SIZE <= RawData.Length)
        {
            var alphaMap = new byte[ALPHA_MAP_SIZE];
            Array.Copy(RawData, offset, alphaMap, 0, ALPHA_MAP_SIZE);
            AlphaMaps.Add(alphaMap);
            offset += ALPHA_MAP_SIZE;
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write all alpha maps back to back
        foreach (var alphaMap in AlphaMaps)
        {
            writer.Write(alphaMap);
        }
    }

    /// <summary>
    /// Gets an alpha map by index.
    /// </summary>
    /// <param name="index">Index of the alpha map.</param>
    /// <returns>The alpha map data if found, null otherwise.</returns>
    public byte[]? GetAlphaMap(int index)
    {
        if (index < 0 || index >= AlphaMaps.Count)
            return null;

        return AlphaMaps[index];
    }

    /// <summary>
    /// Gets an alpha value from a specific alpha map.
    /// </summary>
    /// <param name="mapIndex">Index of the alpha map.</param>
    /// <param name="x">X coordinate (0-63).</param>
    /// <param name="y">Y coordinate (0-63).</param>
    /// <returns>The alpha value if coordinates are valid, 0 otherwise.</returns>
    public byte GetAlpha(int mapIndex, int x, int y)
    {
        if (mapIndex < 0 || mapIndex >= AlphaMaps.Count)
            return 0;

        if (x < 0 || x >= ALPHA_MAP_DIM || y < 0 || y >= ALPHA_MAP_DIM)
            return 0;

        return AlphaMaps[mapIndex][y * ALPHA_MAP_DIM + x];
    }

    /// <summary>
    /// Sets an alpha value in a specific alpha map.
    /// </summary>
    /// <param name="mapIndex">Index of the alpha map.</param>
    /// <param name="x">X coordinate (0-63).</param>
    /// <param name="y">Y coordinate (0-63).</param>
    /// <param name="alpha">The alpha value to set.</param>
    public void SetAlpha(int mapIndex, int x, int y, byte alpha)
    {
        if (mapIndex < 0 || mapIndex >= AlphaMaps.Count)
            return;

        if (x < 0 || x >= ALPHA_MAP_DIM || y < 0 || y >= ALPHA_MAP_DIM)
            return;

        AlphaMaps[mapIndex][y * ALPHA_MAP_DIM + x] = alpha;
    }

    /// <summary>
    /// Adds a new alpha map.
    /// </summary>
    /// <param name="alphaMap">The alpha map data (must be 64x64 bytes).</param>
    public void AddAlphaMap(byte[] alphaMap)
    {
        if (alphaMap.Length != ALPHA_MAP_SIZE)
            throw new ArgumentException($"Alpha map must be {ALPHA_MAP_SIZE} bytes");

        var copy = new byte[ALPHA_MAP_SIZE];
        Array.Copy(alphaMap, copy, ALPHA_MAP_SIZE);
        AlphaMaps.Add(copy);
    }

    /// <summary>
    /// Creates a new alpha map filled with the specified value.
    /// </summary>
    /// <param name="defaultAlpha">The default alpha value to fill with.</param>
    public void CreateAlphaMap(byte defaultAlpha = 0)
    {
        var alphaMap = new byte[ALPHA_MAP_SIZE];
        if (defaultAlpha != 0)
        {
            Array.Fill(alphaMap, defaultAlpha);
        }
        AlphaMaps.Add(alphaMap);
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Alpha Maps: {AlphaMaps.Count}");
        builder.AppendLine($"Total Size: {RawData.Length} bytes");

        for (int i = 0; i < AlphaMaps.Count; i++)
        {
            builder.AppendLine($"\nAlpha Map {i}:");
            var map = AlphaMaps[i];

            // Calculate average alpha
            double avgAlpha = 0;
            for (int j = 0; j < ALPHA_MAP_SIZE; j++)
            {
                avgAlpha += map[j];
            }
            avgAlpha /= ALPHA_MAP_SIZE;

            // Calculate non-zero coverage
            int nonZero = 0;
            for (int j = 0; j < ALPHA_MAP_SIZE; j++)
            {
                if (map[j] > 0) nonZero++;
            }
            double coverage = (double)nonZero / ALPHA_MAP_SIZE * 100;

            builder.AppendLine($"  Average Alpha: {avgAlpha:F2}");
            builder.AppendLine($"  Coverage: {coverage:F1}%");
        }

        return builder.ToString();
    }
} 