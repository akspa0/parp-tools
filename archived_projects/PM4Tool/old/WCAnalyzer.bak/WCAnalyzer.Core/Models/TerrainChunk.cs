using System;
using System.Collections.Generic;
using System.Drawing;
using System.Numerics;
using System.Text;

namespace WCAnalyzer.Core.Models;

/// <summary>
/// Represents a terrain chunk (MCNK) in an ADT file.
/// </summary>
public class TerrainChunk
{
    /// <summary>
    /// Gets or sets the position of the chunk in the ADT grid (x, y).
    /// </summary>
    public Point Position { get; set; }

    /// <summary>
    /// Gets or sets the area ID of the chunk.
    /// </summary>
    public int AreaId { get; set; }

    /// <summary>
    /// Gets or sets the flags for the chunk.
    /// </summary>
    public uint Flags { get; set; }

    /// <summary>
    /// Gets or sets the holes in the chunk.
    /// </summary>
    public uint Holes { get; set; }

    /// <summary>
    /// Gets or sets the liquid level of the chunk.
    /// </summary>
    public float LiquidLevel { get; set; }

    /// <summary>
    /// Gets or sets the position of the chunk in the world.
    /// </summary>
    public Vector3 WorldPosition { get; set; }

    /// <summary>
    /// Gets or sets the normal vectors for the chunk.
    /// </summary>
    public Vector3[] Normals { get; set; } = Array.Empty<Vector3>();

    /// <summary>
    /// Gets or sets the height values for the chunk.
    /// </summary>
    public float[] Heights { get; set; } = Array.Empty<float>();

    /// <summary>
    /// Gets or sets the texture layers for the chunk.
    /// </summary>
    public List<TextureLayer> TextureLayers { get; set; } = new List<TextureLayer>();

    /// <summary>
    /// Gets or sets the vertex colors for the chunk.
    /// </summary>
    public List<Vector3> VertexColors { get; set; } = new List<Vector3>();

    /// <summary>
    /// Gets or sets the shadow map for the chunk.
    /// </summary>
    public byte[] ShadowMap { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// Gets or sets the alpha maps for the chunk.
    /// </summary>
    public List<byte[]> AlphaMaps { get; set; } = new List<byte[]>();

    /// <summary>
    /// Gets or sets the doodad references for the chunk.
    /// </summary>
    public List<int> DoodadRefs { get; set; } = new List<int>();

    /// <summary>
    /// Gets a human-readable representation of the chunk's heightmap data.
    /// </summary>
    /// <returns>A string containing the heightmap data.</returns>
    public string GetHeightmapString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Heightmap for chunk at ({Position.X}, {Position.Y}):");
        
        // Heights are stored in a 9x9 grid (9 vertices per side)
        for (int y = 0; y < 9; y++)
        {
            for (int x = 0; x < 9; x++)
            {
                sb.Append($"{Heights[y * 9 + x]:F2} ");
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }
} 