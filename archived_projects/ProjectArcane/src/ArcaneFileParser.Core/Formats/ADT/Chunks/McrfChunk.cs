using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Chunk Render Flags subchunk containing rendering flags for terrain.
/// </summary>
public class McrfChunk : ChunkBase
{
    public override string ChunkId => "MCRF";

    /// <summary>
    /// The number of render flags in each dimension (8x8 grid).
    /// </summary>
    public const int RENDER_FLAGS_DIM = 8;

    /// <summary>
    /// The total number of render flags (8x8 = 64).
    /// </summary>
    public const int TOTAL_FLAGS = 64;

    [Flags]
    public enum RenderFlags : uint
    {
        None = 0x0,
        Shadows = 0x1,           // Shadows enabled
        DetailDoodads = 0x2,     // Detail doodads enabled
        DetailGrass = 0x4,       // Detail grass enabled
        NoLighting = 0x8,        // Disable lighting
        NoFog = 0x10,           // Disable fog
        NoEnvMapping = 0x20,     // Disable environment mapping
        NoSpecular = 0x40,       // Disable specular lighting
        NoZBuffer = 0x80,        // Disable Z-buffer
        NoBlending = 0x100,      // Disable alpha blending
        NoTrilinear = 0x200,     // Disable trilinear filtering
        NoMipMapping = 0x400,    // Disable mip mapping
        NoSorting = 0x800,       // Disable sorting
        NoClipping = 0x1000,     // Disable clipping
        NoCulling = 0x2000,      // Disable backface culling
        NoDepthTest = 0x4000,    // Disable depth testing
        NoDepthWrite = 0x8000    // Disable depth writing
    }

    /// <summary>
    /// Gets the render flags array (8x8 grid).
    /// </summary>
    public RenderFlags[] Flags { get; } = new RenderFlags[TOTAL_FLAGS];

    public override void Parse(BinaryReader reader, uint size)
    {
        // Read all render flags
        for (int i = 0; i < TOTAL_FLAGS; i++)
        {
            Flags[i] = (RenderFlags)reader.ReadUInt32();
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write all render flags
        for (int i = 0; i < TOTAL_FLAGS; i++)
        {
            writer.Write((uint)Flags[i]);
        }
    }

    /// <summary>
    /// Gets render flags at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-7).</param>
    /// <param name="y">Y coordinate (0-7).</param>
    /// <returns>The render flags at the specified coordinates.</returns>
    public RenderFlags GetFlags(int x, int y)
    {
        if (x < 0 || x >= RENDER_FLAGS_DIM || y < 0 || y >= RENDER_FLAGS_DIM)
            return RenderFlags.None;

        return Flags[y * RENDER_FLAGS_DIM + x];
    }

    /// <summary>
    /// Sets render flags at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-7).</param>
    /// <param name="y">Y coordinate (0-7).</param>
    /// <param name="flags">The render flags to set.</param>
    public void SetFlags(int x, int y, RenderFlags flags)
    {
        if (x < 0 || x >= RENDER_FLAGS_DIM || y < 0 || y >= RENDER_FLAGS_DIM)
            return;

        Flags[y * RENDER_FLAGS_DIM + x] = flags;
    }

    /// <summary>
    /// Adds flags at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-7).</param>
    /// <param name="y">Y coordinate (0-7).</param>
    /// <param name="flags">The render flags to add.</param>
    public void AddFlags(int x, int y, RenderFlags flags)
    {
        if (x < 0 || x >= RENDER_FLAGS_DIM || y < 0 || y >= RENDER_FLAGS_DIM)
            return;

        Flags[y * RENDER_FLAGS_DIM + x] |= flags;
    }

    /// <summary>
    /// Removes flags at the specified grid coordinates.
    /// </summary>
    /// <param name="x">X coordinate (0-7).</param>
    /// <param name="y">Y coordinate (0-7).</param>
    /// <param name="flags">The render flags to remove.</param>
    public void RemoveFlags(int x, int y, RenderFlags flags)
    {
        if (x < 0 || x >= RENDER_FLAGS_DIM || y < 0 || y >= RENDER_FLAGS_DIM)
            return;

        Flags[y * RENDER_FLAGS_DIM + x] &= ~flags;
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine("Render Flags (8x8 grid):");

        for (int y = 0; y < RENDER_FLAGS_DIM; y++)
        {
            for (int x = 0; x < RENDER_FLAGS_DIM; x++)
            {
                var flags = GetFlags(x, y);
                if (flags != RenderFlags.None)
                {
                    builder.AppendLine($"\nPosition ({x}, {y}):");
                    builder.AppendLine($"  Flags: {flags}");
                    
                    // List individual flags
                    if ((flags & RenderFlags.Shadows) != 0) builder.AppendLine("  - Shadows Enabled");
                    if ((flags & RenderFlags.DetailDoodads) != 0) builder.AppendLine("  - Detail Doodads Enabled");
                    if ((flags & RenderFlags.DetailGrass) != 0) builder.AppendLine("  - Detail Grass Enabled");
                    if ((flags & RenderFlags.NoLighting) != 0) builder.AppendLine("  - Lighting Disabled");
                    if ((flags & RenderFlags.NoFog) != 0) builder.AppendLine("  - Fog Disabled");
                    if ((flags & RenderFlags.NoEnvMapping) != 0) builder.AppendLine("  - Environment Mapping Disabled");
                    if ((flags & RenderFlags.NoSpecular) != 0) builder.AppendLine("  - Specular Lighting Disabled");
                    if ((flags & RenderFlags.NoZBuffer) != 0) builder.AppendLine("  - Z-Buffer Disabled");
                    if ((flags & RenderFlags.NoBlending) != 0) builder.AppendLine("  - Alpha Blending Disabled");
                    if ((flags & RenderFlags.NoTrilinear) != 0) builder.AppendLine("  - Trilinear Filtering Disabled");
                    if ((flags & RenderFlags.NoMipMapping) != 0) builder.AppendLine("  - Mip Mapping Disabled");
                    if ((flags & RenderFlags.NoSorting) != 0) builder.AppendLine("  - Sorting Disabled");
                    if ((flags & RenderFlags.NoClipping) != 0) builder.AppendLine("  - Clipping Disabled");
                    if ((flags & RenderFlags.NoCulling) != 0) builder.AppendLine("  - Backface Culling Disabled");
                    if ((flags & RenderFlags.NoDepthTest) != 0) builder.AppendLine("  - Depth Testing Disabled");
                    if ((flags & RenderFlags.NoDepthWrite) != 0) builder.AppendLine("  - Depth Writing Disabled");
                }
            }
        }

        return builder.ToString();
    }
} 