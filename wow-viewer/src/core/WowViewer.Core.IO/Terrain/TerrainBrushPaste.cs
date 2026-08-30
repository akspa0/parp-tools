using System.Numerics;
using System.Text.Json.Serialization;

namespace WowViewer.Core.IO.Terrain;

/// <summary>
/// A reusable, multi-layered terrain motif representing relative elevation contours,
/// multi-layer texture alpha splats, surface normals, and classification tags.
/// Can be stamped onto existing terrain or used as a building block in procedural map synthesis.
/// </summary>
public sealed class TerrainBrushPaste
{
    public string Id { get; init; } = string.Empty;
    public string Name { get; init; } = string.Empty;
    public string Category { get; init; } = "General"; // Road, Hill, Plaza, Ridge, Depression, Garden, Coastline
    public string[] Tags { get; init; } = [];
    public float WidthMeters { get; init; } = 33.33333f;
    public float LengthMeters { get; init; } = 33.33333f;
    public int ResolutionX { get; init; } = 17;
    public int ResolutionY { get; init; } = 17;
    public float[] HeightDeltas { get; init; } = [];
    public List<TerrainPasteLayer> Layers { get; init; } = [];
    public string SourceBuild { get; init; } = "curated";
    public string SourceMap { get; init; } = string.Empty;
    public float MaxSlopeDegrees { get; set; }

    /// <summary>
    /// Samples interpolated relative height delta at normalized [0, 1] coordinates (u, v).
    /// </summary>
    public float SampleHeight(float u, float v)
    {
        if (HeightDeltas.Length == 0 || ResolutionX < 2 || ResolutionY < 2)
            return 0f;

        u = Math.Clamp(u, 0f, 1f);
        v = Math.Clamp(v, 0f, 1f);

        float gx = u * (ResolutionX - 1);
        float gy = v * (ResolutionY - 1);

        int x0 = Math.Clamp((int)Math.Floor(gx), 0, ResolutionX - 1);
        int y0 = Math.Clamp((int)Math.Floor(gy), 0, ResolutionY - 1);
        int x1 = Math.Min(x0 + 1, ResolutionX - 1);
        int y1 = Math.Min(y0 + 1, ResolutionY - 1);

        float fx = gx - x0;
        float fy = gy - y0;

        float h00 = HeightDeltas[y0 * ResolutionX + x0];
        float h10 = HeightDeltas[y0 * ResolutionX + x1];
        float h01 = HeightDeltas[y1 * ResolutionX + x0];
        float h11 = HeightDeltas[y1 * ResolutionX + x1];

        float h0 = h00 + fx * (h10 - h00);
        float h1 = h01 + fx * (h11 - h01);
        return h0 + fy * (h1 - h0);
    }

    /// <summary>
    /// Samples interpolated alpha splat intensity [0, 255] for a given layer index at [0, 1] coordinates (u, v).
    /// </summary>
    public byte SampleLayerAlpha(int layerIndex, float u, float v)
    {
        if (layerIndex < 0 || layerIndex >= Layers.Count)
            return 0;

        TerrainPasteLayer layer = Layers[layerIndex];
        if (layer.AlphaMask.Length == 0 || layer.Resolution < 2)
            return 0;

        u = Math.Clamp(u, 0f, 1f);
        v = Math.Clamp(v, 0f, 1f);

        float gx = u * (layer.Resolution - 1);
        float gy = v * (layer.Resolution - 1);

        int x0 = Math.Clamp((int)Math.Floor(gx), 0, layer.Resolution - 1);
        int y0 = Math.Clamp((int)Math.Floor(gy), 0, layer.Resolution - 1);
        int x1 = Math.Min(x0 + 1, layer.Resolution - 1);
        int y1 = Math.Min(y0 + 1, layer.Resolution - 1);

        float fx = gx - x0;
        float fy = gy - y0;

        byte a00 = layer.AlphaMask[y0 * layer.Resolution + x0];
        byte a10 = layer.AlphaMask[y0 * layer.Resolution + x1];
        byte a01 = layer.AlphaMask[y1 * layer.Resolution + x0];
        byte a11 = layer.AlphaMask[y1 * layer.Resolution + x1];

        float a0 = a00 + fx * (a10 - a00);
        float a1 = a01 + fx * (a11 - a01);
        float a = a0 + fy * (a1 - a0);

        return (byte)Math.Clamp((int)Math.Round(a), 0, 255);
    }

    /// <summary>
    /// Computes the maximum slope gradient across the heightfield in degrees.
    /// </summary>
    public float CalculateMaxSlopeDegrees()
    {
        if (HeightDeltas.Length < 4 || ResolutionX < 2 || ResolutionY < 2)
            return 0f;

        float dx = WidthMeters / (ResolutionX - 1);
        float dy = LengthMeters / (ResolutionY - 1);
        float maxSlopeRad = 0f;

        for (int y = 0; y < ResolutionY - 1; y++)
        {
            for (int x = 0; x < ResolutionX - 1; x++)
            {
                float h00 = HeightDeltas[y * ResolutionX + x];
                float h10 = HeightDeltas[y * ResolutionX + x + 1];
                float h01 = HeightDeltas[(y + 1) * ResolutionX + x];

                float dzx = (h10 - h00) / dx;
                float dzy = (h01 - h00) / dy;
                float grad = MathF.Sqrt(dzx * dzx + dzy * dzy);
                float slope = MathF.Atan(grad);
                if (slope > maxSlopeRad)
                    maxSlopeRad = slope;
            }
        }

        MaxSlopeDegrees = maxSlopeRad * (180f / MathF.PI);
        return MaxSlopeDegrees;
    }
}

/// <summary>
/// A single texture layer in a terrain brush paste, containing texture reference and 2D alpha mask.
/// </summary>
public sealed class TerrainPasteLayer
{
    public string TexturePath { get; init; } = string.Empty;
    public int Resolution { get; init; } = 64; // Standard 64x64 MCAL splat resolution
    public byte[] AlphaMask { get; init; } = [];
    public uint EffectId { get; init; }
    public uint Flags { get; init; }
}
