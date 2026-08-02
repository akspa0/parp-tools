using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.IO.Maps;

/// <summary>One LIT light to plot, already converted to renderer world units.</summary>
/// <param name="WorldPosition">Light centre in world units (NOT raw LIT fixed-point).</param>
/// <param name="CoreRadius">Radius of full influence, world units.</param>
/// <param name="OuterRadius">Radius at which influence reaches zero, world units.</param>
/// <param name="Color">Light colour in sRGB 0..1, used for both the dome tint and the swatch.</param>
public readonly record struct MinimapLightMarker(
    Vector3 WorldPosition,
    float CoreRadius,
    float OuterRadius,
    Vector3 Color,
    string Name);

/// <summary>Controls how the light overlay is drawn.</summary>
/// <param name="DomeOpacity">Peak tint strength of the influence dome at the light's core.</param>
/// <param name="DrawDome">Shade each light's area of influence with its own falloff curve.</param>
/// <param name="DrawSwatch">Mark each light's exact centre with a colour disc and contrast ring.</param>
/// <param name="SwatchRadiusPixels">Radius of the centre swatch, in output pixels.</param>
public sealed record MinimapLightOverlayOptions(
    float DomeOpacity = 0.45f,
    bool DrawDome = true,
    bool DrawSwatch = true,
    int SwatchRadiusPixels = 4)
{
    public static MinimapLightOverlayOptions Default { get; } = new();
}

/// <summary>
/// Draws LIT light positions over a synthesized minimap tile: a translucent dome covering each
/// light's area of influence, plus a solid colour swatch at its exact centre.
///
/// The dome uses the SAME falloff <c>LitSpatialSampler</c> applies when sampling lighting -- full
/// strength inside the core radius, linear to zero at the outer radius -- so what the overlay shows
/// is the influence model actually in use, not a decorative approximation.
///
/// This is a diagnostic overlay, never part of the terrain RGB corpus: it draws authored light
/// metadata on top of a render, so baking it into training rows would teach a model to reproduce
/// annotation marks.
/// </summary>
public static class TerrainMinimapLightOverlayCompositor
{
    public const string RenderProfile = "lit_light_overlay_v1";

    /// <summary>
    /// Returns a copy of <paramref name="baseImage"/> with every light that reaches this tile drawn
    /// on it, and reports how many were actually visible.
    /// </summary>
    public static Image<Rgba32> Compose(
        Image<Rgba32> baseImage,
        int tileX,
        int tileY,
        IReadOnlyList<MinimapLightMarker> lights,
        out int visibleLightCount,
        MinimapLightOverlayOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(baseImage);
        ArgumentNullException.ThrowIfNull(lights);
        options ??= MinimapLightOverlayOptions.Default;

        Image<Rgba32> result = baseImage.Clone();
        visibleLightCount = 0;
        if (lights.Count == 0)
            return result;

        int resolution = result.Width;

        // Keep only lights whose influence circle actually intersects this tile. A light centred on
        // a neighbouring tile can still spill across the seam, so this tests the circle against the
        // tile bounds rather than testing whether the centre is inside.
        var reaching = new List<(MinimapLightMarker Light, float CentreU, float CentreV, float RadiusUv)>();
        foreach (MinimapLightMarker light in lights)
        {
            float outerRadius = MathF.Max(light.OuterRadius, 0f);
            if (outerRadius <= 0f)
                continue;

            MinimapTileProjection.Project(light.WorldPosition, tileX, tileY, out float u, out float v);
            float radiusUv = outerRadius / MinimapTileProjection.TileWorldSize;
            if (u + radiusUv < 0f || u - radiusUv > 1f || v + radiusUv < 0f || v - radiusUv > 1f)
                continue;

            reaching.Add((light, u, v, radiusUv));
        }

        if (reaching.Count == 0)
            return result;

        visibleLightCount = reaching.Count;

        if (options.DrawDome)
        {
            for (int y = 0; y < result.Height; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    Vector3 world = MinimapTileProjection.Unproject(x, y, resolution, tileX, tileY);
                    Vector3 accumulated = Vector3.Zero;
                    float accumulatedWeight = 0f;

                    foreach ((MinimapLightMarker light, _, _, _) in reaching)
                    {
                        // Horizontal distance only: the overlay is a top-down plot, and a light's
                        // vertical offset from the terrain must not shrink its plotted footprint.
                        float dx = world.X - light.WorldPosition.X;
                        float dy = world.Y - light.WorldPosition.Y;
                        float distance = MathF.Sqrt((dx * dx) + (dy * dy));
                        float influence = ComputeInfluence(light, distance);
                        if (influence <= 0f)
                            continue;

                        accumulated += light.Color * influence;
                        accumulatedWeight += influence;
                    }

                    if (accumulatedWeight <= 0f)
                        continue;

                    // Overlapping domes blend toward their weighted mean colour rather than summing
                    // to white, so two adjacent lights stay individually readable.
                    Vector3 tint = accumulated / accumulatedWeight;
                    float alpha = Math.Clamp(accumulatedWeight, 0f, 1f) * Math.Clamp(options.DomeOpacity, 0f, 1f);
                    result[x, y] = Blend(result[x, y], tint, alpha);
                }
            }
        }

        if (options.DrawSwatch)
        {
            foreach ((MinimapLightMarker light, float centreU, float centreV, _) in reaching)
            {
                if (!MinimapTileProjection.IsWithinTile(centreU, centreV))
                    continue;

                DrawSwatch(
                    result,
                    (int)MathF.Round(centreU * resolution),
                    (int)MathF.Round(centreV * resolution),
                    Math.Max(options.SwatchRadiusPixels, 1),
                    light.Color);
            }
        }

        return result;
    }

    /// <summary>
    /// Mirrors <c>LitSpatialSampler.ComputeInfluence</c>: full strength within the core radius, then
    /// linear falloff to zero at the outer radius.
    /// </summary>
    private static float ComputeInfluence(MinimapLightMarker light, float distance)
    {
        float coreRadius = MathF.Max(light.CoreRadius, 0f);
        if (distance <= coreRadius)
            return 1f;

        float outerRadius = light.OuterRadius;
        if (outerRadius <= coreRadius)
            return 0f;

        return Math.Clamp(1f - ((distance - coreRadius) / (outerRadius - coreRadius)), 0f, 1f);
    }

    /// <summary>
    /// Draws a filled colour disc with a contrasting outline. The outline exists so a light whose
    /// colour matches the terrain underneath is still locatable.
    /// </summary>
    private static void DrawSwatch(Image<Rgba32> image, int centreX, int centreY, int radius, Vector3 color)
    {
        Rgba32 fill = ToRgba(color);
        // Pick the outline that contrasts with the swatch itself, not with the terrain, so the
        // marker reads consistently regardless of what it lands on.
        float luminance = (0.2126f * color.X) + (0.7152f * color.Y) + (0.0722f * color.Z);
        Rgba32 outline = luminance > 0.5f ? new Rgba32(0, 0, 0, 255) : new Rgba32(255, 255, 255, 255);

        int outlineRadius = radius + 1;
        for (int dy = -outlineRadius; dy <= outlineRadius; dy++)
        {
            for (int dx = -outlineRadius; dx <= outlineRadius; dx++)
            {
                int x = centreX + dx;
                int y = centreY + dy;
                if ((uint)x >= (uint)image.Width || (uint)y >= (uint)image.Height)
                    continue;

                float distance = MathF.Sqrt((dx * dx) + (dy * dy));
                if (distance <= radius)
                    image[x, y] = fill;
                else if (distance <= outlineRadius)
                    image[x, y] = outline;
            }
        }
    }

    private static Rgba32 Blend(Rgba32 background, Vector3 tint, float alpha)
    {
        float inverse = 1f - alpha;
        return new Rgba32(
            ToByte((background.R / 255f * inverse) + (tint.X * alpha)),
            ToByte((background.G / 255f * inverse) + (tint.Y * alpha)),
            ToByte((background.B / 255f * inverse) + (tint.Z * alpha)),
            background.A);
    }

    private static Rgba32 ToRgba(Vector3 color) =>
        new(ToByte(color.X), ToByte(color.Y), ToByte(color.Z), 255);

    private static byte ToByte(float value) => (byte)Math.Clamp((int)MathF.Round(value * 255f), 0, 255);
}
