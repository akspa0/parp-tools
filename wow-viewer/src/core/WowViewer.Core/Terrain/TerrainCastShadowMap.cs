using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>
/// Analytic sun-driven terrain cast shadows for one tile: marches a ray from every heightfield
/// sample toward the light and records whether the tile's own terrain blocks it.
///
/// This is the shadow signal Lambert N·L cannot produce. Lambert is *slope shading* -- it darkens a
/// face turned away from the sun -- but it can never darken flat, sun-facing ground that happens to
/// sit behind a ridge. Before this pass existed, the only shadow input the compositor had was MCSH,
/// which is off by default for synthetic minimaps and is in any case measurably uncorrelated with
/// authored minimap luminance (Pearson -0.0055 over the v50.1 0.5.3.3368 curriculum).
/// </summary>
/// <remarks>
/// SCOPE LIMIT: single-tile. The heightfield is one 533.33-unit ADT tile with no neighbour data, so
/// a ridge in the adjacent tile casts nothing across the seam. Because the authored solar bearing is
/// fixed north-west (<see cref="WowViewer.Core.Terrain.TerrainSolarDirection"/>), rays always exit
/// through the same two tile edges, and shadow coverage falls off there rather than being wrong in a
/// scattered way. Whole-map stitched exports will show this as a soft discontinuity on those edges.
/// </remarks>
public static class TerrainCastShadowMap
{
    /// <summary>ADT tile edge length in world units.</summary>
    public const float TileWorldSize = 533.33333f;

    /// <summary>
    /// Default occluder-height softness in world units. A blocker is fully shadowing once it clears
    /// the ray by this much; below that the shadow ramps in. Without a ramp, a binary in/out test on
    /// a 257-sample heightfield produces hard stair-stepped shadow edges under the fixed 45-degree
    /// bearing, because the march advances exactly one diagonal cell per step.
    /// </summary>
    public const float DefaultSoftnessWorldUnits = 4f;

    /// <summary>
    /// Builds a shadow-occlusion map matching <paramref name="height"/>'s grid, where 0 is fully lit
    /// and 1 is fully occluded.
    /// </summary>
    /// <param name="height">Square heightfield indexed <c>[row, col]</c> in ADT grid axes (world
    /// units). Typically <c>TerrainTileTensorPack.Height257</c>.</param>
    /// <param name="lightDirectionRenderer">Unit vector pointing *toward* the light, in renderer
    /// world axes -- the same space as <c>TerrainMinimapLighting.LightDirection</c>.</param>
    /// <param name="softnessWorldUnits">See <see cref="DefaultSoftnessWorldUnits"/>.</param>
    /// <returns>
    /// The occlusion map, or <c>null</c> when no cast shadow is geometrically possible: a missing or
    /// degenerate heightfield, or a sun with no horizontal bearing (straight overhead casts nothing).
    /// </returns>
    public static float[,]? Compute(
        float[,]? height,
        Vector3 lightDirectionRenderer,
        float softnessWorldUnits = DefaultSoftnessWorldUnits)
    {
        if (height is null)
            return null;

        int rows = height.GetLength(0);
        int columns = height.GetLength(1);
        if (rows < 2 || columns < 2 || rows != columns)
            return null;

        if (lightDirectionRenderer.LengthSquared() <= 1e-10f)
            return null;

        Vector3 light = Vector3.Normalize(lightDirectionRenderer);
        if (!float.IsFinite(light.X) || !float.IsFinite(light.Y) || !float.IsFinite(light.Z))
            return null;

        // A sun at or below the horizon has no meaningful cast direction, and a sun straight
        // overhead casts nothing at all. Either way there is no shadow map to build.
        float horizontalMagnitude = MathF.Max(MathF.Abs(light.X), MathF.Abs(light.Y));
        if (horizontalMagnitude <= 1e-4f || light.Z <= 1e-4f)
            return null;

        int size = rows;
        float step = TileWorldSize / (size - 1);

        // Renderer/world axes -> ADT grid axes. TerrainNormalGeometry.TransformAdtNormalToRenderer
        // fixes the convention as renderer = (-gridY, -gridX, gridZ), i.e. world +X is grid -row and
        // world +Y is grid -column. Marching "toward the light" in world therefore steps the grid by
        // (-light.Y, -light.X). Getting this backwards would light the wrong side of every ridge --
        // the same class of bug as the hillshade Y-axis inversion fixed in v0.5.2.
        //
        // Scale so the dominant grid axis advances exactly one cell per iteration.
        float worldUnitsPerStep = step / horizontalMagnitude;
        float deltaColumn = -light.Y / horizontalMagnitude;
        float deltaRow = -light.X / horizontalMagnitude;
        float deltaHeight = light.Z * worldUnitsPerStep;
        if (deltaHeight <= 1e-6f)
            return null;

        float maxHeight = float.NegativeInfinity;
        for (int row = 0; row < size; row++)
        {
            for (int column = 0; column < size; column++)
            {
                float sample = height[row, column];
                if (float.IsFinite(sample) && sample > maxHeight)
                    maxHeight = sample;
            }
        }

        if (!float.IsFinite(maxHeight))
            return null;

        float softness = float.IsFinite(softnessWorldUnits) && softnessWorldUnits > 1e-3f
            ? softnessWorldUnits
            : DefaultSoftnessWorldUnits;

        var occlusion = new float[size, size];
        for (int row = 0; row < size; row++)
        {
            for (int column = 0; column < size; column++)
            {
                float originHeight = height[row, column];
                if (!float.IsFinite(originHeight))
                    continue;

                // Once the ray has climbed above the tile's highest sample nothing can block it, so
                // the march is bounded by relief rather than always running the full tile diagonal.
                int reliefLimit = (int)MathF.Ceiling((maxHeight - originHeight) / deltaHeight);
                int maxSteps = Math.Clamp(reliefLimit, 0, size);
                if (maxSteps <= 0)
                    continue;

                float maxOcclusion = 0f;
                for (int marchStep = 1; marchStep <= maxSteps; marchStep++)
                {
                    float sampleColumn = column + (deltaColumn * marchStep);
                    float sampleRow = row + (deltaRow * marchStep);
                    if (sampleColumn < 0f || sampleRow < 0f || sampleColumn > size - 1 || sampleRow > size - 1)
                        break;

                    float terrainHeight = SampleBilinear(height, sampleRow, sampleColumn);
                    if (!float.IsFinite(terrainHeight))
                        continue;

                    float rayHeight = originHeight + (deltaHeight * marchStep);
                    float penetration = terrainHeight - rayHeight;
                    if (penetration <= 0f)
                        continue;

                    float occluded = MathF.Min(1f, penetration / softness);
                    if (occluded > maxOcclusion)
                    {
                        maxOcclusion = occluded;
                        if (maxOcclusion >= 1f)
                            break;
                    }
                }

                occlusion[row, column] = maxOcclusion;
            }
        }

        return occlusion;
    }

    /// <summary>Bilinearly samples a square grid at fractional <c>[row, column]</c> coordinates.</summary>
    public static float SampleBilinear(float[,] grid, float row, float column)
    {
        int size = grid.GetLength(0);
        row = Math.Clamp(row, 0f, size - 1f);
        column = Math.Clamp(column, 0f, grid.GetLength(1) - 1f);

        int row0 = (int)row;
        int column0 = (int)column;
        int row1 = Math.Min(row0 + 1, size - 1);
        int column1 = Math.Min(column0 + 1, grid.GetLength(1) - 1);
        float rowFraction = row - row0;
        float columnFraction = column - column0;

        float top = Lerp(grid[row0, column0], grid[row0, column1], columnFraction);
        float bottom = Lerp(grid[row1, column0], grid[row1, column1], columnFraction);
        return Lerp(top, bottom, rowFraction);
    }

    private static float Lerp(float from, float to, float fraction) => from + ((to - from) * fraction);
}
