using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.Tests;

/// <summary>
/// Decides which side of a hill the sun lights, using the real compositor path rather than reasoning
/// about axis conventions.
///
/// Orientation of a minimap raster, from <see cref="MinimapTileProjection"/>: row increases SOUTH,
/// column increases EAST. So north-west is the TOP-LEFT of the image and south-east the BOTTOM-RIGHT.
/// The authored solar bearing puts the source at 45 degrees -- world +X (North) and +Y (West) -- so
/// under the traced model the top-left flank of a hill is lit and the bottom-right flank is shaded.
/// </summary>
public sealed class TerrainMinimapSunDirectionTests
{
    private const int Grid = 257;
    private const int Resolution = 64;

    /// <summary>A single smooth hill centred on the tile, so every compass flank exists on one tile.</summary>
    private static float[,] BuildCentralHill(float peakHeight = 300f)
    {
        var height = new float[Grid, Grid];
        const float Centre = (Grid - 1) / 2f;
        float sigma = Grid / 6f;
        for (int row = 0; row < Grid; row++)
        {
            for (int column = 0; column < Grid; column++)
            {
                float dr = (row - Centre) / sigma;
                float dc = (column - Centre) / sigma;
                height[row, column] = peakHeight * MathF.Exp(-0.5f * ((dr * dr) + (dc * dc)));
            }
        }

        return height;
    }

    /// <summary>
    /// Derives MCNR-equivalent normals from the heightfield. <see cref="AdtTerrainMath.ComputeNormal"/>
    /// works in ADT grid axes, which is exactly how MCNR is stored, so the compositor's
    /// grid-to-renderer transform is exercised for real.
    /// </summary>
    private static (float[,,] Normals, bool[,] Mask) BuildNormals(float[,] height)
    {
        var flat = new float[Grid * Grid];
        for (int row = 0; row < Grid; row++)
        {
            for (int column = 0; column < Grid; column++)
                flat[(row * Grid) + column] = height[row, column];
        }

        var normals = new float[Grid, Grid, 3];
        var mask = new bool[Grid, Grid];
        for (int row = 0; row < Grid; row++)
        {
            for (int column = 0; column < Grid; column++)
            {
                Vector3 normal = AdtTerrainMath.ComputeNormal(flat, column, row);
                normals[row, column, 0] = normal.X;
                normals[row, column, 1] = normal.Y;
                normals[row, column, 2] = normal.Z;
                mask[row, column] = true;
            }
        }

        return (normals, mask);
    }

    private static TerrainTileTensorPack BuildHillPack(float[,] height)
    {
        (float[,,] normals, bool[,] mask) = BuildNormals(height);
        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        textureIds[0, 0, 1] = -1;
        textureIds[0, 0, 2] = -1;
        textureIds[0, 0, 3] = -1;

        return new TerrainTileTensorPack
        {
            TileX = 32,
            TileY = 32,
            MclyTextureIds = textureIds,
            MclyTextureNames = ["hill.blp"],
            McnrNormalXyz = normals,
            McnrMask257 = mask,
            Height257 = height,
        };
    }

    private static Image<Rgba32> RenderHill(TerrainMinimapLighting lighting, float[,] height) =>
        TerrainMinimapCompositor.Compose(
            BuildHillPack(height),
            new Dictionary<int, byte[,,]> { [0] = new byte[1, 1, 3] { { { 180, 180, 180 } } } },
            new TerrainMinimapCompositionOptions(Resolution, lighting));

    /// <summary>Mean brightness of a small patch, named by compass flank of the central hill.</summary>
    private static float FlankBrightness(Image<Rgba32> image, int offsetColumns, int offsetRows)
    {
        int centre = Resolution / 2;
        int total = 0;
        int count = 0;
        for (int dy = -3; dy <= 3; dy++)
        {
            for (int dx = -3; dx <= 3; dx++)
            {
                int x = Math.Clamp(centre + offsetColumns + dx, 0, Resolution - 1);
                int y = Math.Clamp(centre + offsetRows + dy, 0, Resolution - 1);
                total += image[x, y].R;
                count++;
            }
        }

        return total / (float)count;
    }

    /// <summary>
    /// THE DIRECTION TEST. With the source at the traced north-west bearing, the hill's north-west
    /// flank (up-left in the raster) must be brighter than its south-east flank (down-right).
    /// Reported symptom: the north-west side comes out shadowed instead.
    /// </summary>
    [Fact]
    public void LambertHillshade_LightsTheNorthWestFlankUnderTheTracedBearing()
    {
        float[,] height = BuildCentralHill();
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateShadedTerrain(0.5f) with
        {
            ApplyCastShadows = false,
        };

        using Image<Rgba32> image = RenderHill(lighting, height);

        float northWest = FlankBrightness(image, -12, -12);
        float southEast = FlankBrightness(image, +12, +12);

        Assert.True(
            northWest > southEast,
            $"Sun is north-west, so the NW flank must be lit: NW={northWest:0.0} SE={southEast:0.0}.");
    }

    /// <summary>
    /// Cast shadows must agree with the hillshade. If Lambert lights the north-west flank while the
    /// ray march throws shadows onto the north-west too, the two halves of the lighting model
    /// disagree and the render fights itself -- which reads exactly as "light moved around oddly".
    /// </summary>
    [Fact]
    public void CastShadows_FallOnTheSameFlankLambertShades()
    {
        float[,] height = BuildCentralHill(peakHeight: 600f);
        TerrainMinimapLighting lit = TerrainMinimapLighting.CreateShadedTerrain(8f / 24f);

        using Image<Rgba32> withShadows = RenderHill(lit, height);
        using Image<Rgba32> withoutShadows = RenderHill(lit with { ApplyCastShadows = false }, height);

        // How much each flank darkened once cast shadows were enabled.
        float northWestDarkening = FlankBrightness(withoutShadows, -20, -20) - FlankBrightness(withShadows, -20, -20);
        float southEastDarkening = FlankBrightness(withoutShadows, +20, +20) - FlankBrightness(withShadows, +20, +20);

        Assert.True(
            southEastDarkening > northWestDarkening,
            "Shadows must be thrown away from the sun (onto the SE), not toward it: " +
            $"NW darkened {northWestDarkening:0.0}, SE darkened {southEastDarkening:0.0}.");
    }

    /// <summary>
    /// Under a FIXED bearing the lit side must not change with time of day -- only shadow length
    /// should. The reported symptom is light appearing to move around the terrain, so pin it.
    /// </summary>
    [Theory]
    [InlineData(8f)]
    [InlineData(12f)]
    [InlineData(16f)]
    public void FixedBearing_KeepsTheSameFlankLitAtEveryHour(float hour)
    {
        float[,] height = BuildCentralHill();
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateShadedTerrain(hour / 24f) with
        {
            ApplyCastShadows = false,
        };

        using Image<Rgba32> image = RenderHill(lighting, height);

        Assert.True(
            FlankBrightness(image, -12, -12) > FlankBrightness(image, +12, +12),
            $"At {hour:0}:00 the NW flank must still be the lit one under a fixed bearing.");
    }

    /// <summary>
    /// Shadows must lengthen as the sun drops. Elevation peaks at noon, so an 08:00 sun sits lower
    /// than a 12:00 sun and must throw the longer shadow.
    /// </summary>
    [Fact]
    public void LowerSunThrowsLongerShadowsThanNoon()
    {
        float[,] height = BuildCentralHill(peakHeight: 600f);

        float[,]? noon = TerrainCastShadowMap.Compute(height, TerrainSolarDirection.Evaluate(0.5f));
        float[,]? morning = TerrainCastShadowMap.Compute(height, TerrainSolarDirection.Evaluate(8f / 24f));
        Assert.NotNull(noon);
        Assert.NotNull(morning);

        int noonShadowed = CountShadowed(noon!);
        int morningShadowed = CountShadowed(morning!);

        Assert.True(
            morningShadowed > noonShadowed,
            $"A lower sun must shadow more of the tile: 08:00={morningShadowed}, noon={noonShadowed}.");
    }

    private static int CountShadowed(float[,] occlusion)
    {
        int count = 0;
        for (int row = 0; row < occlusion.GetLength(0); row++)
        {
            for (int column = 0; column < occlusion.GetLength(1); column++)
            {
                if (occlusion[row, column] > 0.01f)
                    count++;
            }
        }

        return count;
    }
}
