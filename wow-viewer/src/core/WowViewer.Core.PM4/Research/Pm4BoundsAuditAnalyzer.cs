using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Measures how much PM4 geometry falls outside the tile it is stored in, and in which direction.
/// </summary>
/// <remarks>
/// A tile is <see cref="Pm4CoordinateService.TileSize"/> yards square in the two horizontal axes.
/// <see cref="Pm4CoordinateService.Pm4LocalToAdtPlacement"/> maps PM4 local X and Z onto the tile
/// plane and passes Y through as height, so X and Z are the axes that can overflow a tile boundary;
/// Y cannot and is not tested.
///
/// Overflow is reported per side rather than as a single magnitude, because the side is the
/// interesting part: geometry that consistently spills toward one edge is evidence about how it was
/// displaced, whereas a symmetric spill is more likely authoring slop.
/// </remarks>
public static class Pm4BoundsAuditAnalyzer
{
    public static Pm4BoundsAuditReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4TileBoundsRecord> tiles = [];
        int filesWithGeometry = 0;
        int filesOverflowing = 0;
        long verticesTotal = 0;
        long verticesOutside = 0;

        double overflowNegX = 0, overflowPosX = 0, overflowNegZ = 0, overflowPosZ = 0;
        int tilesNegX = 0, tilesPosX = 0, tilesNegZ = 0, tilesPosZ = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(path);
            IReadOnlyList<Vector3> vertices = document.KnownChunks.Msvt;
            if (vertices.Count == 0)
                continue;

            if (!Pm4CoordinateService.TryParseTileCoordinates(path, out int tileX, out int tileY))
                continue;

            filesWithGeometry++;

            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            float minZ = float.MaxValue, maxZ = float.MinValue;
            int outside = 0;

            foreach (Vector3 vertex in vertices)
            {
                minX = Math.Min(minX, vertex.X);
                maxX = Math.Max(maxX, vertex.X);
                minY = Math.Min(minY, vertex.Y);
                maxY = Math.Max(maxY, vertex.Y);
                minZ = Math.Min(minZ, vertex.Z);
                maxZ = Math.Max(maxZ, vertex.Z);

                if (vertex.X < 0f || vertex.X > Pm4CoordinateService.TileSize
                    || vertex.Z < 0f || vertex.Z > Pm4CoordinateService.TileSize)
                {
                    outside++;
                }
            }

            verticesTotal += vertices.Count;
            verticesOutside += outside;

            float spillNegX = Math.Max(0f, -minX);
            float spillPosX = Math.Max(0f, maxX - Pm4CoordinateService.TileSize);
            float spillNegZ = Math.Max(0f, -minZ);
            float spillPosZ = Math.Max(0f, maxZ - Pm4CoordinateService.TileSize);

            bool overflows = spillNegX > 0f || spillPosX > 0f || spillNegZ > 0f || spillPosZ > 0f;
            if (overflows)
                filesOverflowing++;

            if (spillNegX > 0f) { overflowNegX += spillNegX; tilesNegX++; }
            if (spillPosX > 0f) { overflowPosX += spillPosX; tilesPosX++; }
            if (spillNegZ > 0f) { overflowNegZ += spillNegZ; tilesNegZ++; }
            if (spillPosZ > 0f) { overflowPosZ += spillPosZ; tilesPosZ++; }

            tiles.Add(new Pm4TileBoundsRecord(
                Path.GetFileName(path),
                tileX,
                tileY,
                vertices.Count,
                outside,
                vertices.Count == 0 ? 0d : (double)outside / vertices.Count,
                minX, maxX, minY, maxY, minZ, maxZ,
                spillNegX, spillPosX, spillNegZ, spillPosZ));
        }

        IReadOnlyList<Pm4TileBoundsRecord> worst = tiles
            .Where(static tile => tile.VerticesOutside > 0)
            .OrderByDescending(static tile => Math.Max(Math.Max(tile.SpillNegX, tile.SpillPosX), Math.Max(tile.SpillNegZ, tile.SpillPosZ)))
            .Take(24)
            .ToList();

        List<string> notes =
        [
            $"Tile extent is {Pm4CoordinateService.TileSize:F2} yards on the X and Z axes; Y is height and is not tested.",
            "Spill is reported per side. A consistent one-sided spill is evidence about displacement; a symmetric spill is more likely authoring slop.",
            $"Side totals (yards, summed over tiles): -X={overflowNegX:F1} over {tilesNegX} tiles, +X={overflowPosX:F1} over {tilesPosX} tiles, -Z={overflowNegZ:F1} over {tilesNegZ} tiles, +Z={overflowPosZ:F1} over {tilesPosZ} tiles."
        ];

        return new Pm4BoundsAuditReport(
            resolvedDirectory,
            filesWithGeometry,
            filesOverflowing,
            verticesTotal,
            verticesOutside,
            verticesTotal == 0 ? 0d : (double)verticesOutside / verticesTotal,
            new Pm4BoundsSideSummary(overflowNegX, overflowPosX, overflowNegZ, overflowPosZ, tilesNegX, tilesPosX, tilesNegZ, tilesPosZ),
            worst,
            tiles,
            notes);
    }
}
