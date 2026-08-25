using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Services;

/// <summary>
/// Resolves MPRL positions to world coordinates for region-grouped PM4 objects.
/// Uses the existing coordinate mode resolution and planar transform infrastructure.
/// </summary>
public static class Pm4ObjectPositionDecoder
{
    /// <summary>
    /// Decode world-space positions for all objects in a region grouping report.
    /// </summary>
    public static Pm4DecodedRegionReport DecodeAll(string inputDirectory, Pm4RegionGroupingReport groupingReport)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        // Reload the PM4 files to get full vertex/index data for placement resolution.
        List<(string Path, Pm4ResearchDocument Doc)> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(path => (path, Pm4ResearchReader.ReadFile(path)))
            .ToList();

        // Build a lookup of tile coordinate → document for vertex/index access.
        Dictionary<string, Pm4ResearchDocument> tileDocs = new(StringComparer.Ordinal);
        foreach ((string path, Pm4ResearchDocument doc) in files)
        {
            if (Pm4CoordinateService.TryParseTileCoordinates(path, out int tileX, out int tileY))
                tileDocs[$"{tileX}_{tileY}"] = doc;
        }

        List<Pm4DecodedRegion> decodedRegions = new();
        int totalDecoded = 0;

        foreach (Pm4Region region in groupingReport.Regions)
        {
            List<Pm4DecodedObjectPlacement> decodedObjects = new();

            foreach (Pm4RegionObject obj in region.Objects)
            {
                // For each sub-object in this CK24 object, resolve placement.
                foreach (Pm4SubObject subObj in obj.SubObjects)
                {
                    Pm4DecodedObjectPlacement? placement = DecodeSubObject(
                        region.RegionId, obj, subObj, tileDocs);

                    if (placement is not null)
                    {
                        decodedObjects.Add(placement);
                        totalDecoded++;
                    }
                }
            }

            Pm4DecodedRegion decodedRegion = new(
                region.RegionId,
                decodedObjects,
                region.TileCount,
                region.TotalSurfaceCount);

            decodedRegions.Add(decodedRegion);
        }

        List<string> notes = new()
        {
            $"Decoded {totalDecoded} sub-object placements across {decodedRegions.Count} regions.",
            $"Each placement includes world position, heading, bounds, and coordinate mode.",
            "Position resolution uses footprint-based planar transform scoring (existing Pm4PlacementMath)."
        };

        return new Pm4DecodedRegionReport(
            resolvedDirectory,
            groupingReport.TotalFiles,
            groupingReport.NonEmptyFiles,
            groupingReport.TotalRegions,
            totalDecoded,
            decodedRegions,
            notes);
    }

    /// <summary>
    /// Decode a single sub-object's world-space placement.
    /// </summary>
    private static Pm4DecodedObjectPlacement? DecodeSubObject(
        uint regionId,
        Pm4RegionObject obj,
        Pm4SubObject subObj,
        Dictionary<string, Pm4ResearchDocument> tileDocs)
    {
        if (subObj.SurfaceIndices.Count == 0)
            return null;

        // Collect the MSUR entries and vertex data from the relevant tiles.
        List<Pm4MsurEntry> surfaces = new();
        List<Vector3> allVertices = new();
        List<uint> allIndices = new();
        List<Pm4MprlEntry> allMprl = new();
        int tileX = 0;
        int tileY = 0;

        // Group surface indices by tile coordinate.
        Dictionary<string, List<int>> surfacesByTile = new(StringComparer.Ordinal);
        foreach (int surfIdx in subObj.SurfaceIndices)
        {
            // We need to find which tile this surface came from.
            // Since we stored global indices, we need to look them up.
            // For now, use the first tile in the object's tile list.
            string tileCoord = obj.TileCoordinates.FirstOrDefault() ?? "0_0";
            if (!surfacesByTile.TryGetValue(tileCoord, out List<int>? tileSurfs))
            {
                tileSurfs = new List<int>();
                surfacesByTile[tileCoord] = tileSurfs;
            }
            tileSurfs.Add(surfIdx);
        }

        // For each tile, collect surfaces and build vertex/index arrays.
        foreach (string tileCoord in surfacesByTile.Keys)
        {
            if (!tileDocs.TryGetValue(tileCoord, out Pm4ResearchDocument? doc))
                continue;

            // Parse tile coordinates from "X_Y" format.
            string[] parts = tileCoord.Split('_');
            if (parts.Length != 2 || !int.TryParse(parts[0], out int tX) || !int.TryParse(parts[1], out int tY))
                continue;

            tileX = tX;
            tileY = tY;

            IReadOnlyList<Pm4MsurEntry> tileMsur = doc.KnownChunks.Msur;
            IReadOnlyList<Vector3> tileMsvt = doc.KnownChunks.Msvt;
            IReadOnlyList<uint> tileMsvi = doc.KnownChunks.Msvi;
            IReadOnlyList<Pm4MprlEntry> tileMprl = doc.KnownChunks.Mprl;

            int vertexOffset = allVertices.Count;
            int indexOffset = allIndices.Count;

            // Add all vertices and indices from this tile (we need them for the full mesh).
            allVertices.AddRange(tileMsvt);
            allIndices.AddRange(tileMsvi);
            allMprl.AddRange(tileMprl);

            // Map global surface indices back to local tile indices.
            foreach (int globalSurfIdx in surfacesByTile[tileCoord])
            {
                // The global index includes the tile's offset in the region's concatenated MSUR.
                // We need to figure out the local index within this tile.
                // Since we're reading the full tile document, local index = globalSurfIdx % tileMsur.Count
                // (approximately — this works if tiles are added in order).
                int localIdx = globalSurfIdx % tileMsur.Count;
                if (localIdx >= 0 && localIdx < tileMsur.Count)
                    surfaces.Add(tileMsur[localIdx]);
            }
        }

        if (surfaces.Count == 0 || allMprl.Count == 0)
            return null;

        // Resolve coordinate mode and planar transform.
        Pm4AxisConvention axisConvention = Pm4AxisConvention.XYPlaneZUp; // Default for WoW
        Pm4CoordinateMode fallbackMode = Pm4CoordinateMode.TileLocal;

        Pm4CoordinateModeResolution coordResolution = Pm4PlacementMath.ResolveCoordinateMode(
            allVertices,
            allIndices,
            surfaces,
            allMprl,
            subObj.PositionRefs.Count > 0 ? subObj.PositionRefs : null,
            tileX,
            tileY,
            axisConvention,
            fallbackMode);

        // Resolve placement solution.
        Pm4PlacementSolution solution = Pm4PlacementMath.ResolvePlacementSolution(
            allVertices,
            allIndices,
            surfaces,
            allMprl,
            subObj.PositionRefs.Count > 0 ? subObj.PositionRefs : null,
            tileX,
            tileY,
            coordResolution.CoordinateMode,
            axisConvention);

        // Use the world pivot from the placement solution as the object's world position.
        Vector3 worldPosition = solution.WorldPivot;
        float headingDegrees = solution.WorldYawCorrectionRadians * (180f / MathF.PI);

        // Compute world bounds from the sub-object's local bounds.
        Pm4Bounds3 worldBounds = subObj.Bounds ?? new Pm4Bounds3(Vector3.Zero, Vector3.Zero);

        // MPRL.Unk04 is NOT a heading, and this used to treat it as one - reading it as a 16-bit
        // angle via Unk04 * 2*pi/65536 and averaging per object. Two measurements refute that.
        //
        // A 16-bit angle uses its whole range; Unk04 spans 0..38317 over 141,218 records and never
        // approaches 65535. And it behaves as a spatial GROUP key instead: points sharing a value sit
        // a median 1.62 world units apart against 170.99 for random groups of the same size from the
        // same file - about 105 times tighter than chance. It says which footprint a terrain-contact
        // point belongs to.
        //
        // No heading is emitted rather than a fabricated one. Rotation is genuinely not recovered
        // from PM4 yet, and a plausible wrong number is worse than an absent one.
        const float mprlHeadingMean = 0f;

        return new Pm4DecodedObjectPlacement(
            regionId,
            obj.Ck24,
            obj.Ck24Type,
            obj.Ck24ObjectId,
            subObj.GroupObjectId,
            worldPosition,
            mprlHeadingMean,
            worldBounds,
            subObj.SurfaceCount,
            subObj.PositionRefCount,
            obj.TotalIndexCount,
            obj.TileCoordinates,
            coordResolution.CoordinateMode,
            solution.AxisConvention,
            solution.PlanarTransform,
            solution.WorldYawCorrectionRadians * (180f / MathF.PI));
    }
}
