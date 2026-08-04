using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Splits the CK24 0 remainder into spatially connected components, checks those components against
/// real doodad placements, and then asks which PM4 field is constant within a component.
/// </summary>
/// <remarks>
/// <para><b>Why the search is inverted.</b> The obvious approach — pick a field, count its distinct
/// values, see whether the count matches the tile's doodad count — was tried and failed. It fails
/// for two reasons. The count is a weak signal: two fields can produce the right number of groups
/// and group entirely different surfaces. And the denominator was wrong, because most M2 doodads
/// generate no collision at all (Blizzard's own editor shows candelabras, cobwebs and banners inside
/// the Karazhan crypts with no nav polygons under them), so no field should ever match the raw MDDF
/// count.</para>
///
/// <para>So this derives the grouping first, from geometry, and only then looks for a field that
/// reproduces it. A component is a set of surfaces reachable from one another by shared vertices —
/// the connectivity the mesh itself asserts, with no field involved.</para>
///
/// <para><b>Two questions, kept apart.</b> First: are components the right unit — do they land on
/// doodad placements one-for-one? That is answered against MDDF with a distance distribution, and it
/// can fail. Second, and independently: is any field constant within a component and different
/// between components? A field can score well on the second while the first is still unresolved, and
/// that is worth knowing, so the two are never combined into one score.</para>
///
/// <para><b>What a good separator looks like.</b> <see cref="Pm4FieldSeparatorScore.Purity"/> is the
/// fraction of components whose surfaces all carry one value — a field that varies inside a
/// component cannot be the identity. <see cref="Pm4FieldSeparatorScore.Distinctness"/> is the
/// fraction of components whose value is used by no other component in the tile. A constant field
/// scores purity 1.0 and distinctness 0.0, which is why both are always reported; neither alone
/// identifies anything.</para>
/// </remarks>
public static class Pm4ComponentIdentityAnalyzer
{
    /// <summary>Distance below which two surface vertices are treated as the same point.</summary>
    public const float WeldEpsilon = 0.25f;

    /// <summary>How close a component centroid must be to an MDDF position to count as landing on it.</summary>
    public const float DoodadMatchRadius = 24f;

    public static Pm4ComponentIdentityReport AnalyzeDirectory(
        string inputDirectory,
        IReadOnlyDictionary<string, Pm4TilePlacements> placementsByFile,
        int maxTiles = 0)
    {
        ArgumentNullException.ThrowIfNull(placementsByFile);
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ComponentRecord> components = [];
        List<Pm4ComponentTileRecord> tiles = [];

        foreach (string path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName, StringComparer.Ordinal))
        {
            if (maxTiles > 0 && tiles.Count >= maxTiles)
                break;

            string fileName = Path.GetFileName(path);
            if (!placementsByFile.TryGetValue(fileName, out Pm4TilePlacements? placements))
                continue;

            Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(path);
            Pm4KnownChunkSet chunks = document.KnownChunks;
            if (chunks.Msvt.Count == 0 || chunks.Msvi.Count == 0 || chunks.Msur.Count == 0)
                continue;

            if (!Pm4CoordinateService.TryParseTileCoordinates(path, out int tileFirst, out int tileSecond))
                continue;

            Pm4ComponentTileRecord? tile = AnalyzeTile(chunks, fileName, tileFirst, tileSecond, placements, components);
            if (tile is not null)
                tiles.Add(tile);
        }

        return BuildReport(resolvedDirectory, tiles, components);
    }

    private static Pm4ComponentTileRecord? AnalyzeTile(
        Pm4KnownChunkSet chunks,
        string fileName,
        int tileFirst,
        int tileSecond,
        Pm4TilePlacements placements,
        List<Pm4ComponentRecord> components)
    {
        // Only the CK24 0 remainder is in scope; keyed CK24 objects are already resolved as WMOs.
        List<int> surfaceIndices = [];
        for (int index = 0; index < chunks.Msur.Count; index++)
        {
            if (chunks.Msur[index].Ck24 == 0u)
                surfaceIndices.Add(index);
        }

        if (surfaceIndices.Count == 0)
            return null;

        Dictionary<int, List<Pm4MslkEntry>> linksBySurface = BuildLinkIndex(chunks);
        List<List<int>> groups = BuildConnectedComponents(chunks, surfaceIndices);

        int landedOnDoodad = 0;
        int componentsBefore = components.Count;

        foreach (List<int> group in groups)
        {
            List<Pm4MsurEntry> surfaces = [.. group.Select(index => chunks.Msur[index])];
            List<Vector3> vertices = Pm4PlacementMath.CollectSurfaceVertices(chunks.Msvt, chunks.Msvi, surfaces);
            if (vertices.Count == 0)
                continue;

            Vector3 centroid = Vector3.Zero;
            Vector3 min = new(float.MaxValue), max = new(float.MinValue);
            foreach (Vector3 vertex in vertices)
            {
                centroid += vertex;
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            centroid /= vertices.Count;
            Vector3 placement = Pm4CoordinateService.Pm4LocalToAdtPlacement(centroid);
            Vector2 planar = new(placement.X, placement.Y);

            float distance = float.MaxValue;
            string modelPath = string.Empty;
            foreach (Pm4NamedPoint doodad in placements.DoodadPositions)
            {
                float candidate = Vector2.Distance(new Vector2(doodad.X, doodad.Y), planar);
                if (candidate < distance)
                {
                    distance = candidate;
                    modelPath = doodad.ModelPath;
                }
            }

            bool onDoodad = distance <= DoodadMatchRadius;
            if (onDoodad)
                landedOnDoodad++;

            HashSet<uint> groupObjectIds = [];
            HashSet<uint> linkIds = [];
            HashSet<byte> typeFlags = [];
            HashSet<byte> groupKeys = [];
            HashSet<byte> attributeMasks = [];
            int anchorLinks = 0;

            foreach (int index in group)
            {
                groupKeys.Add(chunks.Msur[index].GroupKey);
                attributeMasks.Add(chunks.Msur[index].AttributeMask);
                if (!linksBySurface.TryGetValue(index, out List<Pm4MslkEntry>? links))
                    continue;

                foreach (Pm4MslkEntry link in links)
                {
                    groupObjectIds.Add(link.GroupObjectId);
                    linkIds.Add(link.LinkId);
                    typeFlags.Add(link.TypeFlags);
                    if (link.MspiFirstIndex < 0)
                        anchorLinks++;
                }
            }

            Vector3 extent = max - min;
            components.Add(new Pm4ComponentRecord(
                fileName,
                tileFirst,
                tileSecond,
                chunks.Mshd?.Field04 ?? 0u,
                components.Count - componentsBefore,
                group.Count,
                vertices.Count,
                extent.X,
                extent.Y,
                extent.Z,
                distance == float.MaxValue ? -1f : distance,
                onDoodad,
                modelPath,
                groupObjectIds.Count,
                linkIds.Count,
                typeFlags.Count,
                groupKeys.Count,
                attributeMasks.Count,
                anchorLinks,
                groupObjectIds.Count == 1 ? groupObjectIds.First() : 0u,
                linkIds.Count == 1 ? linkIds.First() : 0u,
                typeFlags.Count == 1 ? typeFlags.First() : -1,
                groupKeys.Count == 1 ? groupKeys.First() : -1,
                attributeMasks.Count == 1 ? attributeMasks.First() : -1));
        }

        return new Pm4ComponentTileRecord(
            fileName,
            tileFirst,
            tileSecond,
            chunks.Mshd?.Field04 ?? 0u,
            surfaceIndices.Count,
            components.Count - componentsBefore,
            placements.DoodadPositions.Count,
            landedOnDoodad);
    }

    private static Dictionary<int, List<Pm4MslkEntry>> BuildLinkIndex(Pm4KnownChunkSet chunks)
    {
        Dictionary<int, List<Pm4MslkEntry>> linksBySurface = [];
        foreach (Pm4MslkEntry link in chunks.Mslk)
        {
            if (link.RefIndex >= chunks.Msur.Count)
                continue;

            if (!linksBySurface.TryGetValue(link.RefIndex, out List<Pm4MslkEntry>? bucket))
            {
                bucket = [];
                linksBySurface[link.RefIndex] = bucket;
            }

            bucket.Add(link);
        }

        return linksBySurface;
    }

    /// <summary>
    /// Groups surfaces that share a welded vertex position into connected components.
    /// </summary>
    /// <remarks>
    /// Position welding rather than index identity, because MSVT reuses indices unevenly and two
    /// surfaces of one asset can reference distinct indices at the same point. The quantisation is
    /// <see cref="WeldEpsilon"/>, matching the epsilon the connective-geometry work used when it
    /// measured MSPV/MSVT coincidence, so the two analyses agree on what "the same point" means.
    /// </remarks>
    private static List<List<int>> BuildConnectedComponents(Pm4KnownChunkSet chunks, List<int> surfaceIndices)
    {
        Dictionary<(int, int, int), List<int>> surfacesByCell = [];
        Dictionary<int, List<(int, int, int)>> cellsBySurface = new(surfaceIndices.Count);

        foreach (int surfaceIndex in surfaceIndices)
        {
            Pm4MsurEntry surface = chunks.Msur[surfaceIndex];
            int first = (int)surface.MsviFirstIndex;
            int end = Math.Min(first + surface.IndexCount, chunks.Msvi.Count);
            if (surface.IndexCount <= 0 || first < 0 || end <= first)
                continue;

            List<(int, int, int)> cells = [];
            for (int i = first; i < end; i++)
            {
                int vertexIndex = (int)chunks.Msvi[i];
                if ((uint)vertexIndex >= (uint)chunks.Msvt.Count)
                    continue;

                Vector3 v = chunks.Msvt[vertexIndex];
                (int, int, int) cell = (
                    (int)MathF.Round(v.X / WeldEpsilon),
                    (int)MathF.Round(v.Y / WeldEpsilon),
                    (int)MathF.Round(v.Z / WeldEpsilon));

                cells.Add(cell);
                if (!surfacesByCell.TryGetValue(cell, out List<int>? bucket))
                {
                    bucket = [];
                    surfacesByCell[cell] = bucket;
                }

                bucket.Add(surfaceIndex);
            }

            cellsBySurface[surfaceIndex] = cells;
        }

        Dictionary<int, int> parent = new(surfaceIndices.Count);
        foreach (int surfaceIndex in surfaceIndices)
            parent[surfaceIndex] = surfaceIndex;

        int Find(int index)
        {
            while (parent[index] != index)
            {
                parent[index] = parent[parent[index]];
                index = parent[index];
            }

            return index;
        }

        void Union(int a, int b)
        {
            int rootA = Find(a);
            int rootB = Find(b);
            if (rootA != rootB)
                parent[rootB] = rootA;
        }

        foreach (List<int> shared in surfacesByCell.Values)
        {
            for (int i = 1; i < shared.Count; i++)
                Union(shared[0], shared[i]);
        }

        Dictionary<int, List<int>> byRoot = [];
        foreach (int surfaceIndex in surfaceIndices)
        {
            if (!cellsBySurface.ContainsKey(surfaceIndex))
                continue;

            int root = Find(surfaceIndex);
            if (!byRoot.TryGetValue(root, out List<int>? bucket))
            {
                bucket = [];
                byRoot[root] = bucket;
            }

            bucket.Add(surfaceIndex);
        }

        return [.. byRoot.Values];
    }

    private static Pm4ComponentIdentityReport BuildReport(
        string resolvedDirectory,
        List<Pm4ComponentTileRecord> tiles,
        List<Pm4ComponentRecord> components)
    {
        List<Pm4ComponentRecord> matched = [.. components.Where(static c => c.LandsOnDoodad)];

        List<Pm4FieldSeparatorScore> separators =
        [
            ScoreField("MSLK.GroupObjectId", components, static c => c.DistinctGroupObjectIds, static c => c.SoleGroupObjectId),
            ScoreField("MSLK.LinkId", components, static c => c.DistinctLinkIds, static c => c.SoleLinkId),
            ScoreField("MSLK.TypeFlags", components, static c => c.DistinctTypeFlags, static c => unchecked((uint)c.SoleTypeFlags)),
            ScoreField("MSUR.GroupKey", components, static c => c.DistinctGroupKeys, static c => unchecked((uint)c.SoleGroupKey)),
            ScoreField("MSUR.AttributeMask", components, static c => c.DistinctAttributeMasks, static c => unchecked((uint)c.SoleAttributeMask))
        ];

        // Distinctness needs the per-tile view: a value is only an identity if no OTHER component
        // on the same tile uses it.
        Dictionary<string, int> reusedGroupObjectIds = [];
        foreach (IGrouping<string, Pm4ComponentRecord> tileComponents in components
            .Where(static c => c.DistinctGroupObjectIds == 1)
            .GroupBy(static c => c.FileName, StringComparer.Ordinal))
        {
            HashSet<uint> seen = [];
            int reused = 0;
            foreach (Pm4ComponentRecord component in tileComponents)
            {
                if (!seen.Add(component.SoleGroupObjectId))
                    reused++;
            }

            reusedGroupObjectIds[tileComponents.Key] = reused;
        }

        int pureGroupObjectIdComponents = components.Count(static c => c.DistinctGroupObjectIds == 1);
        int reusedTotal = reusedGroupObjectIds.Values.Sum();

        double componentsPerDoodad = tiles.Count == 0 || tiles.Sum(static t => t.DoodadPlacements) == 0
            ? 0d
            : (double)components.Count / tiles.Sum(static t => t.DoodadPlacements);

        string verdict = components.Count == 0
            ? "no CK24 0 components found"
            : matched.Count * 2 >= components.Count
                ? $"components track doodads — {Frac(matched.Count, components.Count):P1} land within "
                    + $"{DoodadMatchRadius:F0} units of an MDDF placement"
                : $"components do NOT track doodads one-for-one — only {Frac(matched.Count, components.Count):P1} "
                    + "land near an MDDF placement, so the CK24 0 remainder is not simply a pile of doodads "
                    + "(terrain collision is the obvious other occupant)";

        List<string> notes =
        [
            "Components come from geometry alone: surfaces welded through shared vertex positions at "
                + $"epsilon {WeldEpsilon}. No field takes part in forming them, so a field that "
                + "reproduces them is genuine evidence rather than a restatement.",
            "PURITY is the fraction of components whose surfaces all carry one value for the field. "
                + "A field that varies inside a component cannot be the per-component identity.",
            "DISTINCTNESS is the fraction of pure components whose value no OTHER component on the "
                + "same tile shares. A constant field scores purity 1.0 and distinctness 0.0, which is "
                + "why purity alone proves nothing.",
            "Most M2 doodads produce no collision at all — Blizzard's editor shows candelabras, "
                + "cobwebs and banners in the Karazhan crypts with no nav polygons beneath them. So "
                + "component count is EXPECTED to fall well short of the tile's MDDF count, and the "
                + "shortfall is not evidence against the components.",
            "Whether components equal doodads and whether a field equals components are separate "
                + "questions and are never combined into one score; a field can win the second while "
                + "the first is still open."
        ];

        return new Pm4ComponentIdentityReport(
            resolvedDirectory,
            tiles.Count,
            components.Count,
            matched.Count,
            Frac(matched.Count, components.Count),
            componentsPerDoodad,
            pureGroupObjectIdComponents,
            reusedTotal,
            verdict,
            separators,
            [.. tiles.OrderByDescending(static t => t.ComponentCount).Take(20)],
            [.. matched.OrderBy(static c => c.NearestDoodadDistance).Take(24)],
            notes);
    }

    /// <summary>
    /// Scores a field on both halves of the question: is it constant within a component (purity),
    /// and is that value unique among the components of the same tile (distinctness).
    /// </summary>
    /// <remarks>
    /// Distinctness is what separates an identity from a class marker. A field that is the same for
    /// every component scores purity 1.0 and distinctness 0.0; a per-component id scores high on
    /// both. Reuse is judged per tile because PM4 ids are not known to be globally unique, and
    /// judging globally would punish a perfectly good tile-scoped id.
    /// </remarks>
    private static Pm4FieldSeparatorScore ScoreField(
        string field,
        IReadOnlyList<Pm4ComponentRecord> components,
        Func<Pm4ComponentRecord, int> distinctCount,
        Func<Pm4ComponentRecord, uint> soleValue)
    {
        if (components.Count == 0)
            return new Pm4FieldSeparatorScore(field, 0, 0d, 0d, 0, 0d, 0);

        int pure = components.Count(component => distinctCount(component) == 1);
        int absent = components.Count(component => distinctCount(component) == 0);

        int uniqueWithinTile = 0;
        List<double> distinctPerTile = [];

        foreach (IGrouping<string, Pm4ComponentRecord> tile in components
            .Where(component => distinctCount(component) == 1)
            .GroupBy(static component => component.FileName, StringComparer.Ordinal))
        {
            Dictionary<uint, int> counts = [];
            foreach (Pm4ComponentRecord component in tile)
            {
                uint value = soleValue(component);
                counts[value] = counts.TryGetValue(value, out int existing) ? existing + 1 : 1;
            }

            foreach (Pm4ComponentRecord component in tile)
            {
                if (counts[soleValue(component)] == 1)
                    uniqueWithinTile++;
            }

            distinctPerTile.Add(counts.Count);
        }

        return new Pm4FieldSeparatorScore(
            field,
            pure,
            Frac(pure, components.Count),
            Frac(absent, components.Count),
            uniqueWithinTile,
            Frac(uniqueWithinTile, pure),
            (int)Median(distinctPerTile));
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0)
            return 0d;

        values.Sort();
        int mid = values.Count / 2;
        return values.Count % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2d;
    }

    private static double Frac(int numerator, int denominator)
        => denominator == 0 ? 0d : (double)numerator / denominator;
}
