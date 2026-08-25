using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Reports which tiles can have their object placements restored from PM4 data, and writes them as
/// complete monolithic WotLK ADTs.
/// </summary>
/// <remarks>
/// <b>Nothing is ever written over a source file.</b> Output goes to a separate directory.
///
/// <para>An earlier version wrote placement-only <c>_obj0.adt</c> files. That was wrong for the actual
/// use: a split-file <c>_obj0</c> needs the rest of its family to mean anything, and outside this
/// viewer essentially nothing reads one. Testing happens on a 3.3.5 client, so the output has to be a
/// whole monolithic ADT with every chunk that format requires. <see cref="BlankAdtFactory"/> supplies
/// a complete base tile and <see cref="LkAdtWriter"/> emits it, so the chunk set is whatever a real
/// ADT has rather than whatever this file remembered to write.</para>
///
/// <para><b>Only tiles with no surviving terrain are written.</b> A blank base replaces terrain with a
/// flat sheet, which costs nothing on a tile that has none and would destroy a tile that still has its
/// own. Restoring placements onto surviving terrain needs a reader for the existing ADT, which does not
/// exist in this repo yet; those tiles are reported and skipped rather than quietly flattened.</para>
///
/// <para>Every file written is read back through the real <see cref="AdtPlacementReader"/> and compared
/// against what went in. A writer checked only by eye produces plausible rubbish, so the round-trip is
/// part of the operation rather than a separate test.</para>
///
/// <para>Rotation is not recovered and is written as zero rather than guessed.</para>
/// </remarks>
internal static class Pm4AdtPatchSupport
{
    private const int ModfEntrySize = 64;
    private const int SyntheticUniqueIdBase = 0x40000000;
    private const string Unknown = "missingwmo.wmo";
    private const string UnknownPath = @"World\wmo\missingwmo.wmo";

    public static Pm4AdtPatchReport Run(
        string pm4Directory,
        string adtDirectory,
        string? outputDirectory,
        bool includeUnknown = true)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        var tiles = new List<Pm4RestorableTile>();
        int written = 0, roundTripChecked = 0, roundTripFailed = 0, skippedHasTerrain = 0;
        int nextUniqueId = SyntheticUniqueIdBase;

        // Library learned from tiles that still have placements, so names can be proposed elsewhere.
        Dictionary<string, Vector3> library = BuildLibrary(resolved, adtDirectory);

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tFirst, out int tSecond))
                continue;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            Dictionary<uint, (Vector3 Min, Vector3 Max)> objects = ExtractObjects(chunks);
            if (objects.Count == 0)
                continue;

            string stem = Path.GetFileNameWithoutExtension(pm4Path);
            int underscore = stem.IndexOf('_');
            string mapName = underscore > 0 ? stem[..underscore] : stem;
            int existingPlacements = CountExistingPlacements(pm4Path, adtDirectory);
            bool hasTerrain = HasTerrain(pm4Path, adtDirectory, tFirst, tSecond);

            // Restorable means the PM4 knows about objects the tile's own files do not.
            int missing = objects.Count - existingPlacements;
            bool restorable = missing > 0;

            var rows = new List<PatchRow>();
            int named = 0;
            foreach ((uint key, (Vector3 min, Vector3 max)) in objects.OrderBy(static o => o.Key))
            {
                string asset = BestCandidate(library, min, max, out double score) ?? Unknown;
                if (!asset.Equals(Unknown, StringComparison.OrdinalIgnoreCase))
                    named++;
                else if (!includeUnknown)
                    continue;

                rows.Add(new PatchRow(asset, min, max, BitConverter.UInt32BitsToSingle(key), score));
            }

            tiles.Add(new Pm4RestorableTile(
                stem, tFirst, tSecond, hasTerrain, existingPlacements,
                objects.Count, missing, named, restorable));

            if (string.IsNullOrWhiteSpace(outputDirectory) || !restorable || rows.Count == 0)
                continue;

            // A blank base flattens terrain, so it is only safe where there is none to lose.
            if (hasTerrain)
            {
                skippedHasTerrain++;
                continue;
            }

            string outPath = Path.Combine(outputDirectory, $"{stem}.adt");
            AdtPlacementCatalog catalog = BuildCatalog(rows, ref nextUniqueId);
            LkAdtData blank = BlankAdtFactory.CreateBlank(mapName, tFirst, tSecond);
            LkAdtData populated = BlankAdtFactory.WithPlacements(blank, catalog);
            LkAdtWriter.Write(outPath, populated);
            written++;

            roundTripChecked++;
            if (!VerifyRoundTrip(outPath, rows))
                roundTripFailed++;
        }

        return new Pm4AdtPatchReport(
            resolved, adtDirectory, outputDirectory,
            tiles.Count,
            tiles.Count(static t => t.Restorable),
            tiles.Count(static t => t.Restorable && !t.HasTerrain),
            tiles.Count(static t => t.Restorable && t.ExistingPlacements == 0),
            tiles.Sum(static t => t.MissingObjects),
            tiles.Where(static t => t.Restorable).Sum(static t => t.NameableObjects),
            written, roundTripChecked, roundTripFailed, skippedHasTerrain,
            [.. tiles.Where(static t => t.Restorable).OrderByDescending(static t => t.MissingObjects)]);
    }

    private readonly record struct PatchRow(string Asset, Vector3 Min, Vector3 Max, float PlacementZ, double Score);

    /// <summary>
    /// Turns recovered rows into a placement catalog the ADT factory can consume.
    /// </summary>
    /// <remarks>
    /// Positions and extents go in as ADT placement space, which is what the catalog carries and what
    /// the factory expects, so no coordinate maths happens here at all. Unique ids come from a high
    /// base so they cannot collide with ids in surviving files.
    /// </remarks>
    private static AdtPlacementCatalog BuildCatalog(List<PatchRow> rows, ref int nextUniqueId)
    {
        List<string> names = [.. rows
            .Select(static r => r.Asset.Equals(Unknown, StringComparison.OrdinalIgnoreCase) ? UnknownPath : ResolveWmoPath(r.Asset))
            .Distinct(StringComparer.OrdinalIgnoreCase)];

        var placements = new List<AdtWorldModelPlacement>(rows.Count);
        foreach (PatchRow row in rows)
        {
            string path = row.Asset.Equals(Unknown, StringComparison.OrdinalIgnoreCase) ? UnknownPath : ResolveWmoPath(row.Asset);
            int nameId = names.FindIndex(n => n.Equals(path, StringComparison.OrdinalIgnoreCase));

            placements.Add(new AdtWorldModelPlacement(
                Math.Max(nameId, 0),
                path,
                nextUniqueId++,
                new Vector3((row.Min.X + row.Max.X) * 0.5f, (row.Min.Y + row.Max.Y) * 0.5f, row.PlacementZ),
                Vector3.Zero,
                row.Min,
                row.Max,
                0));
        }

        return new AdtPlacementCatalog(
            string.Empty,
            MapFileKind.Adt,
            [],
            names,
            [],
            placements);
    }

    private static bool VerifyRoundTrip(string path, List<PatchRow> rows)
    {
        try
        {
            AdtPlacementCatalog catalog = AdtPlacementReader.Read(path);
            if (catalog.WorldModelPlacements.Count != rows.Count)
                return false;

            for (int i = 0; i < rows.Count; i++)
            {
                AdtWorldModelPlacement read = catalog.WorldModelPlacements[i];
                Vector3 expected = new(
                    (rows[i].Min.X + rows[i].Max.X) * 0.5f,
                    (rows[i].Min.Y + rows[i].Max.Y) * 0.5f,
                    rows[i].PlacementZ);

                if (Vector3.Distance(read.Position, expected) > 0.05f)
                    return false;
                if (Vector3.Distance(read.BoundsMin, rows[i].Min) > 0.05f)
                    return false;
                if (Vector3.Distance(read.BoundsMax, rows[i].Max) > 0.05f)
                    return false;
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private static void WriteChunk(BinaryWriter bw, string tag, byte[] payload)
    {
        for (int i = tag.Length - 1; i >= 0; i--)
            bw.Write((byte)tag[i]);
        bw.Write(payload.Length);
        bw.Write(payload);
    }

    private static void WriteSingle(Span<byte> target, float value) =>
        BinaryPrimitives.WriteUInt32LittleEndian(target, BitConverter.SingleToUInt32Bits(value));

    private static string ResolveWmoPath(string asset) => $@"World\wmo\{asset}";

    private static Dictionary<string, Vector3> BuildLibrary(string resolved, string adtDirectory)
    {
        var shapes = new Dictionary<string, List<Vector3>>(StringComparer.OrdinalIgnoreCase);

        foreach (string pm4Path in Directory.EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly))
        {
            string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtDirectory);
            if (adtPath is null)
                continue;

            AdtPlacementCatalog catalog;
            try
            {
                catalog = AdtPlacementReader.Read(adtPath);
            }
            catch
            {
                continue;
            }

            if (catalog.WorldModelPlacements.Count == 0)
                continue;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            var byHeight = new Dictionary<uint, AdtWorldModelPlacement>();
            foreach (AdtWorldModelPlacement row in catalog.WorldModelPlacements)
                byHeight[BitConverter.SingleToUInt32Bits(row.Position.Z)] = row;

            foreach ((uint key, (Vector3 min, Vector3 max)) in ExtractObjects(chunks))
            {
                if (!byHeight.TryGetValue(key, out AdtWorldModelPlacement truth))
                    continue;

                string name = Path.GetFileName(truth.ModelPath);
                if (string.IsNullOrWhiteSpace(name))
                    continue;

                if (!shapes.TryGetValue(name, out List<Vector3>? list))
                {
                    list = [];
                    shapes[name] = list;
                }

                list.Add(ShapeOf(min, max));
            }
        }

        var library = new Dictionary<string, Vector3>(StringComparer.OrdinalIgnoreCase);
        foreach ((string asset, List<Vector3> list) in shapes)
        {
            List<float> xs = [.. list.Select(static v => v.X).Order()];
            List<float> ys = [.. list.Select(static v => v.Y).Order()];
            List<float> zs = [.. list.Select(static v => v.Z).Order()];
            library[asset] = new Vector3(xs[xs.Count / 2], ys[ys.Count / 2], zs[zs.Count / 2]);
        }

        return library;
    }

    private static string? BestCandidate(Dictionary<string, Vector3> library, Vector3 min, Vector3 max, out double score)
    {
        Vector3 observed = ShapeOf(min, max);
        string? best = null;
        score = double.MaxValue;

        foreach ((string asset, Vector3 median) in library)
        {
            double d = ShapeDistance(observed, median);
            if (d < score)
            {
                score = d;
                best = asset;
            }
        }

        return score <= 0.25 ? best : null;
    }

    private static Vector3 ShapeOf(Vector3 min, Vector3 max)
    {
        float dx = MathF.Abs(max.X - min.X);
        float dy = MathF.Abs(max.Y - min.Y);
        return new Vector3(MathF.Min(dx, dy), MathF.Max(dx, dy), MathF.Abs(max.Z - min.Z));
    }

    private static double ShapeDistance(Vector3 a, Vector3 b)
    {
        static double Rel(float p, float q)
        {
            float scale = MathF.Max(MathF.Max(MathF.Abs(p), MathF.Abs(q)), 1f);
            return MathF.Abs(p - q) / scale;
        }

        return Rel(a.X, b.X) + Rel(a.Y, b.Y) + (2.0 * Rel(a.Z, b.Z));
    }

    private static Dictionary<uint, (Vector3 Min, Vector3 Max)> ExtractObjects(Pm4KnownChunkSet chunks)
    {
        var lo = new Dictionary<uint, Vector3>();
        var hi = new Dictionary<uint, Vector3>();

        foreach (Pm4MsurEntry surface in chunks.Msur)
        {
            if (surface.PackedParams == 0)
                continue;

            long vs = surface.MsviFirstIndex;
            long ve = vs + surface.IndexCount;
            if (surface.IndexCount == 0 || ve > chunks.Msvi.Count)
                continue;

            for (long v = vs; v < ve; v++)
            {
                uint vi = chunks.Msvi[(int)v];
                if (vi >= chunks.Msvt.Count)
                    continue;

                Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(chunks.Msvt[(int)vi]);
                if (!lo.TryGetValue(surface.PackedParams, out Vector3 cur))
                {
                    lo[surface.PackedParams] = p;
                    hi[surface.PackedParams] = p;
                    continue;
                }

                lo[surface.PackedParams] = Vector3.Min(cur, p);
                hi[surface.PackedParams] = Vector3.Max(hi[surface.PackedParams], p);
            }
        }

        var result = new Dictionary<uint, (Vector3, Vector3)>();
        foreach ((uint key, Vector3 min) in lo)
            result[key] = (min, hi[key]);

        return result;
    }

    private static int CountExistingPlacements(string pm4Path, string adtDirectory)
    {
        string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtDirectory);
        if (adtPath is null)
            return 0;

        try
        {
            return AdtPlacementReader.Read(adtPath).WorldModelPlacements.Count;
        }
        catch
        {
            return 0;
        }
    }

    private static bool HasTerrain(string pm4Path, string adtDirectory, int tileFirst, int tileSecond)
    {
        string stem = Path.GetFileNameWithoutExtension(pm4Path);
        int underscore = stem.IndexOf('_');
        string mapName = underscore > 0 ? stem[..underscore] : stem;

        foreach (string candidate in new[]
        {
            Path.Combine(adtDirectory, $"{stem}.adt"),
            Path.Combine(adtDirectory, $"{mapName}_{tileFirst}_{tileSecond}.adt"),
        })
        {
            if (File.Exists(candidate) && new FileInfo(candidate).Length > 1024)
                return true;
        }

        return false;
    }
}

internal sealed record Pm4RestorableTile(
    string Tile,
    int TileFirst,
    int TileSecond,
    bool HasTerrain,
    int ExistingPlacements,
    int Pm4Objects,
    int MissingObjects,
    int NameableObjects,
    bool Restorable);

internal sealed record Pm4AdtPatchReport(
    string Pm4Directory,
    string AdtDirectory,
    string? OutputDirectory,
    int TilesExamined,
    int TilesRestorable,
    int RestorableWithoutTerrain,
    int RestorableWithNoPlacementsAtAll,
    int MissingObjectsTotal,
    int NameableObjects,
    int FilesWritten,
    int RoundTripChecked,
    int RoundTripFailed,
    int SkippedBecauseTerrainWouldBeLost,
    IReadOnlyList<Pm4RestorableTile> Tiles);
