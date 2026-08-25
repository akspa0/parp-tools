using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Reports which tiles can have their object placements restored from PM4 data, and writes those
/// placements as standalone <c>_obj0.adt</c> files.
/// </summary>
/// <remarks>
/// <b>Nothing is ever written over a source file.</b> Output goes to a separate directory, and the
/// files written are placement-only: <c>MVER</c>, the name tables, <c>MDDF</c> and <c>MODF</c>. They
/// carry no terrain, so no amount of getting this wrong can damage a heightmap.
///
/// <para>The <c>MODF</c> layout here is written as the mirror of <see cref="AdtPlacementReader"/>,
/// field for field, and every file written is read back through that reader and compared against what
/// went in. A writer checked only by eye is a writer that silently produces plausible rubbish, so the
/// round-trip is part of the operation rather than a separate test.</para>
///
/// <para>Rotation is not recovered, and is written as zero rather than guessed. Unique ids are
/// synthesised from a high base so they cannot collide with ids in surviving files.</para>
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
        int written = 0, roundTripChecked = 0, roundTripFailed = 0;
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

            string outPath = Path.Combine(outputDirectory, $"{stem}_obj0.adt");
            byte[] bytes = BuildObj0(rows, ref nextUniqueId);
            File.WriteAllBytes(outPath, bytes);
            written++;

            // Read it back through the real reader and compare. A writer nobody checks is a writer
            // that produces plausible rubbish.
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
            written, roundTripChecked, roundTripFailed,
            [.. tiles.Where(static t => t.Restorable).OrderByDescending(static t => t.MissingObjects)]);
    }

    private readonly record struct PatchRow(string Asset, Vector3 Min, Vector3 Max, float PlacementZ, double Score);

    /// <summary>
    /// Builds a placement-only ADT: version, name tables, an empty doodad list and the world-model
    /// placements. Chunk tags are written reversed, which is how they sit on disk.
    /// </summary>
    private static byte[] BuildObj0(List<PatchRow> rows, ref int nextUniqueId)
    {
        List<string> names = [.. rows.Select(static r =>
            r.Asset.Equals(Unknown, StringComparison.OrdinalIgnoreCase) ? UnknownPath : ResolveWmoPath(r.Asset))
            .Distinct(StringComparer.OrdinalIgnoreCase)];

        var nameOffsets = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var mwmo = new List<byte>();
        foreach (string name in names)
        {
            nameOffsets[name] = mwmo.Count;
            mwmo.AddRange(Encoding.ASCII.GetBytes(name));
            mwmo.Add(0);
        }

        byte[] mwid = new byte[names.Count * 4];
        for (int i = 0; i < names.Count; i++)
            BinaryPrimitives.WriteInt32LittleEndian(mwid.AsSpan(i * 4, 4), nameOffsets[names[i]]);

        byte[] modf = new byte[rows.Count * ModfEntrySize];
        for (int i = 0; i < rows.Count; i++)
        {
            PatchRow row = rows[i];
            string path = row.Asset.Equals(Unknown, StringComparison.OrdinalIgnoreCase) ? UnknownPath : ResolveWmoPath(row.Asset);
            int nameId = names.FindIndex(n => n.Equals(path, StringComparison.OrdinalIgnoreCase));

            Vector3 position = new((row.Min.X + row.Max.X) * 0.5f, (row.Min.Y + row.Max.Y) * 0.5f, row.PlacementZ);
            Span<byte> e = modf.AsSpan(i * ModfEntrySize, ModfEntrySize);

            BinaryPrimitives.WriteUInt32LittleEndian(e[0..4], (uint)Math.Max(nameId, 0));
            BinaryPrimitives.WriteUInt32LittleEndian(e[4..8], unchecked((uint)nextUniqueId++));

            // Mirror of AdtPlacementReader: it reads rawX, rawZ, rawY at 8/12/16 and forms
            // (MapOrigin - rawY, MapOrigin - rawX, rawZ), so the inverse swaps X and Y back.
            WriteSingle(e[8..12], Pm4CoordinateService.MapOrigin - position.Y);
            WriteSingle(e[12..16], position.Z);
            WriteSingle(e[16..20], Pm4CoordinateService.MapOrigin - position.X);

            // Rotation is not recovered. Zero is honest; a guess would not be.
            WriteSingle(e[20..24], 0f);
            WriteSingle(e[24..28], 0f);
            WriteSingle(e[28..32], 0f);

            WriteSingle(e[32..36], Pm4CoordinateService.MapOrigin - row.Max.Y);
            WriteSingle(e[36..40], row.Min.Z);
            WriteSingle(e[40..44], Pm4CoordinateService.MapOrigin - row.Max.X);
            WriteSingle(e[44..48], Pm4CoordinateService.MapOrigin - row.Min.Y);
            WriteSingle(e[48..52], row.Max.Z);
            WriteSingle(e[52..56], Pm4CoordinateService.MapOrigin - row.Min.X);

            BinaryPrimitives.WriteUInt16LittleEndian(e[56..58], 0);
            BinaryPrimitives.WriteUInt16LittleEndian(e[58..60], 0);
            BinaryPrimitives.WriteUInt16LittleEndian(e[60..62], 0);
            BinaryPrimitives.WriteUInt16LittleEndian(e[62..64], 0);
        }

        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        WriteChunk(bw, "MVER", BitConverter.GetBytes(18));
        WriteChunk(bw, "MMDX", []);
        WriteChunk(bw, "MMID", []);
        WriteChunk(bw, "MWMO", [.. mwmo]);
        WriteChunk(bw, "MWID", mwid);
        WriteChunk(bw, "MDDF", []);
        WriteChunk(bw, "MODF", modf);
        bw.Flush();
        return ms.ToArray();
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
    IReadOnlyList<Pm4RestorableTile> Tiles);
