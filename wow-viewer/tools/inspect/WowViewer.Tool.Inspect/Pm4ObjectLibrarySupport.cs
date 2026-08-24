using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Builds the inventory that pairs every PM4 object in a corpus with the placed asset that produced
/// it, and reports what did NOT pair and why.
/// </summary>
/// <remarks>
/// <para>
/// The mapping key is <c>MSUR._0x1C</c> read as a float, which was measured on 2026-08-23 to equal
/// the producing placement's Z bit-exactly (93.58% of objects on the original corpus, against a
/// rotated-correspondence control of 2.10%). A candidate must also stand inside the object's
/// horizontal footprint, so the key is never used alone.
/// </para>
/// <para>
/// Every PM4 object appears in the output under some status. Objects that fail to pair are NOT
/// dropped — an inventory that silently omits its failures cannot be used to reason about coverage,
/// and the unpaired population is exactly where the remaining decode work is.
/// </para>
/// </remarks>
internal static class Pm4ObjectLibrarySupport
{
    /// <summary>World units per ADT tile.</summary>
    private const float TileSize = 533.33333f;

    /// <summary>Minimap images are square; this is their edge length in pixels.</summary>
    private const float MinimapPixels = 256f;

    internal sealed record LibraryEntry(
        string Pm4File,
        string? AdtFile,
        string RawHex,
        float PlacementZ,
        int SurfaceCount,
        int IndexCount,
        float MinX, float MinY, float MinZ,
        float MaxX, float MaxY, float MaxZ,
        int TileX,
        int TileY,
        string MinimapFile,
        float MinimapU,
        float MinimapV,
        float MinimapPixelX,
        float MinimapPixelY,
        string Status,
        string? AssetKind,
        string? AssetPath,
        int? UniqueId,
        float? MatchDelta,
        float? PosX, float? PosY, float? PosZ,
        float? RotX, float? RotY, float? RotZ);

    internal sealed record AssetSummary(
        string AssetPath,
        string AssetKind,
        int InstanceCount,
        int DistinctPm4Files,
        int MinSurfaceCount,
        int MaxSurfaceCount,
        double MeanSurfaceCount);

    internal sealed record LibraryReport(
        string Pm4Directory,
        string AdtDirectory,
        int Pm4FilesScanned,
        int Pm4FilesWithCompanionAdt,
        int TotalObjects,
        IReadOnlyList<Pm4ValueFrequency> StatusCounts,
        int DistinctAssets,
        IReadOnlyList<AssetSummary> Assets,
        IReadOnlyList<LibraryEntry> Entries);

    public static LibraryReport Build(string pm4Directory, string? adtDirectory, float tolerance)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);
        string adtRoot = string.IsNullOrWhiteSpace(adtDirectory) ? resolved : adtDirectory;

        var entries = new List<LibraryEntry>();
        var status = new Dictionary<string, int>(StringComparer.Ordinal);
        int filesScanned = 0, filesPaired = 0;

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            filesScanned++;
            string pm4Name = Path.GetFileName(pm4Path);
            string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtRoot);

            List<(string Kind, string Path, int UniqueId, Vector3 Pos, Vector3 Rot)> candidates = [];
            if (adtPath is not null)
            {
                try
                {
                    AdtPlacementCatalog cat = AdtPlacementReader.Read(adtPath);
                    foreach (AdtWorldModelPlacement w in cat.WorldModelPlacements)
                        candidates.Add(("WMO", w.ModelPath, w.UniqueId, w.Position, w.Rotation));
                    foreach (AdtModelPlacement m in cat.ModelPlacements)
                        candidates.Add(("M2", m.ModelPath, m.UniqueId, m.Position, m.Rotation));
                    filesPaired++;
                }
                catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException)
                {
                    adtPath = null;
                }
            }

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            IReadOnlyList<Pm4MsurEntry> msur = chunks.Msur;
            IReadOnlyList<uint> msvi = chunks.Msvi;
            IReadOnlyList<Vector3> msvt = chunks.Msvt;
            if (msur.Count == 0)
                continue;

            var lo = new Dictionary<uint, Vector3>();
            var hi = new Dictionary<uint, Vector3>();
            var surfaceCount = new Dictionary<uint, int>();
            var indexCount = new Dictionary<uint, int>();

            foreach (Pm4MsurEntry s in msur)
            {
                surfaceCount[s.PackedParams] = surfaceCount.GetValueOrDefault(s.PackedParams) + 1;
                indexCount[s.PackedParams] = indexCount.GetValueOrDefault(s.PackedParams) + s.IndexCount;

                long start = s.MsviFirstIndex;
                long end = start + s.IndexCount;
                if (end > msvi.Count)
                    continue;

                for (long k = start; k < end; k++)
                {
                    uint vi = msvi[(int)k];
                    if (vi >= msvt.Count)
                        continue;

                    Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(msvt[(int)vi]);
                    if (!lo.TryGetValue(s.PackedParams, out Vector3 cur))
                    {
                        lo[s.PackedParams] = p;
                        hi[s.PackedParams] = p;
                        continue;
                    }

                    lo[s.PackedParams] = Vector3.Min(cur, p);
                    hi[s.PackedParams] = Vector3.Max(hi[s.PackedParams], p);
                }
            }

            foreach ((uint raw, Vector3 low) in lo.OrderBy(static kv => kv.Key))
            {
                Vector3 high = hi[raw];
                float asFloat = BitConverter.UInt32BitsToSingle(raw);

                string st;
                string? kind = null, asset = null;
                int? uid = null;
                float? delta = null, px = null, py = null, pz = null, rx = null, ry = null, rz = null;

                if (raw == 0)
                {
                    st = "tile_remainder";
                }
                else if (adtPath is null)
                {
                    st = "no_companion_adt";
                }
                else
                {
                    var inside = candidates
                        .Where(c => c.Pos.X >= low.X - 1f && c.Pos.X <= high.X + 1f
                                 && c.Pos.Y >= low.Y - 1f && c.Pos.Y <= high.Y + 1f)
                        .ToList();

                    if (inside.Count == 0)
                    {
                        st = "no_placement_in_footprint";
                    }
                    else
                    {
                        var best = inside.OrderBy(c => MathF.Abs(asFloat - c.Pos.Z)).First();
                        float d = asFloat - best.Pos.Z;
                        delta = d;
                        if (MathF.Abs(d) <= tolerance)
                        {
                            st = "matched";
                            kind = best.Kind;
                            asset = best.Path;
                            uid = best.UniqueId;
                            px = best.Pos.X; py = best.Pos.Y; pz = best.Pos.Z;
                            rx = best.Rot.X; ry = best.Rot.Y; rz = best.Rot.Z;
                        }
                        else
                        {
                            st = "z_mismatch";
                        }
                    }
                }

                // Tile and minimap placement. The index formula was derived empirically rather
                // than assumed: over all 904 matched objects, floor(32 - centreY / TileSize) and
                // floor(32 - centreX / TileSize) reproduce the PM4 filename's two numbers at
                // 100.00%. Minimap images use the same indices as the ADT grid.
                float centreX = (low.X + high.X) / 2f;
                float centreY = (low.Y + high.Y) / 2f;
                float fx = 32f - centreY / TileSize;
                float fy = 32f - centreX / TileSize;
                int tileX = (int)MathF.Floor(fx);
                int tileY = (int)MathF.Floor(fy);
                float u = fx - tileX;
                float v = fy - tileY;

                status[st] = status.GetValueOrDefault(st) + 1;
                entries.Add(new LibraryEntry(
                    pm4Name,
                    adtPath is null ? null : Path.GetFileName(adtPath),
                    $"0x{raw:X8}",
                    asFloat,
                    surfaceCount.GetValueOrDefault(raw),
                    indexCount.GetValueOrDefault(raw),
                    low.X, low.Y, low.Z, high.X, high.Y, high.Z,
                    tileX, tileY,
                    $"map{tileX}_{tileY}.blp",
                    u, v, u * MinimapPixels, v * MinimapPixels,
                    st, kind, asset, uid, delta, px, py, pz, rx, ry, rz));
            }
        }

        // Asset-primary view: the same model placed many times is the ideal generation test set,
        // because one input must reproduce every one of its recorded outputs.
        List<AssetSummary> assets = entries
            .Where(static e => e.Status == "matched" && e.AssetPath is not null)
            .GroupBy(static e => e.AssetPath!, StringComparer.OrdinalIgnoreCase)
            .Select(g => new AssetSummary(
                g.Key,
                g.First().AssetKind ?? "?",
                g.Count(),
                g.Select(static e => e.Pm4File).Distinct(StringComparer.OrdinalIgnoreCase).Count(),
                g.Min(static e => e.SurfaceCount),
                g.Max(static e => e.SurfaceCount),
                g.Average(static e => (double)e.SurfaceCount)))
            .OrderByDescending(static a => a.InstanceCount)
            .ThenBy(static a => a.AssetPath, StringComparer.OrdinalIgnoreCase)
            .ToList();

        IReadOnlyList<Pm4ValueFrequency> statusCounts = status
            .OrderByDescending(static kv => kv.Value)
            .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
            .ToList();

        return new LibraryReport(
            resolved, adtRoot, filesScanned, filesPaired,
            entries.Count, statusCounts, assets.Count, assets, entries);
    }
}
