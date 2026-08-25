using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Tests whether <c>MPRL</c> points are per-object ANCHORS - the placement each asset hangs from -
/// and whether they carry the same float <c>MSUR._0x1C</c> does.
/// </summary>
/// <remarks>
/// Two things must hold if MPRL anchors objects. Its values should collide with the <c>_0x1C</c> set,
/// since that field is bit-identical to <c>MODF.Position.Z</c>; and its points should land on MODF
/// positions rather than merely inside the tile.
///
/// <para>MPRL is the one stream in this format whose axis order differs from the rest, and
/// <c>MprlToAdtPlacement</c> currently returns it unchanged - an assumption, not a measurement. So no
/// axis pairing is assumed here. The float test compares <b>every MPRL component against the
/// <c>_0x1C</c> set bit-exactly</b>, which cannot be fooled by a permutation because it does not care
/// which component matched. The position test reports all three components against all three MODF
/// components, so the permutation falls out of the table.</para>
///
/// <para><b>Controls.</b> The float test is repeated against a DIFFERENT file's <c>_0x1C</c> set, and
/// the position test against a shuffled placement row. Floats collide by chance more often than
/// intuition suggests once a corpus is large, and every point in a tile is within 533 units of every
/// other, so both need a floor to beat.</para>
/// </remarks>
internal static class Pm4MprlAnchorSupport
{
    public static Pm4MprlAnchorReport Analyze(string pm4Directory, string? adtDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);
        string adtRoot = string.IsNullOrWhiteSpace(adtDirectory) ? resolved : adtDirectory;

        int files = 0, filesWithModf = 0;
        long mprlTotal = 0, objectsTotal = 0, modfTotal = 0;
        long countMatchesObjects = 0, countMatchesModf = 0;

        // Bit-exact float collision, per component, raw and origin-flipped, plus a wrong-file control.
        var hits = new long[3];
        var hitsFlipped = new long[3];
        var control = new long[3];
        long floatTested = 0;

        // Nearest MODF position, per MPRL component pairing.
        var near = new Stat[3, 3];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
                near[i, j] = new Stat();
        }

        var nearBest = new Stat();
        var nearControl = new Stat();

        List<uint>? previousZeroSet = null;
        var rng = new Random(4242);

        // Terrain-contact test. If MPRL marks where a building meets the ground, its height must land
        // inside the terrain height range of the cell it sits over. Containment against the chunk
        // [min,max] of MCVT is used rather than a point height, because that needs no assumption about
        // vertex order INSIDE a chunk - only the cell binning, which is separately verified at 100%.
        // All six component permutations are tried, since MPRL is the one permuted stream.
        var contact = new long[6];
        var contactControl = new long[6];
        var contactMiss = new Stat[6];
        for (int i = 0; i < 6; i++)
            contactMiss[i] = new Stat();
        long contactTested = 0;

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Mprl.Count == 0 || chunks.Msur.Count == 0)
                continue;

            files++;
            mprlTotal += chunks.Mprl.Count;

            var packed = new HashSet<uint>();
            foreach (Pm4MsurEntry surface in chunks.Msur)
            {
                if (surface.PackedParams != 0)
                    packed.Add(surface.PackedParams);
            }

            objectsTotal += packed.Count;
            if (chunks.Mprl.Count == packed.Count)
                countMatchesObjects++;

            HashSet<uint> controlSet = previousZeroSet is null ? [] : [.. previousZeroSet];
            previousZeroSet = [.. packed];

            foreach (Pm4MprlEntry entry in chunks.Mprl)
            {
                floatTested++;
                Vector3 p = entry.Position;
                Span<float> comps = [p.X, p.Y, p.Z];

                for (int c = 0; c < 3; c++)
                {
                    uint raw = BitConverter.SingleToUInt32Bits(comps[c]);
                    uint flipped = BitConverter.SingleToUInt32Bits(Pm4CoordinateService.MapOrigin - comps[c]);

                    if (packed.Contains(raw)) hits[c]++;
                    if (packed.Contains(flipped)) hitsFlipped[c]++;
                    if (controlSet.Contains(raw)) control[c]++;
                }
            }

            if (Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tFirst, out int tSecond))
            {
                (float Min, float Max) bX = Pm4CoordinateService.GetPlacementTileBand(tSecond);
                (float Min, float Max) bY = Pm4CoordinateService.GetPlacementTileBand(tFirst);
                string? terrainAdt = FindTerrainAdt(pm4Path, adtRoot, tFirst, tSecond);
                TerrainCell[,]? terrain = terrainAdt is null ? null : ReadTerrain(terrainAdt, bX, bY);

                if (terrain is not null)
                {
                    foreach (Pm4MprlEntry entry in chunks.Mprl)
                    {
                        Vector3 mp = entry.Position;
                        Span<float> mc = [mp.X, mp.Y, mp.Z];
                        contactTested++;

                        for (int perm = 0; perm < 6; perm++)
                        {
                            (int ax, int ay, int az) = Permutation(perm);
                            float wx = Pm4CoordinateService.MapOrigin - mc[ax];
                            float wy = Pm4CoordinateService.MapOrigin - mc[ay];
                            float h = mc[az];

                            int col = (int)MathF.Floor((wx - bX.Min) / ChunkSize);
                            int row = (int)MathF.Floor((wy - bY.Min) / ChunkSize);
                            if (col is < 0 or > 15 || row is < 0 or > 15)
                                continue;

                            TerrainCell cell = terrain[col, row];
                            if (!cell.Present)
                                continue;

                            if (h >= cell.MinHeight && h <= cell.MaxHeight)
                                contact[perm]++;

                            float miss = h < cell.MinHeight ? cell.MinHeight - h
                                : h > cell.MaxHeight ? h - cell.MaxHeight : 0f;
                            contactMiss[perm].Add(miss);

                            TerrainCell other = terrain[rng.Next(16), rng.Next(16)];
                            if (other.Present && h >= other.MinHeight && h <= other.MaxHeight)
                                contactControl[perm]++;
                        }
                    }
                }
            }

            string? adtPath = Pm4PlacementZSupport.FindCompanionAdt(pm4Path, adtRoot);
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

            IReadOnlyList<AdtWorldModelPlacement> modf = catalog.WorldModelPlacements;
            if (modf.Count == 0)
                continue;

            filesWithModf++;
            modfTotal += modf.Count;
            if (chunks.Mprl.Count == modf.Count)
                countMatchesModf++;

            foreach (Pm4MprlEntry entry in chunks.Mprl)
            {
                Vector3 p = entry.Position;
                Span<float> comps = [p.X, p.Y, p.Z];

                for (int c = 0; c < 3; c++)
                {
                    for (int m = 0; m < 3; m++)
                    {
                        float bestOne = float.MaxValue;
                        foreach (AdtWorldModelPlacement row in modf)
                        {
                            float target = m == 0 ? row.Position.X : m == 1 ? row.Position.Y : row.Position.Z;
                            float d = MathF.Abs(comps[c] - target);
                            if (d < bestOne)
                                bestOne = d;
                        }

                        near[c, m].Add(bestOne);
                    }
                }

                // Whole-point distance under the placement transform the other streams use.
                Vector3 asPlacement = Pm4CoordinateService.Pm4LocalToAdtPlacement(p);
                float best = float.MaxValue;
                foreach (AdtWorldModelPlacement row in modf)
                {
                    float d = Vector3.Distance(asPlacement, row.Position);
                    if (d < best)
                        best = d;
                }

                nearBest.Add(best);
                nearControl.Add(Vector3.Distance(asPlacement, modf[rng.Next(modf.Count)].Position));
            }
        }

        var pairings = new List<Pm4ErrorStat>();
        string[] cn = ["MPRL.X", "MPRL.Y", "MPRL.Z"];
        string[] mn = ["MODF.X", "MODF.Y", "MODF.Z"];
        for (int c = 0; c < 3; c++)
        {
            for (int m = 0; m < 3; m++)
                pairings.Add(near[c, m].ToResult($"{cn[c]} -> nearest {mn[m]}"));
        }

        return new Pm4MprlAnchorReport(
            resolved, files, filesWithModf, mprlTotal, objectsTotal, modfTotal,
            files == 0 ? 0 : (double)countMatchesObjects / files,
            filesWithModf == 0 ? 0 : (double)countMatchesModf / filesWithModf,
            floatTested,
            [.. Enumerable.Range(0, 3).Select(c => new Pm4ValueFrequency($"{cn[c]} raw", (int)hits[c]))],
            [.. Enumerable.Range(0, 3).Select(c => new Pm4ValueFrequency($"{cn[c]} origin-flipped", (int)hitsFlipped[c]))],
            [.. Enumerable.Range(0, 3).Select(c => new Pm4ValueFrequency($"{cn[c]} CONTROL wrong file", (int)control[c]))],
            [.. pairings.OrderBy(static x => x.MedianAbsError)],
            nearBest.ToResult("MPRL point -> nearest MODF position"),
            nearControl.ToResult("CONTROL MPRL point -> random MODF position"),
            contactTested,
            [.. Enumerable.Range(0, 6).Select(i => new Pm4ValueFrequency(PermName(i), (int)contact[i]))],
            [.. Enumerable.Range(0, 6).Select(i => new Pm4ValueFrequency(PermName(i) + " CONTROL", (int)contactControl[i]))],
            [.. Enumerable.Range(0, 6).Select(i => contactMiss[i].ToResult(PermName(i) + " miss"))]);
    }

    private const float ChunkSize = Pm4CoordinateService.TileSize / 16f;

    private readonly record struct TerrainCell(bool Present, float MinHeight, float MaxHeight);

    private static (int X, int Y, int Z) Permutation(int i) => i switch
    {
        0 => (0, 1, 2),
        1 => (0, 2, 1),
        2 => (1, 0, 2),
        3 => (1, 2, 0),
        4 => (2, 0, 1),
        _ => (2, 1, 0),
    };

    private static string PermName(int i)
    {
        (int x, int y, int z) = Permutation(i);
        string[] n = ["X", "Y", "Z"];
        return $"xy={n[x]}{n[y]} h={n[z]}";
    }

    private static string? FindTerrainAdt(string pm4Path, string adtDirectory, int tileFirst, int tileSecond)
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
                return candidate;
        }

        return null;
    }

    /// <summary>
    /// Per-cell terrain height RANGE, from each chunk stated base plus the spread of its MCVT heights.
    /// A range rather than a point, so the test needs no assumption about vertex order within a chunk.
    /// </summary>
    private static TerrainCell[,]? ReadTerrain(string adtPath, (float Min, float Max) bandX, (float Min, float Max) bandY)
    {
        byte[] data;
        try
        {
            data = File.ReadAllBytes(adtPath);
        }
        catch
        {
            return null;
        }

        var grid = new TerrainCell[16, 16];
        int placed = 0, offset = 0;

        while (offset + 8 <= data.Length)
        {
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, 4));
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 4, 4));
            int payload = offset + 8;
            if (size > int.MaxValue || payload + (long)size > data.Length)
                break;

            if (magic == 0x4D434E4Bu && size >= 0x74)
            {
                float a = BitConverter.ToSingle(data, payload + 0x68);
                float b = BitConverter.ToSingle(data, payload + 0x6C);
                float baseZ = BitConverter.ToSingle(data, payload + 0x70);

                // Stated position is the chunk max corner; half a chunk back lands inside the cell.
                int col = (int)MathF.Floor((a - (ChunkSize * 0.5f) - bandX.Min) / ChunkSize);
                int row = (int)MathF.Floor((b - (ChunkSize * 0.5f) - bandY.Min) / ChunkSize);
                if (col is >= 0 and <= 15 && row is >= 0 and <= 15
                    && TryMcvtRange(data, payload, (int)size, out float lo, out float hi))
                {
                    grid[col, row] = new TerrainCell(true, baseZ + lo, baseZ + hi);
                    placed++;
                }
            }

            offset = payload + (int)size;
        }

        return placed < 64 ? null : grid;
    }

    private static bool TryMcvtRange(byte[] data, int payload, int size, out float lo, out float hi)
    {
        lo = 0;
        hi = 0;
        int end = payload + size;
        for (int p = payload; p + 8 + (145 * 4) <= end; p += 4)
        {
            if (BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(p, 4)) != 0x4D435654u)
                continue;

            lo = float.MaxValue;
            hi = float.MinValue;
            for (int i = 0; i < 145; i++)
            {
                float v = BitConverter.ToSingle(data, p + 8 + (i * 4));
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }

            return true;
        }

        return false;
    }

    private sealed class Stat

    {
        private readonly List<double> _v = [];

        public void Add(double d) => _v.Add(d);

        public Pm4ErrorStat ToResult(string name)
        {
            if (_v.Count == 0)
                return new Pm4ErrorStat(name, 0, 0, 0, 0, 0);

            _v.Sort();
            return new Pm4ErrorStat(
                name, _v.Count, _v[_v.Count / 2], _v[(int)(_v.Count * 0.90)], _v[^1],
                (double)_v.Count(static x => x < 1.0) / _v.Count);
        }
    }
}

internal sealed record Pm4MprlAnchorReport(
    string Pm4Directory,
    int Files,
    int FilesWithModf,
    long MprlTotal,
    long ObjectsTotal,
    long ModfTotal,
    double FilesWhereMprlCountEqualsObjectCount,
    double FilesWhereMprlCountEqualsModfCount,
    long FloatComparisons,
    IReadOnlyList<Pm4ValueFrequency> RawHits,
    IReadOnlyList<Pm4ValueFrequency> FlippedHits,
    IReadOnlyList<Pm4ValueFrequency> ControlHits,
    IReadOnlyList<Pm4ErrorStat> ComponentPairings,
    Pm4ErrorStat NearestModf,
    Pm4ErrorStat NearestControl,
    long ContactTested,
    IReadOnlyList<Pm4ValueFrequency> ContactHits,
    IReadOnlyList<Pm4ValueFrequency> ContactControl,
    IReadOnlyList<Pm4ErrorStat> ContactMiss);
