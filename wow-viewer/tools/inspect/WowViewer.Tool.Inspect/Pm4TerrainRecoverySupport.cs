using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Measures how well terrain height can be recovered from a PM4 where no ADT mesh survives.
/// </summary>
/// <remarks>
/// Two sources of height samples come out of a PM4. <c>MPRL</c> marks where a placed model meets the
/// ground, so each point is a terrain height at a known position; and every object carries its
/// placement Z in <c>MSUR._0x1C</c>, which for a model standing on the ground is the terrain height
/// under it. This scores both against tiles that still HAVE terrain, which is what licenses using them
/// on tiles that do not.
///
/// <para>Scoring needs a height at a point, not the chunk-wide range used earlier, so the 9x9 outer
/// <c>MCVT</c> grid is assembled into a 129x129 tile heightmap. That requires knowing the vertex order
/// WITHIN a chunk, which is resolved <b>from the ADT alone</b>: neighbouring chunks share an edge, so
/// the right ordering is the one where chunks agree with each other where they overlap. Nothing about
/// PM4 enters that decision, so using it afterwards to judge PM4 is not circular.</para>
///
/// <para>Coverage matters as much as accuracy. A height source that is beautifully accurate across 4%
/// of a tile does not reconstruct terrain, so the fraction of tile cells holding at least one sample
/// is reported next to the error.</para>
/// </remarks>
internal static class Pm4TerrainRecoverySupport
{
    private const float ChunkSize = Pm4CoordinateService.TileSize / 16f;
    private const int Grid = 129;
    private const float SampleSpacing = Pm4CoordinateService.TileSize / (Grid - 1);

    public static Pm4TerrainRecoveryReport Analyze(string pm4Directory, string adtDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);

        // Step 1: resolve the within-chunk vertex order from ADT self-consistency.
        var disagreement = new double[8];
        var disagreementCount = new long[8];

        foreach (string adtPath in Directory
            .EnumerateFiles(adtDirectory, "*.adt", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Take(40))
        {
            List<RawChunk> chunks = ReadChunks(adtPath);
            if (chunks.Count < 64)
                continue;

            for (int o = 0; o < 8; o++)
            {
                (double sum, long n) = OverlapDisagreement(chunks, o);
                disagreement[o] += sum;
                disagreementCount[o] += n;
            }
        }

        var orderScores = new List<Pm4ValueFrequency>();
        int bestOrder = 0;
        double bestRms = double.MaxValue, runnerUpRms = double.MaxValue;
        for (int o = 0; o < 8; o++)
        {
            double rms = disagreementCount[o] == 0 ? double.MaxValue : Math.Sqrt(disagreement[o] / disagreementCount[o]);
            orderScores.Add(new Pm4ValueFrequency($"{OrderName(o)} rms={rms:F4}", (int)disagreementCount[o]));
            if (rms < bestRms)
            {
                runnerUpRms = bestRms;
                bestRms = rms;
                bestOrder = o;
            }
            else if (rms < runnerUpRms)
            {
                runnerUpRms = rms;
            }
        }

        bool orderResolved = bestRms < 0.01 && runnerUpRms > bestRms * 10;

        // Step 2 and 3: score PM4 height sources against the assembled heightmap.
        var mprlError = new Stat();
        var objectError = new Stat();
        var controlError = new Stat();

        long tiles = 0, cellsTotal = 0, cellsWithMprl = 0, cellsWithObject = 0;
        long mprlPoints = 0, objectPoints = 0;
        var rng = new Random(99);

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tFirst, out int tSecond))
                continue;

            string? adtPath = FindTerrainAdt(pm4Path, adtDirectory, tFirst, tSecond);
            if (adtPath is null)
                continue;

            (float Min, float Max) bandX = Pm4CoordinateService.GetPlacementTileBand(tSecond);
            (float Min, float Max) bandY = Pm4CoordinateService.GetPlacementTileBand(tFirst);

            List<RawChunk> chunks = ReadChunks(adtPath);
            if (chunks.Count < 64)
                continue;

            float[,]? height = Assemble(chunks, bandX, bandY, bestOrder);
            if (height is null)
                continue;

            Pm4KnownChunkSet pm4 = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (pm4.Mprl.Count == 0)
                continue;

            tiles++;

            var mprlCells = new bool[16, 16];
            var objectCells = new bool[16, 16];

            foreach (Pm4MprlEntry entry in pm4.Mprl)
            {
                Vector3 w = Pm4CoordinateService.MprlToAdtPlacement(entry.Position);
                if (!TrySample(height, bandX, bandY, w.X, w.Y, out float terrain))
                    continue;

                mprlPoints++;
                mprlError.Add(Math.Abs(w.Z - terrain));

                int col = (int)MathF.Floor((w.X - bandX.Min) / ChunkSize);
                int row = (int)MathF.Floor((w.Y - bandY.Min) / ChunkSize);
                if (col is >= 0 and <= 15 && row is >= 0 and <= 15)
                    mprlCells[col, row] = true;

                // Control: the same terrain lookup at an unrelated spot in the tile.
                float cx = bandX.Min + ((bandX.Max - bandX.Min) * (float)rng.NextDouble());
                float cy = bandY.Min + ((bandY.Max - bandY.Min) * (float)rng.NextDouble());
                if (TrySample(height, bandX, bandY, cx, cy, out float ctl))
                    controlError.Add(Math.Abs(w.Z - ctl));
            }

            // Objects: placement Z against the terrain under the object footprint centre.
            var lo = new Dictionary<uint, Vector3>();
            var hi = new Dictionary<uint, Vector3>();
            foreach (Pm4MsurEntry surface in pm4.Msur)
            {
                if (surface.PackedParams == 0)
                    continue;

                long vs = surface.MsviFirstIndex;
                long ve = vs + surface.IndexCount;
                if (surface.IndexCount == 0 || ve > pm4.Msvi.Count)
                    continue;

                for (long v = vs; v < ve; v++)
                {
                    uint vi = pm4.Msvi[(int)v];
                    if (vi >= pm4.Msvt.Count)
                        continue;

                    Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(pm4.Msvt[(int)vi]);
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

            foreach ((uint key, Vector3 min) in lo)
            {
                Vector3 max = hi[key];
                float cx = (min.X + max.X) * 0.5f;
                float cy = (min.Y + max.Y) * 0.5f;
                if (!TrySample(height, bandX, bandY, cx, cy, out float terrain))
                    continue;

                objectPoints++;
                objectError.Add(Math.Abs(BitConverter.UInt32BitsToSingle(key) - terrain));

                int col = (int)MathF.Floor((cx - bandX.Min) / ChunkSize);
                int row = (int)MathF.Floor((cy - bandY.Min) / ChunkSize);
                if (col is >= 0 and <= 15 && row is >= 0 and <= 15)
                    objectCells[col, row] = true;
            }

            for (int c = 0; c < 16; c++)
            {
                for (int r = 0; r < 16; r++)
                {
                    cellsTotal++;
                    if (mprlCells[c, r]) cellsWithMprl++;
                    if (objectCells[c, r]) cellsWithObject++;
                }
            }
        }

        return new Pm4TerrainRecoveryReport(
            resolved, adtDirectory, tiles, orderScores, OrderName(bestOrder), orderResolved,
            mprlPoints, objectPoints,
            cellsTotal == 0 ? 0 : (double)cellsWithMprl / cellsTotal,
            cellsTotal == 0 ? 0 : (double)cellsWithObject / cellsTotal,
            mprlError.ToResult("MPRL height vs real terrain"),
            objectError.ToResult("object placement Z vs real terrain"),
            controlError.ToResult("CONTROL MPRL height vs terrain elsewhere"));
    }

    private readonly record struct RawChunk(int Col, int Row, float BaseZ, float[] Heights);

    private static string OrderName(int o) =>
        $"{((o & 4) != 0 ? "transpose" : "direct")}{((o & 2) != 0 ? " flipI" : "")}{((o & 1) != 0 ? " flipJ" : "")}";

    /// <summary>
    /// Maps an outer-grid vertex to its offset within the chunk under one candidate ordering.
    /// </summary>
    private static (int U, int V) MapVertex(int i, int j, int order)
    {
        int a = (order & 2) != 0 ? 8 - i : i;
        int b = (order & 1) != 0 ? 8 - j : j;
        return (order & 4) != 0 ? (b, a) : (a, b);
    }

    /// <summary>
    /// Total squared disagreement where neighbouring chunks write the same tile-grid sample. Chunks
    /// share their edge vertices, so the correct ordering makes those writes agree exactly.
    /// </summary>
    private static (double Sum, long Count) OverlapDisagreement(List<RawChunk> chunks, int order)
    {
        var value = new float[Grid, Grid];
        var written = new bool[Grid, Grid];
        double sum = 0;
        long count = 0;

        foreach (RawChunk chunk in chunks)
        {
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    (int u, int v) = MapVertex(i, j, order);
                    int gx = (chunk.Col * 8) + u;
                    int gy = (chunk.Row * 8) + v;
                    if (gx is < 0 or >= Grid || gy is < 0 or >= Grid)
                        continue;

                    float h = chunk.BaseZ + chunk.Heights[(i * 17) + j];
                    if (written[gx, gy])
                    {
                        double d = value[gx, gy] - h;
                        sum += d * d;
                        count++;
                    }
                    else
                    {
                        value[gx, gy] = h;
                        written[gx, gy] = true;
                    }
                }
            }
        }

        return (sum, count);
    }

    private static float[,]? Assemble(List<RawChunk> chunks, (float Min, float Max) bandX, (float Min, float Max) bandY, int order)
    {
        var value = new float[Grid, Grid];
        var written = new bool[Grid, Grid];
        int filled = 0;

        foreach (RawChunk chunk in chunks)
        {
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    (int u, int v) = MapVertex(i, j, order);
                    int gx = (chunk.Col * 8) + u;
                    int gy = (chunk.Row * 8) + v;
                    if (gx is < 0 or >= Grid || gy is < 0 or >= Grid)
                        continue;

                    if (!written[gx, gy])
                    {
                        value[gx, gy] = chunk.BaseZ + chunk.Heights[(i * 17) + j];
                        written[gx, gy] = true;
                        filled++;
                    }
                }
            }
        }

        return filled < Grid * Grid / 2 ? null : value;
    }

    private static bool TrySample(float[,] height, (float Min, float Max) bandX, (float Min, float Max) bandY, float wx, float wy, out float result)
    {
        result = 0;
        float fx = (wx - bandX.Min) / SampleSpacing;
        float fy = (wy - bandY.Min) / SampleSpacing;
        if (fx < 0 || fy < 0 || fx > Grid - 1 || fy > Grid - 1)
            return false;

        int x0 = (int)MathF.Floor(fx), y0 = (int)MathF.Floor(fy);
        int x1 = Math.Min(x0 + 1, Grid - 1), y1 = Math.Min(y0 + 1, Grid - 1);
        float tx = fx - x0, ty = fy - y0;

        float a = (height[x0, y0] * (1 - tx)) + (height[x1, y0] * tx);
        float b = (height[x0, y1] * (1 - tx)) + (height[x1, y1] * tx);
        result = (a * (1 - ty)) + (b * ty);
        return true;
    }

    private static List<RawChunk> ReadChunks(string adtPath)
    {
        var found = new List<RawChunk>();
        byte[] data;
        try
        {
            data = File.ReadAllBytes(adtPath);
        }
        catch
        {
            return found;
        }

        int offset = 0;
        while (offset + 8 <= data.Length)
        {
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, 4));
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 4, 4));
            int payload = offset + 8;
            if (size > int.MaxValue || payload + (long)size > data.Length)
                break;

            if (magic == 0x4D434E4Bu && size >= 0x74)
            {
                uint ix = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(payload + 0x04, 4));
                uint iy = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(payload + 0x08, 4));
                float baseZ = BitConverter.ToSingle(data, payload + 0x70);
                if (ix < 16 && iy < 16 && TryMcvt(data, payload, (int)size, out float[] heights))
                {
                    // col/row from the header indices, using the mapping the cell frame test measured
                    // at 100%: col == 15 - IndexY and row == 15 - IndexX.
                    found.Add(new RawChunk(15 - (int)iy, 15 - (int)ix, baseZ, heights));
                }
            }

            offset = payload + (int)size;
        }

        return found;
    }

    private static bool TryMcvt(byte[] data, int payload, int size, out float[] heights)
    {
        heights = [];
        int end = payload + size;
        for (int p = payload; p + 8 + (145 * 4) <= end; p += 4)
        {
            if (BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(p, 4)) != 0x4D435654u)
                continue;

            var h = new float[145];
            for (int i = 0; i < 145; i++)
                h[i] = BitConverter.ToSingle(data, p + 8 + (i * 4));
            heights = h;
            return true;
        }

        return false;
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

internal sealed record Pm4TerrainRecoveryReport(
    string Pm4Directory,
    string AdtDirectory,
    long Tiles,
    IReadOnlyList<Pm4ValueFrequency> VertexOrderScores,
    string ResolvedOrder,
    bool OrderResolved,
    long MprlSamples,
    long ObjectSamples,
    double CellsWithMprlFraction,
    double CellsWithObjectFraction,
    Pm4ErrorStat MprlError,
    Pm4ErrorStat ObjectError,
    Pm4ErrorStat ControlError);
