using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Tests the stated reason the <c>_0x1C == 0</c> bucket is vertically stretched: that objects standing
/// over water, ice or an open pit get collision run from the top of the object down to whatever is
/// beneath, so a player cannot get behind or under them.
/// </summary>
/// <remarks>
/// The prediction is spatial - stretched surfaces should sit over liquid or over terrain holes - so it
/// needs PM4 surfaces and ADT terrain cells in the same frame. Binning a surface into the wrong
/// sixteenth of a tile produces a confident number that means nothing.
///
/// <para>An earlier version of this test inferred the cell mapping by correlating PM4 surface height
/// against terrain height across all eight candidate orientations. That was the wrong instrument: the
/// right orientation won all three times it was run, but never by more than r = 0.14, because doodad
/// collision height is a poor proxy for ground height. It is not needed at all - <b>every MCNK states
/// its own world position</b>, so its cell is computed with the same band arithmetic used for the PM4
/// surfaces and the two land in one frame by construction. The relationship to the header's
/// <c>IndexX</c>/<c>IndexY</c> is then an OUTPUT of the test rather than an assumption behind it.</para>
///
/// <para>The liquid comparison is tall-vs-short <i>within the same tiles</i>, which controls for the
/// obvious confound: a tile that is mostly lake puts everything over water.</para>
/// </remarks>
internal static class Pm4StretchLocalitySupport
{
    private const float ChunkSize = Pm4CoordinateService.TileSize / 16f;
    private const uint LiquidBits = 0x04u | 0x08u | 0x10u | 0x20u;

    public static Pm4StretchLocalityReport Analyze(string pm4Directory, string adtDirectory, float tallThreshold = 5f)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(pm4Directory);

        int filesPaired = 0, filesSkippedNoAdt = 0, filesSkippedNoGrid = 0;
        long tallTotal = 0, tallLiquid = 0, tallHole = 0;
        long shortTotal = 0, shortLiquid = 0, shortHole = 0;
        long cellsPresent = 0, cellsLiquid = 0, cellsHole = 0;
        long indexChecked = 0, indexColIsFlippedIy = 0, indexRowIsFlippedIx = 0;
        long surfacesOffGrid = 0, surfacesOnMissingCell = 0;

        foreach (string pm4Path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileFirst, out int tileSecond))
                continue;

            string? adtPath = FindTerrainAdt(pm4Path, adtDirectory, tileFirst, tileSecond);
            if (adtPath is null)
            {
                filesSkippedNoAdt++;
                continue;
            }

            // Placement .X is bounded by the filename's SECOND number and .Y by the FIRST - the measured
            // pairing IsWithinPlacementTileBounds uses. Taking them in the obvious order instead puts
            // every surface outside its own tile, which reads as an empty file rather than as an error.
            (float Min, float Max) bandX = Pm4CoordinateService.GetPlacementTileBand(tileSecond);
            (float Min, float Max) bandY = Pm4CoordinateService.GetPlacementTileBand(tileFirst);

            McnkCell[,]? grid = ReadMcnkGrid(
                adtPath, bandX, bandY,
                ref indexChecked, ref indexColIsFlippedIy, ref indexRowIsFlippedIx);
            if (grid is null)
            {
                filesSkippedNoGrid++;
                continue;
            }

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(pm4Path).KnownChunks;
            if (chunks.Msur.Count == 0)
                continue;

            filesPaired++;

            for (int col = 0; col < 16; col++)
            {
                for (int row = 0; row < 16; row++)
                {
                    if (!grid[col, row].Present)
                        continue;

                    cellsPresent++;
                    if (grid[col, row].HasLiquid) cellsLiquid++;
                    if (grid[col, row].HasHole) cellsHole++;
                }
            }

            foreach (Pm4MsurEntry surface in chunks.Msur)
            {
                if (surface.PackedParams != 0)
                    continue; // the stretched bucket only

                long vs = surface.MsviFirstIndex;
                long ve = vs + surface.IndexCount;
                if (surface.IndexCount == 0 || ve > chunks.Msvi.Count)
                    continue;

                float minZ = float.MaxValue, maxZ = float.MinValue;
                double sx = 0, sy = 0;
                int n = 0;
                for (long v = vs; v < ve; v++)
                {
                    uint vi = chunks.Msvi[(int)v];
                    if (vi >= chunks.Msvt.Count)
                        continue;

                    Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(chunks.Msvt[(int)vi]);
                    sx += p.X;
                    sy += p.Y;
                    n++;
                    if (p.Z < minZ) minZ = p.Z;
                    if (p.Z > maxZ) maxZ = p.Z;
                }

                if (n == 0)
                    continue;

                int scol = (int)MathF.Floor((float)((sx / n) - bandX.Min) / ChunkSize);
                int srow = (int)MathF.Floor((float)((sy / n) - bandY.Min) / ChunkSize);
                if (scol is < 0 or > 15 || srow is < 0 or > 15)
                {
                    surfacesOffGrid++;
                    continue;
                }

                McnkCell cell = grid[scol, srow];
                if (!cell.Present)
                {
                    surfacesOnMissingCell++;
                    continue;
                }

                if (maxZ - minZ > tallThreshold)
                {
                    tallTotal++;
                    if (cell.HasLiquid) tallLiquid++;
                    if (cell.HasHole) tallHole++;
                }
                else
                {
                    shortTotal++;
                    if (cell.HasLiquid) shortLiquid++;
                    if (cell.HasHole) shortHole++;
                }
            }
        }

        return new Pm4StretchLocalityReport(
            resolvedDirectory, adtDirectory, filesPaired, filesSkippedNoAdt, filesSkippedNoGrid,
            tallThreshold,
            indexChecked,
            indexChecked == 0 ? 0 : (double)indexColIsFlippedIy / indexChecked,
            indexChecked == 0 ? 0 : (double)indexRowIsFlippedIx / indexChecked,
            surfacesOffGrid, surfacesOnMissingCell,
            tallTotal, shortTotal,
            tallTotal == 0 ? 0 : (double)tallLiquid / tallTotal,
            shortTotal == 0 ? 0 : (double)shortLiquid / shortTotal,
            tallTotal == 0 ? 0 : (double)tallHole / tallTotal,
            shortTotal == 0 ? 0 : (double)shortHole / shortTotal,
            cellsPresent == 0 ? 0 : (double)cellsLiquid / cellsPresent,
            cellsPresent == 0 ? 0 : (double)cellsHole / cellsPresent);
    }

    private readonly record struct McnkCell(bool Present, bool HasLiquid, bool HasHole);

    /// <summary>
    /// Reads an ADT's terrain cells, keyed by the same tile-relative cell coordinates the PM4 surfaces
    /// are binned into, taken from each chunk's own stated world position.
    /// </summary>
    /// <remarks>
    /// Liquid is read from <c>MH2O</c> as well as the <c>MCNK</c> flag bits. In this corpus the flag
    /// bits are clear even on tiles that plainly have water, so trusting them alone would report that
    /// nothing is over liquid - a null result produced by reading the wrong field. The <c>MH2O</c>
    /// header is 256 records of (offsetInstances, layerCount, offsetAttributes) in MCNK order, so its
    /// i-th record belongs to the i-th chunk encountered.
    /// </remarks>
    private static McnkCell[,]? ReadMcnkGrid(
        string adtPath,
        (float Min, float Max) bandX,
        (float Min, float Max) bandY,
        ref long indexChecked,
        ref long indexColIsFlippedIy,
        ref long indexRowIsFlippedIx)
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

        var found = new List<(int Ix, int Iy, float A, float B, uint Flags, ushort Holes)>();
        bool[]? mh2oLayered = null;

        int offset = 0;
        while (offset + 8 <= data.Length)
        {
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, 4));
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 4, 4));
            int payload = offset + 8;
            if (size > int.MaxValue || payload + (long)size > data.Length)
                break;

            // 'MH2O' reversed reads back as 0x4D48324F.
            if (magic == 0x4D48324Fu && size >= 256 * 12)
            {
                mh2oLayered = new bool[256];
                for (int i = 0; i < 256; i++)
                    mh2oLayered[i] = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(payload + (i * 12) + 4, 4)) > 0;
            }

            // 'MCNK' reversed.
            if (magic == 0x4D434E4Bu && size >= 0x74)
            {
                uint flags = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(payload + 0x00, 4));
                uint ix = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(payload + 0x04, 4));
                uint iy = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(payload + 0x08, 4));
                ushort holes = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(payload + 0x3C, 2));
                float a = BitConverter.ToSingle(data, payload + 0x68);
                float b = BitConverter.ToSingle(data, payload + 0x6C);
                if (ix < 16 && iy < 16)
                    found.Add(((int)ix, (int)iy, a, b, flags, holes));
            }

            offset = payload + (int)size;
        }

        if (found.Count < 64)
            return null;

        var grid = new McnkCell[16, 16];
        int placed = 0;

        for (int i = 0; i < found.Count; i++)
        {
            (int ix, int iy, float a, float b, uint flags, ushort holes) = found[i];

            // The chunk position's first component shares an axis with placement .X and the second with
            // placement .Y, so both sides of the comparison are binned by the same arithmetic. The
            // stated position is the chunk's MAXIMUM corner, not its middle: binning it directly puts
            // every chunk one cell high and pushes the first row clean off the grid, so half a chunk is
            // taken off to get a point that is unambiguously inside the cell. PM4 surfaces need no such
            // correction because they are already centroids.
            int col = (int)MathF.Floor((a - (ChunkSize * 0.5f) - bandX.Min) / ChunkSize);
            int row = (int)MathF.Floor((b - (ChunkSize * 0.5f) - bandY.Min) / ChunkSize);
            if (col is < 0 or > 15 || row is < 0 or > 15)
                continue;

            indexChecked++;
            if (col == 15 - iy) indexColIsFlippedIy++;
            if (row == 15 - ix) indexRowIsFlippedIx++;

            bool liquid = (flags & LiquidBits) != 0
                || (mh2oLayered is not null && i < mh2oLayered.Length && mh2oLayered[i]);
            grid[col, row] = new McnkCell(true, liquid, holes != 0);
            placed++;
        }

        return placed < 64 ? null : grid;
    }

    /// <summary>
    /// Finds the ADT that actually carries terrain for a PM4 tile.
    /// </summary>
    /// <remarks>
    /// Not the same job as finding a tile's placements. This corpus is mixed: some tiles are monolithic
    /// and some are split, and for the split ones the root file is present but ZERO BYTES while
    /// <c>_obj0</c> and <c>_tex0</c> hold the content. <c>MCNK</c> terrain lives only in a monolithic
    /// file or a non-empty root - an <c>_obj0</c> has none at all - so reusing the placement resolver
    /// here silently pairs every split tile with a file containing no terrain. Names are tried both
    /// zero-padded and not, since PM4 pads its tile numbers and ADT does not.
    /// </remarks>
    private static string? FindTerrainAdt(string pm4Path, string adtDirectory, int tileFirst, int tileSecond)
    {
        string stem = Path.GetFileNameWithoutExtension(pm4Path);
        int underscore = stem.IndexOf('_');
        string mapName = underscore > 0 ? stem[..underscore] : stem;

        foreach (string candidate in new[]
        {
            Path.Combine(adtDirectory, $"{stem}.adt"),
            Path.Combine(adtDirectory, $"{mapName}_{tileFirst}_{tileSecond}.adt"),
            Path.Combine(adtDirectory, $"{mapName}_{tileFirst:00}_{tileSecond:00}.adt"),
        })
        {
            if (File.Exists(candidate) && new FileInfo(candidate).Length > 1024)
                return candidate;
        }

        return null;
    }
}

internal sealed record Pm4StretchLocalityReport(
    string Pm4Directory,
    string AdtDirectory,
    int FilesPaired,
    int FilesSkippedNoAdt,
    int FilesSkippedNoGrid,
    float TallThreshold,
    long ChunksIndexChecked,
    double ColEqualsFlippedIndexYFraction,
    double RowEqualsFlippedIndexXFraction,
    long SurfacesOffGrid,
    long SurfacesOnMissingCell,
    long TallSurfaces,
    long ShortSurfaces,
    double TallOverLiquidFraction,
    double ShortOverLiquidFraction,
    double TallOverHoleFraction,
    double ShortOverHoleFraction,
    double BaseLiquidCellFraction,
    double BaseHoleCellFraction);
