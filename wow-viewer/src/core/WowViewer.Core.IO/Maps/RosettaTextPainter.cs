using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// One line of text painted into a tile's MCCV vertex colors (Spec 190). <paramref name="PixelMeters"/>
/// is the size of a single font pixel and MUST be an integer multiple of
/// <see cref="RosettaTextPainter.SubCell"/> or the glyph falls between terrain vertices. The
/// <c>Back*</c> channels are what partially covered vertices blend back toward, so they must match
/// whatever plate the text is painted on or the antialiased edge will halo.
/// </summary>
public sealed record RosettaLabel(
    string Text,
    float UStart,
    float VStart,
    float PixelMeters,
    byte R = 255,
    byte G = 255,
    byte B = 255,
    byte BackR = RosettaTextPainter.NeutralChannel,
    byte BackG = RosettaTextPainter.NeutralChannel,
    byte BackB = RosettaTextPainter.NeutralChannel);

/// <summary>A solid MCCV rectangle in tile canvas space — cell plates and grid rules.</summary>
public sealed record RosettaMccvRect(
    float U0,
    float V0,
    float U1,
    float V1,
    byte R,
    byte G,
    byte B);

/// <summary>
/// Paints text and solid rectangles into ADT MCCV vertex colors so a Rosetta tile visually names
/// every placed object (Spec 190). MCCV is stored per chunk as 145 BGRA entries; the terrain shader
/// multiplies terrain color by <c>rgb * 2</c>, so 127 is neutral, 255 doubles brightness and low
/// values darken. Text is therefore painted bright on a darkened plate for maximum contrast.
/// </summary>
/// <remarks>
/// <para>
/// <b>The vertex lattice is a quincunx, not a square grid.</b> Each chunk's 145 vertices are 17
/// interleaved rows: 9 outer (corners of an 8x8 cell grid, pitch <c>chunk/8</c>) then 8 inner (cell
/// centers, offset half a cell on both axes). Vertices therefore exist only where the two half-pitch
/// indices share parity, so a font pixel smaller than one sub-cell lands on nothing half the time.
/// <see cref="SubCell"/> is the smallest font pixel that always covers exactly one outer and one
/// inner vertex; callers must quantize to it.
/// </para>
/// <para>
/// <b>Glyphs are antialiased analytically.</b> Every vertex integrates the glyph over a one-font-pixel
/// box centred on itself and blends by that fractional coverage, so the lattice carries a grayscale
/// field rather than a stencil. Combined with the quincunx that is a genuine 2x resolution win: inner
/// vertices land on font-pixel centres and hold the crisp stroke core, outer vertices land on
/// font-pixel corners and hold the 2x2 resolve.
/// </para>
/// <para>
/// Canvas space <c>(u, v)</c> spans [0, chunkSize*16]² per tile, matching the raw placement
/// coordinates written by <see cref="RosettaTilesetGenerator"/>.
/// </para>
/// </remarks>
public static class RosettaTextPainter
{
    private const int ChunksPerSide = 16;
    private const int VerticesPerChunk = 145;
    private const int RowsPerChunk = 17;
    private const int OuterPerRow = 9;
    private const int InnerPerRow = 8;
    private const int VerticesPerRowPair = OuterPerRow + InnerPerRow;

    /// <summary>Neutral MCCV channel value: terrain color passes through unchanged.</summary>
    public const byte NeutralChannel = 127;

    /// <summary>5x7 bitmap font, one entry per supported character, 7 rows of 5-bit columns (MSB left).</summary>
    private static readonly Dictionary<char, int[]> Glyphs = BuildGlyphs();

    public const int GlyphColumns = 5;
    public const int GlyphRows = 7;

    /// <summary>One blank column between characters, expressed in font pixels.</summary>
    public const int GlyphSpacingColumns = 1;

    /// <summary>Font pixels of horizontal advance per character.</summary>
    public const int CharAdvanceColumns = GlyphColumns + GlyphSpacingColumns;

    /// <summary>Font pixels of vertical advance from one text line to the next (7 rows + 1 blank).</summary>
    public const int LineAdvanceRows = 8;

    /// <summary>
    /// Terrain vertex sub-cell size for the standard 33.333 m chunk: the smallest legible font
    /// pixel. A line of text is exactly <see cref="LineAdvanceRows"/> of these, i.e. one chunk.
    /// </summary>
    public const float SubCell = (533.33333f / 16f) / 8f;

    /// <summary>Sub-cell (minimum font pixel) size for an arbitrary chunk size.</summary>
    public static float SubCellFor(float chunkSizeMeters) => chunkSizeMeters / 8f;

    /// <summary>Width of a rendered text run in meters at the given font pixel size.</summary>
    public static float MeasureWidthMeters(string text, float pixelMeters)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pixelMeters);
        return text.Length * CharAdvanceColumns * pixelMeters;
    }

    /// <summary>How many characters fit on one line of the given width at the given font pixel size.</summary>
    public static int CharsPerLine(float widthMeters, float pixelMeters)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pixelMeters);
        return Math.Max(0, (int)(widthMeters / (CharAdvanceColumns * pixelMeters)));
    }

    /// <summary>
    /// Returns the tile's chunks with <see cref="LkMcnkData.MccvColors"/> populated for every chunk
    /// any rect or label actually touched; untouched chunks keep <c>null</c> MCCV, which renders
    /// identically to a neutral 127 fill and keeps the tiles far smaller. Rects are painted first,
    /// then glyphs on top. Input chunks are not mutated.
    /// </summary>
    public static IReadOnlyList<LkMcnkData> PaintTile(
        IReadOnlyList<LkMcnkData> chunks,
        IReadOnlyList<RosettaMccvRect> rects,
        IReadOnlyList<RosettaLabel> labels,
        float chunkSizeMeters)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(rects);
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkSizeMeters);
        if (chunks.Count != ChunksPerSide * ChunksPerSide)
            throw new ArgumentException($"Expected {ChunksPerSide * ChunksPerSide} chunks, found {chunks.Count}.", nameof(chunks));

        byte[]?[] painted = new byte[chunks.Count][];

        foreach (RosettaMccvRect rect in rects)
            FillRect(painted, rect.U0, rect.V0, rect.U1, rect.V1, rect.R, rect.G, rect.B, chunkSizeMeters);

        foreach (RosettaLabel label in labels)
            PaintLabel(painted, label, chunkSizeMeters);

        var result = new LkMcnkData[chunks.Count];
        for (int i = 0; i < chunks.Count; i++)
            result[i] = painted[i] is { } mccv ? WithMccv(chunks[i], mccv) : chunks[i];
        return result;
    }

    /// <summary>
    /// Paints one text run with analytic antialiasing. A glyph is a union of axis-aligned font pixel
    /// squares; every terrain vertex in reach integrates that union over a box of one font pixel
    /// centred on itself, and blends from the label's background colour to its ink colour by the
    /// resulting fractional coverage.
    /// </summary>
    /// <remarks>
    /// Sampling "is this vertex inside a lit pixel" instead quantizes every stroke to the lattice and
    /// throws away the quincunx's half-pixel offset — that is what made the first painted runs look
    /// chewed.
    /// </remarks>
    /// <summary>
    /// Flattens a run of text into one ink mask per font-pixel column: bit <c>r</c> set means glyph
    /// row <c>r</c> is ink. Shared with <see cref="RosettaAlphaPainter"/> so both the vertex-colour
    /// and texture-alpha paths render the same letterforms.
    /// </summary>
    public static List<int> BuildColumnMasks(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var columns = new List<int>(text.Length * CharAdvanceColumns);
        foreach (char c in text)
        {
            if (!Glyphs.TryGetValue(char.ToUpperInvariant(c), out int[]? rows))
            {
                for (int i = 0; i < CharAdvanceColumns; i++)
                    columns.Add(0);
                continue;
            }

            for (int col = 0; col < GlyphColumns; col++)
            {
                int mask = 0;
                for (int row = 0; row < GlyphRows; row++)
                {
                    if ((rows[row] & (1 << (GlyphColumns - 1 - col))) != 0)
                        mask |= 1 << row;
                }

                columns.Add(mask);
            }

            for (int i = 0; i < GlyphSpacingColumns; i++)
                columns.Add(0);
        }

        return columns;
    }

    private static void PaintLabel(byte[]?[] painted, RosettaLabel label, float chunkSizeMeters)
    {
        float pixel = label.PixelMeters;
        if (pixel <= 0f || label.Text.Length == 0)
            return;

        List<int> columns = BuildColumnMasks(label.Text);

        float half = pixel / 2f;
        // A vertex can pick up ink from up to half a font pixel away in each direction.
        ForEachVertexIn(
            painted,
            label.UStart - half,
            label.VStart - half,
            label.UStart + columns.Count * pixel + half,
            label.VStart + GlyphRows * pixel + half,
            chunkSizeMeters,
            (colors, index, u, v) =>
            {
                float coverage = SampleCoverage(
                    columns, label.UStart, label.VStart, pixel,
                    u - half, v - half, u + half, v + half);
                if (coverage <= 0.002f)
                    return;

                int offset = index * 4;
                colors[offset + 0] = Blend(label.BackB, label.B, coverage);
                colors[offset + 1] = Blend(label.BackG, label.G, coverage);
                colors[offset + 2] = Blend(label.BackR, label.R, coverage);
                colors[offset + 3] = 255;
            });
    }

    /// <summary>
    /// Fraction of the sample box covered by lit font pixels, in the run's own coordinate space.
    /// Exact rather than supersampled: the overlap of two axis-aligned rectangles is a closed form.
    /// </summary>
    public static float SampleCoverage(
        IReadOnlyList<int> columns, float originU, float originV, float pixel,
        float su0, float sv0, float su1, float sv1)
    {
        int colStart = Math.Max(0, (int)MathF.Floor((su0 - originU) / pixel));
        int colEnd = Math.Min(columns.Count, (int)MathF.Ceiling((su1 - originU) / pixel));
        int rowStart = Math.Max(0, (int)MathF.Floor((sv0 - originV) / pixel));
        int rowEnd = Math.Min(GlyphRows, (int)MathF.Ceiling((sv1 - originV) / pixel));

        float area = 0f;
        for (int col = colStart; col < colEnd; col++)
        {
            int mask = columns[col];
            if (mask == 0)
                continue;

            float pixelU0 = originU + col * pixel;
            float overlapU = MathF.Min(su1, pixelU0 + pixel) - MathF.Max(su0, pixelU0);
            if (overlapU <= 0f)
                continue;

            for (int row = rowStart; row < rowEnd; row++)
            {
                if ((mask & (1 << row)) == 0)
                    continue;

                float pixelV0 = originV + row * pixel;
                float overlapV = MathF.Min(sv1, pixelV0 + pixel) - MathF.Max(sv0, pixelV0);
                if (overlapV > 0f)
                    area += overlapU * overlapV;
            }
        }

        return Math.Clamp(area / (pixel * pixel), 0f, 1f);
    }

    private static byte Blend(byte from, byte to, float t) =>
        (byte)Math.Clamp(MathF.Round(from + ((to - from) * t)), 0f, 255f);

    /// <summary>Drives every terrain vertex inside the rectangle to a flat colour.</summary>
    private static void FillRect(
        byte[]?[] painted, float u0, float v0, float u1, float v1,
        byte red, byte green, byte blue, float chunkSizeMeters)
    {
        ForEachVertexIn(painted, u0, v0, u1, v1, chunkSizeMeters, (colors, index, _, _) =>
        {
            int offset = index * 4;
            // CImVector storage is BGRA. Alpha is not an opacity (the shader gate was removed) but
            // Blizzard tiles store 255, so match them.
            colors[offset + 0] = blue;
            colors[offset + 1] = green;
            colors[offset + 2] = red;
            colors[offset + 3] = 255;
        });
    }

    /// <summary>
    /// Invokes <paramref name="action"/> for every terrain vertex whose canvas position falls in
    /// [u0,u1) x [v0,v1), allocating a chunk's MCCV buffer (neutral-filled) the first time it is
    /// touched. Vertex indices come straight from the interleaved row layout, so cost is
    /// proportional to covered vertices rather than to the 145 of every touched chunk.
    /// </summary>
    private static void ForEachVertexIn(
        byte[]?[] painted, float u0, float v0, float u1, float v1, float chunkSizeMeters,
        Action<byte[], int, float, float> action)
    {
        if (u1 <= u0 || v1 <= v0)
            return;

        float tileSize = chunkSizeMeters * ChunksPerSide;
        if (u1 <= 0f || v1 <= 0f || u0 >= tileSize || v0 >= tileSize)
            return;

        float subCell = chunkSizeMeters / 8f;
        int minCx = Math.Clamp((int)MathF.Floor(u0 / chunkSizeMeters), 0, ChunksPerSide - 1);
        int maxCx = Math.Clamp((int)MathF.Floor(u1 / chunkSizeMeters), 0, ChunksPerSide - 1);
        int minCy = Math.Clamp((int)MathF.Floor(v0 / chunkSizeMeters), 0, ChunksPerSide - 1);
        int maxCy = Math.Clamp((int)MathF.Floor(v1 / chunkSizeMeters), 0, ChunksPerSide - 1);

        for (int cy = minCy; cy <= maxCy; cy++)
        {
            float chunkV = cy * chunkSizeMeters;
            for (int cx = minCx; cx <= maxCx; cx++)
            {
                float chunkU = cx * chunkSizeMeters;
                byte[]? colors = null;

                for (int row = 0; row < RowsPerChunk; row++)
                {
                    bool isInner = (row & 1) != 0;
                    float v = chunkV + (isInner ? ((row / 2) + 0.5f) * subCell : (row / 2) * subCell);
                    if (v < v0 || v >= v1)
                        continue;

                    int rowSize = isInner ? InnerPerRow : OuterPerRow;
                    int rowBase = ((row / 2) * VerticesPerRowPair) + (isInner ? OuterPerRow : 0);

                    for (int col = 0; col < rowSize; col++)
                    {
                        float u = chunkU + (isInner ? (col + 0.5f) * subCell : col * subCell);
                        if (u < u0 || u >= u1)
                            continue;

                        colors ??= EnsureChunk(painted, (cy * ChunksPerSide) + cx);
                        action(colors, rowBase + col, u, v);
                    }
                }
            }
        }
    }

    private static byte[] EnsureChunk(byte[]?[] painted, int index)
    {
        if (painted[index] is { } existing)
            return existing;

        byte[] colors = new byte[VerticesPerChunk * 4];
        Array.Fill(colors, NeutralChannel);
        for (int i = 3; i < colors.Length; i += 4)
            colors[i] = 255;
        painted[index] = colors;
        return colors;
    }

    private static LkMcnkData WithMccv(LkMcnkData chunk, byte[] mccv) => new()
    {
        IndexX = chunk.IndexX,
        IndexY = chunk.IndexY,
        Flags = chunk.Flags,
        AreaId = chunk.AreaId,
        NLayers = chunk.NLayers,
        HoleMask = chunk.HoleMask,
        BaseHeight = chunk.BaseHeight,
        Heights = chunk.Heights,
        Normals = chunk.Normals,
        ShadowMap = chunk.ShadowMap,
        AlphaMapData = chunk.AlphaMapData,
        AlphaMapSize = chunk.AlphaMapSize,
        Layers = chunk.Layers,
        DoodadRefs = chunk.DoodadRefs,
        WorldModelRefs = chunk.WorldModelRefs,
        LiquidData = chunk.LiquidData,
        MccvColors = mccv,
        MclvLighting = chunk.MclvLighting,
        PosX = chunk.PosX,
        PosY = chunk.PosY,
        PosZ = chunk.PosZ,
    };

    private static Dictionary<char, int[]> BuildGlyphs()
    {
        // Rows top→bottom, 5 bits per row, MSB = leftmost column.
        var glyphs = new Dictionary<char, int[]>();
        void Add(char c, params int[] rows) => glyphs[c] = rows;

        Add('A', 0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001);
        Add('B', 0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110);
        Add('C', 0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110);
        Add('D', 0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110);
        Add('E', 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111);
        Add('F', 0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000);
        Add('G', 0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110);
        Add('H', 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001);
        Add('I', 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111);
        Add('J', 0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100);
        Add('K', 0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001);
        Add('L', 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111);
        Add('M', 0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001);
        Add('N', 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001);
        Add('O', 0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110);
        Add('P', 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000);
        Add('Q', 0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101);
        Add('R', 0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001);
        Add('S', 0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110);
        Add('T', 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100);
        Add('U', 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110);
        Add('V', 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100);
        Add('W', 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001);
        Add('X', 0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001);
        Add('Y', 0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100);
        Add('Z', 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111);
        Add('0', 0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110);
        Add('1', 0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110);
        Add('2', 0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111);
        Add('3', 0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110);
        Add('4', 0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010);
        Add('5', 0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110);
        Add('6', 0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110);
        Add('7', 0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000);
        Add('8', 0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110);
        Add('9', 0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100);
        Add('_', 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111);
        Add('-', 0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000);
        Add('.', 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100);
        return glyphs;
    }
}
