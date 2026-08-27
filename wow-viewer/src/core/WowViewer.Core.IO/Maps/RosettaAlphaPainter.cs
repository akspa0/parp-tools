namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Paints Rosetta labels into an ADT's texture-layer alpha map (MCAL) rather than its vertex colours
/// (Spec 190).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists.</b> MCCV carries one value per terrain vertex — a 4.167 m lattice — and the
/// 0.5.3 alpha container has no MCCV chunk at all, so alpha-era tiles could carry no painted text
/// whatsoever. MCAL is 64x64 texels per chunk, i.e. <b>1024x1024 across a tile</b>: a texel is
/// 0.52 m, eight times finer on each axis, and it exists in every era. Text painted here is legible
/// at a size the vertex lattice cannot represent, and it survives the LK-to-alpha conversion.
/// </para>
/// <para>
/// The canvas is a single tile-wide byte buffer in the same <c>(u, v)</c> space the placements and
/// the vertex painter use, sliced into per-chunk 64x64 blocks at the end. Coverage is evaluated
/// analytically against the shared 5x7 font in <see cref="RosettaTextPainter"/>, so both painters
/// draw the same letterforms.
/// </para>
/// </remarks>
public static class RosettaAlphaPainter
{
    /// <summary>Alpha texels along one chunk edge — the MCAL resolution.</summary>
    public const int TexelsPerChunk = 64;

    /// <summary>Chunks along one tile edge.</summary>
    public const int ChunksPerSide = 16;

    /// <summary>Alpha texels along one tile edge.</summary>
    public const int TexelsPerTile = TexelsPerChunk * ChunksPerSide;

    /// <summary>Bytes in one chunk's uncompressed 4-bit alpha map (64x64 texels, two per byte).</summary>
    public const int PackedChunkBytes = TexelsPerChunk * TexelsPerChunk / 2;

    /// <summary>A blank tile-wide alpha canvas: layer 1 fully transparent everywhere.</summary>
    public static byte[] CreateCanvas() => new byte[TexelsPerTile * TexelsPerTile];

    /// <summary>Size of one alpha texel in meters, for the given chunk size.</summary>
    public static float TexelSize(float chunkSizeMeters) => chunkSizeMeters / TexelsPerChunk;

    /// <summary>Width of a rendered text run in meters at the given font pixel size.</summary>
    public static float MeasureWidthMeters(string text, float pixelMeters) =>
        RosettaTextPainter.MeasureWidthMeters(text, pixelMeters);

    /// <summary>How many characters fit on one line of the given width.</summary>
    public static int CharsPerLine(float widthMeters, float pixelMeters) =>
        RosettaTextPainter.CharsPerLine(widthMeters, pixelMeters);

    /// <summary>Height of one text line in meters (7 glyph rows plus a blank).</summary>
    public static float LineAdvanceMeters(float pixelMeters) =>
        RosettaTextPainter.LineAdvanceRows * pixelMeters;

    /// <summary>Fills a canvas rectangle to a flat alpha value.</summary>
    public static void FillRect(
        byte[] canvas, float u0, float v0, float u1, float v1, byte value, float chunkSizeMeters)
    {
        ArgumentNullException.ThrowIfNull(canvas);
        if (u1 <= u0 || v1 <= v0)
            return;

        float texel = TexelSize(chunkSizeMeters);
        int minX = Math.Clamp((int)MathF.Floor(u0 / texel), 0, TexelsPerTile - 1);
        int maxX = Math.Clamp((int)MathF.Ceiling(u1 / texel) - 1, 0, TexelsPerTile - 1);
        int minY = Math.Clamp((int)MathF.Floor(v0 / texel), 0, TexelsPerTile - 1);
        int maxY = Math.Clamp((int)MathF.Ceiling(v1 / texel) - 1, 0, TexelsPerTile - 1);

        for (int y = minY; y <= maxY; y++)
        {
            int row = y * TexelsPerTile;
            for (int x = minX; x <= maxX; x++)
                canvas[row + x] = value;
        }
    }

    /// <summary>
    /// Draws one text run, blending from whatever the canvas already holds to <paramref name="ink"/>
    /// by the glyph's exact coverage of each texel, so edges are antialiased rather than stepped.
    /// </summary>
    public static void DrawText(
        byte[] canvas, string text, float uStart, float vStart, float pixelMeters, byte ink, float chunkSizeMeters)
    {
        ArgumentNullException.ThrowIfNull(canvas);
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length == 0 || pixelMeters <= 0f)
            return;

        List<int> columns = RosettaTextPainter.BuildColumnMasks(text);
        float texel = TexelSize(chunkSizeMeters);

        float runU1 = uStart + (columns.Count * pixelMeters);
        float runV1 = vStart + (RosettaTextPainter.GlyphRows * pixelMeters);

        int minX = Math.Clamp((int)MathF.Floor(uStart / texel), 0, TexelsPerTile - 1);
        int maxX = Math.Clamp((int)MathF.Ceiling(runU1 / texel), 0, TexelsPerTile - 1);
        int minY = Math.Clamp((int)MathF.Floor(vStart / texel), 0, TexelsPerTile - 1);
        int maxY = Math.Clamp((int)MathF.Ceiling(runV1 / texel), 0, TexelsPerTile - 1);

        for (int y = minY; y <= maxY; y++)
        {
            float tv0 = y * texel;
            int row = y * TexelsPerTile;

            for (int x = minX; x <= maxX; x++)
            {
                float tu0 = x * texel;
                float coverage = RosettaTextPainter.SampleCoverage(
                    columns, uStart, vStart, pixelMeters, tu0, tv0, tu0 + texel, tv0 + texel);
                if (coverage <= 0.002f)
                    continue;

                byte existing = canvas[row + x];
                canvas[row + x] = (byte)Math.Clamp(
                    MathF.Round(existing + ((ink - existing) * coverage)), 0f, 255f);
            }
        }
    }

    /// <summary>
    /// Slices the tile canvas into one uncompressed 4-bit MCAL block per chunk, in MCNK order.
    /// 4-bit is the era-neutral choice: it needs no MPHD big-alpha flag on the LK side and converts
    /// cleanly for the alpha container, and text only ever needs the extremes anyway.
    /// </summary>
    public static byte[][] SliceToChunks(byte[] canvas)
    {
        ArgumentNullException.ThrowIfNull(canvas);
        if (canvas.Length != TexelsPerTile * TexelsPerTile)
            throw new ArgumentException($"Canvas must be {TexelsPerTile}x{TexelsPerTile}.", nameof(canvas));

        byte[][] chunks = new byte[ChunksPerSide * ChunksPerSide][];
        for (int cy = 0; cy < ChunksPerSide; cy++)
        {
            for (int cx = 0; cx < ChunksPerSide; cx++)
            {
                byte[] packed = new byte[PackedChunkBytes];
                bool any = false;

                for (int y = 0; y < TexelsPerChunk; y++)
                {
                    int srcRow = ((cy * TexelsPerChunk) + y) * TexelsPerTile;
                    int dstRow = y * (TexelsPerChunk / 2);

                    for (int x = 0; x < TexelsPerChunk; x += 2)
                    {
                        int lo = canvas[srcRow + (cx * TexelsPerChunk) + x] >> 4;
                        int hi = canvas[srcRow + (cx * TexelsPerChunk) + x + 1] >> 4;
                        byte value = (byte)(lo | (hi << 4));
                        packed[dstRow + (x / 2)] = value;
                        any |= value != 0;
                    }
                }

                // A chunk the labels never reached needs no second layer at all.
                chunks[(cy * ChunksPerSide) + cx] = any ? packed : [];
            }
        }

        return chunks;
    }
}
