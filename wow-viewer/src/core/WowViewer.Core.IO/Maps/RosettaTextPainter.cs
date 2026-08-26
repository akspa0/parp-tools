using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>A text label painted into a tile's MCCV vertex colors (Spec 190).</summary>
public sealed record RosettaLabel(string Text, float UStart, float VStart, float CapHeightMeters);

/// <summary>
/// Paints human-readable text into ADT MCCV vertex colors so a Rosetta tile visually names every
/// placed object (Spec 190). MCCV is stored per chunk as 145 BGRA entries (17x17 vertices,
/// row-major) where 127/127/127 is the neutral tint; painted glyph vertices are driven to a bright
/// color so they read as glowing text on otherwise neutral terrain.
/// </summary>
/// <remarks>
/// Tile vertex space: 16x16 chunks, each 17x17 vertices, vertex pitch = chunk size / 16
/// (~2.083 m). <c>u</c> runs west→east along +X inside the tile, <c>v</c> runs north→south along
/// −Y. Both shared-edge copies of a boundary vertex are painted identically so labels stay
/// continuous across chunk seams.
/// </remarks>
public static class RosettaTextPainter
{
    private const int ChunksPerSide = 16;
    private const int VerticesPerChunkSide = 17;
    private const int VerticesPerChunk = VerticesPerChunkSide * VerticesPerChunkSide;
    private const float NeutralChannel = 127f;

    /// <summary>5x7 bitmap font, one entry per supported character, 7 rows of 5-bit columns (MSB left).</summary>
    private static readonly Dictionary<char, int[]> Glyphs = BuildGlyphs();

    public const int GlyphColumns = 5;
    public const int GlyphRows = 7;
    /// <summary>One blank column between characters, expressed in font pixels.</summary>
    public const int GlyphSpacingColumns = 1;

    /// <summary>
    /// Returns the tile's chunks with <see cref="LkMcnkData.MccvColors"/> populated: neutral
    /// everywhere except the glyph vertices of each label. Input chunks are not mutated.
    /// </summary>
    public static IReadOnlyList<LkMcnkData> PaintTileLabels(
        IReadOnlyList<LkMcnkData> chunks,
        IReadOnlyList<RosettaLabel> labels,
        float chunkSizeMeters)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkSizeMeters);
        if (chunks.Count != ChunksPerSide * ChunksPerSide)
            throw new ArgumentException($"Expected {ChunksPerSide * ChunksPerSide} chunks, found {chunks.Count}.", nameof(chunks));

        byte[][] painted = new byte[chunks.Count][];
        for (int i = 0; i < chunks.Count; i++)
        {
            byte[] colors = new byte[VerticesPerChunk * 4];
            Array.Fill(colors, (byte)NeutralChannel);
            painted[i] = colors;
        }

        foreach (RosettaLabel label in labels)
            PaintLabel(painted, label, chunkSizeMeters);

        var result = new LkMcnkData[chunks.Count];
        for (int i = 0; i < chunks.Count; i++)
            result[i] = WithMccv(chunks[i], painted[i]);
        return result;
    }

    /// <summary>Width of the rendered text in meters at the label's cap height.</summary>
    public static float MeasureWidthMeters(string text, float capHeightMeters, float chunkSizeMeters)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capHeightMeters);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkSizeMeters);

        float vertexPitch = chunkSizeMeters / (VerticesPerChunkSide - 1);
        float pixelSize = MathF.Max(1f, MathF.Round(capHeightMeters / GlyphRows / vertexPitch));
        int columns = text.Length * (GlyphColumns + GlyphSpacingColumns);
        return columns * pixelSize * vertexPitch;
    }

    private static void PaintLabel(byte[][] paintedChunks, RosettaLabel label, float chunkSizeMeters)
    {
        float vertexPitch = chunkSizeMeters / (VerticesPerChunkSide - 1);
        int pixelSize = Math.Max(1, (int)MathF.Round(label.CapHeightMeters / GlyphRows / vertexPitch));
        float pixelMeters = pixelSize * vertexPitch;

        string text = label.Text;
        for (int charIndex = 0; charIndex < text.Length; charIndex++)
        {
            if (!Glyphs.TryGetValue(char.ToUpperInvariant(text[charIndex]), out int[]? rows))
                continue;

            float glyphU = label.UStart + charIndex * (GlyphColumns + GlyphSpacingColumns) * pixelMeters;

            for (int row = 0; row < GlyphRows; row++)
            {
                int bits = rows[row];
                for (int col = 0; col < GlyphColumns; col++)
                {
                    if ((bits & (1 << (GlyphColumns - 1 - col))) == 0)
                        continue;

                    float u0 = glyphU + col * pixelMeters;
                    float v0 = label.VStart + row * pixelMeters;
                    FillPixel(paintedChunks, u0, v0, pixelMeters, chunkSizeMeters);
                }
            }
        }
    }

    private static void FillPixel(byte[][] paintedChunks, float u0, float v0, float pixelMeters, float chunkSizeMeters)
    {
        float u1 = u0 + pixelMeters;
        float v1 = v0 + pixelMeters;

        int minCx = Math.Clamp((int)MathF.Floor(u0 / chunkSizeMeters), 0, ChunksPerSide - 1);
        int maxCx = Math.Clamp((int)MathF.Floor(u1 / chunkSizeMeters), 0, ChunksPerSide - 1);
        int minCy = Math.Clamp((int)MathF.Floor(v0 / chunkSizeMeters), 0, ChunksPerSide - 1);
        int maxCy = Math.Clamp((int)MathF.Floor(v1 / chunkSizeMeters), 0, ChunksPerSide - 1);

        for (int cy = minCy; cy <= maxCy; cy++)
        {
            for (int cx = minCx; cx <= maxCx; cx++)
            {
                byte[] colors = paintedChunks[cy * ChunksPerSide + cx];
                float chunkU0 = cx * chunkSizeMeters;
                float chunkV0 = cy * chunkSizeMeters;

                for (int vertex = 0; vertex < VerticesPerChunk; vertex++)
                {
                    float u = chunkU0 + (vertex % VerticesPerChunkSide) * chunkSizeMeters / (VerticesPerChunkSide - 1);
                    float v = chunkV0 + (vertex / VerticesPerChunkSide) * chunkSizeMeters / (VerticesPerChunkSide - 1);

                    bool insideHorizontal = u >= u0 && (u < u1 || (pixelMeters <= chunkSizeMeters / (VerticesPerChunkSide - 1) && u <= u1));
                    bool insideVertical = v >= v0 && (v < v1 || (pixelMeters <= chunkSizeMeters / (VerticesPerChunkSide - 1) && v <= v1));
                    if (!insideHorizontal || !insideVertical)
                        continue;

                    int baseIndex = vertex * 4;
                    // BGRA storage; drive RGB high so the glyph reads bright against neutral terrain.
                    colors[baseIndex + 0] = 255;
                    colors[baseIndex + 1] = 255;
                    colors[baseIndex + 2] = 255;
                    colors[baseIndex + 3] = 127;
                }
            }
        }
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
