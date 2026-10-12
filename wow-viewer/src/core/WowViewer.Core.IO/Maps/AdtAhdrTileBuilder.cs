using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.IO.Maps;

/// <summary>One terrain chunk in ADT terms: 145 interleaved heights (yards), optional normals and MCCV, and its layers.</summary>
/// <param name="ChunkX">Chunk column 0..15 (DAT ACNK IndexX; grid column axis).</param>
/// <param name="ChunkY">Chunk row 0..15 (DAT ACNK IndexY; grid row axis).</param>
/// <param name="HeightsYards">145 absolute heights in yards, interleaved 9/8 rows as in MCVT.</param>
/// <param name="GridNormals">145 unit normals in the grid frame (column axis, vertical, row axis), or null.</param>
/// <param name="VertexColors">145 × 4 MCCV bytes, or null (written as neutral 127,127,127,255).</param>
/// <param name="Layers">Texture index into the tile texture list and the ADT sequential alpha (null for layer 0 or an opaque layer).</param>
public sealed record DatV26SourceChunk(
    int ChunkX,
    int ChunkY,
    float[] HeightsYards,
    Vector3[]? GridNormals,
    byte[]? VertexColors,
    IReadOnlyList<(int TextureIndex, byte[]? SequentialAlpha)> Layers);

/// <summary>An object placement in tile-grid terms.</summary>
/// <param name="Column">Fractional outer-grid column 0..128 (8 per chunk).</param>
/// <param name="Row">Fractional outer-grid row 0..128.</param>
/// <param name="HeightYards">Absolute height in yards.</param>
/// <param name="RotationDegrees">Rotation in ACDO order (column axis, vertical, row axis), which is MDDF's file order.</param>
public sealed record DatV26SourcePlacement(string ModelName, uint UniqueId, float Column, float Row, float HeightYards, Vector3 RotationDegrees, float Scale);

/// <summary>
/// Spec 237 (experimental): builds a DAT v26 tile from ADT-style data using the measured frames, so client ADTs can be
/// written as DAT files and read back through <see cref="AdtAhdrReader"/> as a check on the decoding.
/// Unknown fields take the values observed in every corpus file (AHDR +0x14 = 8396383, ALOC[0] = 2869, ACNK +0x08 =
/// 0xD000, ALYR flags 0x100, ACDO +0x20 = 1.0, zero AOCH/ASHD).
/// </summary>
public static class AdtAhdrTileBuilder
{
    private const int Outer = 129;
    private const int Inner = 128;
    private const float InchesPerYard = 36f;

    public static AdtAhdrTile Build(
        int alocX,
        int alocY,
        IReadOnlyList<string> textureNames,
        IReadOnlyList<DatV26SourceChunk> chunks,
        IReadOnlyList<DatV26SourcePlacement> placements,
        string sourcePath)
    {
        var outer = new float[Outer * Outer];
        var inner = new float[Inner * Inner];
        var normals = new byte[(Outer * Outer + Inner * Inner) * 3];
        var colors = new byte[(Outer * Outer + Inner * Inner) * 4];
        for (int i = 0; i < colors.Length; i += 4)
        {
            colors[i] = 127; colors[i + 1] = 127; colors[i + 2] = 127; colors[i + 3] = 255;
        }

        // Neutral "up" normal for vertices no chunk covers.
        for (int i = 0; i < normals.Length; i += 3)
            normals[i + 1] = 127;

        var byPosition = new Dictionary<(int X, int Y), DatV26SourceChunk>();
        foreach (DatV26SourceChunk chunk in chunks)
        {
            if ((uint)chunk.ChunkX > 15 || (uint)chunk.ChunkY > 15 || chunk.HeightsYards.Length < AdtAhdrTileSlicer.VerticesPerChunk)
                continue;

            byPosition[(chunk.ChunkX, chunk.ChunkY)] = chunk;
            int index = 0;
            for (int row = 0; row < 17; row++)
            {
                int r = row / 2;
                bool isOuter = (row & 1) == 0;
                int count = isOuter ? 9 : 8;
                for (int c = 0; c < count; c++, index++)
                {
                    int vertex = isOuter
                        ? (chunk.ChunkY * 8 + r) * Outer + chunk.ChunkX * 8 + c
                        : Outer * Outer + (chunk.ChunkY * 8 + r) * Inner + chunk.ChunkX * 8 + c;
                    float inches = chunk.HeightsYards[index] * InchesPerYard;
                    if (isOuter) outer[vertex] = inches; else inner[vertex - Outer * Outer] = inches;

                    if (chunk.GridNormals is { Length: >= AdtAhdrTileSlicer.VerticesPerChunk } n)
                    {
                        normals[vertex * 3] = ToSByte(n[index].X);
                        normals[vertex * 3 + 1] = ToSByte(n[index].Y);
                        normals[vertex * 3 + 2] = ToSByte(n[index].Z);
                    }

                    if (chunk.VertexColors is { Length: >= AdtAhdrTileSlicer.VerticesPerChunk * 4 } mccv)
                        Buffer.BlockCopy(mccv, index * 4, colors, vertex * 4, 4);
                }
            }
        }

        var models = new List<string>();
        var modelIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var objectsByChunk = new Dictionary<(int X, int Y), List<AdtAhdrObjectDefinition>>();
        var skeleton = new AdtAhdrTile
        {
            SourcePath = sourcePath,
            VerticesX = Outer,
            VerticesY = Outer,
            ChunksX = 16,
            ChunksY = 16,
            OuterHeights = outer,
            InnerHeights = inner,
        };

        foreach (DatV26SourcePlacement placement in placements)
        {
            int chunkX = Math.Clamp((int)MathF.Floor(placement.Column / 8f), 0, 15);
            int chunkY = Math.Clamp((int)MathF.Floor(placement.Row / 8f), 0, 15);
            if (!modelIndex.TryGetValue(placement.ModelName, out int model))
            {
                model = models.Count;
                models.Add(placement.ModelName);
                modelIndex[placement.ModelName] = model;
            }

            float mean = AdtAhdrTileSlicer.ChunkMeanHeight(skeleton, chunkX, chunkY);
            var local = new Vector3(
                (placement.Column - (chunkX * 8 + 4)) * AdtAhdrTileSlicer.InchesPerCell,
                placement.HeightYards * InchesPerYard - mean,
                (placement.Row - (chunkY * 8 + 4)) * AdtAhdrTileSlicer.InchesPerCell);

            if (!objectsByChunk.TryGetValue((chunkX, chunkY), out var list))
                objectsByChunk[(chunkX, chunkY)] = list = [];
            list.Add(new AdtAhdrObjectDefinition(model, local, placement.RotationDegrees, placement.Scale,
                1f, 0u, 0f, placement.UniqueId, 0u, [], []));
        }

        var acnks = new List<AdtAhdrChunk>(256);
        for (int i = 0; i < 256; i++)
        {
            int chunkX = i % 16, chunkY = i / 16;
            var header = new byte[0x40];
            BinaryPrimitives.WriteInt32LittleEndian(header, chunkX);
            BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(4), chunkY);
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(8), 0xD000);

            var layers = new List<AdtAhdrLayer>();
            if (byPosition.TryGetValue((chunkX, chunkY), out DatV26SourceChunk? source) && source.Layers.Count > 0)
            {
                byte[][] weights = AdtAhdrAlpha.SequentialAlphaToWeights(source.Layers.Count, source.Layers.Skip(1).Select(static l => l.SequentialAlpha).ToArray());
                for (int layer = 0; layer < source.Layers.Count; layer++)
                    layers.Add(new AdtAhdrLayer(source.Layers[layer].TextureIndex, 0x100, weights[layer]));
                AdtAhdrAlpha.PredominantLayerMap(weights).CopyTo(header, 0x12);
            }

            objectsByChunk.TryGetValue((chunkX, chunkY), out List<AdtAhdrObjectDefinition>? objects);
            bool hasSubChunks = layers.Count > 0 || objects is { Count: > 0 };
            acnks.Add(new AdtAhdrChunk
            {
                IndexX = chunkX,
                IndexY = chunkY,
                HeaderRaw = header,
                Layers = layers,
                ShadowRaw = hasSubChunks ? new byte[512] : null,
                Objects = (IReadOnlyList<AdtAhdrObjectDefinition>?)objects ?? [],
            });
        }

        return new AdtAhdrTile
        {
            SourcePath = sourcePath,
            MverVersion = 26,
            Version = 26,
            VerticesX = Outer,
            VerticesY = Outer,
            ChunksX = 16,
            ChunksY = 16,
            HeaderReserved = [8396383u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            Aloc = [2869u, (uint)alocX, (uint)alocY, (uint)alocX, (uint)alocY],
            AochRaw = new byte[0x800],
            OuterHeights = outer,
            InnerHeights = inner,
            NormalsRaw = normals,
            VertexShadingRaw = colors,
            TextureNames = textureNames,
            ModelNames = models,
            Chunks = acnks,
        };
    }

    private static byte ToSByte(float component) => unchecked((byte)(sbyte)Math.Clamp((int)MathF.Round(component * 127f), -127, 127));
}
