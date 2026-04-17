using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WorldTerrainVisualSnapshot
{
    public WorldTerrainVisualSnapshot(int width, int height, int sampledPixelCount, string visualHash, byte[] rgbaPixels)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 16);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 16);
        ArgumentOutOfRangeException.ThrowIfNegative(sampledPixelCount);
        ArgumentException.ThrowIfNullOrWhiteSpace(visualHash);
        ArgumentNullException.ThrowIfNull(rgbaPixels);
        if (rgbaPixels.Length != width * height * 4)
            throw new ArgumentException("Terrain preview RGBA payloads must match their declared dimensions.", nameof(rgbaPixels));

        Width = width;
        Height = height;
        SampledPixelCount = sampledPixelCount;
        VisualHash = visualHash;
        RgbaPixels = rgbaPixels;
    }

    public int Width { get; }

    public int Height { get; }

    public int SampledPixelCount { get; }

    public string VisualHash { get; }

    public byte[] RgbaPixels { get; }
}

public static class WorldTerrainVisualSnapshotBuilder
{
    public static WorldTerrainVisualSnapshot Build(WorldTerrainTileData terrainTileData)
    {
        ArgumentNullException.ThrowIfNull(terrainTileData);

        WorldTerrainHeightmapData? heightmap = terrainTileData.Heightmap;
        if (heightmap is null)
            return CreateEmptySnapshot(257, 257);

        byte[] rgbaPixels = new byte[checked(heightmap.Width * heightmap.Height * 4)];
        float range = MathF.Max(1e-6f, heightmap.HeightRange);
        for (int y = 0; y < heightmap.Height; y++)
        {
            int sourceOffset = y * heightmap.Width;
            int pixelOffset = y * heightmap.Width * 4;
            for (int x = 0; x < heightmap.Width; x++)
            {
                float normalized = Math.Clamp((heightmap.Heights[sourceOffset + x] - heightmap.MinHeight) / range, 0f, 1f);
                (byte r, byte g, byte b) = GetPreviewColor(normalized);
                rgbaPixels[pixelOffset + 0] = r;
                rgbaPixels[pixelOffset + 1] = g;
                rgbaPixels[pixelOffset + 2] = b;
                rgbaPixels[pixelOffset + 3] = 255;
                pixelOffset += 4;
            }
        }

        return new WorldTerrainVisualSnapshot(
            heightmap.Width,
            heightmap.Height,
            heightmap.AuthoritativeSampleCount,
            ComputeHash(heightmap.Width, heightmap.Height, rgbaPixels),
            rgbaPixels);
    }

    public static void WriteBmp(Stream output, WorldTerrainVisualSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(snapshot);

        int rowStride = ((snapshot.Width * 3) + 3) & ~3;
        int pixelDataSize = checked(rowStride * snapshot.Height);
        int fileSize = 14 + 40 + pixelDataSize;
        Span<byte> header = stackalloc byte[54];
        header[0] = (byte)'B';
        header[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(header[2..6], fileSize);
        BinaryPrimitives.WriteInt32LittleEndian(header[10..14], 54);
        BinaryPrimitives.WriteInt32LittleEndian(header[14..18], 40);
        BinaryPrimitives.WriteInt32LittleEndian(header[18..22], snapshot.Width);
        BinaryPrimitives.WriteInt32LittleEndian(header[22..26], snapshot.Height);
        BinaryPrimitives.WriteInt16LittleEndian(header[26..28], 1);
        BinaryPrimitives.WriteInt16LittleEndian(header[28..30], 24);
        BinaryPrimitives.WriteInt32LittleEndian(header[34..38], pixelDataSize);
        output.Write(header);

        byte[] padding = new byte[rowStride - (snapshot.Width * 3)];
        for (int y = snapshot.Height - 1; y >= 0; y--)
        {
            int rowOffset = y * snapshot.Width * 4;
            for (int x = 0; x < snapshot.Width; x++)
            {
                int offset = rowOffset + (x * 4);
                output.WriteByte(snapshot.RgbaPixels[offset + 2]);
                output.WriteByte(snapshot.RgbaPixels[offset + 1]);
                output.WriteByte(snapshot.RgbaPixels[offset + 0]);
            }

            if (padding.Length > 0)
                output.Write(padding);
        }
    }

    private static WorldTerrainVisualSnapshot CreateEmptySnapshot(int width, int height)
    {
        byte[] rgbaPixels = new byte[checked(width * height * 4)];
        for (int index = 0; index < rgbaPixels.Length; index += 4)
        {
            rgbaPixels[index + 0] = 15;
            rgbaPixels[index + 1] = 18;
            rgbaPixels[index + 2] = 22;
            rgbaPixels[index + 3] = 255;
        }

        return new WorldTerrainVisualSnapshot(width, height, 0, ComputeHash(width, height, rgbaPixels), rgbaPixels);
    }

    private static (byte r, byte g, byte b) GetPreviewColor(float normalized)
    {
        normalized = Math.Clamp(normalized, 0f, 1f);

        float r;
        float g;
        float b;
        if (normalized < 0.25f)
        {
            float t = normalized / 0.25f;
            r = 0.08f + (t * 0.10f);
            g = 0.14f + (t * 0.30f);
            b = 0.24f + (t * 0.26f);
        }
        else if (normalized < 0.55f)
        {
            float t = (normalized - 0.25f) / 0.30f;
            r = 0.18f + (t * 0.24f);
            g = 0.44f + (t * 0.28f);
            b = 0.50f - (t * 0.24f);
        }
        else if (normalized < 0.80f)
        {
            float t = (normalized - 0.55f) / 0.25f;
            r = 0.42f + (t * 0.30f);
            g = 0.72f - (t * 0.18f);
            b = 0.26f - (t * 0.12f);
        }
        else
        {
            float t = (normalized - 0.80f) / 0.20f;
            r = 0.72f + (t * 0.22f);
            g = 0.54f + (t * 0.24f);
            b = 0.14f + (t * 0.24f);
        }

        return ((byte)Math.Clamp((int)MathF.Round(r * 255f), 0, 255), (byte)Math.Clamp((int)MathF.Round(g * 255f), 0, 255), (byte)Math.Clamp((int)MathF.Round(b * 255f), 0, 255));
    }

    private static string ComputeHash(int width, int height, byte[] rgbaPixels)
    {
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> metadata = stackalloc byte[8];
        BinaryPrimitives.WriteInt32LittleEndian(metadata[0..4], width);
        BinaryPrimitives.WriteInt32LittleEndian(metadata[4..8], height);
        hash.AppendData(metadata);
        hash.AppendData(rgbaPixels);
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }
}