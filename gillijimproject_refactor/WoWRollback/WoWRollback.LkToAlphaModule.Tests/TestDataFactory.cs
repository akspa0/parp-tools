using System;
using System.IO;
using System.Text;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Tests;

/// <summary>
/// Factory for creating test data for managed builder tests.
/// </summary>
public static class TestDataFactory
{
    /// <summary>
    /// Creates a valid LkMcnkSource with synthetic test data.
    /// </summary>
    public static LkMcnkSource CreateTestMcnkSource(int indexX = 0, int indexY = 0, int layerCount = 2)
    {
        var source = new LkMcnkSource
        {
            IndexX = indexX,
            IndexY = indexY,
            Flags = 0x00000001, // Some test flags
            AreaId = 1234,
            HolesLowRes = 0,
            Radius = 100.0f,
            DoodadRefCount = 0,
            MapObjectRefs = 0,
            NoEffectDoodad = 0,
            OffsLiquid = 0,
            OffsSndEmitters = 0,
            SndEmitterCount = 0,
            McvtRaw = CreateTestMcvt(),
            McnrRaw = CreateTestMcnr(),
            MclyRaw = CreateTestMcly(layerCount),
            McrfRaw = Array.Empty<byte>(),
            McshRaw = Array.Empty<byte>(),
            McseRaw = Array.Empty<byte>()
        };

        // Add test alpha layers
        for (int i = 0; i < layerCount; i++)
        {
            source.AlphaLayers.Add(new LkMcnkAlphaLayer
            {
                LayerIndex = i,
                OverrideFlags = null, // Use flags from MCLY
                ColumnMajorAlpha = CreateTestAlphaLayer(i)
            });
        }

        return source;
    }

    /// <summary>
    /// Creates a test MCVT chunk (145 vertex heights).
    /// </summary>
    public static byte[] CreateTestMcvt()
    {
        const int vertexCount = 145; // 9x9 outer + 8x8 inner
        const int mcvtDataSize = vertexCount * sizeof(float);

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write MCVT chunk header (reversed FourCC)
        writer.Write(Encoding.ASCII.GetBytes("TVCM")); // "MCVT" reversed
        writer.Write(mcvtDataSize);

        // Write test heights (simple gradient pattern)
        for (int i = 0; i < vertexCount; i++)
        {
            float height = 100.0f + (i * 0.5f); // Gradual slope
            writer.Write(height);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a test MCNR chunk (145 normals).
    /// </summary>
    public static byte[] CreateTestMcnr()
    {
        const int normalCount = 145;
        const int mcnrDataSize = normalCount * 3; // 3 bytes per normal

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(Encoding.ASCII.GetBytes("RNCM")); // "MCNR" reversed
        writer.Write(mcnrDataSize);

        // Write test normals (pointing up)
        for (int i = 0; i < normalCount; i++)
        {
            writer.Write((sbyte)0);   // X
            writer.Write((sbyte)127); // Y (up)
            writer.Write((sbyte)0);   // Z
        }

        // Pad to even size
        if (ms.Length % 2 == 1)
            writer.Write((byte)0);

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a test MCLY chunk with specified number of layers.
    /// Returns ONLY the data portion (no chunk header), as expected by LkMcnkSource.MclyRaw.
    /// </summary>
    public static byte[] CreateTestMcly(int layerCount)
    {
        const int layerEntrySize = 16;

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Note: Do NOT write chunk header here - MclyRaw is just the data
        uint alphaOffset = 0;
        for (int i = 0; i < layerCount; i++)
        {
            writer.Write((uint)(100 + i)); // TextureId
            writer.Write((uint)0x100);     // Flags (use_alpha_map)
            writer.Write(alphaOffset);      // OffsetInMcal
            writer.Write((uint)0);         // EffectId

            // Each layer uses 4096 bytes (64x64)
            alphaOffset += 4096;
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Creates a test alpha layer (64x64 column-major format).
    /// </summary>
    public static byte[] CreateTestAlphaLayer(int layerIndex)
    {
        byte[] alpha = new byte[4096]; // 64x64

        // Create a test pattern based on layer index
        for (int i = 0; i < 4096; i++)
        {
            // Simple gradient pattern
            alpha[i] = (byte)((i + layerIndex * 50) % 256);
        }

        return alpha;
    }

    /// <summary>
    /// Creates a complete test LkAdtSource with 256 MCNK chunks.
    /// </summary>
    public static LkAdtSource CreateTestAdtSource(string mapName = "TestMap", int tileX = 0, int tileY = 0)
    {
        var source = new LkAdtSource
        {
            MapName = mapName,
            TileX = tileX,
            TileY = tileY
        };

        // Create 256 MCNK chunks (16x16 grid)
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                source.Mcnks.Add(CreateTestMcnkSource(x, y, layerCount: 1));
            }
        }

        return source;
    }

    /// <summary>
    /// Validates that a byte array contains a valid chunk header.
    /// </summary>
    public static bool HasValidChunkHeader(byte[] data, string expectedFourCC)
    {
        if (data.Length < 8) return false;

        string fourCC = Encoding.ASCII.GetString(data, 0, 4);
        int size = BitConverter.ToInt32(data, 4);

        // FourCC should be reversed on disk
        string reversedExpected = new string(expectedFourCC.Reverse().ToArray());

        return fourCC == reversedExpected && size > 0 && size <= data.Length - 8;
    }

    /// <summary>
    /// Extracts chunk data (without header) from a chunk byte array.
    /// </summary>
    public static byte[] ExtractChunkData(byte[] chunkBytes)
    {
        if (chunkBytes.Length < 8) return Array.Empty<byte>();

        int size = BitConverter.ToInt32(chunkBytes, 4);
        if (size <= 0 || size > chunkBytes.Length - 8) return Array.Empty<byte>();

        byte[] data = new byte[size];
        Array.Copy(chunkBytes, 8, data, 0, size);
        return data;
    }
}
