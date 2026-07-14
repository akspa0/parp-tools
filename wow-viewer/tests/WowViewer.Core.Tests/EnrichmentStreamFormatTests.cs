using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public class EnrichmentStreamFormatTests
{
    [Fact]
    public void WriteRead_RoundTripThreeEntries_ReturnsSamePaths()
    {
        using var ms = new MemoryStream();
        var writer = new EnrichmentStreamWriter(ms);

        writer.WriteEntry(MakeM2Entry("World/M2/Peasant.m2"));
        writer.WriteEntry(MakeWmoEntry("World/WMO/House.wmo"));
        writer.WriteEntry(MakeBlpEntry("Tileset/Generic/Dirt.blp"));
        writer.WriteEnds();

        ms.Position = 0;
        var reader = new EnrichmentStreamReader(ms);
        uint version = reader.ReadHeader();

        Assert.Equal(1u, version);

        var entries = reader.ReadEntries().ToList();
        Assert.Equal(3, entries.Count);
        Assert.Equal("World/M2/Peasant.m2", entries[0].CanonicalPath);
        Assert.Equal(AssetKind.M2, entries[0].Kind);
        Assert.Equal("World/WMO/House.wmo", entries[1].CanonicalPath);
        Assert.Equal(AssetKind.Wmo, entries[1].Kind);
        Assert.Equal("Tileset/Generic/Dirt.blp", entries[2].CanonicalPath);
        Assert.Equal(AssetKind.Blp, entries[2].Kind);

        reader.ReadEnds();
    }

    [Fact]
    public void WriteRead_EmptyStream_NoEntries()
    {
        using var ms = new MemoryStream();
        var writer = new EnrichmentStreamWriter(ms);
        writer.WriteEnds();

        ms.Position = 0;
        var reader = new EnrichmentStreamReader(ms);
        reader.ReadHeader();
        var entries = reader.ReadEntries().ToList();
        Assert.Empty(entries);
    }

    [Fact]
    public void WriteRead_SamePathHundredTimes_AllDistinct()
    {
        using var ms = new MemoryStream();
        var writer = new EnrichmentStreamWriter(ms);

        for (int i = 0; i < 100; i++)
        {
            // Each entry has a unique array value so we can confirm they're
            // all present even though paths are the same.
            var array = new EnrichmentArray("value", [1], typeof(int),
                BitConverter.GetBytes(i));
            var entry = new EnrichmentEntry(
                "World/M2/SameModel.m2", AssetKind.M2, 0, [array]);
            writer.WriteEntry(entry);
        }
        writer.WriteEnds();

        ms.Position = 0;
        var reader = new EnrichmentStreamReader(ms);
        reader.ReadHeader();
        var entries = reader.ReadEntries().ToList();
        Assert.Equal(100, entries.Count);
        Assert.All(entries, e => Assert.Equal("World/M2/SameModel.m2", e.CanonicalPath));
    }

    [Fact]
    public void WriteRead_SingleArray_RoundTripsCorrectData()
    {
        using var ms = new MemoryStream();
        var writer = new EnrichmentStreamWriter(ms);

        // Write a small 2x3 float array
        float[] data = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f];
        byte[] flattened = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, flattened, 0, flattened.Length);
        var array = new EnrichmentArray("test_float_array", [2, 3], typeof(float), flattened);
        var entry = new EnrichmentEntry("Test/Asset.m2", AssetKind.M2, 0, [array]);
        writer.WriteEntry(entry);
        writer.WriteEnds();

        ms.Position = 0;
        var reader = new EnrichmentStreamReader(ms);
        reader.ReadHeader();
        var entries = reader.ReadEntries().ToList();
        Assert.Single(entries);

        var readArray = entries[0].Arrays[0];
        Assert.Equal("test_float_array", readArray.Name);
        Assert.Equal(2, readArray.Rank);
        Assert.Equal([2, 3], readArray.Shape);
        Assert.Equal(typeof(float), readArray.DataType);
        Assert.Equal(flattened, readArray.Data);
    }

    [Fact]
    public void LoadError_RoundTrips()
    {
        using var ms = new MemoryStream();
        var writer = new EnrichmentStreamWriter(ms);

        // Entry with load_error = 1 (failed decode)
        var entry = new EnrichmentEntry(
            "World/M2/Corrupt.m2", AssetKind.M2, 1, []);
        writer.WriteEntry(entry);
        writer.WriteEnds();

        ms.Position = 0;
        var reader = new EnrichmentStreamReader(ms);
        reader.ReadHeader();
        var entries = reader.ReadEntries().ToList();
        Assert.Single(entries);
        Assert.Equal(1, entries[0].LoadError);
        Assert.Empty(entries[0].Arrays);
    }

    [Fact]
    public void WriteRead_BlpTexture_RoundTrips()
    {
        using var ms = new MemoryStream();
        var writer = new EnrichmentStreamWriter(ms);

        // Simulate a small 4x4 BLP texture
        int w = 4, h = 4;
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = (byte)(i % 256);
        int[] shape = [4, 4]; // int32 shape packed as bytes
        byte[] shapeBytes = new byte[8];
        Buffer.BlockCopy(shape, 0, shapeBytes, 0, 8);

        var arrays = new List<EnrichmentArray>
        {
            new("texture_rgb", [h, w, 3], typeof(byte), rgb),
            new("texture_shape", [2], typeof(int), shapeBytes),
        };

        var entry = new EnrichmentEntry(
            "Tileset/Generic/Dirt.blp", AssetKind.Blp, 0, arrays);
        writer.WriteEntry(entry);
        writer.WriteEnds();

        ms.Position = 0;
        var reader = new EnrichmentStreamReader(ms);
        reader.ReadHeader();
        var entries = reader.ReadEntries().ToList();
        Assert.Single(entries);
        Assert.Equal(AssetKind.Blp, entries[0].Kind);

        // Check arrays
        var arraysRead = entries[0].Arrays;
        Assert.Equal(2, arraysRead.Count);

        var texRgb = arraysRead[0];
        Assert.Equal("texture_rgb", texRgb.Name);
        Assert.Equal([h, w, 3], texRgb.Shape);
        Assert.Equal(typeof(byte), texRgb.DataType);
        Assert.Equal(rgb, texRgb.Data);

        var texShape = arraysRead[1];
        Assert.Equal("texture_shape", texShape.Name);
        Assert.Equal(typeof(int), texShape.DataType);
    }

    [Fact]
    public void WriteHeaderTwice_NoOp()
    {
        using var ms = new MemoryStream();
        var writer = new EnrichmentStreamWriter(ms);
        writer.WriteHeader();
        writer.WriteHeader(); // Should be a no-op
        writer.WriteEnds();

        ms.Position = 0;
        var reader = new EnrichmentStreamReader(ms);
        uint version = reader.ReadHeader();
        Assert.Equal(1u, version);
        reader.ReadEnds();
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private static EnrichmentEntry MakeM2Entry(string path)
    {
        var arrays = new List<EnrichmentArray>();
        // vertices: 8 cube vertices
        float[] verts = [-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1];
        arrays.Add(MakeArray("vertices", [8, 3], typeof(float), verts));
        // normals: placeholder
        float[] norms = Enumerable.Repeat(0f, 24).ToArray();
        arrays.Add(MakeArray("normals", [8, 3], typeof(float), norms));
        // triangles: 12 triangles (36 indices)
        int[] tris = [0, 1, 2, 0, 2, 3, 1, 5, 6, 1, 6, 2, 5, 4, 7, 5, 7, 6, 4, 0, 3, 4, 3, 7, 3, 2, 6, 3, 6, 7, 4, 5, 1, 4, 1, 0];
        arrays.Add(MakeArray("triangles", [12, 3], typeof(int), tris));
        // bounds
        float[] bounds = [-1, -1, -1, 1, 1, 1];
        arrays.Add(MakeArray("bounds", [2, 3], typeof(float), bounds));
        return new EnrichmentEntry(path, AssetKind.M2, 0, arrays);
    }

    private static EnrichmentEntry MakeWmoEntry(string path)
    {
        var arrays = new List<EnrichmentArray>();
        float[] verts = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
        arrays.Add(MakeArray("vertices", [8, 3], typeof(float), verts));
        int[] tris = [0, 1, 2, 0, 2, 3];
        arrays.Add(MakeArray("triangles", [2, 3], typeof(int), tris));
        arrays.Add(MakeArray("flags", [1], typeof(uint), new uint[] { 0 }));
        arrays.Add(MakeArray("version", [1], typeof(uint), new uint[] { 17 }));
        return new EnrichmentEntry(path, AssetKind.Wmo, 0, arrays);
    }

    private static EnrichmentEntry MakeBlpEntry(string path)
    {
        return new EnrichmentEntry(path, AssetKind.Blp, 0, []);
    }

    private static EnrichmentArray MakeArray(string name, int[] shape, Type elementType, Array values)
    {
        byte[] data = new byte[Buffer.ByteLength(values)];
        Buffer.BlockCopy(values, 0, data, 0, data.Length);
        return new EnrichmentArray(name, shape, elementType, data);
    }
}