using System.Buffers.Binary;
using WowViewer.Core.IO.Wc3;

namespace WowViewer.Core.Tests;

public sealed class Wc3MdxAdapterTests
{
    [Fact]
    public void Version800LaysMaterialIsAdaptedWithoutChangingSourceBytes()
    {
        byte[] model = ModelWithMaterial();
        byte[] original = model.ToArray();
        var parsed = Wc3MdxAdapter.Read(model);
        Assert.Equal(800u, parsed.Version);
        var material = Assert.Single(parsed.Materials);
        Assert.Equal(7, material.PriorityPlane);
        var layer = Assert.Single(material.Layers);
        Assert.Equal(2, layer.TextureId);
        Assert.Equal(0.75f, layer.StaticAlpha);
        Assert.Equal(original, model);
        var preview = Wc3AssetPreviewReader.Read(model, "model.mdx");
        Assert.Null(preview.Error);
        Assert.NotNull(preview.Model);
    }

    [Fact]
    public void InvalidMaterialAndChunkBoundsAreRejected()
    {
        byte[] model = ModelWithMaterial();
        // MTLS begins at byte 16; payload starts at 24, first inclusive size.
        BinaryPrimitives.WriteUInt32LittleEndian(model.AsSpan(24), uint.MaxValue);
        Assert.ThrowsAny<Exception>(() => Wc3MdxAdapter.Read(model));
        Assert.Throws<InvalidDataException>(() => Wc3MdxAdapter.Read(ModelWithMaterial()[..^1]));
    }

    private static byte[] ModelWithMaterial()
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream);
        writer.Write("MDLX"u8); writer.Write("VERS"u8); writer.Write(4); writer.Write(800);
        writer.Write("MTLS"u8); writer.Write(48);
        writer.Write(48); writer.Write(7); writer.Write(0); writer.Write("LAYS"u8); writer.Write(1);
        writer.Write(28); writer.Write(2); writer.Write(1); writer.Write(2);
        writer.Write(-1); writer.Write(0); writer.Write(0.75f);
        return stream.ToArray();
    }
}
