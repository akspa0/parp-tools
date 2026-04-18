using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxMaterialReaderTests
{
    [Fact]
    public void Read_SyntheticClassicMaterialPayload_AssignsLayerTracks()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticMaterialPayload",
            materials:
            [
                (3,
                    [
                        (2u, 0x42u, 7, 5, 1, 0.6f, 1.25f,
                            ("KMTE", 1u, -1, [(0, 0.5f, null, null), (100, 2.0f, null, null)]),
                            ("KMTA", 1u, 4, [(0, 1.0f, null, null), (100, 0.25f, null, null)]),
                            ("KMTF", 1u, -1, [(0, 2, null, null), (100, 6, null, null)]))
                    ])
            ]);

        using MemoryStream stream = new(bytes);
        MdxMaterialFile materialFile = MdxMaterialReader.Read(stream, "synthetic_material_payload.mdx");

        Assert.Equal("SyntheticMaterialPayload", materialFile.ModelName);
        Assert.Equal(1, materialFile.MaterialCount);

        MdxMaterial material = materialFile.Materials[0];
        Assert.Equal(3, material.PriorityPlane);
        Assert.Single(material.Layers);

        MdxMaterialLayer layer = material.Layers[0];
        Assert.Equal(2u, layer.BlendMode);
        Assert.Equal(0x42u, layer.Flags);
        Assert.Equal(7, layer.TextureId);
        Assert.Equal(5, layer.TransformId);
        Assert.Equal(1, layer.CoordId);
        Assert.Equal(0.6f, layer.StaticAlpha, 5);
        Assert.Equal(1.25f, layer.StaticEmissiveGain, 5);
        Assert.NotNull(layer.EmissiveTrack);
        Assert.Equal(2.0f, layer.EmissiveTrack!.Keys[1].Value, 5);
        Assert.NotNull(layer.AlphaTrack);
        Assert.Equal(4, layer.AlphaTrack!.GlobalSequenceId);
        Assert.Equal(0.25f, layer.AlphaTrack.Keys[1].Value, 5);
        Assert.NotNull(layer.TextureLayerTrack);
        Assert.Equal(6, layer.TextureLayerTrack!.Keys[1].Value);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        IReadOnlyList<(int PriorityPlane, IReadOnlyList<(uint BlendMode, uint Flags, int TextureId, int TransformId, int CoordId, float StaticAlpha, float StaticEmissiveGain, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? EmissiveTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? AlphaTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, int Value, int? InTangent, int? OutTangent)> Keys)? TextureLayerTrack)> Layers)> materials)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName)));
        bytes.AddRange(CreateChunk("MTLS", CreateMtlsPayload(materials)));
        return [.. bytes];
    }

    private static byte[] CreateMtlsPayload(IReadOnlyList<(int PriorityPlane, IReadOnlyList<(uint BlendMode, uint Flags, int TextureId, int TransformId, int CoordId, float StaticAlpha, float StaticEmissiveGain, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? EmissiveTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? AlphaTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, int Value, int? InTangent, int? OutTangent)> Keys)? TextureLayerTrack)> Layers)> materials)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)materials.Count));
        payload.AddRange(CreateUInt32Payload(0));

        foreach ((int priorityPlane, IReadOnlyList<(uint BlendMode, uint Flags, int TextureId, int TransformId, int CoordId, float StaticAlpha, float StaticEmissiveGain, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? EmissiveTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? AlphaTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, int Value, int? InTangent, int? OutTangent)> Keys)? TextureLayerTrack)> layers) in materials)
        {
            List<byte> materialPayload = [];
            materialPayload.AddRange(CreateInt32Payload(priorityPlane));
            materialPayload.AddRange(CreateUInt32Payload((uint)layers.Count));

            foreach ((uint blendMode, uint flags, int textureId, int transformId, int coordId, float staticAlpha, float staticEmissiveGain, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? emissiveTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? alphaTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, int Value, int? InTangent, int? OutTangent)> Keys)? textureLayerTrack) in layers)
            {
                List<byte> layerPayload = [];
                layerPayload.AddRange(CreateUInt32Payload(blendMode));
                layerPayload.AddRange(CreateUInt32Payload(flags));
                layerPayload.AddRange(CreateInt32Payload(textureId));
                layerPayload.AddRange(CreateInt32Payload(transformId));
                layerPayload.AddRange(CreateInt32Payload(coordId));
                layerPayload.AddRange(CreateSinglePayload(staticAlpha));
                layerPayload.AddRange(CreateSinglePayload(staticEmissiveGain));
                if (emissiveTrack is not null)
                    layerPayload.AddRange(CreateScalarTrack(emissiveTrack.Value.Tag, emissiveTrack.Value.InterpolationType, emissiveTrack.Value.GlobalSequenceId, emissiveTrack.Value.Keys));
                if (alphaTrack is not null)
                    layerPayload.AddRange(CreateScalarTrack(alphaTrack.Value.Tag, alphaTrack.Value.InterpolationType, alphaTrack.Value.GlobalSequenceId, alphaTrack.Value.Keys));
                if (textureLayerTrack is not null)
                    layerPayload.AddRange(CreateIntTrack(textureLayerTrack.Value.Tag, textureLayerTrack.Value.InterpolationType, textureLayerTrack.Value.GlobalSequenceId, textureLayerTrack.Value.Keys));

                materialPayload.AddRange(CreateSizedPayload(layerPayload));
            }

            payload.AddRange(CreateSizedPayload(materialPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateScalarTrack(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> keys)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)keys.Count));
        payload.AddRange(CreateUInt32Payload(interpolationType));
        payload.AddRange(CreateInt32Payload(globalSequenceId));
        foreach ((int time, float value, float? inTangent, float? outTangent) in keys)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateSinglePayload(value));
            if (interpolationType >= 2u)
            {
                payload.AddRange(CreateSinglePayload(inTangent ?? 0.0f));
                payload.AddRange(CreateSinglePayload(outTangent ?? 0.0f));
            }
        }

        return [.. payload];
    }

    private static byte[] CreateIntTrack(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, int Value, int? InTangent, int? OutTangent)> keys)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)keys.Count));
        payload.AddRange(CreateUInt32Payload(interpolationType));
        payload.AddRange(CreateInt32Payload(globalSequenceId));
        foreach ((int time, int value, int? inTangent, int? outTangent) in keys)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateInt32Payload(value));
            if (interpolationType >= 2u)
            {
                payload.AddRange(CreateInt32Payload(inTangent ?? 0));
                payload.AddRange(CreateInt32Payload(outTangent ?? 0));
            }
        }

        return [.. payload];
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        List<byte> chunk = [];
        chunk.AddRange(Encoding.ASCII.GetBytes(id));
        chunk.AddRange(CreateUInt32Payload((uint)payload.Length));
        chunk.AddRange(payload);
        return [.. chunk];
    }

    private static byte[] CreateSizedPayload(List<byte> payload)
    {
        List<byte> sized = [];
        sized.AddRange(CreateUInt32Payload((uint)(payload.Count + 4)));
        sized.AddRange(payload);
        return [.. sized];
    }

    private static byte[] CreateModlPayload(string modelName)
    {
        byte[] payload = new byte[0x50];
        Encoding.ASCII.GetBytes(modelName[..Math.Min(modelName.Length, payload.Length)]).CopyTo(payload, 0);
        return payload;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] payload = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(payload, value);
        return payload;
    }

    private static byte[] CreateInt32Payload(int value)
    {
        byte[] payload = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(payload, value);
        return payload;
    }

    private static byte[] CreateSinglePayload(float value)
    {
        byte[] payload = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(payload, value);
        return payload;
    }
}
