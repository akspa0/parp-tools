using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxGeosetAnimationReaderTests
{
    [Fact]
    public void Read_SyntheticClassicGeoa_ProducesExpectedPayload()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticGeosetAnimationPayload",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            extraChunks:
            [
                CreateChunk("GEOA", CreateGeoaPayload(
                [
                    (0u, 0.75f, new Vector3(1.0f, 0.5f, 0.25f), 0x1u,
                        (2u, 4, [(10, 0.25f, 0.1f, 0.2f), (40, 1.0f, 0.8f, 0.9f)]),
                        (1u, -1, [(15, new Vector3(0.2f, 0.4f, 0.6f), null, null)])),
                    (uint.MaxValue, 1.0f, Vector3.One, 0x0u, null, null),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxGeosetAnimationFile geosetAnimationFile = MdxGeosetAnimationReader.Read(stream, "synthetic_geoa_payload.mdx");

        Assert.Equal((uint)1300, geosetAnimationFile.Version);
        Assert.Equal("SyntheticGeosetAnimationPayload", geosetAnimationFile.ModelName);
        Assert.Equal(2, geosetAnimationFile.GeosetAnimationCount);

        MdxGeosetAnimation first = geosetAnimationFile.GeosetAnimations[0];
        Assert.Equal(0u, first.GeosetId);
        Assert.Equal(0.75f, first.StaticAlpha);
        Assert.Equal(new Vector3(1.0f, 0.5f, 0.25f), first.StaticColor);
        Assert.True(first.UsesStaticColor);
        Assert.NotNull(first.AlphaTrack);
        Assert.Equal("KGAO", first.AlphaTrack!.Tag);
        Assert.Equal(MdxTrackInterpolationType.Hermite, first.AlphaTrack.InterpolationType);
        Assert.Equal(4, first.AlphaTrack.GlobalSequenceId);
        Assert.Equal(2, first.AlphaTrack.KeyCount);
        Assert.Equal(0.25f, first.AlphaTrack.Keys[0].Value, 4);
        Assert.Equal(0.1f, first.AlphaTrack.Keys[0].InTangent ?? float.NaN, 4);
        Assert.Equal(0.2f, first.AlphaTrack.Keys[0].OutTangent ?? float.NaN, 4);
        Assert.NotNull(first.ColorTrack);
        Assert.Equal("KGAC", first.ColorTrack!.Tag);
        Assert.Equal(MdxTrackInterpolationType.Linear, first.ColorTrack.InterpolationType);
        Assert.Equal(-1, first.ColorTrack.GlobalSequenceId);
        Assert.Single(first.ColorTrack.Keys);
        Assert.Equal(new Vector3(0.2f, 0.4f, 0.6f), first.ColorTrack.Keys[0].Value);

        MdxGeosetAnimation second = geosetAnimationFile.GeosetAnimations[1];
        Assert.Equal(uint.MaxValue, second.GeosetId);
        Assert.Null(second.AlphaTrack);
        Assert.Null(second.ColorTrack);
    }

    private static byte[] CreateMdxBytes(uint version, string modelName, Vector3 boundsMin, Vector3 boundsMax, IReadOnlyList<byte[]> extraChunks)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName, boundsMin, boundsMax)));
        foreach (byte[] chunk in extraChunks)
            bytes.AddRange(chunk);

        return [.. bytes];
    }

    private static byte[] CreateGeoaPayload(IReadOnlyList<(uint GeosetId, float StaticAlpha, Vector3 StaticColor, uint Flags, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? AlphaTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ColorTrack)> geosetAnimations)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)geosetAnimations.Count));

        foreach ((uint geosetId, float staticAlpha, Vector3 staticColor, uint flags, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? alphaTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? colorTrack) in geosetAnimations)
        {
            List<byte> geosetAnimationPayload = [];
            geosetAnimationPayload.AddRange(CreateUInt32Payload(geosetId));
            geosetAnimationPayload.AddRange(CreateSinglePayload(staticAlpha));
            geosetAnimationPayload.AddRange(CreateVector3Payload(staticColor));
            geosetAnimationPayload.AddRange(CreateUInt32Payload(flags));

            if (alphaTrack is not null)
                geosetAnimationPayload.AddRange(CreateScalarTrack("KGAO", alphaTrack.Value.InterpolationType, alphaTrack.Value.GlobalSequenceId, alphaTrack.Value.Keys));

            if (colorTrack is not null)
                geosetAnimationPayload.AddRange(CreateColorTrack("KGAC", colorTrack.Value.InterpolationType, colorTrack.Value.GlobalSequenceId, colorTrack.Value.Keys));

            payload.AddRange(CreateSizedPayload(geosetAnimationPayload));
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

    private static byte[] CreateColorTrack(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> keys)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)keys.Count));
        payload.AddRange(CreateUInt32Payload(interpolationType));
        payload.AddRange(CreateInt32Payload(globalSequenceId));

        foreach ((int time, Vector3 value, Vector3? inTangent, Vector3? outTangent) in keys)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateVector3Payload(value));
            if (interpolationType >= 2u)
            {
                payload.AddRange(CreateVector3Payload(inTangent ?? Vector3.Zero));
                payload.AddRange(CreateVector3Payload(outTangent ?? Vector3.Zero));
            }
        }

        return [.. payload];
    }

    private static byte[] CreateModlPayload(string modelName, Vector3 boundsMin, Vector3 boundsMax)
    {
        byte[] payload = new byte[0x6C];
        WriteFixedAscii(payload, 0, 0x50, modelName);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x50, 4), boundsMin.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x54, 4), boundsMin.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x58, 4), boundsMin.Z);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x5C, 4), boundsMax.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x60, 4), boundsMax.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0x64, 4), boundsMax.Z);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x68, 4), 0u);
        return payload;
    }

    private static byte[] CreateChunk(string tag, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Encoding.ASCII.GetBytes(tag, bytes.AsSpan(0, 4));
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), (uint)payload.Length);
        payload.CopyTo(bytes.AsSpan(8));
        return bytes;
    }

    private static byte[] CreateSizedPayload(List<byte> payload)
    {
        byte[] bytes = new byte[4 + payload.Count];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), (uint)(4 + payload.Count));
        payload.CopyTo(bytes, 4);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateInt32Payload(int value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateSinglePayload(float value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateVector3Payload(Vector3 value)
    {
        byte[] bytes = new byte[12];
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0, 4), value.X);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4, 4), value.Y);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(8, 4), value.Z);
        return bytes;
    }

    private static void WriteFixedAscii(byte[] buffer, int offset, int length, string value)
    {
        int count = Math.Min(length, value.Length);
        for (int index = 0; index < count; index++)
            buffer[offset + index] = (byte)value[index];
    }
}
