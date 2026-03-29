using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxTextureAnimationReaderTests
{
    [Fact]
    public void Read_SyntheticClassicTxan_ProducesExpectedPayload()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticTextureAnimationPayload",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            extraChunks:
            [
                CreateChunk("TXAN", CreateTextureAnimationPayload(
                [
                    (CreateVector3Track("KTAT", 2u, 4, [(10, new Vector3(1.0f, 2.0f, 3.0f), new Vector3(0.1f, 0.2f, 0.3f), new Vector3(0.4f, 0.5f, 0.6f))]), null, null),
                    (null, CreateQuaternionTrack("KTAR", 2u, 7, [(15, (1u, 2u), (3u, 4u), (5u, 6u))]), CreateVector3Track("KTAS", 1u, 5, [(20, new Vector3(7.0f, 8.0f, 9.0f), null, null)])),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxTextureAnimationFile textureAnimationFile = MdxTextureAnimationReader.Read(stream, "synthetic_txan_payload.mdx");

        Assert.Equal((uint)1300, textureAnimationFile.Version);
        Assert.Equal("SyntheticTextureAnimationPayload", textureAnimationFile.ModelName);
        Assert.Equal(2, textureAnimationFile.TextureAnimationCount);

        MdxTextureAnimation first = textureAnimationFile.TextureAnimations[0];
        Assert.Equal(0, first.Index);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal("KTAT", first.TranslationTrack!.Tag);
        Assert.Equal(MdxTrackInterpolationType.Hermite, first.TranslationTrack.InterpolationType);
        Assert.Equal(4, first.TranslationTrack.GlobalSequenceId);
        Assert.Single(first.TranslationTrack.Keys);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), first.TranslationTrack.Keys[0].Value);
        Assert.Equal(new Vector3(0.1f, 0.2f, 0.3f), first.TranslationTrack.Keys[0].InTangent);
        Assert.Equal(new Vector3(0.4f, 0.5f, 0.6f), first.TranslationTrack.Keys[0].OutTangent);
        Assert.Null(first.RotationTrack);
        Assert.Null(first.ScalingTrack);

        MdxTextureAnimation second = textureAnimationFile.TextureAnimations[1];
        Assert.NotNull(second.RotationTrack);
        Assert.Equal("KTAR", second.RotationTrack!.Tag);
        Assert.Equal(MdxTrackInterpolationType.Hermite, second.RotationTrack.InterpolationType);
        Assert.Equal(DecompressQuaternion(1u, 2u), second.RotationTrack.Keys[0].Value);
        Assert.Equal(DecompressQuaternion(3u, 4u), second.RotationTrack.Keys[0].InTangent);
        Assert.Equal(DecompressQuaternion(5u, 6u), second.RotationTrack.Keys[0].OutTangent);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal("KTAS", second.ScalingTrack!.Tag);
        Assert.Equal(MdxTrackInterpolationType.Linear, second.ScalingTrack.InterpolationType);
        Assert.Equal(new Vector3(7.0f, 8.0f, 9.0f), second.ScalingTrack.Keys[0].Value);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithoutTxan_ReturnsNoTextureAnimations()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxTextureAnimationFile textureAnimationFile = MdxTextureAnimationReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", textureAnimationFile.ModelName);
        Assert.Equal(0, textureAnimationFile.TextureAnimationCount);
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithTxan_ProducesPayloadSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using var catalog = new MpqArchiveCatalog();
        _ = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);

        string? virtualPath = MdxTextureAnimationTestPaths.Standard060TxanCandidates.FirstOrDefault(catalog.FileExists);
        if (virtualPath is null)
            return;

        byte[]? bytes = catalog.ReadFile(virtualPath);
        Assert.NotNull(bytes);

        using MemoryStream detectionStream = new(bytes!);
        WowFileDetection detection = WowFileDetector.Detect(detectionStream, virtualPath);
        Assert.Equal(WowFileKind.Mdx, detection.Kind);

        using MemoryStream animationStream = new(bytes!);
        MdxTextureAnimationFile textureAnimationFile = MdxTextureAnimationReader.Read(animationStream, virtualPath);

        Assert.True(textureAnimationFile.TextureAnimationCount > 0);
        Assert.Contains(textureAnimationFile.TextureAnimations, animation => animation.HasTranslationTrack || animation.HasRotationTrack || animation.HasScalingTrack);

        if (string.Equals(virtualPath, MdxTextureAnimationTestPaths.Standard060AirElementalMdxVirtualPath, StringComparison.OrdinalIgnoreCase))
        {
            Assert.Equal("AirElemental", textureAnimationFile.ModelName);
            Assert.Equal(12, textureAnimationFile.TextureAnimationCount);
            MdxTextureAnimation first = textureAnimationFile.TextureAnimations[0];
            Assert.NotNull(first.TranslationTrack);
            Assert.Equal("KTAT", first.TranslationTrack!.Tag);
            Assert.Equal(MdxTrackInterpolationType.Bezier, first.TranslationTrack.InterpolationType);
            Assert.Equal(1, first.TranslationTrack.GlobalSequenceId);
            Assert.Equal(2, first.TranslationTrack.KeyCount);
            Assert.Equal(0, first.TranslationTrack.FirstKeyTime);
            Assert.Equal(1245, first.TranslationTrack.LastKeyTime);
            Assert.Equal(0.0f, first.TranslationTrack.Keys[0].Value.X, 4);
            Assert.Equal(0.9999968f, first.TranslationTrack.Keys[1].Value.X, 4);
            Assert.Null(first.RotationTrack);
            Assert.Null(first.ScalingTrack);
        }
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

    private static byte[] CreateTextureAnimationPayload(IReadOnlyList<(byte[]? TranslationTrack, byte[]? RotationTrack, byte[]? ScalingTrack)> animations)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)animations.Count));

        foreach ((byte[]? translationTrack, byte[]? rotationTrack, byte[]? scalingTrack) in animations)
        {
            List<byte> entryPayload = [];
            if (translationTrack is not null)
                entryPayload.AddRange(translationTrack);

            if (rotationTrack is not null)
                entryPayload.AddRange(rotationTrack);

            if (scalingTrack is not null)
                entryPayload.AddRange(scalingTrack);

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateVector3Track(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> keys)
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

    private static byte[] CreateQuaternionTrack(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, (uint Data0, uint Data1) Value, (uint Data0, uint Data1)? InTangent, (uint Data0, uint Data1)? OutTangent)> keys)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)keys.Count));
        payload.AddRange(CreateUInt32Payload(interpolationType));
        payload.AddRange(CreateInt32Payload(globalSequenceId));

        foreach ((int time, (uint Data0, uint Data1) value, (uint Data0, uint Data1)? inTangent, (uint Data0, uint Data1)? outTangent) in keys)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateUInt32Payload(value.Data0));
            payload.AddRange(CreateUInt32Payload(value.Data1));
            if (interpolationType >= 2u)
            {
                payload.AddRange(CreateUInt32Payload((inTangent ?? (0u, 0u)).Data0));
                payload.AddRange(CreateUInt32Payload((inTangent ?? (0u, 0u)).Data1));
                payload.AddRange(CreateUInt32Payload((outTangent ?? (0u, 0u)).Data0));
                payload.AddRange(CreateUInt32Payload((outTangent ?? (0u, 0u)).Data1));
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

    private static byte[] CreateVector3Payload(Vector3 value)
    {
        List<byte> payload = [];
        payload.AddRange(CreateSinglePayload(value.X));
        payload.AddRange(CreateSinglePayload(value.Y));
        payload.AddRange(CreateSinglePayload(value.Z));
        return [.. payload];
    }

    private static byte[] CreateSinglePayload(float value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        return bytes;
    }

    private static void WriteFixedAscii(byte[] destination, int offset, int length, string value)
    {
        Encoding.ASCII.GetBytes(value, destination.AsSpan(offset, Math.Min(length, value.Length)));
    }

    private static Quaternion DecompressQuaternion(uint data0, uint data1)
    {
        int xq = ((int)data1) >> 10;
        int yq = ((int)((data1 << 22) | (data0 >> 10))) >> 11;
        int zq = ((int)(data0 << 11)) >> 11;

        const float scaleX = 1.0f / (1 << 21);
        const float scaleYZ = 1.0f / (1 << 20);

        float x = xq * scaleX;
        float y = yq * scaleYZ;
        float z = zq * scaleYZ;
        float s = x * x + y * y + z * z;
        float w = MathF.Abs(s - 1.0f) < scaleYZ ? 0.0f : MathF.Sqrt(MathF.Max(0.0f, 1.0f - s));
        return new Quaternion(x, y, z, w);
    }
}

internal static class MdxTextureAnimationTestPaths
{
    public const string Standard060AirElementalMdxVirtualPath = "creature/airelemental/airelemental.mdx";

    public static readonly string[] Standard060TxanCandidates =
    [
        Standard060AirElementalMdxVirtualPath,
    ];
}