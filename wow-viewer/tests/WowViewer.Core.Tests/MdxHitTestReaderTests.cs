using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxHitTestReaderTests
{
    [Fact]
    public void Read_SyntheticClassicHtst_ProducesExpectedPayload()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticHtstPayload",
            boundsMin: new Vector3(-1.0f, -1.0f, -1.0f),
            boundsMax: new Vector3(1.0f, 1.0f, 1.0f),
            extraChunks:
            [
                CreateChunk("HTST", CreateHitTestShapePayload(
                [
                    ("HitBox", 21, 7, 0x1000u, MdxGeometryShapeType.Box, new Vector3(-1.0f, -2.0f, -3.0f), new Vector3(4.0f, 5.0f, 6.0f), 0.0f, 0.0f,
                        CreateVector3Track(2u, 4, [(10, new Vector3(1.0f, 2.0f, 3.0f), new Vector3(0.1f, 0.2f, 0.3f), new Vector3(0.4f, 0.5f, 0.6f))]), null, null),
                    ("HitSphere", 23, -1, 0x1002u, MdxGeometryShapeType.Sphere, new Vector3(0.25f, 0.5f, 0.75f), Vector3.Zero, 1.25f, 0.0f,
                        null,
                        CreateQuaternionTrack(2u, 7, [(15, (1u, 2u), (3u, 4u), (5u, 6u))]),
                        CreateVector3Track(1u, 5, [(20, new Vector3(7.0f, 8.0f, 9.0f), null, null)])),
                ])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxHitTestFile hitTestFile = MdxHitTestReader.Read(stream, "synthetic_htst_payload.mdx");

        Assert.Equal((uint)1300, hitTestFile.Version);
        Assert.Equal("SyntheticHtstPayload", hitTestFile.ModelName);
        Assert.Equal(2, hitTestFile.ShapeCount);

        MdxHitTestShape box = hitTestFile.Shapes[0];
        Assert.Equal("HitBox", box.Name);
        Assert.Equal(MdxGeometryShapeType.Box, box.ShapeType);
        Assert.Equal(new Vector3(-1.0f, -2.0f, -3.0f), box.Minimum);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), box.Maximum);
        Assert.NotNull(box.TranslationTrack);
        Assert.Equal(MdxTrackInterpolationType.Hermite, box.TranslationTrack!.InterpolationType);
        Assert.Equal(4, box.TranslationTrack.GlobalSequenceId);
        Assert.Single(box.TranslationTrack.Keys);
        Assert.Equal(10, box.TranslationTrack.Keys[0].Time);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), box.TranslationTrack.Keys[0].Value);
        Assert.Equal(new Vector3(0.1f, 0.2f, 0.3f), box.TranslationTrack.Keys[0].InTangent);
        Assert.Equal(new Vector3(0.4f, 0.5f, 0.6f), box.TranslationTrack.Keys[0].OutTangent);

        MdxHitTestShape sphere = hitTestFile.Shapes[1];
        Assert.Equal(MdxGeometryShapeType.Sphere, sphere.ShapeType);
        Assert.Equal(new Vector3(0.25f, 0.5f, 0.75f), sphere.Center);
        Assert.Equal(1.25f, sphere.Radius);
        Assert.NotNull(sphere.RotationTrack);
        Assert.Equal(MdxTrackInterpolationType.Hermite, sphere.RotationTrack!.InterpolationType);
        Assert.Single(sphere.RotationTrack.Keys);
        Assert.Equal(15, sphere.RotationTrack.Keys[0].Time);
        Assert.Equal(DecompressQuaternion(1u, 2u), sphere.RotationTrack.Keys[0].Value);
        Assert.Equal(DecompressQuaternion(3u, 4u), sphere.RotationTrack.Keys[0].InTangent);
        Assert.Equal(DecompressQuaternion(5u, 6u), sphere.RotationTrack.Keys[0].OutTangent);
        Assert.NotNull(sphere.ScalingTrack);
        Assert.Equal(MdxTrackInterpolationType.Linear, sphere.ScalingTrack!.InterpolationType);
        Assert.Equal(new Vector3(7.0f, 8.0f, 9.0f), sphere.ScalingTrack.Keys[0].Value);
    }

    [Fact]
    public void Read_RealAlpha053WispMdx_WithHtst_ProducesExpectedPayloadSignals()
    {
        if (!File.Exists(MdxTestPaths.Alpha053WispMdxPath))
            return;

        MdxHitTestFile hitTestFile = MdxHitTestReader.Read(MdxTestPaths.Alpha053WispMdxPath);

        Assert.Equal("Wisp", hitTestFile.ModelName);
        Assert.Equal(1, hitTestFile.ShapeCount);

        MdxHitTestShape shape = hitTestFile.Shapes[0];
        Assert.Equal("HIT01", shape.Name);
        Assert.Equal(51, shape.ObjectId);
        Assert.Equal(24, shape.ParentId);
        Assert.Equal(0x1002u, shape.Flags);
        Assert.Equal(MdxGeometryShapeType.Sphere, shape.ShapeType);
        Assert.Equal(0.3661754f, shape.Center!.Value.X, 6);
        Assert.Equal(0.008944444f, shape.Center!.Value.Y, 6);
        Assert.Equal(1.889694f, shape.Center!.Value.Z, 6);
        Assert.Equal(0.8333333f, shape.Radius!.Value, 6);
        Assert.Null(shape.TranslationTrack);
        Assert.Null(shape.RotationTrack);
        Assert.Null(shape.ScalingTrack);
    }

    [Fact]
    public void Read_RealStandardArchiveMdx_WithHtst_ProducesExpectedPayloadSignals()
    {
        if (!Directory.Exists(MdxTestPaths.Standard060DataPath) || !File.Exists(MdxTestPaths.ListfilePath))
            return;

        using var catalog = new MpqArchiveCatalog();
        _ = ArchiveCatalogBootstrapper.Bootstrap(catalog, [MdxTestPaths.Standard060DataPath], MdxTestPaths.ListfilePath);

        string? virtualPath = MdxHitTestPayloadPaths.Standard060HitTestCandidates.FirstOrDefault(catalog.FileExists);
        if (virtualPath is null)
            return;

        byte[]? bytes = catalog.ReadFile(virtualPath);
        Assert.NotNull(bytes);

        using MemoryStream detectionStream = new(bytes!);
        WowFileDetection detection = WowFileDetector.Detect(detectionStream, virtualPath);
        Assert.Equal(WowFileKind.Mdx, detection.Kind);

        using MemoryStream hitTestStream = new(bytes!);
        MdxHitTestFile hitTestFile = MdxHitTestReader.Read(hitTestStream, virtualPath);

        Assert.True(hitTestFile.ShapeCount > 0);

        if (string.Equals(virtualPath, MdxHitTestPayloadPaths.Standard060AnubisathMdxVirtualPath, StringComparison.OrdinalIgnoreCase))
        {
            Assert.Equal(2, hitTestFile.ShapeCount);
            Assert.Equal(MdxGeometryShapeType.Cylinder, hitTestFile.Shapes[0].ShapeType);
            Assert.Equal(0.3170731f, hitTestFile.Shapes[0].BasePoint!.Value.X, 3);
            Assert.Equal(4.166667f, hitTestFile.Shapes[0].Height!.Value, 3);
            Assert.Equal(1.944444f, hitTestFile.Shapes[0].Radius!.Value, 3);
            Assert.Equal(MdxGeometryShapeType.Sphere, hitTestFile.Shapes[1].ShapeType);
            Assert.Equal(0.2408537f, hitTestFile.Shapes[1].Center!.Value.X, 3);
            Assert.Equal(5.027778f, hitTestFile.Shapes[1].Center!.Value.Z, 3);
            Assert.Equal(3.009774f, hitTestFile.Shapes[1].Radius!.Value, 3);
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

    private static byte[] CreateHitTestShapePayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, MdxGeometryShapeType ShapeType, Vector3 PrimaryVector, Vector3 SecondaryVector, float PrimaryScalar, float SecondaryScalar, Vector3TrackData? TranslationTrack, QuaternionTrackData? RotationTrack, Vector3TrackData? ScalingTrack)> shapes)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)shapes.Count));

        foreach ((string name, int objectId, int parentId, uint flags, MdxGeometryShapeType shapeType, Vector3 primaryVector, Vector3 secondaryVector, float primaryScalar, float secondaryScalar, Vector3TrackData? translationTrack, QuaternionTrackData? rotationTrack, Vector3TrackData? scalingTrack) in shapes)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(objectId));
            nodePayload.AddRange(CreateInt32Payload(parentId));
            nodePayload.AddRange(CreateUInt32Payload(flags));

            if (translationTrack is not null)
                nodePayload.AddRange(CreateVector3TrackChunk("KGTR", translationTrack));

            if (rotationTrack is not null)
                nodePayload.AddRange(CreateQuaternionTrackChunk("KGRT", rotationTrack));

            if (scalingTrack is not null)
                nodePayload.AddRange(CreateVector3TrackChunk("KGSC", scalingTrack));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));
            entryPayload.Add((byte)shapeType);

            switch (shapeType)
            {
                case MdxGeometryShapeType.Box:
                    entryPayload.AddRange(CreateVector3Payload(primaryVector));
                    entryPayload.AddRange(CreateVector3Payload(secondaryVector));
                    break;
                case MdxGeometryShapeType.Cylinder:
                    entryPayload.AddRange(CreateVector3Payload(primaryVector));
                    entryPayload.AddRange(CreateSinglePayload(primaryScalar));
                    entryPayload.AddRange(CreateSinglePayload(secondaryScalar));
                    break;
                case MdxGeometryShapeType.Sphere:
                    entryPayload.AddRange(CreateVector3Payload(primaryVector));
                    entryPayload.AddRange(CreateSinglePayload(primaryScalar));
                    break;
                case MdxGeometryShapeType.Plane:
                    entryPayload.AddRange(CreateSinglePayload(primaryScalar));
                    entryPayload.AddRange(CreateSinglePayload(secondaryScalar));
                    break;
            }

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateVector3TrackChunk(string tag, Vector3TrackData track)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)track.Keys.Count));
        payload.AddRange(CreateUInt32Payload(track.InterpolationType));
        payload.AddRange(CreateInt32Payload(track.GlobalSequenceId));

        foreach ((int time, Vector3 value, Vector3? inTangent, Vector3? outTangent) in track.Keys)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateVector3Payload(value));
            if (track.InterpolationType >= 2u)
            {
                payload.AddRange(CreateVector3Payload(inTangent ?? Vector3.Zero));
                payload.AddRange(CreateVector3Payload(outTangent ?? Vector3.Zero));
            }
        }

        return [.. payload];
    }

    private static byte[] CreateQuaternionTrackChunk(string tag, QuaternionTrackData track)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)track.Keys.Count));
        payload.AddRange(CreateUInt32Payload(track.InterpolationType));
        payload.AddRange(CreateInt32Payload(track.GlobalSequenceId));

        foreach ((int time, (uint Data0, uint Data1) value, (uint Data0, uint Data1)? inTangent, (uint Data0, uint Data1)? outTangent) in track.Keys)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateUInt32Payload(value.Data0));
            payload.AddRange(CreateUInt32Payload(value.Data1));
            if (track.InterpolationType >= 2u)
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

    private static byte[] CreateFixedAsciiPayload(string value, int length)
    {
        byte[] payload = new byte[length];
        WriteFixedAscii(payload, 0, length, value);
        return payload;
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
        List<byte> payload = [];
        payload.AddRange(CreateSinglePayload(value.X));
        payload.AddRange(CreateSinglePayload(value.Y));
        payload.AddRange(CreateSinglePayload(value.Z));
        return [.. payload];
    }

    private static void WriteFixedAscii(byte[] destination, int offset, int length, string value)
    {
        Encoding.ASCII.GetBytes(value, destination.AsSpan(offset, Math.Min(length, value.Length)));
    }

    private static Vector3TrackData CreateVector3Track(uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> keys)
    {
        return new Vector3TrackData(interpolationType, globalSequenceId, keys);
    }

    private static QuaternionTrackData CreateQuaternionTrack(uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, (uint Data0, uint Data1) Value, (uint Data0, uint Data1)? InTangent, (uint Data0, uint Data1)? OutTangent)> keys)
    {
        return new QuaternionTrackData(interpolationType, globalSequenceId, keys);
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

    private sealed record Vector3TrackData(uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys);

    private sealed record QuaternionTrackData(uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, (uint Data0, uint Data1) Value, (uint Data0, uint Data1)? InTangent, (uint Data0, uint Data1)? OutTangent)> Keys);
}

internal static class MdxHitTestPayloadPaths
{
    public const string Standard060AnubisathMdxVirtualPath = "creature/anubisath/anubisath.mdx";
    public const string Standard060WispMdxVirtualPath = "creature/wisp/wisp.mdx";

    public static readonly string[] Standard060HitTestCandidates =
    [
        Standard060AnubisathMdxVirtualPath,
        Standard060WispMdxVirtualPath,
    ];
}