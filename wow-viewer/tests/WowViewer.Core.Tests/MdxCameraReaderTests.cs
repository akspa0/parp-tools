using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxCameraReaderTests
{
    [Fact]
    public void Read_SyntheticClassicCameraPayload_ParsesStaticFieldsAndTracks()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticCameraPayload",
            cameras:
            [
                (
                    "Portrait",
                    new Vector3(1.0f, 2.0f, 3.0f),
                    0.95f,
                    27.0f,
                    0.2f,
                    new Vector3(4.0f, 5.0f, 6.0f),
                    (1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (40, new Vector3(2.0f, 0.0f, 0.0f), null, null)]),
                    (1u, 3, [(20, 0.0f, null, null), (60, 1.5707964f, null, null)]),
                    (0u, -1, [(0, 1.0f, null, null), (100, 0.0f, null, null)]),
                    (1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (40, new Vector3(0.0f, 3.0f, 0.0f), null, null)])),
                (
                    "Paperdoll",
                    new Vector3(-1.0f, -2.0f, -3.0f),
                    1.1f,
                    50.0f,
                    0.5f,
                    new Vector3(-4.0f, -5.0f, -6.0f),
                    null,
                    null,
                    null,
                    null),
            ]);

        using MemoryStream stream = new(bytes);
        MdxCameraFile cameraFile = MdxCameraReader.Read(stream, "synthetic_camera_payload.mdx");

        Assert.Equal(2, cameraFile.CameraCount);
        Assert.Equal("SyntheticCameraPayload", cameraFile.ModelName);

        MdxCamera first = cameraFile.Cameras[0];
        Assert.Equal("Portrait", first.Name);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), first.PivotPoint);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), first.TargetPivotPoint);
        Assert.NotNull(first.PositionTrack);
        Assert.Equal(2, first.PositionTrack!.KeyCount);
        Assert.Equal(new Vector3(2.0f, 0.0f, 0.0f), first.PositionTrack.Keys[1].Value);
        Assert.NotNull(first.RollTrack);
        Assert.Equal(3, first.RollTrack!.GlobalSequenceId);
        Assert.Equal(1.5707964f, first.RollTrack.Keys[1].Value, 5);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal(0.0f, first.VisibilityTrack!.Keys[1].Value, 5);
        Assert.NotNull(first.TargetPositionTrack);
        Assert.Equal(new Vector3(0.0f, 3.0f, 0.0f), first.TargetPositionTrack!.Keys[1].Value);

        MdxCamera second = cameraFile.Cameras[1];
        Assert.Equal("Paperdoll", second.Name);
        Assert.Null(second.PositionTrack);
        Assert.Null(second.RollTrack);
        Assert.Null(second.VisibilityTrack);
        Assert.Null(second.TargetPositionTrack);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        IReadOnlyList<(string Name, Vector3 PivotPoint, float FieldOfView, float FarClip, float NearClip, Vector3 TargetPivotPoint, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? PositionTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? RollTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? VisibilityTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TargetPositionTrack)> cameras)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName)));
        bytes.AddRange(CreateChunk("CAMS", CreateCameraPayload(cameras)));
        return [.. bytes];
    }

    private static byte[] CreateCameraPayload(IReadOnlyList<(string Name, Vector3 PivotPoint, float FieldOfView, float FarClip, float NearClip, Vector3 TargetPivotPoint, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? PositionTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? RollTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? VisibilityTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TargetPositionTrack)> cameras)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)cameras.Count));

        foreach (var camera in cameras)
        {
            List<byte> cameraPayload = [];
            cameraPayload.AddRange(CreateFixedAsciiPayload(camera.Name, 0x50));
            cameraPayload.AddRange(CreateVector3Payload(camera.PivotPoint));
            cameraPayload.AddRange(CreateSinglePayload(camera.FieldOfView));
            cameraPayload.AddRange(CreateSinglePayload(camera.FarClip));
            cameraPayload.AddRange(CreateSinglePayload(camera.NearClip));
            cameraPayload.AddRange(CreateVector3Payload(camera.TargetPivotPoint));

            if (camera.PositionTrack is not null)
                cameraPayload.AddRange(CreateVector3Track("KCTR", camera.PositionTrack.Value.InterpolationType, camera.PositionTrack.Value.GlobalSequenceId, camera.PositionTrack.Value.Keys));
            if (camera.RollTrack is not null)
                cameraPayload.AddRange(CreateScalarTrack("KCRL", camera.RollTrack.Value.InterpolationType, camera.RollTrack.Value.GlobalSequenceId, camera.RollTrack.Value.Keys));
            if (camera.VisibilityTrack is not null)
                cameraPayload.AddRange(CreateScalarTrack("KVIS", camera.VisibilityTrack.Value.InterpolationType, camera.VisibilityTrack.Value.GlobalSequenceId, camera.VisibilityTrack.Value.Keys));
            if (camera.TargetPositionTrack is not null)
                cameraPayload.AddRange(CreateVector3Track("KTTR", camera.TargetPositionTrack.Value.InterpolationType, camera.TargetPositionTrack.Value.GlobalSequenceId, camera.TargetPositionTrack.Value.Keys));

            payload.AddRange(CreateSizedPayload(cameraPayload));
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

    private static byte[] CreateModlPayload(string modelName)
    {
        byte[] payload = new byte[0x6C];
        WriteFixedAscii(payload, 0, 0x50, modelName);
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
        byte[] bytes = new byte[length];
        WriteFixedAscii(bytes, 0, length, value);
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
