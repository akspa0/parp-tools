using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxHelperReaderTests
{
    [Fact]
    public void Read_SyntheticClassicHelperPayload_AssignsPivotsAndTracks()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticHelperPayload",
            pivotPoints:
            [
                new Vector3(1.0f, 2.0f, 3.0f),
                new Vector3(4.0f, 5.0f, 6.0f),
            ],
            helpers:
            [
                ("SocketRoot", 0, -1, 0x80u,
                    (1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (30, new Vector3(1.0f, 0.0f, 0.0f), null, null)]),
                    null,
                    null),
                ("SocketChild", 1, 0, 0x81u,
                    null,
                    null,
                    (1u, 7, [(20, new Vector3(1.5f, 1.5f, 1.5f), null, null)])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxHelperFile helperFile = MdxHelperReader.Read(stream, "synthetic_helper_payload.mdx");

        Assert.Equal("SyntheticHelperPayload", helperFile.ModelName);
        Assert.Equal(2, helperFile.HelperCount);

        MdxHelper root = helperFile.Helpers[0];
        Assert.Equal("SocketRoot", root.Name);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), root.PivotPoint);
        Assert.NotNull(root.TranslationTrack);
        Assert.Equal(2, root.TranslationTrack!.KeyCount);
        Assert.Equal(new Vector3(1.0f, 0.0f, 0.0f), root.TranslationTrack.Keys[1].Value);

        MdxHelper child = helperFile.Helpers[1];
        Assert.True(child.HasParent);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), child.PivotPoint);
        Assert.NotNull(child.ScalingTrack);
        Assert.Equal(7, child.ScalingTrack!.GlobalSequenceId);
        Assert.Equal(new Vector3(1.5f, 1.5f, 1.5f), child.ScalingTrack.Keys[0].Value);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        IReadOnlyList<Vector3> pivotPoints,
        IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack)> helpers)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName)));
        bytes.AddRange(CreateChunk("HELP", CreateHelperPayload(helpers)));
        bytes.AddRange(CreateChunk("PIVT", CreatePivtPayload(pivotPoints)));
        return [.. bytes];
    }

    private static byte[] CreateHelperPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack)> helpers)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)helpers.Count));

        foreach (var helper in helpers)
        {
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(helper.Name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(helper.ObjectId));
            nodePayload.AddRange(CreateInt32Payload(helper.ParentId));
            nodePayload.AddRange(CreateUInt32Payload(helper.Flags));

            if (helper.TranslationTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGTR", helper.TranslationTrack.Value.InterpolationType, helper.TranslationTrack.Value.GlobalSequenceId, helper.TranslationTrack.Value.Keys));
            if (helper.RotationTrack is not null)
                nodePayload.AddRange(CreateQuaternionTrack("KGRT", helper.RotationTrack.Value.InterpolationType, helper.RotationTrack.Value.GlobalSequenceId, helper.RotationTrack.Value.Keys));
            if (helper.ScalingTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGSC", helper.ScalingTrack.Value.InterpolationType, helper.ScalingTrack.Value.GlobalSequenceId, helper.ScalingTrack.Value.Keys));

            payload.AddRange(CreateSizedPayload(nodePayload));
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

    private static byte[] CreateQuaternionTrack(string tag, uint interpolationType, int globalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> keys)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)keys.Count));
        payload.AddRange(CreateUInt32Payload(interpolationType));
        payload.AddRange(CreateInt32Payload(globalSequenceId));
        foreach ((int time, Quaternion value, Quaternion? inTangent, Quaternion? outTangent) in keys)
        {
            payload.AddRange(CreateInt32Payload(time));
            payload.AddRange(CreateQuaternionPayload(value));
            if (interpolationType >= 2u)
            {
                payload.AddRange(CreateQuaternionPayload(inTangent ?? Quaternion.Identity));
                payload.AddRange(CreateQuaternionPayload(outTangent ?? Quaternion.Identity));
            }
        }

        return [.. payload];
    }

    private static byte[] CreatePivtPayload(IReadOnlyList<Vector3> pivotPoints)
    {
        byte[] payload = new byte[pivotPoints.Count * 12];
        for (int index = 0; index < pivotPoints.Count; index++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan((index * 12) + 0, 4), pivotPoints[index].X);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan((index * 12) + 4, 4), pivotPoints[index].Y);
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan((index * 12) + 8, 4), pivotPoints[index].Z);
        }

        return payload;
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

    private static byte[] CreateVector3Payload(Vector3 value)
    {
        byte[] bytes = new byte[12];
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0, 4), value.X);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4, 4), value.Y);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(8, 4), value.Z);
        return bytes;
    }

    private static byte[] CreateQuaternionPayload(Quaternion value)
    {
        byte[] bytes = new byte[8];
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0, 4), value.X);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4, 4), value.Y);
        return bytes;
    }

    private static void WriteFixedAscii(byte[] buffer, int offset, int length, string value)
    {
        int count = Math.Min(length, value.Length);
        for (int index = 0; index < count; index++)
            buffer[offset + index] = (byte)value[index];
    }
}
