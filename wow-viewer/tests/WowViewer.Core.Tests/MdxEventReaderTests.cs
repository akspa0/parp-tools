using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxEventReaderTests
{
    [Fact]
    public void Read_SyntheticClassicEventPayload_AssignsPivotsTracksAndKeyTimes()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticEventPayload",
            pivotPoints:
            [
                new Vector3(3.0f, 4.0f, 5.0f),
                new Vector3(6.0f, 7.0f, 8.0f),
            ],
            events:
            [
                ("FootstepRight", 0, -1, 0x80u,
                    (1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (30, new Vector3(0.3f, 0.0f, 0.0f), null, null)]),
                    null,
                    null,
                    ("KEVT", 5, [120, 480, 960])),
                ("FootstepLeft", 1, 0, 0x81u,
                    null,
                    null,
                    (1u, 4, [(20, new Vector3(1.2f, 1.3f, 1.4f), null, null)]),
                    null),
            ]);

        using MemoryStream stream = new(bytes);
        MdxEventFile eventFile = MdxEventReader.Read(stream, "synthetic_event_payload.mdx");

        Assert.Equal("SyntheticEventPayload", eventFile.ModelName);
        Assert.Equal(2, eventFile.EventCount);

        MdxEvent first = eventFile.Events[0];
        Assert.Equal("FootstepRight", first.Name);
        Assert.Equal(new Vector3(3.0f, 4.0f, 5.0f), first.PivotPoint);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal(new Vector3(0.3f, 0.0f, 0.0f), first.TranslationTrack!.Keys[1].Value);
        Assert.NotNull(first.EventTrack);
        Assert.Equal("KEVT", first.EventTrack!.Tag);
        Assert.Equal(5, first.EventTrack.GlobalSequenceId);
        Assert.Equal(3, first.EventTrack.KeyCount);
        Assert.Equal(120, first.EventTrack.FirstKeyTime);
        Assert.Equal(960, first.EventTrack.LastKeyTime);
        Assert.Equal([120, 480, 960], first.EventTrack.KeyTimes);

        MdxEvent second = eventFile.Events[1];
        Assert.True(second.HasParent);
        Assert.Equal(new Vector3(6.0f, 7.0f, 8.0f), second.PivotPoint);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal(4, second.ScalingTrack!.GlobalSequenceId);
        Assert.Null(second.EventTrack);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        IReadOnlyList<Vector3> pivotPoints,
        IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack, (string Tag, int GlobalSequenceId, IReadOnlyList<int> KeyTimes)? EventTrack)> events)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName)));
        bytes.AddRange(CreateChunk("EVTS", CreateEventPayload(events)));
        bytes.AddRange(CreateChunk("PIVT", CreatePivtPayload(pivotPoints)));
        return [.. bytes];
    }

    private static byte[] CreateEventPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack, (string Tag, int GlobalSequenceId, IReadOnlyList<int> KeyTimes)? EventTrack)> events)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)events.Count));

        foreach (var mdxEvent in events)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(mdxEvent.Name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(mdxEvent.ObjectId));
            nodePayload.AddRange(CreateInt32Payload(mdxEvent.ParentId));
            nodePayload.AddRange(CreateUInt32Payload(mdxEvent.Flags));

            if (mdxEvent.TranslationTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGTR", mdxEvent.TranslationTrack.Value.InterpolationType, mdxEvent.TranslationTrack.Value.GlobalSequenceId, mdxEvent.TranslationTrack.Value.Keys));
            if (mdxEvent.RotationTrack is not null)
                nodePayload.AddRange(CreateQuaternionTrack("KGRT", mdxEvent.RotationTrack.Value.InterpolationType, mdxEvent.RotationTrack.Value.GlobalSequenceId, mdxEvent.RotationTrack.Value.Keys));
            if (mdxEvent.ScalingTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGSC", mdxEvent.ScalingTrack.Value.InterpolationType, mdxEvent.ScalingTrack.Value.GlobalSequenceId, mdxEvent.ScalingTrack.Value.Keys));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));
            if (mdxEvent.EventTrack is not null)
                entryPayload.AddRange(CreateEventTrack(mdxEvent.EventTrack.Value.Tag, mdxEvent.EventTrack.Value.GlobalSequenceId, mdxEvent.EventTrack.Value.KeyTimes));

            payload.AddRange(CreateSizedPayload(entryPayload));
        }

        return [.. payload];
    }

    private static byte[] CreateEventTrack(string tag, int globalSequenceId, IReadOnlyList<int> keyTimes)
    {
        List<byte> payload = [];
        payload.AddRange(Encoding.ASCII.GetBytes(tag));
        payload.AddRange(CreateUInt32Payload((uint)keyTimes.Count));
        payload.AddRange(CreateInt32Payload(globalSequenceId));
        foreach (int keyTime in keyTimes)
            payload.AddRange(CreateInt32Payload(keyTime));

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
