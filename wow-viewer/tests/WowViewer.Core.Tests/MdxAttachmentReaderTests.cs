using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxAttachmentReaderTests
{
    [Fact]
    public void Read_SyntheticClassicAttachmentPayload_AssignsPivotsPathsAndVisibility()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticAttachmentPayload",
            pivotPoints:
            [
                new Vector3(2.0f, 3.0f, 4.0f),
                new Vector3(5.0f, 6.0f, 7.0f),
            ],
            attachments:
            [
                ("HandRight", 0, -1, 0x80u, 5u, "Textures\\WeaponAttachment.mdx",
                    (1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (20, new Vector3(0.5f, 0.0f, 0.0f), null, null)]),
                    null,
                    null,
                    ("KATV", 1u, 9, [(0, 1.0f, null, null), (100, 0.0f, null, null)])),
                ("HandLeft", 1, 0, 0x81u, 8u, null,
                    null,
                    null,
                    (1u, -1, [(20, new Vector3(1.1f, 1.2f, 1.3f), null, null)]),
                    null),
            ]);

        using MemoryStream stream = new(bytes);
        MdxAttachmentFile attachmentFile = MdxAttachmentReader.Read(stream, "synthetic_attachment_payload.mdx");

        Assert.Equal("SyntheticAttachmentPayload", attachmentFile.ModelName);
        Assert.Equal(2, attachmentFile.AttachmentCount);

        MdxAttachment first = attachmentFile.Attachments[0];
        Assert.Equal(5u, first.AttachmentId);
        Assert.Equal("Textures\\WeaponAttachment.mdx", first.Path);
        Assert.Equal(new Vector3(2.0f, 3.0f, 4.0f), first.PivotPoint);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal(new Vector3(0.5f, 0.0f, 0.0f), first.TranslationTrack!.Keys[1].Value);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal("KATV", first.VisibilityTrack!.Tag);
        Assert.Equal(9, first.VisibilityTrack.GlobalSequenceId);
        Assert.Equal(0.0f, first.VisibilityTrack.Keys[1].Value, 5);

        MdxAttachment second = attachmentFile.Attachments[1];
        Assert.True(second.HasParent);
        Assert.False(second.HasPath);
        Assert.Equal(new Vector3(5.0f, 6.0f, 7.0f), second.PivotPoint);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal(new Vector3(1.1f, 1.2f, 1.3f), second.ScalingTrack!.Keys[0].Value);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        IReadOnlyList<Vector3> pivotPoints,
        IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, uint AttachmentId, string? Path, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? VisibilityTrack)> attachments)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName)));
        bytes.AddRange(CreateChunk("ATCH", CreateAttachmentPayload(attachments)));
        bytes.AddRange(CreateChunk("PIVT", CreatePivtPayload(pivotPoints)));
        return [.. bytes];
    }

    private static byte[] CreateAttachmentPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, uint AttachmentId, string? Path, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? VisibilityTrack)> attachments)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)attachments.Count));
        payload.AddRange(CreateUInt32Payload(0u));

        foreach (var attachment in attachments)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(attachment.Name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(attachment.ObjectId));
            nodePayload.AddRange(CreateInt32Payload(attachment.ParentId));
            nodePayload.AddRange(CreateUInt32Payload(attachment.Flags));

            if (attachment.TranslationTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGTR", attachment.TranslationTrack.Value.InterpolationType, attachment.TranslationTrack.Value.GlobalSequenceId, attachment.TranslationTrack.Value.Keys));
            if (attachment.RotationTrack is not null)
                nodePayload.AddRange(CreateQuaternionTrack("KGRT", attachment.RotationTrack.Value.InterpolationType, attachment.RotationTrack.Value.GlobalSequenceId, attachment.RotationTrack.Value.Keys));
            if (attachment.ScalingTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGSC", attachment.ScalingTrack.Value.InterpolationType, attachment.ScalingTrack.Value.GlobalSequenceId, attachment.ScalingTrack.Value.Keys));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));
            entryPayload.AddRange(CreateUInt32Payload(attachment.AttachmentId));
            entryPayload.Add(0);
            entryPayload.AddRange(CreateAttachmentPathPayload(attachment.Path));
            if (attachment.VisibilityTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(attachment.VisibilityTrack.Value.Tag, attachment.VisibilityTrack.Value.InterpolationType, attachment.VisibilityTrack.Value.GlobalSequenceId, attachment.VisibilityTrack.Value.Keys));

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

    private static byte[] CreateAttachmentPathPayload(string? value)
    {
        byte[] bytes = new byte[0x104];
        if (string.IsNullOrWhiteSpace(value))
            return bytes;

        int count = Math.Min(bytes.Length, value.Length);
        for (int index = 0; index < count; index++)
            bytes[index] = (byte)value[index];

        return bytes;
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
