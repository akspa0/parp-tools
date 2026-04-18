using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxBoneReaderTests
{
    [Fact]
    public void Read_SyntheticClassicBonePayload_AssignsPivotsAndTracks()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticBonePayload",
            boundsMin: new Vector3(-1.0f),
            boundsMax: new Vector3(1.0f),
            pivotPoints:
            [
                new Vector3(1.0f, 2.0f, 3.0f),
                new Vector3(4.0f, 5.0f, 6.0f),
            ],
            bones:
            [
                ("Root", 0, -1, 0x80u, 0u, uint.MaxValue,
                    (1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (40, new Vector3(2.0f, 0.0f, 0.0f), null, null)]),
                    null,
                    null),
                ("Child", 1, 0, 0x81u, uint.MaxValue, uint.MaxValue,
                    null,
                    (1u, -1, [(20, Quaternion.Identity, null, null)]),
                    (1u, 3, [(20, new Vector3(1.5f, 1.5f, 1.5f), null, null)])),
            ]);

        using MemoryStream stream = new(bytes);
        MdxBoneFile boneFile = MdxBoneReader.Read(stream, "synthetic_bone_payload.mdx");

        Assert.Equal(2, boneFile.BoneCount);
        Assert.Equal(new Vector3(1.0f, 2.0f, 3.0f), boneFile.Bones[0].PivotPoint);
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), boneFile.Bones[1].PivotPoint);
        Assert.NotNull(boneFile.Bones[0].TranslationTrack);
        Assert.Equal(2, boneFile.Bones[0].TranslationTrack!.KeyCount);
        Assert.NotNull(boneFile.Bones[1].RotationTrack);
        Assert.NotNull(boneFile.Bones[1].ScalingTrack);
        Assert.Equal(3, boneFile.Bones[1].ScalingTrack!.GlobalSequenceId);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        Vector3 boundsMin,
        Vector3 boundsMax,
        IReadOnlyList<Vector3> pivotPoints,
        IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, uint GeosetId, uint GeosetAnimationId, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack)> bones)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName, boundsMin, boundsMax)));
        bytes.AddRange(CreateChunk("BONE", CreateBonePayload(bones)));
        bytes.AddRange(CreateChunk("PIVT", CreatePivtPayload(pivotPoints)));
        return [.. bytes];
    }

    private static byte[] CreateBonePayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, uint GeosetId, uint GeosetAnimationId, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack)> bones)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)bones.Count));

        foreach (var bone in bones)
        {
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(bone.Name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(bone.ObjectId));
            nodePayload.AddRange(CreateInt32Payload(bone.ParentId));
            nodePayload.AddRange(CreateUInt32Payload(bone.Flags));

            if (bone.TranslationTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGTR", bone.TranslationTrack.Value.InterpolationType, bone.TranslationTrack.Value.GlobalSequenceId, bone.TranslationTrack.Value.Keys));
            if (bone.RotationTrack is not null)
                nodePayload.AddRange(CreateQuaternionTrack("KGRT", bone.RotationTrack.Value.InterpolationType, bone.RotationTrack.Value.GlobalSequenceId, bone.RotationTrack.Value.Keys));
            if (bone.ScalingTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGSC", bone.ScalingTrack.Value.InterpolationType, bone.ScalingTrack.Value.GlobalSequenceId, bone.ScalingTrack.Value.Keys));

            payload.AddRange(CreateSizedPayload(nodePayload));
            payload.AddRange(CreateUInt32Payload(bone.GeosetId));
            payload.AddRange(CreateUInt32Payload(bone.GeosetAnimationId));
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

    private static byte[] CreateQuaternionPayload(Quaternion value)
    {
        byte[] bytes = new byte[8];
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0, 4), value.X);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4, 4), value.Y);
        return bytes;
    }

    private static byte[] CreateFixedAsciiPayload(string value, int length)
    {
        byte[] bytes = new byte[length];
        WriteFixedAscii(bytes, 0, length, value);
        return bytes;
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
