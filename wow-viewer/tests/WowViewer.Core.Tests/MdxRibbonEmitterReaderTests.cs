using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxRibbonEmitterReaderTests
{
    [Fact]
    public void Read_SyntheticClassicRibbonPayload_AssignsPivotsAndTracks()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticRibbonPayload",
            pivotPoints:
            [
                new Vector3(7.0f, 8.0f, 9.0f),
                new Vector3(10.0f, 11.0f, 12.0f),
            ],
            ribbons:
            [
                ("TrailMain", 0, -1, 0x80u, 2.5f, 1.5f, 0.8f, new Vector3(1.0f, 0.5f, 0.25f), 4.0f, 2u, 32u, 4u, 2u, 7u, 9.5f,
                    (1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (30, new Vector3(0.2f, 0.0f, 0.0f), null, null)]),
                    null,
                    null,
                    ("KRHA", 1u, -1, [(0, 1.0f, null, null), (100, 2.0f, null, null)]),
                    ("KRHB", 1u, -1, [(0, 0.5f, null, null), (100, 1.5f, null, null)]),
                    ("KRAL", 1u, 3, [(0, 0.9f, null, null), (100, 0.1f, null, null)]),
                    ("KRCO", 1u, -1, [(0, new Vector3(1.0f, 0.0f, 0.0f), null, null), (100, new Vector3(0.0f, 1.0f, 0.0f), null, null)]),
                    ("KRTX", 1u, -1, [(0, 2, null, null), (100, 4, null, null)]),
                    ("KVIS", 0u, -1, [(0, 1.0f, null, null), (100, 0.0f, null, null)])),
                ("TrailChild", 1, 0, 0x81u, 3.5f, 2.5f, 1.0f, new Vector3(0.25f, 0.5f, 1.0f), 6.0f, 1u, 16u, 1u, 1u, 8u, 3.0f,
                    null,
                    null,
                    (1u, 4, [(20, new Vector3(1.0f, 1.1f, 1.2f), null, null)]),
                    null,
                    null,
                    null,
                    null,
                    null,
                    null),
            ]);

        using MemoryStream stream = new(bytes);
        MdxRibbonEmitterFile ribbonFile = MdxRibbonEmitterReader.Read(stream, "synthetic_ribbon_payload.mdx");

        Assert.Equal("SyntheticRibbonPayload", ribbonFile.ModelName);
        Assert.Equal(2, ribbonFile.RibbonCount);

        MdxRibbonEmitter first = ribbonFile.Ribbons[0];
        Assert.Equal(new Vector3(7.0f, 8.0f, 9.0f), first.PivotPoint);
        Assert.Equal(2.5f, first.StaticHeightAbove, 5);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal(new Vector3(0.2f, 0.0f, 0.0f), first.TranslationTrack!.Keys[1].Value);
        Assert.NotNull(first.HeightAboveTrack);
        Assert.Equal(2.0f, first.HeightAboveTrack!.Keys[1].Value, 5);
        Assert.NotNull(first.ColorTrack);
        Assert.Equal(new Vector3(0.0f, 1.0f, 0.0f), first.ColorTrack!.Keys[1].Value);
        Assert.NotNull(first.TextureSlotTrack);
        Assert.Equal(4, first.TextureSlotTrack!.Keys[1].Value);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal(0.0f, first.VisibilityTrack!.Keys[1].Value, 5);

        MdxRibbonEmitter second = ribbonFile.Ribbons[1];
        Assert.True(second.HasParent);
        Assert.Equal(new Vector3(10.0f, 11.0f, 12.0f), second.PivotPoint);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal(4, second.ScalingTrack!.GlobalSequenceId);
        Assert.Equal(new Vector3(1.0f, 1.1f, 1.2f), second.ScalingTrack.Keys[0].Value);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        IReadOnlyList<Vector3> pivotPoints,
        IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, float StaticHeightAbove, float StaticHeightBelow, float StaticAlpha, Vector3 StaticColor, float EdgeLifetime, uint StaticTextureSlot, uint EdgesPerSecond, uint TextureRows, uint TextureColumns, uint MaterialId, float Gravity, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? HeightAboveTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? HeightBelowTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? AlphaTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ColorTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, int Value, int? InTangent, int? OutTangent)> Keys)? TextureSlotTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? VisibilityTrack)> ribbons)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName)));
        bytes.AddRange(CreateChunk("RIBB", CreateRibbonPayload(ribbons)));
        bytes.AddRange(CreateChunk("PIVT", CreatePivtPayload(pivotPoints)));
        return [.. bytes];
    }

    private static byte[] CreateRibbonPayload(IReadOnlyList<(string Name, int ObjectId, int ParentId, uint Flags, float StaticHeightAbove, float StaticHeightBelow, float StaticAlpha, Vector3 StaticColor, float EdgeLifetime, uint StaticTextureSlot, uint EdgesPerSecond, uint TextureRows, uint TextureColumns, uint MaterialId, float Gravity, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? TranslationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys)? RotationTrack, (uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ScalingTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? HeightAboveTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? HeightBelowTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? AlphaTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys)? ColorTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, int Value, int? InTangent, int? OutTangent)> Keys)? TextureSlotTrack, (string Tag, uint InterpolationType, int GlobalSequenceId, IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys)? VisibilityTrack)> ribbons)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)ribbons.Count));

        foreach (var ribbon in ribbons)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(ribbon.Name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(ribbon.ObjectId));
            nodePayload.AddRange(CreateInt32Payload(ribbon.ParentId));
            nodePayload.AddRange(CreateUInt32Payload(ribbon.Flags));

            if (ribbon.TranslationTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGTR", ribbon.TranslationTrack.Value.InterpolationType, ribbon.TranslationTrack.Value.GlobalSequenceId, ribbon.TranslationTrack.Value.Keys));
            if (ribbon.RotationTrack is not null)
                nodePayload.AddRange(CreateQuaternionTrack("KGRT", ribbon.RotationTrack.Value.InterpolationType, ribbon.RotationTrack.Value.GlobalSequenceId, ribbon.RotationTrack.Value.Keys));
            if (ribbon.ScalingTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGSC", ribbon.ScalingTrack.Value.InterpolationType, ribbon.ScalingTrack.Value.GlobalSequenceId, ribbon.ScalingTrack.Value.Keys));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));

            List<byte> staticPayload = [];
            staticPayload.AddRange(CreateSinglePayload(ribbon.StaticHeightAbove));
            staticPayload.AddRange(CreateSinglePayload(ribbon.StaticHeightBelow));
            staticPayload.AddRange(CreateSinglePayload(ribbon.StaticAlpha));
            staticPayload.AddRange(CreateVector3Payload(ribbon.StaticColor));
            staticPayload.AddRange(CreateSinglePayload(ribbon.EdgeLifetime));
            staticPayload.AddRange(CreateUInt32Payload(ribbon.StaticTextureSlot));
            staticPayload.AddRange(CreateUInt32Payload(ribbon.EdgesPerSecond));
            staticPayload.AddRange(CreateUInt32Payload(ribbon.TextureRows));
            staticPayload.AddRange(CreateUInt32Payload(ribbon.TextureColumns));
            staticPayload.AddRange(CreateUInt32Payload(ribbon.MaterialId));
            staticPayload.AddRange(CreateSinglePayload(ribbon.Gravity));

            entryPayload.AddRange(CreateUInt32Payload((uint)(4 + staticPayload.Count)));
            entryPayload.AddRange(staticPayload);

            if (ribbon.HeightAboveTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(ribbon.HeightAboveTrack.Value.Tag, ribbon.HeightAboveTrack.Value.InterpolationType, ribbon.HeightAboveTrack.Value.GlobalSequenceId, ribbon.HeightAboveTrack.Value.Keys));
            if (ribbon.HeightBelowTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(ribbon.HeightBelowTrack.Value.Tag, ribbon.HeightBelowTrack.Value.InterpolationType, ribbon.HeightBelowTrack.Value.GlobalSequenceId, ribbon.HeightBelowTrack.Value.Keys));
            if (ribbon.AlphaTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(ribbon.AlphaTrack.Value.Tag, ribbon.AlphaTrack.Value.InterpolationType, ribbon.AlphaTrack.Value.GlobalSequenceId, ribbon.AlphaTrack.Value.Keys));
            if (ribbon.ColorTrack is not null)
                entryPayload.AddRange(CreateVector3Track(ribbon.ColorTrack.Value.Tag, ribbon.ColorTrack.Value.InterpolationType, ribbon.ColorTrack.Value.GlobalSequenceId, ribbon.ColorTrack.Value.Keys));
            if (ribbon.TextureSlotTrack is not null)
                entryPayload.AddRange(CreateIntTrack(ribbon.TextureSlotTrack.Value.Tag, ribbon.TextureSlotTrack.Value.InterpolationType, ribbon.TextureSlotTrack.Value.GlobalSequenceId, ribbon.TextureSlotTrack.Value.Keys));
            if (ribbon.VisibilityTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(ribbon.VisibilityTrack.Value.Tag, ribbon.VisibilityTrack.Value.InterpolationType, ribbon.VisibilityTrack.Value.GlobalSequenceId, ribbon.VisibilityTrack.Value.Keys));

            payload.AddRange(CreateSizedPayload(entryPayload));
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
