using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxParticleEmitter2ReaderTests
{
    [Fact]
    public void Read_SyntheticClassicParticleEmitterPayload_AssignsPivotsAndTracks()
    {
        byte[] bytes = CreateMdxBytes(
            version: 1300,
            modelName: "SyntheticPre2Payload",
            pivotPoints:
            [
                new Vector3(4.0f, 5.0f, 6.0f),
                new Vector3(7.0f, 8.0f, 9.0f),
            ],
            emitters:
            [
                new ParticleEmitterFixture(
                    "EmitterMain", 0, -1, 0x200u, 3,
                    8.0f, 0.25f, 0.5f, 1.0f, -9.8f, 2.0f, 6.0f, 12.0f, 9.0f, 4.0f,
                    2u, 3u, 7u, 5.0f, 0.35f,
                    new Vector3(1.0f, 0.0f, 0.0f), new Vector3(0.0f, 1.0f, 0.0f), new Vector3(0.0f, 0.0f, 1.0f),
                    255, 128, 64,
                    1.2f, 0.8f, 0.4f,
                    6u, 5, 2, 11u,
                    "Particles/Glow.mdx", "Particles/Recur.mdx",
                    [10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u, 90u, 100u, 110u, 120u],
                    [0.1f, 0.2f, 0.3f, 0.4f, 0.5f],
                    [1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f],
                    [2.1f, 2.2f],
                    new Vector3(9.0f, 8.0f, 7.0f),
                    [3.1f, 3.2f, 3.3f, 3.4f, 3.5f],
                    [new Vector3(0.0f, 0.0f, 0.0f), new Vector3(1.0f, 1.0f, 1.0f)],
                    1,
                    new Vector3TrackFixture(1u, -1, [(10, new Vector3(0.0f, 0.0f, 0.0f), null, null), (20, new Vector3(0.5f, 0.0f, 0.0f), null, null)]),
                    null,
                    null,
                    new ScalarTrackFixture("KVIS", 1u, -1, [(0, 1.0f, null, null), (100, 0.0f, null, null)]),
                    new ScalarTrackFixture("KP2S", 1u, -1, [(0, 8.0f, null, null), (100, 12.0f, null, null)]),
                    null,
                    null,
                    null,
                    null,
                    null,
                    new ScalarTrackFixture("KP2E", 1u, -1, [(0, 12.0f, null, null), (100, 15.0f, null, null)]),
                    null,
                    null,
                    null),
                new ParticleEmitterFixture(
                    "EmitterChild", 1, 0, 0x201u, 1,
                    4.0f, 0.1f, 0.2f, 0.3f, -1.0f, 0.0f, 2.0f, 3.0f, 2.5f, 1.5f,
                    1u, 1u, 2u, 0.0f, 0.5f,
                    new Vector3(0.25f, 0.25f, 0.25f), new Vector3(0.5f, 0.5f, 0.5f), new Vector3(0.75f, 0.75f, 0.75f),
                    10, 20, 30,
                    0.5f, 0.6f, 0.7f,
                    1u, 2, 3, 4u,
                    null, null,
                    [1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u],
                    [0f, 0f, 0f, 0f, 0f],
                    [0f, 0f, 0f, 0f, 0f, 0f],
                    [0f, 0f],
                    Vector3.Zero,
                    [0f, 0f, 0f, 0f, 0f],
                    [],
                    0,
                    null,
                    null,
                    new Vector3TrackFixture(1u, 4, [(30, new Vector3(1.0f, 1.1f, 1.2f), null, null)]),
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                        null,
                    null),
            ]);

        using MemoryStream stream = new(bytes);
        MdxParticleEmitter2File particleFile = MdxParticleEmitter2Reader.Read(stream, "synthetic_pre2_payload.mdx");

        Assert.Equal("SyntheticPre2Payload", particleFile.ModelName);
        Assert.Equal(2, particleFile.ParticleEmitterCount);

        MdxParticleEmitter2 first = particleFile.ParticleEmitters[0];
        Assert.Equal(new Vector3(4.0f, 5.0f, 6.0f), first.PivotPoint);
        Assert.Equal(3, first.EmitterType);
        Assert.Equal(6u, first.BlendMode);
        Assert.Equal(5, first.TextureId);
        Assert.Equal("Particles/Glow.mdx", first.GeometryModel);
        Assert.Equal("Particles/Recur.mdx", first.RecursionModel);
        Assert.Equal(12, first.UnknownIntervals.Count);
        Assert.Equal(2, first.SplinePoints.Count);
        Assert.NotNull(first.TranslationTrack);
        Assert.Equal(new Vector3(0.5f, 0.0f, 0.0f), first.TranslationTrack!.Keys[1].Value);
        Assert.NotNull(first.VisibilityTrack);
        Assert.Equal(0.0f, first.VisibilityTrack!.Keys[1].Value, 5);
        Assert.NotNull(first.SpeedTrack);
        Assert.Equal(12.0f, first.SpeedTrack!.Keys[1].Value, 5);
        Assert.NotNull(first.EmissionRateTrack);
        Assert.Equal(15.0f, first.EmissionRateTrack!.Keys[1].Value, 5);

        MdxParticleEmitter2 second = particleFile.ParticleEmitters[1];
        Assert.True(second.HasParent);
        Assert.Equal(new Vector3(7.0f, 8.0f, 9.0f), second.PivotPoint);
        Assert.NotNull(second.ScalingTrack);
        Assert.Equal(4, second.ScalingTrack!.GlobalSequenceId);
        Assert.Equal(new Vector3(1.0f, 1.1f, 1.2f), second.ScalingTrack.Keys[0].Value);
    }

    private static byte[] CreateMdxBytes(
        uint version,
        string modelName,
        IReadOnlyList<Vector3> pivotPoints,
        IReadOnlyList<ParticleEmitterFixture> emitters)
    {
        List<byte> bytes = [];
        bytes.AddRange(Encoding.ASCII.GetBytes("MDLX"));
        bytes.AddRange(CreateChunk("VERS", CreateUInt32Payload(version)));
        bytes.AddRange(CreateChunk("MODL", CreateModlPayload(modelName)));
        bytes.AddRange(CreateChunk("PRE2", CreatePre2Payload(emitters)));
        bytes.AddRange(CreateChunk("PIVT", CreatePivtPayload(pivotPoints)));
        return [.. bytes];
    }

    private static byte[] CreatePre2Payload(IReadOnlyList<ParticleEmitterFixture> emitters)
    {
        List<byte> payload = [];
        payload.AddRange(CreateUInt32Payload((uint)emitters.Count));

        foreach (ParticleEmitterFixture emitter in emitters)
        {
            List<byte> entryPayload = [];
            List<byte> nodePayload = [];
            nodePayload.AddRange(CreateFixedAsciiPayload(emitter.Name, 0x50));
            nodePayload.AddRange(CreateInt32Payload(emitter.ObjectId));
            nodePayload.AddRange(CreateInt32Payload(emitter.ParentId));
            nodePayload.AddRange(CreateUInt32Payload(emitter.Flags));

            if (emitter.TranslationTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGTR", emitter.TranslationTrack.InterpolationType, emitter.TranslationTrack.GlobalSequenceId, emitter.TranslationTrack.Keys));
            if (emitter.RotationTrack is not null)
                nodePayload.AddRange(CreateQuaternionTrack("KGRT", emitter.RotationTrack.InterpolationType, emitter.RotationTrack.GlobalSequenceId, emitter.RotationTrack.Keys));
            if (emitter.ScalingTrack is not null)
                nodePayload.AddRange(CreateVector3Track("KGSC", emitter.ScalingTrack.InterpolationType, emitter.ScalingTrack.GlobalSequenceId, emitter.ScalingTrack.Keys));

            entryPayload.AddRange(CreateSizedPayload(nodePayload));

            List<byte> staticPayload = [];
            staticPayload.AddRange(CreateInt32Payload(emitter.EmitterType));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticSpeed));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticVariation));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticLatitude));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticLongitude));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticGravity));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticZSource));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticLife));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticEmissionRate));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticLength));
            staticPayload.AddRange(CreateSinglePayload(emitter.StaticWidth));
            staticPayload.AddRange(CreateUInt32Payload(emitter.Rows));
            staticPayload.AddRange(CreateUInt32Payload(emitter.Columns));
            staticPayload.AddRange(CreateUInt32Payload(emitter.ParticleType));
            staticPayload.AddRange(CreateSinglePayload(emitter.TailLength));
            staticPayload.AddRange(CreateSinglePayload(emitter.MiddleTime));
            staticPayload.AddRange(CreateVector3Payload(emitter.StartColor));
            staticPayload.AddRange(CreateVector3Payload(emitter.MiddleColor));
            staticPayload.AddRange(CreateVector3Payload(emitter.EndColor));
            staticPayload.Add(emitter.StartAlpha);
            staticPayload.Add(emitter.MiddleAlpha);
            staticPayload.Add(emitter.EndAlpha);
            staticPayload.AddRange(CreateSinglePayload(emitter.StartScale));
            staticPayload.AddRange(CreateSinglePayload(emitter.MiddleScale));
            staticPayload.AddRange(CreateSinglePayload(emitter.EndScale));

            foreach (uint interval in emitter.UnknownIntervals)
                staticPayload.AddRange(CreateUInt32Payload(interval));

            staticPayload.AddRange(CreateUInt32Payload(emitter.BlendMode));
            staticPayload.AddRange(CreateInt32Payload(emitter.TextureId));
            staticPayload.AddRange(CreateInt32Payload(emitter.PriorityPlane));
            staticPayload.AddRange(CreateUInt32Payload(emitter.ReplaceableId));
            staticPayload.AddRange(CreateFixedAsciiPayload(emitter.GeometryModel ?? string.Empty, 0x104));
            staticPayload.AddRange(CreateFixedAsciiPayload(emitter.RecursionModel ?? string.Empty, 0x104));

            foreach (float value in emitter.UnknownFloatBlockA)
                staticPayload.AddRange(CreateSinglePayload(value));
            foreach (float value in emitter.UnknownTumbleValues)
                staticPayload.AddRange(CreateSinglePayload(value));
            foreach (float value in emitter.UnknownFloatBlockB)
                staticPayload.AddRange(CreateSinglePayload(value));
            staticPayload.AddRange(CreateVector3Payload(emitter.UnknownVector));
            foreach (float value in emitter.UnknownFloatBlockC)
                staticPayload.AddRange(CreateSinglePayload(value));

            staticPayload.AddRange(CreateUInt32Payload((uint)emitter.SplinePoints.Count));
            foreach (Vector3 splinePoint in emitter.SplinePoints)
                staticPayload.AddRange(CreateVector3Payload(splinePoint));
            staticPayload.AddRange(CreateInt32Payload(emitter.Squirts));

            entryPayload.AddRange(CreateUInt32Payload((uint)staticPayload.Count));
            entryPayload.AddRange(staticPayload);

            if (emitter.VisibilityTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.VisibilityTrack.Tag, emitter.VisibilityTrack.InterpolationType, emitter.VisibilityTrack.GlobalSequenceId, emitter.VisibilityTrack.Keys));
            if (emitter.SpeedTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.SpeedTrack.Tag, emitter.SpeedTrack.InterpolationType, emitter.SpeedTrack.GlobalSequenceId, emitter.SpeedTrack.Keys));
            if (emitter.VariationTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.VariationTrack.Tag, emitter.VariationTrack.InterpolationType, emitter.VariationTrack.GlobalSequenceId, emitter.VariationTrack.Keys));
            if (emitter.LatitudeTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.LatitudeTrack.Tag, emitter.LatitudeTrack.InterpolationType, emitter.LatitudeTrack.GlobalSequenceId, emitter.LatitudeTrack.Keys));
            if (emitter.LongitudeTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.LongitudeTrack.Tag, emitter.LongitudeTrack.InterpolationType, emitter.LongitudeTrack.GlobalSequenceId, emitter.LongitudeTrack.Keys));
            if (emitter.GravityTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.GravityTrack.Tag, emitter.GravityTrack.InterpolationType, emitter.GravityTrack.GlobalSequenceId, emitter.GravityTrack.Keys));
            if (emitter.LifeTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.LifeTrack.Tag, emitter.LifeTrack.InterpolationType, emitter.LifeTrack.GlobalSequenceId, emitter.LifeTrack.Keys));
            if (emitter.EmissionRateTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.EmissionRateTrack.Tag, emitter.EmissionRateTrack.InterpolationType, emitter.EmissionRateTrack.GlobalSequenceId, emitter.EmissionRateTrack.Keys));
            if (emitter.WidthTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.WidthTrack.Tag, emitter.WidthTrack.InterpolationType, emitter.WidthTrack.GlobalSequenceId, emitter.WidthTrack.Keys));
            if (emitter.LengthTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.LengthTrack.Tag, emitter.LengthTrack.InterpolationType, emitter.LengthTrack.GlobalSequenceId, emitter.LengthTrack.Keys));
            if (emitter.ZSourceTrack is not null)
                entryPayload.AddRange(CreateScalarTrack(emitter.ZSourceTrack.Tag, emitter.ZSourceTrack.InterpolationType, emitter.ZSourceTrack.GlobalSequenceId, emitter.ZSourceTrack.Keys));

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

        payload.CopyTo(bytes.AsSpan(4));
        return bytes;
    }

    private static byte[] CreateFixedAsciiPayload(string value, int size)
    {
        byte[] payload = new byte[size];
        WriteFixedAscii(payload, 0, size, value);
        return payload;

    }

    private static byte[] CreateUInt32Payload(uint value)
    {

        byte[] payload = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(payload, value);

        return payload;
    }

    private static byte[] CreateInt32Payload(int value) => CreateUInt32Payload(unchecked((uint)value));

    private static byte[] CreateSinglePayload(float value)
    {
        byte[] payload = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(payload, value);
        return payload;
    }

    private static byte[] CreateVector3Payload(Vector3 value)
    {
        byte[] payload = new byte[12];
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0, 4), value.X);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(4, 4), value.Y);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(8, 4), value.Z);
        return payload;
    }

    private static byte[] CreateQuaternionPayload(Quaternion value)

    {
        byte[] payload = new byte[8];

        PackCompressedQuaternion(value, out uint data0, out uint data1);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0, 4), data0);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(4, 4), data1);
        return payload;
    }

    private static void PackCompressedQuaternion(Quaternion value, out uint data0, out uint data1)
    {
        Quaternion normalized = Quaternion.Normalize(value);
        int xq = (int)MathF.Round(normalized.X * (1 << 21));
        int yq = (int)MathF.Round(normalized.Y * (1 << 20));
        int zq = (int)MathF.Round(normalized.Z * (1 << 20));
        xq = Math.Clamp(xq, -(1 << 21), (1 << 21) - 1);
        yq = Math.Clamp(yq, -(1 << 20), (1 << 20) - 1);
        zq = Math.Clamp(zq, -(1 << 20), (1 << 20) - 1);

        uint ux = unchecked((uint)xq) & 0x003F_FFFFu;
        uint uy = unchecked((uint)yq) & 0x001F_FFFFu;
        uint uz = unchecked((uint)zq) & 0x001F_FFFFu;

        data0 = (uz & 0x001F_FFFFu) | ((uy & 0x0000_07FFu) << 21);
        data1 = ((uy >> 11) & 0x0000_03FFu) | (ux << 10);
    }

    private static void WriteFixedAscii(byte[] buffer, int offset, int size, string value)
    {
        string safeValue = value ?? string.Empty;
        int count = Math.Min(size, Encoding.ASCII.GetByteCount(safeValue));
        Encoding.ASCII.GetBytes(safeValue.AsSpan(0, safeValue.Length), buffer.AsSpan(offset, count));
        if (count < size)
            buffer[offset + count] = 0;
    }

    private sealed record ScalarTrackFixture(
        string Tag,
        uint InterpolationType,
        int GlobalSequenceId,
        IReadOnlyList<(int Time, float Value, float? InTangent, float? OutTangent)> Keys);

    private sealed record Vector3TrackFixture(
        uint InterpolationType,
        int GlobalSequenceId,
        IReadOnlyList<(int Time, Vector3 Value, Vector3? InTangent, Vector3? OutTangent)> Keys);

    private sealed record QuaternionTrackFixture(
        uint InterpolationType,
        int GlobalSequenceId,
        IReadOnlyList<(int Time, Quaternion Value, Quaternion? InTangent, Quaternion? OutTangent)> Keys);

    private sealed record ParticleEmitterFixture(
        string Name,
        int ObjectId,
        int ParentId,
        uint Flags,
        int EmitterType,
        float StaticSpeed,
        float StaticVariation,
        float StaticLatitude,
        float StaticLongitude,
        float StaticGravity,
        float StaticZSource,
        float StaticLife,
        float StaticEmissionRate,
        float StaticLength,
        float StaticWidth,
        uint Rows,
        uint Columns,
        uint ParticleType,
        float TailLength,
        float MiddleTime,
        Vector3 StartColor,
        Vector3 MiddleColor,
        Vector3 EndColor,
        byte StartAlpha,
        byte MiddleAlpha,
        byte EndAlpha,
        float StartScale,
        float MiddleScale,
        float EndScale,
        uint BlendMode,
        int TextureId,
        int PriorityPlane,
        uint ReplaceableId,
        string? GeometryModel,
        string? RecursionModel,
        IReadOnlyList<uint> UnknownIntervals,
        IReadOnlyList<float> UnknownFloatBlockA,
        IReadOnlyList<float> UnknownTumbleValues,
        IReadOnlyList<float> UnknownFloatBlockB,
        Vector3 UnknownVector,
        IReadOnlyList<float> UnknownFloatBlockC,
        IReadOnlyList<Vector3> SplinePoints,
        int Squirts,
        Vector3TrackFixture? TranslationTrack,
        QuaternionTrackFixture? RotationTrack,
        Vector3TrackFixture? ScalingTrack,
        ScalarTrackFixture? VisibilityTrack,
        ScalarTrackFixture? SpeedTrack,
        ScalarTrackFixture? VariationTrack,
        ScalarTrackFixture? LatitudeTrack,
        ScalarTrackFixture? LongitudeTrack,
        ScalarTrackFixture? GravityTrack,
        ScalarTrackFixture? LifeTrack,
        ScalarTrackFixture? EmissionRateTrack,
        ScalarTrackFixture? WidthTrack,
        ScalarTrackFixture? LengthTrack,
        ScalarTrackFixture? ZSourceTrack);
}
