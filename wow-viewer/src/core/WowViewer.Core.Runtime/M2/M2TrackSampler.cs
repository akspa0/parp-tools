using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public static class M2TrackSampler
{
    private const int ArrayReferenceSize = 0x08;

    public static Vector3 SampleVector3(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<Vector3> track, Vector3 fallback)
    {
        return Sample(payload, model, sequenceIndex, timeMs, track, fallback, ReadVector3, InterpolateVector3);
    }

    public static short SampleInt16(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<short> track, short fallback)
    {
        return Sample(payload, model, sequenceIndex, timeMs, track, fallback, ReadInt16, InterpolateInt16);
    }

    public static ushort SampleUInt16(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<ushort> track, ushort fallback)
    {
        return Sample(payload, model, sequenceIndex, timeMs, track, fallback, ReadUInt16, InterpolateUInt16);
    }

    public static float SampleSingle(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<float> track, float fallback)
    {
        return Sample(payload, model, sequenceIndex, timeMs, track, fallback, ReadSingle, InterpolateSingle);
    }

    public static byte SampleByte(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<byte> track, byte fallback)
    {
        return Sample(payload, model, sequenceIndex, timeMs, track, fallback, ReadByte, InterpolateByte);
    }

    public static Quaternion SampleQuaternion(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<Quaternion> track, Quaternion fallback)
    {
        return Sample(payload, model, sequenceIndex, timeMs, track, fallback, ReadQuaternion, InterpolateQuaternion);
    }

    public static Quaternion SampleCompressedQuaternion(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<M2CompQuaternion> track, Quaternion fallback)
    {
        return Sample(payload, model, sequenceIndex, timeMs, track, fallback, ReadCompressedQuaternion, InterpolateQuaternion);
    }

    private static TValue Sample<TTrack, TValue>(
        byte[] payload,
        M2ModelDocument model,
        int sequenceIndex,
        int timeMs,
        M2TrackDefinition<TTrack> track,
        TValue fallback,
        Func<byte[], int, M2TrackInterpolation, TrackSample<TValue>> readSample,
        Func<TrackKeyFrame<TValue>, TrackKeyFrame<TValue>, float, M2TrackInterpolation, TValue> interpolate)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(track);

        if (!TryReadTrackKeyFrames(payload, track, sequenceIndex, readSample, out List<TrackKeyFrame<TValue>> keyFrames) || keyFrames.Count == 0)
            return fallback;

        uint duration = track.UsesGlobalSequence
            ? (track.GlobalSequenceIndex >= 0 && track.GlobalSequenceIndex < model.GlobalLoops.Count ? model.GlobalLoops[track.GlobalSequenceIndex] : 0u)
            : model.Sequences[sequenceIndex].Duration;

        int sampleTime = ResolveSampleTime(timeMs, duration);
        if (track.Interpolation == M2TrackInterpolation.None || keyFrames.Count == 1)
            return SampleStep(keyFrames, sampleTime);

        for (int index = 0; index < keyFrames.Count - 1; index++)
        {
            TrackKeyFrame<TValue> current = keyFrames[index];
            TrackKeyFrame<TValue> next = keyFrames[index + 1];
            if (sampleTime < current.Time || sampleTime > next.Time)
                continue;

            int span = next.Time - current.Time;
            if (span <= 0)
                return current.Value;

            float factor = Math.Clamp((sampleTime - current.Time) / (float)span, 0.0f, 1.0f);
            return interpolate(current, next, factor, track.Interpolation);
        }

        return sampleTime <= keyFrames[0].Time ? keyFrames[0].Value : keyFrames[^1].Value;
    }

    private static bool TryReadTrackKeyFrames<TTrack, TValue>(
        byte[] payload,
        M2TrackDefinition<TTrack> track,
        int sequenceIndex,
        Func<byte[], int, M2TrackInterpolation, TrackSample<TValue>> readSample,
        out List<TrackKeyFrame<TValue>> keyFrames)
    {
        keyFrames = [];
        int targetTrackIndex = track.UsesGlobalSequence ? 0 : sequenceIndex;
        if (!TryReadSequenceSlice(payload, track.TimestampArray, track.ValueArray, targetTrackIndex, out M2TrackSequenceSlice slice) || !slice.HasData)
            return false;

        int keyCount = checked((int)Math.Min(slice.TimestampCount, slice.ValueCount));
        if (keyCount <= 0)
            return false;

        int timestampBytes = checked(keyCount * sizeof(uint));
        if (!IsReadable(payload, slice.TimestampOffset, timestampBytes))
            return false;

        TrackSample<TValue> firstSample = readSample(payload, checked((int)slice.ValueOffset), track.Interpolation);
        int valueStride = GetValueStride(track.Interpolation, firstSample.ValueSizeBytes);
        int valueBytes = checked(keyCount * valueStride);
        if (!IsReadable(payload, slice.ValueOffset, valueBytes))
            return false;

        keyFrames = new List<TrackKeyFrame<TValue>>(keyCount);
        for (int keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int timestampOffset = checked((int)slice.TimestampOffset + (keyIndex * sizeof(uint)));
            int valueOffset = checked((int)slice.ValueOffset + (keyIndex * valueStride));
            int time = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(timestampOffset, sizeof(uint))));
            TrackSample<TValue> sample = readSample(payload, valueOffset, track.Interpolation);
            keyFrames.Add(new TrackKeyFrame<TValue>(time, sample.Value, sample.InTangent, sample.OutTangent));
        }

        return true;
    }

    private static bool TryReadSequenceSlice(byte[] payload, M2TrackArrayReference timestampArray, M2TrackArrayReference valueArray, int sequenceIndex, out M2TrackSequenceSlice slice)
    {
        slice = default;
        if (timestampArray.Count == 0 || valueArray.Count == 0)
            return false;

        if (sequenceIndex < 0 || sequenceIndex >= timestampArray.Count || sequenceIndex >= valueArray.Count)
            return false;

        int timestampRefOffset = checked((int)timestampArray.Offset + (sequenceIndex * ArrayReferenceSize));
        int valueRefOffset = checked((int)valueArray.Offset + (sequenceIndex * ArrayReferenceSize));
        if (!IsReadable(payload, (uint)timestampRefOffset, ArrayReferenceSize) || !IsReadable(payload, (uint)valueRefOffset, ArrayReferenceSize))
            return false;

        uint timestampCount = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(timestampRefOffset, sizeof(uint)));
        uint timestampOffset = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(timestampRefOffset + 0x04, sizeof(uint)));
        uint valueCount = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(valueRefOffset, sizeof(uint)));
        uint valueOffset = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(valueRefOffset + 0x04, sizeof(uint)));
        slice = new M2TrackSequenceSlice(timestampCount, timestampOffset, valueCount, valueOffset);
        return true;
    }

    private static bool IsReadable(byte[] payload, uint offset, int size)
    {
        return offset <= payload.Length && size >= 0 && offset <= payload.Length - size;
    }

    private static int GetValueStride(M2TrackInterpolation interpolation, int scalarSize)
    {
        return interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier
            ? checked(scalarSize * 3)
            : scalarSize;
    }

    private static int ResolveSampleTime(int timeMs, uint duration)
    {
        if (duration == 0)
            return Math.Max(timeMs, 0);

        int period = checked((int)duration);
        int sampleTime = timeMs % period;
        if (sampleTime < 0)
            sampleTime += period;

        return sampleTime;
    }

    private static T SampleStep<T>(IReadOnlyList<TrackKeyFrame<T>> keyFrames, int sampleTime)
    {
        TrackKeyFrame<T> current = keyFrames[0];
        for (int index = 1; index < keyFrames.Count; index++)
        {
            if (sampleTime < keyFrames[index].Time)
                break;

            current = keyFrames[index];
        }

        return current.Value;
    }

    private static TrackSample<byte> ReadByte(byte[] payload, int offset, M2TrackInterpolation interpolation)
    {
        byte value = payload[offset];
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
            return new TrackSample<byte>(value, payload[offset + 1], payload[offset + 2], sizeof(byte));

        return new TrackSample<byte>(value, value, value, sizeof(byte));
    }

    private static TrackSample<short> ReadInt16(byte[] payload, int offset, M2TrackInterpolation interpolation)
    {
        short value = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset, sizeof(short)));
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
        {
            return new TrackSample<short>(
                value,
                BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset + sizeof(short), sizeof(short))),
                BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset + (sizeof(short) * 2), sizeof(short))),
                sizeof(short));
        }

        return new TrackSample<short>(value, value, value, sizeof(short));
    }

    private static TrackSample<ushort> ReadUInt16(byte[] payload, int offset, M2TrackInterpolation interpolation)
    {
        ushort value = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, sizeof(ushort)));
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
        {
            return new TrackSample<ushort>(
                value,
                BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + sizeof(ushort), sizeof(ushort))),
                BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + (sizeof(ushort) * 2), sizeof(ushort))),
                sizeof(ushort));
        }

        return new TrackSample<ushort>(value, value, value, sizeof(ushort));
    }

    private static TrackSample<float> ReadSingle(byte[] payload, int offset, M2TrackInterpolation interpolation)
    {
        float value = ReadSingleValue(payload, offset);
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
        {
            return new TrackSample<float>(
                value,
                ReadSingleValue(payload, offset + sizeof(float)),
                ReadSingleValue(payload, offset + (sizeof(float) * 2)),
                sizeof(float));
        }

        return new TrackSample<float>(value, value, value, sizeof(float));
    }

    private static TrackSample<Vector3> ReadVector3(byte[] payload, int offset, M2TrackInterpolation interpolation)
    {
        Vector3 value = ReadVector3Value(payload, offset);
        const int Size = sizeof(float) * 3;
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
        {
            return new TrackSample<Vector3>(
                value,
                ReadVector3Value(payload, offset + Size),
                ReadVector3Value(payload, offset + (Size * 2)),
                Size);
        }

        return new TrackSample<Vector3>(value, value, value, Size);
    }

    private static TrackSample<Quaternion> ReadQuaternion(byte[] payload, int offset, M2TrackInterpolation interpolation)
    {
        Quaternion value = ReadQuaternionValue(payload, offset);
        const int Size = sizeof(float) * 4;
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
        {
            return new TrackSample<Quaternion>(
                value,
                ReadQuaternionValue(payload, offset + Size),
                ReadQuaternionValue(payload, offset + (Size * 2)),
                Size);
        }

        return new TrackSample<Quaternion>(value, value, value, Size);
    }

    private static TrackSample<Quaternion> ReadCompressedQuaternion(byte[] payload, int offset, M2TrackInterpolation interpolation)
    {
        Quaternion value = ReadCompQuaternionValue(payload, offset);
        const int Size = sizeof(ushort) * 4;
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
        {
            return new TrackSample<Quaternion>(
                value,
                ReadCompQuaternionValue(payload, offset + Size),
                ReadCompQuaternionValue(payload, offset + (Size * 2)),
                Size);
        }

        return new TrackSample<Quaternion>(value, value, value, Size);
    }

    private static byte InterpolateByte(TrackKeyFrame<byte> start, TrackKeyFrame<byte> end, float factor, M2TrackInterpolation interpolation)
    {
        return (byte)Math.Clamp(MathF.Round(InterpolateScalar(start.Value, start.OutTangent, end.InTangent, end.Value, factor, interpolation)), byte.MinValue, byte.MaxValue);
    }

    private static short InterpolateInt16(TrackKeyFrame<short> start, TrackKeyFrame<short> end, float factor, M2TrackInterpolation interpolation)
    {
        return (short)Math.Clamp(MathF.Round(InterpolateScalar(start.Value, start.OutTangent, end.InTangent, end.Value, factor, interpolation)), short.MinValue, short.MaxValue);
    }

    private static ushort InterpolateUInt16(TrackKeyFrame<ushort> start, TrackKeyFrame<ushort> end, float factor, M2TrackInterpolation interpolation)
    {
        return (ushort)Math.Clamp(MathF.Round(InterpolateScalar(start.Value, start.OutTangent, end.InTangent, end.Value, factor, interpolation)), ushort.MinValue, ushort.MaxValue);
    }

    private static float InterpolateSingle(TrackKeyFrame<float> start, TrackKeyFrame<float> end, float factor, M2TrackInterpolation interpolation)
    {
        return InterpolateScalar(start.Value, start.OutTangent, end.InTangent, end.Value, factor, interpolation);
    }

    private static Vector3 InterpolateVector3(TrackKeyFrame<Vector3> start, TrackKeyFrame<Vector3> end, float factor, M2TrackInterpolation interpolation)
    {
        return interpolation switch
        {
            M2TrackInterpolation.Hermite => InterpolateCubic(start.Value, start.OutTangent, end.InTangent, end.Value, factor, hermite: true),
            M2TrackInterpolation.Bezier => InterpolateCubic(start.Value, start.OutTangent, end.InTangent, end.Value, factor, hermite: false),
            _ => Vector3.Lerp(start.Value, end.Value, factor),
        };
    }

    private static Quaternion InterpolateQuaternion(TrackKeyFrame<Quaternion> start, TrackKeyFrame<Quaternion> end, float factor, M2TrackInterpolation interpolation)
    {
        Quaternion value = interpolation switch
        {
            M2TrackInterpolation.Hermite => NormalizeWeighted(start.Value, start.OutTangent, end.InTangent, end.Value, HermiteWeights(factor)),
            M2TrackInterpolation.Bezier => NormalizeWeighted(start.Value, start.OutTangent, end.InTangent, end.Value, BezierWeights(factor)),
            _ => Quaternion.Slerp(start.Value, end.Value, factor),
        };

        return value.LengthSquared() > 0.000001f ? Quaternion.Normalize(value) : Quaternion.Identity;
    }

    private static float InterpolateScalar(float start, float startOut, float endIn, float end, float factor, M2TrackInterpolation interpolation)
    {
        return interpolation switch
        {
            M2TrackInterpolation.Hermite => Dot(HermiteWeights(factor), start, startOut, endIn, end),
            M2TrackInterpolation.Bezier => Dot(BezierWeights(factor), start, startOut, endIn, end),
            _ => (start * (1.0f - factor)) + (end * factor),
        };
    }

    private static Vector3 InterpolateCubic(Vector3 start, Vector3 startOut, Vector3 endIn, Vector3 end, float factor, bool hermite)
    {
        Vector4 weights = hermite ? HermiteWeights(factor) : BezierWeights(factor);
        return (start * weights.X) + (startOut * weights.Y) + (endIn * weights.Z) + (end * weights.W);
    }

    private static Quaternion NormalizeWeighted(Quaternion start, Quaternion startOut, Quaternion endIn, Quaternion end, Vector4 weights)
    {
        Vector4 weighted = (ToVector4(start) * weights.X)
            + (ToVector4(startOut) * weights.Y)
            + (ToVector4(endIn) * weights.Z)
            + (ToVector4(end) * weights.W);

        return Quaternion.Normalize(new Quaternion(weighted.X, weighted.Y, weighted.Z, weighted.W));
    }

    private static Vector4 HermiteWeights(float factor)
    {
        float t2 = factor * factor;
        float t3 = t2 * factor;
        return new Vector4(
            (2.0f * t3) - (3.0f * t2) + 1.0f,
            t3 - (2.0f * t2) + factor,
            t3 - t2,
            (-2.0f * t3) + (3.0f * t2));
    }

    private static Vector4 BezierWeights(float factor)
    {
        float inv = 1.0f - factor;
        return new Vector4(
            inv * inv * inv,
            3.0f * inv * inv * factor,
            3.0f * inv * factor * factor,
            factor * factor * factor);
    }

    private static float Dot(Vector4 weights, float first, float second, float third, float fourth)
    {
        return (first * weights.X) + (second * weights.Y) + (third * weights.Z) + (fourth * weights.W);
    }

    private static float ReadSingleValue(byte[] payload, int offset)
    {
        return BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset, sizeof(float))));
    }

    private static Vector3 ReadVector3Value(byte[] payload, int offset)
    {
        return new Vector3(
            ReadSingleValue(payload, offset + 0x00),
            ReadSingleValue(payload, offset + 0x04),
            ReadSingleValue(payload, offset + 0x08));
    }

    private static Quaternion ReadQuaternionValue(byte[] payload, int offset)
    {
        Quaternion value = new(
            ReadSingleValue(payload, offset + 0x00),
            ReadSingleValue(payload, offset + 0x04),
            ReadSingleValue(payload, offset + 0x08),
            ReadSingleValue(payload, offset + 0x0C));

        return value.LengthSquared() > 0.000001f ? Quaternion.Normalize(value) : Quaternion.Identity;
    }

    private static Quaternion ReadCompQuaternionValue(byte[] payload, int offset)
    {
        M2CompQuaternion value = new(
            BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 0x00, sizeof(ushort))),
            BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 0x02, sizeof(ushort))),
            BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 0x04, sizeof(ushort))),
            BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 0x06, sizeof(ushort))));
        return value.ToQuaternion();
    }

    private static Vector4 ToVector4(Quaternion value)
    {
        return new Vector4(value.X, value.Y, value.Z, value.W);
    }

    private readonly record struct TrackSample<T>(T Value, T InTangent, T OutTangent, int ValueSizeBytes);

    private readonly record struct TrackKeyFrame<T>(int Time, T Value, T InTangent, T OutTangent);
}
