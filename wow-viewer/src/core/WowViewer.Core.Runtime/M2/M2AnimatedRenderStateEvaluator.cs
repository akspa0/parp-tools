using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2AnimatedRenderState
{
    public M2AnimatedRenderState(
        int requestedSequenceIndex,
        int resolvedSequenceIndex,
        int timeMs,
        bool usesExternalPayload,
        IReadOnlyList<M2AnimatedRenderPassState> passes,
        IReadOnlyList<M2AnimatedLightState> lights)
    {
        ArgumentNullException.ThrowIfNull(passes);
        ArgumentNullException.ThrowIfNull(lights);

        RequestedSequenceIndex = requestedSequenceIndex;
        ResolvedSequenceIndex = resolvedSequenceIndex;
        TimeMs = timeMs;
        UsesExternalPayload = usesExternalPayload;
        Passes = passes;
        Lights = lights;
    }

    public int RequestedSequenceIndex { get; }

    public int ResolvedSequenceIndex { get; }

    public int TimeMs { get; }

    public bool UsesExternalPayload { get; }

    public IReadOnlyList<M2AnimatedRenderPassState> Passes { get; }

    public IReadOnlyList<M2AnimatedLightState> Lights { get; }
}

public sealed class M2AnimatedRenderPassState
{
    public M2AnimatedRenderPassState(
        int sectionIndex,
        int passIndex,
        int batchIndex,
        Vector3 color,
        float colorAlpha,
        float combinedAlpha,
        IReadOnlyList<M2AnimatedTextureBindingState> textureBindings)
    {
        ArgumentNullException.ThrowIfNull(textureBindings);

        SectionIndex = sectionIndex;
        PassIndex = passIndex;
        BatchIndex = batchIndex;
        Color = color;
        ColorAlpha = colorAlpha;
        CombinedAlpha = combinedAlpha;
        TextureBindings = textureBindings;
    }

    public int SectionIndex { get; }

    public int PassIndex { get; }

    public int BatchIndex { get; }

    public Vector3 Color { get; }

    public float ColorAlpha { get; }

    public float CombinedAlpha { get; }

    public IReadOnlyList<M2AnimatedTextureBindingState> TextureBindings { get; }
}

public sealed class M2AnimatedTextureBindingState
{
    public M2AnimatedTextureBindingState(int stageIndex, float transparencyAlpha, Vector3 translation, Quaternion rotation, Vector3 scaling)
    {
        StageIndex = stageIndex;
        TransparencyAlpha = transparencyAlpha;
        Translation = translation;
        Rotation = rotation;
        Scaling = scaling;
    }

    public int StageIndex { get; }

    public float TransparencyAlpha { get; }

    public Vector3 Translation { get; }

    public Quaternion Rotation { get; }

    public Vector3 Scaling { get; }
}

public sealed class M2AnimatedLightState
{
    public M2AnimatedLightState(
        int lightIndex,
        ushort type,
        short boneIndex,
        Vector3 position,
        Vector3 ambientColor,
        float ambientIntensity,
        Vector3 diffuseColor,
        float diffuseIntensity,
        float attenuationStart,
        float attenuationEnd,
        bool visible)
    {
        LightIndex = lightIndex;
        Type = type;
        BoneIndex = boneIndex;
        Position = position;
        AmbientColor = ambientColor;
        AmbientIntensity = ambientIntensity;
        DiffuseColor = diffuseColor;
        DiffuseIntensity = diffuseIntensity;
        AttenuationStart = attenuationStart;
        AttenuationEnd = attenuationEnd;
        Visible = visible;
    }

    public int LightIndex { get; }

    public ushort Type { get; }

    public short BoneIndex { get; }

    public Vector3 Position { get; }

    public Vector3 AmbientColor { get; }

    public float AmbientIntensity { get; }

    public Vector3 DiffuseColor { get; }

    public float DiffuseIntensity { get; }

    public float AttenuationStart { get; }

    public float AttenuationEnd { get; }

    public bool Visible { get; }
}

public static class M2AnimatedRenderStateEvaluator
{
    private const int ArrayReferenceSize = 0x08;

    public static M2AnimatedRenderState Evaluate(
        M2ModelDocument model,
        M2StaticRenderModel renderModel,
        int sequenceIndex,
        int timeMs,
        M2ExternalAnimationRuntimeState? externalAnimationState = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(renderModel);

        if (sequenceIndex < 0 || sequenceIndex >= model.Sequences.Count)
            throw new ArgumentOutOfRangeException(nameof(sequenceIndex), $"Sequence index {sequenceIndex} is out of range for model '{model.Identity.CanonicalModelPath}'.");

        if (externalAnimationState is not null && externalAnimationState.RequestedSequenceIndex != sequenceIndex)
        {
            throw new ArgumentException(
                $"External animation state targets sequence {externalAnimationState.RequestedSequenceIndex} but evaluation requested sequence {sequenceIndex}.",
                nameof(externalAnimationState));
        }

        int resolvedSequenceIndex = externalAnimationState?.ResolvedSequenceIndex ?? sequenceIndex;
        byte[] payload = ResolvePayload(model, externalAnimationState);
        bool usesExternalPayload = !ReferenceEquals(payload, model.RawBytes);

        List<M2AnimatedRenderPassState> passes = new();
        for (int sectionIndex = 0; sectionIndex < renderModel.StructuredSections.Count; sectionIndex++)
        {
            M2StructuredRenderSection section = renderModel.StructuredSections[sectionIndex];
            for (int passIndex = 0; passIndex < section.Passes.Count; passIndex++)
            {
                M2StructuredRenderPass pass = section.Passes[passIndex];
                M2StaticRenderMaterial material = pass.Material;

                Vector3 color = Vector3.One;
                float colorAlpha = 1.0f;
                if (material.ColorIndex >= 0 && material.ColorIndex < model.Colors.Count)
                {
                    M2ColorDefinition colorDefinition = model.Colors[material.ColorIndex];
                    color = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, colorDefinition.ColorTrack, Vector3.One);
                    colorAlpha = NormalizeFixed16Alpha(EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, colorDefinition.AlphaTrack, (short)0x7FFF));
                }

                List<M2AnimatedTextureBindingState> bindingStates = new(material.TextureBindings.Count);
                float combinedAlpha = colorAlpha;
                foreach (M2StaticRenderTextureBinding binding in material.TextureBindings)
                {
                    float transparencyAlpha = 1.0f;
                    if (binding.TransparencyLookupValue is ushort transparencyIndex && transparencyIndex != ushort.MaxValue && transparencyIndex < model.TextureWeights.Count)
                    {
                        M2TextureWeightDefinition weightDefinition = model.TextureWeights[transparencyIndex];
                        transparencyAlpha = NormalizeFixed16Alpha(EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, weightDefinition.WeightTrack, (short)0x7FFF));
                    }

                    Vector3 translation = Vector3.Zero;
                    Quaternion rotation = Quaternion.Identity;
                    Vector3 scaling = Vector3.One;
                    if (binding.TextureAnimationLookupValue is ushort transformIndex && transformIndex != ushort.MaxValue && transformIndex < model.TextureTransforms.Count)
                    {
                        M2TextureTransformDefinition transformDefinition = model.TextureTransforms[transformIndex];
                        translation = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, transformDefinition.TranslationTrack, Vector3.Zero);
                        rotation = Quaternion.Normalize(EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, transformDefinition.RotationTrack, Quaternion.Identity));
                        scaling = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, transformDefinition.ScalingTrack, Vector3.One);
                    }

                    if (!material.EffectRecipe.SuppressCombinedTransparency)
                        combinedAlpha *= transparencyAlpha;

                    bindingStates.Add(new M2AnimatedTextureBindingState(binding.StageIndex, transparencyAlpha, translation, rotation, scaling));
                }

                passes.Add(new M2AnimatedRenderPassState(sectionIndex, passIndex, material.BatchIndex, color, colorAlpha, combinedAlpha, bindingStates));
            }
        }

        List<M2AnimatedLightState> lights = new(model.Lights.Count);
        foreach (M2LightDefinition light in model.Lights)
        {
            Vector3 ambientColor = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, light.AmbientColorTrack, Vector3.Zero);
            float ambientIntensity = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, light.AmbientIntensityTrack, 1.0f);
            Vector3 diffuseColor = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, light.DiffuseColorTrack, Vector3.Zero);
            float diffuseIntensity = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, light.DiffuseIntensityTrack, 1.0f);
            float attenuationStart = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, light.AttenuationStartTrack, 0.0f);
            float attenuationEnd = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, light.AttenuationEndTrack, 0.0f);
            byte visibility = EvaluateTrack(payload, model, resolvedSequenceIndex, timeMs, light.VisibilityTrack, byte.MaxValue);

            lights.Add(new M2AnimatedLightState(
                light.Index,
                light.Type,
                light.BoneIndex,
                light.Position,
                ambientColor,
                ambientIntensity,
                diffuseColor,
                diffuseIntensity,
                attenuationStart,
                attenuationEnd,
                visibility != 0));
        }

        return new M2AnimatedRenderState(sequenceIndex, resolvedSequenceIndex, timeMs, usesExternalPayload, passes, lights);
    }

    private static byte[] ResolvePayload(M2ModelDocument model, M2ExternalAnimationRuntimeState? externalAnimationState)
    {
        if (externalAnimationState?.UsesExternalFile == true && externalAnimationState.LoadedAnimation is not null)
            return externalAnimationState.LoadedAnimation.Payload;

        return model.RawBytes;
    }

    private static T EvaluateTrack<T>(byte[] payload, M2ModelDocument model, int sequenceIndex, int timeMs, M2TrackDefinition<T> track, T fallback)
    {
        if (!TryReadTrackKeyFrames(payload, track, sequenceIndex, out List<TrackKeyFrame<T>> keyFrames) || keyFrames.Count == 0)
            return fallback;

        uint duration = track.UsesGlobalSequence
            ? (track.GlobalSequenceIndex >= 0 && track.GlobalSequenceIndex < model.GlobalLoops.Count ? model.GlobalLoops[track.GlobalSequenceIndex] : 0u)
            : model.Sequences[sequenceIndex].Duration;

        int sampleTime = ResolveSampleTime(timeMs, duration);
        if (track.Interpolation == M2TrackInterpolation.None || keyFrames.Count == 1)
            return SampleStep(keyFrames, sampleTime);

        for (int index = 0; index < keyFrames.Count - 1; index++)
        {
            TrackKeyFrame<T> current = keyFrames[index];
            TrackKeyFrame<T> next = keyFrames[index + 1];
            if (sampleTime < current.Time || sampleTime > next.Time)
                continue;

            int span = next.Time - current.Time;
            if (span <= 0)
                return current.Value;

            float factor = Math.Clamp((sampleTime - current.Time) / (float)span, 0.0f, 1.0f);
            return track.Interpolation switch
            {
                M2TrackInterpolation.Hermite => InterpolateHermite(current, next, factor),
                M2TrackInterpolation.Bezier => InterpolateBezier(current, next, factor),
                _ => InterpolateLinear(current.Value, next.Value, factor),
            };
        }

        return sampleTime <= keyFrames[0].Time ? keyFrames[0].Value : keyFrames[^1].Value;
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

    private static bool TryReadTrackKeyFrames<T>(byte[] payload, M2TrackDefinition<T> track, int sequenceIndex, out List<TrackKeyFrame<T>> keyFrames)
    {
        keyFrames = [];
        int targetTrackIndex = track.UsesGlobalSequence ? 0 : sequenceIndex;
        if (!TryReadSequenceSlice(payload, track.TimestampArray, track.ValueArray, targetTrackIndex, out M2TrackSequenceSlice slice) || !slice.HasData)
            return false;

        int keyCount = checked((int)Math.Min(slice.TimestampCount, slice.ValueCount));
        if (keyCount <= 0)
            return false;

        int scalarSize = GetScalarSize<T>();
        int timestampBytes = checked(keyCount * sizeof(uint));
        if (!IsReadable(payload, slice.TimestampOffset, timestampBytes))
            return false;

        int valueStride = GetValueStride<T>(track.Interpolation, scalarSize);
        int valueBytes = checked(keyCount * valueStride);
        if (!IsReadable(payload, slice.ValueOffset, valueBytes))
            return false;

        keyFrames = new List<TrackKeyFrame<T>>(keyCount);
        for (int keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int timestampOffset = checked((int)slice.TimestampOffset + (keyIndex * sizeof(uint)));
            int valueOffset = checked((int)slice.ValueOffset + (keyIndex * valueStride));
            int time = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(timestampOffset, sizeof(uint))));
            ReadTrackSample(payload, valueOffset, track.Interpolation, out T value, out T inTangent, out T outTangent);
            keyFrames.Add(new TrackKeyFrame<T>(time, value, inTangent, outTangent));
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

    private static int GetScalarSize<T>()
    {
        if (typeof(T) == typeof(byte))
            return sizeof(byte);
        if (typeof(T) == typeof(short))
            return sizeof(short);
        if (typeof(T) == typeof(float))
            return sizeof(float);
        if (typeof(T) == typeof(Vector3))
            return sizeof(float) * 3;
        if (typeof(T) == typeof(Quaternion))
            return sizeof(float) * 4;

        throw new NotSupportedException($"Unsupported M2 track value type '{typeof(T).FullName}'.");
    }

    private static int GetValueStride<T>(M2TrackInterpolation interpolation, int scalarSize)
    {
        return interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier
            ? checked(scalarSize * 3)
            : scalarSize;
    }

    private static void ReadTrackSample<T>(byte[] payload, int offset, M2TrackInterpolation interpolation, out T value, out T inTangent, out T outTangent)
    {
        if (interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier)
        {
            int scalarSize = GetScalarSize<T>();
            value = ReadValue<T>(payload, offset);
            inTangent = ReadValue<T>(payload, offset + scalarSize);
            outTangent = ReadValue<T>(payload, offset + (scalarSize * 2));
            return;
        }

        value = ReadValue<T>(payload, offset);
        inTangent = value;
        outTangent = value;
    }

    private static T ReadValue<T>(byte[] payload, int offset)
    {
        if (typeof(T) == typeof(byte))
        {
            byte value = payload[offset];
            return Unsafe.As<byte, T>(ref value);
        }

        if (typeof(T) == typeof(short))
        {
            short value = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset, sizeof(short)));
            return Unsafe.As<short, T>(ref value);
        }

        if (typeof(T) == typeof(float))
        {
            float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset, sizeof(float))));
            return Unsafe.As<float, T>(ref value);
        }

        if (typeof(T) == typeof(Vector3))
        {
            Vector3 value = new(
                BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + 0x00, sizeof(float)))),
                BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + 0x04, sizeof(float)))),
                BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + 0x08, sizeof(float)))));
            return Unsafe.As<Vector3, T>(ref value);
        }

        if (typeof(T) == typeof(Quaternion))
        {
            Quaternion value = Quaternion.Normalize(new Quaternion(
                BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + 0x00, sizeof(float)))),
                BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + 0x04, sizeof(float)))),
                BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + 0x08, sizeof(float)))),
                BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(offset + 0x0C, sizeof(float))))));
            return Unsafe.As<Quaternion, T>(ref value);
        }

        throw new NotSupportedException($"Unsupported M2 track value type '{typeof(T).FullName}'.");
    }

    private static T InterpolateLinear<T>(T start, T end, float factor)
    {
        if (typeof(T) == typeof(byte))
        {
            byte startValue = Unsafe.As<T, byte>(ref start);
            byte endValue = Unsafe.As<T, byte>(ref end);
            byte value = (byte)Math.Clamp(MathF.Round((startValue * (1.0f - factor)) + (endValue * factor)), byte.MinValue, byte.MaxValue);
            return Unsafe.As<byte, T>(ref value);
        }

        if (typeof(T) == typeof(short))
        {
            short startValue = Unsafe.As<T, short>(ref start);
            short endValue = Unsafe.As<T, short>(ref end);
            short value = (short)Math.Clamp(MathF.Round((startValue * (1.0f - factor)) + (endValue * factor)), short.MinValue, short.MaxValue);
            return Unsafe.As<short, T>(ref value);
        }

        if (typeof(T) == typeof(float))
        {
            float startValue = Unsafe.As<T, float>(ref start);
            float endValue = Unsafe.As<T, float>(ref end);
            float value = (startValue * (1.0f - factor)) + (endValue * factor);
            return Unsafe.As<float, T>(ref value);
        }

        if (typeof(T) == typeof(Vector3))
        {
            Vector3 startValue = Unsafe.As<T, Vector3>(ref start);
            Vector3 endValue = Unsafe.As<T, Vector3>(ref end);
            Vector3 value = Vector3.Lerp(startValue, endValue, factor);
            return Unsafe.As<Vector3, T>(ref value);
        }

        if (typeof(T) == typeof(Quaternion))
        {
            Quaternion startValue = Unsafe.As<T, Quaternion>(ref start);
            Quaternion endValue = Unsafe.As<T, Quaternion>(ref end);
            Quaternion value = Quaternion.Normalize(Quaternion.Slerp(startValue, endValue, factor));
            return Unsafe.As<Quaternion, T>(ref value);
        }

        throw new NotSupportedException($"Unsupported M2 track interpolation type '{typeof(T).FullName}'.");
    }

    private static T InterpolateHermite<T>(TrackKeyFrame<T> start, TrackKeyFrame<T> end, float factor)
    {
        float t2 = factor * factor;
        float t3 = t2 * factor;
        float h00 = (2.0f * t3) - (3.0f * t2) + 1.0f;
        float h10 = t3 - (2.0f * t2) + factor;
        float h01 = (-2.0f * t3) + (3.0f * t2);
        float h11 = t3 - t2;
        return CombineCubic(start.Value, start.OutTangent, end.Value, end.InTangent, h00, h10, h01, h11);
    }

    private static T InterpolateBezier<T>(TrackKeyFrame<T> start, TrackKeyFrame<T> end, float factor)
    {
        float inv = 1.0f - factor;
        float b0 = inv * inv * inv;
        float b1 = 3.0f * inv * inv * factor;
        float b2 = 3.0f * inv * factor * factor;
        float b3 = factor * factor * factor;
        return CombineCubic(start.Value, start.OutTangent, end.InTangent, end.Value, b0, b1, b2, b3);
    }

    private static T CombineCubic<T>(T first, T second, T third, T fourth, float w0, float w1, float w2, float w3)
    {
        if (typeof(T) == typeof(byte))
        {
            byte firstValue = Unsafe.As<T, byte>(ref first);
            byte secondValue = Unsafe.As<T, byte>(ref second);
            byte thirdValue = Unsafe.As<T, byte>(ref third);
            byte fourthValue = Unsafe.As<T, byte>(ref fourth);
            byte value = (byte)Math.Clamp(MathF.Round((firstValue * w0) + (secondValue * w1) + (thirdValue * w2) + (fourthValue * w3)), byte.MinValue, byte.MaxValue);
            return Unsafe.As<byte, T>(ref value);
        }

        if (typeof(T) == typeof(short))
        {
            short firstValue = Unsafe.As<T, short>(ref first);
            short secondValue = Unsafe.As<T, short>(ref second);
            short thirdValue = Unsafe.As<T, short>(ref third);
            short fourthValue = Unsafe.As<T, short>(ref fourth);
            short value = (short)Math.Clamp(MathF.Round((firstValue * w0) + (secondValue * w1) + (thirdValue * w2) + (fourthValue * w3)), short.MinValue, short.MaxValue);
            return Unsafe.As<short, T>(ref value);
        }

        if (typeof(T) == typeof(float))
        {
            float firstValue = Unsafe.As<T, float>(ref first);
            float secondValue = Unsafe.As<T, float>(ref second);
            float thirdValue = Unsafe.As<T, float>(ref third);
            float fourthValue = Unsafe.As<T, float>(ref fourth);
            float value = (firstValue * w0) + (secondValue * w1) + (thirdValue * w2) + (fourthValue * w3);
            return Unsafe.As<float, T>(ref value);
        }

        if (typeof(T) == typeof(Vector3))
        {
            Vector3 firstValue = Unsafe.As<T, Vector3>(ref first);
            Vector3 secondValue = Unsafe.As<T, Vector3>(ref second);
            Vector3 thirdValue = Unsafe.As<T, Vector3>(ref third);
            Vector3 fourthValue = Unsafe.As<T, Vector3>(ref fourth);
            Vector3 value = (firstValue * w0) + (secondValue * w1) + (thirdValue * w2) + (fourthValue * w3);
            return Unsafe.As<Vector3, T>(ref value);
        }

        if (typeof(T) == typeof(Quaternion))
        {
            Quaternion firstValue = Unsafe.As<T, Quaternion>(ref first);
            Quaternion secondValue = Unsafe.As<T, Quaternion>(ref second);
            Quaternion thirdValue = Unsafe.As<T, Quaternion>(ref third);
            Quaternion fourthValue = Unsafe.As<T, Quaternion>(ref fourth);
            Quaternion value = NormalizeQuaternionWeighted(firstValue, secondValue, thirdValue, fourthValue, w0, w1, w2, w3);
            return Unsafe.As<Quaternion, T>(ref value);
        }

        throw new NotSupportedException($"Unsupported M2 cubic interpolation type '{typeof(T).FullName}'.");
    }

    private static Quaternion NormalizeQuaternionWeighted(Quaternion first, Quaternion second, Quaternion third, Quaternion fourth, float w0, float w1, float w2, float w3)
    {
        Vector4 weighted = (QuaternionToVector(first) * w0)
            + (QuaternionToVector(second) * w1)
            + (QuaternionToVector(third) * w2)
            + (QuaternionToVector(fourth) * w3);

        return Quaternion.Normalize(new Quaternion(weighted.X, weighted.Y, weighted.Z, weighted.W));
    }

    private static Vector4 QuaternionToVector(Quaternion value)
    {
        return new Vector4(value.X, value.Y, value.Z, value.W);
    }

    private static float NormalizeFixed16Alpha(short value)
    {
        return Math.Clamp(value / 32767.0f, 0.0f, 1.0f);
    }

    private readonly record struct TrackKeyFrame<T>(int Time, T Value, T InTangent, T OutTangent);
}