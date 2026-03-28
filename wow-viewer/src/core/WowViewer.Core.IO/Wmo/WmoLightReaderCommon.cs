using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

internal static class WmoLightReaderCommon
{
    internal const int LegacyEntrySize = 32;
    internal const int StandardEntrySize = 48;
    internal const float IdentityTolerance = 0.0001f;

    internal static IReadOnlyList<WmoLightDetail> ReadDetails(byte[] payload, string sourcePath, uint? version)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        int entrySize = InferEntrySize(payload.Length, version);
        if (payload.Length % entrySize != 0)
            throw new InvalidDataException($"MOLT payload size {payload.Length} in '{sourcePath}' is not divisible by inferred entry size {entrySize}.");

        int entryCount = payload.Length / entrySize;
        if (entryCount == 0)
            return [];

        List<WmoLightDetail> details = new(entryCount);
        for (int lightIndex = 0; lightIndex < entryCount; lightIndex++)
        {
            int entryOffset = lightIndex * entrySize;
            Quaternion? rotation = entrySize == StandardEntrySize
                ? ReadRotation(payload, entryOffset)
                : null;

            details.Add(new WmoLightDetail(
                lightIndex,
                entryOffset,
                entrySize,
                payload[entryOffset],
                payload[entryOffset + 1] != 0,
                BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(entryOffset + 4, sizeof(uint))),
                new Vector3(
                    BitConverter.ToSingle(payload, entryOffset + 8),
                    BitConverter.ToSingle(payload, entryOffset + 12),
                    BitConverter.ToSingle(payload, entryOffset + 16)),
                BitConverter.ToSingle(payload, entryOffset + 20),
                ReadAttenuationValue(payload, entryOffset, entrySize, endValue: false),
                ReadAttenuationValue(payload, entryOffset, entrySize, endValue: true),
                entrySize == StandardEntrySize ? ReadHeaderFlagsWord(payload, entryOffset) : null,
                rotation,
                rotation?.Length()));
        }

        return details;
    }

    internal static Quaternion ReadRotation(byte[] payload, int entryOffset)
    {
        return new Quaternion(
            BitConverter.ToSingle(payload, entryOffset + 24),
            BitConverter.ToSingle(payload, entryOffset + 28),
            BitConverter.ToSingle(payload, entryOffset + 32),
            BitConverter.ToSingle(payload, entryOffset + 36));
    }

    internal static ushort ReadHeaderFlagsWord(byte[] payload, int entryOffset)
    {
        return BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(entryOffset + 2, sizeof(ushort)));
    }

    internal static bool IsIdentityRotation(Quaternion rotation)
    {
        return MathF.Abs(rotation.X) <= IdentityTolerance
            && MathF.Abs(rotation.Y) <= IdentityTolerance
            && MathF.Abs(rotation.Z) <= IdentityTolerance
            && MathF.Abs(rotation.W - 1f) <= IdentityTolerance;
    }

    internal static float ReadAttenuationValue(byte[] payload, int entryOffset, int entrySize, bool endValue)
    {
        int offsetWithinEntry = entrySize switch
        {
            LegacyEntrySize => endValue ? 28 : 24,
            StandardEntrySize => endValue ? 44 : 40,
            _ => throw new InvalidDataException($"Unsupported MOLT entry size {entrySize}."),
        };

        return BitConverter.ToSingle(payload, entryOffset + offsetWithinEntry);
    }

    internal static int InferEntrySize(int payloadLength, uint? version)
    {
        if (payloadLength == 0)
            return version is not null && version <= 14 ? LegacyEntrySize : StandardEntrySize;

        if (version is not null && version <= 14)
            return LegacyEntrySize;

        if (version is not null && version >= 17)
            return StandardEntrySize;

        bool divisibleByLegacy = payloadLength % LegacyEntrySize == 0;
        bool divisibleByStandard = payloadLength % StandardEntrySize == 0;

        if (divisibleByStandard && !divisibleByLegacy)
            return StandardEntrySize;

        if (divisibleByLegacy && !divisibleByStandard)
            return LegacyEntrySize;

        if (divisibleByStandard)
            return StandardEntrySize;

        if (divisibleByLegacy)
            return LegacyEntrySize;

        return StandardEntrySize;
    }
}