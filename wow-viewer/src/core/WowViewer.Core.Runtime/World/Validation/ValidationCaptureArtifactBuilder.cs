using System.Buffers.Binary;
using System.Security.Cryptography;

namespace WowViewer.Core.Runtime.World.Validation;

public static class ValidationCaptureArtifactBuilder
{
    public static ValidationObjectMaskStrategy ResolveMaskStrategy(
        string? buildLabel,
        ValidationCaptureArtifactPolicy policy)
    {
        return IsEarlyBuild(buildLabel)
            ? policy.EarlyBuildStrategy
            : policy.LaterBuildStrategy;
    }

    public static ValidationCaptureArtifactOutputs Build(
        ValidationCaptureArtifactInputs inputs,
        ValidationCaptureArtifactPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(inputs);

        bool havePrimaryAndNoObjects = inputs.PrimaryRgbaPixels is not null && inputs.NoObjectsRgbaPixels is not null;
        bool haveObjectsOnly = inputs.ObjectsOnlyRgbaPixels is not null;

        byte[] objectVisibilityMask;
        ValidationObjectMaskStrategy preferredStrategy = ResolveMaskStrategy(inputs.BuildLabel, policy);
        ValidationObjectMaskStrategy strategy;
        if (preferredStrategy == ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff && havePrimaryAndNoObjects)
        {
            strategy = ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff;
            objectVisibilityMask = BuildDiffMask(inputs.Width, inputs.Height, inputs.PrimaryRgbaPixels!, inputs.NoObjectsRgbaPixels!, policy.DiffMaskThreshold);
        }
        else if (preferredStrategy == ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette && haveObjectsOnly)
        {
            strategy = ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette;
            objectVisibilityMask = BuildDirectObjectsOnlyMask(inputs.Width, inputs.Height, inputs.ObjectsOnlyRgbaPixels!, policy.ObjectsOnlyIntensityThreshold);
        }
        else if (havePrimaryAndNoObjects)
        {
            strategy = ValidationObjectMaskStrategy.PrimaryVsNoObjectsDiff;
            objectVisibilityMask = BuildDiffMask(inputs.Width, inputs.Height, inputs.PrimaryRgbaPixels!, inputs.NoObjectsRgbaPixels!, policy.DiffMaskThreshold);
        }
        else if (haveObjectsOnly)
        {
            strategy = ValidationObjectMaskStrategy.DirectObjectsOnlySilhouette;
            objectVisibilityMask = BuildDirectObjectsOnlyMask(inputs.Width, inputs.Height, inputs.ObjectsOnlyRgbaPixels!, policy.ObjectsOnlyIntensityThreshold);
        }
        else
        {
            throw new InvalidOperationException($"Cannot build object visibility mask for '{inputs.TileName}': need at least ObjectsOnlyRgbaPixels or (PrimaryRgbaPixels + NoObjectsRgbaPixels).");
        }

        byte[] noObjectMinimap = havePrimaryAndNoObjects
            ? BuildNoObjectMinimap(inputs.Width, inputs.Height, inputs.NoObjectsRgbaPixels!)
            : Array.Empty<byte>();

        return new ValidationCaptureArtifactOutputs(
            inputs.TileName,
            inputs.BuildLabel,
            inputs.Width,
            inputs.Height,
            strategy,
            objectVisibilityMask,
            ComputeHash(inputs.Width, inputs.Height, objectVisibilityMask),
            noObjectMinimap,
            ComputeHash(inputs.Width, inputs.Height, noObjectMinimap));
    }

    private static byte[] BuildNoObjectMinimap(int width, int height, byte[] noObjectsRgbaPixels)
    {
        byte[] noObjectMinimap = new byte[noObjectsRgbaPixels.Length];
        Buffer.BlockCopy(noObjectsRgbaPixels, 0, noObjectMinimap, 0, noObjectMinimap.Length);
        return noObjectMinimap;
    }

    private static bool IsEarlyBuild(string? buildLabel)
    {
        if (string.IsNullOrWhiteSpace(buildLabel))
            return false;

        ReadOnlySpan<char> span = buildLabel.AsSpan().Trim();
        int dotIndex = span.IndexOf('.');
        ReadOnlySpan<char> majorSpan = dotIndex >= 0 ? span[..dotIndex] : span;
        return int.TryParse(majorSpan, out int major) && major == 0;
    }

    private static byte[] BuildDirectObjectsOnlyMask(int width, int height, byte[] objectsOnlyRgbaPixels, int threshold)
    {
        byte[] mask = new byte[checked(width * height)];
        int rgbaOffset = 0;
        for (int index = 0; index < mask.Length; index++)
        {
            int intensity = Math.Max(objectsOnlyRgbaPixels[rgbaOffset], Math.Max(objectsOnlyRgbaPixels[rgbaOffset + 1], objectsOnlyRgbaPixels[rgbaOffset + 2]));
            mask[index] = (byte)(intensity > threshold ? 255 : 0);
            rgbaOffset += 4;
        }

        return mask;
    }

    private static byte[] BuildDiffMask(int width, int height, byte[] primaryRgbaPixels, byte[] noObjectsRgbaPixels, int threshold)
    {
        byte[] mask = new byte[checked(width * height)];
        int rgbaOffset = 0;
        for (int index = 0; index < mask.Length; index++)
        {
            int diffR = Math.Abs(primaryRgbaPixels[rgbaOffset] - noObjectsRgbaPixels[rgbaOffset]);
            int diffG = Math.Abs(primaryRgbaPixels[rgbaOffset + 1] - noObjectsRgbaPixels[rgbaOffset + 1]);
            int diffB = Math.Abs(primaryRgbaPixels[rgbaOffset + 2] - noObjectsRgbaPixels[rgbaOffset + 2]);
            int diff = Math.Max(diffR, Math.Max(diffG, diffB));
            mask[index] = (byte)(diff >= threshold ? 255 : 0);
            rgbaOffset += 4;
        }

        return mask;
    }

    private static string ComputeHash(int width, int height, byte[] payload)
    {
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> metadata = stackalloc byte[8];
        BinaryPrimitives.WriteInt32LittleEndian(metadata[0..4], width);
        BinaryPrimitives.WriteInt32LittleEndian(metadata[4..8], height);
        hash.AppendData(metadata);
        hash.AppendData(payload);
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }
}
