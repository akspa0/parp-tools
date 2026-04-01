namespace WowViewer.Core.M2;

public sealed class M2SkinProfileSelection
{
    public M2SkinProfileSelection(int profileIndex, string companionPath)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(profileIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(profileIndex, 99);
        ArgumentException.ThrowIfNullOrWhiteSpace(companionPath);

        ProfileIndex = profileIndex;
        CompanionPath = M2ModelIdentity.NormalizePath(companionPath);
    }

    public int ProfileIndex { get; }

    public string CompanionPath { get; }
}