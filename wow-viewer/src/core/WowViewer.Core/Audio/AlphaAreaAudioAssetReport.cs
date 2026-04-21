namespace WowViewer.Core.Audio;

public enum AlphaAreaAudioAssetSource
{
    None = 0,
    Disk = 1,
    Archive = 2,
}

public enum AlphaAreaAudioAssetRole
{
    DaySequence = 0,
    NightSequence = 1,
    DlsFile = 2,
    UnderwaterDaySequence = 3,
    UnderwaterNightSequence = 4,
    UnderwaterDlsFile = 5,
}

public sealed record AlphaAreaAudioAssetProbe(
    AlphaAreaAudioAssetRole Role,
    string RequestedPath,
    string? ResolvedPath,
    AlphaAreaAudioAssetSource Source)
{
    public bool IsReferenced => !string.IsNullOrWhiteSpace(RequestedPath);

    public bool Exists => !string.IsNullOrWhiteSpace(ResolvedPath);
}

public sealed record AlphaAreaAudioBindingAssetReport(
    AlphaAreaAudioBinding Binding,
    AlphaAreaAudioAssetProbe DaySequence,
    AlphaAreaAudioAssetProbe NightSequence,
    AlphaAreaAudioAssetProbe DlsFile,
    AlphaAreaAudioAssetProbe UnderwaterDaySequence,
    AlphaAreaAudioAssetProbe UnderwaterNightSequence,
    AlphaAreaAudioAssetProbe UnderwaterDlsFile)
{
    public IEnumerable<AlphaAreaAudioAssetProbe> EnumerateAssets()
    {
        yield return DaySequence;
        yield return NightSequence;
        yield return DlsFile;
        yield return UnderwaterDaySequence;
        yield return UnderwaterNightSequence;
        yield return UnderwaterDlsFile;
    }
}
