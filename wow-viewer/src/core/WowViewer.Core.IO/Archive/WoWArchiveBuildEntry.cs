namespace WowViewer.Core.IO.Archive;

public sealed record WoWArchiveBuildEntry(
    string BuildVersion,
    string Platform,
    string Locale,
    string Era,
    string InnerPath,
    WoWArchiveBuildStatus Status);
