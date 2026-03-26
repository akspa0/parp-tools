namespace WowViewer.Core.Files;

public sealed record WowFileDetection(string SourcePath, WowFileKind Kind, uint? Version)
{
    public bool IsKnown => Kind != WowFileKind.Unknown;
}