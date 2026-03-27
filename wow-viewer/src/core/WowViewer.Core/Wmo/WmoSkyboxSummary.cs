namespace WowViewer.Core.Wmo;

public sealed class WmoSkyboxSummary
{
    public WmoSkyboxSummary(string sourcePath, uint? version, int payloadSizeBytes, string skyboxName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(skyboxName);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        SkyboxName = skyboxName;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public string SkyboxName { get; }
}
