using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoSkyboxSummaryReader
{
    public static WmoSkyboxSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoSkyboxSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] payload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mosb);
        int length = Array.IndexOf(payload, (byte)0);
        if (length < 0)
            length = payload.Length;

        string skyboxName = System.Text.Encoding.UTF8.GetString(payload, 0, length);
        return new WmoSkyboxSummary(sourcePath, version, payload.Length, skyboxName);
    }

}
