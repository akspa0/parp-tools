using System.Diagnostics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.App;

internal sealed class WmoPreviewLoadRequest
{
    public string? InputPath { get; init; }

    public string? ArchiveRoot { get; init; }

    public string? VirtualPath { get; init; }

    public string? BuildLabel { get; init; }

    public string? LooseOverlayRoot { get; init; }

    public int VisualWidth { get; init; } = 384;

    public int VisualHeight { get; init; } = 384;

    public PreviewCameraSettings Camera { get; init; } = new();

    public bool UsesArchiveSource =>
        !string.IsNullOrWhiteSpace(ArchiveRoot)
        && !string.IsNullOrWhiteSpace(VirtualPath);

    public string SourceLabel => UsesArchiveSource
        ? VirtualPath!
        : InputPath ?? string.Empty;

    public void Validate()
    {
        if (!UsesArchiveSource && string.IsNullOrWhiteSpace(InputPath))
            throw new ArgumentException("Provide a local WMO input path or an archive root with virtual path.");

        if (VisualWidth < 16 || VisualHeight < 16)
            throw new ArgumentOutOfRangeException(nameof(VisualWidth), "Visual size must be at least 16x16.");

        Camera.Validate();
    }
}

internal sealed class WmoPreviewLoadResult
{
    public WmoPreviewLoadResult(WmoPreviewLoadRequest request, WmoRenderDocument document, TimeSpan loadDuration)
    {
        Request = request ?? throw new ArgumentNullException(nameof(request));
        Document = document ?? throw new ArgumentNullException(nameof(document));
        LoadDuration = loadDuration;
    }

    public WmoPreviewLoadRequest Request { get; }

    public WmoRenderDocument Document { get; }

    public TimeSpan LoadDuration { get; }
}

internal static class WmoPreviewLoader
{
    public static WmoPreviewLoadResult Load(WmoPreviewLoadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        request.Validate();

        Stopwatch stopwatch = Stopwatch.StartNew();
        byte[] wmoBytes = ReadBytes(request);
        using MemoryStream stream = new(wmoBytes, writable: false);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, request.SourceLabel);
        stopwatch.Stop();
        return new WmoPreviewLoadResult(request, document, stopwatch.Elapsed);
    }

    private static byte[] ReadBytes(WmoPreviewLoadRequest request)
    {
        if (request.UsesArchiveSource)
            return VirtualAssetOverlayResolver.ReadVirtualFilePreferLoose(request.VirtualPath!, request.ArchiveRoot!, request.LooseOverlayRoot);

        if (!string.IsNullOrWhiteSpace(request.InputPath) && File.Exists(request.InputPath))
            return File.ReadAllBytes(request.InputPath);

        throw new FileNotFoundException($"Could not read WMO input '{request.SourceLabel}'.", request.SourceLabel);
    }
}