using System.Diagnostics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.App;

internal sealed class MdxPreviewLoadRequest
{
    public string? InputPath { get; init; }

    public string? ArchiveRoot { get; init; }

    public string? VirtualPath { get; init; }

    public string? BuildLabel { get; init; }

    public int SequenceIndex { get; init; }

    public int TimeMs { get; init; }

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
            throw new ArgumentException("Provide --input <file.mdx> or --archive-root <dir> with --virtual-path <path/to/file.mdx>.");

        if (SequenceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(SequenceIndex), "Sequence index must be non-negative.");

        if (TimeMs < 0)
            throw new ArgumentOutOfRangeException(nameof(TimeMs), "Time must be non-negative.");

        if (VisualWidth < 16 || VisualHeight < 16)
            throw new ArgumentOutOfRangeException(nameof(VisualWidth), "Visual size must be at least 16x16.");

        Camera.Validate();
    }
}

internal sealed class MdxPreviewLoadResult
{
    public MdxPreviewLoadResult(
        MdxPreviewLoadRequest request,
        MdxSummary summary,
        MdxGeometryFile geometry,
        TimeSpan loadDuration)
    {
        Request = request ?? throw new ArgumentNullException(nameof(request));
        Summary = summary ?? throw new ArgumentNullException(nameof(summary));
        Geometry = geometry ?? throw new ArgumentNullException(nameof(geometry));
        LoadDuration = loadDuration;
    }

    public MdxPreviewLoadRequest Request { get; }

    public MdxSummary Summary { get; }

    public MdxGeometryFile Geometry { get; }

    public TimeSpan LoadDuration { get; }
}

internal static class MdxPreviewLoader
{
    public static MdxPreviewLoadResult Load(MdxPreviewLoadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        request.Validate();

        Stopwatch stopwatch = Stopwatch.StartNew();
        byte[] modelBytes = ReadBytes(request);

        using MemoryStream summaryStream = new(modelBytes, writable: false);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, request.SourceLabel);

        using MemoryStream geometryStream = new(modelBytes, writable: false);
        MdxGeometryFile geometry = MdxGeometryReader.Read(geometryStream, request.SourceLabel);

        stopwatch.Stop();
        return new MdxPreviewLoadResult(request, summary, geometry, stopwatch.Elapsed);
    }

    private static byte[] ReadBytes(MdxPreviewLoadRequest request)
    {
        if (request.UsesArchiveSource)
            return ArchiveVirtualFileReader.ReadVirtualFile(request.VirtualPath!, [request.ArchiveRoot!], new ArchiveCatalogBootstrapOptions());

        if (!string.IsNullOrWhiteSpace(request.InputPath) && File.Exists(request.InputPath))
            return File.ReadAllBytes(request.InputPath);

        throw new FileNotFoundException($"Could not read MDX input '{request.SourceLabel}'.", request.SourceLabel);
    }
}
