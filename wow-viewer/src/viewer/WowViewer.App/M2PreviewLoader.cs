using System.Diagnostics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.App;

internal sealed class M2PreviewLoadRequest
{
    public string? InputPath { get; init; }

    public string? ArchiveRoot { get; init; }

    public string? VirtualPath { get; init; }

    public string? BuildLabel { get; init; }

    public int ProfileIndex { get; init; }

    public int SequenceIndex { get; init; }

    public int TimeMs { get; init; }

    public int VisualWidth { get; init; } = 384;

    public int VisualHeight { get; init; } = 384;

    public bool UsesArchiveSource =>
        !string.IsNullOrWhiteSpace(ArchiveRoot)
        && !string.IsNullOrWhiteSpace(VirtualPath);

    public string SourceLabel => UsesArchiveSource
        ? VirtualPath!
        : InputPath ?? string.Empty;

    public void Validate(bool requireSequenceIndex = true)
    {
        if (!UsesArchiveSource && string.IsNullOrWhiteSpace(InputPath))
            throw new ArgumentException("Provide --input <file.m2|file.mdx|file.mdl> or --archive-root <dir> with --virtual-path <path/to/file.m2|file.mdx|file.mdl>.");

        if (ProfileIndex < 0 || ProfileIndex > 99)
            throw new ArgumentOutOfRangeException(nameof(ProfileIndex), "Profile index must be in the range 0..99.");

        if (requireSequenceIndex && SequenceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(SequenceIndex), "Sequence index must be non-negative.");

        if (VisualWidth < 16 || VisualHeight < 16)
            throw new ArgumentOutOfRangeException(nameof(VisualWidth), "Visual size must be at least 16x16.");
    }

    public string DescribeSource()
    {
        string source = UsesArchiveSource
            ? $"archive:{ArchiveRoot}::{VirtualPath}"
            : Path.GetFullPath(InputPath!);

        return string.IsNullOrWhiteSpace(BuildLabel)
            ? source
            : $"{source} [{BuildLabel}]";
    }
}

internal sealed class M2PreviewLoadResult
{
    public M2PreviewLoadResult(
        M2PreviewLoadRequest request,
        M2RuntimeFrameResult frameResult,
        TimeSpan loadDuration)
    {
        Request = request ?? throw new ArgumentNullException(nameof(request));
        FrameResult = frameResult ?? throw new ArgumentNullException(nameof(frameResult));
        LoadDuration = loadDuration;
    }

    public M2PreviewLoadRequest Request { get; }

    public M2RuntimeFrameResult FrameResult { get; }

    public TimeSpan LoadDuration { get; }
}

internal static class M2PreviewLoader
{
    public static M2PreviewLoadResult Load(M2PreviewLoadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        request.Validate();

        Stopwatch stopwatch = Stopwatch.StartNew();
        M2RuntimeFrameResult result = BuildFrame(request);
        stopwatch.Stop();
        return new M2PreviewLoadResult(request, result, stopwatch.Elapsed);
    }

    private static M2RuntimeFrameResult BuildFrame(M2PreviewLoadRequest request)
    {
        byte[] modelBytes = ReadBytes(request);
        using MemoryStream modelStream = new(modelBytes, writable: false);
        M2ModelDocument model = M2ModelReader.Read(modelStream, request.SourceLabel);

        using MemoryStream geometryStream = new(modelBytes, writable: false);
        M2GeometryDocument geometry = M2GeometryReader.Read(geometryStream, request.SourceLabel);

        M2SkinProfileRuntimeState skinState = M2SkinProfileRuntime.Choose(model, request.ProfileIndex);
        byte[] skinBytes = ReadCompanionBytes(request, skinState.Selection.CompanionPath);
        using MemoryStream skinStream = new(skinBytes, writable: false);
        M2SkinDocument skin = M2SkinReader.Read(skinStream, skinState.Selection.CompanionPath);
        skinState = M2SkinProfileRuntime.Initialize(M2SkinProfileRuntime.Load(skinState, skin));

        M2StaticRenderModel renderModel = M2StaticRenderModelBuilder.Build(geometry, skinState);
        M2ExternalAnimationRuntimeState externalAnimationState = M2ExternalAnimationRuntime.Choose(model, request.SequenceIndex);
        if (externalAnimationState.UsesExternalFile && !string.IsNullOrWhiteSpace(externalAnimationState.CompanionPath))
        {
            byte[] animBytes = ReadCompanionBytes(request, externalAnimationState.CompanionPath);
            using MemoryStream animStream = new(animBytes, writable: false);
            M2ExternalAnimationDocument animation = M2AnimationReader.Read(animStream, externalAnimationState.CompanionPath);
            externalAnimationState = M2ExternalAnimationRuntime.Load(externalAnimationState, animation);
        }

        return M2RuntimeFramePipeline.Build(
            model,
            renderModel,
            request.SequenceIndex,
            request.TimeMs,
            externalAnimationState,
            visualWidth: request.VisualWidth,
            visualHeight: request.VisualHeight);
    }

    private static byte[] ReadBytes(M2PreviewLoadRequest request)
    {
        if (request.UsesArchiveSource)
            return ArchiveVirtualFileReader.ReadVirtualFile(request.VirtualPath!, [request.ArchiveRoot!], new ArchiveCatalogBootstrapOptions());

        if (!string.IsNullOrWhiteSpace(request.InputPath) && File.Exists(request.InputPath))
            return File.ReadAllBytes(request.InputPath);

        throw new FileNotFoundException($"Could not read M2 input '{request.SourceLabel}'.", request.SourceLabel);
    }

    private static byte[] ReadCompanionBytes(M2PreviewLoadRequest request, string companionPath)
    {
        if (request.UsesArchiveSource)
            return ArchiveVirtualFileReader.ReadVirtualFile(companionPath, [request.ArchiveRoot!], new ArchiveCatalogBootstrapOptions());

        string localPath = companionPath;
        if (!Path.IsPathRooted(localPath) && !string.IsNullOrWhiteSpace(request.InputPath))
        {
            string? inputDirectory = Path.GetDirectoryName(Path.GetFullPath(request.InputPath));
            if (!string.IsNullOrWhiteSpace(inputDirectory))
                localPath = Path.Combine(inputDirectory, Path.GetFileName(companionPath));
        }

        return File.ReadAllBytes(localPath);
    }
}