using System.Diagnostics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;
using WowViewer.Core.Runtime.Mdx;

namespace WowViewer.App;

internal sealed class MdxPreviewLoadRequest
{
    public string? InputPath { get; init; }

    public string? ArchiveRoot { get; init; }

    public string? VirtualPath { get; init; }

    public string? BuildLabel { get; init; }

    public string? LooseOverlayRoot { get; init; }

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
        MdxBoneFile bones,
        MdxHelperFile helpers,
        MdxAttachmentFile attachments,
        MdxEventFile events,
        MdxParticleEmitter2File particleEmitters,
        MdxRibbonEmitterFile ribbons,
        MdxEffectRuntimeState effectRuntimeState,
        MdxCameraFile cameras,
        MdxMaterialFile materials,
        MdxTextureAnimationFile textureAnimations,
        MdxGeosetAnimationFile geosetAnimations,
        TimeSpan loadDuration)
    {
        Request = request ?? throw new ArgumentNullException(nameof(request));
        Summary = summary ?? throw new ArgumentNullException(nameof(summary));
        Geometry = geometry ?? throw new ArgumentNullException(nameof(geometry));
        Bones = bones ?? throw new ArgumentNullException(nameof(bones));
        Helpers = helpers ?? throw new ArgumentNullException(nameof(helpers));
        Attachments = attachments ?? throw new ArgumentNullException(nameof(attachments));
        Events = events ?? throw new ArgumentNullException(nameof(events));
        ParticleEmitters = particleEmitters ?? throw new ArgumentNullException(nameof(particleEmitters));
        Ribbons = ribbons ?? throw new ArgumentNullException(nameof(ribbons));
        EffectRuntimeState = effectRuntimeState ?? throw new ArgumentNullException(nameof(effectRuntimeState));
        Cameras = cameras ?? throw new ArgumentNullException(nameof(cameras));
        Materials = materials ?? throw new ArgumentNullException(nameof(materials));
        TextureAnimations = textureAnimations ?? throw new ArgumentNullException(nameof(textureAnimations));
        GeosetAnimations = geosetAnimations ?? throw new ArgumentNullException(nameof(geosetAnimations));
        LoadDuration = loadDuration;
    }

    public MdxPreviewLoadRequest Request { get; }

    public MdxSummary Summary { get; }

    public MdxGeometryFile Geometry { get; }

    public MdxBoneFile Bones { get; }

    public MdxHelperFile Helpers { get; }

    public MdxAttachmentFile Attachments { get; }

    public MdxEventFile Events { get; }

    public MdxParticleEmitter2File ParticleEmitters { get; }

    public MdxRibbonEmitterFile Ribbons { get; }

    public MdxEffectRuntimeState EffectRuntimeState { get; }

    public MdxCameraFile Cameras { get; }

    public MdxMaterialFile Materials { get; }

    public MdxTextureAnimationFile TextureAnimations { get; }

    public MdxGeosetAnimationFile GeosetAnimations { get; }

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

        using MemoryStream boneStream = new(modelBytes, writable: false);
        MdxBoneFile bones = MdxBoneReader.Read(boneStream, request.SourceLabel);

        using MemoryStream helperStream = new(modelBytes, writable: false);
        MdxHelperFile helpers = MdxHelperReader.Read(helperStream, request.SourceLabel);

        using MemoryStream attachmentStream = new(modelBytes, writable: false);
        MdxAttachmentFile attachments = MdxAttachmentReader.Read(attachmentStream, request.SourceLabel);

        using MemoryStream eventStream = new(modelBytes, writable: false);
        MdxEventFile events = MdxEventReader.Read(eventStream, request.SourceLabel);

        using MemoryStream particleEmitterStream = new(modelBytes, writable: false);
        MdxParticleEmitter2File particleEmitters = MdxParticleEmitter2Reader.Read(particleEmitterStream, request.SourceLabel);

        using MemoryStream ribbonStream = new(modelBytes, writable: false);
        MdxRibbonEmitterFile ribbons = MdxRibbonEmitterReader.Read(ribbonStream, request.SourceLabel);

        MdxEffectRuntimeState effectRuntimeState = MdxEffectRuntimeEvaluator.Evaluate(summary, events, particleEmitters, ribbons, request.SequenceIndex, request.TimeMs);

        using MemoryStream cameraStream = new(modelBytes, writable: false);
        MdxCameraFile cameras = MdxCameraReader.Read(cameraStream, request.SourceLabel);

        using MemoryStream materialStream = new(modelBytes, writable: false);
        MdxMaterialFile materials = MdxMaterialReader.Read(materialStream, request.SourceLabel);

        using MemoryStream textureAnimationStream = new(modelBytes, writable: false);
        MdxTextureAnimationFile textureAnimations = MdxTextureAnimationReader.Read(textureAnimationStream, request.SourceLabel);

        using MemoryStream geosetAnimationStream = new(modelBytes, writable: false);
        MdxGeosetAnimationFile geosetAnimations = MdxGeosetAnimationReader.Read(geosetAnimationStream, request.SourceLabel);

        stopwatch.Stop();
        return new MdxPreviewLoadResult(request, summary, geometry, bones, helpers, attachments, events, particleEmitters, ribbons, effectRuntimeState, cameras, materials, textureAnimations, geosetAnimations, stopwatch.Elapsed);
    }

    private static byte[] ReadBytes(MdxPreviewLoadRequest request)
    {
        if (request.UsesArchiveSource)
            return VirtualAssetOverlayResolver.ReadVirtualFilePreferLoose(request.VirtualPath!, request.ArchiveRoot!, request.LooseOverlayRoot, request.BuildLabel);

        if (!string.IsNullOrWhiteSpace(request.InputPath) && File.Exists(request.InputPath))
            return File.ReadAllBytes(request.InputPath);

        throw new FileNotFoundException($"Could not read MDX input '{request.SourceLabel}'.", request.SourceLabel);
    }
}
