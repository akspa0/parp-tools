namespace WowViewer.App;

internal enum WowViewerWorkspaceMode
{
    StandaloneM2 = 0,
}

internal enum WowViewerAssetSourceKind
{
    ArchiveVirtualPath = 0,
    LocalFile = 1,
}

internal sealed class WowViewerAssetSource
{
    public WowViewerAssetSourceKind Kind { get; set; } = WowViewerAssetSourceKind.ArchiveVirtualPath;

    public string ArchiveRoot { get; set; } = string.Empty;

    public string VirtualPath { get; set; } = string.Empty;

    public string InputPath { get; set; } = string.Empty;

    public string BuildLabel { get; set; } = string.Empty;

    public bool UsesArchiveSource => Kind == WowViewerAssetSourceKind.ArchiveVirtualPath;

    public string Describe()
    {
        string source = UsesArchiveSource
            ? $"archive:{ArchiveRoot}::{VirtualPath}"
            : Path.GetFullPath(InputPath);

        return string.IsNullOrWhiteSpace(BuildLabel)
            ? source
            : $"{source} [{BuildLabel}]";
    }
}

internal sealed class WowViewerSession
{
    public WowViewerWorkspaceMode WorkspaceMode { get; set; } = WowViewerWorkspaceMode.StandaloneM2;

    public WowViewerAssetSource Source { get; set; } = new();

    public int ProfileIndex { get; set; }

    public int SequenceIndex { get; set; }

    public int TimeMs { get; set; }

    public int VisualSize { get; set; } = 384;

    public static WowViewerSession CreateDefault() => new();

    public void Normalize()
    {
        ProfileIndex = Math.Clamp(ProfileIndex, 0, 99);
        SequenceIndex = Math.Max(0, SequenceIndex);
        TimeMs = Math.Max(0, TimeMs);
        VisualSize = Math.Clamp(VisualSize, 128, 1024);
        Source ??= new WowViewerAssetSource();
    }

    public M2PreviewLoadRequest BuildM2PreviewRequest()
    {
        Normalize();
        return new M2PreviewLoadRequest
        {
            ArchiveRoot = Source.UsesArchiveSource ? Source.ArchiveRoot : null,
            VirtualPath = Source.UsesArchiveSource ? Source.VirtualPath : null,
            InputPath = Source.UsesArchiveSource ? null : Source.InputPath,
            BuildLabel = Source.BuildLabel,
            ProfileIndex = ProfileIndex,
            SequenceIndex = SequenceIndex,
            TimeMs = TimeMs,
            VisualWidth = VisualSize,
            VisualHeight = VisualSize,
        };
    }

    public void ApplyM2PreviewRequest(M2PreviewLoadRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        WorkspaceMode = WowViewerWorkspaceMode.StandaloneM2;
        Source.Kind = request.UsesArchiveSource ? WowViewerAssetSourceKind.ArchiveVirtualPath : WowViewerAssetSourceKind.LocalFile;
        Source.ArchiveRoot = request.ArchiveRoot ?? string.Empty;
        Source.VirtualPath = request.VirtualPath ?? string.Empty;
        Source.InputPath = request.InputPath ?? string.Empty;
        Source.BuildLabel = request.BuildLabel ?? string.Empty;
        ProfileIndex = request.ProfileIndex;
        SequenceIndex = request.SequenceIndex;
        TimeMs = request.TimeMs;
        VisualSize = Math.Clamp(Math.Max(request.VisualWidth, request.VisualHeight), 128, 1024);
        Normalize();
    }

    public string GetWorkspaceLabel()
    {
        return WorkspaceMode switch
        {
            WowViewerWorkspaceMode.StandaloneM2 => "Standalone M2",
            _ => "Unknown",
        };
    }
}