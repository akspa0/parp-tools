namespace WowViewer.App;

internal enum WowViewerWorkspaceMode
{
    StandaloneM2 = 0,
    StandaloneWmo = 1,
    StandaloneMdx = 2,
    WorldSession = 3,
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

internal sealed class WowViewerWorldSessionState
{
    public string ClientRoot { get; set; } = string.Empty;

    public string MapInput { get; set; } = string.Empty;

    public string BuildLabel { get; set; } = string.Empty;

    public int TileX { get; set; } = -1;

    public int TileY { get; set; } = -1;

    public void Normalize()
    {
        ClientRoot = ClientRoot?.Trim() ?? string.Empty;
        MapInput = MapInput?.Trim() ?? string.Empty;
        BuildLabel = BuildLabel?.Trim() ?? string.Empty;
        TileX = TileX < 0 ? -1 : Math.Clamp(TileX, 0, 63);
        TileY = TileY < 0 ? -1 : Math.Clamp(TileY, 0, 63);
    }

    public WowViewerWorldSessionOpenRequest BuildRequest()
    {
        Normalize();
        return new WowViewerWorldSessionOpenRequest(ClientRoot, MapInput, BuildLabel);
    }

    public WowViewerWorldRuntimeFrameRequest BuildRuntimeFrameRequest()
    {
        Normalize();
        return new WowViewerWorldRuntimeFrameRequest(ClientRoot, MapInput, BuildLabel, TileX, TileY);
    }

    public string Describe()
    {
        string source = string.IsNullOrWhiteSpace(ClientRoot)
            ? "<client root not set>"
            : Path.GetFullPath(ClientRoot);

        string map = string.IsNullOrWhiteSpace(MapInput)
            ? "<map not set>"
            : MapInput;

        string tile = TileX >= 0 && TileY >= 0
            ? $" tile=({TileX},{TileY})"
            : " tile=<auto>";

        return string.IsNullOrWhiteSpace(BuildLabel)
            ? $"{source} :: {map}{tile}"
            : $"{source} :: {map}{tile} [{BuildLabel}]";
    }

    public bool HasBootstrapInput()
    {
        Normalize();
        return !string.IsNullOrWhiteSpace(ClientRoot) && !string.IsNullOrWhiteSpace(MapInput);
    }
}

internal sealed class WowViewerSession
{
    public WowViewerWorkspaceMode WorkspaceMode { get; set; } = WowViewerWorkspaceMode.StandaloneM2;

    public WowViewerAssetSource Source { get; set; } = new();

    public WowViewerWorldSessionState World { get; set; } = new();

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
        World ??= new WowViewerWorldSessionState();
        World.Normalize();
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
            WowViewerWorkspaceMode.StandaloneWmo => "Standalone WMO",
            WowViewerWorkspaceMode.StandaloneMdx => "Standalone MDX",
            WowViewerWorkspaceMode.WorldSession => "World Session",
            _ => "Unknown",
        };
    }

    public bool IsImplementedWorkspace()
    {
        return WorkspaceMode is WowViewerWorkspaceMode.StandaloneM2 or WowViewerWorkspaceMode.WorldSession;
    }

    public bool HasBootstrapInput()
    {
        Normalize();
        return WorkspaceMode switch
        {
            WowViewerWorkspaceMode.StandaloneM2 => Source.UsesArchiveSource
                ? !string.IsNullOrWhiteSpace(Source.ArchiveRoot) && !string.IsNullOrWhiteSpace(Source.VirtualPath)
                : !string.IsNullOrWhiteSpace(Source.InputPath),
            WowViewerWorkspaceMode.WorldSession => World.HasBootstrapInput(),
            _ => false,
        };
    }
}