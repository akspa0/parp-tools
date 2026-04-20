using System.Numerics;

namespace WowViewer.App;

using WowViewer.Core.Runtime.World.Passes;

internal enum WowViewerWorkspaceMode
{
    StandaloneM2 = 0,
    StandaloneWmo = 1,
    StandaloneMdx = 2,
    WorldSession = 3,
    DatasetTooling = 4,
    ModelOutputs = 5,
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

    public string LooseOverlayRoot { get; set; } = string.Empty;

    public bool UsesArchiveSource => Kind == WowViewerAssetSourceKind.ArchiveVirtualPath;

    public string Describe()
    {
        if (UsesArchiveSource)
        {
            string overlay = string.IsNullOrWhiteSpace(LooseOverlayRoot)
                ? string.Empty
                : $" + loose:{LooseOverlayRoot}";

            return string.IsNullOrWhiteSpace(BuildLabel)
                ? $"archive:{ArchiveRoot}::{VirtualPath}{overlay}"
                : $"archive:{ArchiveRoot}::{VirtualPath} [{BuildLabel}]{overlay}";
        }

        if (string.IsNullOrWhiteSpace(InputPath))
            return "(no file selected)";

        return Path.GetFullPath(InputPath);
    }
}

internal sealed class WowViewerWorldSessionState
{
    public string ClientRoot { get; set; } = string.Empty;

    public string MapInput { get; set; } = string.Empty;

    public string BuildLabel { get; set; } = string.Empty;

    public string LooseOverlayRoot { get; set; } = string.Empty;

    public int TileX { get; set; } = -1;

    public int TileY { get; set; } = -1;

    public bool ShowWmos { get; set; } = true;

    public bool ShowDoodads { get; set; } = true;

    public bool ShowSky { get; set; } = true;

    public bool ShowWdl { get; set; } = true;

    public bool ShowTerrain { get; set; } = true;

    public bool ShowLiquid { get; set; } = true;

    public bool ShowOverlay { get; set; } = true;

    public void Normalize()
    {
        ClientRoot = ClientRoot?.Trim() ?? string.Empty;
        MapInput = MapInput?.Trim() ?? string.Empty;
        BuildLabel = BuildLabel?.Trim() ?? string.Empty;
        LooseOverlayRoot = LooseOverlayRoot?.Trim() ?? string.Empty;
        TileX = TileX < 0 ? -1 : Math.Clamp(TileX, 0, 63);
        TileY = TileY < 0 ? -1 : Math.Clamp(TileY, 0, 63);
    }

    public WowViewerWorldSessionOpenRequest BuildRequest()
    {
        Normalize();
        return new WowViewerWorldSessionOpenRequest(ClientRoot, MapInput, BuildLabel, LooseOverlayRoot);
    }

    public WowViewerWorldRuntimeFrameRequest BuildRuntimeFrameRequest()
    {
        Normalize();
        return new WowViewerWorldRuntimeFrameRequest(ClientRoot, MapInput, BuildLabel, LooseOverlayRoot, TileX, TileY, BuildPassOptions());
    }

    public WorldFramePassOptions BuildPassOptions()
    {
        Normalize();
        bool objectsVisible = ShowWmos || ShowDoodads;
        return new WorldFramePassOptions(
            objectsVisible,
            ShowWmos,
            ShowDoodads,
            ShowSky,
            ShowWdl,
            ShowTerrain,
            ShowLiquid,
            ShowOverlay);
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

        string loose = string.IsNullOrWhiteSpace(LooseOverlayRoot)
            ? string.Empty
            : $" loose={Path.GetFullPath(LooseOverlayRoot)}";

        bool usingDefaultLayers = ShowWmos && ShowDoodads && ShowSky && ShowWdl && ShowTerrain && ShowLiquid && ShowOverlay;
        string layerSummary = usingDefaultLayers
            ? string.Empty
            : $" layers[wmo={ShowWmos}, mdx={ShowDoodads}, sky={ShowSky}, wdl={ShowWdl}, terrain={ShowTerrain}, liquid={ShowLiquid}, overlay={ShowOverlay}]";

        return string.IsNullOrWhiteSpace(BuildLabel)
            ? $"{source} :: {map}{tile}{loose}{layerSummary}"
            : $"{source} :: {map}{tile} [{BuildLabel}]{loose}{layerSummary}";
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

    public WowViewerModelOutputState ModelOutput { get; set; } = new();

    public int ProfileIndex { get; set; }

    public int SequenceIndex { get; set; }

    public int TimeMs { get; set; }

    public int VisualSize { get; set; } = 384;

    public float M2CameraAzimuthDegrees { get; set; } = 45.0f;

    public float M2CameraElevationDegrees { get; set; } = 20.0f;

    public float M2CameraZoomFactor { get; set; } = 1.15f;

    public float M2CameraTargetOffsetX { get; set; }

    public float M2CameraTargetOffsetY { get; set; }

    public float M2CameraTargetOffsetZ { get; set; }

    public PreviewCameraMode MdxCameraMode { get; set; } = PreviewCameraMode.Frame;

    public string MdxCameraPreset { get; set; } = PreviewCameraSettings.DefaultPresetName;

    public float MdxCameraAzimuthDegrees { get; set; } = 35.0f;

    public float MdxCameraElevationDegrees { get; set; } = 25.0f;

    public float MdxCameraFieldOfViewDegrees { get; set; } = 60.0f;

    public float MdxCameraZoomFactor { get; set; } = 0.72f;

    public float MdxCameraTargetOffsetX { get; set; }

    public float MdxCameraTargetOffsetY { get; set; }

    public float MdxCameraTargetOffsetZ { get; set; }

    public static WowViewerSession CreateDefault() => new();

    public void Normalize()
    {
        ProfileIndex = Math.Clamp(ProfileIndex, 0, 99);
        SequenceIndex = Math.Max(0, SequenceIndex);
        TimeMs = Math.Max(0, TimeMs);
        VisualSize = Math.Clamp(VisualSize, 128, 1024);
        M2CameraAzimuthDegrees = float.IsFinite(M2CameraAzimuthDegrees) ? M2CameraAzimuthDegrees : 45.0f;
        M2CameraElevationDegrees = float.IsFinite(M2CameraElevationDegrees) ? Math.Clamp(M2CameraElevationDegrees, -89.0f, 89.0f) : 20.0f;
        M2CameraZoomFactor = float.IsFinite(M2CameraZoomFactor) ? Math.Clamp(M2CameraZoomFactor, 0.05f, 10.0f) : 1.15f;
        M2CameraTargetOffsetX = float.IsFinite(M2CameraTargetOffsetX) ? M2CameraTargetOffsetX : 0.0f;
        M2CameraTargetOffsetY = float.IsFinite(M2CameraTargetOffsetY) ? M2CameraTargetOffsetY : 0.0f;
        M2CameraTargetOffsetZ = float.IsFinite(M2CameraTargetOffsetZ) ? M2CameraTargetOffsetZ : 0.0f;
        if (!Enum.IsDefined(MdxCameraMode))
            MdxCameraMode = PreviewCameraMode.Frame;
        MdxCameraPreset = MdxCameraPreset?.Trim() ?? string.Empty;
        MdxCameraAzimuthDegrees = float.IsFinite(MdxCameraAzimuthDegrees) ? MdxCameraAzimuthDegrees : 35.0f;
        MdxCameraElevationDegrees = float.IsFinite(MdxCameraElevationDegrees) ? Math.Clamp(MdxCameraElevationDegrees, -89.0f, 89.0f) : 25.0f;
        MdxCameraFieldOfViewDegrees = float.IsFinite(MdxCameraFieldOfViewDegrees) ? Math.Clamp(MdxCameraFieldOfViewDegrees, 1.0f, 170.0f) : 60.0f;
        MdxCameraZoomFactor = float.IsFinite(MdxCameraZoomFactor) ? Math.Clamp(MdxCameraZoomFactor, 0.05f, 10.0f) : 0.72f;
        MdxCameraTargetOffsetX = float.IsFinite(MdxCameraTargetOffsetX) ? MdxCameraTargetOffsetX : 0.0f;
        MdxCameraTargetOffsetY = float.IsFinite(MdxCameraTargetOffsetY) ? MdxCameraTargetOffsetY : 0.0f;
        MdxCameraTargetOffsetZ = float.IsFinite(MdxCameraTargetOffsetZ) ? MdxCameraTargetOffsetZ : 0.0f;
        Source ??= new WowViewerAssetSource();
        World ??= new WowViewerWorldSessionState();
        ModelOutput ??= new WowViewerModelOutputState();
        Source.ArchiveRoot = Source.ArchiveRoot?.Trim() ?? string.Empty;
        Source.VirtualPath = Source.VirtualPath?.Trim() ?? string.Empty;
        Source.InputPath = Source.InputPath?.Trim() ?? string.Empty;
        Source.BuildLabel = Source.BuildLabel?.Trim() ?? string.Empty;
        Source.LooseOverlayRoot = Source.LooseOverlayRoot?.Trim() ?? string.Empty;
        World.Normalize();
        ModelOutput.Normalize();
    }

    public M2PreviewLoadRequest BuildM2PreviewRequest()
    {
        Normalize();
        return new M2PreviewLoadRequest
        {
            ArchiveRoot = Source.UsesArchiveSource ? Source.ArchiveRoot : null,
            VirtualPath = Source.UsesArchiveSource ? Source.VirtualPath : null,
            LooseOverlayRoot = Source.UsesArchiveSource ? Source.LooseOverlayRoot : null,
            InputPath = Source.UsesArchiveSource ? null : Source.InputPath,
            BuildLabel = Source.BuildLabel,
            ProfileIndex = ProfileIndex,
            SequenceIndex = SequenceIndex,
            TimeMs = TimeMs,
            VisualWidth = VisualSize,
            VisualHeight = VisualSize,
        };
    }

    public MdxPreviewLoadRequest BuildMdxPreviewRequest()
    {
        Normalize();
        return new MdxPreviewLoadRequest
        {
            ArchiveRoot = Source.UsesArchiveSource ? Source.ArchiveRoot : null,
            VirtualPath = Source.UsesArchiveSource ? Source.VirtualPath : null,
            LooseOverlayRoot = Source.UsesArchiveSource ? Source.LooseOverlayRoot : null,
            InputPath = Source.UsesArchiveSource ? null : Source.InputPath,
            BuildLabel = Source.BuildLabel,
            SequenceIndex = SequenceIndex,
            TimeMs = TimeMs,
            VisualWidth = VisualSize,
            VisualHeight = VisualSize,
            Camera = new PreviewCameraSettings
            {
                Mode = MdxCameraMode,
                PresetName = string.IsNullOrWhiteSpace(MdxCameraPreset) ? null : MdxCameraPreset,
                AzimuthDegrees = MdxCameraAzimuthDegrees,
                ElevationDegrees = MdxCameraElevationDegrees,
                FieldOfViewDegrees = MdxCameraFieldOfViewDegrees,
                ZoomFactor = MdxCameraZoomFactor,
                TargetOffset = GetMdxCameraTargetOffset(),
            },
        };
    }

    public Vector3 GetM2CameraTargetOffset()
    {
        Normalize();
        return new Vector3(M2CameraTargetOffsetX, M2CameraTargetOffsetY, M2CameraTargetOffsetZ);
    }

    public void SetM2CameraTargetOffset(Vector3 value)
    {
        M2CameraTargetOffsetX = value.X;
        M2CameraTargetOffsetY = value.Y;
        M2CameraTargetOffsetZ = value.Z;
        Normalize();
    }

    public Vector3 GetMdxCameraTargetOffset()
    {
        Normalize();
        return new Vector3(MdxCameraTargetOffsetX, MdxCameraTargetOffsetY, MdxCameraTargetOffsetZ);
    }

    public void SetMdxCameraTargetOffset(Vector3 value)
    {
        MdxCameraTargetOffsetX = value.X;
        MdxCameraTargetOffsetY = value.Y;
        MdxCameraTargetOffsetZ = value.Z;
        Normalize();
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
        Source.LooseOverlayRoot = request.LooseOverlayRoot ?? string.Empty;
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
            WowViewerWorkspaceMode.DatasetTooling => "Dataset Tooling",
            WowViewerWorkspaceMode.ModelOutputs => "Model Outputs",
            _ => "Unknown",
        };
    }

    public bool IsImplementedWorkspace()
    {
        return WorkspaceMode is WowViewerWorkspaceMode.StandaloneM2 or WowViewerWorkspaceMode.StandaloneMdx or WowViewerWorkspaceMode.WorldSession or WowViewerWorkspaceMode.DatasetTooling or WowViewerWorkspaceMode.ModelOutputs;
    }

    public bool HasBootstrapInput()
    {
        Normalize();
        return WorkspaceMode switch
        {
            WowViewerWorkspaceMode.StandaloneM2 => Source.UsesArchiveSource
                ? !string.IsNullOrWhiteSpace(Source.ArchiveRoot) && !string.IsNullOrWhiteSpace(Source.VirtualPath)
                : !string.IsNullOrWhiteSpace(Source.InputPath),
            WowViewerWorkspaceMode.StandaloneMdx => Source.UsesArchiveSource
                ? !string.IsNullOrWhiteSpace(Source.ArchiveRoot) && !string.IsNullOrWhiteSpace(Source.VirtualPath)
                : !string.IsNullOrWhiteSpace(Source.InputPath),
            WowViewerWorkspaceMode.WorldSession => World.HasBootstrapInput(),
            WowViewerWorkspaceMode.DatasetTooling => false,
            WowViewerWorkspaceMode.ModelOutputs => ModelOutput.HasInput(),
            _ => false,
        };
    }
}