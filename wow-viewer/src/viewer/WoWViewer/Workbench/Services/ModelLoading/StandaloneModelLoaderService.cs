using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Standalone model loading: M2/MDX/WMO from disk or data source, container probing, skin/companion resolution, M2->MDX fallback, camera-path models and character customization.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class StandaloneModelLoaderService
{
    private readonly IViewerAppHost _host;

    internal StandaloneModelLoaderService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref bool _autoFrameModelOnLoad => ref _host.AutoFrameModelOnLoad;
    private ref Camera _camera => ref _host.Camera;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref GL _gl => ref _host.Gl;
    private ref string? _lastVirtualPath => ref _host.LastVirtualPath;
    private ref float _lastWorldSceneCameraPitch => ref _host.LastWorldSceneCameraPitch;
    private ref Vector3 _lastWorldSceneCameraPosition => ref _host.LastWorldSceneCameraPosition;
    private ref float _lastWorldSceneCameraYaw => ref _host.LastWorldSceneCameraYaw;
    private ref string? _lastWorldSceneWdtPath => ref _host.LastWorldSceneWdtPath;
    private ref string? _loadedFileName => ref _host.LoadedFileName;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ref M2StaticRenderModel? _loadedM2Runtime => ref _host.LoadedM2Runtime;
    private ref MdxFile? _loadedMdx => ref _host.LoadedMdx;
    private ref WmoV14ToV17Converter.WmoV14Data? _loadedWmo => ref _host.LoadedWmo;
    private ref Rendering.LoadingScreen? _loadingScreen => ref _host.LoadingScreen;
    private HashSet<string> _loggedStandaloneMissingSkinPaths => _host.LoggedStandaloneMissingSkinPaths;
    private ref string _modelInfo => ref _host.ModelInfo;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private SqlSpawnStreamingService _sqlSpawnStreaming => _host.SqlSpawnStreaming;
    private Dictionary<string, string?> _standaloneSkinPathCache => _host.StandaloneSkinPathCache;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref ReplaceableTextureResolver? _texResolver => ref _host.TexResolver;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private WdlPreviewService _wdlPreview => _host.WdlPreview;
    private ref IWindow _window => ref _host.Window;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void FrameCurrentModel() => _host.FrameCurrentModel();
    private void LoadWdtTerrain(string wdtPath) => _host.LoadWdtTerrain(wdtPath);
    private string? TryGetLoadedLocalWdtPath() => _host.TryGetLoadedLocalWdtPath();


    private enum ModelContainerKind
    {
        Unknown,
        Mdlx,
        Md20,
        Md21,
    }
    private string? _standaloneCharacterCustomizationModelPath;
    private readonly List<int> _standaloneCharacterHairVariationIds = new();
    private readonly List<int> _standaloneCharacterFacialHairVariationIds = new();
    private int _standaloneCharacterHairVariationOverride = -1;
    private int _standaloneCharacterFacialHairVariationOverride = -1;
    private bool _preserveStandaloneCharacterCustomizationOnNextLoad;

    private void CaptureWorldReturnState()
    {
        if (_worldScene == null || _terrainManager == null)
            return;

        string? wdtPath = TryGetLoadedLocalWdtPath();
        if (string.IsNullOrWhiteSpace(wdtPath))
            return;

        _lastWorldSceneWdtPath = wdtPath;
        _lastWorldSceneCameraPosition = _camera.Position;
        _lastWorldSceneCameraYaw = _camera.Yaw;
        _lastWorldSceneCameraPitch = _camera.Pitch;
    }

    internal void LoadFileFromDisk(string filePath)
    {
        _loadedFilePath = filePath;
        _loadedFileName = Path.GetFileName(filePath);
        _window.Title = $"{ViewerProductName} - {_loadedFileName}";

        var ext = Path.GetExtension(filePath).ToLowerInvariant();
        string dir = Path.GetDirectoryName(filePath) ?? ".";

        if (ext != ".wdt")
            CaptureWorldReturnState();

        try
        {
            _renderer?.Dispose();
            _renderer = null;

            switch (ext)
            {
                case ".mdx":
                case ".mdl":
                case ".m2":
                    var modelBytes = File.ReadAllBytes(filePath);
                    LoadModelFromBytesWithContainerProbe(modelBytes, filePath, dir, "Disk");
                    break;

                case ".wmo":
                    LoadWmoFromDisk(filePath, dir);
                    break;

                case ".wdt":
                    LoadWdtTerrain(filePath);
                    break;

                default:
                    _statusMessage = $"Unsupported format: {ext}";
                    break;
            }
        }
        catch (Exception ex)
        {
            LogLoadFailure("DiskLoad", filePath, ex);
            _statusMessage = $"Failed to load: {BuildStatusExceptionSummary(ex)}";
            _modelInfo = "";
        }
    }

    /// <summary>
    /// Load an M2 model from disk using Warcraft.NET parser + companion .skin geometry.
    /// </summary>
    private void LoadM2FromDisk(string filePath, string dir)
    {
        var m2Bytes = File.ReadAllBytes(filePath);
        LoadM2FromBytes(m2Bytes, filePath, dir);
    }

    /// <summary>
    /// Load an M2 model from raw bytes using Warcraft.NET model/skin support.
    /// </summary>
    private void LoadM2FromBytes(byte[] m2Bytes, string originalPath, string dir)
    {
        string resolvedModelPath = ResolveStandaloneCanonicalModelPath(originalPath);

        // Detect era FIRST — 1.0.0 and 1.12.1 models have embedded geometry and don't
        // need a format profile or external .skin files. Only WotLK+ (264+) needs the
        // profile registry + external .skin companion path.
        M2Era1121EraTag detectedEra = M2ModelReaderDispatcher.DetectEra(m2Bytes.AsSpan(), resolvedModelPath);

        if (TryLoadStandaloneCameraPathM2(m2Bytes, resolvedModelPath))
        {
            CaptureWorldReturnState();
            return;
        }

        if (detectedEra is M2Era1121EraTag.Md20_1X_V100_Era100)
        {
            try
            {
                M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildEra100StaticRenderModel(m2Bytes, resolvedModelPath);
                LoadM2RuntimeModel(runtimeModel, modelDir: dir, virtualPath: resolvedModelPath);
                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Loaded native 1.0.0 M2 geometry for {Path.GetFileName(originalPath)} (era={detectedEra.ToDisplayString()})");
                _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                return;
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] Embedded 1.0.0 fallback failed for {Path.GetFileName(originalPath)}: {ex.Message}");
                throw new InvalidDataException(
                    $"Failed to load embedded 1.0.0 geometry for {Path.GetFileName(originalPath)}: {ex.Message}", ex);
            }
        }

        if (detectedEra is M2Era1121EraTag.Md20_1X_V100 or M2Era1121EraTag.Md20_1X_V101)
        {
            try
            {
                var embeddedMdx = WarcraftNetM2Adapter.BuildRuntimeModel(m2Bytes, null, resolvedModelPath, _dbcBuild);
                LoadMdxModel(embeddedMdx, dir, resolvedModelPath, isM2AdapterModel: true);
                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Loaded embedded 1.12.1 geometry for {Path.GetFileName(originalPath)} (era={detectedEra.ToDisplayString()})");
                _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                return;
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] Embedded 1.12.1 fallback failed for {Path.GetFileName(originalPath)}: {ex.Message}");
                throw new InvalidDataException(
                    $"Failed to load embedded 1.12.1 geometry for {Path.GetFileName(originalPath)}: {ex.Message}", ex);
            }
        }

        // WotLK+ (264+) path: requires a format profile + external .skin companion.
        var profile = FormatProfileRegistry.ResolveModelProfile(_dbcBuild);
        if (profile == null)
        {
            string buildLabel = string.IsNullOrWhiteSpace(_dbcBuild) ? "unknown" : _dbcBuild;
            throw new InvalidDataException(
                $"Standalone M2-family loading is not yet implemented for build {buildLabel}. " +
                "This asset is an M2-family model; .mdx/.mdl is not a substitute for 1.x M2 data. " +
                "Use the version-specific M2 reader path or load a supported client build.");
        }

        WarcraftNetM2Adapter.ValidateModelProfile(m2Bytes, resolvedModelPath, profile, _dbcBuild);

        var candidatePaths = new List<string>(WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath));

        Exception? lastError = null;
        bool anySkinFound = false;
        bool triedBestSkinPath = false;

        while (true)
        {
            foreach (var skinPath in candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                byte[]? skinBytes = ReadStandaloneFileData(skinPath);
                if (skinBytes == null || skinBytes.Length == 0)
                    continue;

                anySkinFound = true;

                try
                {
                    ViewerLog.Trace($"[M2] Trying skin: {skinPath} ({skinBytes.Length} bytes)");
                    M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(m2Bytes, skinBytes, resolvedModelPath, skinPath);
                    MdxFile? adaptedMdx = null;
                    try
                    {
                        adaptedMdx = WarcraftNetM2Adapter.BuildRuntimeModel(m2Bytes, skinBytes, resolvedModelPath, _dbcBuild);
                    }
                    catch (Exception adapterEx)
                    {
                        ViewerLog.Debug(ViewerLog.Category.Mdx,
                            $"[M2] M2->MDX adapter fallback failed for {Path.GetFileName(resolvedModelPath)}: {adapterEx.Message} (native renderer will be used)");
                    }
                    LoadM2RuntimeModel(runtimeModel, adaptedMdx, dir, resolvedModelPath);
                    CaptureWorldReturnState();
                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Selected skin for {Path.GetFileName(originalPath)}: {skinPath} ({skinBytes.Length} bytes)");
                    _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                    return;
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Skin candidate failed for {Path.GetFileName(originalPath)}: {skinPath} ({ex.Message})");
                }
            }

            if (triedBestSkinPath)
                break;

            triedBestSkinPath = true;
            string? bestSkinPath = ResolveBestStandaloneSkinPath(resolvedModelPath);
            if (string.IsNullOrWhiteSpace(bestSkinPath))
                break;

            candidatePaths.Add(bestSkinPath);
        }

        if (!anySkinFound && string.Equals(FormatProfileRegistry.ResolveModelProfile(_dbcBuild)?.ProfileId, FormatProfileRegistry.M2Profile3018303.ProfileId, StringComparison.Ordinal))
        {
            try
            {
                var embeddedMdx = WarcraftNetM2Adapter.BuildRuntimeModel(m2Bytes, null, resolvedModelPath, _dbcBuild);
                LoadMdxModel(embeddedMdx, dir, resolvedModelPath, isM2AdapterModel: true);
                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Loaded embedded root-profile geometry for {Path.GetFileName(originalPath)} after no external .skin resolved");
                _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                return;
            }
            catch (Exception ex)
            {
                lastError = ex;
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] Embedded root-profile fallback failed for {Path.GetFileName(originalPath)}: {ex.Message}");
            }
        }

        if (WarcraftNetM2Adapter.IsMd20(m2Bytes))
        {
            byte[]? convertedBytes = ConvertStandaloneM2ToMdx(m2Bytes, resolvedModelPath);
            if (convertedBytes != null && convertedBytes.Length > 0)
            {
                try
                {
                    using var convertedStream = new MemoryStream(convertedBytes);
                    var convertedMdx = MdxFile.Load(convertedStream);
                    if (WarcraftNetM2Adapter.HasRenderableGeometry(convertedMdx))
                    {
                        LoadMdxModel(convertedMdx, dir, resolvedModelPath, isM2AdapterModel: true);
                        ViewerLog.Info(ViewerLog.Category.Mdx,
                            $"[M2] Falling back to M2->MDX conversion for {Path.GetFileName(originalPath)} after adapter failure");
                        _statusMessage = $"Loaded M2: {Path.GetFileName(originalPath)}";
                        return;
                    }

                    lastError = new InvalidDataException(
                        $"M2->MDX fallback produced no renderable geometry for {Path.GetFileName(originalPath)} ({WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)})");
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Rejecting converted fallback for {Path.GetFileName(originalPath)}: {WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)}");
                }
                catch (Exception ex)
                {
                    lastError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Converted fallback load failed for {Path.GetFileName(originalPath)}: {ex.Message}");
                }
            }
        }

        if (!anySkinFound)
        {
            bool isTracedPreRelease301 = string.Equals(
                FormatProfileRegistry.ResolveModelProfile(_dbcBuild)?.ProfileId,
                FormatProfileRegistry.M2Profile3018303.ProfileId,
                StringComparison.Ordinal);

            InvalidDataException missingSkinError = isTracedPreRelease301
                ? new InvalidDataException(
                    $"No external .skin resolved for pre-release M2: {Path.GetFileName(originalPath)}. wow.exe 3.0.1.8303 traces root-contained profile tables for CM2Shared; WoWViewer root-profile geometry parsing is still incomplete.")
                : new InvalidDataException($"Missing companion .skin for M2: {Path.GetFileName(originalPath)}");

            if (_loggedStandaloneMissingSkinPaths.Add(resolvedModelPath))
            {
                ViewerLog.Error(ViewerLog.Category.Mdx,
                    $"[M2] {missingSkinError.Message} (build={_dbcBuild ?? "unknown"}, resolved='{resolvedModelPath}', candidateCount={candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase).Count()})");
            }

            throw missingSkinError;
        }

        var adaptFailure = new InvalidDataException(
            $"Failed to adapt M2 with available .skin candidates: {Path.GetFileName(originalPath)}",
            lastError);
        ViewerLog.Error(ViewerLog.Category.Mdx,
            $"[M2] {adaptFailure.Message} for '{resolvedModelPath}' (build={_dbcBuild ?? "unknown"}): {DescribeExceptionChain(lastError ?? adaptFailure)}");
        throw adaptFailure;
    }

    private bool TryLoadStandaloneCameraPathM2(byte[] m2Bytes, string resolvedModelPath)
    {
        if (!WarcraftNetM2Adapter.IsMd20(m2Bytes))
            return false;

        try
        {
            using MemoryStream stream = new(m2Bytes, writable: false);
            M2ModelDocument model = M2ModelReader.Read(stream, resolvedModelPath);
            if (!M2CameraPathOverlayBuilder.CanBuild(model))
                return false;

            M2CameraPathVisualization visualization = M2CameraPathOverlayBuilder.Build(model);
            LoadStandaloneCameraPathModel(model, visualization, resolvedModelPath);
            ViewerLog.Info(ViewerLog.Category.Mdx,
                $"[M2] Loaded camera-path visualization for {Path.GetFileName(resolvedModelPath)}: cameras={model.CameraCount}, sequences={model.SequenceCount}");
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] Camera-path probe skipped for {Path.GetFileName(resolvedModelPath)}: {ex.Message}");
            return false;
        }
    }

    private static string DescribeExceptionChain(Exception ex, int maxDepth = 6)
    {
        var parts = new List<string>();
        Exception? current = ex;
        while (current != null && parts.Count < maxDepth)
        {
            parts.Add($"{current.GetType().Name}: {current.Message}");
            current = current.InnerException;
        }

        return string.Join(" -> ", parts);
    }

    private static string BuildStatusExceptionSummary(Exception ex)
    {
        string summary = DescribeExceptionChain(ex, 3);
        return summary.Length <= 240 ? summary : summary[..237] + "...";
    }

    private void LogLoadFailure(string operation, string sourcePath, Exception ex, byte[]? modelBytes = null)
    {
        string byteSummary = modelBytes == null
            ? string.Empty
            : $" magic={GetModelMagicLabel(modelBytes)} md20Version={GetMd20VersionLabel(modelBytes)} bytes={modelBytes.Length}";
        ViewerLog.Error(ViewerLog.Category.General,
            $"[{operation}] Failed for '{sourcePath}': {DescribeExceptionChain(ex)}{byteSummary}");
    }

    private void LogDataSourceReadFailure(string requestedPath, string resolvedPath, string ext)
    {
        bool requestedExists = false;
        bool resolvedExists = false;
        try { requestedExists = _dataSource?.FileExists(requestedPath) ?? false; } catch { }
        try { resolvedExists = _dataSource?.FileExists(resolvedPath) ?? false; } catch { }

        string indexedRequested = "-";
        string indexedResolved = "-";
        if (_dataSource is MpqDataSource mpqDataSource)
        {
            try
            {
                indexedRequested = mpqDataSource.FindInFileSet(requestedPath.Replace('/', '\\')) ?? "-";
                indexedResolved = mpqDataSource.FindInFileSet(resolvedPath.Replace('/', '\\')) ?? "-";
            }
            catch { }
        }

        ViewerLog.Error(ViewerLog.Category.General,
            $"[DataSourceRead] Failed to read requested='{requestedPath}' resolved='{resolvedPath}' ext={ext} source={_dataSource?.GetType().Name ?? "<null>"} exists(requested)={requestedExists} exists(resolved)={resolvedExists} indexedRequested='{indexedRequested}' indexedResolved='{indexedResolved}'");
    }

    private static ModelContainerKind DetectModelContainer(byte[] modelBytes)
    {
        if (modelBytes.Length < 4) return ModelContainerKind.Unknown;

        uint magic = BitConverter.ToUInt32(modelBytes, 0);
        if (magic == MdxHeaders.MAGIC) return ModelContainerKind.Mdlx;
        if (magic == 0x3032444D) return ModelContainerKind.Md20; // "MD20"
        if (magic == 0x3132444D) return ModelContainerKind.Md21; // "MD21"

        return ModelContainerKind.Unknown;
    }

    private static string GetModelMagicLabel(byte[] modelBytes)
    {
        if (modelBytes.Length < 4) return "<short>";

        uint magic = BitConverter.ToUInt32(modelBytes, 0);
        return magic switch
        {
            MdxHeaders.MAGIC => "MDLX",
            0x3032444D => "MD20",
            0x3132444D => "MD21",
            _ => $"0x{magic:X8}"
        };
    }

    private static string GetMd20VersionLabel(byte[] modelBytes)
    {
        if (modelBytes.Length < 8 || BitConverter.ToUInt32(modelBytes, 0) != 0x3032444D)
            return "n/a";

        uint version = BitConverter.ToUInt32(modelBytes, 4);
        return $"0x{version:X}";
    }

    private void LogModelRouteProbe(string entrypoint, string sourcePath, string ext, byte[] modelBytes, ModelContainerKind container)
    {
        ViewerLog.Trace(
            $"[ModelRouting] probe build={_dbcBuild ?? "unknown"} entrypoint={entrypoint} file={sourcePath} ext={ext} magic={GetModelMagicLabel(modelBytes)} md20Version={GetMd20VersionLabel(modelBytes)} container={container}");
    }

    private void LoadModelFromBytesWithContainerProbe(byte[] modelBytes, string sourcePath, string dir, string entrypoint,
        IReadOnlyList<string>? explicitTextureVariations = null)
    {
        var container = DetectModelContainer(modelBytes);
        string ext = Path.GetExtension(sourcePath).ToLowerInvariant();
        LogModelRouteProbe(entrypoint, sourcePath, ext, modelBytes, container);

        switch (container)
        {
            case ModelContainerKind.Mdlx:
                if (ext != ".mdx")
                    ViewerLog.Important(ViewerLog.Category.Mdx,
                        $"[ModelRouting] Extension/container mismatch: '{ext}' with MDLX root. Routing as MDX: {Path.GetFileName(sourcePath)}");

                // Use the legacy MDX renderer for .mdx files. The chunked MDX-to-M2
                // runtime conversion path produces incorrect animation for converted
                // MDX data (M2 CPU skinning doesn't properly handle Alpha-era models).
                MdxRuntimeSharedInfo? sharedRuntimeInfo = TryReadSharedMdxRuntimeInfo(sourcePath, modelBytes);

                using (var ms = new MemoryStream(modelBytes))
                using (var br = new BinaryReader(ms))
                {
                    var mdx = MdxFile.Load(br);
                    LoadMdxModel(mdx, dir, sourcePath, sharedRuntimeInfo: sharedRuntimeInfo,
                        explicitTextureVariations: explicitTextureVariations);
                }
                return;

            case ModelContainerKind.Md20:
            case ModelContainerKind.Md21:
                if (ext == ".mdx" || ext == ".mdl")
                    ViewerLog.Important(ViewerLog.Category.Mdx,
                        $"[ModelRouting] Extension/container mismatch: '{ext}' with {GetModelMagicLabel(modelBytes)} root. Routing as M2-family: {Path.GetFileName(sourcePath)}");

                LoadM2FromBytes(modelBytes, sourcePath, dir);
                return;

            default:
                throw new InvalidDataException(
                    $"Unsupported model root magic ({GetModelMagicLabel(modelBytes)}) for '{Path.GetFileName(sourcePath)}'. Expected MDLX or MD20.");
        }
    }

    private void LoadChunkedMdxFromBytes(byte[] modelBytes, string sourcePath, string dir)
    {
        ArgumentNullException.ThrowIfNull(modelBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        string resolvedModelPath = ResolveStandaloneCanonicalModelPath(sourcePath);
        using MemoryStream stream = new(modelBytes, writable: false);
        M2ChunkedReadResult chunked = M2ChunkedModelReader.ReadDetailed(stream, resolvedModelPath, ReadStandaloneFileData);

        if (TryLoadStandaloneCameraPathM2(chunked.Conversion.ModelBytes, chunked.Conversion.ModelPath))
        {
            CaptureWorldReturnState();
            return;
        }

        M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(
            chunked.Conversion.ModelBytes,
            chunked.Conversion.SkinBytes,
            chunked.Conversion.ModelPath,
            chunked.Conversion.SkinPath);

        LoadM2RuntimeModel(runtimeModel, modelDir: dir, virtualPath: resolvedModelPath);
        ViewerLog.Info(ViewerLog.Category.Mdx,
            $"[ModelRouting] Loaded chunked MDX through M2 runtime: file={Path.GetFileName(sourcePath)} chunks={chunked.Chunks.Count} geosets={chunked.Geometry.GeosetCount} vertices={chunked.VertexCount} triangles={chunked.TriangleCount}");
    }

    /// <summary>
    /// Load a WMO from disk, auto-detecting v14 (Alpha) vs v17+ (standard) format.
    /// v17 files are converted to v14 in-memory before rendering.
    /// </summary>
    private void LoadWmoFromDisk(string filePath, string dir)
    {
        int version = DetectWmoVersion(filePath);
        ViewerLog.Trace($"[WMO] Detected version {version} for {Path.GetFileName(filePath)}");

        if (version >= 17)
        {
            // v17+: parse directly into WmoV14Data — no lossy binary roundtrip
            var v17RootBytes = File.ReadAllBytes(filePath);

            var groupBytesList = new List<byte[]>();
            string baseName = Path.GetFileNameWithoutExtension(filePath);
            for (int gi = 0; gi < 512; gi++)
            {
                string groupPath = Path.Combine(dir, $"{baseName}_{gi:D3}.wmo");
                if (!File.Exists(groupPath)) break;
                groupBytesList.Add(File.ReadAllBytes(groupPath));
                ViewerLog.Trace($"[WMO] Loaded group file: {Path.GetFileName(groupPath)}");
            }

            var v17Parser = new WmoV17ToV14Converter();
            var wmo = v17Parser.ParseV17ToModel(v17RootBytes, groupBytesList);
            ViewerLog.Trace($"[WMO] Parsed v{version} direct ({wmo.Groups.Count} groups)");
            LoadWmoModel(wmo, dir);
            _statusMessage = $"Loaded WMO v{version}: {Path.GetFileName(filePath)}";
        }
        else
        {
            // v14 (Alpha): use existing pipeline directly
            var converter = new WmoV14ToV17Converter();
            var wmo = converter.ParseWmoV14(filePath);
            LoadWmoModel(wmo, dir);
        }
    }

    /// <summary>
    /// Load a WMO from data source bytes, auto-detecting v14 vs v17+ format.
    /// </summary>
    private void LoadWmoFromDataSource(byte[] rootBytes, string virtualPath, string cachePath)
    {
        // Detect version from bytes
        int version;
        using (var ms = new MemoryStream(rootBytes))
        using (var br = new BinaryReader(ms))
            version = DetectWmoVersionFromBytes(br);

        ViewerLog.Trace($"[WMO] Detected version {version} for {Path.GetFileName(virtualPath)}");

        if (version >= 17)
        {
            // v17+: parse directly into WmoV14Data — no lossy binary roundtrip
            var wmoDir = Path.GetDirectoryName(virtualPath)?.Replace('/', '\\') ?? "";
            var wmoBase = Path.GetFileNameWithoutExtension(virtualPath);

            var groupBytesList = new List<byte[]>();
            for (int gi = 0; gi < 512; gi++)
            {
                var groupName = $"{wmoBase}_{gi:D3}.wmo";
                var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                var groupBytes = _dataSource?.ReadFile(groupPath);
                if (groupBytes == null || groupBytes.Length == 0) break;
                groupBytesList.Add(groupBytes);
                ViewerLog.Trace($"[WMO] Group {gi}: loaded {groupBytes.Length} bytes");
            }

            var v17Parser = new WmoV17ToV14Converter();
            var wmo = v17Parser.ParseV17ToModel(rootBytes, groupBytesList);
            ViewerLog.Trace($"[WMO] Parsed v{version} direct ({wmo.Groups.Count} groups)");
            LoadWmoModel(wmo, CacheDir);
            _statusMessage = $"Loaded WMO v{version}: {Path.GetFileName(virtualPath)}";
        }
        else
        {
            // v14 (Alpha): use existing pipeline
            var converter = new WmoV14ToV17Converter();
            var wmo = converter.ParseWmoV14(cachePath);

            // v16 split format: root has GroupCount but no embedded MOGP chunks
            if (wmo.Groups.Count == 0 && wmo.GroupCount > 0 && _dataSource != null)
            {
                var wmoDir = Path.GetDirectoryName(virtualPath)?.Replace('/', '\\') ?? "";
                var wmoBase = Path.GetFileNameWithoutExtension(virtualPath);
                ViewerLog.Trace($"[WMO] v14/v16 split: loading {wmo.GroupCount} group files from data source");

                for (int gi = 0; gi < wmo.GroupCount; gi++)
                {
                    var groupName = $"{wmoBase}_{gi:D3}.wmo";
                    var groupPath = string.IsNullOrEmpty(wmoDir) ? groupName : $"{wmoDir}\\{groupName}";
                    var groupBytes = _dataSource.ReadFile(groupPath);
                    if (groupBytes != null && groupBytes.Length > 0)
                    {
                        ViewerLog.Trace($"[WMO] Group {gi}: loaded {groupBytes.Length} bytes from '{groupPath}'");
                        converter.ParseGroupFile(groupBytes, wmo, gi);
                    }
                    else
                    {
                        ViewerLog.Trace($"[WMO] Group {gi}: NOT FOUND '{groupPath}'");
                    }
                }

                for (int gi = 0; gi < wmo.Groups.Count && gi < wmo.GroupInfos.Count; gi++)
                {
                    if (wmo.Groups[gi].Name == null)
                        wmo.Groups[gi].Name = $"group_{gi}";
                }

                var bMin = new Vector3(float.MaxValue);
                var bMax = new Vector3(float.MinValue);
                foreach (var g in wmo.Groups)
                {
                    foreach (var v in g.Vertices)
                    {
                        bMin = Vector3.Min(bMin, v);
                        bMax = Vector3.Max(bMax, v);
                    }
                }
                if (bMin.X < float.MaxValue)
                {
                    wmo.BoundsMin = bMin;
                    wmo.BoundsMax = bMax;
                    ViewerLog.Trace($"[WMO] Recalculated bounds: ({bMin.X:F1},{bMin.Y:F1},{bMin.Z:F1}) - ({bMax.X:F1},{bMax.Y:F1},{bMax.Z:F1})");
                }
            }

            LoadWmoModel(wmo, CacheDir);
        }
    }

    /// <summary>
    /// Detect Alpha WDT format by examining MPHD data.
    /// Alpha MPHD stores absolute file offsets to MDNM (byte 4) and MONM (byte 12).
    /// Standard MPHD stores flags at byte 0 and has no MDNM/MONM offsets.
    /// If MPHD byte 4 contains a large value (absolute offset to MDNM), it's Alpha.
    /// </summary>
    internal static bool DetectAlphaWdt(byte[] wdtBytes)
    {
        // Find MPHD chunk (reversed on disk: "DHPM")
        for (int i = 0; i + 8 <= wdtBytes.Length;)
        {
            string fcc = System.Text.Encoding.ASCII.GetString(wdtBytes, i, 4);
            int sz = BitConverter.ToInt32(wdtBytes, i + 4);
            if (sz < 0 || i + 8 + sz > wdtBytes.Length) break;

            string reversed = new string(fcc.Reverse().ToArray());
            if (fcc == "DHPM" || reversed == "DHPM") // MPHD
            {
                int dataStart = i + 8;
                if (sz >= 16)
                {
                    // Alpha MPHD: [0..3]=nTextures, [4..7]=MDNM abs offset, [8..11]=nMapObjNames, [12..15]=MONM abs offset
                    // Standard MPHD: [0..3]=flags (small: 0,1,4,8), rest is different
                    int mdnmOffset = BitConverter.ToInt32(wdtBytes, dataStart + 4);
                    // MDNM offset in Alpha is always after MVER+MPHD+MAIN, so > ~32KB
                    // Standard MPHD byte 4 is 0 or a small relative offset
                    if (mdnmOffset > 1000 && mdnmOffset < wdtBytes.Length)
                        return true;
                }
                break;
            }

            int next = i + 8 + sz;
            if (next <= i) break;
            i = next;
        }

        return false;
    }

    private string ResolveStandaloneCanonicalModelPath(string sourcePath)
    {
        string normalizedPath = sourcePath.Replace('/', '\\');
        if (_dataSource == null)
            return normalizedPath;

        if (_dataSource is not MpqDataSource mpqDataSource)
            return normalizedPath;

        foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
        {
            string? found = mpqDataSource.FindInFileSet(candidate);
            if (!string.IsNullOrWhiteSpace(found))
                return found.Replace('/', '\\');
        }

        string baseName = Path.GetFileNameWithoutExtension(normalizedPath);
        if (!string.IsNullOrWhiteSpace(baseName))
        {
            string? indexed = mpqDataSource.FindByBaseName(baseName, GetLikelyStandaloneModelExtensions(normalizedPath));
            if (!string.IsNullOrWhiteSpace(indexed))
                return indexed.Replace('/', '\\');
        }

        foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
        {
            if (_dataSource.FileExists(candidate))
                return candidate.Replace('/', '\\');
        }

        return normalizedPath;
    }

    private string? ResolveBestStandaloneSkinPath(string resolvedModelPath)
    {
        if (_dataSource == null)
            return null;

        if (_standaloneSkinPathCache.TryGetValue(resolvedModelPath, out string? cachedPath))
            return cachedPath;

        string? bestSkinPath = WarcraftNetM2Adapter.FindSkinInFileList(resolvedModelPath, _dataSource.GetFileList(".skin"));
        _standaloneSkinPathCache[resolvedModelPath] = bestSkinPath;
        return bestSkinPath;
    }

    internal byte[]? ReadStandaloneFileData(string path)
    {
        if (File.Exists(path))
            return File.ReadAllBytes(path);

        if (_dataSource == null)
            return null;

        byte[]? data = _dataSource.ReadFile(path);
        if (data != null && data.Length > 0)
            return data;

        string normalizedPath = path.Replace('/', '\\');
        if (!normalizedPath.Equals(path, StringComparison.OrdinalIgnoreCase))
        {
            data = _dataSource.ReadFile(normalizedPath);
            if (data != null && data.Length > 0)
                return data;
        }

        if (IsStandaloneModelPath(normalizedPath))
        {
            foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
            {
                if (candidate.Equals(normalizedPath, StringComparison.OrdinalIgnoreCase))
                    continue;

                data = _dataSource.ReadFile(candidate);
                if (data != null && data.Length > 0)
                    return data;
            }
        }

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            foreach (string candidate in BuildStandaloneFileSetCandidates(normalizedPath))
            {
                string? found = mpqDataSource.FindInFileSet(candidate);
                if (string.IsNullOrWhiteSpace(found))
                    continue;

                data = _dataSource.ReadFile(found);
                if (data != null && data.Length > 0)
                    return data;
            }

            string baseName = Path.GetFileNameWithoutExtension(normalizedPath);
            if (!string.IsNullOrWhiteSpace(baseName))
            {
                string? indexed = mpqDataSource.FindByBaseName(baseName, GetLikelyStandaloneModelExtensions(normalizedPath));
                if (!string.IsNullOrWhiteSpace(indexed))
                {
                    data = _dataSource.ReadFile(indexed);
                    if (data != null && data.Length > 0)
                        return data;
                }
            }
        }

        return null;
    }

    private static bool IsStandaloneModelPath(string path)
    {
        string ext = Path.GetExtension(path);
        return ext.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".mdl", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".m2", StringComparison.OrdinalIgnoreCase);
    }

    private static IEnumerable<string> BuildStandaloneFileSetCandidates(string path)
    {
        yield return path;

        foreach (string alternatePath in EnumerateStandaloneAlternateModelPaths(path))
            yield return alternatePath;

        string fileName = Path.GetFileName(path);
        if (!string.IsNullOrWhiteSpace(fileName) && !fileName.Equals(path, StringComparison.OrdinalIgnoreCase))
        {
            yield return fileName;

            foreach (string alternatePath in EnumerateStandaloneAlternateModelPaths(fileName))
                yield return alternatePath;
        }

        string baseName = Path.GetFileNameWithoutExtension(path);
        if (!string.IsNullOrWhiteSpace(baseName))
        {
            yield return $"Creature\\{baseName}\\{baseName}.mdx";
            yield return $"Creature\\{baseName}\\{baseName}.m2";
            yield return $"Creature\\{baseName}\\{baseName}.mdl";
        }
    }

    private byte[]? ConvertStandaloneM2ToMdx(byte[] m2Bytes, string resolvedModelPath)
    {
        try
        {
            byte[]? skinBytes = null;
            foreach (string skinPath in WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath).Distinct(StringComparer.OrdinalIgnoreCase))
            {
                skinBytes = ReadStandaloneFileData(skinPath);
                if (skinBytes != null && skinBytes.Length > 0)
                    break;
            }

            if ((skinBytes == null || skinBytes.Length == 0) && _dataSource != null)
            {
                string? bestSkinPath = ResolveBestStandaloneSkinPath(resolvedModelPath);
                if (!string.IsNullOrWhiteSpace(bestSkinPath))
                    skinBytes = ReadStandaloneFileData(bestSkinPath);
            }

            throw new NotSupportedException("M2 to MDX conversion is not supported in the standalone viewer."); return new byte[0];
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] Standalone M2->MDX converter fallback failed for {Path.GetFileName(resolvedModelPath)}: {ex.Message}");
            return null;
        }
    }

    private static IEnumerable<string> EnumerateStandaloneAlternateModelPaths(string path)
    {
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".m2";
            yield return path[..^4] + ".mdl";
            yield break;
        }

        if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".mdx";
            yield return path[..^4] + ".m2";
            yield break;
        }

        if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^3] + ".mdx";
            yield return path[..^3] + ".mdl";
        }
    }

    private static IEnumerable<string> GetLikelyStandaloneModelExtensions(string path)
    {
        string ext = Path.GetExtension(path);
        if (ext.Equals(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".m2";
            yield return ".mdx";
            yield return ".mdl";
            yield break;
        }

        if (ext.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return ".mdl";
            yield return ".mdx";
            yield return ".m2";
            yield break;
        }

        yield return ".mdx";
        yield return ".m2";
        yield return ".mdl";
    }

    /// <summary>
    /// Detect WMO version by reading the MVER chunk from the file.
    /// Returns 14 for Alpha, 17 for standard WotLK+, or 0 if detection fails.
    /// </summary>
    private static int DetectWmoVersion(string filePath)
    {
        try
        {
            using var fs = File.OpenRead(filePath);
            using var br = new BinaryReader(fs);
            return DetectWmoVersionFromBytes(br);
        }
        catch { return 0; }
    }

    /// <summary>
    /// Detect WMO version from a BinaryReader by scanning for MVER chunk.
    /// Handles both forward and reversed FourCC ordering.
    /// </summary>
    private static int DetectWmoVersionFromBytes(BinaryReader br)
    {
        long startPos = br.BaseStream.Position;
        try
        {
            // Read first 8 bytes to check for MOMO container (v14) or MVER (v17)
            if (br.BaseStream.Length < 12) return 0;

            var magic = System.Text.Encoding.ASCII.GetString(br.ReadBytes(4));
            var reversed = new string(magic.Reverse().ToArray());

            // v14 Alpha: starts with MOMO container
            if (magic == "MOMO" || reversed == "MOMO")
                return 14;

            // v17+: starts with MVER chunk directly
            if (magic == "MVER" || reversed == "MVER")
            {
                uint size = br.ReadUInt32();
                if (size >= 4)
                {
                    uint version = br.ReadUInt32();
                    return (int)version;
                }
            }

            // Fallback: scan first 64 bytes for MVER
            br.BaseStream.Position = startPos;
            byte[] header = br.ReadBytes((int)Math.Min(64, br.BaseStream.Length));
            string headerStr = System.Text.Encoding.ASCII.GetString(header);
            int mverIdx = headerStr.IndexOf("MVER");
            if (mverIdx < 0) mverIdx = headerStr.IndexOf("REVM"); // reversed
            if (mverIdx >= 0 && mverIdx + 12 <= header.Length)
            {
                uint ver = BitConverter.ToUInt32(header, mverIdx + 8);
                return (int)ver;
            }

            return 0;
        }
        finally
        {
            br.BaseStream.Position = startPos;
        }
    }

    /// <summary>
    /// Called when the user double-clicks an entry in the Asset Catalog.
    /// Loads the model into the viewer using the same pipeline as the file browser.
    /// </summary>
    internal void OnCatalogLoadModel(string modelPath, bool isWmo, AssetCatalogEntry entry)
    {
        if (_dataSource == null)
        {
            _statusMessage = "No data source loaded";
            return;
        }

        // Try exact path first, then fuzzy resolve via the data source file list
        byte[]? data = _dataSource.ReadFile(modelPath);
        string resolvedPath = modelPath;

        if (data == null)
        {
            // Fuzzy: try Creature\Name\Name.mdx pattern and case variations
            string baseName = Path.GetFileNameWithoutExtension(modelPath);
            string[] candidates = {
                modelPath,
                $"Creature\\{baseName}\\{baseName}.mdx",
                modelPath.Replace('/', '\\'),
                modelPath.Replace('\\', '/'),
            };
            foreach (var c in candidates)
            {
                data = _dataSource.ReadFile(c);
                if (data != null) { resolvedPath = c; break; }
            }

            // Last resort: search file list
            if (data == null)
            {
                string ext = isWmo ? ".wmo" : ".mdx";
                var files = _dataSource.GetFileList(ext);
                string target = baseName.ToLowerInvariant();
                var match = files.FirstOrDefault(f =>
                    Path.GetFileNameWithoutExtension(f).Equals(target, StringComparison.OrdinalIgnoreCase));
                if (match != null)
                {
                    data = _dataSource.ReadFile(match);
                    if (data != null) resolvedPath = match;
                }
            }
        }

        if (data == null || data.Length == 0)
        {
            _statusMessage = $"Model not found: {modelPath}";
            return;
        }

        try
        {
            _renderer?.Dispose();
            _renderer = null;
            _loadedFileName = Path.GetFileName(resolvedPath);
            _lastVirtualPath = resolvedPath;

            string dir = Path.GetDirectoryName(resolvedPath)?.Replace('/', '\\') ?? "";

            if (isWmo)
            {
                // WMO: write to temp, parse, load
                string tempFile = Path.Combine(Path.GetTempPath(), $"catalog_wmo_{entry.EntryId}.wmo");
                File.WriteAllBytes(tempFile, data);
                var converter = new WmoV14ToV17Converter();
                var wmo = converter.ParseWmoV14(tempFile);

                // Handle split WMO groups
                if (wmo.Groups.Count == 0 && wmo.GroupCount > 0)
                {
                    string wmoBase = Path.GetFileNameWithoutExtension(resolvedPath);
                    for (int gi = 0; gi < wmo.GroupCount; gi++)
                    {
                        var groupName = $"{wmoBase}_{gi:D3}.wmo";
                        var groupPath = string.IsNullOrEmpty(dir) ? groupName : $"{dir}\\{groupName}";
                        var groupBytes = _dataSource.ReadFile(groupPath);
                        if (groupBytes != null)
                            converter.ParseGroupFile(groupBytes, wmo, gi);
                    }
                }

                try { File.Delete(tempFile); } catch { }
                LoadWmoModel(wmo, dir);
            }
            else
            {
                LoadModelFromBytesWithContainerProbe(data, resolvedPath, dir, "Catalog", entry.TextureVariations);
            }

            _window.Title = $"{ViewerProductName} - {entry.Name} ({_loadedFileName})";
            _statusMessage = $"Loaded from catalog: {entry.Name} [{entry.EntryId}]";
        }
        catch (Exception ex)
        {
            LogLoadFailure("CatalogLoad", resolvedPath, ex, isWmo ? null : data);
            _statusMessage = $"Failed to load {entry.Name}: {BuildStatusExceptionSummary(ex)}";
            _modelInfo = "";
        }
    }

    internal void LoadFileFromDataSource(string virtualPath)
    {
        if (_dataSource == null) return;

        _statusMessage = $"Loading {Path.GetFileName(virtualPath)}...";
        _loadedFileName = Path.GetFileName(virtualPath);
        _lastVirtualPath = virtualPath;

        string resolvedVirtualPath = virtualPath;
        string ext = Path.GetExtension(virtualPath).ToLowerInvariant();
        byte[]? data = null;

        if (ext != ".wdt")
            CaptureWorldReturnState();

        try
        {
            if (ext is ".mdx" or ".mdl" or ".m2")
            {
                resolvedVirtualPath = ResolveStandaloneCanonicalModelPath(virtualPath);
                data = ReadStandaloneFileData(resolvedVirtualPath);
                if ((data == null || data.Length == 0) && !resolvedVirtualPath.Equals(virtualPath, StringComparison.OrdinalIgnoreCase))
                    data = ReadStandaloneFileData(virtualPath);
            }
            else
            {
                data = _dataSource.ReadFile(virtualPath);
            }

            if (data == null || data.Length == 0)
            {
                LogDataSourceReadFailure(virtualPath, resolvedVirtualPath, ext);
                _statusMessage = resolvedVirtualPath.Equals(virtualPath, StringComparison.OrdinalIgnoreCase)
                    ? $"Failed to read: {virtualPath}"
                    : $"Failed to read: {virtualPath} (resolved: {resolvedVirtualPath})";
                return;
            }

            _renderer?.Dispose();
            _renderer = null;

            _lastVirtualPath = resolvedVirtualPath;
            _loadedFileName = Path.GetFileName(resolvedVirtualPath);

            // Write to cache folder for parsers that expect file paths. The cache is VERSIONED by
            // client root: a flat cache let a stale Shadowfang.wdt extracted from one client
            // version shadow the 0.5.3 alphaWDT from another, silently feeding the terrain
            // pipeline a WDT that was never from the active client (Spec 222, 2026-09-04).
            Directory.CreateDirectory(CacheDir);
            string clientCacheSegment = WdlPreviewService.BuildCacheSegment(_wdlPreview.BuildWdlPreviewCacheIdentity());
            var cachePath = Path.Combine(CacheDir, clientCacheSegment, _loadedFileName!);
            Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
            File.WriteAllBytes(cachePath, data);
            _loadedFilePath = cachePath;

            switch (ext)
            {
                case ".mdx":
                case ".m2":
                case ".mdl":
                    LoadModelFromBytesWithContainerProbe(data, resolvedVirtualPath, CacheDir, "DataSource");
                    break;

                case ".wmo":
                    LoadWmoFromDataSource(data, virtualPath, cachePath);
                    break;

                case ".wdt":
                    LoadWdtTerrain(cachePath);
                    break;

                default:
                    _statusMessage = $"Viewing {ext} not yet supported.";
                    break;
            }

            _window.Title = $"{ViewerProductName} - {_loadedFileName}";
        }
        catch (Exception ex)
        {
            LogLoadFailure("DataSourceLoad", resolvedVirtualPath, ex,
                ext is ".mdx" or ".mdl" or ".m2" ? data : null);
            _statusMessage = $"Load failed: {BuildStatusExceptionSummary(ex)}";
            _modelInfo = "";
        }
    }

    private void LoadMdxModel(MdxFile mdx, string dir, string? virtualPath = null, bool isM2AdapterModel = false,
        MdxRuntimeSharedInfo? sharedRuntimeInfo = null, IReadOnlyList<string>? explicitTextureVariations = null)
    {
        CaptureWorldReturnState();
        ExitToStandaloneView();

        _loadedWmo = null;
        _loadedMdx = mdx;
        _loadedM2Runtime = null;

        CoreMdxSummary? sharedSummary = sharedRuntimeInfo?.Summary;
        CoreMdxGeometryFile? sharedGeometry = sharedRuntimeInfo?.Geometry;

        int geosetCount = sharedGeometry?.GeosetCount ?? mdx.Geosets.Count;
        int validGeosets = sharedGeometry != null
            ? sharedGeometry.Geosets.Count(g => g.VertexCount > 0 && g.IndexCount > 0)
            : mdx.Geosets.Count(g => g.Vertices.Count > 0 && g.Indices.Count > 0);
        int totalVerts = sharedGeometry != null
            ? sharedGeometry.Geosets.Sum(g => g.VertexCount)
            : mdx.Geosets.Sum(g => g.Vertices.Count);
        int totalTris = sharedGeometry != null
            ? sharedGeometry.Geosets.Sum(g => g.TriangleCount)
            : mdx.Geosets.Sum(g => g.Indices.Count / 3);
        string versionLabel = sharedSummary?.Version?.ToString()
            ?? sharedGeometry?.Version?.ToString()
            ?? mdx.Version.ToString();
        string modelName = sharedSummary?.ModelName
            ?? sharedGeometry?.ModelName
            ?? mdx.Model.Name;
        int textureCount = sharedSummary?.TextureCount ?? mdx.Textures.Count;
        int materialCount = sharedSummary?.MaterialCount ?? mdx.Materials.Count;
        int boneCount = sharedSummary?.BoneCount ?? mdx.Bones.Count;
        int sequenceCount = sharedSummary?.SequenceCount ?? mdx.Sequences.Count;
        int pivotPointCount = sharedSummary?.PivotPointCount ?? mdx.PivotPoints.Count;
        CoreMdxCollisionSummary? collision = sharedSummary?.Collision;

        _renderer = new MdxRenderer(_gl, mdx, dir, _dataSource, _texResolver, virtualPath, isM2AdapterModel, _dbcBuild,
            explicitTextureVariations: explicitTextureVariations);
        RefreshStandaloneCharacterCustomizationState(virtualPath, isM2AdapterModel);

        if (sharedRuntimeInfo != null)
        {
            ViewerLog.Trace(
                $"[SharedMDX] Runtime metadata consumer: summary={(sharedSummary != null ? "yes" : "no")} geometry={(sharedGeometry != null ? "yes" : "no")} file={Path.GetFileName(virtualPath ?? _loadedFileName ?? "<memory>")}");
        }

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        string typeLabel = isM2AdapterModel
            ? "M2 (compatibility runtime via MDX renderer)"
            : "MDX (Alpha 0.5.3)";
        string statusTypeLabel = isM2AdapterModel ? "M2" : "MDX";

        _modelInfo = $"Path: {virtualPath ?? _loadedFileName ?? "<unknown>"}\n" +
                     $"Type: {typeLabel}\n" +
                     $"Version: {versionLabel}\n" +
                     $"Name: {modelName}\n\n" +
                     $"Geosets: {geosetCount} ({validGeosets} valid)\n" +
                     $"Vertices: {totalVerts:N0}\n" +
                     $"Triangles: {totalTris:N0}\n" +
                     $"Pivot Points: {pivotPointCount}\n" +
                     (collision != null
                        ? $"Collision: {collision.VertexCount} verts, {collision.TriangleCount} tris\n"
                        : string.Empty) +
                     "\n" +
                     $"Materials: {materialCount}\n" +
                     $"Textures: {textureCount}\n" +
                     $"Bones: {boneCount}\n" +
                     $"Sequences: {sequenceCount}\n";

        if (mdx.Sequences.Count > 0)
        {
            _modelInfo += "\nAnimations:\n";
            foreach (var seq in mdx.Sequences)
                _modelInfo += $"  {seq.Name} ({seq.Time.Start}-{seq.Time.End})\n";
        }

        if (mdx.Textures.Count > 0)
        {
            _modelInfo += "\nTextures:\n";
            foreach (var tex in mdx.Textures)
            {
                string name = string.IsNullOrEmpty(tex.Path) ? $"Replaceable #{tex.ReplaceableId}" : tex.Path;
                _modelInfo += $"  {name}\n";
            }
        }

        if (isM2AdapterModel)
        {
            _modelInfo += "\nCompatibility Notes:\n" +
                          "  Source asset is M2, but the current viewer path still adapts it into MdxFile/MdxRenderer state.\n" +
                          "  Animated M2 compatibility is currently disabled by default because that path is not reliable.\n";
        }

        _statusMessage = $"Loaded {statusTypeLabel}: {_loadedFileName} ({validGeosets} geosets, {totalVerts:N0} verts)";
    }

    private void LoadM2RuntimeModel(M2StaticRenderModel runtimeModel, MdxFile? adaptedMdx = null, string? modelDir = null, string? virtualPath = null)
    {
        ArgumentNullException.ThrowIfNull(runtimeModel);

        CaptureWorldReturnState();
        ExitToStandaloneView();

        _loadedWmo = null;
        _loadedMdx = null;
        _loadedM2Runtime = runtimeModel;
        string sourceModelPath = virtualPath ?? runtimeModel.Model.Identity.CanonicalModelPath;
        _renderer = WowViewerM2RuntimeBridge.CreateRenderer(
            _gl,
            runtimeModel,
            adaptedMdx,
            modelDir,
            _dataSource,
            _texResolver,
            _dbcBuild,
            sourceModelPath);
        RefreshStandaloneCharacterCustomizationState(sourceModelPath, isM2AdapterModel: adaptedMdx != null);
        ApplyStandaloneCharacterCustomizationOverrides();

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        int sectionCount = runtimeModel.Sections.Count;
        int vertexCount = runtimeModel.Sections.Sum(static section => section.Vertices.Count);
        int triangleCount = runtimeModel.Sections.Sum(static section => section.Indices.Count / 3);
        int transparentSectionCount = runtimeModel.Sections.Count(static section => section.Material.IsTransparent);
        List<string> textureNames = runtimeModel.Sections
            .Select(static section => section.Material.TexturePath)
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList()!;

        bool usesNativeStaticRenderer = WowViewerM2RuntimeBridge.ShouldUseNativeStaticRenderer(adaptedMdx);
        string runtimeTypeLabel = usesNativeStaticRenderer
            ? "M2 (wow-viewer static renderer in WoWViewer)"
            : "M2 (wow-viewer runtime + legacy draw backend)";

        _modelInfo = $"Path: {virtualPath ?? runtimeModel.Model.Identity.CanonicalModelPath}\n" +
                     $"Type: {runtimeTypeLabel}\n" +
                     $"Version: {runtimeModel.Model.Version}\n" +
                     $"Name: {runtimeModel.Model.ModelName ?? Path.GetFileNameWithoutExtension(runtimeModel.Model.Identity.CanonicalModelPath)}\n\n" +
                     $"Sections: {sectionCount}\n" +
                     $"Transparent Sections: {transparentSectionCount}\n" +
                     $"Vertices: {vertexCount:N0}\n" +
                     $"Triangles: {triangleCount:N0}\n" +
                     $"Bounds Radius: {runtimeModel.Model.BoundsRadius:F3}\n";

        if (textureNames.Count > 0)
        {
            _modelInfo += "\nTextures:\n";
            foreach (string textureName in textureNames)
                _modelInfo += $"  {textureName}\n";
        }

        _modelInfo += "\nRuntime Notes:\n" +
                      "  Geometry is submitted from wow-viewer active skin sections.\n" +
                      (usesNativeStaticRenderer
                          ? "  Draw path: Native wow-viewer runtime renderer in WoWViewer.\n  Skeletal sequence playback advances through wow-viewer pose evaluation.\n  Shading: Native runtime material pipeline (textured diffuse, directional + ambient lighting, alpha cutout & blending).\n"
                          : "  Draw path: Legacy MDX backend compatibility pipeline.\n");

        _statusMessage = $"Loaded M2: {_loadedFileName} ({sectionCount} sections, {vertexCount:N0} verts, {triangleCount:N0} tris)";
    }

    private void LoadStandaloneCameraPathModel(M2ModelDocument cameraModel, M2CameraPathVisualization visualization, string virtualPath)
    {
        ArgumentNullException.ThrowIfNull(cameraModel);
        ArgumentNullException.ThrowIfNull(visualization);

        CaptureWorldReturnState();
        ExitToStandaloneView();

        _loadedWmo = null;
        _loadedMdx = null;
        _loadedM2Runtime = null;
        _renderer = new M2CameraPathRenderer(_gl, visualization, virtualPath);
        ClearStandaloneCharacterCustomizationState(resetOverrides: true);

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        var info = new StringBuilder();
        info.AppendLine($"Path: {virtualPath}");
        info.AppendLine("Type: M2 camera path");
        info.AppendLine($"Version: {cameraModel.Version}");
        info.AppendLine($"Name: {cameraModel.ModelName ?? Path.GetFileNameWithoutExtension(cameraModel.Identity.CanonicalModelPath)}");
        info.AppendLine();
        info.AppendLine($"Cameras: {cameraModel.CameraCount}");
        info.AppendLine($"Sequences: {cameraModel.SequenceCount}");
        info.AppendLine($"Bounds Radius: {cameraModel.BoundsRadius:F3}");
        info.AppendLine();
        info.AppendLine("Camera Definitions:");

        foreach (M2CameraDefinition camera in cameraModel.Cameras)
        {
            string typeLabel = DescribeStandaloneCameraType(camera.Type);
            string fovLabel = camera.HasAnimatedFieldOfView
                ? "animated FoV"
                : $"FoV {camera.StaticFieldOfView.GetValueOrDefault():F3} rad";
            info.AppendLine($"  [{camera.Index}] {typeLabel}: near {camera.NearClip:F2}, far {camera.FarClip:F2}, {fovLabel}");
        }

        info.AppendLine();
        info.AppendLine("Runtime Notes:");
        info.AppendLine("  Geometry-less camera-only M2 assets are visualized as sampled camera and target paths.");
        info.AppendLine("  This path intentionally bypasses .skin resolution because flyby cameras can be valid MD20 assets without mesh data.");

        _modelInfo = info.ToString();
        _statusMessage = $"Loaded M2 camera path: {_loadedFileName} ({cameraModel.CameraCount} cameras)";
    }

    private static string DescribeStandaloneCameraType(int cameraType)
    {
        return cameraType switch
        {
            0 => "portrait",
            1 => "character info",
            -1 => "flyby",
            _ => $"type {cameraType}",
        };
    }

    private MdxRuntimeSharedInfo? TryReadSharedMdxRuntimeInfo(string sourcePath, byte[] modelBytes)
    {
        CoreMdxSummary? summary = null;
        CoreMdxGeometryFile? geometry = null;

        try
        {
            using var summaryStream = new MemoryStream(modelBytes, writable: false);
            summary = MdxSummaryReader.Read(summaryStream, sourcePath);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[SharedMDX] Summary metadata unavailable for runtime consumer {Path.GetFileName(sourcePath)}: {ex.Message}");
        }

        try
        {
            using var geometryStream = new MemoryStream(modelBytes, writable: false);
            geometry = MdxGeometryReader.Read(geometryStream, sourcePath);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[SharedMDX] GEOS metadata unavailable for runtime consumer {Path.GetFileName(sourcePath)}: {ex.Message}");
        }

        if (summary == null && geometry == null)
            return null;

        return new MdxRuntimeSharedInfo(summary, geometry);
    }

    private readonly record struct MdxRuntimeSharedInfo(
        CoreMdxSummary? Summary,
        CoreMdxGeometryFile? Geometry);

    /// <summary>
    /// Tears down the world/terrain scene so the viewer switches to standalone
    /// object-view mode (WMO or M2 model rendering without the world scene).
    /// </summary>
    private void ExitToStandaloneView()
    {
        _loadingScreen?.Disable();
        _worldScene?.Dispose();
        _worldScene = null;
        _terrainManager?.Dispose();
        _terrainManager = null;
        _vlmTerrainManager?.Dispose();
        _vlmTerrainManager = null;
        _sqlSpawnStreaming.ResetSqlSpawnStreamingState(clearSceneSpawns: false);
    }

    private void LoadWmoModel(WmoV14ToV17Converter.WmoV14Data wmo, string dir)
    {
        // Loading a standalone WMO fully switches the viewer to object-view mode — the render
        // path draws this WMO and no longer draws the world scene. Tear down any lingering
        // world/terrain scene so the object-view UI drives THIS renderer. Otherwise the sidebar
        // keys off the still-alive _worldScene/_terrainManager and shows world-scene controls
        // (e.g. the "M2/WMO WF" wireframe checkbox drives the dormant world scene, so toggling it
        // has no visible effect on the loaded WMO — the object-view wireframe checkbox is skipped
        // because a stale terrain renderer is still present).
        ExitToStandaloneView();

        _loadedMdx = null;
        _loadedM2Runtime = null;
        _loadedWmo = wmo;
        
        int totalVerts = wmo.Groups.Sum(g => g.Vertices.Count);
        int totalTris = wmo.Groups.Sum(g => g.Indices.Count / 3);

        _renderer = new WmoRenderer(_gl, wmo, dir, _dataSource, _texResolver, _dbcBuild,
            enableRuntimeGroupVisibility: false);

        if (_autoFrameModelOnLoad)
            FrameCurrentModel();

        var wmoCenter = (wmo.BoundsMin + wmo.BoundsMax) * 0.5f;
        var wmoExtent = wmo.BoundsMax - wmo.BoundsMin;

        // Position camera offset from WMO center
        float dist = Math.Max(wmoExtent.Length() * 1.5f, 100f);
        _camera.Position = wmoCenter + new System.Numerics.Vector3(dist, 0, wmoExtent.Z * 0.3f);
        _camera.Yaw = 180f;
        _camera.Pitch = -10f;

        _modelInfo = $"Path: {_loadedFileName ?? "<unknown>"}\n" +
                     $"Type: WMO v{wmo.Version}\n\n" +
                     $"Groups: {wmo.Groups.Count}\n" +
                     $"Vertices: {totalVerts:N0}\n" +
                     $"Triangles: {totalTris:N0}\n\n" +
                     $"Materials: {wmo.Materials.Count}\n" +
                     $"Textures: {wmo.Textures.Count}\n" +
                     $"Doodad Sets: {wmo.DoodadSets.Count}\n" +
                     $"Doodad Defs: {wmo.DoodadDefs.Count}\n" +
                     $"Portals: {wmo.Portals.Count}\n" +
                     $"Lights: {wmo.Lights.Count}\n";

        if (wmo.DoodadSets.Count > 0)
        {
            _modelInfo += "\nDoodad Sets:\n";
            for (int i = 0; i < wmo.DoodadSets.Count; i++)
            {
                var ds = wmo.DoodadSets[i];
                _modelInfo += $"  [{i}] {ds.Name ?? "unnamed"} ({ds.Count} doodads)\n";
            }
        }

        if (wmo.Textures.Count > 0)
        {
            _modelInfo += "\nTextures:\n";
            foreach (var tex in wmo.Textures)
                _modelInfo += $"  {tex}\n";
        }

        if (wmo.Groups.Count > 0)
        {
            _modelInfo += "\nGroups:\n";
            for (int i = 0; i < wmo.Groups.Count; i++)
            {
                var g = wmo.Groups[i];
                string name = g.Name ?? $"group_{i}";
                _modelInfo += $"  [{i}] {name} ({g.Vertices.Count}v, {g.Indices.Count / 3}t)\n";
            }
        }

        _statusMessage = $"Loaded WMO: {_loadedFileName} ({wmo.Groups.Count} groups, {totalVerts:N0} verts, {wmo.DoodadDefs.Count} doodads)";
    }

    private void RefreshStandaloneCharacterCustomizationState(string? modelPath, bool isM2AdapterModel)
    {
        if (_texResolver == null || string.IsNullOrWhiteSpace(modelPath))
        {
            ClearStandaloneCharacterCustomizationState(resetOverrides: true);
            return;
        }

        string normalizedPath = modelPath.Replace('/', '\\');
        if (_texResolver.GetDefaultCharacterSelectionGroups(normalizedPath) == null)
        {
            ClearStandaloneCharacterCustomizationState(resetOverrides: true);
            return;
        }

        bool preserveExistingSelection = _preserveStandaloneCharacterCustomizationOnNextLoad
            || string.Equals(_standaloneCharacterCustomizationModelPath, normalizedPath, StringComparison.OrdinalIgnoreCase);

        _standaloneCharacterCustomizationModelPath = normalizedPath;
        _standaloneCharacterHairVariationIds.Clear();
        _standaloneCharacterHairVariationIds.AddRange(_texResolver.GetCharacterHairVariationIds(normalizedPath));
        _standaloneCharacterFacialHairVariationIds.Clear();
        _standaloneCharacterFacialHairVariationIds.AddRange(_texResolver.GetCharacterFacialHairVariationIds(normalizedPath));

        if (!preserveExistingSelection)
        {
            _standaloneCharacterHairVariationOverride = -1;
            _standaloneCharacterFacialHairVariationOverride = -1;
        }

        NormalizeStandaloneCharacterCustomizationSelection();
        _preserveStandaloneCharacterCustomizationOnNextLoad = false;

        ApplyStandaloneCharacterCustomizationOverrides();
    }

    private void ClearStandaloneCharacterCustomizationState(bool resetOverrides)
    {
        _standaloneCharacterCustomizationModelPath = null;
        _standaloneCharacterHairVariationIds.Clear();
        _standaloneCharacterFacialHairVariationIds.Clear();
        _preserveStandaloneCharacterCustomizationOnNextLoad = false;

        if (!resetOverrides)
            return;

        _standaloneCharacterHairVariationOverride = -1;
        _standaloneCharacterFacialHairVariationOverride = -1;
    }

    internal void PrepareStandaloneCharacterCustomizationForNextLoad(int? hairVariationId, int? facialHairVariationId)
    {
        _standaloneCharacterHairVariationOverride = hairVariationId is >= 0 ? hairVariationId.Value : -1;
        _standaloneCharacterFacialHairVariationOverride = facialHairVariationId is >= 0 ? facialHairVariationId.Value : -1;
        _preserveStandaloneCharacterCustomizationOnNextLoad = hairVariationId.HasValue || facialHairVariationId.HasValue;
    }

    private void NormalizeStandaloneCharacterCustomizationSelection()
    {
        if (_standaloneCharacterHairVariationOverride >= 0
            && !_standaloneCharacterHairVariationIds.Contains(_standaloneCharacterHairVariationOverride))
        {
            _standaloneCharacterHairVariationOverride = -1;
        }

        if (_standaloneCharacterFacialHairVariationOverride >= 0
            && !_standaloneCharacterFacialHairVariationIds.Contains(_standaloneCharacterFacialHairVariationOverride))
        {
            _standaloneCharacterFacialHairVariationOverride = -1;
        }
    }

    private void ApplyStandaloneCharacterCustomizationOverrides()
    {
        if (_texResolver == null || string.IsNullOrWhiteSpace(_standaloneCharacterCustomizationModelPath))
            return;

        IReadOnlyCollection<uint>? selectedGroups = _texResolver.GetCharacterSelectionGroups(
            _standaloneCharacterCustomizationModelPath,
            _standaloneCharacterHairVariationOverride >= 0 ? _standaloneCharacterHairVariationOverride : null,
            _standaloneCharacterFacialHairVariationOverride >= 0 ? _standaloneCharacterFacialHairVariationOverride : null);
        if (selectedGroups == null)
            return;

        string reasonLabel = _standaloneCharacterHairVariationOverride >= 0 || _standaloneCharacterFacialHairVariationOverride >= 0
            ? $"character geosets (hair={FormatStandaloneCharacterVariationLabel(_standaloneCharacterHairVariationOverride)}, facial={FormatStandaloneCharacterVariationLabel(_standaloneCharacterFacialHairVariationOverride)})"
            : "default character geosets";

        int? hairVariationId = _standaloneCharacterHairVariationOverride >= 0 ? _standaloneCharacterHairVariationOverride : null;
        int? facialHairVariationId = _standaloneCharacterFacialHairVariationOverride >= 0 ? _standaloneCharacterFacialHairVariationOverride : null;

        switch (_renderer)
        {
            case MdxRenderer mdxRenderer:
                mdxRenderer.TryApplyCharacterCustomization(selectedGroups, hairVariationId, facialHairVariationId, reasonLabel);
                break;

            case M2Renderer m2Renderer:
                m2Renderer.TryApplyCharacterCustomization(selectedGroups, hairVariationId, facialHairVariationId, reasonLabel);
                break;
        }
    }

    private static string FormatStandaloneCharacterVariationLabel(int variationId)
        => variationId >= 0 ? variationId.ToString() : "default";
}
