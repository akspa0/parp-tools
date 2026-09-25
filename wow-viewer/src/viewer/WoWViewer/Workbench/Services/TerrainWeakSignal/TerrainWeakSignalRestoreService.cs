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
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Terrain weak-signal restore: rebuilds flattened/low-signal terrain tiles from WDL, neighbours, shadow and texture evidence.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class TerrainWeakSignalRestoreService
{
    private readonly IViewerAppHost _host;

    internal TerrainWeakSignalRestoreService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private HashSet<(int tileX, int tileY, int chunkX, int chunkY)> _selectedChunks => _host.SelectedChunks;
    private ref WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode _stratigraphyAnchorMode => ref _host.StratigraphyAnchorMode;
    private ref bool _stratigraphyPolarityInverted => ref _host.StratigraphyPolarityInverted;
    private ref bool _stratigraphyPreserveNegativeFloor => ref _host.StratigraphyPreserveNegativeFloor;
    private ref bool _stratigraphyUnhideDevMeshes => ref _host.StratigraphyUnhideDevMeshes;
    private ref bool _stratigraphyUseNeighborAutoFit => ref _host.StratigraphyUseNeighborAutoFit;
    private ref bool _stratigraphyUseWdlMagnetization => ref _host.StratigraphyUseWdlMagnetization;
    private ref float _stratigraphyWdlMagnetizationStrength => ref _host.StratigraphyWdlMagnetizationStrength;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref TerrainTileScope _terrainTileScope => ref _host.TerrainTileScope;
    private ref bool _terrainWeakSignalRestoreAllLoadedTiles => ref _host.TerrainWeakSignalRestoreAllLoadedTiles;
    private ref float _terrainWeakSignalRestoreCandidateMaxHeight => ref _host.TerrainWeakSignalRestoreCandidateMaxHeight;
    private ref float _terrainWeakSignalRestoreCandidateMinHeight => ref _host.TerrainWeakSignalRestoreCandidateMinHeight;
    private ref bool _terrainWeakSignalRestoreEnabled => ref _host.TerrainWeakSignalRestoreEnabled;
    private ref float _terrainWeakSignalRestoreManualFactor => ref _host.TerrainWeakSignalRestoreManualFactor;
    private ref string _terrainWeakSignalRestoreStatus => ref _host.TerrainWeakSignalRestoreStatus;
    private ref bool _terrainWeakSignalRestoreUseAutoFactor => ref _host.TerrainWeakSignalRestoreUseAutoFactor;
    private ref bool _terrainWeakSignalRestoreUseTextureSubdivisions => ref _host.TerrainWeakSignalRestoreUseTextureSubdivisions;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WdlPreviewCacheService? _wdlPreviewCacheService => ref _host.WdlPreviewCacheService;
    private (int tileX, int tileY) GetCameraTile() => _host.GetCameraTile();
    private string? GetCurrentSessionMapName() => _host.GetCurrentSessionMapName();
    private IReadOnlyList<(int tileX, int tileY)> GetTileScopeList(TerrainTileScope scope) => _host.GetTileScopeList(scope);

    private const float TerrainWeakSignalRestoreMinZLimit = -8192f;
    private const float TerrainWeakSignalRestoreMaxZLimit = 512f;
    private const float TerrainWeakSignalShadowEdgeMinCoverage = 0.55f;
    private const float TerrainWeakSignalShadowLitMaxCoverage = 0.45f;
    private const float TerrainWeakSignalShadowEdgeMinHeightDelta = 0.5f;
    private readonly Dictionary<(int tileX, int tileY), List<Terrain.TerrainChunkData>> _terrainWeakSignalOriginalTiles = new();
    private readonly Dictionary<(int tileX, int tileY), int> _terrainWeakSignalAppliedPlans = new();
    private readonly HashSet<(int tileX, int tileY)> _terrainWeakSignalApplyingTiles = new();
    private Terrain.TerrainManager? _terrainWeakSignalHookedTerrainManager;
    private Terrain.VlmTerrainManager? _terrainWeakSignalHookedVlmTerrainManager;
    private string? _terrainWeakSignalWdlMapName;
    private WdlParser.WdlData? _terrainWeakSignalWdlData;
    private (int tileX, int tileY)? _terrainWeakSignalRestoreLastCameraTile;
    private bool _terrainWeakSignalRestoreNeedsRefresh = true;
    private float _stratigraphyVerticalOffsetZ = 0f;
    private readonly System.Collections.Concurrent.ConcurrentQueue<(int tileX, int tileY, List<Terrain.TerrainChunkData> chunks, int planSignature, string reason)> _pendingRestoredTilesQueue = new();
    private readonly HashSet<(int tileX, int tileY)> _terrainWeakSignalBackgroundComputingTiles = new();

    internal readonly struct TerrainWeakSignalSubChunkCell
    {
        public int CellX { get; init; }
        public int CellY { get; init; }
        public int DominantLayerIndex { get; init; }
        public float MinHeight { get; init; }
        public float MaxHeight { get; init; }
        public float AverageHeight { get; init; }
        public bool IsWeakSignalCandidate { get; init; }
        public bool TouchesBorder { get; init; }
    }

    internal sealed class TerrainWeakSignalTextureGuidance
    {
        public TerrainWeakSignalSubChunkCell[] Cells { get; init; } = Array.Empty<TerrainWeakSignalSubChunkCell>();
        public bool[] SelectedMask { get; init; } = Array.Empty<bool>();
        public int DominantLayerIndex { get; init; }
        public int SelectedCellCount { get; init; }
        public int BorderSelectedCellCount { get; init; }
        public float ObservedMinHeight { get; init; }
        public float ObservedMaxHeight { get; init; }
        public float ObservedAverageHeight { get; init; }
    }

    internal void ResetTerrainWeakSignalRestoreSessionState(bool preserveToggle)
    {
        DetachTerrainWeakSignalRestoreHooks();
        _terrainWeakSignalOriginalTiles.Clear();
        _terrainWeakSignalAppliedPlans.Clear();
        _terrainWeakSignalApplyingTiles.Clear();
        _terrainWeakSignalWdlMapName = null;
        _terrainWeakSignalWdlData = null;
        _terrainWeakSignalRestoreLastCameraTile = null;
        _terrainWeakSignalRestoreNeedsRefresh = true;

        if (!preserveToggle)
            _terrainWeakSignalRestoreEnabled = false;

        _terrainWeakSignalRestoreStatus = string.Empty;
    }

    internal void RefreshTerrainWeakSignalRestoreHooks()
    {
        if (!ReferenceEquals(_terrainWeakSignalHookedTerrainManager, _terrainManager))
        {
            if (_terrainWeakSignalHookedTerrainManager != null)
                _terrainWeakSignalHookedTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

            _terrainWeakSignalHookedTerrainManager = _terrainManager;
            if (_terrainWeakSignalHookedTerrainManager != null)
                _terrainWeakSignalHookedTerrainManager.OnTileLoaded += OnTerrainWeakSignalTileLoaded;
        }

        if (!ReferenceEquals(_terrainWeakSignalHookedVlmTerrainManager, _vlmTerrainManager))
        {
            if (_terrainWeakSignalHookedVlmTerrainManager != null)
                _terrainWeakSignalHookedVlmTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

            _terrainWeakSignalHookedVlmTerrainManager = _vlmTerrainManager;
            if (_terrainWeakSignalHookedVlmTerrainManager != null)
                _terrainWeakSignalHookedVlmTerrainManager.OnTileLoaded += OnTerrainWeakSignalTileLoaded;
        }
    }

    private void DetachTerrainWeakSignalRestoreHooks()
    {
        if (_terrainWeakSignalHookedTerrainManager != null)
            _terrainWeakSignalHookedTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

        if (_terrainWeakSignalHookedVlmTerrainManager != null)
            _terrainWeakSignalHookedVlmTerrainManager.OnTileLoaded -= OnTerrainWeakSignalTileLoaded;

        _terrainWeakSignalHookedTerrainManager = null;
        _terrainWeakSignalHookedVlmTerrainManager = null;
    }

    internal bool SetTerrainWeakSignalRestoreEnabled(bool enabled)
    {
        if (_terrainWeakSignalRestoreEnabled == enabled)
            return false;

        _terrainWeakSignalRestoreEnabled = enabled;
        _terrainWeakSignalRestoreNeedsRefresh = true;
        _terrainWeakSignalRestoreLastCameraTile = null;
        if (enabled)
        {
            RefreshTerrainWeakSignalRestoreHooks();
            RefreshTerrainWeakSignalRestoreForLoadedTiles();
        }
        else
        {
            RestoreAllTerrainWeakSignalTiles();
            _terrainWeakSignalRestoreStatus = "Weak-signal terrain restore disabled.";
        }

        return true;
    }

    internal void MarkTerrainWeakSignalRestoreDirty()
    {
        _terrainWeakSignalRestoreNeedsRefresh = true;
    }

    private bool TerrainWeakSignalRestoreUsesWorkbenchScope()
        => _terrainTileScope == TerrainTileScope.CurrentTile
            || _terrainTileScope == TerrainTileScope.CustomList
            || _terrainTileScope == TerrainTileScope.RectRange;

    private IReadOnlyList<(int tileX, int tileY)> GetTerrainWeakSignalRestoreScopedTiles()
    {
        if (_selectedChunks.Count > 0)
        {
            return _selectedChunks
                .Select(chunk => (chunk.tileX, chunk.tileY))
                .Distinct()
                .OrderBy(tile => tile.tileX)
                .ThenBy(tile => tile.tileY)
                .ToList();
        }

        if (TerrainWeakSignalRestoreUsesWorkbenchScope())
            return GetTileScopeList(_terrainTileScope);

        if (_terrainWeakSignalRestoreAllLoadedTiles)
            return GetTileScopeList(TerrainTileScope.LoadedTiles);

        return new List<(int tileX, int tileY)> { GetCameraTile() };
    }

    private bool HasTerrainWeakSignalScopedChunkSelectionForTile(int tileX, int tileY)
        => _selectedChunks.Count > 0 && _selectedChunks.Any(chunk => chunk.tileX == tileX && chunk.tileY == tileY);

    private bool IsTerrainWeakSignalRestoreTileInScope(int tileX, int tileY)
    {
        if (_selectedChunks.Count > 0)
            return HasTerrainWeakSignalScopedChunkSelectionForTile(tileX, tileY);

        if (TerrainWeakSignalRestoreUsesWorkbenchScope())
            return GetTerrainWeakSignalRestoreScopedTiles().Contains((tileX, tileY));

        if (_terrainWeakSignalRestoreAllLoadedTiles)
            return true;

        return GetCameraTile() == (tileX, tileY);
    }

    private bool IsTerrainWeakSignalRestoreChunkInScope(int tileX, int tileY, int chunkX, int chunkY)
    {
        if (_selectedChunks.Count > 0)
            return _selectedChunks.Contains((tileX, tileY, chunkX, chunkY));

        return IsTerrainWeakSignalRestoreTileInScope(tileX, tileY);
    }

    internal string GetTerrainWeakSignalRestoreScopeSummary()
    {
        return "camera tile + 4 neighbors";
    }

    internal void UpdateTerrainWeakSignalRestoreForCamera()
    {
        if (!_terrainWeakSignalRestoreEnabled)
            return;

        // Drain any asynchronously computed tile restorations with minimal GPU overhead (<0.5ms per tile)
        int processedThisFrame = 0;
        while (_pendingRestoredTilesQueue.TryDequeue(out var readyTile) && processedThisFrame < 4)
        {
            var key = (readyTile.tileX, readyTile.tileY);
            try
            {
                if (_terrainManager != null)
                    _terrainManager.ReplaceTileChunksAndRebuild(readyTile.tileX, readyTile.tileY, readyTile.chunks);
                else
                    _vlmTerrainManager?.ReplaceTileChunksAndRebuild(readyTile.tileX, readyTile.tileY, readyTile.chunks);

                _terrainWeakSignalAppliedPlans[key] = readyTile.planSignature;
                _terrainWeakSignalRestoreStatus = $"Weak-signal restore applied to tile ({readyTile.tileY}, {readyTile.tileX}) using {readyTile.reason}.";
                processedThisFrame++;
            }
            finally
            {
                _terrainWeakSignalBackgroundComputingTiles.Remove(key);
                _terrainWeakSignalApplyingTiles.Remove(key);
            }
        }

        var cameraTile = GetCameraTile();
        if (_terrainWeakSignalRestoreNeedsRefresh
            || _terrainWeakSignalRestoreLastCameraTile == null
            || _terrainWeakSignalRestoreLastCameraTile.Value != cameraTile)
        {
            _terrainWeakSignalRestoreLastCameraTile = cameraTile;
            RefreshTerrainWeakSignalRestoreForLoadedTiles();
            _terrainWeakSignalRestoreNeedsRefresh = false;
        }
    }

    internal void RefreshTerrainWeakSignalRestoreForLoadedTiles()
    {
        if (!_terrainWeakSignalRestoreEnabled)
            return;

        var loadedKeys = new HashSet<(int tileX, int tileY)>();

        if (_terrainManager != null)
        {
            foreach (var (tileX, tileY) in _terrainManager.LoadedTiles.ToList())
            {
                loadedKeys.Add((tileX, tileY));
                if (_terrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
                {
                    if (ShouldApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks))
                        ApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks);
                    else
                        RestoreTerrainWeakSignalTile((tileX, tileY), clearCache: true);
                }
            }
        }

        if (_vlmTerrainManager != null)
        {
            foreach (var (tileX, tileY) in _vlmTerrainManager.LoadedTiles.ToList())
            {
                loadedKeys.Add((tileX, tileY));
                if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
                {
                    if (ShouldApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks))
                        ApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks);
                    else
                        RestoreTerrainWeakSignalTile((tileX, tileY), clearCache: true);
                }
            }
        }

        foreach (var key in _terrainWeakSignalOriginalTiles.Keys.Where(key => !loadedKeys.Contains(key)).ToList())
        {
            _terrainWeakSignalOriginalTiles.Remove(key);
            _terrainWeakSignalAppliedPlans.Remove(key);
        }
    }

    private void RestoreAllTerrainWeakSignalTiles()
    {
        foreach (var key in _terrainWeakSignalOriginalTiles.Keys.ToList())
            RestoreTerrainWeakSignalTile(key, clearCache: true);

        _terrainWeakSignalOriginalTiles.Clear();
        _terrainWeakSignalAppliedPlans.Clear();
        _terrainWeakSignalApplyingTiles.Clear();
    }

    private void OnTerrainWeakSignalTileLoaded(int tileX, int tileY, WoWViewer.Terrain.TileLoadResult result)
    {
        if (!_terrainWeakSignalRestoreEnabled || result.Chunks.Count == 0)
            return;

        if (ShouldApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks))
            ApplyTerrainWeakSignalRestoreToTile(tileX, tileY, result.Chunks);
    }

    private bool ShouldApplyTerrainWeakSignalRestoreToTile(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        if (sourceChunks.Count == 0)
            return false;

        var cameraTile = GetCameraTile();
        int deltaX = Math.Abs(tileX - cameraTile.tileX);
        int deltaY = Math.Abs(tileY - cameraTile.tileY);
        if (deltaX + deltaY > 1)
            return false;

        var key = (tileX, tileY);
        IReadOnlyList<Terrain.TerrainChunkData> baseChunks = _terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks)
            ? originalChunks
            : sourceChunks;

        return HasTerrainWeakSignalRestoreWholeTileEvidence(tileX, tileY, baseChunks);
    }

    private bool IsTerrainWeakSignalRestoreCandidateHeightmap(TerrainHeightmapIo.TileHeightmap257 tileHeightmap)
    {
        GetTerrainWeakSignalRestoreCandidateRange(out float minHeight, out float maxHeight);
        return tileHeightmap.MinHeight >= minHeight && tileHeightmap.MaxHeight <= maxHeight;
    }

    private bool IsTerrainWeakSignalCandidateRange(float minHeight, float maxHeight)
    {
        GetTerrainWeakSignalRestoreCandidateRange(out float candidateMinHeight, out float candidateMaxHeight);
        return minHeight >= candidateMinHeight && maxHeight <= candidateMaxHeight;
    }

    private bool HasTerrainWeakSignalRestoreCandidateChunks(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        if (sourceChunks.Count == 0)
            return false;

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(sourceChunks);
        bool tileWeakSignalCandidate = IsTerrainWeakSignalRestoreCandidateHeightmap(tileHeightmap);

        for (int index = 0; index < sourceChunks.Count; index++)
        {
            Terrain.TerrainChunkData chunk = sourceChunks[index];
            if (!IsTerrainWeakSignalRestoreChunkInScope(tileX, tileY, chunk.ChunkX, chunk.ChunkY))
                continue;

            if (!TryBuildTerrainWeakSignalTextureGuidance(chunk, out TerrainWeakSignalTextureGuidance? textureGuidance) || textureGuidance == null)
                continue;

            if (!_terrainWeakSignalRestoreUseAutoFactor)
                return true;

            if (TryEstimateTerrainWeakSignalRestoreFactorForChunk(tileX, tileY, chunk, tileHeightmap, tileWeakSignalCandidate, textureGuidance, out _, out _))
                return true;
        }

        return false;
    }

    private bool HasTerrainWeakSignalRestoreWholeTileEvidence(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        if (sourceChunks.Count == 0)
            return false;

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(sourceChunks);
        if (IsTerrainWeakSignalRestoreCandidateHeightmap(tileHeightmap))
            return true;

        return TryGetTerrainWeakSignalTileObservedRange(sourceChunks, out _, out _, out _, out _);
    }

    internal void GetTerrainWeakSignalRestoreCandidateRange(out float minHeight, out float maxHeight)
    {
        minHeight = ClampTerrainWeakSignalRestoreZ(_terrainWeakSignalRestoreCandidateMinHeight);
        maxHeight = ClampTerrainWeakSignalRestoreZ(_terrainWeakSignalRestoreCandidateMaxHeight);
        if (minHeight > maxHeight)
            (minHeight, maxHeight) = (maxHeight, minHeight);
    }

    internal static float ClampTerrainWeakSignalRestoreZ(float value)
        => Math.Clamp(value, TerrainWeakSignalRestoreMinZLimit, TerrainWeakSignalRestoreMaxZLimit);

    private void ApplyTerrainWeakSignalRestoreToTile(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> sourceChunks)
    {
        var key = (tileX, tileY);
        if (_terrainWeakSignalApplyingTiles.Contains(key) || _terrainWeakSignalBackgroundComputingTiles.Contains(key) || sourceChunks.Count == 0)
            return;

        bool hasOriginal = _terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks);
        IReadOnlyList<Terrain.TerrainChunkData> baseChunks = hasOriginal
            ? originalChunks!
            : sourceChunks;

        if (!hasOriginal)
            _terrainWeakSignalOriginalTiles[key] = CloneTerrainChunkList(sourceChunks);

        _terrainWeakSignalBackgroundComputingTiles.Add(key);
        var chunkSnapshot = CloneTerrainChunkList(baseChunks);

        Task.Run(() =>
        {
            try
            {
                if (TryBuildTerrainWeakSignalRestoredChunks(tileX, tileY, chunkSnapshot, out var restoredChunks, out int planSignature, out string reason))
                {
                    if (hasOriginal
                        && _terrainWeakSignalAppliedPlans.TryGetValue(key, out int appliedPlanSignature)
                        && appliedPlanSignature == planSignature)
                    {
                        _terrainWeakSignalBackgroundComputingTiles.Remove(key);
                        return;
                    }

                    _pendingRestoredTilesQueue.Enqueue((tileX, tileY, restoredChunks, planSignature, reason));
                }
                else
                {
                    _terrainWeakSignalBackgroundComputingTiles.Remove(key);
                }
            }
            catch
            {
                _terrainWeakSignalBackgroundComputingTiles.Remove(key);
            }
        });
    }

    private void RestoreTerrainWeakSignalTile((int tileX, int tileY) key, bool clearCache)
    {
        if (!_terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks) || _terrainWeakSignalApplyingTiles.Contains(key))
            return;

        _terrainWeakSignalApplyingTiles.Add(key);
        try
        {
            if (_terrainManager != null)
                _terrainManager.ReplaceTileChunksAndRebuild(key.tileX, key.tileY, CloneTerrainChunkList(originalChunks));
            else
                _vlmTerrainManager?.ReplaceTileChunksAndRebuild(key.tileX, key.tileY, CloneTerrainChunkList(originalChunks));
        }
        finally
        {
            _terrainWeakSignalApplyingTiles.Remove(key);
        }

        if (clearCache)
        {
            _terrainWeakSignalOriginalTiles.Remove(key);
            _terrainWeakSignalAppliedPlans.Remove(key);
        }
    }

    private bool TryBuildTerrainWeakSignalRestoredChunks(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks,
        out List<Terrain.TerrainChunkData> restoredChunks,
        out int planSignature,
        out string reason)
    {
        return TryBuildTerrainWeakSignalRestoredWholeTile(tileX, tileY, sourceChunks, out restoredChunks, out planSignature, out reason);
    }

    private bool TryBuildTerrainWeakSignalRestoredWholeTile(
        int tileX,
        int tileY,
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks,
        out List<Terrain.TerrainChunkData> restoredChunks,
        out int planSignature,
        out string reason)
    {
        restoredChunks = new List<Terrain.TerrainChunkData>();
        planSignature = 0;
        reason = string.Empty;

        TerrainHeightmapIo.TileHeightmap257 tileHeightmap = TerrainHeightmapIo.BuildTileHeightmap257(sourceChunks);
        bool tileWeakSignalCandidate = IsTerrainWeakSignalRestoreCandidateHeightmap(tileHeightmap);
        bool hasPartialSignal = TryGetTerrainWeakSignalTileObservedRange(sourceChunks, out float observedMinHeight, out float observedMaxHeight, out int observedSignalCount, out bool usedTextureGuidance);
        if (!tileWeakSignalCandidate && !hasPartialSignal)
            return false;

        float factor;
        if (_terrainWeakSignalRestoreUseAutoFactor)
        {
            if (tileWeakSignalCandidate)
            {
                if (!TryEstimateTerrainWeakSignalRestoreFactor(tileX, tileY, tileHeightmap, out factor, out reason))
                    return false;
            }
            else if (!TryEstimateTerrainWeakSignalRestoreFactorForObservedRange(tileX, tileY, observedMinHeight, observedMaxHeight, out factor, out reason))
            {
                return false;
            }
        }
        else
        {
            factor = Math.Clamp(_terrainWeakSignalRestoreManualFactor, 1f, TerrainWeakSignalRestoreMaxFactor);
            if (factor <= 1.001f)
                return false;

            reason = "manual scale";
        }

        float? globalMaxHeight = TryGetTerrainWeakSignalGlobalMaxHeight(tileX, tileY, out float resolvedGlobalMaxHeight)
            ? resolvedGlobalMaxHeight
            : null;

        float anchorHeight = _stratigraphyAnchorMode switch
        {
            WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode.HighestZ_Ceiling => tileHeightmap.MaxHeight,
            WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode.MeanZ => (tileHeightmap.MinHeight + tileHeightmap.MaxHeight) * 0.5f,
            WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyAnchorMode.CustomDatum => 0f,
            _ => tileHeightmap.MinHeight < 0f ? tileHeightmap.MinHeight : 0f
        };

        float signedFactor = _stratigraphyPolarityInverted ? -factor : factor;
        float offsetZ = _stratigraphyVerticalOffsetZ;

        if (_stratigraphyUseNeighborAutoFit && TrySolveNeighborAutoFit(tileX, tileY, sourceChunks, out var autoFitResult))
        {
            signedFactor = autoFitResult.BestPolarityInverted ? -autoFitResult.BestFactor : autoFitResult.BestFactor;
            offsetZ = autoFitResult.BestVerticalOffsetZ;
            reason = $"neighbor auto-fit (scale={autoFitResult.BestFactor:F1}x, RMSE={autoFitResult.ResidualRmseMeters:F2}m)";
        }

        bool preserveNegativeFloor = anchorHeight < 0f && !_stratigraphyPolarityInverted;
        WdlParser.WdlTile? wdlTile = null;
        bool useWdlMagnetization = _stratigraphyUseWdlMagnetization && TryGetTerrainWeakSignalWdlTile(tileX, tileY, out wdlTile) && wdlTile != null;

        float[] restoredHeightmap = new float[tileHeightmap.Heights.Length];
        for (int index = 0; index < tileHeightmap.Heights.Length; index++)
        {
            float sourceHeight = tileHeightmap.Heights[index];
            float delta = sourceHeight - anchorHeight;
            float restoredHeight;

            if (useWdlMagnetization && wdlTile != null)
            {
                int yIdx = index / 257;
                int xIdx = index % 257;
                float normX = xIdx / 256f;
                float normY = yIdx / 256f;
                float wdlMacro = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.WdlLatticeMagnetizer.SampleWdlHeight(wdlTile!.Height17, normX, normY);
                float microRelief = delta * signedFactor;
                float direct = anchorHeight + microRelief + offsetZ;
                float magnetized = wdlMacro + microRelief + offsetZ;
                restoredHeight = (1f - _stratigraphyWdlMagnetizationStrength) * direct + (_stratigraphyWdlMagnetizationStrength * magnetized);
            }
            else
            {
                restoredHeight = anchorHeight + (delta * signedFactor) + offsetZ;
            }

            if (!preserveNegativeFloor && !_stratigraphyPolarityInverted && restoredHeight < 0f && _stratigraphyPreserveNegativeFloor)
                restoredHeight = 0f;
            if (globalMaxHeight.HasValue && restoredHeight > globalMaxHeight.Value)
                restoredHeight = globalMaxHeight.Value;

            restoredHeightmap[index] = restoredHeight;
        }

        List<Terrain.TerrainChunkData> wholeTileRestoredChunks = TerrainHeightmapIo.ApplyHeightmap257ToChunks(sourceChunks, restoredHeightmap);
        if (_stratigraphyUnhideDevMeshes)
        {
            for (int i = 0; i < wholeTileRestoredChunks.Count; i++)
            {
                var ch = wholeTileRestoredChunks[i];
                if (ch.HoleMask != 0)
                    wholeTileRestoredChunks[i] = CloneTerrainChunk(ch, holeMask: 0);
            }
        }

        if (_terrainWeakSignalRestoreUseTextureSubdivisions)
        {
            List<Terrain.TerrainChunkData> maskedChunks = CloneTerrainChunkList(sourceChunks);
            var maskedPlanHash = new HashCode();
            int maskedChunkCount = 0;
            int guidedCellCount = 0;

            for (int index = 0; index < sourceChunks.Count; index++)
            {
                Terrain.TerrainChunkData chunk = sourceChunks[index];
                if (!TryBuildTerrainWeakSignalTextureGuidance(chunk, out TerrainWeakSignalTextureGuidance? textureGuidance) || textureGuidance == null)
                    continue;

                float[]? vertexWeights = BuildTerrainWeakSignalTextureGuidanceVertexWeights(textureGuidance);
                Terrain.TerrainChunkData restoredChunk = wholeTileRestoredChunks[index];

                float[] restoredHeights = BlendTerrainWeakSignalMaskedChunkHeights(chunk.Heights, restoredChunk.Heights, vertexWeights);
                Vector3[] restoredNormals = GenerateNormalsForChunk(chunk, restoredHeights, _stratigraphyUnhideDevMeshes ? 0 : chunk.HoleMask);
                maskedChunks[index] = CloneTerrainChunk(chunk, heights: restoredHeights, normals: restoredNormals, holeMask: _stratigraphyUnhideDevMeshes ? 0 : (int?)null);

                maskedPlanHash.Add(chunk.ChunkX);
                maskedPlanHash.Add(chunk.ChunkY);
                maskedPlanHash.Add(textureGuidance.SelectedCellCount);
                maskedPlanHash.Add(GetTerrainWeakSignalSelectedMaskHash(textureGuidance.SelectedMask));
                maskedPlanHash.Add((int)MathF.Round(factor * 1000f));

                maskedChunkCount++;
                guidedCellCount += textureGuidance.SelectedCellCount;
            }

            if (maskedChunkCount > 0)
            {
                restoredChunks = maskedChunks;
                planSignature = maskedPlanHash.ToHashCode();
                string maskedSignalSummary = tileWeakSignalCandidate
                    ? "whole-tile weak range"
                    : $"{observedSignalCount} partial weak signal source(s){(usedTextureGuidance ? ", cell-guided" : string.Empty)}";
                reason += $", whole-tile factor clamped to {maskedChunkCount} cell-guided chunk(s) / {guidedCellCount} weak sub-cell(s) via {maskedSignalSummary}";
                return true;
            }
        }

        restoredChunks = wholeTileRestoredChunks;
        planSignature = HashCode.Combine(false, (int)MathF.Round(factor * 1000f));
        string signalSummary = tileWeakSignalCandidate
            ? "whole-tile weak range"
            : $"{observedSignalCount} partial weak signal source(s){(usedTextureGuidance ? ", cell-guided" : string.Empty)}";
        reason += preserveNegativeFloor
            ? $", whole tile from source floor via {signalSummary}"
            : $", whole tile from z=0 via {signalSummary}";
        return true;
    }

    private bool TrySolveNeighborAutoFit(
        int targetTileX,
        int targetTileY,
        IReadOnlyList<Terrain.TerrainChunkData> targetChunks,
        out WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborAutoFitResult result)
    {
        result = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborAutoFitResult.None;
        var boundaryPairs = new List<WowViewer.Core.Runtime.World.Terrain.Stratigraphy.BoundaryVertexPair>();

        var adjacentOffsets = new (int dx, int dy, WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction dir)[]
        {
            (-1, 0, WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.North),
            (1, 0, WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.South),
            (0, -1, WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.West),
            (0, 1, WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.East),
        };

        Span<float> tEdge = stackalloc float[9];
        Span<float> nEdge = stackalloc float[9];

        foreach (var (dx, dy, dir) in adjacentOffsets)
        {
            int nTileX = targetTileX + dx;
            int nTileY = targetTileY + dy;
            if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(nTileX, nTileY, out var nResult) && nResult.Chunks.Count > 0)
            {
                for (int c = 0; c < 16; c++)
                {
                    int targetCx = dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.North ? 0 :
                                   dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.South ? 15 : c;
                    int targetCy = dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.West ? 0 :
                                   dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.East ? 15 : c;

                    int neighborCx = dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.North ? 15 :
                                     dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.South ? 0 : c;
                    int neighborCy = dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.West ? 15 :
                                     dir is WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.East ? 0 : c;

                    var targetChunk = targetChunks.FirstOrDefault(ch => ch.ChunkX == targetCx && ch.ChunkY == targetCy);
                    var neighborChunk = nResult.Chunks.FirstOrDefault(ch => ch.ChunkX == neighborCx && ch.ChunkY == neighborCy);

                    if (targetChunk?.Heights != null && neighborChunk?.Heights != null)
                    {
                        WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.ExtractEdge145(targetChunk.Heights, dir, tEdge);

                        var oppDir = dir switch
                        {
                            WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.North => WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.South,
                            WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.South => WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.North,
                            WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.West => WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.East,
                            _ => WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.Direction.West
                        };
                        WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.ExtractEdge145(neighborChunk.Heights, oppDir, nEdge);

                        for (int i = 0; i < 9; i++)
                            boundaryPairs.Add(new WowViewer.Core.Runtime.World.Terrain.Stratigraphy.BoundaryVertexPair(nEdge[i], tEdge[i]));
                    }
                }
            }
        }

        if (boundaryPairs.Count >= 9)
        {
            result = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.NeighborMeshHeightSolver.SolveFromBoundaryPairs(boundaryPairs);
            return result.FoundNeighbor;
        }

        return false;
    }

    private bool TryEstimateTerrainWeakSignalRestoreFactorForObservedRange(
        int tileX,
        int tileY,
        float observedMin,
        float observedMax,
        out float factor,
        out string reason)
    {
        factor = 1f;
        reason = string.Empty;

        float observedRange = Math.Max(observedMax - observedMin, 0f);
        if (observedRange < 0.25f)
            return false;

        if (TryGetTerrainWeakSignalLoadedBounds(out float loadedMin, out float loadedMax, out int loadedTileCount))
        {
            float loadedRange = Math.Max(loadedMax - loadedMin, 0f);
            float visibilityRatio = loadedRange > 0.001f
                ? observedRange / loadedRange
                : 1f;
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, loadedMin, loadedMax);
            if (visibilityRatio <= 0.25f && rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"partial-signal relief {observedMin:F1}..{observedMax:F1} vs loaded {loadedMin:F1}..{loadedMax:F1} across {loadedTileCount} tile(s)";
                return true;
            }
        }

        if (TryGetTerrainWeakSignalWdlBounds(tileX, tileY, out float coarseMin, out float coarseMax))
        {
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, coarseMin, coarseMax);
            if (rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"partial-signal relief {observedMin:F1}..{observedMax:F1} vs WDL {coarseMin:F1}..{coarseMax:F1}";
                return true;
            }
        }

        float fallbackFactor = EstimateTerrainWeakSignalFallbackFactor(observedMin, observedMax);
        if (fallbackFactor >= 1.25f)
        {
            factor = fallbackFactor;
            reason = $"partial-signal fallback {observedMin:F1}..{observedMax:F1}";
            return true;
        }

        return false;
    }

    private bool TryGetTerrainWeakSignalTileObservedRange(
        IReadOnlyList<Terrain.TerrainChunkData> sourceChunks,
        out float minHeight,
        out float maxHeight,
        out int signalCount,
        out bool usedTextureGuidance)
    {
        minHeight = float.MaxValue;
        maxHeight = float.MinValue;
        signalCount = 0;
        usedTextureGuidance = false;

        for (int index = 0; index < sourceChunks.Count; index++)
        {
            Terrain.TerrainChunkData chunk = sourceChunks[index];
            if (!_terrainWeakSignalRestoreUseTextureSubdivisions)
                continue;

            if (!TryBuildTerrainWeakSignalTextureGuidance(chunk, out TerrainWeakSignalTextureGuidance? textureGuidance) || textureGuidance == null)
                continue;

            if (textureGuidance.ObservedMinHeight < minHeight)
                minHeight = textureGuidance.ObservedMinHeight;
            if (textureGuidance.ObservedMaxHeight > maxHeight)
                maxHeight = textureGuidance.ObservedMaxHeight;
            signalCount++;
            usedTextureGuidance = true;
        }

        return signalCount > 0 && minHeight != float.MaxValue && maxHeight != float.MinValue && maxHeight > minHeight;
    }

    private bool TryEstimateTerrainWeakSignalRestoreFactorForChunk(
        int tileX,
        int tileY,
        Terrain.TerrainChunkData chunk,
        TerrainHeightmapIo.TileHeightmap257 tileHeightmap,
        bool tileWeakSignalCandidate,
        TerrainWeakSignalTextureGuidance? textureGuidance,
        out float factor,
        out string reason)
    {
        factor = 1f;
        reason = string.Empty;

        float observedMin;
        float observedMax;
        if (textureGuidance != null)
        {
            observedMin = textureGuidance.ObservedMinHeight;
            observedMax = textureGuidance.ObservedMaxHeight;
        }
        else if (!TryGetTerrainChunkHeightRange(chunk, out observedMin, out observedMax))
        {
            return false;
        }

        if (TryGetTerrainWeakSignalWdlChunkBounds(tileX, tileY, chunk.ChunkX, chunk.ChunkY, out float chunkGuideMin, out float chunkGuideMax, out _))
        {
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, chunkGuideMin, chunkGuideMax);
            if (rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = textureGuidance != null
                    ? $"WDL chunk relief {chunkGuideMin:F1}..{chunkGuideMax:F1} from {DescribeTerrainWeakSignalGuidance(textureGuidance)}"
                    : $"WDL chunk relief {chunkGuideMin:F1}..{chunkGuideMax:F1}";
                return true;
            }
        }

        if (!tileWeakSignalCandidate)
        {
            float mixedTileFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, tileHeightmap.MinHeight, tileHeightmap.MaxHeight);
            if (mixedTileFactor >= 1.25f)
            {
                factor = mixedTileFactor;
                reason = textureGuidance != null
                    ? $"mixed-tile relief {tileHeightmap.MinHeight:F1}..{tileHeightmap.MaxHeight:F1} from {DescribeTerrainWeakSignalGuidance(textureGuidance)}"
                    : $"mixed-tile relief {tileHeightmap.MinHeight:F1}..{tileHeightmap.MaxHeight:F1}";
                return true;
            }
        }

        if (tileWeakSignalCandidate && TryEstimateTerrainWeakSignalRestoreFactor(tileX, tileY, tileHeightmap, out float tileFactor, out string tileReason))
        {
            factor = tileFactor;
            reason = tileReason;
            return true;
        }

        float fallbackFactor = EstimateTerrainWeakSignalFallbackFactor(observedMin, observedMax);
        if (fallbackFactor >= 1.25f)
        {
            factor = fallbackFactor;
            reason = "chunk sea-level fallback";
            return true;
        }

        return false;
    }

    private bool TryEstimateTerrainWeakSignalRestoreFactor(
        int tileX,
        int tileY,
        TerrainHeightmapIo.TileHeightmap257 tileHeightmap,
        out float factor,
        out string reason)
    {
        factor = 1f;
        reason = string.Empty;

        float observedMin = tileHeightmap.MinHeight;
        float observedMax = tileHeightmap.MaxHeight;
        float observedRange = Math.Max(observedMax - observedMin, 0f);
        if (observedRange < 0.25f)
            return false;

        if (TryGetTerrainWeakSignalLoadedBounds(out float loadedMin, out float loadedMax, out int loadedTileCount))
        {
            float loadedRange = Math.Max(loadedMax - loadedMin, 0f);
            float visibilityRatio = loadedRange > 0.001f
                ? observedRange / loadedRange
                : 1f;
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, loadedMin, loadedMax);
            if (visibilityRatio <= 0.25f && rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"loaded-tile relief {loadedMin:F1}..{loadedMax:F1} across {loadedTileCount} tile(s)";
                return true;
            }
        }

        if (TryGetTerrainWeakSignalWdlBounds(tileX, tileY, out float coarseMin, out float coarseMax))
        {
            float rawFactor = EstimateTerrainWeakSignalRestoreFactorFromRanges(observedMin, observedMax, coarseMin, coarseMax);
            if (rawFactor >= 1.25f)
            {
                factor = rawFactor;
                reason = $"WDL coarse relief {coarseMin:F1}..{coarseMax:F1}";
                return true;
            }
        }

        float fallbackFactor = EstimateTerrainWeakSignalFallbackFactor(observedMin, observedMax);
        if (fallbackFactor >= 1.25f)
        {
            factor = fallbackFactor;
            reason = "sea-level fallback";
            return true;
        }

        return false;
    }

    private bool TryGetTerrainWeakSignalLoadedBounds(out float minHeight, out float maxHeight, out int tileCount)
    {
        float localMinHeight = float.MaxValue;
        float localMaxHeight = float.MinValue;
        int localTileCount = 0;

        void AccumulateTile(int tileX, int tileY, IReadOnlyList<Terrain.TerrainChunkData> chunks)
        {
            var key = (tileX, tileY);
            IReadOnlyList<Terrain.TerrainChunkData> baseChunks = _terrainWeakSignalOriginalTiles.TryGetValue(key, out var originalChunks)
                ? originalChunks
                : chunks;

            TerrainHeightmapIo.TileHeightmap257 heightmap = TerrainHeightmapIo.BuildTileHeightmap257(baseChunks);
            if (float.IsNaN(heightmap.MinHeight) || float.IsNaN(heightmap.MaxHeight))
                return;

            if (heightmap.MinHeight < localMinHeight)
                localMinHeight = heightmap.MinHeight;

            if (heightmap.MaxHeight > localMaxHeight)
                localMaxHeight = heightmap.MaxHeight;

            localTileCount++;
        }

        if (_terrainManager != null)
        {
            foreach (var (tileX, tileY) in _terrainManager.LoadedTiles)
            {
                if (_terrainManager.TryGetTileLoadResult(tileX, tileY, out var result) && result.Chunks.Count > 0)
                    AccumulateTile(tileX, tileY, result.Chunks);
            }
        }

        if (_vlmTerrainManager != null)
        {
            foreach (var (tileX, tileY) in _vlmTerrainManager.LoadedTiles)
            {
                if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var result) && result.Chunks.Count > 0)
                    AccumulateTile(tileX, tileY, result.Chunks);
            }
        }

        minHeight = localMinHeight;
        maxHeight = localMaxHeight;
        tileCount = localTileCount;
        return tileCount > 0 && minHeight != float.MaxValue && maxHeight != float.MinValue && maxHeight > minHeight;
    }

    private bool TryGetTerrainWeakSignalGlobalMaxHeight(int tileX, int tileY, out float maxHeight)
    {
        maxHeight = float.MinValue;

        if (TryGetTerrainWeakSignalWdlTile(tileX, tileY, out _) && _terrainWeakSignalWdlData != null)
        {
            for (int index = 0; index < _terrainWeakSignalWdlData.Tiles.Length; index++)
            {
                WdlParser.WdlTile? tile = _terrainWeakSignalWdlData.Tiles[index];
                if (tile?.HasData != true)
                    continue;

                if (float.IsNaN(tile.MaxZ) || float.IsInfinity(tile.MaxZ))
                    continue;

                if (tile.MaxZ > maxHeight)
                    maxHeight = tile.MaxZ;
            }

            if (maxHeight != float.MinValue)
                return true;
        }

        if (TryGetTerrainWeakSignalLoadedBounds(out _, out float loadedMaxHeight, out _))
        {
            maxHeight = loadedMaxHeight;
            return true;
        }

        return false;
    }

    internal bool TryGetTerrainWeakSignalWdlTile(int tileX, int tileY, out WdlParser.WdlTile? tile)
    {
        tile = null;

        if (_dataSource == null)
            return false;

        string? mapName = _terrainManager?.MapName ?? GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(mapName))
            return false;

        if (!string.Equals(_terrainWeakSignalWdlMapName, mapName, StringComparison.OrdinalIgnoreCase))
        {
            _terrainWeakSignalWdlMapName = mapName;
            _terrainWeakSignalWdlData = null;
        }

        if (_terrainWeakSignalWdlData == null)
        {
            if (_wdlPreviewCacheService != null)
                _wdlPreviewCacheService.EnsurePrefetch(mapName);

            if (!WdlDataSourceResolver.TryReadWdlBytes(_dataSource, mapName, out byte[]? wdlBytes, out _)
                || wdlBytes == null
                || wdlBytes.Length == 0)
            {
                return false;
            }

            _terrainWeakSignalWdlData = WdlParser.Parse(wdlBytes);
        }

        if (_terrainWeakSignalWdlData == null)
            return false;

        int tileIndex = tileX * 64 + tileY;
        if ((uint)tileIndex >= _terrainWeakSignalWdlData.Tiles.Length)
            return false;

        tile = _terrainWeakSignalWdlData.Tiles[tileIndex];
        return tile?.HasData == true;
    }

    /// <summary>
    /// Spec 232 T064: feeds the base map's parsed WDL to the terrain adapter so composed layers
    /// with a magnetic edge-snap strength can blend their footprint-boundary heights toward it.
    /// The parse is shared with the stratigraphy weak-signal path (same cache fields).
    /// </summary>
    internal void WireBaseWdlEdgeBlendLookup()
    {
        if (_terrainManager?.Adapter is not Terrain.ITerrainAdapter adapter)
            return;

        if (adapter is Terrain.AlphaTerrainAdapter alphaAdapter)
            alphaAdapter.BaseWdlTileLookup = ResolveBaseWdlTile;
        else if (adapter is Terrain.StandardTerrainAdapter standardAdapter)
            standardAdapter.BaseWdlTileLookup = ResolveBaseWdlTile;
    }

    private WdlParser.WdlTile? ResolveBaseWdlTile(int tileX, int tileY)
        => TryGetTerrainWeakSignalWdlTile(tileX, tileY, out WdlParser.WdlTile? tile) ? tile : null;

    private bool TryGetTerrainWeakSignalWdlBounds(int tileX, int tileY, out float minHeight, out float maxHeight)
    {
        minHeight = 0f;
        maxHeight = 0f;
        if (!TryGetTerrainWeakSignalWdlTile(tileX, tileY, out WdlParser.WdlTile? tile) || tile == null)
            return false;

        minHeight = tile.MinZ;
        maxHeight = tile.MaxZ;
        return maxHeight > minHeight;
    }

    private bool TryGetTerrainWeakSignalWdlChunkBounds(
        int tileX,
        int tileY,
        int chunkX,
        int chunkY,
        out float minHeight,
        out float maxHeight,
        out float centerHeight)
    {
        minHeight = 0f;
        maxHeight = 0f;
        centerHeight = 0f;

        if ((uint)chunkX >= 16u || (uint)chunkY >= 16u)
            return false;

        if (!TryGetTerrainWeakSignalWdlTile(tileX, tileY, out WdlParser.WdlTile? tile) || tile == null)
            return false;

        float h00 = tile.Height17[chunkY, chunkX];
        float h10 = tile.Height17[chunkY, chunkX + 1];
        float h01 = tile.Height17[chunkY + 1, chunkX];
        float h11 = tile.Height17[chunkY + 1, chunkX + 1];
        centerHeight = tile.Height16[chunkY, chunkX];

        minHeight = MathF.Min(MathF.Min(h00, h10), MathF.Min(MathF.Min(h01, h11), centerHeight));
        maxHeight = MathF.Max(MathF.Max(h00, h10), MathF.Max(MathF.Max(h01, h11), centerHeight));
        return maxHeight > minHeight;
    }

    private static float EstimateTerrainWeakSignalRestoreFactorFromRanges(float observedMin, float observedMax, float coarseMin, float coarseMax)
    {
        const float epsilon = 0.001f;
        float observedRange = Math.Max(observedMax - observedMin, 0f);
        float coarseRange = Math.Max(coarseMax - coarseMin, 0f);
        if (observedRange <= epsilon || coarseRange <= epsilon)
            return 1f;

        float rawFactor = 1f;
        if (coarseRange > observedRange * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseRange / observedRange);

        float observedBelow = Math.Max(0f, -observedMin);
        float coarseBelow = Math.Max(0f, -coarseMin);
        if (observedBelow > epsilon && coarseBelow > observedBelow * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseBelow / observedBelow);

        float observedAbove = Math.Max(0f, observedMax);
        float coarseAbove = Math.Max(0f, coarseMax);
        if (observedAbove > epsilon && coarseAbove > observedAbove * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseAbove / observedAbove);

        return Math.Clamp(rawFactor, 1f, TerrainWeakSignalRestoreMaxFactor);
    }

    private static float EstimateTerrainWeakSignalFallbackFactor(float observedMin, float observedMax)
    {
        float observedBelow = Math.Max(0f, -observedMin);
        float observedRange = Math.Max(observedMax - observedMin, 0f);
        if (observedBelow < 0.5f || observedRange > 12f || observedMax > 4f)
            return 1f;

        float rawFactor = 16f / observedBelow;
        return Math.Clamp(rawFactor, 1f, TerrainWeakSignalRestoreMaxFactor);
    }

    private static float SnapTerrainWeakSignalRestoreFactor(float rawFactor)
    {
        if (rawFactor <= 1f)
            return 1f;

        float[] supported = { 1f, 2f, 4f, 8f, 16f, 32f, 64f, 128f, 256f, 512f };
        foreach (float value in supported)
        {
            if (rawFactor <= value)
                return value;
        }

        return supported[^1];
    }

    private static bool TryGetTerrainChunkHeightRange(Terrain.TerrainChunkData chunk, out float minHeight, out float maxHeight)
    {
        minHeight = float.MaxValue;
        maxHeight = float.MinValue;

        if (chunk.Heights == null || chunk.Heights.Length == 0)
            return false;

        for (int index = 0; index < chunk.Heights.Length; index++)
        {
            float height = chunk.Heights[index];
            if (float.IsNaN(height) || float.IsInfinity(height))
                continue;

            if (height < minHeight)
                minHeight = height;

            if (height > maxHeight)
                maxHeight = height;
        }

        return minHeight != float.MaxValue && maxHeight != float.MinValue && maxHeight > minHeight;
    }

    private static bool HasTerrainWeakSignalShadowSignal(Terrain.TerrainChunkData chunk)
    {
        if (chunk.ShadowMap == null || chunk.ShadowMap.Length == 0)
            return false;

        for (int index = 0; index < chunk.ShadowMap.Length; index++)
        {
            if (chunk.ShadowMap[index] != 0)
                return true;
        }

        return false;
    }

    private bool TryBuildTerrainWeakSignalShadowEdgeVertexWeights(
        Terrain.TerrainChunkData chunk,
        TerrainWeakSignalTextureGuidance? textureGuidance,
        out float[]? vertexWeights)
    {
        vertexWeights = null;
        if (!HasTerrainWeakSignalShadowSignal(chunk) || chunk.ShadowMap == null || chunk.ShadowMap.Length < 64 * 64)
            return false;

        const int subDivisions = 8;
        bool[] selectedMask = new bool[subDivisions * subDivisions];
        float[] shadowCoverage = new float[subDivisions * subDivisions];
        float[] averageHeights = new float[subDivisions * subDivisions];
        int[] dominantLayers = new int[subDivisions * subDivisions];

        for (int cellY = 0; cellY < subDivisions; cellY++)
        {
            for (int cellX = 0; cellX < subDivisions; cellX++)
            {
                int cellIndex = cellY * subDivisions + cellX;
                shadowCoverage[cellIndex] = ComputeTerrainWeakSignalShadowCoverage(chunk.ShadowMap, cellX, cellY, subDivisions);
                averageHeights[cellIndex] = ComputeTerrainWeakSignalAverageHeightForSubCell(chunk, cellX, cellY, subDivisions);
                dominantLayers[cellIndex] = GetTerrainWeakSignalDominantLayerForSubChunkCell(chunk, cellX, cellY, subDivisions);
                if (textureGuidance != null && textureGuidance.SelectedMask.Length == selectedMask.Length && textureGuidance.SelectedMask[cellIndex])
                    selectedMask[cellIndex] = true;
            }
        }

        int preferredLayer = textureGuidance?.DominantLayerIndex ?? -1;
        bool hasDirectionalShadowAnchor = TryInferTerrainWeakSignalShadowDirection(
            textureGuidance?.SelectedMask,
            shadowCoverage,
            averageHeights,
            dominantLayers,
            subDivisions,
            preferredLayer,
            out var litToShadowOffset);
        int selectedCellCount = selectedMask.Count(static value => value);
        for (int cellY = 0; cellY < subDivisions; cellY++)
        {
            for (int cellX = 0; cellX < subDivisions; cellX++)
            {
                int cellIndex = cellY * subDivisions + cellX;
                float coverage = shadowCoverage[cellIndex];
                if (coverage > TerrainWeakSignalShadowLitMaxCoverage)
                    continue;

                if (preferredLayer >= 0 && dominantLayers[cellIndex] != preferredLayer)
                    continue;

                bool touchesSeed = textureGuidance == null
                    || CellTouchesSelectedMask(textureGuidance.SelectedMask, subDivisions, cellX, cellY);
                if (!touchesSeed)
                    continue;

                bool foundShadowNeighbor = hasDirectionalShadowAnchor
                    ? TryGetTerrainWeakSignalDirectionalShadowNeighborAverageHeight(
                        shadowCoverage,
                        averageHeights,
                        subDivisions,
                        cellX,
                        cellY,
                        litToShadowOffset.offsetX,
                        litToShadowOffset.offsetY,
                        out float shadowNeighborAverageHeight)
                    : TryGetTerrainWeakSignalShadowNeighborHeightRange(shadowCoverage, averageHeights, subDivisions, cellX, cellY, out shadowNeighborAverageHeight);
                if (!foundShadowNeighbor)
                    continue;

                if (averageHeights[cellIndex] + TerrainWeakSignalShadowEdgeMinHeightDelta < shadowNeighborAverageHeight)
                    continue;

                if (!selectedMask[cellIndex])
                {
                    selectedMask[cellIndex] = true;
                    selectedCellCount++;
                }
            }
        }

        for (int pass = 0; pass < 2; pass++)
        {
            bool changed = false;
            bool[] nextMask = (bool[])selectedMask.Clone();
            for (int cellY = 0; cellY < subDivisions; cellY++)
            {
                for (int cellX = 0; cellX < subDivisions; cellX++)
                {
                    int cellIndex = cellY * subDivisions + cellX;
                    if (nextMask[cellIndex] || shadowCoverage[cellIndex] > TerrainWeakSignalShadowLitMaxCoverage)
                        continue;

                    if (preferredLayer >= 0 && dominantLayers[cellIndex] != preferredLayer)
                        continue;

                    bool foundSelectedNeighbor = hasDirectionalShadowAnchor
                        ? TryGetTerrainWeakSignalDirectionalSelectedNeighborAverageHeight(
                            selectedMask,
                            averageHeights,
                            subDivisions,
                            cellX,
                            cellY,
                            litToShadowOffset.offsetX,
                            litToShadowOffset.offsetY,
                            out float selectedNeighborAverageHeight)
                        : TryGetTerrainWeakSignalSelectedNeighborAverageHeight(selectedMask, averageHeights, subDivisions, cellX, cellY, out selectedNeighborAverageHeight);
                    if (!foundSelectedNeighbor)
                        continue;

                    if (averageHeights[cellIndex] + 0.25f < selectedNeighborAverageHeight)
                        continue;

                    nextMask[cellIndex] = true;
                    selectedCellCount++;
                    changed = true;
                }
            }

            selectedMask = nextMask;
            if (!changed)
                break;
        }

        if (selectedCellCount == 0)
            return false;

        vertexWeights = BuildTerrainWeakSignalSubCellVertexWeights(selectedMask, subDivisions);
        return true;
    }

    internal bool TryBuildTerrainWeakSignalTextureGuidance(Terrain.TerrainChunkData chunk, out TerrainWeakSignalTextureGuidance? guidance)
    {
        guidance = null;

        const int subDivisions = 8;
        float cellSize = WoWConstants.ChunkSize / subDivisions;
        var cells = new TerrainWeakSignalSubChunkCell[subDivisions * subDivisions];
        float selectedMinHeight = float.MaxValue;
        float selectedMaxHeight = float.MinValue;
        float selectedAverageHeightSum = 0f;

        for (int cellY = 0; cellY < subDivisions; cellY++)
        {
            for (int cellX = 0; cellX < subDivisions; cellX++)
            {
                float minHeight = float.MaxValue;
                float maxHeight = float.MinValue;
                float averageHeight = 0f;
                int sampleCount = 0;

                for (int sampleY = 0; sampleY < 3; sampleY++)
                {
                    for (int sampleX = 0; sampleX < 3; sampleX++)
                    {
                        float localX = cellX * cellSize + ((sampleX + 0.5f) / 3f) * cellSize;
                        float localY = cellY * cellSize + ((sampleY + 0.5f) / 3f) * cellSize;
                        float height = SampleHeightOuterGrid(chunk, localX, localY);
                        if (height < minHeight)
                            minHeight = height;
                        if (height > maxHeight)
                            maxHeight = height;
                        averageHeight += height;
                        sampleCount++;
                    }
                }

                averageHeight = sampleCount > 0 ? averageHeight / sampleCount : 0f;
                int dominantLayerIndex = GetTerrainWeakSignalDominantLayerForSubChunkCell(chunk, cellX, cellY, subDivisions);
                bool isWeakSignalCandidate = minHeight != float.MaxValue
                    && maxHeight != float.MinValue
                    && IsTerrainWeakSignalCandidateRange(minHeight, maxHeight);
                bool touchesBorder = cellX == 0 || cellY == 0 || cellX == subDivisions - 1 || cellY == subDivisions - 1;

                var cell = new TerrainWeakSignalSubChunkCell
                {
                    CellX = cellX,
                    CellY = cellY,
                    DominantLayerIndex = dominantLayerIndex,
                    MinHeight = minHeight,
                    MaxHeight = maxHeight,
                    AverageHeight = averageHeight,
                    IsWeakSignalCandidate = isWeakSignalCandidate,
                    TouchesBorder = touchesBorder,
                };
                cells[cellY * subDivisions + cellX] = cell;
            }
        }

        bool[] selectedMask = new bool[subDivisions * subDivisions];
        int selectedCellCount = 0;
        int borderSelectedCellCount = 0;
        for (int index = 0; index < cells.Length; index++)
        {
            TerrainWeakSignalSubChunkCell cell = cells[index];
            if (!cell.IsWeakSignalCandidate)
                continue;

            selectedMask[index] = true;
            selectedCellCount++;
            if (cell.TouchesBorder)
                borderSelectedCellCount++;
            if (cell.MinHeight < selectedMinHeight)
                selectedMinHeight = cell.MinHeight;
            if (cell.MaxHeight > selectedMaxHeight)
                selectedMaxHeight = cell.MaxHeight;
            selectedAverageHeightSum += cell.AverageHeight;
        }

        if (selectedCellCount == 0)
            return false;

        guidance = new TerrainWeakSignalTextureGuidance
        {
            Cells = cells,
            SelectedMask = selectedMask,
            DominantLayerIndex = -1,
            SelectedCellCount = selectedCellCount,
            BorderSelectedCellCount = borderSelectedCellCount,
            ObservedMinHeight = selectedMinHeight,
            ObservedMaxHeight = selectedMaxHeight,
            ObservedAverageHeight = selectedAverageHeightSum / selectedCellCount,
        };
        return true;
    }

    private static int GetTerrainWeakSignalDominantLayerForSubChunkCell(Terrain.TerrainChunkData chunk, int cellX, int cellY, int subDivisions)
    {
        const int alphaSize = 64;
        int pixelStartX = cellX * (alphaSize / subDivisions);
        int pixelStartY = cellY * (alphaSize / subDivisions);
        int pixelEndX = pixelStartX + (alphaSize / subDivisions);
        int pixelEndY = pixelStartY + (alphaSize / subDivisions);

        float[] layerSums = new float[Math.Max(chunk.Layers.Length, 1)];
        for (int y = pixelStartY; y < pixelEndY; y++)
        {
            for (int x = pixelStartX; x < pixelEndX; x++)
            {
                int pixelIndex = y * alphaSize + x;
                float maxOverlay = 0f;
                for (int layerIndex = 1; layerIndex < chunk.Layers.Length; layerIndex++)
                {
                    if (!chunk.AlphaMaps.TryGetValue(layerIndex, out byte[]? alphaMap) || alphaMap.Length <= pixelIndex)
                        continue;

                    float weight = alphaMap[pixelIndex];
                    layerSums[layerIndex] += weight;
                    if (weight > maxOverlay)
                        maxOverlay = weight;
                }

                layerSums[0] += Math.Max(0f, 255f - maxOverlay);
            }
        }

        int dominantLayerIndex = 0;
        float dominantWeight = layerSums[0];
        for (int layerIndex = 1; layerIndex < layerSums.Length; layerIndex++)
        {
            if (layerSums[layerIndex] > dominantWeight)
            {
                dominantWeight = layerSums[layerIndex];
                dominantLayerIndex = layerIndex;
            }
        }

        return dominantLayerIndex;
    }

    private static float[] BuildTerrainWeakSignalTextureGuidanceVertexWeights(TerrainWeakSignalTextureGuidance guidance)
    {
        const int subDivisions = 8;
        return BuildTerrainWeakSignalSubCellVertexWeights(guidance.SelectedMask, subDivisions);
    }

    private static string DescribeTerrainWeakSignalGuidance(TerrainWeakSignalTextureGuidance guidance)
    {
        return $"cell-guided weak cells across {guidance.SelectedCellCount} cell(s)";
    }

    private static float[] BlendTerrainWeakSignalMaskedChunkHeights(float[] sourceHeights, float[] restoredHeights, float[]? vertexWeights)
    {
        float[] blendedHeights = new float[sourceHeights.Length];
        for (int index = 0; index < sourceHeights.Length; index++)
        {
            float sourceHeight = sourceHeights[index];
            float targetHeight = index < restoredHeights.Length ? restoredHeights[index] : sourceHeight;
            float weight = vertexWeights != null && index < vertexWeights.Length ? Math.Clamp(vertexWeights[index], 0f, 1f) : 1f;
            blendedHeights[index] = sourceHeight + ((targetHeight - sourceHeight) * weight);
        }

        return blendedHeights;
    }

    private static int GetTerrainWeakSignalSelectedMaskHash(bool[] selectedMask)
    {
        var hash = new HashCode();
        for (int index = 0; index < selectedMask.Length; index++)
            hash.Add(selectedMask[index]);

        return hash.ToHashCode();
    }

    private static float[] BuildTerrainWeakSignalSubCellVertexWeights(bool[] selectedMask, int subDivisions)
    {
        float[] weights = new float[145];
        for (int vertexIndex = 0; vertexIndex < weights.Length; vertexIndex++)
        {
            GetChunkVertexLocalPosition(vertexIndex, out float localX, out float localY);
            int cellX = Math.Clamp((int)MathF.Floor(Math.Clamp(localX / WoWConstants.ChunkSize, 0f, 0.999f) * subDivisions), 0, subDivisions - 1);
            int cellY = Math.Clamp((int)MathF.Floor(Math.Clamp(localY / WoWConstants.ChunkSize, 0f, 0.999f) * subDivisions), 0, subDivisions - 1);
            weights[vertexIndex] = selectedMask[cellY * subDivisions + cellX] ? 1f : 0f;
        }

        return weights;
    }

    private static bool CellTouchesSelectedMask(bool[] mask, int size, int cellX, int cellY)
    {
        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                if ((uint)neighborX >= size || (uint)neighborY >= size)
                    continue;

                if (mask[neighborY * size + neighborX])
                    return true;
            }
        }

        return false;
    }

    private static bool TryGetTerrainWeakSignalShadowNeighborHeightRange(float[] shadowCoverage, float[] averageHeights, int size, int cellX, int cellY, out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                if (offsetX == 0 && offsetY == 0)
                    continue;

                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                if ((uint)neighborX >= size || (uint)neighborY >= size)
                    continue;

                int neighborIndex = neighborY * size + neighborX;
                if (shadowCoverage[neighborIndex] < TerrainWeakSignalShadowEdgeMinCoverage)
                    continue;

                averageHeight += averageHeights[neighborIndex];
                count++;
            }
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryGetTerrainWeakSignalDirectionalShadowNeighborAverageHeight(
        float[] shadowCoverage,
        float[] averageHeights,
        int size,
        int cellX,
        int cellY,
        int offsetX,
        int offsetY,
        out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        foreach (var (sampleOffsetX, sampleOffsetY) in EnumerateDirectionalOffsets(offsetX, offsetY))
        {
            int neighborX = cellX + sampleOffsetX;
            int neighborY = cellY + sampleOffsetY;
            if ((uint)neighborX >= size || (uint)neighborY >= size)
                continue;

            int neighborIndex = neighborY * size + neighborX;
            if (shadowCoverage[neighborIndex] < TerrainWeakSignalShadowEdgeMinCoverage)
                continue;

            averageHeight += averageHeights[neighborIndex];
            count++;
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryGetTerrainWeakSignalSelectedNeighborAverageHeight(bool[] selectedMask, float[] averageHeights, int size, int cellX, int cellY, out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        for (int offsetY = -1; offsetY <= 1; offsetY++)
        {
            for (int offsetX = -1; offsetX <= 1; offsetX++)
            {
                if (offsetX == 0 && offsetY == 0)
                    continue;

                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                if ((uint)neighborX >= size || (uint)neighborY >= size)
                    continue;

                int neighborIndex = neighborY * size + neighborX;
                if (!selectedMask[neighborIndex])
                    continue;

                averageHeight += averageHeights[neighborIndex];
                count++;
            }
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryGetTerrainWeakSignalDirectionalSelectedNeighborAverageHeight(
        bool[] selectedMask,
        float[] averageHeights,
        int size,
        int cellX,
        int cellY,
        int offsetX,
        int offsetY,
        out float averageHeight)
    {
        averageHeight = 0f;
        int count = 0;
        foreach (var (sampleOffsetX, sampleOffsetY) in EnumerateDirectionalOffsets(offsetX, offsetY))
        {
            int neighborX = cellX + sampleOffsetX;
            int neighborY = cellY + sampleOffsetY;
            if ((uint)neighborX >= size || (uint)neighborY >= size)
                continue;

            int neighborIndex = neighborY * size + neighborX;
            if (!selectedMask[neighborIndex])
                continue;

            averageHeight += averageHeights[neighborIndex];
            count++;
        }

        if (count == 0)
            return false;

        averageHeight /= count;
        return true;
    }

    private static bool TryInferTerrainWeakSignalShadowDirection(
        bool[]? seedMask,
        float[] shadowCoverage,
        float[] averageHeights,
        int[] dominantLayers,
        int size,
        int preferredLayer,
        out (int offsetX, int offsetY) litToShadowOffset)
    {
        litToShadowOffset = default;
        float bestScore = float.MinValue;
        int bestMatchCount = 0;

        foreach ((int offsetX, int offsetY) in EnumerateNeighborDirections())
        {
            float score = 0f;
            int matchCount = 0;

            for (int cellY = 0; cellY < size; cellY++)
            {
                for (int cellX = 0; cellX < size; cellX++)
                {
                    int cellIndex = cellY * size + cellX;
                    if (shadowCoverage[cellIndex] > TerrainWeakSignalShadowLitMaxCoverage)
                        continue;

                    if (preferredLayer >= 0 && dominantLayers[cellIndex] != preferredLayer)
                        continue;

                    bool relevantToSeed = seedMask == null
                        || seedMask[cellIndex]
                        || CellTouchesSelectedMask(seedMask, size, cellX, cellY);
                    if (!relevantToSeed)
                        continue;

                    if (!TryGetTerrainWeakSignalDirectionalShadowNeighborAverageHeight(
                        shadowCoverage,
                        averageHeights,
                        size,
                        cellX,
                        cellY,
                        offsetX,
                        offsetY,
                        out float shadowNeighborAverageHeight))
                    {
                        continue;
                    }

                    float heightDelta = averageHeights[cellIndex] - shadowNeighborAverageHeight;
                    if (heightDelta + TerrainWeakSignalShadowEdgeMinHeightDelta < 0f)
                        continue;

                    float localScore = 1f + Math.Max(heightDelta, 0f);
                    if (seedMask != null && seedMask[cellIndex])
                        localScore += 0.5f;

                    score += localScore;
                    matchCount++;
                }
            }

            if (matchCount == 0)
                continue;

            if (score > bestScore || (Math.Abs(score - bestScore) < 0.001f && matchCount > bestMatchCount))
            {
                bestScore = score;
                bestMatchCount = matchCount;
                litToShadowOffset = (offsetX, offsetY);
            }
        }

        return bestMatchCount > 0;
    }

    private static IEnumerable<(int offsetX, int offsetY)> EnumerateNeighborDirections()
    {
        yield return (-1, -1);
        yield return (0, -1);
        yield return (1, -1);
        yield return (-1, 0);
        yield return (1, 0);
        yield return (-1, 1);
        yield return (0, 1);
        yield return (1, 1);
    }

    private static IEnumerable<(int offsetX, int offsetY)> EnumerateDirectionalOffsets(int offsetX, int offsetY)
    {
        yield return (offsetX, offsetY);

        if (offsetX == 0)
        {
            yield return (-1, offsetY);
            yield return (1, offsetY);
            yield break;
        }

        if (offsetY == 0)
        {
            yield return (offsetX, -1);
            yield return (offsetX, 1);
            yield break;
        }

        yield return (offsetX, 0);
        yield return (0, offsetY);
    }

    private static float ComputeTerrainWeakSignalShadowCoverage(byte[] shadowMap, int cellX, int cellY, int subDivisions)
    {
        const int shadowSize = 64;
        int pixelsPerCell = shadowSize / subDivisions;
        int pixelStartX = cellX * pixelsPerCell;
        int pixelStartY = cellY * pixelsPerCell;
        float sum = 0f;
        int count = 0;

        for (int y = pixelStartY; y < pixelStartY + pixelsPerCell; y++)
        {
            for (int x = pixelStartX; x < pixelStartX + pixelsPerCell; x++)
            {
                int pixelIndex = y * shadowSize + x;
                if ((uint)pixelIndex >= shadowMap.Length)
                    continue;

                sum += shadowMap[pixelIndex] / 255f;
                count++;
            }
        }

        return count > 0 ? sum / count : 0f;
    }

    private static float ComputeTerrainWeakSignalAverageHeightForSubCell(Terrain.TerrainChunkData chunk, int cellX, int cellY, int subDivisions)
    {
        float cellSize = WoWConstants.ChunkSize / subDivisions;
        float averageHeight = 0f;
        int sampleCount = 0;
        for (int sampleY = 0; sampleY < 3; sampleY++)
        {
            for (int sampleX = 0; sampleX < 3; sampleX++)
            {
                float localX = cellX * cellSize + ((sampleX + 0.5f) / 3f) * cellSize;
                float localY = cellY * cellSize + ((sampleY + 0.5f) / 3f) * cellSize;
                averageHeight += SampleHeightOuterGrid(chunk, localX, localY);
                sampleCount++;
            }
        }

        return sampleCount > 0 ? averageHeight / sampleCount : 0f;
    }

    private static float[] BuildTerrainWeakSignalRestoredChunkHeights(
        Terrain.TerrainChunkData chunk,
        float factor,
        float[]? vertexWeights = null,
        float? globalMaxHeight = null)
    {
        float[] restoredHeights = new float[chunk.Heights.Length];
        float anchorHeight = 0f;
        bool preserveNegativeFloor = false;
        if (TryGetTerrainChunkHeightRange(chunk, out float chunkMinHeight, out _))
        {
            anchorHeight = chunkMinHeight < 0f ? chunkMinHeight : 0f;
            preserveNegativeFloor = anchorHeight < 0f;
        }

        for (int index = 0; index < chunk.Heights.Length; index++)
        {
            float sourceHeight = chunk.Heights[index];
            float amplifiedHeight = anchorHeight + ((sourceHeight - anchorHeight) * factor);
            float weight = vertexWeights != null && index < vertexWeights.Length ? Math.Clamp(vertexWeights[index], 0f, 1f) : 1f;
            float restoredHeight = sourceHeight + ((amplifiedHeight - sourceHeight) * weight);
            if (!preserveNegativeFloor && restoredHeight < 0f)
                restoredHeight = 0f;
            if (globalMaxHeight.HasValue && restoredHeight > globalMaxHeight.Value)
                restoredHeight = globalMaxHeight.Value;

            restoredHeights[index] = restoredHeight;
        }

        return restoredHeights;
    }
}
