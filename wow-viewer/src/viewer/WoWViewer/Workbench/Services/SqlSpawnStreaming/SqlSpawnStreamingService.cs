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
using WowViewer.Core.Runtime.PromoVideo;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// SQL world population: alpha-core spawn loading, camera-AOI streaming into the scene, the population sub-tab and game-object animation controls.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class SqlSpawnStreamingService
{
    private readonly IViewerAppHost _host;

    internal SqlSpawnStreamingService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private ref int _currentMapId => ref _host.CurrentMapId;
    private ref bool _sqlForceStreamRefresh => ref _host.SqlForceStreamRefresh;
    private ref SqlWorldPopulationService? _sqlPopulationService => ref _host.SqlPopulationService;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private void DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent) => _host.DrawToolbarPopupButton(label, summary, popupId, drawContent);
    private void ExportAnimationStateJson(IAnimationController animator, int currentSeq, string currentSeqName, float seqStart, float seqEnd) => _host.ExportAnimationStateJson(animator, currentSeq, currentSeqName, seqStart, seqEnd);
    private (int tileX, int tileY) GetCameraTile() => _host.GetCameraTile();

    public enum PopulationDatasetMode
    {
        Auto = 0,
        Wrath335a = 1,
        Alpha053 = 2,
        Custom = 3
    }

    private PopulationDatasetMode _datasetMode = PopulationDatasetMode.Auto;
    private string _activeVersionDetected = "3.3.5a";
    private bool _sqlAutoLoadOnMapChange = false;
    private string _sqlAlphaCoreRoot = "";
    private bool _sqlIncludeCreatures = true;
    private bool _sqlIncludeGameObjects = true;
    private int _sqlMaxSpawns = 2000;
    private float _sqlGameObjectMdxScaleMultiplier = 1.0f;
    private bool _sqlUseAoiFilter = true;
    private int _sqlAoiTileRadius = 3;
    private bool _sqlStreamWithCamera = true;
    private string _sqlSpawnStatus = "Not loaded";
    private string _sqlServiceRoot = "";
    private List<WorldSpawnRecord>? _sqlMapSpawnsCache;
    private int _sqlMapSpawnsCacheMapId = -1;
    private (int tileX, int tileY)? _sqlLastCameraTile;
    private bool _isLoadingSpawns;
    private readonly HashSet<int> _lastAppliedSpawnIds = new();

    internal string DetectActiveClientVersion()
    {
        string dbcBuild = _dbcBuild ?? _host.GetCurrentCaptureBuildVersion() ?? "";
        string dsPath = _dataSource?.Name ?? "";

        if (dbcBuild.Contains("0.5.3", StringComparison.OrdinalIgnoreCase) ||
            dbcBuild.Contains("0.5.4", StringComparison.OrdinalIgnoreCase) ||
            dbcBuild.Contains("0.5.5", StringComparison.OrdinalIgnoreCase) ||
            dbcBuild.Contains("3368", StringComparison.OrdinalIgnoreCase) ||
            dbcBuild.Contains("alpha", StringComparison.OrdinalIgnoreCase) ||
            dsPath.Contains("alpha", StringComparison.OrdinalIgnoreCase) ||
            dsPath.Contains("0.5.3", StringComparison.OrdinalIgnoreCase))
        {
            return "0.5.3";
        }

        return "3.3.5a";
    }

    internal static string? ResolveDatasetPathForVersion(string version)
    {
        string[] candidates =
        {
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "data", "population", version)),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "data", "population", version)),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "population", version)),
            $@"I:\parp\parp-tools\wow-viewer\data\population\{version}"
        };

        foreach (var c in candidates)
        {
            if (Directory.Exists(c) && SqlWorldPopulationService.DetectIsJsonDataset(c))
                return c;
        }

        return null;
    }

    internal void EnsureDatasetRootConfigured()
    {
        if (_datasetMode == PopulationDatasetMode.Custom)
        {
            if (string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
                TryAutoPopulateAlphaCoreRoot();
            return;
        }

        string targetVersion = _datasetMode switch
        {
            PopulationDatasetMode.Alpha053 => "0.5.3",
            PopulationDatasetMode.Wrath335a => "3.3.5a",
            _ => DetectActiveClientVersion()
        };

        _activeVersionDetected = targetVersion;
        string? resolved = ResolveDatasetPathForVersion(targetVersion);
        if (resolved != null)
        {
            if (!string.Equals(_sqlAlphaCoreRoot, resolved, StringComparison.OrdinalIgnoreCase))
            {
                _sqlAlphaCoreRoot = resolved;
                _sqlSpawnStatus = $"Active population dataset: {targetVersion} ({resolved})";
            }
        }
        else if (string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
        {
            TryAutoPopulateAlphaCoreRoot();
        }
    }

    internal void TryAutoPopulateAlphaCoreRoot()
    {
        EnsureDatasetRootConfigured();
        if (!string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
            return;

        // Fallback to alpha-core SQL roots
        string[] candidates =
        {
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "external", "alpha-core")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "external", "alpha-core")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "external", "alpha-core")),
            @"I:\parp\parp-tools\wow-viewer\libs\alpha-core"
        };

        foreach (var candidate in candidates)
        {
            string worldDir = Path.Combine(candidate, "etc", "databases", "world");
            string dbcDir = Path.Combine(candidate, "etc", "databases", "dbc");
            if (Directory.Exists(worldDir) && Directory.Exists(dbcDir))
            {
                _sqlAlphaCoreRoot = candidate;
                _sqlSpawnStatus = $"Auto-detected alpha-core SQL root: {candidate}";
                return;
            }
        }
    }

    internal void DrawSelectedSqlGameObjectAnimationControls()
    {
        if (_worldScene == null || !_worldScene.Selection.SelectedInstance.HasValue)
            return;
        if (_worldScene.Selection.SelectedObjectType != Terrain.ObjectType.Mdx)
            return;
        if (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId)
            return;

        var inst = _worldScene.Selection.SelectedInstance.Value;
        var spawn = _sqlMapSpawnsCache.FirstOrDefault(s =>
            s.SpawnId == inst.UniqueId &&
            (string.IsNullOrEmpty(s.ModelPath) || string.Equals(Path.GetFileName(s.ModelPath), inst.ModelName, StringComparison.OrdinalIgnoreCase)));
        if (spawn == null)
            return;

        var mdxRenderer = _worldScene.Assets.GetMdx(inst.ModelKey);
        var animator = mdxRenderer?.Animator;

        ImGui.Separator();
        string headerTitle = spawn.SpawnType == WorldSpawnType.Creature
            ? $"SQL NPC: {spawn.Name}" + (!string.IsNullOrEmpty(spawn.Subname) ? $" <{spawn.Subname}>" : "")
            : $"SQL GameObject: {spawn.Name}";
        ImGui.TextColored(new Vector4(0.85f, 1f, 0.85f, 1f), headerTitle);
        string typeDetail = spawn.SpawnType == WorldSpawnType.Creature
            ? $"SpawnId: {spawn.SpawnId}  Entry: {spawn.EntryId}  Type: NPC"
            : $"SpawnId: {spawn.SpawnId}  Entry: {spawn.EntryId}  Type: {spawn.GameObjectType}";
        ImGui.TextDisabled(typeDetail);

        if (animator == null || !animator.HasAnimation || animator.Sequences.Count == 0)
        {
            ImGui.TextDisabled("This model has no animation sequences.");
            return;
        }

        int currentSeq = animator.CurrentSequence;
        string currentSeqName = currentSeq >= 0 && currentSeq < animator.Sequences.Count
            ? animator.Sequences[currentSeq].Name
            : "None";
        if (string.IsNullOrWhiteSpace(currentSeqName))
            currentSeqName = $"Sequence {currentSeq}";

        if (ImGui.BeginCombo("##sqlgo_anim_seq", currentSeqName))
        {
            for (int s = 0; s < animator.Sequences.Count; s++)
            {
                bool selected = s == currentSeq;
                string seqName = animator.Sequences[s].Name;
                if (string.IsNullOrWhiteSpace(seqName))
                    seqName = $"Sequence {s}";
                if (ImGui.Selectable(seqName, selected))
                    animator.SetSequence(s);
                if (selected) ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

var seq = animator.Sequences[animator.CurrentSequence];
        float seqStart = seq.Time.Start;
        float seqEnd = seq.Time.End;

        bool isPlaying = animator.IsPlaying;
        string animTag = spawn.SpawnType == WorldSpawnType.Creature ? "NPC" : "GO";
        if (ImGui.Button(isPlaying ? $"Pause {animTag} Anim" : $"Play {animTag} Anim"))
            animator.IsPlaying = !isPlaying;

        ImGui.SameLine();
        if (ImGui.Button($"Stop {animTag} Anim"))
        {
            animator.IsPlaying = false;
            animator.CurrentFrame = seqStart;
        }

        ImGui.SameLine();
        if (ImGui.Button("Prev Key"))
        {
            animator.IsPlaying = false;
            animator.StepToPrevKeyframe();
        }

        ImGui.SameLine();
        if (ImGui.Button("Next Key"))
        {
            animator.IsPlaying = false;
            animator.StepToNextKeyframe();
        }

        float currentFrame = Math.Clamp(animator.CurrentFrame, seqStart, seqEnd);
        if (ImGui.SliderFloat("GO Frame", ref currentFrame, seqStart, seqEnd, "%.0f"))
        {
            animator.IsPlaying = false;
            animator.CurrentFrame = currentFrame;
        }

        ImGui.SameLine();
        if (ImGui.Button("Export JSON##GO"))
            ExportAnimationStateJson(animator, currentSeq, currentSeqName, seqStart, seqEnd);

        ImGui.TextDisabled("Note: this affects all visible instances using the same MDX model renderer.");
    }

    internal void UpdateSqlSpawnStreaming()
    {
        if (_worldScene == null || !_sqlStreamWithCamera || !_sqlUseAoiFilter)
            return;

        if (_currentMapId < 0)
            return;

        // Auto-stream population when loading any map or switching maps
        if (_sqlAutoLoadOnMapChange && (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId))
        {
            EnsureDatasetRootConfigured();
            if (!string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
            {
                LoadSqlSpawnsForCurrentMap();
            }
        }

        if (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId)
            return;

        var camTile = GetCameraTile();
        if (_sqlForceStreamRefresh || _sqlLastCameraTile == null || _sqlLastCameraTile.Value != camTile)
        {
            _sqlLastCameraTile = camTile;
            ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: false);
            _sqlForceStreamRefresh = false;
        }
    }

    internal void ResetSqlSpawnStreamingState(bool clearSceneSpawns)
    {
        _sqlMapSpawnsCache = null;
        _sqlMapSpawnsCacheMapId = -1;
        _sqlLastCameraTile = null;
        _sqlForceStreamRefresh = false;
        _lastAppliedSpawnIds.Clear();
        if (clearSceneSpawns && _worldScene != null)
            _worldScene.ExternalSpawns.ClearExternalSpawns();
    }

    private void LoadSqlSpawnsForCurrentMap()
    {
        if (_worldScene == null)
        {
            _sqlSpawnStatus = "No world loaded.";
            return;
        }

        if (_currentMapId < 0)
        {
            _sqlSpawnStatus = "Current map ID unavailable.";
            return;
        }

        if (string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
        {
            _sqlSpawnStatus = "Enter population dataset root path first.";
            return;
        }

        if (_isLoadingSpawns)
            return;

        int targetMapId = _currentMapId;
        string root = _sqlAlphaCoreRoot;
        int requestedMax = (_sqlUseAoiFilter || _sqlStreamWithCamera) ? 0 : _sqlMaxSpawns;
        bool incCreatures = _sqlIncludeCreatures;
        bool incGos = _sqlIncludeGameObjects;

        _isLoadingSpawns = true;
        _sqlSpawnStatus = $"Loading population partition for map {targetMapId} in background...";

        Task.Run(async () =>
        {
            try
            {
                if (_sqlPopulationService == null ||
                    !string.Equals(_sqlServiceRoot, root, StringComparison.OrdinalIgnoreCase))
                {
                    _sqlPopulationService?.Dispose();
                    _sqlPopulationService = new SqlWorldPopulationService(root);
                    _sqlServiceRoot = root;
                }

                var (ok, message) = _sqlPopulationService.Validate();
                if (!ok)
                {
                    _sqlSpawnStatus = message;
                    _isLoadingSpawns = false;
                    return;
                }

                var mapSpawns = await _sqlPopulationService
                    .LoadMapSpawnsAsync(targetMapId, requestedMax, incCreatures, incGos)
                    .ConfigureAwait(false);

                var spawnList = mapSpawns.ToList();

                // If active client is 0.5.3 Alpha, normalize any .m2 model paths to .mdx directly
                string clientVer = DetectActiveClientVersion();
                if (string.Equals(clientVer, "0.5.3", StringComparison.OrdinalIgnoreCase))
                {
                    for (int i = 0; i < spawnList.Count; i++)
                    {
                        var sp = spawnList[i];
                        if (!string.IsNullOrEmpty(sp.ModelPath) && sp.ModelPath.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                        {
                            sp.ModelPath = sp.ModelPath[..^3] + ".mdx";
                        }
                    }
                }

                _sqlMapSpawnsCache = spawnList;
                _sqlMapSpawnsCacheMapId = targetMapId;
                _sqlLastCameraTile = null;
                _sqlForceStreamRefresh = true;
                _isLoadingSpawns = false;
            }
            catch (Exception ex)
            {
                _sqlSpawnStatus = $"Error: {ex.Message}";
                _sqlMapSpawnsCache = new List<WorldSpawnRecord>();
                _sqlMapSpawnsCacheMapId = targetMapId;
                _isLoadingSpawns = false;
            }
        });
    }

    private void ApplySqlSpawnsToScene(IReadOnlyList<WorldSpawnRecord> mapSpawns, bool updateStatus)
    {
        if (_worldScene == null)
            return;

        _worldScene.ExternalSpawns.SqlGameObjectMdxScaleMultiplier = _sqlGameObjectMdxScaleMultiplier;

        IReadOnlyList<WorldSpawnRecord> finalSpawns = mapSpawns;
        if (_sqlUseAoiFilter)
            finalSpawns = FilterSpawnsToCameraAoi(mapSpawns, _sqlAoiTileRadius, _sqlMaxSpawns);
        else if (_sqlMaxSpawns > 0 && mapSpawns.Count > _sqlMaxSpawns)
            finalSpawns = mapSpawns.Take(_sqlMaxSpawns).ToList();

        // Avoid re-dirtying scene graph and instance lists if spawn set has not changed
        bool changed = finalSpawns.Count != _lastAppliedSpawnIds.Count;
        if (!changed)
        {
            for (int i = 0; i < finalSpawns.Count; i++)
            {
                if (!_lastAppliedSpawnIds.Contains(finalSpawns[i].SpawnId))
                {
                    changed = true;
                    break;
                }
            }
        }

        if (changed)
        {
            _lastAppliedSpawnIds.Clear();
            for (int i = 0; i < finalSpawns.Count; i++)
                _lastAppliedSpawnIds.Add(finalSpawns[i].SpawnId);

            _worldScene.ExternalSpawns.SetExternalSpawns(finalSpawns);
        }

        if (updateStatus)
        {
            _sqlSpawnStatus = _sqlUseAoiFilter
                ? $"Loaded {finalSpawns.Count}/{mapSpawns.Count} spawns for map {_currentMapId} (AOI radius {_sqlAoiTileRadius} tiles{(_sqlStreamWithCamera ? ", streaming" : "")})."
                : $"Loaded {finalSpawns.Count} spawns for map {_currentMapId}.";
        }
    }

    private List<WorldSpawnRecord> FilterSpawnsToCameraAoi(IReadOnlyList<WorldSpawnRecord> spawns, int tileRadius, int maxCount)
    {
        if (spawns.Count == 0) return new List<WorldSpawnRecord>();

        float camTileX = (WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize;
        float camTileY = (WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize;

        var inRange = new List<(WorldSpawnRecord spawn, float distSq)>();
        foreach (var spawn in spawns)
        {
            var pos = SqlSpawnCoordinateConverter.ToRendererPosition(spawn.PositionWow);
            float spawnTileX = (WoWConstants.MapOrigin - pos.X) / WoWConstants.ChunkSize;
            float spawnTileY = (WoWConstants.MapOrigin - pos.Y) / WoWConstants.ChunkSize;

            if (MathF.Abs(spawnTileX - camTileX) > tileRadius || MathF.Abs(spawnTileY - camTileY) > tileRadius)
                continue;

            float dx = pos.X - _camera.Position.X;
            float dy = pos.Y - _camera.Position.Y;
            float dz = pos.Z - _camera.Position.Z;
            inRange.Add((spawn, dx * dx + dy * dy + dz * dz));
        }

        inRange.Sort((a, b) => a.distSq.CompareTo(b.distSq));

        int take = maxCount > 0 ? Math.Min(maxCount, inRange.Count) : inRange.Count;
        var result = new List<WorldSpawnRecord>(take);
        for (int i = 0; i < take; i++)
            result.Add(inRange[i].spawn);

        return result;
    }

    internal bool HasSqlGameObjectForSelectedInstance()
    {
        if (_worldScene == null || !_worldScene.Selection.SelectedInstance.HasValue)
            return false;
        if (_worldScene.Selection.SelectedObjectType != Terrain.ObjectType.Mdx)
            return false;
        if (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId)
            return false;

        var inst = _worldScene.Selection.SelectedInstance.Value;
        return _sqlMapSpawnsCache.Any(s =>
            s.SpawnId == inst.UniqueId &&
            (string.IsNullOrEmpty(s.ModelPath) || string.Equals(Path.GetFileName(s.ModelPath), inst.ModelName, StringComparison.OrdinalIgnoreCase)));
    }

    internal void DrawPopulationSubTabContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to use SQL Population.");
            return;
        }

        ImGui.TextDisabled("Authentic SQL world population (NPCs & Game Objects).");

        // Version / Dataset selector
        string currentModeLabel = _datasetMode switch
        {
            PopulationDatasetMode.Auto => $"Auto (Detected: {_activeVersionDetected})",
            PopulationDatasetMode.Wrath335a => "3.3.5a (TrinityCore Full World)",
            PopulationDatasetMode.Alpha053 => "0.5.3 (Alpha-Core)",
            PopulationDatasetMode.Custom => "Custom Path",
            _ => "Auto"
        };

        if (ImGui.BeginCombo("Dataset Version", currentModeLabel))
        {
            if (ImGui.Selectable($"Auto (Detect Active Client: {_activeVersionDetected})", _datasetMode == PopulationDatasetMode.Auto))
            {
                _datasetMode = PopulationDatasetMode.Auto;
                EnsureDatasetRootConfigured();
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                LoadSqlSpawnsForCurrentMap();
            }
            if (ImGui.Selectable("3.3.5a (TrinityCore Full World - 113 Maps, 246k Spawns)", _datasetMode == PopulationDatasetMode.Wrath335a))
            {
                _datasetMode = PopulationDatasetMode.Wrath335a;
                EnsureDatasetRootConfigured();
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                LoadSqlSpawnsForCurrentMap();
            }
            if (ImGui.Selectable("0.5.3 (Alpha-Core - 31 Maps, 92k Spawns)", _datasetMode == PopulationDatasetMode.Alpha053))
            {
                _datasetMode = PopulationDatasetMode.Alpha053;
                EnsureDatasetRootConfigured();
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                LoadSqlSpawnsForCurrentMap();
            }
            if (ImGui.Selectable("Custom Path / Alpha-Core SQL", _datasetMode == PopulationDatasetMode.Custom))
            {
                _datasetMode = PopulationDatasetMode.Custom;
            }
            ImGui.EndCombo();
        }

        if (_datasetMode == PopulationDatasetMode.Custom)
        {
            ImGui.InputTextWithHint("##populationSqlRoot", "Path to dataset or alpha-core root", ref _sqlAlphaCoreRoot, 1024);
        }
        else
        {
            ImGui.TextDisabled($"Active Dataset Root: {_sqlAlphaCoreRoot}");
        }

        ImGui.Checkbox("Auto-Stream Spawns On Map Load", ref _sqlAutoLoadOnMapChange);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Automatically loads and streams population data whenever a map is loaded or switched.");

        DrawToolbarPopupButton("SQL Actions", string.Empty, "##PopulationSqlActionsPopup", () =>
        {
            if (ImGui.Button("Reload Spawns (Current Map)"))
            {
                _sqlAutoLoadOnMapChange = true;
                EnsureDatasetRootConfigured();
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                LoadSqlSpawnsForCurrentMap();
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button("Use 3.3.5a Dataset"))
            {
                _datasetMode = PopulationDatasetMode.Wrath335a;
                EnsureDatasetRootConfigured();
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                LoadSqlSpawnsForCurrentMap();
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button("Use 0.5.3 Alpha Dataset"))
            {
                _datasetMode = PopulationDatasetMode.Alpha053;
                EnsureDatasetRootConfigured();
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                LoadSqlSpawnsForCurrentMap();
                ImGui.CloseCurrentPopup();
            }

            if (ImGui.Button("Clear SQL Spawns"))
            {
                _sqlAutoLoadOnMapChange = false;
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                _sqlSpawnStatus = "Cleared SQL spawns. Auto-load paused until next map.";
                ImGui.CloseCurrentPopup();
            }
        });

        bool settingsChanged = false;
        if (ImGui.Checkbox("NPC Spawns", ref _sqlIncludeCreatures))
        {
            ResetSqlSpawnStreamingState(clearSceneSpawns: true);
            LoadSqlSpawnsForCurrentMap();
        }
        ImGui.SameLine();
        if (ImGui.Checkbox("GameObject Spawns", ref _sqlIncludeGameObjects))
        {
            ResetSqlSpawnStreamingState(clearSceneSpawns: true);
            LoadSqlSpawnsForCurrentMap();
        }
        settingsChanged |= ImGui.Checkbox("AOI Tile Filter", ref _sqlUseAoiFilter);
        if (_sqlUseAoiFilter)
            settingsChanged |= ImGui.SliderInt("AOI Tile Radius", ref _sqlAoiTileRadius, 1, 16);
        settingsChanged |= ImGui.Checkbox("Stream With Camera", ref _sqlStreamWithCamera);
        settingsChanged |= ImGui.SliderInt("Max SQL Spawns", ref _sqlMaxSpawns, 100, 20000);
        settingsChanged |= ImGui.SliderFloat("GO MDX Scale", ref _sqlGameObjectMdxScaleMultiplier, 0.10f, 3.00f, "%.2fx");
        _worldScene.ExternalSpawns.SqlGameObjectMdxScaleMultiplier = _sqlGameObjectMdxScaleMultiplier;
        if (settingsChanged && _sqlMapSpawnsCache != null)
        {
            _sqlForceStreamRefresh = true;
            if (!_sqlStreamWithCamera || !_sqlUseAoiFilter)
                ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: true);
        }

        ImGui.TextDisabled($"Status: {_sqlSpawnStatus}");
        ImGui.TextDisabled($"Injected: {_worldScene.ExternalSpawns.ExternalSpawnInstanceCount} total ({_worldScene.ExternalSpawns.ExternalSpawnMdxCount} MDX, {_worldScene.ExternalSpawns.ExternalSpawnWmoCount} WMO)");
    }
}
