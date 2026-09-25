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
    private void DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent) => _host.DrawToolbarPopupButton(label, summary, popupId, drawContent);
    private void ExportAnimationStateJson(IAnimationController animator, int currentSeq, string currentSeqName, float seqStart, float seqEnd) => _host.ExportAnimationStateJson(animator, currentSeq, currentSeqName, seqStart, seqEnd);
    private (int tileX, int tileY) GetCameraTile() => _host.GetCameraTile();

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

    internal void TryAutoPopulateAlphaCoreRoot()
    {
        if (!string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot))
            return;

        string[] candidates =
        {
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "external", "alpha-core")),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "external", "alpha-core")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "external", "alpha-core"))
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
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return;
        if (_worldScene.SelectedObjectType != Terrain.ObjectType.Mdx)
            return;
        if (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId)
            return;

        var inst = _worldScene.SelectedInstance.Value;
        var spawn = _sqlMapSpawnsCache.FirstOrDefault(s =>
            s.SpawnType == WorldSpawnType.GameObject &&
            s.SpawnId == inst.UniqueId &&
            (string.IsNullOrEmpty(s.ModelPath) || string.Equals(Path.GetFileName(s.ModelPath), inst.ModelName, StringComparison.OrdinalIgnoreCase)));
        if (spawn == null)
            return;

        var mdxRenderer = _worldScene.Assets.GetMdx(inst.ModelKey);
        var animator = mdxRenderer?.Animator;

        ImGui.Separator();
        ImGui.TextColored(new Vector4(0.85f, 1f, 0.85f, 1f), "SQL GameObject Animation");
        ImGui.TextDisabled($"SpawnId: {spawn.SpawnId}  Entry: {spawn.EntryId}  Type: {spawn.GameObjectType}");

        if (animator == null || !animator.HasAnimation || animator.Sequences.Count == 0)
        {
            ImGui.TextDisabled("This gameobject model has no animation sequences.");
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
        if (ImGui.Button(isPlaying ? "Pause GO Anim" : "Play GO Anim"))
            animator.IsPlaying = !isPlaying;

        ImGui.SameLine();
        if (ImGui.Button("Stop GO Anim"))
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
        if (clearSceneSpawns && _worldScene != null)
            _worldScene.ClearExternalSpawns();
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
            _sqlSpawnStatus = "Enter alpha-core root path first.";
            return;
        }

        try
        {
            if (_sqlPopulationService == null ||
                !string.Equals(_sqlServiceRoot, _sqlAlphaCoreRoot, StringComparison.OrdinalIgnoreCase))
            {
                _sqlPopulationService?.Dispose();
                _sqlPopulationService = new SqlWorldPopulationService(_sqlAlphaCoreRoot);
                _sqlServiceRoot = _sqlAlphaCoreRoot;
            }

            var (ok, message) = _sqlPopulationService.Validate();
            if (!ok)
            {
                _sqlSpawnStatus = message;
                return;
            }

            _sqlSpawnStatus = "Parsing SQL and building spawn list...";

            int requestedMax = (_sqlUseAoiFilter || _sqlStreamWithCamera) ? 0 : _sqlMaxSpawns;
            var mapSpawns = _sqlPopulationService
                .LoadMapSpawnsAsync(_currentMapId, requestedMax, _sqlIncludeCreatures, _sqlIncludeGameObjects)
                .GetAwaiter()
                .GetResult();

            _sqlMapSpawnsCache = mapSpawns.ToList();
            _sqlMapSpawnsCacheMapId = _currentMapId;
            _sqlLastCameraTile = null;
            _sqlForceStreamRefresh = true;

            ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: true);
        }
        catch (Exception ex)
        {
            _sqlSpawnStatus = $"Error: {ex.Message}";
        }
    }

    private void ApplySqlSpawnsToScene(IReadOnlyList<WorldSpawnRecord> mapSpawns, bool updateStatus)
    {
        if (_worldScene == null)
            return;

        _worldScene.SqlGameObjectMdxScaleMultiplier = _sqlGameObjectMdxScaleMultiplier;

        IReadOnlyList<WorldSpawnRecord> finalSpawns = mapSpawns;
        if (_sqlUseAoiFilter)
            finalSpawns = FilterSpawnsToCameraAoi(mapSpawns, _sqlAoiTileRadius, _sqlMaxSpawns);
        else if (_sqlMaxSpawns > 0 && mapSpawns.Count > _sqlMaxSpawns)
            finalSpawns = mapSpawns.Take(_sqlMaxSpawns).ToList();

        _worldScene.SetExternalSpawns(finalSpawns);

        if (updateStatus)
        {
            _sqlSpawnStatus = _sqlUseAoiFilter
                ? $"Loaded {finalSpawns.Count}/{mapSpawns.Count} SQL spawns for map {_currentMapId} (AOI radius {_sqlAoiTileRadius} tiles{(_sqlStreamWithCamera ? ", streaming" : "")})."
                : $"Loaded {finalSpawns.Count} SQL spawns for map {_currentMapId}.";
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
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return false;
        if (_worldScene.SelectedObjectType != Terrain.ObjectType.Mdx)
            return false;
        if (_sqlMapSpawnsCache == null || _sqlMapSpawnsCacheMapId != _currentMapId)
            return false;

        var inst = _worldScene.SelectedInstance.Value;
        return _sqlMapSpawnsCache.Any(s =>
            s.SpawnType == WorldSpawnType.GameObject &&
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

        ImGui.TextDisabled("Optional alpha-core SQL population. This is separate from ADT/WMO/MDX placement data.");
        ImGui.InputTextWithHint("##populationSqlRoot", "Path to alpha-core root (example: external/alpha-core)", ref _sqlAlphaCoreRoot, 1024);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("WoWViewer reads NPC/GameObject spawns from alpha-core SQL dumps (etc/databases/world + dbc).");

        DrawToolbarPopupButton("SQL Actions", string.Empty, "##PopulationSqlActionsPopup", () =>
        {
            if (ImGui.Button("Use Submodule Path"))
            {
                string candidate = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "external", "alpha-core"));
                _sqlAlphaCoreRoot = candidate;
                ImGui.CloseCurrentPopup();
            }

            bool canLoadSql = _currentMapId >= 0 && !string.IsNullOrWhiteSpace(_sqlAlphaCoreRoot);
            if (!canLoadSql)
                ImGui.BeginDisabled();
            if (ImGui.Button("Load SQL Spawns (Current Map)"))
            {
                LoadSqlSpawnsForCurrentMap();
                ImGui.CloseCurrentPopup();
            }
            if (!canLoadSql)
                ImGui.EndDisabled();

            if (ImGui.Button("Clear SQL Spawns"))
            {
                ResetSqlSpawnStreamingState(clearSceneSpawns: true);
                _sqlSpawnStatus = "Cleared SQL spawns.";
                ImGui.CloseCurrentPopup();
            }
        });

        bool settingsChanged = false;
        settingsChanged |= ImGui.Checkbox("NPC Spawns", ref _sqlIncludeCreatures);
        ImGui.SameLine();
        settingsChanged |= ImGui.Checkbox("GameObject Spawns", ref _sqlIncludeGameObjects);
        settingsChanged |= ImGui.Checkbox("AOI Tile Filter", ref _sqlUseAoiFilter);
        if (_sqlUseAoiFilter)
            settingsChanged |= ImGui.SliderInt("AOI Tile Radius", ref _sqlAoiTileRadius, 1, 16);
        settingsChanged |= ImGui.Checkbox("Stream With Camera", ref _sqlStreamWithCamera);
        settingsChanged |= ImGui.SliderInt("Max SQL Spawns", ref _sqlMaxSpawns, 100, 20000);
        settingsChanged |= ImGui.SliderFloat("GO MDX Scale", ref _sqlGameObjectMdxScaleMultiplier, 0.10f, 3.00f, "%.2fx");
        _worldScene.SqlGameObjectMdxScaleMultiplier = _sqlGameObjectMdxScaleMultiplier;
        if (settingsChanged && _sqlMapSpawnsCache != null)
        {
            _sqlForceStreamRefresh = true;
            if (!_sqlStreamWithCamera || !_sqlUseAoiFilter)
                ApplySqlSpawnsToScene(_sqlMapSpawnsCache, updateStatus: true);
        }

        ImGui.TextDisabled($"Status: {_sqlSpawnStatus}");
        ImGui.TextDisabled($"Injected: {_worldScene.ExternalSpawnInstanceCount} total ({_worldScene.ExternalSpawnMdxCount} MDX, {_worldScene.ExternalSpawnWmoCount} WMO)");
    }
}
