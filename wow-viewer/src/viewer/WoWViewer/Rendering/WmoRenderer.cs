using System.Diagnostics;
using System.Numerics;
using System.Text;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using Silk.NET.OpenGL;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.Wmo;
using WowViewer.Core.IO.Converters;

namespace WoWViewer.Rendering;

public readonly record struct WmoDoodadInfo(
    int Index,
    string ModelPath,
    int DoodadDefIndex,
    Vector3 LocalPosition,
    bool Visible,
    bool IsLoaded);

public readonly record struct WmoOpaqueDoodadBatchItem(
    IModelRenderer Renderer,
    Matrix4x4 ModelMatrix);

public enum WmoRenderPass
{
    Both,
    Opaque,
    Transparent,
}

public readonly record struct WmoRenderStats(
    int DrawCalls,
    int BatchDrawCalls,
    int OpaqueBatchInstanceCount,
    int GroupFallbackDrawCalls,
    int LiquidDrawCalls,
    int DoodadSubmissions,
    int VisibleGroupSubmissions,
    int VisibleLiquidMeshes,
    int PortalTestedCount,
    int PortalFallbackCount,
    int PortalAdmittedGroupCount);

/// <summary>
/// Renders a WMO (World Map Object) using OpenGL.
/// Uses WowViewer.Core.IO.Converters' WmoV14Data model for geometry.
/// Supports loading and rendering MDX doodads from DoodadSets.
/// </summary>
public class WmoRenderer : ISceneRenderer, IGpuInstancedWmoRenderer
{
    private readonly GL _gl;
    private readonly WmoV14ToV17Converter.WmoV14Data _wmo;
    private readonly string _modelDir;
    private readonly IDataSource? _dataSource;
    private readonly ReplaceableTextureResolver? _texResolver;
    private readonly string? _buildVersion;
    private readonly bool _deferInitialDoodadLoads;
    private readonly bool _deferInitialMaterialTextureLoads;
    private bool _enableRuntimeGroupVisibility;

    // Shared static shader program — prevents race condition when multiple WmoRenderers
    // exist and one is disposed (same fix as MdxRenderer)
    private static uint _shaderProgram;
    private static int _uModel, _uView, _uProj, _uHasTexture, _uColor, _uAlphaTest;
    private static int _uFogColor, _uFogStart, _uFogEnd, _uCameraPos;
    private static int _uLightDir, _uLightColor, _uAmbientColor;
    private static int _uUseInstanceModel;
    private static int _shaderRefCount;
    private uint _gpuInstanceVbo;
    private readonly List<Matrix4x4> _gpuInstanceMatrices = new();
    private float[] _gpuInstanceUploadScratch = Array.Empty<float>();
    private bool _gpuInstanceBatchActive;

    private readonly List<GroupBuffers> _groups = new();
    private readonly List<(int groupBufferIndex, float distSq)> _transparentGroupSortScratch = new();
    private readonly FrustumCuller _groupFrustumCuller = new();
    private readonly WmoPortalVisibilityGroup[] _portalVisibilityGroups;
    private readonly WmoPortalVisibilityPortal[] _portalVisibilityPortals;
    private readonly bool[] _runtimeVisibleGroups;
    private readonly bool[] _frustumVisibleScratch;
    private readonly bool[] _portalVisibleScratch;
    private WmoAdmissionTally _groupAdmission;
    private readonly HashSet<int> _runtimeVisibleDoodadDefIndices = new();
    private readonly HashSet<string> _updatedDoodadModelsScratch = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<(int idx, float distSq)> _visibleDoodadsScratch = new();
    private readonly Dictionary<IModelRenderer, List<int>> _opaqueDoodadBatchGroups = new();
    private readonly List<IModelRenderer> _opaqueDoodadBatchRenderers = new();
    private bool _doodadAnimationsPreparedForWorldFrame;
    private bool _wireframe;

    // Material textures: materialIndex → GL texture handle
    private readonly Dictionary<int, uint> _materialTextures = new();
    private readonly HashSet<string> _materialFallbackLogKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _invalidBatchRangeLogKeys = new(StringComparer.OrdinalIgnoreCase);
    private int _materialFallbackLogCount;
    private int _invalidBatchRangeLogCount;
    private const int MaxMaterialFallbackLogs = 200;
    private const int MaxInvalidBatchRangeLogs = 100;
    private readonly Queue<int> _pendingMaterialTextureLoads = new();
    private const int DeferredMaterialTextureLoadsPerFrame = 1;
    private const double DeferredMaterialTextureLoadBudgetMs = 2.0;

// Doodad support
    private readonly Dictionary<string, IModelRenderer?> _doodadModelCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, M2RouteDecision?> _doodadRouteDecisions = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<DoodadInstance> _doodadInstances = new();
    private readonly List<string> _doodadNames = new(); // resolved from MODN
    private readonly Dictionary<string, string> _canonicalDoodadPathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string?> _bestSkinPathCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _loggedMissingDoodadSkinPaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly Queue<string> _pendingDoodadModelLoads = new();
    private readonly HashSet<string> _queuedDoodadModelLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<int>> _doodadInstanceIndicesByModel = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _doodadSourceModelPaths = new(StringComparer.OrdinalIgnoreCase);
    private int _activeDoodadSet = 0;
    private bool _doodadsVisible = true;
    private bool _runtimeDoodadsVisible = true;
    private bool _runtimeGroupLiquidsVisible = true;
    private int _currentDrawCalls;
    private int _currentBatchDrawCalls;
    private int _currentOpaqueBatchInstanceCount;
    private int _currentGroupFallbackDrawCalls;
    private int _currentLiquidDrawCalls;
    private int _currentDoodadSubmissions;
    private int _currentVisibleGroupSubmissions;
    private int _currentVisibleLiquidMeshes;
    private const int DefaultDeferredDoodadLoads = 1;
    private const double DefaultDeferredDoodadBudgetMs = 2.0;
    private const int ExteriorPortalTraversalDepth = 1;
    private const int InteriorPortalTraversalDepth = 4;

    // Doodad culling constants
    private const float DoodadCullDistance = 4000f;  // Minimum distance from camera to render WMO doodads; expanded further by fog range at runtime
    private const float DoodadMaxRenderCount = 1024; // Soft cap to avoid large WMO doodad sets dropping out too early

    public int PendingDoodadModelLoadCount => _pendingDoodadModelLoads.Count;
    public int PendingMaterialTextureLoadCount => _pendingMaterialTextureLoads.Count;
    public WmoRenderStats LastRenderStats { get; private set; }
    public WmoPortalVisibilityDiagnostics LastPortalVisibilityDiagnostics { get; private set; } = new();

    /// <summary>
    /// Group admission accounting for the most recent submission, recording which rule admitted each
    /// group rather than only how many were admitted. Spec 151 instrumentation; read it before
    /// changing any admission rule.
    /// </summary>
    public WmoAdmissionTally LastGroupAdmission => _groupAdmission;

    /// <summary>
    /// Opaque WMO shells can be instanced when portal visibility cannot distinguish individual
    /// placements. Group-level frustum admission is intentionally traded for a conservative
    /// object-level batch; transparent/liquid/doodad work remains placement-aware.
    /// </summary>
    public bool SupportsGpuInstancedOpaque
        => _groups.Count > 0
            && _wmo.Portals.Count == 0
            && _groups.All(static group => group.ManualVisible);

    public M2RouteDecision? GetDoodadRouteDecision(string normalizedPath)
        => _doodadRouteDecisions.TryGetValue(normalizedPath, out var decision) ? decision : null;

    // WMO liquid meshes (from MLIQ chunks in groups)
    private readonly List<LiquidMeshData> _liquidMeshes = new();
    private static uint _liquidShader;
    private static int _uLiqModel, _uLiqView, _uLiqProj, _uLiqColor;
    private static int _liquidShaderRefCount;
    // Additional user-configurable quarter-turns applied after the shared WMO family baseline.
    private static int _mliqRotationQuarterTurns;
    private static int _mliqRotationRevision;
    private int _builtMliqRotationRevision = -1;

    public static int MliqRotationQuarterTurns
    {
        get => _mliqRotationQuarterTurns;
        set
        {
            int normalized = ((value % 4) + 4) % 4;
            if (_mliqRotationQuarterTurns == normalized)
                return;

            _mliqRotationQuarterTurns = normalized;
            _mliqRotationRevision++;
            ViewerLog.Important(ViewerLog.Category.Wmo,
                $"[WmoRenderer] MLIQ additional rotation override set to {normalized * 90}°");
        }
    }

    public WmoRenderer(GL gl, WmoV14ToV17Converter.WmoV14Data wmo, string modelDir,
        IDataSource? dataSource = null, ReplaceableTextureResolver? texResolver = null, string? buildVersion = null,
        bool deferInitialDoodadLoads = false, bool deferInitialMaterialTextureLoads = false,
        bool enableRuntimeGroupVisibility = true)
    {
        var initStopwatch = Stopwatch.StartNew();
        _gl = gl;
        _wmo = wmo;
        _modelDir = modelDir;
        _dataSource = dataSource;
        _texResolver = texResolver;
        _buildVersion = buildVersion;
        _deferInitialDoodadLoads = deferInitialDoodadLoads;
        _deferInitialMaterialTextureLoads = deferInitialMaterialTextureLoads;
        _enableRuntimeGroupVisibility = enableRuntimeGroupVisibility;

        _portalVisibilityGroups = BuildPortalVisibilityGroups();
        _portalVisibilityPortals = BuildPortalVisibilityPortals();

        _runtimeVisibleGroups = new bool[_wmo.Groups.Count];
        _frustumVisibleScratch = new bool[_wmo.Groups.Count];
        _portalVisibleScratch = new bool[_wmo.Groups.Count];

        InitShaders();
        InitLiquidShader();
        InitBuffers();
        BuildLiquidMeshes();
        if (_deferInitialMaterialTextureLoads)
            QueueDeferredMaterialTextureLoads();
        else
            LoadMaterialTextures();
        ResolveDoodadNames();
        LoadActiveDoodadSet();

        if (initStopwatch.Elapsed.TotalMilliseconds >= 50)
        {
            ViewerLog.Info(
                ViewerLog.Category.Wmo,
                $"[WMO-LOAD] {modelDir}: init {initStopwatch.Elapsed.TotalMilliseconds:F1} ms (groups={_wmo.Groups.Count}, materials={_wmo.Materials.Count}, doodadDefs={_wmo.DoodadDefs.Count}, deferredMaterials={_deferInitialMaterialTextureLoads}, deferredDoodads={_deferInitialDoodadLoads})");
        }
    }

    /// <summary>MOHD bounding box min in WMO local space.</summary>
    public Vector3 BoundsMin => _wmo.BoundsMin;
    /// <summary>MOHD bounding box max in WMO local space.</summary>
    public Vector3 BoundsMax => _wmo.BoundsMax;
    public int GroupRenderCount => _groups.Count;

    /// <summary>
    /// Exposes the already-loaded WMO portal read model for the opt-in scene-graph bridge.
    /// This does not change the renderer's existing portal visibility path.
    /// </summary>
    public IReadOnlyList<WorldSceneWmoPortalGroupReadModel> GetSceneGraphPortalGroups()
        => _wmo.Groups
            .Select((_, groupIndex) => new WorldSceneWmoPortalGroupReadModel(groupIndex))
            .ToArray();

    /// <summary>
    /// Converts the existing renderer-owned WMO portal data to the graph adapter contract.
    /// Invalid vertex ranges are represented as missing geometry so the adapter can fail open.
    /// </summary>
    public IReadOnlyList<WorldSceneWmoPortalReadModel> GetSceneGraphPortalReadModels()
    {
        List<WorldSceneWmoPortalReadModel> portals = new(_wmo.Portals.Count);
        for (int portalIndex = 0; portalIndex < _wmo.Portals.Count; portalIndex++)
        {
            WmoV14ToV17Converter.WmoPortal portal = _wmo.Portals[portalIndex];
            IReadOnlyList<Vector3>? vertices = null;
            int startVertex = portal.StartVertex;
            int vertexCount = portal.Count;
            if (vertexCount >= 3 && startVertex <= _wmo.PortalVertices.Count - vertexCount)
            {
                vertices = _wmo.PortalVertices
                    .Skip(startVertex)
                    .Take(vertexCount)
                    .ToArray();
            }

            portals.Add(new WorldSceneWmoPortalReadModel(
                portalIndex,
                vertices,
                new Vector3(portal.PlaneA, portal.PlaneB, portal.PlaneC),
                portal.PlaneD,
                _wmo.PortalRefs
                    .Select((reference, referenceIndex) => (reference, referenceIndex))
                    .Where(item => item.reference.PortalIndex == portalIndex)
                    .Select(item => new WorldSceneWmoPortalReferenceReadModel(
                        item.referenceIndex,
                        item.reference.PortalIndex,
                        item.reference.GroupIndex,
                        item.reference.Side))
                    .ToArray()));
        }

        return portals;
    }

    // Sub-object visibility: WMO groups + doodad toggle
    // Layout: [0..N-1] = WMO groups, [N] = "Doodads" toggle, [N+1..] = individual doodad models
    public int SubObjectCount => _groups.Count + 1 + _doodadInstances.Count;

    public int GetRenderGroupId(int renderGroupIndex)
        => renderGroupIndex >= 0 && renderGroupIndex < _groups.Count ? _groups[renderGroupIndex].GroupIndex : -1;

    public string GetRenderGroupName(int renderGroupIndex)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
            return string.Empty;

        int groupIndex = _groups[renderGroupIndex].GroupIndex;
        string name = (groupIndex < _wmo.Groups.Count ? _wmo.Groups[groupIndex].Name : null) ?? $"Group {groupIndex}";
        return $"[{groupIndex}] {name}";
    }

    public bool GetRenderGroupManualVisible(int renderGroupIndex)
        => renderGroupIndex >= 0 && renderGroupIndex < _groups.Count && _groups[renderGroupIndex].ManualVisible;

    public bool GetRenderGroupRuntimeVisible(int renderGroupIndex)
        => renderGroupIndex >= 0 && renderGroupIndex < _groups.Count && _groups[renderGroupIndex].RuntimeVisible;

    public bool GetRenderGroupEffectiveVisible(int renderGroupIndex)
        => renderGroupIndex >= 0 && renderGroupIndex < _groups.Count && _groups[renderGroupIndex].IsVisible;

    public void SetRenderGroupVisible(int renderGroupIndex, bool visible)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
            return;

        _groups[renderGroupIndex].ManualVisible = visible;
    }

    public void SetAllRenderGroupsVisible(bool visible)
    {
        for (int i = 0; i < _groups.Count; i++)
            _groups[i].ManualVisible = visible;
    }

    public void IsolateRenderGroup(int renderGroupIndex)
    {
        for (int i = 0; i < _groups.Count; i++)
            _groups[i].ManualVisible = i == renderGroupIndex;
    }

    public void GetRenderGroupBounds(int renderGroupIndex, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
        {
            boundsMin = boundsMax = Vector3.Zero;
            return;
        }

        var group = _wmo.Groups[_groups[renderGroupIndex].GroupIndex];
        boundsMin = group.BoundsMin;
        boundsMax = group.BoundsMax;
    }

    public Vector3 GetRenderGroupCenter(int renderGroupIndex)
        => renderGroupIndex >= 0 && renderGroupIndex < _groups.Count
            ? _groups[renderGroupIndex].GroupCenter
            : Vector3.Zero;

    public Vector3 GetRenderGroupDebugColor(int renderGroupIndex)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
            return new Vector3(0.8f, 0.8f, 0.8f);

        int groupIndex = _groups[renderGroupIndex].GroupIndex;
        return new Vector3(
            ((groupIndex * 67 + 13) % 255) / 255f,
            ((groupIndex * 131 + 7) % 255) / 255f,
            ((groupIndex * 43 + 29) % 255) / 255f);
    }

    public string GetSubObjectName(int index)
    {
        if (index < _groups.Count)
        {
            int gi = _groups[index].GroupIndex;
            string name = (gi < _wmo.Groups.Count ? _wmo.Groups[gi].Name : null) ?? $"Group {gi}";
            return $"[{gi}] {name}";
        }
        if (index == _groups.Count)
            return $"--- Doodads ({_doodadInstances.Count}) ---";
        int di = index - _groups.Count - 1;
        if (di < _doodadInstances.Count)
        {
            var inst = _doodadInstances[di];
            return $"  Doodad: {Path.GetFileNameWithoutExtension(inst.ModelPath)}";
        }
        return "";
    }

    public bool GetSubObjectVisible(int index)
    {
        if (index < _groups.Count)
            return _groups[index].ManualVisible;
        if (index == _groups.Count)
            return _doodadsVisible;
        int di = index - _groups.Count - 1;
        if (di < _doodadInstances.Count)
            return _doodadInstances[di].Visible;
        return false;
    }

    public void SetSubObjectVisible(int index, bool visible)
    {
        if (index < _groups.Count)
            _groups[index].ManualVisible = visible;
        else if (index == _groups.Count)
            _doodadsVisible = visible;
        else
        {
            int di = index - _groups.Count - 1;
            if (di < _doodadInstances.Count)
                _doodadInstances[di].Visible = visible;
        }
    }

    // DoodadSet management
    public int DoodadSetCount => _wmo.DoodadSets.Count;
    public int ActiveDoodadSet => _activeDoodadSet;
    public int DoodadInstanceCount => _doodadInstances.Count;
    public int DoodadDefCount => _wmo.DoodadDefs.Count;
    public string GetDoodadSetName(int index) =>
        index < _wmo.DoodadSets.Count ? (_wmo.DoodadSets[index].Name ?? $"Set {index}") : "";

    public bool TryGetDoodadInfo(int index, out WmoDoodadInfo info)
    {
        if (index >= 0 && index < _doodadInstances.Count)
        {
            DoodadInstance doodad = _doodadInstances[index];
            info = new WmoDoodadInfo(
                index,
                doodad.ModelPath,
                doodad.DoodadDefIndex,
                doodad.LocalPosition,
                doodad.Visible,
                doodad.Renderer != null);
            return true;
        }

        info = default;
        return false;
    }

    public bool TryGetDoodadBounds(int index, in Matrix4x4 modelMatrix, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (index >= 0 && index < _doodadInstances.Count)
        {
            DoodadInstance doodad = _doodadInstances[index];
            Matrix4x4 doodadWorld = doodad.Transform * modelMatrix;
            if (doodad.Renderer is IModelRenderer modelRenderer)
            {
                TransformAabb(modelRenderer.BoundsMin, modelRenderer.BoundsMax, doodadWorld, out boundsMin, out boundsMax);
                return true;
            }

            Vector3 worldPosition = Vector3.Transform(doodad.LocalPosition, modelMatrix);
            boundsMin = worldPosition - new Vector3(2f);
            boundsMax = worldPosition + new Vector3(2f);
            return true;
        }

        boundsMin = boundsMax = Vector3.Zero;
        return false;
    }

    public bool TryGetDoodadDef(int doodadDefIndex, out WmoV14ToV17Converter.WmoDoodadDef def)
    {
        if (doodadDefIndex >= 0 && doodadDefIndex < _wmo.DoodadDefs.Count)
        {
            def = _wmo.DoodadDefs[doodadDefIndex];
            return true;
        }
        def = default;
        return false;
    }

    public string GetDoodadDefName(int doodadDefIndex)
    {
        if (doodadDefIndex >= 0 && doodadDefIndex < _wmo.DoodadDefs.Count)
        {
            var def = _wmo.DoodadDefs[doodadDefIndex];
            return GetDoodadName(def.NameIndex);
        }
        return "";
    }

    public List<int> GetRenderGroupsForDoodadDef(int doodadDefIndex)
    {
        var result = new List<int>();
        if (doodadDefIndex < 0 || doodadDefIndex >= _wmo.DoodadDefs.Count)
            return result;
        for (int renderGroupIndex = 0; renderGroupIndex < _groups.Count; renderGroupIndex++)
        {
            int groupIndex = _groups[renderGroupIndex].GroupIndex;
            if (groupIndex >= 0 && groupIndex < _wmo.Groups.Count && _wmo.Groups[groupIndex].DoodadRefs.Contains((ushort)doodadDefIndex))
                result.Add(renderGroupIndex);
        }
        return result;
    }

    public int GetDoodadCountForRenderGroup(int renderGroupIndex)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
            return 0;
        int groupIndex = _groups[renderGroupIndex].GroupIndex;
        if (groupIndex < 0 || groupIndex >= _wmo.Groups.Count)
            return 0;
        return _wmo.Groups[groupIndex].DoodadRefs.Count;
    }

    public void SetActiveDoodadSet(int index)
    {
        if (index == _activeDoodadSet || index < 0 || index >= _wmo.DoodadSets.Count) return;
        _activeDoodadSet = index;
        LoadActiveDoodadSet();
    }

    public void SetRuntimeDoodadsVisible(bool visible)
    {
        _runtimeDoodadsVisible = visible;
    }

    /// <summary>
    /// Prepares shared WMO doodad animation state once for the current world frame.
    /// WMO placements share this renderer, so doing this inside every placement draw
    /// scales animation CPU cost with placement count instead of unique assets.
    /// </summary>
    public void BeginWorldFrame()
    {
        _doodadAnimationsPreparedForWorldFrame = true;
        UpdateDoodadAnimations();
    }

    public void EndWorldFrame()
    {
        _doodadAnimationsPreparedForWorldFrame = false;
    }

    public void SetRuntimeGroupVisibilityEnabled(bool enabled)
    {
        _enableRuntimeGroupVisibility = enabled;
    }

    public void SetRuntimeGroupLiquidsVisible(bool visible)
    {
        _runtimeGroupLiquidsVisible = visible;
    }

    public bool IsWireframe => _wireframe;

    public void ToggleWireframe()
    {
        _wireframe = !_wireframe;
    }

    public void ApplyTextureSamplingSettings()
    {
        foreach (var textureId in _materialTextures.Values)
        {
            if (textureId == 0)
                continue;

            _gl.BindTexture(TextureTarget.Texture2D, textureId);
            RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: true,
                TextureWrapMode.Repeat, TextureWrapMode.Repeat);
        }

        foreach (var renderer in _doodadModelCache.Values)
            renderer?.ApplyTextureSamplingSettings();

        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    public unsafe void RenderWireframeOverlay(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj,
        Vector3? fogColor = null, float fogStart = 200f, float fogEnd = 1500f, Vector3? cameraPos = null,
        Vector3? lightDir = null, Vector3? lightColor = null, Vector3? ambientColor = null)
    {
        _gl.UseProgram(_shaderProgram);
        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);
        _gl.Disable(EnableCap.Blend);

        var model = modelMatrix;
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&model);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);

        var fc = fogColor ?? new Vector3(0.6f, 0.7f, 0.85f);
        var cp = cameraPos ?? Vector3.Zero;
        _gl.Uniform3(_uFogColor, fc.X, fc.Y, fc.Z);
        _gl.Uniform1(_uFogStart, fogStart);
        _gl.Uniform1(_uFogEnd, fogEnd);
        _gl.Uniform3(_uCameraPos, cp.X, cp.Y, cp.Z);

        var ld = lightDir ?? Vector3.Normalize(new Vector3(0.5f, 0.3f, 1.0f));
        var lc = lightColor ?? new Vector3(1.0f, 0.95f, 0.85f);
        var ac = ambientColor ?? new Vector3(0.35f, 0.35f, 0.4f);
        _gl.Uniform3(_uLightDir, ld.X, ld.Y, ld.Z);
        _gl.Uniform3(_uLightColor, lc.X, lc.Y, lc.Z);
        _gl.Uniform3(_uAmbientColor, ac.X, ac.Y, ac.Z);

        _gl.Uniform1(_uHasTexture, 0);
        _gl.Uniform1(_uAlphaTest, 0.0f);
        _gl.Uniform4(_uColor, 0.95f, 1.0f, 0.65f, 1.0f);

        _gl.LineWidth(1.5f);
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);

        foreach (var gb in _groups)
        {
            if (!gb.IsVisible) continue;
            _gl.BindVertexArray(gb.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, DrawElementsType.UnsignedShort, null);
        }

        _gl.BindVertexArray(0);
        _gl.LineWidth(1.0f);
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
        _gl.DepthMask(true);
        _gl.Enable(EnableCap.CullFace);
    }

    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        RenderWithTransform(Matrix4x4.Identity, view, proj, WmoRenderPass.Both);
    }

    private int GetBaselineMliqRotationQuarterTurns()
    {
        return WmoLiquidLayoutResolver.GetBaselineRotationQuarterTurns(_wmo.Version, _buildVersion);
    }

    /// <summary>
    /// Render this WMO with a custom world transform (for placed WMO instances in WorldScene).
    /// </summary>
    public unsafe void RenderWithTransform(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj,
        Vector3? fogColor = null, float fogStart = 200f, float fogEnd = 1500f, Vector3? cameraPos = null,
        Vector3? lightDir = null, Vector3? lightColor = null, Vector3? ambientColor = null)
    {
        RenderWithTransform(modelMatrix, view, proj, WmoRenderPass.Both,
            fogColor, fogStart, fogEnd, cameraPos,
            lightDir, lightColor, ambientColor);
    }

    /// <summary>
    /// Render this WMO with a custom world transform (for placed WMO instances in WorldScene).
    /// </summary>
    public unsafe void RenderWithTransform(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj, WmoRenderPass pass,
        Vector3? fogColor = null, float fogStart = 200f, float fogEnd = 1500f, Vector3? cameraPos = null,
        Vector3? lightDir = null, Vector3? lightColor = null, Vector3? ambientColor = null)
    {
        ResetRenderStats();
        ProcessDeferredMaterialTextureLoads();
        EnsureLiquidMeshesUpToDate();

        bool renderOpaquePass = pass != WmoRenderPass.Transparent;
        bool renderTransparentPass = pass != WmoRenderPass.Opaque;

        // WMO render order: opaque shell → doodad opaque → liquids → doodad transparent → transparent shell.
        _gl.UseProgram(_shaderProgram);
        ApplySurfaceCulling();
        _gl.Uniform1(_uUseInstanceModel, 0);

        var model = modelMatrix;
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&model);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);

        // Fog uniforms (match terrain fog for seamless blending)
        var fc = fogColor ?? new Vector3(0.6f, 0.7f, 0.85f);
        var cp = cameraPos ?? Vector3.Zero;
        _gl.Uniform3(_uFogColor, fc.X, fc.Y, fc.Z);
        _gl.Uniform1(_uFogStart, fogStart);
        _gl.Uniform1(_uFogEnd, fogEnd);
        _gl.Uniform3(_uCameraPos, cp.X, cp.Y, cp.Z);

        // Lighting uniforms (match terrain lighting for consistent scene illumination)
        var ld = lightDir ?? Vector3.Normalize(new Vector3(0.5f, 0.3f, 1.0f));
        var lc = lightColor ?? new Vector3(1.0f, 0.95f, 0.85f);
        var ac = ambientColor ?? new Vector3(0.35f, 0.35f, 0.4f);
        _gl.Uniform3(_uLightDir, ld.X, ld.Y, ld.Z);
        _gl.Uniform3(_uLightColor, lc.X, lc.Y, lc.Z);
        _gl.Uniform3(_uAmbientColor, ac.X, ac.Y, ac.Z);

        UpdateRuntimeVisibility(modelMatrix, view, proj, cp);

        if (_wireframe)
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);
        else
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // Pass 1: Opaque geometry (BlendMode 0) — depth write ON, no blending
        if (renderOpaquePass)
        {
            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthMask(true);
            _gl.Disable(EnableCap.Blend);
            _gl.Uniform1(_uAlphaTest, 0.0f);

            foreach (var gb in _groups)
            {
                if (!gb.IsVisible) continue;
                _currentVisibleGroupSubmissions++;
                var group = _wmo.Groups[gb.GroupIndex];
                _gl.BindVertexArray(gb.Vao);

                if (group.Batches.Count > 0)
                {
                    foreach (var batch in group.Batches)
                    {
                        int matId = ResolveBatchMaterialId(group, batch);
                        uint rawBlendMode = matId < _wmo.Materials.Count ? _wmo.Materials[matId].BlendMode : 0;
                        EGxBlend blendMode = ResolveWmoBlendMode(rawBlendMode);
                        if (blendMode != EGxBlend.Opaque && blendMode != EGxBlend.AlphaKey)
                            continue;

                        if (blendMode == EGxBlend.AlphaKey)
                        {
                            _gl.Disable(EnableCap.Blend);
                            _gl.DepthMask(true);
                            _gl.Uniform1(_uAlphaTest, WoWConstants.AlphaKeyThreshold);
                        }
                        else
                        {
                            _gl.Disable(EnableCap.Blend);
                            _gl.DepthMask(true);
                            _gl.Uniform1(_uAlphaTest, 0.0f);
                        }

                        DrawBatch(gb, batch, matId);
                    }
                }
                else
                {
                    DrawGroupFallback(gb);
                }
                _gl.BindVertexArray(0);
            }
        }

        // Pass 2: Doodad opaque layers.
        // Distance-culled, sorted nearest-first, capped at DoodadMaxRenderCount.
        int visibleDoodadRenderCount = 0;
        if (_doodadsVisible && _runtimeDoodadsVisible && _doodadInstances.Count > 0)
        {
            if (renderOpaquePass)
                visibleDoodadRenderCount = PrepareVisibleDoodads(modelMatrix, cp, fogEnd, updateAnimation: true);
            else
                visibleDoodadRenderCount = PrepareVisibleDoodads(modelMatrix, cp, fogEnd, updateAnimation: false);

            if (renderOpaquePass)
                RenderOpaqueDoodads(visibleDoodadRenderCount, modelMatrix, view, proj,
                    fc, fogStart, fogEnd, cp, ld, lc, ac);
        }

        // Pass 3: Liquid surfaces (semi-transparent, before transparent WMO geometry)
        if (renderTransparentPass && _runtimeGroupLiquidsVisible && _liquidMeshes.Count > 0)
        {
            _gl.UseProgram(_liquidShader);
            _gl.UniformMatrix4(_uLiqModel, 1, false, (float*)&model);
            _gl.UniformMatrix4(_uLiqView, 1, false, (float*)&view);
            _gl.UniformMatrix4(_uLiqProj, 1, false, (float*)&proj);

            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.DepthMask(false);

            foreach (var liq in _liquidMeshes)
            {
                if (liq.GroupIndex >= 0 && liq.GroupIndex < _runtimeVisibleGroups.Length && !_runtimeVisibleGroups[liq.GroupIndex])
                    continue;

                _currentDrawCalls++;
                _currentLiquidDrawCalls++;
                _currentVisibleLiquidMeshes++;
                _gl.Uniform4(_uLiqColor, liq.ColorR, liq.ColorG, liq.ColorB, liq.ColorA);
                _gl.BindVertexArray(liq.Vao);
                _gl.DrawElements(PrimitiveType.Triangles, liq.IndexCount, DrawElementsType.UnsignedShort, null);
            }

            _gl.BindVertexArray(0);
            _gl.DepthMask(true);
            _gl.Disable(EnableCap.Blend);
        }

        // Pass 4: Doodad transparent layers back-to-front so model glass/reflection stays above liquids.
        if (renderTransparentPass && visibleDoodadRenderCount > 0)
        {
            for (int vi = visibleDoodadRenderCount - 1; vi >= 0; vi--)
            {
                var inst = _doodadInstances[_visibleDoodadsScratch[vi].idx];
                if (!inst.Renderer!.HasTransparentWorldPass)
                    continue;

                var doodadWorld = inst.Transform * modelMatrix;
                _currentDoodadSubmissions++;
                inst.Renderer.RenderWithTransform(doodadWorld, view, proj, RenderPass.Transparent, 1.0f,
                    fogColor, fogStart, fogEnd, cameraPos,
                    lightDir, lightColor, ambientColor);
            }
        }

        // Pass 5: Transparent geometry (BlendMode 1+ = alpha key/blend)
        // Alpha key (BlendMode 1): hard cutout at alpha < 0.5
        // Alpha blend (BlendMode 2+): smooth blending with depth writes off
        if (renderTransparentPass)
        {
            _gl.UseProgram(_shaderProgram);
            _gl.UniformMatrix4(_uModel, 1, false, (float*)&model);
            _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
            _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
            // Re-set fog uniforms after UseProgram (doodad rendering may have changed active program)
            _gl.Uniform3(_uFogColor, fc.X, fc.Y, fc.Z);
            _gl.Uniform1(_uFogStart, fogStart);
            _gl.Uniform1(_uFogEnd, fogEnd);
            _gl.Uniform3(_uCameraPos, cp.X, cp.Y, cp.Z);

            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            _transparentGroupSortScratch.Clear();
            for (int groupBufferIndex = 0; groupBufferIndex < _groups.Count; groupBufferIndex++)
            {
                var groupBuffer = _groups[groupBufferIndex];
                if (!groupBuffer.IsVisible)
                    continue;

                Vector3 worldCenter = Vector3.Transform(groupBuffer.GroupCenter, modelMatrix);
                float distSq = Vector3.DistanceSquared(cp, worldCenter);
                _transparentGroupSortScratch.Add((groupBufferIndex, distSq));
            }

            _transparentGroupSortScratch.Sort((a, b) => b.distSq.CompareTo(a.distSq));

            foreach (var (groupBufferIndex, _) in _transparentGroupSortScratch)
            {
                var gb = _groups[groupBufferIndex];
                if (!gb.IsVisible) continue;
                _currentVisibleGroupSubmissions++;
                var group = _wmo.Groups[gb.GroupIndex];
                _gl.BindVertexArray(gb.Vao);

                if (group.Batches.Count > 0)
                {
                    foreach (var batch in group.Batches)
                    {
                        int matId = ResolveBatchMaterialId(group, batch);
                        uint rawBlendMode = matId < _wmo.Materials.Count ? _wmo.Materials[matId].BlendMode : 0;
                        EGxBlend blendMode = ResolveWmoBlendMode(rawBlendMode);
                        if (blendMode == EGxBlend.Opaque || blendMode == EGxBlend.AlphaKey)
                            continue;

                        _gl.DepthMask(false);
                        _gl.Uniform1(_uAlphaTest, 0.0f);

                        switch (blendMode)
                        {
                            case EGxBlend.Blend:
                                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                                break;
                            case EGxBlend.Add:
                                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                                break;
                            default:
                                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                                break;
                        }

                        DrawBatch(gb, batch, matId);
                    }
                }
                _gl.BindVertexArray(0);
            }

            _gl.DepthMask(true);
            _gl.Disable(EnableCap.Blend);
            _gl.Uniform1(_uAlphaTest, 0.0f);
        }
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
        _gl.Enable(EnableCap.CullFace);
        LastRenderStats = new WmoRenderStats(
            _currentDrawCalls,
            _currentBatchDrawCalls,
            _currentOpaqueBatchInstanceCount,
            _currentGroupFallbackDrawCalls,
            _currentLiquidDrawCalls,
            _currentDoodadSubmissions,
            _currentVisibleGroupSubmissions,
            _currentVisibleLiquidMeshes,
            LastPortalVisibilityDiagnostics.TestedPortalCount,
            LastPortalVisibilityDiagnostics.Mode == WmoPortalVisibilityMode.ConservativeFallback ? 1 : 0,
            LastPortalVisibilityDiagnostics.Mode == WmoPortalVisibilityMode.ConservativeFallback
                ? _runtimeVisibleGroups.Length
                : LastPortalVisibilityDiagnostics.AdmittedGroupCount);
    }

    public unsafe void BeginGpuInstanceBatch(Matrix4x4 view, Matrix4x4 proj,
        Vector3 fogColor, float fogStart, float fogEnd, Vector3 cameraPos,
        Vector3 lightDir, Vector3 lightColor, Vector3 ambientColor)
    {
        ResetRenderStats();
        ProcessDeferredMaterialTextureLoads();
        EnsureGpuInstanceBuffer();

        _gpuInstanceMatrices.Clear();
        _gpuInstanceBatchActive = SupportsGpuInstancedOpaque;
        if (!_gpuInstanceBatchActive)
            return;

        _gl.UseProgram(_shaderProgram);
        ApplySurfaceCulling();
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.PolygonMode(TriangleFace.FrontAndBack, _wireframe ? PolygonMode.Line : PolygonMode.Fill);
        _gl.Uniform1(_uUseInstanceModel, 0);

        var identity = Matrix4x4.Identity;
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&identity);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        _gl.Uniform3(_uFogColor, fogColor.X, fogColor.Y, fogColor.Z);
        _gl.Uniform1(_uFogStart, fogStart);
        _gl.Uniform1(_uFogEnd, fogEnd);
        _gl.Uniform3(_uCameraPos, cameraPos.X, cameraPos.Y, cameraPos.Z);
        _gl.Uniform3(_uLightDir, lightDir.X, lightDir.Y, lightDir.Z);
        _gl.Uniform3(_uLightColor, lightColor.X, lightColor.Y, lightColor.Z);
        _gl.Uniform3(_uAmbientColor, ambientColor.X, ambientColor.Y, ambientColor.Z);
        _gl.Uniform1(_uAlphaTest, 0.0f);
    }

    public void QueueGpuInstance(Matrix4x4 modelMatrix)
    {
        if (_gpuInstanceBatchActive)
            _gpuInstanceMatrices.Add(modelMatrix);
    }

    public unsafe void EndGpuInstanceBatch()
    {
        if (!_gpuInstanceBatchActive)
            return;

        _gpuInstanceBatchActive = false;
        if (_gpuInstanceMatrices.Count == 0)
            return;

        UploadGpuInstanceData();
        uint instanceCount = (uint)_gpuInstanceMatrices.Count;
        _currentOpaqueBatchInstanceCount = (int)instanceCount;
        _gl.UseProgram(_shaderProgram);
        _gl.Uniform1(_uUseInstanceModel, 1);
        _gl.Uniform1(_uAlphaTest, 0.0f);

        try
        {
            // The instanced shell never consults runtime group visibility: every manually visible
            // group is submitted once per instance. Recorded per instance so the counts line up
            // with VisibleGroupSubmissions instead of quietly under-reporting by the batch factor.
            for (uint instance = 0; instance < instanceCount; instance++)
            {
                int admittedInPlacement = 0;
                foreach (GroupBuffers gb in _groups)
                {
                    bool admitted = gb.ManualVisible;
                    _groupAdmission.RecordGroup(admitted
                        ? WmoGroupAdmissionRule.GpuInstancedShell
                        : WmoGroupAdmissionRule.None);
                    if (admitted)
                        admittedInPlacement++;
                }

                _groupAdmission.RecordGroupPlacementEvaluation(admittedInPlacement, _modelDir, null);
            }

            foreach (GroupBuffers gb in _groups)
            {
                if (!gb.ManualVisible)
                    continue;

                _currentVisibleGroupSubmissions += (int)instanceCount;
                var group = _wmo.Groups[gb.GroupIndex];
                _gl.BindVertexArray(gb.Vao);

                if (group.Batches.Count > 0)
                {
                    foreach (var batch in group.Batches)
                    {
                        int matId = ResolveBatchMaterialId(group, batch);
                        uint rawBlendMode = matId < _wmo.Materials.Count ? _wmo.Materials[matId].BlendMode : 0;
                        EGxBlend blendMode = ResolveWmoBlendMode(rawBlendMode);
                        if (blendMode != EGxBlend.Opaque && blendMode != EGxBlend.AlphaKey)
                            continue;

                        _gl.Uniform1(_uAlphaTest,
                            blendMode == EGxBlend.AlphaKey ? WoWConstants.AlphaKeyThreshold : 0.0f);
                        DrawInstancedBatch(gb, batch, matId, instanceCount);
                    }
                }
                else
                {
                    DrawInstancedGroupFallback(gb, instanceCount);
                }

                _gl.BindVertexArray(0);
            }
        }
        finally
        {
            _gl.Uniform1(_uUseInstanceModel, 0);
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, 0);
            _gl.BindVertexArray(0);
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
            UpdateLastRenderStats();
        }
    }

    public unsafe void RenderOpaqueDoodadsForPlacement(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj,
        Vector3 fogColor, float fogStart, float fogEnd, Vector3 cameraPos,
        Vector3 lightDir, Vector3 lightColor, Vector3 ambientColor)
    {
        if (!SupportsGpuInstancedOpaque || !_doodadsVisible || !_runtimeDoodadsVisible || _doodadInstances.Count == 0)
            return;

        // The opaque WMO shell was admitted to a shared instance batch. For this
        // route the renderer has no portals and all manual groups are visible, so
        // repeating placement-local portal/frustum traversal is redundant.
        int visibleDoodadRenderCount = PrepareVisibleDoodads(
            modelMatrix,
            cameraPos,
            fogEnd,
            updateAnimation: false,
            sortByDistance: false,
            respectRuntimeDoodadVisibility: false);
        RenderOpaqueDoodads(visibleDoodadRenderCount, modelMatrix, view, proj,
            fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
        UpdateLastRenderStats();
    }

    public void CollectOpaqueDoodadsForPlacement(
        Matrix4x4 modelMatrix,
        Vector3 cameraPos,
        float fogEnd,
        Action<WmoOpaqueDoodadBatchItem> collect)
    {
        ArgumentNullException.ThrowIfNull(collect);
        if (!SupportsGpuInstancedOpaque || !_doodadsVisible || !_runtimeDoodadsVisible || _doodadInstances.Count == 0)
            return;

        int visibleDoodadRenderCount = PrepareVisibleDoodads(
            modelMatrix,
            cameraPos,
            fogEnd,
            updateAnimation: false,
            sortByDistance: false,
            respectRuntimeDoodadVisibility: false);
        for (int vi = 0; vi < visibleDoodadRenderCount; vi++)
        {
            DoodadInstance inst = _doodadInstances[_visibleDoodadsScratch[vi].idx];
            if (inst.Renderer is not IModelRenderer renderer)
                continue;

            _currentDoodadSubmissions++;
            collect(new WmoOpaqueDoodadBatchItem(renderer, inst.Transform * modelMatrix));
        }

        UpdateLastRenderStats();
    }

    private int PrepareVisibleDoodads(
        Matrix4x4 modelMatrix,
        Vector3 cameraPos,
        float fogEnd,
        bool updateAnimation,
        bool sortByDistance = true,
        bool respectRuntimeDoodadVisibility = true)
    {
        _updatedDoodadModelsScratch.Clear();
        _visibleDoodadsScratch.Clear();

        if (updateAnimation && !_doodadAnimationsPreparedForWorldFrame)
            UpdateDoodadAnimations();

        float doodadCullDistance = MathF.Max(DoodadCullDistance, MathF.Min(fogEnd + 800f, 6000f));
        float cullDistSq = doodadCullDistance * doodadCullDistance;
        for (int di = 0; di < _doodadInstances.Count; di++)
        {
            DoodadInstance inst = _doodadInstances[di];
            if (!inst.Visible || inst.Renderer == null)
                continue;
            if (respectRuntimeDoodadVisibility
                && _runtimeVisibleDoodadDefIndices.Count > 0
                && !_runtimeVisibleDoodadDefIndices.Contains(inst.DoodadDefIndex))
                continue;

            Vector3 worldPos = Vector3.Transform(inst.LocalPosition, modelMatrix);
            float distSq = Vector3.DistanceSquared(cameraPos, worldPos);
            if (distSq > cullDistSq)
                continue;

            _visibleDoodadsScratch.Add((di, distSq));
        }

        if (sortByDistance && _visibleDoodadsScratch.Count > 1)
            _visibleDoodadsScratch.Sort((a, b) => a.distSq.CompareTo(b.distSq));

        return Math.Min(_visibleDoodadsScratch.Count, (int)DoodadMaxRenderCount);
    }

    private void UpdateDoodadAnimations()
    {
        _updatedDoodadModelsScratch.Clear();
        foreach (DoodadInstance inst in _doodadInstances)
        {
            if (inst.Renderer != null && _updatedDoodadModelsScratch.Add(inst.ModelPath))
                inst.Renderer.UpdateAnimation();
        }
    }

    private unsafe void RenderOpaqueDoodads(int visibleDoodadRenderCount, Matrix4x4 modelMatrix,
        Matrix4x4 view, Matrix4x4 proj, Vector3 fogColor, float fogStart, float fogEnd,
        Vector3 cameraPos, Vector3 lightDir, Vector3 lightColor, Vector3 ambientColor)
    {
        _opaqueDoodadBatchGroups.Clear();
        _opaqueDoodadBatchRenderers.Clear();

        for (int vi = 0; vi < visibleDoodadRenderCount; vi++)
        {
            int doodadIndex = _visibleDoodadsScratch[vi].idx;
            DoodadInstance inst = _doodadInstances[doodadIndex];
            IModelRenderer renderer = inst.Renderer!;
            if (renderer.RequiresUnbatchedWorldRender)
            {
                _currentDoodadSubmissions++;
                renderer.RenderWithTransform(inst.Transform * modelMatrix, view, proj, RenderPass.Opaque, 1.0f,
                    fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
                continue;
            }

            if (!_opaqueDoodadBatchGroups.TryGetValue(renderer, out List<int>? indices))
            {
                indices = new List<int>();
                _opaqueDoodadBatchGroups.Add(renderer, indices);
                _opaqueDoodadBatchRenderers.Add(renderer);
            }

            indices.Add(doodadIndex);
        }

        foreach (IModelRenderer renderer in _opaqueDoodadBatchRenderers)
        {
            List<int> indices = _opaqueDoodadBatchGroups[renderer];
            if (renderer is IGpuInstancedModelRenderer gpuRenderer
                && gpuRenderer.SupportsGpuInstancedOpaque)
            {
                gpuRenderer.BeginGpuInstanceBatch(view, proj, fogColor, fogStart, fogEnd,
                    cameraPos, lightDir, lightColor, ambientColor);
                foreach (int doodadIndex in indices)
                {
                    DoodadInstance inst = _doodadInstances[doodadIndex];
                    gpuRenderer.QueueGpuInstance(inst.Transform * modelMatrix, 1.0f);
                    _currentDoodadSubmissions++;
                }

                gpuRenderer.EndGpuInstanceBatch();
            }
            else
            {
                renderer.BeginBatch(view, proj, fogColor, fogStart, fogEnd,
                    cameraPos, lightDir, lightColor, ambientColor);
                foreach (int doodadIndex in indices)
                {
                    DoodadInstance inst = _doodadInstances[doodadIndex];
                    renderer.RenderInstance(inst.Transform * modelMatrix, RenderPass.Opaque, 1.0f);
                    _currentDoodadSubmissions++;
                }
            }
        }
    }

    private void UpdateLastRenderStats()
    {
        LastRenderStats = new WmoRenderStats(
            _currentDrawCalls,
            _currentBatchDrawCalls,
            _currentOpaqueBatchInstanceCount,
            _currentGroupFallbackDrawCalls,
            _currentLiquidDrawCalls,
            _currentDoodadSubmissions,
            _currentVisibleGroupSubmissions,
            _currentVisibleLiquidMeshes,
            LastPortalVisibilityDiagnostics.TestedPortalCount,
            LastPortalVisibilityDiagnostics.Mode == WmoPortalVisibilityMode.ConservativeFallback ? 1 : 0,
            LastPortalVisibilityDiagnostics.Mode == WmoPortalVisibilityMode.ConservativeFallback
                ? _runtimeVisibleGroups.Length
                : LastPortalVisibilityDiagnostics.AdmittedGroupCount);
    }

    private void ResetRenderStats()
    {
        _currentDrawCalls = 0;
        _currentBatchDrawCalls = 0;
        _currentOpaqueBatchInstanceCount = 0;
        _currentGroupFallbackDrawCalls = 0;
        _currentLiquidDrawCalls = 0;
        _currentDoodadSubmissions = 0;
        _currentVisibleGroupSubmissions = 0;
        _currentVisibleLiquidMeshes = 0;
        LastPortalVisibilityDiagnostics = new();
        LastRenderStats = default;
        _groupAdmission.Reset();
    }

    private void ApplySurfaceCulling()
    {
        // WMO materials aren't reliably single-sided (double-sided flags vary per material),
        // so WMO backface culling was never safely on by default; removed as an option.
        _gl.Disable(EnableCap.CullFace);
    }

    private static EGxBlend ResolveWmoBlendMode(uint rawBlendMode)
    {
        return rawBlendMode switch
        {
            0 => EGxBlend.Opaque,
            // WMO MOMT blend-mode mapping parity with Alpha-era EGx semantics:
            // 0 = Opaque, 1 = AlphaKey (cutout), 2 = Blend, 3 = Add.
            // Treating mode 1 as full Blend causes shell cutouts (e.g., windows/cloth)
            // to render in transparent pass and can expose interior surfaces through walls.
            1 => EGxBlend.AlphaKey,
            2 => EGxBlend.Blend,
            3 => EGxBlend.Add,
            _ => EGxBlend.Blend,
        };
    }

    private unsafe void DrawBatch(GroupBuffers gb, WmoV14ToV17Converter.WmoBatch batch, int matId)
    {
        if (!TryValidateBatchDrawRange(gb, batch))
            return;

        if (!TryBindGroupGeometry(gb, batch))
            return;

        if (_materialTextures.TryGetValue(matId, out uint glTex))
        {
            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, glTex);
            _gl.Uniform1(_uHasTexture, 1);
            _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 1.0f);
        }
        else
        {
            _gl.Uniform1(_uHasTexture, 0);
            float r = ((gb.GroupIndex * 67 + 13) % 255) / 255f;
            float g = ((gb.GroupIndex * 131 + 7) % 255) / 255f;
            float b = ((gb.GroupIndex * 43 + 29) % 255) / 255f;
            _gl.Uniform4(_uColor, r, g, b, 1.0f);
        }
        _currentDrawCalls++;
        _currentBatchDrawCalls++;
        _gl.DrawElements(PrimitiveType.Triangles, batch.IndexCount,
            DrawElementsType.UnsignedShort, null);
    }

    private unsafe void DrawInstancedBatch(GroupBuffers gb, WmoV14ToV17Converter.WmoBatch batch,
        int matId, uint instanceCount)
    {
        if (!TryValidateBatchDrawRange(gb, batch))
            return;

        if (!TryBindGroupGeometry(gb, batch))
            return;

        if (_materialTextures.TryGetValue(matId, out uint glTex))
        {
            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, glTex);
            _gl.Uniform1(_uHasTexture, 1);
            _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 1.0f);
        }
        else
        {
            _gl.Uniform1(_uHasTexture, 0);
            float r = ((gb.GroupIndex * 67 + 13) % 255) / 255f;
            float g = ((gb.GroupIndex * 131 + 7) % 255) / 255f;
            float b = ((gb.GroupIndex * 43 + 29) % 255) / 255f;
            _gl.Uniform4(_uColor, r, g, b, 1.0f);
        }

        _currentDrawCalls++;
        _currentBatchDrawCalls++;
        _gl.DrawElementsInstanced(PrimitiveType.Triangles, batch.IndexCount,
            DrawElementsType.UnsignedShort, null, instanceCount);
    }

    private bool TryBindGroupGeometry(GroupBuffers gb, WmoV14ToV17Converter.WmoBatch batch)
    {
        if (!gb.BatchEbos.TryGetValue((batch.FirstIndex, batch.IndexCount), out uint batchEbo))
        {
            LogInvalidBatchRange(gb, batch, "no compact batch EBO was uploaded");
            return false;
        }

        if (gb.Vao == 0 || batchEbo == 0 || !_gl.IsVertexArray(gb.Vao) || !_gl.IsBuffer(batchEbo))
        {
            LogInvalidBatchRange(gb, batch,
                $"GPU geometry handle is not live (vao={gb.Vao}, batchEbo={batchEbo})");
            return false;
        }

        // Rebind both objects at the draw site. Doodad/model passes can change the active
        // VAO and an element-array binding belongs to the currently bound VAO in OpenGL.
        _gl.BindVertexArray(gb.Vao);
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, batchEbo);

        long bufferSizeBytes = _gl.GetBufferParameter(
            BufferTargetARB.ElementArrayBuffer,
            BufferPNameARB.BufferSize);
        ulong requiredBytes = (ulong)batch.IndexCount * sizeof(ushort);
        if (bufferSizeBytes < 0 || (ulong)bufferSizeBytes < requiredBytes)
        {
            LogInvalidBatchRange(gb, batch,
                $"GPU EBO size {bufferSizeBytes} bytes is smaller than required {requiredBytes} bytes");
            return false;
        }

        return true;
    }

    private bool TryValidateBatchDrawRange(GroupBuffers gb, WmoV14ToV17Converter.WmoBatch batch)
    {
        ulong firstIndex = batch.FirstIndex;
        ulong indexCount = batch.IndexCount;
        ulong indexEnd = firstIndex + indexCount;
        bool validRange = gb.Ebo != 0
            && indexCount > 0
            && indexEnd <= gb.IndexCount
            && gb.GroupIndex >= 0
            && gb.GroupIndex < _wmo.Groups.Count;

        if (validRange)
        {
            var group = _wmo.Groups[gb.GroupIndex];
            validRange = indexEnd <= (ulong)group.Indices.Count;
            if (validRange)
            {
                int first = (int)firstIndex;
                int end = (int)indexEnd;
                for (int index = first; index < end; index++)
                {
                    if (group.Indices[index] < gb.VertexCount)
                        continue;

                    LogInvalidBatchRange(
                        gb,
                        batch,
                        $"vertex index {group.Indices[index]} exceeds vertex count {gb.VertexCount}");
                    return false;
                }

                return true;
            }
        }

        LogInvalidBatchRange(gb, batch, "index range exceeds uploaded or source index data");
        return false;
    }

    private void LogInvalidBatchRange(
        GroupBuffers gb,
        WmoV14ToV17Converter.WmoBatch batch,
        string reason)
    {
        if (_invalidBatchRangeLogCount >= MaxInvalidBatchRangeLogs)
            return;

        string key = $"{gb.GroupIndex}|{batch.FirstIndex}|{batch.IndexCount}|{gb.IndexCount}|{gb.VertexCount}|{gb.Ebo}";
        if (_invalidBatchRangeLogKeys.Add(key))
        {
            _invalidBatchRangeLogCount++;
            ViewerLog.Error(
                ViewerLog.Category.Wmo,
                $"[WMO] Skipping invalid batch draw: model='{_modelDir}' group={gb.GroupIndex} firstIndex={batch.FirstIndex} " +
                $"indexCount={batch.IndexCount} indexBufferCount={gb.IndexCount} vertexCount={gb.VertexCount} " +
                $"ebo={gb.Ebo} reason={reason}");
        }
    }

    private unsafe void DrawGroupFallback(GroupBuffers gb)
    {
        _gl.Uniform1(_uHasTexture, 0);
        float r = ((gb.GroupIndex * 67 + 13) % 255) / 255f;
        float g = ((gb.GroupIndex * 131 + 7) % 255) / 255f;
        float b = ((gb.GroupIndex * 43 + 29) % 255) / 255f;
        _gl.Uniform4(_uColor, r, g, b, 1.0f);
        _currentDrawCalls++;
        _currentGroupFallbackDrawCalls++;
        _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, DrawElementsType.UnsignedShort, null);
    }

    private unsafe void DrawInstancedGroupFallback(GroupBuffers gb, uint instanceCount)
    {
        _gl.Uniform1(_uHasTexture, 0);
        float r = ((gb.GroupIndex * 67 + 13) % 255) / 255f;
        float g = ((gb.GroupIndex * 131 + 7) % 255) / 255f;
        float b = ((gb.GroupIndex * 43 + 29) % 255) / 255f;
        _gl.Uniform4(_uColor, r, g, b, 1.0f);
        _currentDrawCalls++;
        _currentGroupFallbackDrawCalls++;
        _gl.DrawElementsInstanced(PrimitiveType.Triangles, gb.IndexCount,
            DrawElementsType.UnsignedShort, null, instanceCount);
    }

    private unsafe void EnsureGpuInstanceBuffer()
    {
        if (_gpuInstanceVbo != 0)
            return;

        _gpuInstanceVbo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _gpuInstanceVbo);

        // Every WMO VAO carries the divisor-1 instance attributes so it can be reused by
        // opaque instanced draws. Ordinary DrawElements calls still traverse those enabled
        // attributes on some drivers, even when uUseInstanceModel selects uModel. Seed the
        // buffer with one complete matrix so the non-instanced path never fetches from a
        // zero-byte buffer before the first real batch upload.
        float[] identity =
        {
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f,
        };
        fixed (float* data = identity)
        {
            _gl.BufferData(
                BufferTargetARB.ArrayBuffer,
                (nuint)(identity.Length * sizeof(float)),
                data,
                BufferUsageARB.StreamDraw);
        }
    }

    private unsafe void UploadGpuInstanceData()
    {
        int requiredFloatCount = _gpuInstanceMatrices.Count * 16;
        if (_gpuInstanceUploadScratch.Length < requiredFloatCount)
            _gpuInstanceUploadScratch = new float[requiredFloatCount];

        for (int index = 0; index < _gpuInstanceMatrices.Count; index++)
        {
            Matrix4x4 model = _gpuInstanceMatrices[index];
            int offset = index * 16;
            _gpuInstanceUploadScratch[offset + 0] = model.M11;
            _gpuInstanceUploadScratch[offset + 1] = model.M12;
            _gpuInstanceUploadScratch[offset + 2] = model.M13;
            _gpuInstanceUploadScratch[offset + 3] = model.M14;
            _gpuInstanceUploadScratch[offset + 4] = model.M21;
            _gpuInstanceUploadScratch[offset + 5] = model.M22;
            _gpuInstanceUploadScratch[offset + 6] = model.M23;
            _gpuInstanceUploadScratch[offset + 7] = model.M24;
            _gpuInstanceUploadScratch[offset + 8] = model.M31;
            _gpuInstanceUploadScratch[offset + 9] = model.M32;
            _gpuInstanceUploadScratch[offset + 10] = model.M33;
            _gpuInstanceUploadScratch[offset + 11] = model.M34;
            _gpuInstanceUploadScratch[offset + 12] = model.M41;
            _gpuInstanceUploadScratch[offset + 13] = model.M42;
            _gpuInstanceUploadScratch[offset + 14] = model.M43;
            _gpuInstanceUploadScratch[offset + 15] = model.M44;
        }

        EnsureGpuInstanceBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _gpuInstanceVbo);
        fixed (float* data = _gpuInstanceUploadScratch)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer,
                (nuint)(requiredFloatCount * sizeof(float)), data, BufferUsageARB.StreamDraw);
        }
    }

    private unsafe void ConfigureGpuInstanceAttributes()
    {
        EnsureGpuInstanceBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _gpuInstanceVbo);
        uint instanceStride = 16 * sizeof(float);
        for (uint column = 0; column < 4; column++)
        {
            uint location = 5 + column;
            _gl.EnableVertexAttribArray(location);
            _gl.VertexAttribPointer(location, 4, VertexAttribPointerType.Float, false,
                instanceStride, (void*)(column * 4 * sizeof(float)));
            _gl.VertexAttribDivisor(location, 1);
        }
    }

    private void InitShaders()
    {
        _shaderRefCount++;
        if (_shaderProgram != 0) return; // Already initialized by another instance

        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aVertexLight;
layout(location = 4) in float aBakedWeight;
layout(location = 5) in vec4 aInstanceModel0;
layout(location = 6) in vec4 aInstanceModel1;
layout(location = 7) in vec4 aInstanceModel2;
layout(location = 8) in vec4 aInstanceModel3;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform int uUseInstanceModel;

out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vFragPos;
out vec4 vVertexLight;
out float vBakedWeight;

void main() {
    mat4 model = uUseInstanceModel == 1
        ? mat4(aInstanceModel0, aInstanceModel1, aInstanceModel2, aInstanceModel3)
        : uModel;
    vec4 worldPos = model * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(model))) * aNormal;
    vTexCoord = aTexCoord;
    vVertexLight = aVertexLight;
    vBakedWeight = aBakedWeight;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vNormal;
in vec2 vTexCoord;
in vec3 vFragPos;
in vec4 vVertexLight;
in float vBakedWeight;

uniform sampler2D uSampler;
uniform int uHasTexture;
uniform vec4 uColor;
uniform float uAlphaTest;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;

out vec4 FragColor;

void main() {
    vec3 norm = normalize(vNormal);
    // Half-Lambert diffuse: wraps lighting around surfaces for softer shading
    // Prevents harsh black shadows that don't match WoW's look
    float NdotL = dot(norm, normalize(uLightDir));
    float diff = NdotL * 0.5 + 0.5; // half-Lambert: remap [-1,1] to [0,1]
    diff = diff * diff; // square for slightly sharper falloff
    vec3 lighting = uAmbientColor + uLightColor * diff;
    float bakedWeight = clamp(vBakedWeight, 0.0, 1.0);
    vec3 bakedLighting = mix(vec3(1.0), clamp(vVertexLight.rgb, vec3(0.0), vec3(1.0)), bakedWeight);

    vec4 texColor;
    if (uHasTexture == 1) {
        texColor = texture(uSampler, vTexCoord);
    } else {
        texColor = uColor;
    }

    // Alpha test: discard fragments below threshold (for cutout/transparent materials)
    if (uAlphaTest > 0.0 && texColor.a < uAlphaTest)
        discard;

    // Fog: blend to fog color based on distance from camera
    vec3 litColor = texColor.rgb * lighting * bakedLighting;
    float dist = length(vFragPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 foggedColor = mix(uFogColor, litColor, fogFactor);

    FragColor = vec4(foggedColor, texColor.a);
}
";

        uint vert = CompileShader(ShaderType.VertexShader, vertSrc);
        uint frag = CompileShader(ShaderType.FragmentShader, fragSrc);

        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vert);
        _gl.AttachShader(_shaderProgram, frag);
        _gl.LinkProgram(_shaderProgram);

        _gl.GetProgram(_shaderProgram, ProgramPropertyARB.LinkStatus, out int status);
        if (status == 0)
            throw new Exception($"Shader link error: {_gl.GetProgramInfoLog(_shaderProgram)}");

        _gl.DeleteShader(vert);
        _gl.DeleteShader(frag);

        _gl.UseProgram(_shaderProgram);
        _uModel = _gl.GetUniformLocation(_shaderProgram, "uModel");
        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uUseInstanceModel = _gl.GetUniformLocation(_shaderProgram, "uUseInstanceModel");
        _uHasTexture = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uColor = _gl.GetUniformLocation(_shaderProgram, "uColor");
        _uAlphaTest = _gl.GetUniformLocation(_shaderProgram, "uAlphaTest");
        _uFogColor = _gl.GetUniformLocation(_shaderProgram, "uFogColor");
        _uFogStart = _gl.GetUniformLocation(_shaderProgram, "uFogStart");
        _uFogEnd = _gl.GetUniformLocation(_shaderProgram, "uFogEnd");
        _uCameraPos = _gl.GetUniformLocation(_shaderProgram, "uCameraPos");
        _uLightDir = _gl.GetUniformLocation(_shaderProgram, "uLightDir");
        _uLightColor = _gl.GetUniformLocation(_shaderProgram, "uLightColor");
        _uAmbientColor = _gl.GetUniformLocation(_shaderProgram, "uAmbientColor");
    }

    private uint CompileShader(ShaderType type, string source)
    {
        uint shader = _gl.CreateShader(type);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);

        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status == 0)
            throw new Exception($"Shader compile error ({type}): {_gl.GetShaderInfoLog(shader)}");

        return shader;
    }

    private WmoPortalVisibilityGroup[] BuildPortalVisibilityGroups()
        => _wmo.Groups
            .Select((group, groupIndex) => new WmoPortalVisibilityGroup(
                groupIndex,
                group.Flags,
                group.BoundsMin,
                group.BoundsMax))
            .ToArray();

    private WmoPortalVisibilityPortal[] BuildPortalVisibilityPortals()
    {
        var portals = new WmoPortalVisibilityPortal[_wmo.Portals.Count];
        for (int portalIndex = 0; portalIndex < _wmo.Portals.Count; portalIndex++)
        {
            WmoV14ToV17Converter.WmoPortal portal = _wmo.Portals[portalIndex];
            var vertices = new List<Vector3>();
            int startVertex = portal.StartVertex;
            int vertexCount = portal.Count;
            if (vertexCount >= 3 && startVertex <= _wmo.PortalVertices.Count - vertexCount)
            {
                vertices.AddRange(_wmo.PortalVertices.Skip(startVertex).Take(vertexCount));
            }

            WmoPortalVisibilityReference[] references = _wmo.PortalRefs
                .Where(reference => reference.PortalIndex == portalIndex)
                .Select(reference => new WmoPortalVisibilityReference(reference.GroupIndex, reference.Side))
                .ToArray();
            portals[portalIndex] = new WmoPortalVisibilityPortal(
                portalIndex,
                vertices,
                new Vector3(portal.PlaneA, portal.PlaneB, portal.PlaneC),
                portal.PlaneD,
                references);
        }

        return portals;
    }

    private void UpdateRuntimeVisibility(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos)
    {
        Array.Clear(_runtimeVisibleGroups, 0, _runtimeVisibleGroups.Length);
        Array.Clear(_frustumVisibleScratch, 0, _frustumVisibleScratch.Length);
        Array.Clear(_portalVisibleScratch, 0, _portalVisibleScratch.Length);
        _runtimeVisibleDoodadDefIndices.Clear();
        _groupAdmission.Reset();

        if (_wmo.Groups.Count == 0)
            return;

        if (!_enableRuntimeGroupVisibility)
        {
            for (int groupIndex = 0; groupIndex < _runtimeVisibleGroups.Length; groupIndex++)
            {
                _runtimeVisibleGroups[groupIndex] = true;
                _groupAdmission.RecordGroup(WmoGroupAdmissionRule.RuntimeVisibilityDisabled);
            }

            _groupAdmission.RecordGroupPlacementEvaluation(_runtimeVisibleGroups.Length, _modelDir, null);
            ApplyRuntimeVisibilityToBuffers();
            CollectVisibleDoodadDefs();
            return;
        }

        if (!Matrix4x4.Invert(modelMatrix, out var inverseModel))
        {
            for (int i = 0; i < _runtimeVisibleGroups.Length; i++)
            {
                _runtimeVisibleGroups[i] = true;
                _groupAdmission.RecordGroup(WmoGroupAdmissionRule.PlacementTransformInvalid);
            }

            LastPortalVisibilityDiagnostics = WmoPortalVisibilityDiagnostics.CreateFallback("placement_transform_invalid");
            _groupAdmission.RecordGroupPlacementEvaluation(
                _runtimeVisibleGroups.Length, _modelDir, LastPortalVisibilityDiagnostics.FallbackReason);
            ApplyRuntimeVisibilityToBuffers();
            CollectVisibleDoodadDefs();
            return;
        }

        Vector3 localCameraPos = Vector3.Transform(cameraPos, inverseModel);
        _groupFrustumCuller.Update(view * proj);

        for (int groupIndex = 0; groupIndex < _wmo.Groups.Count; groupIndex++)
        {
            TransformAabb(_wmo.Groups[groupIndex].BoundsMin, _wmo.Groups[groupIndex].BoundsMax,
                modelMatrix, out Vector3 worldMin, out Vector3 worldMax);
            _frustumVisibleScratch[groupIndex] = _groupFrustumCuller.TestAABB(worldMin, worldMax);
        }

        // Native 0.5.3 uses transformed portal polygons and a recursively narrowed view volume.
        // The pure evaluator mirrors that contract from decoded data and returns all groups when
        // any required evidence is invalid, keeping the renderer fail-open for old WMO variants.
        WmoPortalVisibilityDecision decision = WmoPortalVisibilityEvaluator.Evaluate(
            _portalVisibilityGroups,
            _portalVisibilityPortals,
            localCameraPos,
            groupIndex => (uint)groupIndex < (uint)_frustumVisibleScratch.Length
                && _frustumVisibleScratch[groupIndex],
            interiorMaximumDepth: InteriorPortalTraversalDepth,
            exteriorMaximumDepth: ExteriorPortalTraversalDepth);
        LastPortalVisibilityDiagnostics = decision.Diagnostics;
        foreach (int groupIndex in decision.VisibleGroupIndices)
        {
            if ((uint)groupIndex < (uint)_runtimeVisibleGroups.Length)
            {
                _runtimeVisibleGroups[groupIndex] = true;
                _portalVisibleScratch[groupIndex] = true;
            }
        }

        // Portal traversal is a conservative optimization, never the final
        // correctness culler. A group whose transformed bounds are in the
        // camera frustum must remain drawable even when portal winding,
        // incomplete exterior flags, or an old-build portal edge disagrees.
        // Connected groups admitted by the portal walk remain visible too.
        for (int groupIndex = 0; groupIndex < _frustumVisibleScratch.Length; groupIndex++)
        {
            if (!_frustumVisibleScratch[groupIndex])
                continue;

            _runtimeVisibleGroups[groupIndex] = true;
        }

        // Accounting only — the decisions above are unchanged. A conservative fallback admits every
        // group by construction, so it is recorded as its own rule instead of being credited to the
        // portal walk that did not actually run.
        bool portalFallback = decision.Diagnostics.Mode == WmoPortalVisibilityMode.ConservativeFallback;
        int admittedInPlacement = 0;
        for (int groupIndex = 0; groupIndex < _runtimeVisibleGroups.Length; groupIndex++)
        {
            bool byPortal = _portalVisibleScratch[groupIndex];
            bool byFrustum = _frustumVisibleScratch[groupIndex];
            WmoGroupAdmissionRule rule = (portalFallback && byPortal, byPortal, byFrustum) switch
            {
                (true, _, _) => WmoGroupAdmissionRule.PortalFallback,
                (_, true, true) => WmoGroupAdmissionRule.PortalAndFrustum,
                (_, true, false) => WmoGroupAdmissionRule.Portal,
                (_, false, true) => WmoGroupAdmissionRule.Frustum,
                _ => WmoGroupAdmissionRule.None,
            };

            _groupAdmission.RecordGroup(rule);
            if (rule != WmoGroupAdmissionRule.None)
                admittedInPlacement++;
        }

        _groupAdmission.RecordGroupPlacementEvaluation(
            admittedInPlacement, _modelDir, portalFallback ? decision.Diagnostics.FallbackReason : null);

        ApplyRuntimeVisibilityToBuffers();
        CollectVisibleDoodadDefs();
    }

    private void ApplyRuntimeVisibilityToBuffers()
    {
        foreach (var groupBuffer in _groups)
        {
            if ((uint)groupBuffer.GroupIndex < (uint)_runtimeVisibleGroups.Length)
                groupBuffer.RuntimeVisible = _runtimeVisibleGroups[groupBuffer.GroupIndex];
            else
                groupBuffer.RuntimeVisible = true;
        }
    }

    private void CollectVisibleDoodadDefs()
    {
        for (int groupIndex = 0; groupIndex < _wmo.Groups.Count; groupIndex++)
        {
            if (!_runtimeVisibleGroups[groupIndex])
                continue;

            foreach (ushort doodadRef in _wmo.Groups[groupIndex].DoodadRefs)
            {
                if ((uint)doodadRef < (uint)_wmo.DoodadDefs.Count)
                    _runtimeVisibleDoodadDefIndices.Add(doodadRef);
            }
        }
    }

    private static void TransformAabb(Vector3 min, Vector3 max, Matrix4x4 transform, out Vector3 outMin, out Vector3 outMax)
    {
        outMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        outMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        Span<float> xs = stackalloc float[] { min.X, max.X };
        Span<float> ys = stackalloc float[] { min.Y, max.Y };
        Span<float> zs = stackalloc float[] { min.Z, max.Z };

        foreach (float x in xs)
        foreach (float y in ys)
        foreach (float z in zs)
        {
            Vector3 transformed = Vector3.Transform(new Vector3(x, y, z), transform);
            outMin = Vector3.Min(outMin, transformed);
            outMax = Vector3.Max(outMax, transformed);
        }
    }

    private unsafe void InitBuffers()
    {
        EnsureGpuInstanceBuffer();
        for (int gi = 0; gi < _wmo.Groups.Count; gi++)
        {
            var group = _wmo.Groups[gi];
            if (group.Vertices.Count == 0 || group.Indices.Count == 0)
                continue;

            var gb = new GroupBuffers
            {
                GroupIndex = gi,
                GroupCenter = (group.BoundsMin + group.BoundsMax) * 0.5f
            };

            // Prefer parsed MONR normals when available; fallback to generated normals.
            // 3.3.5 WMOs can carry authored normals that better match client lighting.
            var normals = BuildRenderNormals(group);

            int vertCount = group.Vertices.Count;
            bool hasUVs = group.UVs.Count == vertCount;
            if (!hasUVs)
                ViewerLog.Trace($"[WmoRenderer] Group {gi} '{group.Name}': UV count mismatch! Verts={vertCount}, UVs={group.UVs.Count}");

            Vector4[] vertexLightColors = BuildVertexLightColors(group);

            // Interleave: pos(3) + normal(3) + uv(2) + vertexLight(4) = 12 floats
            float[] vertexData = new float[vertCount * 12];
            for (int v = 0; v < vertCount; v++)
            {
                // Pass through raw WoW model-local coords.
                // Coordinate conversion is handled by the placement transform.
                var pos = group.Vertices[v];
                int baseOffset = v * 12;
                vertexData[baseOffset + 0] = pos.X;
                vertexData[baseOffset + 1] = pos.Y;
                vertexData[baseOffset + 2] = pos.Z;

                var n = v < normals.Count ? normals[v] : Vector3.UnitY;
                vertexData[baseOffset + 3] = n.X;
                vertexData[baseOffset + 4] = n.Y;
                vertexData[baseOffset + 5] = n.Z;

                if (hasUVs)
                {
                    var uv = group.UVs[v];
                    vertexData[baseOffset + 6] = uv.X;
                    vertexData[baseOffset + 7] = uv.Y;
                }

                Vector4 vertexLight = vertexLightColors[v];
                vertexData[baseOffset + 8] = vertexLight.X;
                vertexData[baseOffset + 9] = vertexLight.Y;
                vertexData[baseOffset + 10] = vertexLight.Z;
                vertexData[baseOffset + 11] = vertexLight.W;
            }

            gb.Vao = _gl.GenVertexArray();
            _gl.BindVertexArray(gb.Vao);

            gb.Vbo = _gl.GenBuffer();
            _gl.BindBuffer(BufferTargetARB.ArrayBuffer, gb.Vbo);
            fixed (float* ptr = vertexData)
                _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexData.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

            gb.Ebo = _gl.GenBuffer();
            _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, gb.Ebo);
            var indices = group.Indices.ToArray();
            // Reverse triangle winding: WoW/D3D uses CW front faces, OpenGL uses CCW.
            // Swap v1↔v2 in each triangle to convert CW→CCW.
            for (int t = 0; t + 2 < indices.Length; t += 3)
                (indices[t + 1], indices[t + 2]) = (indices[t + 2], indices[t + 1]);
            fixed (ushort* ptr = indices)
                _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

            foreach (var batch in group.Batches)
            {
                ulong batchEnd = batch.FirstIndex + (ulong)batch.IndexCount;
                if (batch.IndexCount == 0 || batchEnd > (ulong)indices.Length)
                    continue;

                ushort[] batchIndices = new ushort[batch.IndexCount];
                Array.Copy(indices, (int)batch.FirstIndex, batchIndices, 0, batchIndices.Length);

                uint batchEbo = _gl.GenBuffer();
                _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, batchEbo);
                fixed (ushort* batchPtr = batchIndices)
                {
                    _gl.BufferData(
                        BufferTargetARB.ElementArrayBuffer,
                        (nuint)(batchIndices.Length * sizeof(ushort)),
                        batchPtr,
                        BufferUsageARB.StaticDraw);
                }

                gb.BatchEbos[(batch.FirstIndex, batch.IndexCount)] = batchEbo;
            }

            // Keep the full-group EBO attached for the no-batch fallback path.
            _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, gb.Ebo);

            uint stride = 12 * sizeof(float);
            _gl.EnableVertexAttribArray(0);
            _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
            _gl.EnableVertexAttribArray(1);
            _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
            _gl.EnableVertexAttribArray(2);
            _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));
            _gl.EnableVertexAttribArray(3);
            _gl.VertexAttribPointer(3, 4, VertexAttribPointerType.Float, false, stride, (void*)(8 * sizeof(float)));

            ConfigureGpuInstanceAttributes();

            _gl.BindVertexArray(0);

            gb.IndexCount = (uint)indices.Length;
            gb.VertexCount = (uint)group.Vertices.Count;
            _groups.Add(gb);
        }
    }

    private static Vector4[] BuildVertexLightColors(WmoV14ToV17Converter.WmoGroupData group)
    {
        int vertexCount = group.Vertices.Count;
        var vertexLightColors = new Vector4[vertexCount];
        if (vertexCount == 0)
            return vertexLightColors;

        if (TryCopyParsedVertexColors(group, vertexLightColors))
            return vertexLightColors;

        if (TrySampleVertexColorsFromLightmaps(group, vertexLightColors))
            return vertexLightColors;

        for (int i = 0; i < vertexCount; i++)
            vertexLightColors[i] = Vector4.One;

        return vertexLightColors;
    }

    private static bool TryCopyParsedVertexColors(WmoV14ToV17Converter.WmoGroupData group, Vector4[] vertexLightColors)
    {
        if (group.VertexColors.Count != vertexLightColors.Length || group.VertexColors.Count == 0)
            return false;

        double averageLuminosity = 0.0;
        foreach (uint packedColor in group.VertexColors)
        {
            byte blue = (byte)(packedColor & 0xFF);
            byte green = (byte)((packedColor >> 8) & 0xFF);
            byte red = (byte)((packedColor >> 16) & 0xFF);
            averageLuminosity += (red + green + blue) / 3.0;
        }

        averageLuminosity /= group.VertexColors.Count;
        if (averageLuminosity < 10.0)
            return false;

        for (int i = 0; i < vertexLightColors.Length; i++)
            vertexLightColors[i] = DecodePackedBgra(group.VertexColors[i]);

        return true;
    }

    private static bool TrySampleVertexColorsFromLightmaps(WmoV14ToV17Converter.WmoGroupData group, Vector4[] vertexLightColors)
    {
        if (group.LightmapData.Length == 0 || group.LightmapUVs.Count == 0 || group.LightmapInfos.Count == 0)
            return false;

        int vertexCount = vertexLightColors.Length;
        var redSums = new float[vertexCount];
        var greenSums = new float[vertexCount];
        var blueSums = new float[vertexCount];
        var sampleCounts = new int[vertexCount];
        int faceCount = group.Indices.Count / 3;
        bool hasSamples = false;

        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            var lightmapInfo = group.LightmapInfos[Math.Min(faceIndex, group.LightmapInfos.Count - 1)];
            if (lightmapInfo.Width == 0 || lightmapInfo.Height == 0)
                continue;

            for (int corner = 0; corner < 3; corner++)
            {
                int uvIndex = faceIndex * 3 + corner;
                int indexOffset = faceIndex * 3 + corner;
                if (uvIndex >= group.LightmapUVs.Count || indexOffset >= group.Indices.Count)
                    continue;

                int vertexIndex = group.Indices[indexOffset];
                if ((uint)vertexIndex >= (uint)vertexCount)
                    continue;

                Vector2 uv = group.LightmapUVs[uvIndex];
                if (!float.IsFinite(uv.X) || !float.IsFinite(uv.Y))
                    continue;

                float u = Math.Clamp(uv.X, 0f, 1f);
                float v = Math.Clamp(uv.Y, 0f, 1f);
                int pixelX = (int)(u * (lightmapInfo.Width - 1));
                int pixelY = (int)(v * (lightmapInfo.Height - 1));

                long pixelOffset = (long)lightmapInfo.DataOffset + (((long)pixelY * lightmapInfo.Width) + pixelX) * 4L;
                if (pixelOffset < 0 || pixelOffset + 4 > group.LightmapData.LongLength)
                    continue;

                int pixelOffsetInt = (int)pixelOffset;
                blueSums[vertexIndex] += group.LightmapData[pixelOffsetInt + 0] / 255f;
                greenSums[vertexIndex] += group.LightmapData[pixelOffsetInt + 1] / 255f;
                redSums[vertexIndex] += group.LightmapData[pixelOffsetInt + 2] / 255f;
                sampleCounts[vertexIndex]++;
                hasSamples = true;
            }
        }

        if (!hasSamples)
            return false;

        double averageLuminosity = 0.0;
        for (int i = 0; i < vertexCount; i++)
        {
            if (sampleCounts[i] > 0)
            {
                float invCount = 1f / sampleCounts[i];
                float red = redSums[i] * invCount;
                float green = greenSums[i] * invCount;
                float blue = blueSums[i] * invCount;
                vertexLightColors[i] = new Vector4(red, green, blue, 1f);
                averageLuminosity += (red + green + blue) / 3.0;
            }
            else
            {
                vertexLightColors[i] = Vector4.One;
                averageLuminosity += 1.0;
            }
        }

        averageLuminosity /= vertexCount;
        if (averageLuminosity < 0.08)
            return false;

        return true;
    }

    private static Vector4 DecodePackedBgra(uint packedColor)
    {
        float blue = (packedColor & 0xFF) / 255f;
        float green = ((packedColor >> 8) & 0xFF) / 255f;
        float red = ((packedColor >> 16) & 0xFF) / 255f;
        float alpha = ((packedColor >> 24) & 0xFF) / 255f;
        return new Vector4(red, green, blue, alpha > 0f ? alpha : 1f);
    }

    private void LoadMaterialTextures()
    {
        if (_dataSource == null) return;

        int loaded = 0, failed = 0;
        for (int i = 0; i < _wmo.Materials.Count; i++)
            TryLoadMaterialTexture(i, ref loaded, ref failed);

        ViewerLog.Trace($"[WmoRenderer] Textures: {loaded} loaded, {failed} failed out of {_wmo.Materials.Count} materials");
    }

    private void QueueDeferredMaterialTextureLoads()
    {
        _pendingMaterialTextureLoads.Clear();
        for (int i = 0; i < _wmo.Materials.Count; i++)
            _pendingMaterialTextureLoads.Enqueue(i);

        if (_pendingMaterialTextureLoads.Count > 0)
        {
            ViewerLog.Info(ViewerLog.Category.Wmo,
                $"[WMO-LOAD] Deferred {_pendingMaterialTextureLoads.Count} material textures for {_modelDir}");
        }
    }

    public int ProcessDeferredMaterialTextureLoads(
        int maxLoads = DeferredMaterialTextureLoadsPerFrame,
        double maxBudgetMs = DeferredMaterialTextureLoadBudgetMs)
    {
        if (!_deferInitialMaterialTextureLoads || _pendingMaterialTextureLoads.Count == 0 || _dataSource == null)
            return 0;

        if (maxLoads <= 0 || maxBudgetMs <= 0)
            return 0;

        var stopwatch = Stopwatch.StartNew();
        int loadsCompleted = 0;
        int loaded = 0, failed = 0;

        while (loadsCompleted < maxLoads
            && stopwatch.Elapsed.TotalMilliseconds < maxBudgetMs
            && _pendingMaterialTextureLoads.TryDequeue(out int materialIndex))
        {
            TryLoadMaterialTexture(materialIndex, ref loaded, ref failed);
            loadsCompleted++;
        }

        return loadsCompleted;
    }

    private void TryLoadMaterialTexture(int i, ref int loaded, ref int failed)
    {
        var mat = _wmo.Materials[i];
        string? texName = ResolveMaterialTextureName(mat);
        if (string.IsNullOrEmpty(texName))
            return;

        if (!texName.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
            texName += ".blp";

        byte[]? blpData = _dataSource?.ReadFile(texName);

        if (blpData == null)
            blpData = _dataSource?.ReadFile(texName.Replace('/', '\\'));

        if (blpData == null && _dataSource is MpqDataSource mpqDs)
        {
            var found = mpqDs.FindInFileSet(texName);
            if (found != null)
                blpData = _dataSource.ReadFile(found);
        }

        if (blpData != null && blpData.Length > 0)
        {
            uint glTex = LoadWmoTexture(blpData, texName);
            if (glTex != 0)
            {
                _materialTextures[i] = glTex;
                loaded++;
            }
            else
            {
                ViewerLog.Trace($"[WmoRenderer] Mat {i}: BLP decode failed for '{texName}'");
                failed++;
            }
        }
        else
        {
            ViewerLog.Trace($"[WmoRenderer] Mat {i}: texture not found '{texName}'");
            failed++;
        }
    }

    private int ResolveBatchMaterialId(WmoV14ToV17Converter.WmoGroupData group, WmoV14ToV17Converter.WmoBatch batch)
    {
        int originalMaterialId = batch.MaterialId;
        int materialId = originalMaterialId;
        if ((uint)materialId < (uint)_wmo.Materials.Count)
            return materialId;

        int firstFace = (int)(batch.FirstIndex / 3u);
        if ((uint)firstFace < (uint)group.FaceMaterials.Count)
        {
            int faceMaterial = group.FaceMaterials[firstFace];
            if ((uint)faceMaterial < (uint)_wmo.Materials.Count)
            {
                LogMaterialFallback(group, batch, originalMaterialId, faceMaterial, "MOPY");
                return faceMaterial;
            }
        }

        int defaultMaterial = _wmo.Materials.Count > 0 ? 0 : -1;
        LogMaterialFallback(group, batch, originalMaterialId, defaultMaterial, "DEFAULT");
        return defaultMaterial;
    }

    private void LogMaterialFallback(WmoV14ToV17Converter.WmoGroupData group, WmoV14ToV17Converter.WmoBatch batch,
        int originalMaterialId, int resolvedMaterialId, string source)
    {
        if (_materialFallbackLogCount >= MaxMaterialFallbackLogs)
            return;

        string groupName = string.IsNullOrWhiteSpace(group.Name) ? "<unnamed>" : group.Name;
        string key = $"{groupName}|{batch.FirstIndex}|{batch.IndexCount}|{originalMaterialId}|{resolvedMaterialId}|{source}";
        if (!_materialFallbackLogKeys.Add(key))
            return;

        _materialFallbackLogCount++;
        ViewerLog.Info(ViewerLog.Category.Wmo,
            $"[WMO-MAT] Fallback source={source} group='{groupName}' firstIndex={batch.FirstIndex} indexCount={batch.IndexCount} material {originalMaterialId} -> {resolvedMaterialId}");
    }

    private string? ResolveMaterialTextureName(WmoV14ToV17Converter.WmoMaterial material)
    {
        string? textureName = material.Texture1Name;
        if (!string.IsNullOrWhiteSpace(textureName))
            return textureName;

        if (_wmo.MotxRaw.Length == 0)
            return null;

        textureName = ResolveStringFromRaw(_wmo.MotxRaw, material.Texture1Offset);
        if (string.IsNullOrWhiteSpace(textureName) && material.Texture1Offset >= 8)
            textureName = ResolveStringFromRaw(_wmo.MotxRaw, material.Texture1Offset - 8);

        return textureName;
    }

    private static string? ResolveStringFromRaw(byte[] raw, uint offset)
    {
        if (raw.Length == 0 || offset >= raw.Length)
            return null;

        int start = (int)offset;
        int end = Array.IndexOf(raw, (byte)0, start);
        if (end < 0)
            end = raw.Length;
        if (end <= start)
            return null;

        return Encoding.ASCII.GetString(raw, start, end - start).Trim();
    }

    private unsafe uint LoadWmoTexture(byte[] blpData, string name)
    {
        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new SereniaBLPLib.BlpFile(ms);
            using var image = blp.GetImage(0);

            // ImageSharp Image<Rgba32> is already tightly-packed RGBA
            int w = image.Width, h = image.Height;
            var pixels = new byte[w * h * 4];
            image.CopyPixelDataTo(pixels);

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: true,
                TextureWrapMode.Repeat, TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            return tex;
        }
        catch (Exception ex)
        {
            ViewerLog.Trace($"[WmoRenderer] Failed to decode BLP {name}: {ex.Message}");
            return 0;
        }
    }

    private static List<Vector3> BuildRenderNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        if (group.Normals.Count == group.Vertices.Count && group.Normals.Count > 0)
        {
            var normalized = new List<Vector3>(group.Normals.Count);
            bool hasUsableNormal = false;

            for (int i = 0; i < group.Normals.Count; i++)
            {
                Vector3 n = group.Normals[i];
                if (!float.IsFinite(n.X) || !float.IsFinite(n.Y) || !float.IsFinite(n.Z))
                {
                    normalized.Add(Vector3.UnitY);
                    continue;
                }

                float lengthSq = n.LengthSquared();
                if (lengthSq > 1e-8f)
                {
                    normalized.Add(Vector3.Normalize(n));
                    hasUsableNormal = true;
                }
                else
                {
                    normalized.Add(Vector3.UnitY);
                }
            }

            if (hasUsableNormal)
                return normalized;
        }

        return GenerateNormals(group);
    }

    private static List<Vector3> GenerateNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            int i0 = group.Indices[i], i1 = group.Indices[i + 1], i2 = group.Indices[i + 2];
            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                continue;
            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var n = Vector3.Normalize(Vector3.Cross(e1, e2));
            if (float.IsNaN(n.X)) continue;
            normals[i0] += n;
            normals[i1] += n;
            normals[i2] += n;
        }
        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitY).ToList();
    }

    // --- Doodad loading ---

    private void ResolveDoodadNames()
    {
        // Parse null-terminated string table from DoodadNamesRaw
        // DoodadDef.NameIndex is a byte offset into this table
        _doodadNames.Clear();
        if (_wmo.DoodadNamesRaw.Length == 0) return;

        // Build offset→name map for quick lookup
        var raw = _wmo.DoodadNamesRaw;
        int start = 0;
        for (int i = 0; i <= raw.Length; i++)
        {
            if (i == raw.Length || raw[i] == 0)
            {
                if (i > start)
                {
                    // We don't store by index here — we'll resolve by offset in GetDoodadName
                }
                start = i + 1;
            }
        }
    }

    private string GetDoodadName(uint nameOffset)
    {
        if (nameOffset >= _wmo.DoodadNamesRaw.Length) return "";
        int end = (int)nameOffset;
        while (end < _wmo.DoodadNamesRaw.Length && _wmo.DoodadNamesRaw[end] != 0)
            end++;
        if (end == (int)nameOffset) return "";
        return Encoding.UTF8.GetString(_wmo.DoodadNamesRaw, (int)nameOffset, end - (int)nameOffset);
    }

    private void LoadActiveDoodadSet()
    {
        _doodadInstances.Clear();
        _pendingDoodadModelLoads.Clear();
        _queuedDoodadModelLoads.Clear();
        _doodadInstanceIndicesByModel.Clear();
        _doodadSourceModelPaths.Clear();

        if (_wmo.DoodadSets.Count == 0 || _wmo.DoodadDefs.Count == 0)
            return;

        if (_activeDoodadSet >= _wmo.DoodadSets.Count)
            _activeDoodadSet = 0;

        var set = _wmo.DoodadSets[_activeDoodadSet];
        ViewerLog.Trace($"[WmoRenderer] Loading DoodadSet [{_activeDoodadSet}] \"{set.Name}\": {set.Count} doodads (start={set.StartIndex}), DoodadDefs.Count={_wmo.DoodadDefs.Count}, DoodadNamesRaw.Length={_wmo.DoodadNamesRaw.Length}");

        int loaded = 0, failed = 0, emptyName = 0, notFound = 0, parseError = 0, deferredUniqueModels = 0;
        for (uint i = set.StartIndex; i < set.StartIndex + set.Count && i < (uint)_wmo.DoodadDefs.Count; i++)
        {
            var def = _wmo.DoodadDefs[(int)i];
            string modelPath = GetDoodadName(def.NameIndex);

            if (string.IsNullOrEmpty(modelPath))
            {
                emptyName++;
                failed++;
                continue;
            }

            // Build transform matrix: Scale * Rotation * Translation
            var transform = Matrix4x4.CreateScale(def.Scale)
                          * Matrix4x4.CreateFromQuaternion(def.Orientation)
                          * Matrix4x4.CreateTranslation(def.Position);

            string normalizedModelPath = NormalizeDoodadPath(modelPath).ToLowerInvariant();
            _doodadSourceModelPaths[normalizedModelPath] = modelPath;
            if (!_doodadInstanceIndicesByModel.TryGetValue(normalizedModelPath, out List<int>? instanceIndices))
            {
                instanceIndices = new List<int>();
                _doodadInstanceIndicesByModel[normalizedModelPath] = instanceIndices;
            }

            IModelRenderer? renderer = null;
            if (_deferInitialDoodadLoads)
            {
                if (_queuedDoodadModelLoads.Add(normalizedModelPath))
                {
                    _pendingDoodadModelLoads.Enqueue(normalizedModelPath);
                    deferredUniqueModels++;
                }
            }
            else
            {
                renderer = GetOrLoadDoodadModel(modelPath);
            }

            _doodadInstances.Add(new DoodadInstance
            {
                ModelPath = modelPath,
                NormalizedModelPath = normalizedModelPath,
                Renderer = renderer,
                Transform = transform,
                Visible = true,
                DoodadDefIndex = (int)i,
                LocalPosition = def.Position
            });
            instanceIndices.Add(_doodadInstances.Count - 1);

            if (_deferInitialDoodadLoads)
                continue;

            if (renderer != null)
                loaded++;
            else
            {
                failed++;
                if (_lastLoadResult == DoodadLoadResult.NotFound) notFound++;
                else if (_lastLoadResult == DoodadLoadResult.ParseError) parseError++;
            }
        }

        if (_deferInitialDoodadLoads)
        {
            ViewerLog.Trace($"[WmoRenderer] Doodads queued for deferred loading: {_doodadInstances.Count} instances, {deferredUniqueModels} unique models");
        }
        else
        {
            ViewerLog.Trace($"[WmoRenderer] Doodads: {loaded} loaded, {failed} failed ({emptyName} empty names, {notFound} not found, {parseError} parse errors), {_doodadModelCache.Count} unique models cached");
        }
    }

    public int ProcessDeferredDoodadLoads(
        int maxLoads = DefaultDeferredDoodadLoads,
        double maxBudgetMs = DefaultDeferredDoodadBudgetMs)
    {
        if (!_deferInitialDoodadLoads || _pendingDoodadModelLoads.Count == 0)
            return 0;

        if (maxLoads <= 0 || maxBudgetMs <= 0)
            return 0;

        var stopwatch = Stopwatch.StartNew();
        int loadsCompleted = 0;
        while (loadsCompleted < maxLoads
            && stopwatch.Elapsed.TotalMilliseconds < maxBudgetMs
            && _pendingDoodadModelLoads.TryDequeue(out string? normalizedModelPath))
        {
            _queuedDoodadModelLoads.Remove(normalizedModelPath);
            if (!_doodadInstanceIndicesByModel.TryGetValue(normalizedModelPath, out List<int>? indices) || indices.Count == 0)
                continue;

            string modelPath = _doodadSourceModelPaths.TryGetValue(normalizedModelPath, out string? sourceModelPath)
                ? sourceModelPath
                : _doodadInstances[indices[0]].ModelPath;
            IModelRenderer? renderer = GetOrLoadDoodadModel(modelPath);
            foreach (int idx in indices)
                _doodadInstances[idx].Renderer = renderer;

            loadsCompleted++;
        }

        return loadsCompleted;
    }

    private enum DoodadLoadResult { Loaded, NotFound, ParseError }
    private DoodadLoadResult _lastLoadResult;

    private IModelRenderer? GetOrLoadDoodadModel(string modelPath)
    {
        string normalized = NormalizeDoodadPath(modelPath).ToLowerInvariant();

        if (_doodadModelCache.TryGetValue(normalized, out var cached))
        {
            _lastLoadResult = cached != null ? DoodadLoadResult.Loaded : DoodadLoadResult.NotFound;
            return cached;
        }

        IModelRenderer? renderer = null;
        _lastLoadResult = DoodadLoadResult.NotFound;
        try
        {
            string resolvedModelPath;
            byte[]? modelData;
            string normalizedModelPath = NormalizeDoodadPath(modelPath);

            if (!TryReadPreferredClassicDoodadData(normalizedModelPath, out resolvedModelPath, out modelData))
            {
                resolvedModelPath = ResolveCanonicalDoodadPath(modelPath);
                modelData = ReadDoodadFileData(resolvedModelPath);
                if ((modelData == null || modelData.Length == 0) && !resolvedModelPath.Equals(modelPath, StringComparison.OrdinalIgnoreCase))
                    modelData = ReadDoodadFileData(modelPath);
            }

            if (modelData == null || modelData.Length == 0)
            {
                if (_doodadModelCache.Count < 30) // only log first 30 unique misses
                    ViewerLog.Trace($"  Doodad not found: {modelPath}");

                _doodadModelCache[normalized] = null;
                return null;
            }

            bool isM2Family = WarcraftNetM2Adapter.IsM2FamilyContainer(modelData);

            if (isM2Family)
            {
                renderer = LoadM2DoodadRenderer(modelPath, resolvedModelPath, modelData);
            }
            else
            {
                using var stream = new MemoryStream(modelData);
                var mdx = MdxFile.Load(stream);
                string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;
                renderer = new MdxRenderer(_gl, mdx, modelDir, _dataSource, _texResolver, resolvedModelPath);
            }

            if (renderer != null)
            {
                _lastLoadResult = DoodadLoadResult.Loaded;
                ViewerLog.Trace($"  Doodad loaded: {Path.GetFileName(modelPath)}");
            }
        }
        catch (Exception ex)
        {
            _lastLoadResult = DoodadLoadResult.ParseError;
            ViewerLog.Trace($"  Doodad load failed: {modelPath} — {ex.Message}");
        }

        _doodadModelCache[normalized] = renderer;
        return renderer;
    }

private IModelRenderer? LoadM2DoodadRenderer(string originalModelPath, string resolvedModelPath, byte[] modelData)
    {
        WarcraftNetM2Adapter.ValidateModelProfile(modelData, resolvedModelPath, _buildVersion);
        string buildProfileId = FormatProfileRegistry.ResolveModelProfile(_buildVersion)?.ProfileId ?? "unknown";

        var candidatePaths = new List<string>(WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath));
        string? bestSkinPath = ResolveBestSkinPath(resolvedModelPath);
        if (!string.IsNullOrWhiteSpace(bestSkinPath))
            candidatePaths.Add(bestSkinPath);

        Exception? lastSkinError = null;
        bool anySkinFound = false;

        foreach (string skinPath in candidatePaths.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            byte[]? skinBytes = ReadDoodadFileData(skinPath);
            if (skinBytes == null || skinBytes.Length == 0)
                continue;

            anySkinFound = true;

            try
            {
                ViewerLog.Trace($"[M2] Trying WMO doodad skin for {Path.GetFileName(originalModelPath)}: {skinPath} ({skinBytes.Length} bytes)");
                M2StaticRenderModel runtimeModel = WowViewerM2RuntimeBridge.BuildStaticRenderModel(modelData, skinBytes, resolvedModelPath, skinPath);
                MdxFile? adapted = null;
                try
                {
                    adapted = WarcraftNetM2Adapter.BuildRuntimeModel(modelData, skinBytes, resolvedModelPath, _buildVersion);
                }
                catch (Exception adapterEx)
                {
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] M2->MDX adapter fallback failed for {Path.GetFileName(resolvedModelPath)}: {adapterEx.Message} (native renderer will be used)");
                }

                var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterSkin, M2RouteType.AdapterSkin, skinPath);
                _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                M2RouteDiagnostics.LogRouteDecision(route);

                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Selected WMO doodad skin for {Path.GetFileName(originalModelPath)}: {skinPath} ({skinBytes.Length} bytes)");
                return WowViewerM2RuntimeBridge.CreateRenderer(
                    _gl,
                    runtimeModel,
                    adapted,
                    Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir,
                    _dataSource,
                    _texResolver,
                    _buildVersion,
                    resolvedModelPath);
            }
            catch (Exception ex)
            {
                lastSkinError = ex;
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] WMO doodad skin candidate failed for {Path.GetFileName(originalModelPath)}: {skinPath} ({ex.Message})");
            }
        }

        if (!anySkinFound)
        {
            if (WarcraftNetM2Adapter.SupportsEmbeddedNativeRoute(_buildVersion))
            {
                try
                {
                    M2StaticRenderModel runtimeModel = WarcraftNetM2Adapter.BuildEmbeddedStaticRenderModel(modelData, resolvedModelPath, _buildVersion);
                    var route = M2RouteDecision.Create(
                        originalModelPath,
                        buildProfileId,
                        M2RouteType.NativeEmbeddedProfile,
                        M2RouteType.NativeEmbeddedProfile,
                        fallbackReason: "No external .skin resolved for WMO doodad; using native embedded root-profile geometry");
                    _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                    M2RouteDiagnostics.LogRouteDecision(route);

                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Loaded native embedded root-profile geometry for WMO doodad {Path.GetFileName(originalModelPath)} after no external .skin resolved");
                    return WowViewerM2RuntimeBridge.CreateRenderer(
                        _gl,
                        runtimeModel,
                        adaptedMdx: null,
                        modelDir: Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir,
                        dataSource: _dataSource,
                        texResolver: _texResolver,
                        buildVersion: _buildVersion,
                        sourceModelPath: resolvedModelPath);
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Native embedded root-profile WMO doodad route failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }

            if (string.Equals(FormatProfileRegistry.ResolveModelProfile(_buildVersion)?.ProfileId, FormatProfileRegistry.M2Profile3018303.ProfileId, StringComparison.Ordinal))
            {
                try
                {
                    var adapted = WarcraftNetM2Adapter.BuildRuntimeModel(modelData, null, resolvedModelPath, _buildVersion);
                    string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;

                    var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterEmbeddedProfile, M2RouteType.AdapterEmbeddedProfile, fallbackReason: "No external .skin resolved for WMO doodad, using embedded root-profile");
                    _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                    M2RouteDiagnostics.LogRouteDecision(route);

                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Loaded embedded root-profile geometry for WMO doodad {Path.GetFileName(originalModelPath)} after no external .skin resolved");
                    return new M2Renderer(
                        new MdxRenderer(_gl, adapted, modelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion),
                        resolvedModelPath);
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Embedded root-profile WMO doodad fallback failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }

            M2Era1121EraTag detectedEra = M2ModelReaderDispatcher.DetectEra(modelData.AsSpan(), resolvedModelPath);
            if (detectedEra is M2Era1121EraTag.Md20_1X_V100 or M2Era1121EraTag.Md20_1X_V101)
            {
                try
                {
                    var adapted = WarcraftNetM2Adapter.BuildRuntimeModel(modelData, null, resolvedModelPath, _buildVersion);
                    string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;

                    var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterEmbeddedProfile, M2RouteType.AdapterEmbeddedProfile, fallbackReason: $"1.12.1 WMO doodad (era={detectedEra.ToDisplayString()}), no external .skin needed");
                    _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                    M2RouteDiagnostics.LogRouteDecision(route);

                    ViewerLog.Info(ViewerLog.Category.Mdx,
                        $"[M2] Loaded embedded 1.12.1 geometry for WMO doodad {Path.GetFileName(originalModelPath)} (era={detectedEra.ToDisplayString()})");
                    return new M2Renderer(
                        new MdxRenderer(_gl, adapted, modelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion),
                        resolvedModelPath);
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Embedded 1.12.1 WMO doodad fallback failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }

            if (_loggedMissingDoodadSkinPaths.Add(resolvedModelPath))
                ViewerLog.Important(ViewerLog.Category.Mdx, $"[M2] Missing WMO doodad .skin for: {Path.GetFileName(originalModelPath)}");
        }

        if (WarcraftNetM2Adapter.IsMd20(modelData))
        {
            byte[]? convertedBytes = ConvertM2ToMdx(modelData, resolvedModelPath);
            if (convertedBytes != null && convertedBytes.Length > 0)
            {
                try
                {
                    using var convertedStream = new MemoryStream(convertedBytes);
                    var convertedMdx = MdxFile.Load(convertedStream);
                    if (WarcraftNetM2Adapter.HasRenderableGeometry(convertedMdx))
                    {
                        string modelDir = Path.GetDirectoryName(resolvedModelPath)?.Replace('/', '\\') ?? _modelDir;

                        var route = M2RouteDecision.Create(originalModelPath, buildProfileId, M2RouteType.AdapterSkin, M2RouteType.ConversionFallback, fallbackReason: "Adapter/skin path failed for WMO doodad, fell back to M2->MDX conversion");
                        _doodadRouteDecisions[NormalizeDoodadPath(originalModelPath)] = route;
                        M2RouteDiagnostics.LogRouteDecision(route);

                        ViewerLog.Info(ViewerLog.Category.Mdx,
                            $"[M2] Falling back to M2->MDX conversion for WMO doodad {Path.GetFileName(originalModelPath)} after adapter failure");
                        return new M2Renderer(
                            new MdxRenderer(_gl, convertedMdx, modelDir, _dataSource, _texResolver, resolvedModelPath, true, _buildVersion),
                            resolvedModelPath);
                    }

                    lastSkinError = new InvalidDataException(
                        $"M2->MDX fallback produced no renderable geometry for WMO doodad {Path.GetFileName(originalModelPath)} ({WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)})");
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Rejecting converted WMO doodad fallback for {Path.GetFileName(originalModelPath)}: {WarcraftNetM2Adapter.SummarizeGeometry(convertedMdx)}");
                }
                catch (Exception ex)
                {
                    lastSkinError = ex;
                    ViewerLog.Debug(ViewerLog.Category.Mdx,
                        $"[M2] Converted WMO doodad fallback load failed for {Path.GetFileName(originalModelPath)}: {ex.Message}");
                }
            }
        }

        if (lastSkinError != null)
            throw new InvalidDataException($"All .skin candidates failed for WMO doodad M2: {Path.GetFileName(originalModelPath)}", lastSkinError);

        return null;
    }

    private byte[]? ConvertM2ToMdx(byte[] modelData, string resolvedModelPath)
    {
        try
        {
            byte[]? skinBytes = null;
            foreach (string skinPath in WarcraftNetM2Adapter.BuildSkinCandidates(resolvedModelPath).Distinct(StringComparer.OrdinalIgnoreCase))
            {
                skinBytes = ReadDoodadFileData(skinPath);
                if (skinBytes != null && skinBytes.Length > 0)
                    break;
            }

            var converter = new WoWViewer.Transfer.M2ToMdxConverter();
            return converter.ConvertToBytes(modelData, skinBytes, _buildVersion);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] WMO doodad M2->MDX converter fallback failed for {Path.GetFileName(resolvedModelPath)}: {ex.Message}");
            return null;
        }
    }

    private string ResolveCanonicalDoodadPath(string modelPath)
    {
        string normalizedPath = NormalizeDoodadPath(modelPath);
        if (_canonicalDoodadPathCache.TryGetValue(normalizedPath, out string? cachedPath))
            return cachedPath;

        string resolvedPath = normalizedPath;
        if (_dataSource is MpqDataSource mpqDataSource)
        {
            string? found = mpqDataSource.FindInFileSet(normalizedPath);
            if (!string.IsNullOrWhiteSpace(found))
            {
                resolvedPath = NormalizeDoodadPath(found);
            }
            else
            {
                foreach (string alternatePath in EnumerateAlternateDoodadPaths(normalizedPath))
                {
                    found = mpqDataSource.FindInFileSet(alternatePath);
                    if (string.IsNullOrWhiteSpace(found))
                        continue;

                    resolvedPath = NormalizeDoodadPath(found);
                    break;
                }
            }
        }

        _canonicalDoodadPathCache[normalizedPath] = resolvedPath;
        return resolvedPath;
    }

    private string? ResolveBestSkinPath(string resolvedModelPath)
    {
        if (_bestSkinPathCache.TryGetValue(resolvedModelPath, out string? cachedPath))
            return cachedPath;

        string? bestSkinPath = WarcraftNetM2Adapter.FindSkinInFileList(
            resolvedModelPath,
            _dataSource?.GetFileList(".skin") ?? Array.Empty<string>());

        _bestSkinPathCache[resolvedModelPath] = bestSkinPath;
        return bestSkinPath;
    }

    private byte[]? ReadDoodadFileData(string path)
    {
        string normalizedPath = NormalizeDoodadPath(path);

        if (_dataSource != null)
        {
            byte[]? data = _dataSource.ReadFile(path);
            if ((data == null || data.Length == 0) && !normalizedPath.Equals(path, StringComparison.OrdinalIgnoreCase))
                data = _dataSource.ReadFile(normalizedPath);

            if ((data == null || data.Length == 0) && _dataSource is MpqDataSource mpqDataSource)
            {
                string? found = mpqDataSource.FindInFileSet(normalizedPath);
                if (!string.IsNullOrWhiteSpace(found))
                    data = _dataSource.ReadFile(found);
            }

            if (data != null && data.Length > 0)
                return data;
        }

        string diskPath = path;
        if (!Path.IsPathRooted(diskPath))
            diskPath = Path.Combine(_modelDir, normalizedPath);

        if (File.Exists(diskPath))
            return File.ReadAllBytes(diskPath);

        string fallbackPath = Path.Combine(_modelDir, Path.GetFileName(normalizedPath));
        if (!fallbackPath.Equals(diskPath, StringComparison.OrdinalIgnoreCase) && File.Exists(fallbackPath))
            return File.ReadAllBytes(fallbackPath);

        return null;
    }

    private static string NormalizeDoodadPath(string path)
    {
        return path.Replace('/', '\\');
    }

    private static bool IsClassicDoodadRequest(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase);
    }

    private bool TryReadPreferredClassicDoodadData(string normalizedPath, out string resolvedPath, out byte[]? data)
    {
        resolvedPath = normalizedPath;
        data = null;

        if (!IsClassicDoodadRequest(normalizedPath))
            return false;

        foreach (string candidate in EnumeratePreferredClassicDoodadPaths(normalizedPath))
        {
            data = ReadDoodadFileData(candidate);
            if (data == null || data.Length == 0)
                continue;

            resolvedPath = NormalizeDoodadPath(candidate);
            return true;
        }

        data = null;
        resolvedPath = normalizedPath;
        return false;
    }

    private static IEnumerable<string> EnumeratePreferredClassicDoodadPaths(string normalizedPath)
    {
        yield return normalizedPath;

        if (normalizedPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return normalizedPath[..^4] + ".mdl";
            yield break;
        }

        if (normalizedPath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
            yield return normalizedPath[..^4] + ".mdx";
    }

    private static IEnumerable<string> EnumerateAlternateDoodadPaths(string normalizedPath)
    {
        if (normalizedPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return normalizedPath[..^4] + ".m2";
            yield return normalizedPath[..^4] + ".mdl";
            yield break;
        }

        if (normalizedPath.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return normalizedPath[..^4] + ".mdx";
            yield return normalizedPath[..^4] + ".m2";
            yield break;
        }

        if (normalizedPath.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
            yield return normalizedPath[..^3] + ".mdx";
    }

    private void InitLiquidShader()
    {
        _liquidShaderRefCount++;
        if (_liquidShader != 0) return; // Already initialized by another instance

        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = uProj * uView * worldPos;
}
";
        string fragSrc = @"
#version 330 core
in vec3 vWorldPos;

uniform vec4 uColor;

out vec4 FragColor;

void main() {
    // Simple semi-transparent liquid with slight depth variation
    float depthShade = 0.85 + 0.15 * sin(vWorldPos.x * 0.5 + vWorldPos.y * 0.5);
    FragColor = vec4(uColor.rgb * depthShade, uColor.a);
}
";
        uint vert = CompileShader(ShaderType.VertexShader, vertSrc);
        uint frag = CompileShader(ShaderType.FragmentShader, fragSrc);

        _liquidShader = _gl.CreateProgram();
        _gl.AttachShader(_liquidShader, vert);
        _gl.AttachShader(_liquidShader, frag);
        _gl.LinkProgram(_liquidShader);

        _gl.GetProgram(_liquidShader, ProgramPropertyARB.LinkStatus, out int status);
        if (status == 0)
            ViewerLog.Trace($"[WmoRenderer] Liquid shader link error: {_gl.GetProgramInfoLog(_liquidShader)}");

        _gl.DeleteShader(vert);
        _gl.DeleteShader(frag);

        _gl.UseProgram(_liquidShader);
        _uLiqModel = _gl.GetUniformLocation(_liquidShader, "uModel");
        _uLiqView = _gl.GetUniformLocation(_liquidShader, "uView");
        _uLiqProj = _gl.GetUniformLocation(_liquidShader, "uProj");
        _uLiqColor = _gl.GetUniformLocation(_liquidShader, "uColor");
    }

    private unsafe void BuildLiquidMeshes()
    {
        _liquidMeshes.Clear();

        for (int gi = 0; gi < _wmo.Groups.Count; gi++)
        {
            var group = _wmo.Groups[gi];
            if (group.LiquidData == null || group.LiquidData.Length < 30)
                continue;

            try
            {
                using var ms = new MemoryStream(group.LiquidData);
                using var reader = new BinaryReader(ms);

                // MLIQ header: C2iVector verts(8), C2iVector tiles(8), C3Vector corner(12), uint16 matId(2) = 30 bytes
                int xverts = reader.ReadInt32();
                int yverts = reader.ReadInt32();
                int xtiles = reader.ReadInt32();
                int ytiles = reader.ReadInt32();
                float cornerX = reader.ReadSingle();
                float cornerY = reader.ReadSingle();
                float cornerZ = reader.ReadSingle();
                ushort matId = reader.ReadUInt16();


                if (xverts <= 0 || yverts <= 0 || xverts > 256 || yverts > 256)
                {
                    ViewerLog.Trace($"[WmoRenderer] MLIQ group {gi}: invalid dimensions {xverts}x{yverts}, skipping");
                    continue;
                }

                int expectedVertBytes = xverts * yverts * 8;
                int expectedTileBytes = xtiles * ytiles;
                int totalExpected = 30 + expectedVertBytes + expectedTileBytes;
                if (ms.Length - ms.Position < expectedVertBytes)
                {
                    ViewerLog.Trace($"[WmoRenderer] MLIQ group {gi}: not enough data for {xverts}x{yverts} verts (need {expectedVertBytes}, have {ms.Length - ms.Position}), totalExpected={totalExpected} vs dataLen={group.LiquidData.Length}");
                    continue;
                }

                // Read vertex heights (8 bytes per vertex: 4 bytes flow data + 4 bytes float height)
                float[] heights = new float[xverts * yverts];
                for (int v = 0; v < xverts * yverts; v++)
                {
                    reader.ReadInt32(); // flow/filler data (skip)
                    heights[v] = reader.ReadSingle();
                }

                // Read tile flags (1 byte per tile) — check for visible tiles
                byte[] tileFlags = new byte[xtiles * ytiles];
                if (ms.Length - ms.Position >= expectedTileBytes)
                {
                    for (int t = 0; t < xtiles * ytiles; t++)
                        tileFlags[t] = reader.ReadByte();
                }

                // WMO MLIQ tile size = 1/8th of a map chunk = UNIT_SIZE/2 ≈ 4.16666
                float liquidTileSize = 4.16666f;

                // Build vertex positions in WMO-local space (raw file coords, Z-up).
                // Auto-fit the liquid quad to the owning group's bounds, then apply
                // any known build baseline plus the user-selected adjustment.
                int liquidOrientation = SelectBestLiquidOrientation(group, cornerX, cornerY, xverts, yverts, liquidTileSize);
                int baselineRotation = GetBaselineMliqRotationQuarterTurns();
                int effectiveOrientation = (liquidOrientation + baselineRotation + _mliqRotationQuarterTurns) & 3;
                int nverts = xverts * yverts;
                var vertices = new float[nverts * 3];
                for (int j = 0; j < yverts; j++)
                {
                    for (int i = 0; i < xverts; i++)
                    {
                        int idx = j * xverts + i;
                        var p = MapLiquidVertex(effectiveOrientation, cornerX, cornerY, liquidTileSize, i, j);
                        vertices[idx * 3 + 0] = p.X;
                        vertices[idx * 3 + 1] = p.Y;
                        vertices[idx * 3 + 2] = heights[idx];
                    }
                }

                if (liquidOrientation != 2 || baselineRotation != 0 || _mliqRotationQuarterTurns != 0)
                {
                    ViewerLog.Trace($"[WmoRenderer] MLIQ group {gi}: orientation={effectiveOrientation} (auto={liquidOrientation}, baselineRot={baselineRotation * 90}°, userRot={_mliqRotationQuarterTurns * 90}°)");
                }

                // Build indices: one quad per visible tile
                // Per 0.8.0 Ghidra spec: (tileByte & 0x0F) == 0x0F means no liquid at tile
                var indices = new List<ushort>();
                for (int j = 0; j < ytiles; j++)
                {
                    for (int i = 0; i < xtiles; i++)
                    {
                        int tileIdx = j * xtiles + i;
                        if (tileIdx >= tileFlags.Length) continue;
                        if ((tileFlags[tileIdx] & 0x0F) == 0x0F)
                            continue; // no liquid at this tile

                        ushort p = (ushort)(j * xverts + i);
                        ushort tl = p;
                        ushort tr = (ushort)(p + 1);
                        ushort bl = (ushort)(p + xverts);
                        ushort br = (ushort)(p + xverts + 1);

                        // Two triangles per quad (same winding as noggit)
                        indices.Add(tl); indices.Add(tr); indices.Add(br);
                        indices.Add(br); indices.Add(bl); indices.Add(tl);
                    }
                }

                if (indices.Count == 0)
                {
                    ViewerLog.Trace($"[WmoRenderer] MLIQ group {gi}: no visible tiles");
                    continue;
                }

                // Determine liquid type from per-tile nibble (primary) and MOGP flags (hint).
                // Per 0.8.0 Ghidra spec (FUN_006c0740 / FUN_006ae130):
                //   Runtime returns first non-0x0F tile nibble, then dispatches:
                //     nibble 0/4/8 → water renderer
                //     nibble 2/3/6/7 → magma/slime renderer
                // For our basic type mapping: water=0, ocean=1, magma=2, slime=3
                bool isOcean = (group.Flags & 0x80000) != 0;
                int liquidBasicType = 0; // default water

                // Sample first visible tile nibble for liquid type dispatch
                for (int t = 0; t < tileFlags.Length; t++)
                {
                    int nibble = tileFlags[t] & 0x0F;
                    if (nibble == 0x0F) continue; // empty tile
                    // Map nibble to basic type per 0.8.0 dispatch table
                    switch (nibble)
                    {
                        case 0: case 4: case 8:
                            liquidBasicType = 0; // water
                            break;
                        case 2: case 6:
                            liquidBasicType = 2; // magma
                            break;
                        case 3: case 7:
                            liquidBasicType = 3; // slime
                            break;
                        default:
                            liquidBasicType = 0; // unknown nibble → water fallback
                            break;
                    }
                    break; // use first visible tile
                }

                // Ocean flag override
                if (isOcean && liquidBasicType == 0) liquidBasicType = 1;

                // Assign color based on liquid type
                float cr, cg, cb, ca;
                switch (liquidBasicType)
                {
                    case 1: // ocean
                        cr = 0.10f; cg = 0.25f; cb = 0.55f; ca = 0.60f;
                        break;
                    case 2: // magma/lava
                        cr = 0.85f; cg = 0.25f; cb = 0.05f; ca = 0.70f;
                        break;
                    case 3: // slime
                        cr = 0.20f; cg = 0.65f; cb = 0.10f; ca = 0.65f;
                        break;
                    default: // water
                        cr = 0.15f; cg = 0.35f; cb = 0.65f; ca = 0.55f;
                        break;
                }
                string liquidTypeName = liquidBasicType switch { 1 => "ocean", 2 => "magma", 3 => "slime", _ => "water" };

                // Upload to GPU
                uint vao = _gl.GenVertexArray();
                uint vbo = _gl.GenBuffer();
                uint ebo = _gl.GenBuffer();

                _gl.BindVertexArray(vao);

                _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
                fixed (float* ptr = vertices)
                    _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

                _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
                var indexArr = indices.ToArray();
                fixed (ushort* ptr = indexArr)
                    _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indexArr.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

                _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), (void*)0);
                _gl.EnableVertexAttribArray(0);
                _gl.BindVertexArray(0);

                _liquidMeshes.Add(new LiquidMeshData
                {
                    GroupIndex = gi,
                    Vao = vao, Vbo = vbo, Ebo = ebo,
                    IndexCount = (uint)indexArr.Length,
                    ColorR = cr, ColorG = cg, ColorB = cb, ColorA = ca
                });

                ViewerLog.Trace($"[WmoRenderer] MLIQ group {gi}: {xverts}x{yverts} verts, {xtiles}x{ytiles} tiles, {indices.Count / 3} tris, corner=({cornerX:F1},{cornerY:F1},{cornerZ:F1}), type={liquidTypeName}, groupLiquid={group.GroupLiquid}, matId={matId}");
            }
            catch (Exception ex)
            {
                ViewerLog.Trace($"[WmoRenderer] MLIQ group {gi}: parse error — {ex.Message}");
            }
        }

        if (_liquidMeshes.Count > 0)
            ViewerLog.Trace($"[WmoRenderer] Built {_liquidMeshes.Count} liquid meshes");

        _builtMliqRotationRevision = _mliqRotationRevision;
    }

    private void EnsureLiquidMeshesUpToDate()
    {
        if (_builtMliqRotationRevision == _mliqRotationRevision)
            return;

        DisposeLiquidMeshes();
        BuildLiquidMeshes();
    }

    private void DisposeLiquidMeshes()
    {
        foreach (var liq in _liquidMeshes)
        {
            _gl.DeleteVertexArray(liq.Vao);
            _gl.DeleteBuffer(liq.Vbo);
            _gl.DeleteBuffer(liq.Ebo);
        }

        _liquidMeshes.Clear();
    }

    private static Vector2 MapLiquidVertex(int orientation, float cornerX, float cornerY, float tileSize, int i, int j)
    {
        return orientation switch
        {
            // No rotation
            0 => new Vector2(cornerX + i * tileSize, cornerY + j * tileSize),
            // 90° CW
            1 => new Vector2(cornerX + j * tileSize, cornerY - i * tileSize),
            // 90° CCW (legacy behavior)
            2 => new Vector2(cornerX - j * tileSize, cornerY + i * tileSize),
            // 180°
            3 => new Vector2(cornerX - i * tileSize, cornerY - j * tileSize),
            _ => new Vector2(cornerX - j * tileSize, cornerY + i * tileSize)
        };
    }

    private static int SelectBestLiquidOrientation(
        WmoV14ToV17Converter.WmoGroupData group,
        float cornerX,
        float cornerY,
        int xverts,
        int yverts,
        float tileSize)
    {
        int maxI = Math.Max(0, xverts - 1);
        int maxJ = Math.Max(0, yverts - 1);

        var groupMin = group.BoundsMin;
        var groupMax = group.BoundsMax;
        float groupCenterX = (groupMin.X + groupMax.X) * 0.5f;
        float groupCenterY = (groupMin.Y + groupMax.Y) * 0.5f;

        // Keep legacy mapping as tie-break default.
        int bestOrientation = 2;
        float bestScore = float.MaxValue;

        for (int orientation = 0; orientation < 4; orientation++)
        {
            var p00 = MapLiquidVertex(orientation, cornerX, cornerY, tileSize, 0, 0);
            var p10 = MapLiquidVertex(orientation, cornerX, cornerY, tileSize, maxI, 0);
            var p01 = MapLiquidVertex(orientation, cornerX, cornerY, tileSize, 0, maxJ);
            var p11 = MapLiquidVertex(orientation, cornerX, cornerY, tileSize, maxI, maxJ);

            float minX = MathF.Min(MathF.Min(p00.X, p10.X), MathF.Min(p01.X, p11.X));
            float maxX = MathF.Max(MathF.Max(p00.X, p10.X), MathF.Max(p01.X, p11.X));
            float minY = MathF.Min(MathF.Min(p00.Y, p10.Y), MathF.Min(p01.Y, p11.Y));
            float maxY = MathF.Max(MathF.Max(p00.Y, p10.Y), MathF.Max(p01.Y, p11.Y));

            float overflow = 0f;
            if (minX < groupMin.X) overflow += groupMin.X - minX;
            if (maxX > groupMax.X) overflow += maxX - groupMax.X;
            if (minY < groupMin.Y) overflow += groupMin.Y - minY;
            if (maxY > groupMax.Y) overflow += maxY - groupMax.Y;

            float centerX = (minX + maxX) * 0.5f;
            float centerY = (minY + maxY) * 0.5f;
            float centerDx = centerX - groupCenterX;
            float centerDy = centerY - groupCenterY;
            float centerDistance = MathF.Sqrt(centerDx * centerDx + centerDy * centerDy);

            // Prioritize staying inside group bounds, then center proximity.
            float score = overflow * 1000f + centerDistance;

            if (orientation == bestOrientation)
            {
                bestScore = score;
                continue;
            }

            if (score + 0.001f < bestScore)
            {
                bestScore = score;
                bestOrientation = orientation;
            }
        }

        return bestOrientation;
    }

    public void Dispose()
    {
        foreach (var gb in _groups)
        {
            _gl.DeleteVertexArray(gb.Vao);
            _gl.DeleteBuffer(gb.Vbo);
            _gl.DeleteBuffer(gb.Ebo);
            foreach (uint batchEbo in gb.BatchEbos.Values)
                _gl.DeleteBuffer(batchEbo);
        }

        if (_gpuInstanceVbo != 0)
        {
            _gl.DeleteBuffer(_gpuInstanceVbo);
            _gpuInstanceVbo = 0;
        }

        // Delete material textures
        foreach (var tex in _materialTextures.Values)
            _gl.DeleteTexture(tex);
        _materialTextures.Clear();

        // Dispose liquid meshes
        DisposeLiquidMeshes();

        // Dispose cached doodad renderers
        foreach (var renderer in _doodadModelCache.Values)
            renderer?.Dispose();
        _doodadModelCache.Clear();
        _doodadInstances.Clear();
        _loggedMissingDoodadSkinPaths.Clear();

        _shaderRefCount--;
        if (_shaderRefCount <= 0 && _shaderProgram != 0)
        {
            _gl.DeleteProgram(_shaderProgram);
            _shaderProgram = 0;
            _shaderRefCount = 0;
        }

        _liquidShaderRefCount--;
        if (_liquidShaderRefCount <= 0 && _liquidShader != 0)
        {
            _gl.DeleteProgram(_liquidShader);
            _liquidShader = 0;
            _liquidShaderRefCount = 0;
        }
    }

    private class GroupBuffers
    {
        public int GroupIndex;
        public Vector3 GroupCenter;
        public uint Vao, Vbo, Ebo;
        public uint IndexCount, VertexCount;
        public Dictionary<(uint FirstIndex, ushort IndexCount), uint> BatchEbos { get; } = new();
        public bool ManualVisible = true;
        public bool RuntimeVisible = true;
        public bool IsVisible => ManualVisible && RuntimeVisible;
    }

    private class DoodadInstance
    {
        public string ModelPath = "";
        public string NormalizedModelPath = "";
        public IModelRenderer? Renderer;
        public Matrix4x4 Transform;
        public bool Visible = true;
        public int DoodadDefIndex;
        public Vector3 LocalPosition; // WMO-local position for fast culling
    }

    private class LiquidMeshData
    {
        public int GroupIndex;
        public uint Vao, Vbo, Ebo;
        public uint IndexCount;
        public float ColorR, ColorG, ColorB, ColorA;
    }
}
