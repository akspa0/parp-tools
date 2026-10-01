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

/// <summary>
/// Renders a WMO (World Map Object) using OpenGL.
/// Uses WowViewer.Core.IO.Converters' WmoV14Data model for geometry.
/// Supports loading and rendering MDX doodads from DoodadSets.
/// </summary>
public class WmoRenderer : ISceneRenderer, IGpuInstancedWmoRenderer, ISceneLightEmitter
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
    private readonly bool _isCollisionWall;

    public static float GlobalOpacity { get; set; } = 1.0f;
    public static bool TexturesEnabled { get; set; } = true;
    public static float CollisionWallOpacity { get; set; } = 1.0f;
    public static bool CollisionWallTexturesEnabled { get; set; } = true;

    public bool IsCollisionWall => _isCollisionWall;
    public float GetEffectiveOpacity() => _isCollisionWall ? Math.Clamp(GlobalOpacity * CollisionWallOpacity, 0f, 1f) : Math.Clamp(GlobalOpacity, 0f, 1f);
    public bool AreTexturesEffective() => TexturesEnabled && (!_isCollisionWall || CollisionWallTexturesEnabled);

    private readonly WmoLiquidRenderer _liquidRenderer;
    private readonly WmoMaterialManager _materialManager;
    private readonly WmoDoodadController _doodadController;

    // Shared static shader program — prevents race condition when multiple WmoRenderers
    // exist and one is disposed (same fix as MdxRenderer)
    private static uint _shaderProgram;
    private static int _uModel, _uView, _uProj, _uHasTexture, _uUnlit, _uColor, _uAlphaTest, _uOpacity;
    private static int _uFogColor, _uFogStart, _uFogEnd, _uCameraPos;
    private static int _uLightDir, _uLightColor, _uAmbientColor;
    private const int MaxWmoLocalLights = SceneLightManager.MaxShaderLights;
    private static int _uLocalLightCount;
    private static readonly int[] _uLocalLightPos = new int[MaxWmoLocalLights];
    private static readonly int[] _uLocalLightColor = new int[MaxWmoLocalLights];
    private static readonly int[] _uLocalLightIntensity = new int[MaxWmoLocalLights];
    private static readonly int[] _uLocalLightStart = new int[MaxWmoLocalLights];
    private static readonly int[] _uLocalLightEnd = new int[MaxWmoLocalLights];
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
    private readonly SceneLight[] _localLightUploadScratch = new SceneLight[MaxWmoLocalLights];
    private bool _wireframe;

    private readonly HashSet<string> _invalidBatchRangeLogKeys = new(StringComparer.OrdinalIgnoreCase);
    private int _invalidBatchRangeLogCount;
    private const int MaxInvalidBatchRangeLogs = 100;

    internal WmoDoodadModelShare? DoodadModelShare
    {
        get => _doodadController.DoodadModelShare;
        init => _doodadController.DoodadModelShare = value;
    }

    public bool DoodadsVisible
    {
        get => _doodadController.DoodadsVisible;
        set => _doodadController.DoodadsVisible = value;
    }

    public bool RuntimeDoodadsVisible => _doodadController.RuntimeDoodadsVisible;

    private bool _runtimeGroupLiquidsVisible = true;
    private int _currentDrawCalls;
    private int _currentBatchDrawCalls;
    private int _currentOpaqueBatchInstanceCount;
    private int _currentGroupFallbackDrawCalls;
    private int _currentLiquidDrawCalls;
    private int _currentDoodadSubmissions;
    private int _currentVisibleGroupSubmissions;
    private int _currentVisibleLiquidMeshes;
    private const int ExteriorPortalTraversalDepth = 1;
    private const int InteriorPortalTraversalDepth = 4;

    public int PendingDoodadModelLoadCount => _doodadController.PendingDoodadModelLoadsCount;
    public int PendingMaterialTextureLoadCount => _materialManager.PendingTextureLoadsCount;
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
        => !_wireframe
            && _groups.Count > 0
            && _wmo.Portals.Count == 0
            && _groups.All(static group => group.ManualVisible);

    public M2RouteDecision? GetDoodadRouteDecision(string normalizedPath)
        => _doodadController.GetDoodadRouteDecision(normalizedPath);

    public static int MliqRotationQuarterTurns
    {
        get => WmoLiquidRenderer.MliqRotationQuarterTurns;
        set => WmoLiquidRenderer.MliqRotationQuarterTurns = value;
    }

    public int LiquidMeshCount => _liquidRenderer.LiquidMeshCount;

    public WmoRenderer(GL gl, WmoV14ToV17Converter.WmoV14Data wmo, string modelDir,
        IDataSource? dataSource = null, ReplaceableTextureResolver? texResolver = null, string? buildVersion = null,
        bool deferInitialDoodadLoads = false, bool deferInitialMaterialTextureLoads = false,
        bool enableRuntimeGroupVisibility = true)
    {
        var initStopwatch = Stopwatch.StartNew();
        _gl = gl;
        _wmo = wmo;
        _modelDir = modelDir;
        _isCollisionWall = modelDir.Contains("collisionwall", StringComparison.OrdinalIgnoreCase) || modelDir.Contains("collision_wall", StringComparison.OrdinalIgnoreCase);
        _dataSource = dataSource;
        _texResolver = texResolver;
        _buildVersion = buildVersion;
        _deferInitialDoodadLoads = deferInitialDoodadLoads;
        _deferInitialMaterialTextureLoads = deferInitialMaterialTextureLoads;
        _enableRuntimeGroupVisibility = enableRuntimeGroupVisibility;

        _liquidRenderer = new WmoLiquidRenderer(_gl, _wmo, _buildVersion);
        _materialManager = new WmoMaterialManager(_gl, _wmo, _modelDir, _dataSource, _deferInitialMaterialTextureLoads);
        _doodadController = new WmoDoodadController(_gl, _wmo, _modelDir, _dataSource, _texResolver, _buildVersion, _deferInitialDoodadLoads);

        _portalVisibilityGroups = BuildPortalVisibilityGroups();
        _portalVisibilityPortals = BuildPortalVisibilityPortals();

        _runtimeVisibleGroups = new bool[_wmo.Groups.Count];
        _frustumVisibleScratch = new bool[_wmo.Groups.Count];
        _portalVisibleScratch = new bool[_wmo.Groups.Count];

        InitShaders();
        InitBuffers();

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
    public uint WmoId => _wmo.WmoId;

    /// <summary>Whether this model's own MOLT lights feed the scene light set.</summary>
    public bool EmitsSceneLights => _wmo.Lights.Count > 0;

    /// <summary>World-space AABB of the model placed with <paramref name="modelMatrix"/>.</summary>
    public void GetWorldBounds(Matrix4x4 modelMatrix, out Vector3 worldMin, out Vector3 worldMax)
        => TransformAabb(BoundsMin, BoundsMax, modelMatrix, out worldMin, out worldMax);

    public void CollectSceneLights(Matrix4x4 modelMatrix, ICollection<SceneLight> lights, string sourceKey)
    {
        ArgumentNullException.ThrowIfNull(lights);

        for (int i = 0; i < _wmo.Lights.Count; i++)
        {
            WmoV14ToV17Converter.WmoLight light = _wmo.Lights[i];
            Vector4 decodedColor = WmoGeometryHelper.DecodePackedBgra(light.Color);
            float intensity = Math.Clamp(FiniteOrDefault(light.Intensity, 0.0f), 0.0f, 4.0f);
            float start = Math.Clamp(FiniteOrDefault(light.AttenStart, 0.0f), 0.0f, 100000.0f);
            float end = Math.Clamp(FiniteOrDefault(light.AttenEnd, 0.0f), 0.0f, 100000.0f);

            if (end <= start)
                end = start + 0.001f;

            lights.Add(new SceneLight(
                Vector3.Transform(light.Position, modelMatrix),
                new Vector3(decodedColor.X, decodedColor.Y, decodedColor.Z),
                intensity,
                start,
                end,
                "WMO-MOLT",
                sourceKey));
        }

        _doodadController.CollectSceneLights(modelMatrix, lights, sourceKey);
    }

    public uint GetRenderGroupAreaId(int renderGroupIndex)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
            return 0;
        int groupIndex = _groups[renderGroupIndex].GroupIndex;
        return groupIndex < _wmo.Groups.Count ? _wmo.Groups[groupIndex].WmoGroupId : 0;
    }

    public string? GetRenderGroupRawName(int renderGroupIndex)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
            return null;
        int groupIndex = _groups[renderGroupIndex].GroupIndex;
        return groupIndex < _wmo.Groups.Count ? _wmo.Groups[groupIndex].Name : null;
    }

    public bool TryGetGroupBounds(int renderGroupIndex, out Vector3 min, out Vector3 max)
    {
        if (renderGroupIndex < 0 || renderGroupIndex >= _groups.Count)
        {
            min = default;
            max = default;
            return false;
        }
        int groupIndex = _groups[renderGroupIndex].GroupIndex;
        if (groupIndex < _wmo.Groups.Count)
        {
            min = _wmo.Groups[groupIndex].BoundsMin;
            max = _wmo.Groups[groupIndex].BoundsMax;
            return true;
        }
        min = default;
        max = default;
        return false;
    }

    /// <summary>
    /// Tests a WMO-local point against group bounding boxes, returning the matching render group index, or -1.
    /// Prefers groups whose bounds strictly contain the point; if multiple overlap, selects the smallest volume.
    /// </summary>
    public int FindGroupContainingPoint(Vector3 localPoint)
    {
        int bestIdx = -1;
        float bestVolume = float.MaxValue;

        for (int i = 0; i < _groups.Count; i++)
        {
            int gi = _groups[i].GroupIndex;
            if (gi < 0 || gi >= _wmo.Groups.Count) continue;
            var g = _wmo.Groups[gi];
            if (localPoint.X >= g.BoundsMin.X && localPoint.X <= g.BoundsMax.X &&
                localPoint.Y >= g.BoundsMin.Y && localPoint.Y <= g.BoundsMax.Y &&
                localPoint.Z >= g.BoundsMin.Z && localPoint.Z <= g.BoundsMax.Z)
            {
                var size = g.BoundsMax - g.BoundsMin;
                float volume = Math.Abs(size.X * size.Y * size.Z);
                if (volume < bestVolume)
                {
                    bestVolume = volume;
                    bestIdx = i;
                }
            }
        }

        return bestIdx;
    }

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
    public int SubObjectCount => _groups.Count + 1 + _doodadController.DoodadInstanceCount;

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
            return $"--- Doodads ({_doodadController.DoodadInstanceCount}) ---";
        int di = index - _groups.Count - 1;
        if (di < _doodadController.DoodadInstanceCount)
        {
            var inst = _doodadController.DoodadInstances[di];
            return $"  Doodad: {Path.GetFileNameWithoutExtension(inst.ModelPath)}";
        }
        return "";
    }

    public bool GetSubObjectVisible(int index)
    {
        if (index < _groups.Count)
            return _groups[index].ManualVisible;
        if (index == _groups.Count)
            return _doodadController.DoodadsVisible;
        int di = index - _groups.Count - 1;
        if (di < _doodadController.DoodadInstanceCount)
            return _doodadController.DoodadInstances[di].Visible;
        return false;
    }

    public void SetSubObjectVisible(int index, bool visible)
    {
        if (index < _groups.Count)
            _groups[index].ManualVisible = visible;
        else if (index == _groups.Count)
            _doodadController.DoodadsVisible = visible;
        else
        {
            int di = index - _groups.Count - 1;
            if (di < _doodadController.DoodadInstanceCount)
                _doodadController.SetDoodadVisible(di, visible);
        }
    }

    // DoodadSet management
    public int DoodadSetCount => _doodadController.DoodadSetCount;
    public int ActiveDoodadSet => _doodadController.ActiveDoodadSet;
    public int DoodadInstanceCount => _doodadController.DoodadInstanceCount;
    public int DoodadDefCount => _doodadController.DoodadDefCount;
    public string GetDoodadSetName(int index) => _doodadController.GetDoodadSetName(index);

    public bool TryGetDoodadSetRange(int index, out string name, out int startIndex, out int count) =>
        _doodadController.TryGetDoodadSetRange(index, out name, out startIndex, out count);

    public bool TryGetDoodadInfo(int index, out WmoDoodadInfo info) =>
        _doodadController.TryGetDoodadInfo(index, out info);

    public bool TryGetDoodadBounds(int index, in Matrix4x4 modelMatrix, out Vector3 boundsMin, out Vector3 boundsMax)
        => _doodadController.TryGetDoodadBounds(index, modelMatrix, out boundsMin, out boundsMax, out _);

    public bool TryGetDoodadBounds(int index, in Matrix4x4 modelMatrix, out Vector3 boundsMin, out Vector3 boundsMax, out bool boundsResolved) =>
        _doodadController.TryGetDoodadBounds(index, modelMatrix, out boundsMin, out boundsMax, out boundsResolved);

    public bool TryGetDoodadWorldTransform(int index, in Matrix4x4 modelMatrix, out Matrix4x4 transform) =>
        _doodadController.TryGetDoodadWorldTransform(index, modelMatrix, out transform);

    public bool TryGetDoodadLocalBounds(int index, out Vector3 boundsMin, out Vector3 boundsMax) =>
        _doodadController.TryGetDoodadLocalBounds(index, out boundsMin, out boundsMax);

    public void RequestDoodadModelLoad(int index) =>
        _doodadController.RequestDoodadModelLoad(index);

    public bool TryPickDoodadsByRay(
        Vector3 rayOrigin,
        Vector3 rayDir,
        in Matrix4x4 modelMatrix,
        List<(int index, float distance, Vector3 hitPoint, Vector3 boundsMin, Vector3 boundsMax, WmoDoodadInfo info)> hits) =>
        _doodadController.TryPickDoodadsByRay(rayOrigin, rayDir, modelMatrix, hits);

    public bool TryGetDoodadDef(int doodadDefIndex, out WmoV14ToV17Converter.WmoDoodadDef def) =>
        _doodadController.TryGetDoodadDef(doodadDefIndex, out def);

    public string GetDoodadDefName(int doodadDefIndex) =>
        _doodadController.GetDoodadDefName(doodadDefIndex);

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

    public void SetActiveDoodadSet(int index) => _doodadController.SetActiveDoodadSet(index);

    public void SetRuntimeDoodadsVisible(bool visible) => _doodadController.SetRuntimeDoodadsVisible(visible);

    public void BeginWorldFrame() => _doodadController.BeginWorldFrame();

    public void EndWorldFrame() => _doodadController.EndWorldFrame();

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
        foreach (var textureId in _materialManager.MaterialTextures.Values)
        {
            if (textureId == 0)
                continue;

            _gl.BindTexture(TextureTarget.Texture2D, textureId);
            RenderQualitySettings.ApplySampling(_gl, TextureTarget.Texture2D, hasMipmaps: true,
                TextureWrapMode.Repeat, TextureWrapMode.Repeat);
        }

        _doodadController.ApplyTextureSamplingSettings();

        _gl.BindTexture(TextureTarget.Texture2D, 0);
    }

    public unsafe void RenderWireframeOverlay(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj,
        Vector3? fogColor = null, float fogStart = 200f, float fogEnd = 1500f, Vector3? cameraPos = null,
        Vector3? lightDir = null, Vector3? lightColor = null, Vector3? ambientColor = null,
        Vector3? wireframeColor = null)
    {
        float effectiveOpacity = GetEffectiveOpacity();
        if (effectiveOpacity <= 0.001f)
            return;

        _gl.UseProgram(_shaderProgram);
        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);
        _gl.Disable(EnableCap.Blend);
        _gl.Uniform1(_uOpacity, effectiveOpacity);

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
        _gl.Uniform1(_uUnlit, 1);
        _gl.Uniform1(_uAlphaTest, 0.0f);
        Vector3 lineColor = wireframeColor ?? new Vector3(0.95f, 1.0f, 0.65f);
        _gl.Uniform4(_uColor, lineColor.X, lineColor.Y, lineColor.Z, 1.0f);

        _gl.LineWidth(1.5f);
        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);

        foreach (var gb in _groups)
        {
            if (!gb.IsVisible) continue;
            _gl.BindVertexArray(gb.Vao);
            _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, gb.Ebo);
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
        Vector3? lightDir = null, Vector3? lightColor = null, Vector3? ambientColor = null,
        SceneLightManager? sceneLights = null)
    {
        float effectiveOpacity = GetEffectiveOpacity();
        if (effectiveOpacity <= 0.001f)
            return;

        ResetRenderStats();
        ProcessDeferredMaterialTextureLoads();
        _liquidRenderer.EnsureLiquidMeshesUpToDate();

        bool renderOpaquePass = pass != WmoRenderPass.Transparent;
        bool renderTransparentPass = pass != WmoRenderPass.Opaque;

        // WMO render order: opaque shell → doodad opaque → liquids → doodad transparent → transparent shell.
        _gl.UseProgram(_shaderProgram);
        ApplySurfaceCulling();
        _gl.Uniform1(_uUseInstanceModel, 0);
        _gl.Uniform1(_uOpacity, effectiveOpacity);

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
        _gl.Uniform1(_uUnlit, 0);
        UploadLocalLights(sceneLights, modelMatrix);

        UpdateRuntimeVisibility(modelMatrix, view, proj, cp);

        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // Pass 1: Opaque geometry (BlendMode 0) — depth write ON, 33% alpha blend if ghost wireframe
        if (renderOpaquePass)
        {
            _gl.Enable(EnableCap.DepthTest);
            if (_wireframe)
            {
                _gl.DepthMask(true);
                _gl.Enable(EnableCap.Blend);
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 0.33f * effectiveOpacity);
                // Spec 232: ghost wireframe lines must not be blackened by baked MOCV vertex
                // light (frequent on WMO interiors) — draw them unlit so they stay visible.
                _gl.Uniform1(_uUnlit, 1);
            }
            else
            {
                _gl.DepthMask(true);
                if (effectiveOpacity < 1.0f)
                {
                    _gl.Enable(EnableCap.Blend);
                    _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                }
                else
                {
                    _gl.Disable(EnableCap.Blend);
                }
                _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 1.0f);
                _gl.Uniform1(_uUnlit, 0);
            }
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
                            if (!_wireframe && effectiveOpacity >= 1.0f) _gl.Disable(EnableCap.Blend);
                            else if (effectiveOpacity < 1.0f) { _gl.Enable(EnableCap.Blend); _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha); }
                            _gl.DepthMask(true);
                            _gl.Uniform1(_uAlphaTest, WoWConstants.AlphaKeyThreshold);
                        }
                        else
                        {
                            if (!_wireframe && effectiveOpacity >= 1.0f) _gl.Disable(EnableCap.Blend);
                            else if (effectiveOpacity < 1.0f) { _gl.Enable(EnableCap.Blend); _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha); }
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
        if (_doodadController.DoodadsVisible && _doodadController.RuntimeDoodadsVisible && _doodadController.DoodadInstanceCount > 0)
        {
            visibleDoodadRenderCount = _doodadController.PrepareVisibleDoodads(modelMatrix, cp, fogEnd, updateAnimation: renderOpaquePass);

            if (renderOpaquePass)
            {
                _doodadController.RenderOpaqueDoodads(visibleDoodadRenderCount, modelMatrix, view, proj,
                    fc, fogStart, fogEnd, cp, ld, lc, ac, ref _currentDoodadSubmissions, sceneLights);
            }
        }

        // Pass 3: Liquid surfaces (semi-transparent, before transparent WMO geometry)
        if (renderTransparentPass && _runtimeGroupLiquidsVisible && _liquidRenderer.LiquidMeshCount > 0)
        {
            _liquidRenderer.Render(model, view, proj, _runtimeVisibleGroups,
                ref _currentDrawCalls, ref _currentLiquidDrawCalls, ref _currentVisibleLiquidMeshes);
        }

        // Pass 4: Doodad transparent layers back-to-front so model glass/reflection stays above liquids.
        if (renderTransparentPass && visibleDoodadRenderCount > 0)
        {
            _doodadController.RenderTransparentDoodads(visibleDoodadRenderCount, modelMatrix, view, proj,
                fc, fogStart, fogEnd, cp, ld, lc, ac, ref _currentDoodadSubmissions, sceneLights);
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

        if (_wireframe)
        {
            RenderWireframeOverlay(modelMatrix, view, proj, fc, fogStart, fogEnd, cp, ld, lc, ac);
        }

        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
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
        float effectiveOpacity = GetEffectiveOpacity();
        if (effectiveOpacity <= 0.001f)
        {
            _gpuInstanceBatchActive = false;
            return;
        }

        _gpuInstanceBatchActive = SupportsGpuInstancedOpaque;
        if (!_gpuInstanceBatchActive)
            return;

        _gl.UseProgram(_shaderProgram);
        ApplySurfaceCulling();
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true);
        if (effectiveOpacity < 1.0f)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        }
        else
        {
            _gl.Disable(EnableCap.Blend);
        }
        _gl.Uniform1(_uOpacity, effectiveOpacity);
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
        UploadLocalLights(null, Matrix4x4.Identity);
        _gl.Uniform1(_uAlphaTest, 0.0f);
    }

    public unsafe void BeginGpuInstanceBatch(Matrix4x4 view, Matrix4x4 proj,
        Vector3 fogColor, float fogStart, float fogEnd, Vector3 cameraPos,
        Vector3 lightDir, Vector3 lightColor, Vector3 ambientColor,
        SceneLightManager? sceneLights)
    {
        BeginGpuInstanceBatch(view, proj, fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor);
        UploadLocalLights(sceneLights, Matrix4x4.Identity);
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
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
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
        if (!SupportsGpuInstancedOpaque)
            return;

        _doodadController.RenderOpaqueDoodadsForPlacement(modelMatrix, view, proj,
            fogColor, fogStart, fogEnd, cameraPos, lightDir, lightColor, ambientColor, ref _currentDoodadSubmissions);
        UpdateLastRenderStats();
    }

    public void CollectOpaqueDoodadsForPlacement(
        Matrix4x4 modelMatrix,
        Vector3 cameraPos,
        float fogEnd,
        Action<WmoOpaqueDoodadBatchItem> collect)
    {
        ArgumentNullException.ThrowIfNull(collect);
        if (!SupportsGpuInstancedOpaque)
            return;

        _doodadController.CollectOpaqueDoodadsForPlacement(modelMatrix, cameraPos, fogEnd, collect, ref _currentDoodadSubmissions);
        UpdateLastRenderStats();
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

        if (AreTexturesEffective() && _materialManager.TryGetTexture(matId, out uint glTex))
        {
            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, glTex);
            _gl.Uniform1(_uHasTexture, 1);
            _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 1.0f);
        }
        else
        {
            _gl.Uniform1(_uHasTexture, 0);
            float r = _isCollisionWall ? 0.35f : ((gb.GroupIndex * 67 + 13) % 255) / 255f;
            float g = _isCollisionWall ? 0.65f : ((gb.GroupIndex * 131 + 7) % 255) / 255f;
            float b = _isCollisionWall ? 0.95f : ((gb.GroupIndex * 43 + 29) % 255) / 255f;
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

        if (AreTexturesEffective() && _materialManager.TryGetTexture(matId, out uint glTex))
        {
            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, glTex);
            _gl.Uniform1(_uHasTexture, 1);
            _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, 1.0f);
        }
        else
        {
            _gl.Uniform1(_uHasTexture, 0);
            float r = _isCollisionWall ? 0.35f : ((gb.GroupIndex * 67 + 13) % 255) / 255f;
            float g = _isCollisionWall ? 0.65f : ((gb.GroupIndex * 131 + 7) % 255) / 255f;
            float b = _isCollisionWall ? 0.95f : ((gb.GroupIndex * 43 + 29) % 255) / 255f;
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
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, gb.Ebo);
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
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, gb.Ebo);
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
uniform int uUnlit;
uniform vec4 uColor;
uniform float uAlphaTest;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform int uLocalLightCount;
uniform vec3 uLocalLightPos[8];
uniform vec3 uLocalLightColor[8];
uniform float uLocalLightIntensity[8];
uniform float uLocalLightStart[8];
uniform float uLocalLightEnd[8];
uniform float uOpacity;

out vec4 FragColor;

vec3 safeNormalize(vec3 value) {
    float len = length(value);
    if (len <= 0.0001)
        return vec3(0.0, 0.0, 1.0);
    return value / len;
}

void main() {
    vec3 norm = normalize(vNormal);
    // Half-Lambert diffuse: wraps lighting around surfaces for softer shading
    // Prevents harsh black shadows that don't match WoW's look
    float NdotL = dot(norm, normalize(uLightDir));
    float diff = NdotL * 0.5 + 0.5; // half-Lambert: remap [-1,1] to [0,1]
    diff = diff * diff; // square for slightly sharper falloff
    vec3 lighting = uAmbientColor + uLightColor * diff;
    vec3 localLight = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        if (i >= uLocalLightCount)
            break;

        vec3 toLight = uLocalLightPos[i] - vFragPos;
        float distanceToLight = length(toLight);
        float attenuationRange = max(uLocalLightEnd[i] - uLocalLightStart[i], 0.001);
        float attenuation = clamp((uLocalLightEnd[i] - distanceToLight) / attenuationRange, 0.0, 1.0);
        float localDiffuse = max(dot(norm, safeNormalize(toLight)), 0.0);
        localLight += max(uLocalLightColor[i], vec3(0.0))
            * clamp(uLocalLightIntensity[i], 0.0, 4.0)
            * attenuation
            * localDiffuse;
    }
    lighting += localLight;
    float bakedWeight = clamp(vBakedWeight, 0.0, 1.0);
    vec3 bakedLighting = mix(vec3(1.0), clamp(vVertexLight.rgb, vec3(0.0), vec3(1.0)), bakedWeight);

    vec4 texColor;
    if (uHasTexture == 1) {
        texColor = texture(uSampler, vTexCoord);
    } else {
        texColor = uColor;
    }

    // Spec 232: wireframe/selection passes emit flat unlit lines. WMO interior geometry
    // carries baked MOCV vertex light that is frequently black, which previously rendered
    // the selection outline invisible (black lines on dark terrain).
    if (uUnlit == 1) {
        FragColor = vec4(texColor.rgb, texColor.a * uOpacity);
        return;
    }

    // Alpha test: discard fragments below threshold (for cutout/transparent materials)
    if (uAlphaTest > 0.0 && texColor.a < uAlphaTest)
        discard;

    // Fog: blend to fog color based on distance from camera
    vec3 litColor = texColor.rgb * lighting * bakedLighting;
    float dist = length(vFragPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 foggedColor = mix(uFogColor, litColor, fogFactor);

    FragColor = vec4(foggedColor, texColor.a * uOpacity);
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
        _uOpacity = _gl.GetUniformLocation(_shaderProgram, "uOpacity");
        _uColor = _gl.GetUniformLocation(_shaderProgram, "uColor");
        _uUnlit = _gl.GetUniformLocation(_shaderProgram, "uUnlit");
        _uAlphaTest = _gl.GetUniformLocation(_shaderProgram, "uAlphaTest");
        _uFogColor = _gl.GetUniformLocation(_shaderProgram, "uFogColor");
        _uFogStart = _gl.GetUniformLocation(_shaderProgram, "uFogStart");
        _uFogEnd = _gl.GetUniformLocation(_shaderProgram, "uFogEnd");
        _uCameraPos = _gl.GetUniformLocation(_shaderProgram, "uCameraPos");
        _uLightDir = _gl.GetUniformLocation(_shaderProgram, "uLightDir");
        _uLightColor = _gl.GetUniformLocation(_shaderProgram, "uLightColor");
        _uAmbientColor = _gl.GetUniformLocation(_shaderProgram, "uAmbientColor");
        _uLocalLightCount = _gl.GetUniformLocation(_shaderProgram, "uLocalLightCount");
        for (int i = 0; i < MaxWmoLocalLights; i++)
        {
            _uLocalLightPos[i] = _gl.GetUniformLocation(_shaderProgram, $"uLocalLightPos[{i}]");
            _uLocalLightColor[i] = _gl.GetUniformLocation(_shaderProgram, $"uLocalLightColor[{i}]");
            _uLocalLightIntensity[i] = _gl.GetUniformLocation(_shaderProgram, $"uLocalLightIntensity[{i}]");
            _uLocalLightStart[i] = _gl.GetUniformLocation(_shaderProgram, $"uLocalLightStart[{i}]");
            _uLocalLightEnd[i] = _gl.GetUniformLocation(_shaderProgram, $"uLocalLightEnd[{i}]");
        }
    }

    private void UploadLocalLights(SceneLightManager? sceneLights, Matrix4x4 modelMatrix)
    {
        if (_uLocalLightCount < 0)
            return;

        int count = 0;
        if (sceneLights != null)
        {
            TransformAabb(BoundsMin, BoundsMax, modelMatrix, out Vector3 worldMin, out Vector3 worldMax);
            count = sceneLights.QueryAffecting(worldMin, worldMax, _localLightUploadScratch);
        }

        _gl.Uniform1(_uLocalLightCount, count);
        for (int i = 0; i < count; i++)
        {
            SceneLight light = _localLightUploadScratch[i];
            Vector3 color = ClampVector(light.Color, 0.0f, 4.0f);
            float intensity = Math.Clamp(FiniteOrDefault(light.Intensity, 0.0f), 0.0f, 4.0f);
            float start = Math.Clamp(FiniteOrDefault(light.AttenuationStart, 0.0f), 0.0f, 100000.0f);
            float end = MathF.Max(Math.Clamp(FiniteOrDefault(light.AttenuationEnd, 0.0f), 0.0f, 100000.0f), start + 0.001f);

            _gl.Uniform3(_uLocalLightPos[i], light.Position.X, light.Position.Y, light.Position.Z);
            _gl.Uniform3(_uLocalLightColor[i], color.X, color.Y, color.Z);
            _gl.Uniform1(_uLocalLightIntensity[i], intensity);
            _gl.Uniform1(_uLocalLightStart[i], start);
            _gl.Uniform1(_uLocalLightEnd[i], end);
        }
    }

    private static Vector3 ClampVector(Vector3 value, float min, float max)
    {
        return new Vector3(
            Math.Clamp(FiniteOrDefault(value.X, 0.0f), min, max),
            Math.Clamp(FiniteOrDefault(value.Y, 0.0f), min, max),
            Math.Clamp(FiniteOrDefault(value.Z, 0.0f), min, max));
    }

    private static float FiniteOrDefault(float value, float fallback)
        => float.IsFinite(value) ? value : fallback;

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
        _doodadController.ClearRuntimeVisibleDoodadDefs();
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
        _groupFrustumCuller.Update(view * proj, cameraPos);

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
                    _doodadController.AddRuntimeVisibleDoodadDef(doodadRef);
            }
        }
    }

    private static void TransformAabb(Vector3 min, Vector3 max, Matrix4x4 transform, out Vector3 outMin, out Vector3 outMax)
        => WmoGeometryHelper.TransformAabb(min, max, transform, out outMin, out outMax);

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
            var normals = WmoGeometryHelper.BuildRenderNormals(group);

            int vertCount = group.Vertices.Count;
            bool hasUVs = group.UVs.Count == vertCount;
            if (!hasUVs)
                ViewerLog.Trace($"[WmoRenderer] Group {gi} '{group.Name}': UV count mismatch! Verts={vertCount}, UVs={group.UVs.Count}");

            Vector4[] vertexLightColors = WmoGeometryHelper.BuildVertexLightColors(group);

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
    public const int DeferredMaterialTextureLoadsPerFrame = WmoMaterialManager.DeferredMaterialTextureLoadsPerFrame;
    public const double DeferredMaterialTextureLoadBudgetMs = WmoMaterialManager.DeferredMaterialTextureLoadBudgetMs;
    public const int DefaultDeferredDoodadLoads = WmoDoodadController.DefaultDeferredDoodadLoads;
    public const double DefaultDeferredDoodadBudgetMs = WmoDoodadController.DefaultDeferredDoodadBudgetMs;

    public int ProcessDeferredMaterialTextureLoads(
        int maxLoads = DeferredMaterialTextureLoadsPerFrame,
        double maxBudgetMs = DeferredMaterialTextureLoadBudgetMs)
        => _materialManager.ProcessDeferredMaterialTextureLoads(maxLoads, maxBudgetMs);

    public int ProcessDeferredDoodadLoads(
        int maxLoads = DefaultDeferredDoodadLoads,
        double maxBudgetMs = DefaultDeferredDoodadBudgetMs)
        => _doodadController.ProcessDeferredDoodadLoads(maxLoads, maxBudgetMs);

    private int ResolveBatchMaterialId(WmoV14ToV17Converter.WmoGroupData group, WmoV14ToV17Converter.WmoBatch batch)
        => _materialManager.ResolveBatchMaterialId(group, batch);

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

        _materialManager.Dispose();
        _liquidRenderer.Dispose();
        _doodadController.Dispose();

        _shaderRefCount--;
        if (_shaderRefCount <= 0 && _shaderProgram != 0)
        {
            _gl.DeleteProgram(_shaderProgram);
            _shaderProgram = 0;
            _shaderRefCount = 0;
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
}
