using System.IO;
using System.Numerics;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Validation;
using WowViewer.Core.Wmo;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;
using WowViewer.Core.Runtime.Mdx;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.App;

internal sealed class WorldGpuPreviewRenderer : IDisposable
{
    private const float WorldFieldOfViewDegrees = 45.0f;
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16.0f;
    private const float ChunkSubCellSize = ChunkSize / 8.0f;
    private const int ChunkAlphaSize = 64;
    private const float MapOrigin = 32.0f * TileSize;
    private const float TerrainTextureWorldScale = 8.0f / ChunkSize;

    private readonly GL _gl;
    private readonly IViewerIoService _viewerIoService;
    private readonly ViewerIoSourceKey _sourceKey;
    private uint _skyProgram;
    private uint _skyVao;
    private int _skyInverseViewProjectionLocation;
    private int _skyCameraPositionLocation;
    private int _skyZenithColorLocation;
    private int _skyHorizonColorLocation;
    private int _skyFogColorLocation;
    private int _skyBackdropStrengthLocation;
    private int _skyBackdropTintLocation;
    private int _skyBackdropSeedLocation;
    private uint _terrainProgram;
    private int _terrainViewLocation;
    private int _terrainProjectionLocation;
    private int _terrainLightDirectionLocation;
    private int _terrainLightColorLocation;
    private int _terrainAmbientColorLocation;
    private uint _overlayProgram;
    private int _overlayViewLocation;
    private int _overlayProjectionLocation;
    private uint _markerProgram;
    private int _markerViewLocation;
    private int _markerProjectionLocation;
    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private int _frameWidth;
    private int _frameHeight;
    private int _terrainDiffuseLayerCountLocation;
    private readonly List<TerrainTileMesh> _terrainTiles = [];
    private uint _overlayVao;
    private uint _overlayVbo;
    private uint _overlayVertexCount;
    private uint _markerVao;
    private uint _markerVbo;
    private uint _markerVertexCount;
    private Vector3 _boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
    private Vector3 _boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
    private bool _showSky;
    private Vector3 _skyZenithColor = new(0.16f, 0.30f, 0.54f);
    private Vector3 _skyHorizonColor = new(0.58f, 0.58f, 0.50f);
    private Vector3 _skyFogColor = new(0.34f, 0.38f, 0.42f);
    private Vector3 _skyBackdropTint = new(0.46f, 0.52f, 0.64f);
    private float _skyBackdropStrength;
    private float _skyBackdropSeed;
    private readonly Dictionary<string, TerrainTextureSample?> _terrainTextureCache = new(StringComparer.OrdinalIgnoreCase);

    // MDX Blend Modes
    private const uint MdxBlendModeTransparentKey = 1;
    private const uint MdxBlendModeBlend = 2;
    private const uint MdxBlendModeAdditive = 3;
    private const uint MdxBlendModeAddAlpha = 4;
    private const uint MdxBlendModeModulate = 5;
    private const uint MdxBlendModeModulate2X = 6;

    // Diagnostics — object loading counters
    private int _lastVisibleWmoCount;
    private int _lastVisibleMdxCount;
    private int _lastLoadedWmoCount;
    private int _lastLoadedMdxCount;

    // WMO Shader Program and Uniform Locations
    private uint _wmoProgram;
    private int _uWmoView;
    private int _uWmoProj;
    private int _uWmoModel;
    private int _uWmoLightDir;
    private int _uWmoAmbientColor;
    private int _uWmoBaseColor;
    private int _uWmoHasTexture;
    private int _uWmoTexture0;
    private int _uWmoAlphaTestThreshold;
    private int _uWmoUseTextureAlpha;

    // MDX Shader Program and Uniform Locations
    private uint _mdxProgram;
    private int _uMdxView;
    private int _uMdxProj;
    private int _uMdxModel;
    private int _uMdxLightDir;
    private int _uMdxLightColor;
    private int _uMdxAmbientColor;
    private int _uMdxBaseColor;
    private int _uMdxEmissiveColor;
    private int _uMdxAlpha;
    private int _uMdxHasTexture;
    private int _uMdxTexture0;
    private int _uMdxAlphaCutout;
    private int _uMdxAlphaThreshold;
    private int _uMdxReceivesLighting;
    private int _uMdxUseTextureAlpha;
    private int _uMdxPremultiplyAlpha;
    private int _uMdxSphereEnvMap;
    private int _uMdxUseBoneSkinning;

    // Object and Texture Caches
    private readonly Dictionary<string, CachedWmo> _wmoCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, CachedMdx> _mdxCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, uint> _loadedTextureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<uint> _ownedTextureIds = [];
    private uint _fallbackWhiteTexture;

    private WowViewerWorldRuntimeFrameResult? _activeFrame;
    private bool _disposed;

    public WorldGpuPreviewRenderer(GL gl, IViewerIoService viewerIoService, ViewerIoSourceKey sourceKey)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        _viewerIoService = viewerIoService ?? throw new ArgumentNullException(nameof(viewerIoService));
        _sourceKey = sourceKey;
        InitializeSkyShader();
        InitializeTerrainShader();
        InitializeOverlayShader();
        InitializeMarkerShader();
        InitializeWmoShader();
        InitializeMdxShader();
        _fallbackWhiteTexture = CreateFallbackWhiteTexture();
    }

    public uint PreviewTextureHandle => _colorTexture;

    public bool HasRenderableGeometry => _showSky || _terrainTiles.Count > 0 || _overlayVertexCount > 0 || _markerVertexCount > 0
        || (_activeFrame != null && _activeFrame.Visibility != null && (_activeFrame.Visibility.VisibleWmos.Count > 0 || _activeFrame.Visibility.VisibleMdx.Count > 0));

    public int TerrainTriangleCount => _terrainTiles.Sum(static tile => checked((int)(tile.IndexCount / 3)));

    public int MarkerCount => checked((int)_markerVertexCount);

    public float SceneScale => MathF.Max((_boundsMax - _boundsMin).Length(), 128f);

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        ClearPreview();

        foreach (var cached in _wmoCache.Values)
        {
            foreach (var cmd in cached.Commands)
            {
                cmd.Dispose(_gl);
            }
        }
        _wmoCache.Clear();

        foreach (var cached in _mdxCache.Values)
        {
            foreach (var cmd in cached.Commands)
            {
                cmd.Dispose(_gl);
            }
        }
        _mdxCache.Clear();

        foreach (uint textureId in _ownedTextureIds)
            _gl.DeleteTexture(textureId);

        _ownedTextureIds.Clear();
        _loadedTextureCache.Clear();

        if (_fallbackWhiteTexture != 0)
        {
            _gl.DeleteTexture(_fallbackWhiteTexture);
            _fallbackWhiteTexture = 0;
        }

        if (_skyProgram != 0)
        {
            _gl.DeleteProgram(_skyProgram);
            _skyProgram = 0;
        }

        if (_skyVao != 0)
        {
            _gl.DeleteVertexArray(_skyVao);
            _skyVao = 0;
        }

        if (_terrainProgram != 0)
        {
            _gl.DeleteProgram(_terrainProgram);
            _terrainProgram = 0;
        }

        if (_overlayProgram != 0)
        {
            _gl.DeleteProgram(_overlayProgram);
            _overlayProgram = 0;
        }

        if (_markerProgram != 0)
        {
            _gl.DeleteProgram(_markerProgram);
            _markerProgram = 0;
        }

        if (_wmoProgram != 0)
        {
            _gl.DeleteProgram(_wmoProgram);
            _wmoProgram = 0;
        }

        if (_mdxProgram != 0)
        {
            _gl.DeleteProgram(_mdxProgram);
            _mdxProgram = 0;
        }

        DeleteFramebuffer();
    }

    public void ClearPreview()
    {
        DeleteTerrainBuffers();
        DeleteOverlayBuffers();
        DeleteMarkerBuffers();
        _boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        _boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
        _showSky = false;
        _skyBackdropStrength = 0.0f;
        _activeFrame = null;
    }

    public void LoadPreview(WowViewerWorldRuntimeFrameResult frame, bool ignoreTerrainHoles = false, bool showHoleOverlay = false)
    {
        ArgumentNullException.ThrowIfNull(frame);

        ClearPreview();
        _activeFrame = frame;
        _showSky = frame.PassOptions.SkyVisible;
        ConfigureSkyColors(frame);
        BuildTerrainBuffers(frame, ignoreTerrainHoles);
        if (showHoleOverlay)
            BuildHoleOverlayBuffers(frame);
        BuildMarkerBuffers(frame);

        // Pre-load visible WMOs and MDXs to cache them
        if (frame.Visibility != null)
        {
            int wmoLoaded = 0;
            int mdxLoaded = 0;
            foreach (var entry in frame.Visibility.VisibleWmos)
            {
                if (GetOrLoadWmo(entry.Instance.ModelPath) != null)
                    wmoLoaded++;
            }
            foreach (var entry in frame.Visibility.VisibleMdx)
            {
                if (GetOrLoadMdx(entry.Instance.ModelPath) != null)
                    mdxLoaded++;
            }
            _lastVisibleWmoCount = frame.Visibility.VisibleWmos.Count;
            _lastVisibleMdxCount = frame.Visibility.VisibleMdx.Count;
            _lastLoadedWmoCount = wmoLoaded;
            _lastLoadedMdxCount = mdxLoaded;
            string diag = $"Objects: WMO visible={_lastVisibleWmoCount} loaded={_lastLoadedWmoCount}, MDX visible={_lastVisibleMdxCount} loaded={_lastLoadedMdxCount}";
            Console.Error.WriteLine($"[WorldGpuPreviewRenderer] {diag}");
            try { File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}] {diag}{Environment.NewLine}"); } catch { }
            try
            {
                string tileCount = $"Terrain tile meshes: {_terrainTiles.Count}";
                File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}] {tileCount}{Environment.NewLine}");
                for (int i = 0; i < _terrainTiles.Count; i++)
                {
                    var t = _terrainTiles[i];
                    File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}]   Tile[{i}]: ({t.TileX},{t.TileY}) indexCount={t.IndexCount} chunkCount={t.ChunkCount} layerCount={t.DiffuseLayerCount}{Environment.NewLine}");
                }
            }
            catch { }
        }
    }

    public unsafe void Render(int width, int height, WorldViewCamera camera)
    {
        ArgumentNullException.ThrowIfNull(camera);

        BuildMatrices(width, height, camera, out Matrix4x4 view, out Matrix4x4 projection);
        RenderCore(width, height, camera.Position, view, projection);
    }

    public unsafe void Render(int width, int height, ValidationCaptureCameraFrame cameraFrame)
    {
        RenderCore(width, height, cameraFrame.Eye, cameraFrame.View, cameraFrame.Projection);
    }

    private unsafe void RenderCore(int width, int height, Vector3 cameraPosition, Matrix4x4 view, Matrix4x4 projection)
    {
        try
        {
            File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"),
                $"[{DateTime.UtcNow:O}] RenderCore: showSky={_showSky} terrainTiles={_terrainTiles.Count} markers={_markerVertexCount} overlay={_overlayVertexCount} wmos={_activeFrame?.Visibility?.VisibleWmos.Count} mdxs={_activeFrame?.Visibility?.VisibleMdx.Count}{Environment.NewLine}");
        }
        catch { }

        if (!HasRenderableGeometry)
        {
            try { File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}] RenderCore: SKIP — no renderable geometry{Environment.NewLine}"); } catch { }
            return;
        }

        EnsureFramebuffer(width, height);
        Matrix4x4 viewProjection = view * projection;
        Matrix4x4.Invert(viewProjection, out Matrix4x4 inverseViewProjection);

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Disable(EnableCap.CullFace);
        _gl.ClearColor(_skyFogColor.X, _skyFogColor.Y, _skyFogColor.Z, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        if (_showSky)
        {
            _gl.Disable(EnableCap.DepthTest);
            _gl.UseProgram(_skyProgram);
            _gl.UniformMatrix4(_skyInverseViewProjectionLocation, 1, false, (float*)&inverseViewProjection.M11);
            _gl.Uniform3(_skyCameraPositionLocation, cameraPosition.X, cameraPosition.Y, cameraPosition.Z);
            _gl.Uniform3(_skyZenithColorLocation, _skyZenithColor.X, _skyZenithColor.Y, _skyZenithColor.Z);
            _gl.Uniform3(_skyHorizonColorLocation, _skyHorizonColor.X, _skyHorizonColor.Y, _skyHorizonColor.Z);
            _gl.Uniform3(_skyFogColorLocation, _skyFogColor.X, _skyFogColor.Y, _skyFogColor.Z);
            _gl.Uniform1(_skyBackdropStrengthLocation, _skyBackdropStrength);
            _gl.Uniform3(_skyBackdropTintLocation, _skyBackdropTint.X, _skyBackdropTint.Y, _skyBackdropTint.Z);
            _gl.Uniform1(_skyBackdropSeedLocation, _skyBackdropSeed);
            _gl.BindVertexArray(_skyVao);
            _gl.DrawArrays(PrimitiveType.Triangles, 0, 3);
            _gl.Enable(EnableCap.DepthTest);
        }

        if (_terrainTiles.Count > 0)
        {
            _gl.UseProgram(_terrainProgram);
            _gl.UniformMatrix4(_terrainViewLocation, 1, false, (float*)&view.M11);
            _gl.UniformMatrix4(_terrainProjectionLocation, 1, false, (float*)&projection.M11);
            _gl.Uniform3(_terrainLightDirectionLocation, 0.51f, 0.38f, 0.77f);
            _gl.Uniform3(_terrainLightColorLocation, 0.55f, 0.54f, 0.50f);
            _gl.Uniform3(_terrainAmbientColorLocation, 0.43f, 0.43f, 0.50f);
            foreach (TerrainTileMesh tileMesh in _terrainTiles)
            {
                _gl.ActiveTexture(TextureUnit.Texture0);
                _gl.BindTexture(TextureTarget.Texture2DArray, tileMesh.DiffuseArrayTexture);
                _gl.ActiveTexture(TextureUnit.Texture1);
                _gl.BindTexture(TextureTarget.Texture2DArray, tileMesh.AlphaShadowArrayTexture);
                _gl.Uniform1(_terrainDiffuseLayerCountLocation, tileMesh.DiffuseLayerCount);
                _gl.BindVertexArray(tileMesh.Vao);
                _gl.DrawElements(PrimitiveType.Triangles, tileMesh.IndexCount, DrawElementsType.UnsignedInt, null);
            }

            _gl.ActiveTexture(TextureUnit.Texture1);
            _gl.BindTexture(TextureTarget.Texture2DArray, 0);
            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2DArray, 0);
        }

        // Render Opaque WMOs
        var visibleWmos = _activeFrame?.Visibility?.VisibleWmos ?? new List<WorldVisibleWmoEntry>();
        if (visibleWmos.Count > 0)
        {
            _gl.UseProgram(_wmoProgram);
            _gl.UniformMatrix4(_uWmoView, 1, false, (float*)&view.M11);
            _gl.UniformMatrix4(_uWmoProj, 1, false, (float*)&projection.M11);
_gl.Uniform3(_uWmoLightDir, 0.51f, 0.38f, 0.77f);
_gl.Uniform3(_uWmoAmbientColor, 0.43f, 0.43f, 0.50f);
            _gl.Uniform1(_uWmoTexture0, 0);

            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthMask(true);
            _gl.Disable(EnableCap.Blend);

            foreach (var entry in visibleWmos)
            {
                var cached = GetOrLoadWmo(entry.Instance.ModelPath);
                if (cached == null) continue;

                Matrix4x4 modelMatrix = entry.Instance.Transform;
                _gl.UniformMatrix4(_uWmoModel, 1, false, (float*)&modelMatrix.M11);

                foreach (var command in cached.Commands)
                {
                    if (command.IsTransparent) continue;

                    _gl.Uniform3(_uWmoBaseColor, command.BaseColor.X, command.BaseColor.Y, command.BaseColor.Z);
                    _gl.Uniform1(_uWmoHasTexture, command.HasTexture ? 1 : 0);
                    _gl.Uniform1(_uWmoAlphaTestThreshold, command.AlphaTestThreshold);
                    _gl.Uniform1(_uWmoUseTextureAlpha, command.UseTextureAlpha ? 1 : 0);

                    _gl.ActiveTexture(TextureUnit.Texture0);
                    _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
                    _gl.BindVertexArray(command.Vao);
                    _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedShort, null);
                }
            }
        }

        // Render Opaque MDXs
        var visibleMdxs = _activeFrame?.Visibility?.VisibleMdx ?? new List<WorldVisibleMdxEntry>();
        if (visibleMdxs.Count > 0)
        {
            _gl.UseProgram(_mdxProgram);
            _gl.UniformMatrix4(_uMdxView, 1, false, (float*)&view.M11);
            _gl.UniformMatrix4(_uMdxProj, 1, false, (float*)&projection.M11);
_gl.Uniform3(_uMdxLightDir, 0.51f, 0.38f, 0.77f);
_gl.Uniform3(_uMdxLightColor, 0.55f, 0.54f, 0.50f);
_gl.Uniform3(_uMdxAmbientColor, 0.43f, 0.43f, 0.50f);
            _gl.Uniform1(_uMdxTexture0, 0);
            _gl.Uniform1(_uMdxUseBoneSkinning, 0);

            _gl.Enable(EnableCap.DepthTest);
            _gl.Disable(EnableCap.Blend);

            foreach (var entry in visibleMdxs)
            {
                var cached = GetOrLoadMdx(entry.Instance.ModelPath);
                if (cached == null) continue;

                Matrix4x4 modelMatrix = entry.Instance.Transform;
                _gl.UniformMatrix4(_uMdxModel, 1, false, (float*)&modelMatrix.M11);

                foreach (var command in cached.Commands)
                {
                    if (command.IsTransparent) continue;

                    if (command.DepthTest)
                        _gl.Enable(EnableCap.DepthTest);
                    else
                        _gl.Disable(EnableCap.DepthTest);

                    _gl.DepthMask(command.DepthWrite);

                    ResolveAlphaHandling(command, out bool useTextureAlpha, out bool premultiplyAlpha, out float alphaThreshold);

                    _gl.Uniform3(_uMdxBaseColor, command.BaseColor.X, command.BaseColor.Y, command.BaseColor.Z);
                    _gl.Uniform3(_uMdxEmissiveColor, command.EmissiveColor.X, command.EmissiveColor.Y, command.EmissiveColor.Z);
                    _gl.Uniform1(_uMdxAlpha, command.Alpha);
                    _gl.Uniform1(_uMdxHasTexture, command.HasTexture ? 1 : 0);
                    _gl.Uniform1(_uMdxAlphaCutout, command.AlphaCutout ? 1 : 0);
                    _gl.Uniform1(_uMdxAlphaThreshold, alphaThreshold);
                    _gl.Uniform1(_uMdxReceivesLighting, command.ReceivesLighting ? 1 : 0);
                    _gl.Uniform1(_uMdxUseTextureAlpha, useTextureAlpha ? 1 : 0);
                    _gl.Uniform1(_uMdxPremultiplyAlpha, premultiplyAlpha ? 1 : 0);
                    _gl.Uniform1(_uMdxSphereEnvMap, command.UsesSphereEnvMap ? 1 : 0);

                    _gl.ActiveTexture(TextureUnit.Texture0);
                    _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
                    _gl.BindVertexArray(command.Vao);
                    _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedShort, null);
                }
            }
        }

        if (_overlayVertexCount > 0)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.UseProgram(_overlayProgram);
            _gl.UniformMatrix4(_overlayViewLocation, 1, false, (float*)&view.M11);
            _gl.UniformMatrix4(_overlayProjectionLocation, 1, false, (float*)&projection.M11);
            _gl.BindVertexArray(_overlayVao);
            _gl.DrawArrays(PrimitiveType.Triangles, 0, _overlayVertexCount);
            _gl.Disable(EnableCap.Blend);
        }

        // Gather, sort and render transparent WMOs and MDXs
        List<TransparentDrawEntry> transparentDraws = [];
        foreach (var entry in visibleWmos)
        {
            var cached = GetOrLoadWmo(entry.Instance.ModelPath);
            if (cached == null) continue;

            foreach (var command in cached.Commands)
            {
                if (!command.IsTransparent) continue;

                Vector3 worldSortCenter = Vector3.Transform(command.SortCenter, entry.Instance.Transform);
                float distSq = Vector3.DistanceSquared(cameraPosition, worldSortCenter);
                transparentDraws.Add(new TransparentDrawEntry
                {
                    IsWmo = true,
                    WmoCommand = command,
                    Transform = entry.Instance.Transform,
                    DistanceSq = distSq
                });
            }
        }

        foreach (var entry in visibleMdxs)
        {
            var cached = GetOrLoadMdx(entry.Instance.ModelPath);
            if (cached == null) continue;

            foreach (var command in cached.Commands)
            {
                if (!command.IsTransparent) continue;

                Vector3 worldSortCenter = Vector3.Transform(command.TransparentSortCenter, entry.Instance.Transform);
                float distSq = Vector3.DistanceSquared(cameraPosition, worldSortCenter);
                transparentDraws.Add(new TransparentDrawEntry
                {
                    IsWmo = false,
                    MdxCommand = command,
                    Transform = entry.Instance.Transform,
                    DistanceSq = distSq
                });
            }
        }

        if (transparentDraws.Count > 0)
        {
            // Sort back-to-front (descending by DistanceSq)
            transparentDraws.Sort(static (a, b) => b.DistanceSq.CompareTo(a.DistanceSq));

            _gl.Enable(EnableCap.Blend);
            _gl.DepthMask(false);

            uint lastProgram = 0;

            foreach (var draw in transparentDraws)
            {
                if (draw.IsWmo)
                {
                    if (lastProgram != _wmoProgram)
                    {
                        _gl.UseProgram(_wmoProgram);
                        _gl.UniformMatrix4(_uWmoView, 1, false, (float*)&view.M11);
                        _gl.UniformMatrix4(_uWmoProj, 1, false, (float*)&projection.M11);
                        _gl.Uniform3(_uWmoLightDir, -0.45f, -0.55f, 0.70f);
                        _gl.Uniform3(_uWmoAmbientColor, 0.30f, 0.30f, 0.34f);
                        _gl.Uniform1(_uWmoTexture0, 0);
                        _gl.Enable(EnableCap.DepthTest);
                        _gl.DepthFunc(DepthFunction.Lequal);
                        lastProgram = _wmoProgram;
                    }

                    var command = draw.WmoCommand;
                    _gl.UniformMatrix4(_uWmoModel, 1, false, (float*)&draw.Transform.M11);
                    _gl.Uniform3(_uWmoBaseColor, command.BaseColor.X, command.BaseColor.Y, command.BaseColor.Z);
                    _gl.Uniform1(_uWmoHasTexture, command.HasTexture ? 1 : 0);
                    _gl.Uniform1(_uWmoAlphaTestThreshold, command.AlphaTestThreshold);
                    _gl.Uniform1(_uWmoUseTextureAlpha, command.UseTextureAlpha ? 1 : 0);

                    _gl.BlendFunc(command.SourceBlendFactor, command.DestinationBlendFactor);

                    _gl.ActiveTexture(TextureUnit.Texture0);
                    _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
                    _gl.BindVertexArray(command.Vao);
                    _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedShort, null);
                }
                else
                {
                    if (lastProgram != _mdxProgram)
                    {
                        _gl.UseProgram(_mdxProgram);
                        _gl.UniformMatrix4(_uMdxView, 1, false, (float*)&view.M11);
                        _gl.UniformMatrix4(_uMdxProj, 1, false, (float*)&projection.M11);
                        _gl.Uniform3(_uMdxLightDir, -0.45f, -0.55f, 0.70f);
                        _gl.Uniform3(_uMdxLightColor, 0.80f, 0.82f, 0.78f);
                        _gl.Uniform3(_uMdxAmbientColor, 0.30f, 0.30f, 0.34f);
                        _gl.Uniform1(_uMdxTexture0, 0);
                        _gl.Uniform1(_uMdxUseBoneSkinning, 0);
                        lastProgram = _mdxProgram;
                    }

                    var command = draw.MdxCommand;
                    _gl.UniformMatrix4(_uMdxModel, 1, false, (float*)&draw.Transform.M11);

                    if (command.DepthTest)
                        _gl.Enable(EnableCap.DepthTest);
                    else
                        _gl.Disable(EnableCap.DepthTest);

                    ConfigureBlendMode(command.IsAdditive, command.BlendMode);
                    ResolveAlphaHandling(command, out bool useTextureAlpha, out bool premultiplyAlpha, out float alphaThreshold);

                    _gl.Uniform3(_uMdxBaseColor, command.BaseColor.X, command.BaseColor.Y, command.BaseColor.Z);
                    _gl.Uniform3(_uMdxEmissiveColor, command.EmissiveColor.X, command.EmissiveColor.Y, command.EmissiveColor.Z);
                    _gl.Uniform1(_uMdxAlpha, command.Alpha);
                    _gl.Uniform1(_uMdxHasTexture, command.HasTexture ? 1 : 0);
                    _gl.Uniform1(_uMdxAlphaCutout, command.AlphaCutout ? 1 : 0);
                    _gl.Uniform1(_uMdxAlphaThreshold, alphaThreshold);
                    _gl.Uniform1(_uMdxReceivesLighting, command.ReceivesLighting ? 1 : 0);
                    _gl.Uniform1(_uMdxUseTextureAlpha, useTextureAlpha ? 1 : 0);
                    _gl.Uniform1(_uMdxPremultiplyAlpha, premultiplyAlpha ? 1 : 0);
                    _gl.Uniform1(_uMdxSphereEnvMap, command.UsesSphereEnvMap ? 1 : 0);

                    _gl.ActiveTexture(TextureUnit.Texture0);
                    _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
                    _gl.BindVertexArray(command.Vao);
                    _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedShort, null);
                }
            }

            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
        }

        _gl.BindVertexArray(0);
        _gl.UseProgram(0);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    private unsafe void BuildTerrainBuffers(WowViewerWorldRuntimeFrameResult frame, bool ignoreTerrainHoles)
    {
        IReadOnlyList<WowViewerWorldRuntimeTileFrame> activeTiles = GetActiveTerrainTiles(frame);
        float minHeight = activeTiles
            .Where(static tile => tile.TerrainTileData.Heightmap is not null)
            .Select(static tile => tile.TerrainTileData.Heightmap!.MinHeight)
            .DefaultIfEmpty(0f)
            .Min();
        float maxHeight = activeTiles
            .Where(static tile => tile.TerrainTileData.Heightmap is not null)
            .Select(static tile => tile.TerrainTileData.Heightmap!.MaxHeight)
            .DefaultIfEmpty(0f)
            .Max();
        float heightRange = MathF.Max(maxHeight - minHeight, 1.0f);

        foreach (WowViewerWorldRuntimeTileFrame tile in activeTiles)
        {
            TerrainTileMesh? terrainTile = BuildTerrainTileMesh(tile, minHeight, heightRange, ignoreTerrainHoles);
            if (terrainTile is not null)
                _terrainTiles.Add(terrainTile);
        }
    }

    private unsafe TerrainTileMesh? BuildTerrainTileMesh(WowViewerWorldRuntimeTileFrame tile, float minHeight, float heightRange, bool ignoreTerrainHoles)
    {
        List<string> tileTextureNames = CollectTileTextureNames(tile.TerrainTileData);
        Dictionary<string, int> textureIndices = BuildTextureIndexMap(tileTextureNames);
        if (tileTextureNames.Count > 0)
            Console.WriteLine($"[WorldGpuPreviewRenderer] Tile textures: {tileTextureNames.Count} names, first={tileTextureNames[0]}");
        else
            Console.WriteLine($"[WorldGpuPreviewRenderer] Tile textures: 0 names (chunks={tile.TerrainTileData.Chunks.Count}, layersChunks={tile.TerrainTileData.TextureLayerChunkCount})");

        const int chunkAlphaSliceCount = 256;
        byte[] alphaShadow = new byte[ChunkAlphaSize * ChunkAlphaSize * 4 * chunkAlphaSliceCount];
        List<float> vertexData = [];
        List<byte> chunkSliceData = [];
        List<ushort> texIndexData = [];
        List<uint> indexData = [];

        Vector3[] globalNormalsSum = new Vector3[257 * 257];
        ushort[] globalNormalsCount = new ushort[257 * 257];

        var chunksPositions = new Vector3[tile.TerrainTileData.Chunks.Count][];
        var chunksIndices = new int[tile.TerrainTileData.Chunks.Count][];
        var chunksValid = new bool[tile.TerrainTileData.Chunks.Count];

        for (int chunkIdx = 0; chunkIdx < tile.TerrainTileData.Chunks.Count; chunkIdx++)
        {
            var chunk = tile.TerrainTileData.Chunks[chunkIdx];
            if (!chunk.HasHeights || chunk.Heights is null)
                continue;

            Vector3[] positions = BuildChunkPositions(tile.TileX, tile.TileY, chunk);
            int[] indices = BuildChunkIndices(chunk.HoleMask, ignoreTerrainHoles);
            if (indices.Length == 0)
                continue;

            chunksPositions[chunkIdx] = positions;
            chunksIndices[chunkIdx] = indices;
            chunksValid[chunkIdx] = true;

            for (int triangle = 0; triangle + 2 < indices.Length; triangle += 3)
            {
                int i0 = indices[triangle + 0];
                int i1 = indices[triangle + 1];
                int i2 = indices[triangle + 2];

                Vector3 edge1 = positions[i1] - positions[i0];
                Vector3 edge2 = positions[i2] - positions[i0];
                Vector3 normal = Vector3.Cross(edge1, edge2);
                if (normal.LengthSquared() < 1e-10f)
                    continue;

                normal = Vector3.Normalize(normal);

                AccumulateGlobalNormal(chunk.IndexY, chunk.IndexX, i0, normal, globalNormalsSum, globalNormalsCount);
                AccumulateGlobalNormal(chunk.IndexY, chunk.IndexX, i1, normal, globalNormalsSum, globalNormalsCount);
                AccumulateGlobalNormal(chunk.IndexY, chunk.IndexX, i2, normal, globalNormalsSum, globalNormalsCount);
            }
        }

        for (int chunkIdx = 0; chunkIdx < tile.TerrainTileData.Chunks.Count; chunkIdx++)
        {
            if (!chunksValid[chunkIdx])
                continue;

            var chunk = tile.TerrainTileData.Chunks[chunkIdx];
            Vector3[] positions = chunksPositions[chunkIdx];
            int[] indices = chunksIndices[chunkIdx];

            int baseVertex = vertexData.Count / 11;
            byte chunkSlice = GetChunkSlice(chunk);
            ushort[] chunkTextureIndices = BuildChunkTextureIndices(chunk, textureIndices);

            for (int index = 0; index < positions.Length; index++)
            {
                Vector3 position = positions[index];
                Vector3 normal = GetGlobalNormal(chunk.IndexY, chunk.IndexX, index, globalNormalsSum, globalNormalsCount);
                float normalizedHeight = (position.Z - minHeight) / heightRange;
                float slopeFactor = Math.Clamp(1.0f - normal.Z, 0.0f, 1.0f);
                Vector3 fallbackColor = ComputeTerrainColor(normalizedHeight, slopeFactor);
                Vector2 alphaUv = GetChunkAlphaUv(index);

                vertexData.Add(position.X);
                vertexData.Add(position.Y);
                vertexData.Add(position.Z);
                vertexData.Add(normal.X);
                vertexData.Add(normal.Y);
                vertexData.Add(normal.Z);
                vertexData.Add(alphaUv.X);
                vertexData.Add(alphaUv.Y);
                vertexData.Add(1.0f);
                vertexData.Add(1.0f);
                vertexData.Add(1.0f);

                chunkSliceData.Add(chunkSlice);
                texIndexData.Add(chunkTextureIndices[0]);
                texIndexData.Add(chunkTextureIndices[1]);
                texIndexData.Add(chunkTextureIndices[2]);
                texIndexData.Add(chunkTextureIndices[3]);
                ExpandBounds(position);
            }

            foreach (int localIndex in indices)
                indexData.Add((uint)(baseVertex + localIndex));

            FillAlphaShadowSlice(alphaShadow, chunkSlice, chunk);
        }

        if (indexData.Count == 0)
            return null;

        TerrainTileMesh terrainTile = new(tile.TileX, tile.TileY);
        terrainTile.Vao = _gl.GenVertexArray();
        terrainTile.VboVertices = _gl.GenBuffer();
        terrainTile.VboChunkSlice = _gl.GenBuffer();
        terrainTile.VboTexIndices = _gl.GenBuffer();
        terrainTile.Ebo = _gl.GenBuffer();
        terrainTile.IndexCount = (uint)indexData.Count;
        terrainTile.ChunkCount = tile.TerrainTileData.ChunkCount;

        _gl.BindVertexArray(terrainTile.Vao);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, terrainTile.VboVertices);
        float[] vertexArray = vertexData.ToArray();
        fixed (float* vertexPtr = vertexArray)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexArray.Length * sizeof(float)), vertexPtr, BufferUsageARB.StaticDraw);

        const uint vertexStride = 11u * sizeof(float);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, vertexStride, (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, vertexStride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, vertexStride, (void*)(6 * sizeof(float)));
        _gl.EnableVertexAttribArray(2);
        _gl.VertexAttribPointer(5, 3, VertexAttribPointerType.Float, false, vertexStride, (void*)(8 * sizeof(float)));
        _gl.EnableVertexAttribArray(5);
        byte[] chunkSliceArray = chunkSliceData.ToArray();
        fixed (byte* chunkSlicePtr = chunkSliceArray)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)chunkSliceArray.Length, chunkSlicePtr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(3);
        _gl.VertexAttribIPointer(3, 1, VertexAttribIType.UnsignedByte, 1, (void*)0);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, terrainTile.VboTexIndices);
        ushort[] texIndexArray = texIndexData.ToArray();
        fixed (ushort* texIndexPtr = texIndexArray)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(texIndexArray.Length * sizeof(ushort)), texIndexPtr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(4);
        _gl.VertexAttribIPointer(4, 4, VertexAttribIType.UnsignedShort, (uint)(4 * sizeof(ushort)), (void*)0);

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, terrainTile.Ebo);
        uint[] indexArray = indexData.ToArray();
        fixed (uint* indexPtr = indexArray)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indexArray.Length * sizeof(uint)), indexPtr, BufferUsageARB.StaticDraw);

        _gl.BindVertexArray(0);

        CreateDiffuseArrayTexture(terrainTile, tileTextureNames);
        CreateAlphaShadowArrayTexture(terrainTile, alphaShadow);
        return terrainTile;
    }

    private unsafe void BuildHoleOverlayBuffers(WowViewerWorldRuntimeFrameResult frame)
    {
        List<float> overlayData = [];

        foreach (WowViewerWorldRuntimeTileFrame tile in GetActiveTerrainTiles(frame))
        {
            foreach (var chunk in tile.TerrainTileData.Chunks)
            {
                if (!chunk.HasHeights || chunk.Heights is null || !chunk.HasHoles)
                    continue;

                Vector3[] positions = BuildChunkPositions(tile.TileX, tile.TileY, chunk);
                for (int holeY = 0; holeY < 4; holeY++)
                {
                    for (int holeX = 0; holeX < 4; holeX++)
                    {
                        int holeBit = 1 << ((holeY * 4) + holeX);
                        if ((chunk.HoleMask & holeBit) == 0)
                            continue;

                        int startRow = holeY * 2;
                        int startCol = holeX * 2;
                        Vector3 topLeft = positions[OuterVertexIndex(startRow, startCol)] + new Vector3(0f, 0f, 1.25f);
                        Vector3 topRight = positions[OuterVertexIndex(startRow, startCol + 2)] + new Vector3(0f, 0f, 1.25f);
                        Vector3 bottomLeft = positions[OuterVertexIndex(startRow + 2, startCol)] + new Vector3(0f, 0f, 1.25f);
                        Vector3 bottomRight = positions[OuterVertexIndex(startRow + 2, startCol + 2)] + new Vector3(0f, 0f, 1.25f);
                        AppendOverlayTriangle(overlayData, topLeft, topRight, bottomRight, new Vector4(0.92f, 0.20f, 0.18f, 0.34f));
                        AppendOverlayTriangle(overlayData, topLeft, bottomRight, bottomLeft, new Vector4(0.92f, 0.20f, 0.18f, 0.34f));
                    }
                }
            }
        }

        if (overlayData.Count == 0)
            return;

        _overlayVertexCount = (uint)(overlayData.Count / 7);
        _overlayVao = _gl.GenVertexArray();
        _overlayVbo = _gl.GenBuffer();
        _gl.BindVertexArray(_overlayVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _overlayVbo);
        float[] overlayArray = overlayData.ToArray();
        fixed (float* overlayPtr = overlayArray)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(overlayArray.Length * sizeof(float)), overlayPtr, BufferUsageARB.StaticDraw);
        }

        const uint stride = 7u * sizeof(float);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.BindVertexArray(0);
    }

    private static void AppendOverlayTriangle(List<float> overlayData, Vector3 a, Vector3 b, Vector3 c, Vector4 color)
    {
        AppendOverlayVertex(overlayData, a, color);
        AppendOverlayVertex(overlayData, b, color);
        AppendOverlayVertex(overlayData, c, color);
    }

    private static void AppendOverlayVertex(List<float> overlayData, Vector3 position, Vector4 color)
    {
        overlayData.Add(position.X);
        overlayData.Add(position.Y);
        overlayData.Add(position.Z);
        overlayData.Add(color.X);
        overlayData.Add(color.Y);
        overlayData.Add(color.Z);
        overlayData.Add(color.W);
    }

    private static IReadOnlyList<WowViewerWorldRuntimeTileFrame> GetActiveTerrainTiles(WowViewerWorldRuntimeFrameResult frame)
    {
        if (frame.ActiveTerrainTiles.Count > 0)
            return frame.ActiveTerrainTiles;

        return
        [
            new WowViewerWorldRuntimeTileFrame(
                frame.SelectedTileX,
                frame.SelectedTileY,
                frame.PlacementSourcePath,
                frame.TileStageSummary,
                frame.TerrainTileData,
                frame.LiquidTileData,
                frame.PlacementCatalog),
        ];
    }

    private unsafe void BuildMarkerBuffers(WowViewerWorldRuntimeFrameResult frame)
    {
        List<float> markerData = [];
        AppendMarkers(markerData, frame.WmoInstances, new Vector4(0.98f, 0.76f, 0.32f, 0.92f));
        AppendMarkers(markerData, frame.MdxInstances, new Vector4(0.42f, 0.80f, 0.98f, 0.88f));
        if (markerData.Count == 0)
            return;

        _markerVertexCount = (uint)(markerData.Count / 7);
        _markerVao = _gl.GenVertexArray();
        _markerVbo = _gl.GenBuffer();
        _gl.BindVertexArray(_markerVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _markerVbo);
        float[] markerArray = markerData.ToArray();
        fixed (float* markerPtr = markerArray)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(markerArray.Length * sizeof(float)), markerPtr, BufferUsageARB.StaticDraw);
        }

        const uint stride = 7u * sizeof(float);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.BindVertexArray(0);
    }

    private static void AppendMarkers(List<float> markerData, IReadOnlyList<WowViewer.Core.Runtime.World.WorldObjectInstance> instances, Vector4 color)
    {
        foreach (var instance in instances)
        {
            Vector3 position = instance.BoundsResolved
                ? (instance.BoundsMin + instance.BoundsMax) * 0.5f
                : instance.PlacementPosition;

            markerData.Add(position.X);
            markerData.Add(position.Y);
            markerData.Add(position.Z + 2.0f);
            markerData.Add(color.X);
            markerData.Add(color.Y);
            markerData.Add(color.Z);
            markerData.Add(color.W);
        }
    }

    private void ConfigureSkyColors(WowViewerWorldRuntimeFrameResult frame)
    {
        float minHeight = frame.TerrainTileData.Heightmap?.MinHeight ?? 0f;
        float maxHeight = frame.TerrainTileData.Heightmap?.MaxHeight ?? 0f;
        float waterInfluence = frame.TileStageSummary.LiquidLayerCount > 0 ? 1.0f : 0.0f;
        float highRelief = Math.Clamp((maxHeight - minHeight) / 900.0f, 0.0f, 1.0f);

        Vector3 alphaZenith = new(0.13f, 0.27f, 0.50f);
        Vector3 highZenith = new(0.10f, 0.20f, 0.42f);
        Vector3 dryHorizon = new(0.64f, 0.58f, 0.44f);
        Vector3 wetHorizon = new(0.48f, 0.56f, 0.57f);
        Vector3 dryFog = new(0.38f, 0.36f, 0.31f);
        Vector3 wetFog = new(0.31f, 0.38f, 0.41f);

        _skyZenithColor = Vector3.Lerp(alphaZenith, highZenith, highRelief);
        _skyHorizonColor = Vector3.Lerp(dryHorizon, wetHorizon, waterInfluence * 0.65f);
        _skyFogColor = Vector3.Lerp(dryFog, wetFog, waterInfluence * 0.65f);

        ConfigureBackdropLayer(frame);
    }

    private void ConfigureBackdropLayer(WowViewerWorldRuntimeFrameResult frame)
    {
        if (frame.SkyboxBackdropInstances.Count == 0)
        {
            _skyBackdropStrength = 0.0f;
            _skyBackdropSeed = 0.0f;
            return;
        }

        _skyBackdropSeed = ComputeBackdropSeed(frame.SkyboxBackdropInstances);
        float countInfluence = Math.Clamp(MathF.Log2(frame.SkyboxBackdropInstances.Count + 1) / 6.0f, 0.0f, 1.0f);
        _skyBackdropStrength = Math.Clamp(0.16f + (countInfluence * 0.18f), 0.12f, 0.34f);

        float warmShift = Fract(_skyBackdropSeed * 1.731f);
        Vector3 moonlit = new(0.38f, 0.45f, 0.60f);
        Vector3 dusty = new(0.62f, 0.54f, 0.43f);
        _skyBackdropTint = Vector3.Lerp(moonlit, dusty, warmShift * 0.45f);
    }

    private static float ComputeBackdropSeed(IReadOnlyList<WowViewer.Core.Runtime.World.WorldObjectInstance> instances)
    {
        uint hash = 2166136261u;
        foreach (var instance in instances.Take(8))
        {
            string path = instance.ModelPath ?? string.Empty;
            for (int index = 0; index < path.Length; index++)
            {
                hash ^= (uint)char.ToUpperInvariant(path[index]);
                hash *= 16777619u;
            }
        }

        return (hash & 0x00FFFFFFu) / 16777215.0f;
    }

    private static float Fract(float value)
    {
        return value - MathF.Floor(value);
    }

    private void BuildMatrices(int width, int height, WorldViewCamera camera, out Matrix4x4 view, out Matrix4x4 projection)
    {
        view = camera.GetViewMatrix();

        Vector3 extent = _boundsMax - _boundsMin;
        float radius = MathF.Max(extent.Length() * 0.5f, 128f);
        float distance = Vector3.Distance(camera.Position, camera.Target);
        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float farPlane = MathF.Max(2048f, distance + (radius * 4.0f));
        projection = Matrix4x4.CreatePerspectiveFieldOfView(WorldFieldOfViewDegrees * MathF.PI / 180.0f, aspect, 1.0f, farPlane);
    }

    private static Vector3[] BuildChunkPositions(int tileX, int tileY, WowViewer.Core.Runtime.World.Terrain.WorldTerrainChunkData chunk)
    {
        float chunkWorldX = MapOrigin - (tileX * TileSize) - (chunk.IndexY * ChunkSize);
        float chunkWorldY = MapOrigin - (tileY * TileSize) - (chunk.IndexX * ChunkSize);
        Vector3[] positions = new Vector3[chunk.Heights!.Length];
        for (int index = 0; index < chunk.Heights.Length; index++)
        {
            GetChunkVertexLayout(index, out int row, out int col, out bool isInner);
            float localX = isInner ? (col + 0.5f) * ChunkSubCellSize : col * ChunkSubCellSize;
            float localY = isInner ? ((row / 2) + 0.5f) * ChunkSubCellSize : (row / 2) * ChunkSubCellSize;
            positions[index] = new Vector3(chunkWorldX - localY, chunkWorldY - localX, chunk.Heights[index]);
        }

        return positions;
    }

    private static void GetCameraAngles(Vector3 forward, out float azimuthDegrees, out float elevationDegrees)
    {
        azimuthDegrees = MathF.Atan2(forward.Y, forward.X) * 180.0f / MathF.PI;
        elevationDegrees = MathF.Asin(Math.Clamp(forward.Z, -1.0f, 1.0f)) * 180.0f / MathF.PI;
    }

    private static Vector3 ComputeForwardVector(float azimuthDegrees, float elevationDegrees)
    {
        float azimuthRadians = azimuthDegrees * MathF.PI / 180.0f;
        float elevationRadians = elevationDegrees * MathF.PI / 180.0f;
        float cosElevation = MathF.Cos(elevationRadians);
        return Vector3.Normalize(new Vector3(
            cosElevation * MathF.Cos(azimuthRadians),
            cosElevation * MathF.Sin(azimuthRadians),
            MathF.Sin(elevationRadians)));
    }

    private static Vector3[] BuildChunkNormals(int[] indices, IReadOnlyList<Vector3> positions)
    {
        Vector3[] accum = new Vector3[positions.Count];
        for (int triangle = 0; triangle + 2 < indices.Length; triangle += 3)
        {
            int i0 = indices[triangle + 0];
            int i1 = indices[triangle + 1];
            int i2 = indices[triangle + 2];
            Vector3 edge1 = positions[i1] - positions[i0];
            Vector3 edge2 = positions[i2] - positions[i0];
            Vector3 normal = Vector3.Cross(edge1, edge2);
            if (normal.LengthSquared() < 1e-10f)
                continue;

            normal = Vector3.Normalize(normal);
            accum[i0] += normal;
            accum[i1] += normal;
            accum[i2] += normal;
        }

        Vector3[] normals = new Vector3[positions.Count];
        for (int index = 0; index < normals.Length; index++)
            normals[index] = accum[index].LengthSquared() > 1e-10f ? Vector3.Normalize(accum[index]) : Vector3.UnitZ;

        return normals;
    }

    private static Vector3 ComputeTerrainColor(float normalizedHeight, float slopeFactor)
    {
        normalizedHeight = Math.Clamp(normalizedHeight, 0.0f, 1.0f);
        slopeFactor = Math.Clamp(slopeFactor, 0.0f, 1.0f);

        Vector3 low = new(0.19f, 0.29f, 0.17f);
        Vector3 mid = new(0.48f, 0.41f, 0.24f);
        Vector3 high = new(0.66f, 0.66f, 0.62f);
        Vector3 baseColor = normalizedHeight < 0.55f
            ? Vector3.Lerp(low, mid, normalizedHeight / 0.55f)
            : Vector3.Lerp(mid, high, (normalizedHeight - 0.55f) / 0.45f);

        return Vector3.Lerp(baseColor, high, slopeFactor * 0.35f);
    }

    private TerrainTextureSample? LoadTerrainTexture(string texturePath)
    {
        string normalizedPath = NormalizeTerrainTexturePath(texturePath);
        if (!_viewerIoService.TryReadVirtualFile(_sourceKey, normalizedPath, out byte[]? data, out _)
            || data is not { Length: > 0 })
        {
            Console.WriteLine($"[WorldGpuPreviewRenderer] Texture NOT found: '{normalizedPath}' (original: '{texturePath}')");
            return null;
        }

        try
        {
            using MemoryStream stream = new(data, writable: false);
            using BlpFile blp = new(stream);
            byte[] rgbaPixels = blp.GetPixels(0, out int width, out int height, bgra: false);
            Console.WriteLine($"[WorldGpuPreviewRenderer] Texture LOADED: '{normalizedPath}' {width}x{height}");
            return new TerrainTextureSample(width, height, rgbaPixels);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WorldGpuPreviewRenderer] Texture decode FAILED: '{normalizedPath}' - {ex.Message}");
            return null;
        }
    }

    private static string EnsureBlpExtension(string texturePath)
    {
        return texturePath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase)
            ? texturePath
            : $"{texturePath}.blp";
    }

    private static string NormalizeTerrainTexturePath(string texturePath)
    {
        return EnsureBlpExtension(texturePath)
            .Replace('/', '\\')
            .TrimStart('\\');
    }

    private static Vector2 GetChunkAlphaUv(int vertexIndex)
    {
        GetChunkVertexLayout(vertexIndex, out int row, out int col, out bool isInner);
        float localX = isInner ? (col + 0.5f) * ChunkSubCellSize : col * ChunkSubCellSize;
        float localY = isInner ? ((row / 2) + 0.5f) * ChunkSubCellSize : (row / 2) * ChunkSubCellSize;
        return new Vector2(localX / ChunkSize, localY / ChunkSize);
    }

    private static int SampleAlpha(byte[]? alphaMap, Vector2 uv)
    {
        if (alphaMap is not { Length: ChunkAlphaSize * ChunkAlphaSize })
            return 0;

        int sampleX = Math.Clamp((int)(Math.Clamp(uv.X, 0.0f, 1.0f) * (ChunkAlphaSize - 1)), 0, ChunkAlphaSize - 1);
        int sampleY = Math.Clamp((int)(Math.Clamp(uv.Y, 0.0f, 1.0f) * (ChunkAlphaSize - 1)), 0, ChunkAlphaSize - 1);
        return alphaMap[(sampleY * ChunkAlphaSize) + sampleX];
    }

    private static List<string> CollectTileTextureNames(WowViewer.Core.Runtime.World.Terrain.WorldTerrainTileData terrainTileData)
    {
        List<string> textureNames = [];
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        foreach (var chunk in terrainTileData.Chunks)
        {
            if (!chunk.HasTextureLayers)
                continue;

            int layerLimit = Math.Min(chunk.TextureLayers.Count, 4);
            for (int layerIndex = 0; layerIndex < layerLimit; layerIndex++)
            {
                string? texturePath = chunk.TextureLayers[layerIndex].TexturePath;
                if (string.IsNullOrWhiteSpace(texturePath))
                    continue;

                string normalizedPath = NormalizeTerrainTexturePath(texturePath);
                if (seen.Add(normalizedPath))
                    textureNames.Add(normalizedPath);
            }
        }

        return textureNames;
    }

    private static Dictionary<string, int> BuildTextureIndexMap(IReadOnlyList<string> textureNames)
    {
        Dictionary<string, int> textureIndices = new(StringComparer.OrdinalIgnoreCase);
        for (int index = 0; index < textureNames.Count; index++)
            textureIndices[textureNames[index]] = index;

        return textureIndices;
    }

    private static ushort[] BuildChunkTextureIndices(WowViewer.Core.Runtime.World.Terrain.WorldTerrainChunkData chunk, IReadOnlyDictionary<string, int> textureIndices)
    {
        ushort[] indices = [0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF];
        int layerLimit = Math.Min(chunk.TextureLayers.Count, 4);
        for (int layerIndex = 0; layerIndex < layerLimit; layerIndex++)
        {
            string? texturePath = chunk.TextureLayers[layerIndex].TexturePath;
            if (string.IsNullOrWhiteSpace(texturePath))
                continue;

            string normalizedPath = NormalizeTerrainTexturePath(texturePath);
            if (textureIndices.TryGetValue(normalizedPath, out int textureIndex))
                indices[layerIndex] = (ushort)Math.Clamp(textureIndex, 0, ushort.MaxValue - 1);
        }

        return indices;
    }

    private static byte GetChunkSlice(WowViewer.Core.Runtime.World.Terrain.WorldTerrainChunkData chunk)
    {
        int slice = (chunk.IndexY * 16) + chunk.IndexX;
        if ((uint)slice >= 256u)
            slice = chunk.ChunkIndex & 0xFF;

        return (byte)slice;
    }

    private static void AccumulateGlobalNormal(int chunkX, int chunkY, int vertexIdx, Vector3 normal, Vector3[] sum, ushort[] count)
    {
        GetChunkVertexLayout(vertexIdx, out int row, out int col, out bool isInner);
        int sampleX = isInner ? (col * 2) + 1 : col * 2;
        int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;

        int px = (chunkX * 16) + sampleX;
        int py = (chunkY * 16) + sampleY;

        if (px >= 0 && px < 257 && py >= 0 && py < 257)
        {
            int gridIndex = (py * 257) + px;
            sum[gridIndex] += normal;
            count[gridIndex]++;
        }
    }

    private static Vector3 GetGlobalNormal(int chunkX, int chunkY, int vertexIdx, Vector3[] sum, ushort[] count)
    {
        GetChunkVertexLayout(vertexIdx, out int row, out int col, out bool isInner);
        int sampleX = isInner ? (col * 2) + 1 : col * 2;
        int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;

        int px = (chunkX * 16) + sampleX;
        int py = (chunkY * 16) + sampleY;

        if (px >= 0 && px < 257 && py >= 0 && py < 257)
        {
            int gridIndex = (py * 257) + px;
            if (count[gridIndex] > 0)
            {
                Vector3 avg = sum[gridIndex];
                if (avg.LengthSquared() > 1e-10f)
                    return Vector3.Normalize(avg);
            }
        }

        return Vector3.UnitZ;
    }

    private static void FillAlphaShadowSlice(byte[] alphaShadow, byte slice, WowViewer.Core.Runtime.World.Terrain.WorldTerrainChunkData chunk)
    {
        int sliceBase = slice * ChunkAlphaSize * ChunkAlphaSize * 4;
        for (int layerIndex = 1; layerIndex <= 3; layerIndex++)
        {
            int channel = layerIndex - 1;

            if (layerIndex >= chunk.TextureLayers.Count)
                continue;

            AdtTextureChunkLayer layer = chunk.TextureLayers[layerIndex];
            byte[]? alphaMap = layer.DecodedAlpha?.AlphaMap;

            if (alphaMap is { Length: ChunkAlphaSize * ChunkAlphaSize })
            {
                for (int y = 0; y < ChunkAlphaSize; y++)
                {
                    for (int x = 0; x < ChunkAlphaSize; x++)
                    {
                        int srcX = Math.Min(x, ChunkAlphaSize - 2);
                        int srcY = Math.Min(y, ChunkAlphaSize - 2);
                        int dst = y * ChunkAlphaSize + x;
                        int src = srcY * ChunkAlphaSize + srcX;
                        alphaShadow[sliceBase + (dst * 4) + channel] = alphaMap[src];
                    }
                }
            }
            else if ((layer.Flags & 0x100u) == 0)
            {
                for (int sampleIndex = 0; sampleIndex < ChunkAlphaSize * ChunkAlphaSize; sampleIndex++)
                    alphaShadow[sliceBase + (sampleIndex * 4) + channel] = 255;
            }
        }

        if (chunk.ShadowMap is { Length: >= ChunkAlphaSize * ChunkAlphaSize })
        {
            for (int y = 0; y < ChunkAlphaSize; y++)
            {
                for (int x = 0; x < ChunkAlphaSize; x++)
                {
                    int srcX = Math.Min(x, ChunkAlphaSize - 2);
                    int srcY = Math.Min(y, ChunkAlphaSize - 2);
                    int dst = y * ChunkAlphaSize + x;
                    int src = srcY * ChunkAlphaSize + srcX;
                    alphaShadow[sliceBase + (dst * 4) + 3] = chunk.ShadowMap[src];
                }
            }
        }
    }

    private unsafe void CreateAlphaShadowArrayTexture(TerrainTileMesh tileMesh, byte[] alphaShadow)
    {
        tileMesh.AlphaShadowArrayTexture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, tileMesh.AlphaShadowArrayTexture);
        fixed (byte* alphaShadowPtr = alphaShadow)
        {
            _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, ChunkAlphaSize, ChunkAlphaSize, 256, 0, PixelFormat.Rgba, PixelType.UnsignedByte, alphaShadowPtr);
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);
    }

    private unsafe void CreateDiffuseArrayTexture(TerrainTileMesh tileMesh, IReadOnlyList<string> textureNames)
    {
        int maxLayers = 256;
        try
        {
            maxLayers = _gl.GetInteger(GetPName.MaxArrayTextureLayers);
        }
        catch
        {
        }

        int layerCount = Math.Max(1, Math.Min(textureNames.Count, maxLayers));
        Console.WriteLine($"[WorldGpuPreviewRenderer] CreateDiffuseArrayTexture: names={textureNames.Count} layerCount={layerCount}");
        int maxDimension = 0;
        for (int index = 0; index < Math.Min(textureNames.Count, layerCount); index++)
        {
            TerrainTextureSample? sample = LoadTerrainTexture(textureNames[index]);
            if (sample is null)
                continue;

            maxDimension = Math.Max(maxDimension, Math.Max(sample.Width, sample.Height));
        }

        int targetDimension = maxDimension switch
        {
            <= 0 => 256,
            <= 64 => 64,
            <= 128 => 128,
            _ => 256,
        };

        tileMesh.DiffuseArrayTexture = _gl.GenTexture();
        tileMesh.DiffuseLayerCount = layerCount;
        _gl.BindTexture(TextureTarget.Texture2DArray, tileMesh.DiffuseArrayTexture);
        _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, (uint)targetDimension, (uint)targetDimension, (uint)layerCount, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);

        for (int layer = 0; layer < layerCount; layer++)
        {
            byte[] pixels;
            if (layer < textureNames.Count && LoadTerrainTexture(textureNames[layer]) is TerrainTextureSample sample)
            {
                pixels = sample.Width == targetDimension && sample.Height == targetDimension
                    ? sample.RgbaPixels
                    : ResampleRgbaNearest(sample.RgbaPixels, sample.Width, sample.Height, targetDimension, targetDimension);
            }
            else
            {
                pixels = CreateSolidRgba(targetDimension, targetDimension, 255, 255, 255, 255);
            }

            fixed (byte* pixelPtr = pixels)
            {
                _gl.TexSubImage3D(TextureTarget.Texture2DArray, 0, 0, 0, layer, (uint)targetDimension, (uint)targetDimension, 1, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
        _gl.GenerateMipmap(TextureTarget.Texture2DArray);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);
    }

    private static byte[] CreateSolidRgba(int width, int height, byte red, byte green, byte blue, byte alpha)
    {
        byte[] data = new byte[width * height * 4];
        for (int index = 0; index < data.Length; index += 4)
        {
            data[index + 0] = red;
            data[index + 1] = green;
            data[index + 2] = blue;
            data[index + 3] = alpha;
        }

        return data;
    }

    private static byte[] ResampleRgbaNearest(byte[] source, int sourceWidth, int sourceHeight, int destinationWidth, int destinationHeight)
    {
        byte[] destination = new byte[destinationWidth * destinationHeight * 4];
        for (int y = 0; y < destinationHeight; y++)
        {
            int sourceY = (int)((long)y * sourceHeight / destinationHeight);
            for (int x = 0; x < destinationWidth; x++)
            {
                int sourceX = (int)((long)x * sourceWidth / destinationWidth);
                int sourceIndex = (sourceY * sourceWidth + sourceX) * 4;
                int destinationIndex = (y * destinationWidth + x) * 4;
                destination[destinationIndex + 0] = source[sourceIndex + 0];
                destination[destinationIndex + 1] = source[sourceIndex + 1];
                destination[destinationIndex + 2] = source[sourceIndex + 2];
                destination[destinationIndex + 3] = source[sourceIndex + 3];
            }
        }

        return destination;
    }

    private void ExpandBounds(Vector3 position)
    {
        _boundsMin = Vector3.Min(_boundsMin, position);
        _boundsMax = Vector3.Max(_boundsMax, position);
    }

    private sealed class TerrainTextureSample
    {
        public TerrainTextureSample(int width, int height, byte[] rgbaPixels)
        {
            Width = width;
            Height = height;
            RgbaPixels = rgbaPixels;
        }

        public int Width { get; }

        public int Height { get; }

        public byte[] RgbaPixels { get; }
    }

    private sealed class TerrainTileMesh
    {
        public TerrainTileMesh(int tileX, int tileY)
        {
            TileX = tileX;
            TileY = tileY;
        }

        public int TileX { get; }

        public int TileY { get; }

        public uint Vao { get; set; }

        public uint VboVertices { get; set; }

        public uint VboChunkSlice { get; set; }

        public uint VboTexIndices { get; set; }

        public uint Ebo { get; set; }

        public uint IndexCount { get; set; }

        public int ChunkCount { get; set; }

        public uint AlphaShadowArrayTexture { get; set; }

        public uint DiffuseArrayTexture { get; set; }

        public int DiffuseLayerCount { get; set; }

        public void Dispose(GL gl)
        {
            if (Vao != 0)
                gl.DeleteVertexArray(Vao);
            if (VboVertices != 0)
                gl.DeleteBuffer(VboVertices);
            if (VboChunkSlice != 0)
                gl.DeleteBuffer(VboChunkSlice);
            if (VboTexIndices != 0)
                gl.DeleteBuffer(VboTexIndices);
            if (Ebo != 0)
                gl.DeleteBuffer(Ebo);
            if (AlphaShadowArrayTexture != 0)
                gl.DeleteTexture(AlphaShadowArrayTexture);
            if (DiffuseArrayTexture != 0)
                gl.DeleteTexture(DiffuseArrayTexture);
        }
    }

    private static void GetChunkVertexLayout(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;
        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow & 1) == 0 ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow & 1) != 0;
                return;
            }

            remaining -= rowSize;
        }
    }

    private static int OuterVertexIndex(int row, int col) => row * 17 + col;

    private static int InnerVertexIndex(int row, int col) => row * 17 + 9 + col;

    private static int[] BuildChunkIndices(ushort holeMask, bool ignoreTerrainHoles)
    {
        List<int> indices = new(256 * 3);
        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                if (!ignoreTerrainHoles && holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << ((holeY * 4) + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                int topLeft = OuterVertexIndex(cellY, cellX);
                int topRight = OuterVertexIndex(cellY, cellX + 1);
                int bottomLeft = OuterVertexIndex(cellY + 1, cellX);
                int bottomRight = OuterVertexIndex(cellY + 1, cellX + 1);
                int center = InnerVertexIndex(cellY, cellX);

                indices.Add(center);
                indices.Add(topRight);
                indices.Add(topLeft);
                indices.Add(center);
                indices.Add(bottomRight);
                indices.Add(topRight);
                indices.Add(center);
                indices.Add(bottomLeft);
                indices.Add(bottomRight);
                indices.Add(center);
                indices.Add(topLeft);
                indices.Add(bottomLeft);
            }
        }

        return indices.ToArray();
    }

    private void InitializeTerrainShader()
    {
        const string vertexSource = """
            #version 330 core
            layout (location = 0) in vec3 aPosition;
            layout (location = 1) in vec3 aNormal;
            layout (location = 2) in vec2 aTexCoord;
            layout (location = 3) in uint aChunkSlice;
            layout (location = 4) in uvec4 aTexIndices;
            layout (location = 5) in vec3 aFallbackColor;

            uniform mat4 uView;
            uniform mat4 uProjection;

            out vec3 vWorldPosition;
            out vec3 vNormal;
            out vec2 vTexCoord;
            flat out uint vChunkSlice;
            flat out uvec4 vTexIndices;

            void main()
            {
                vWorldPosition = aPosition;
                vNormal = aNormal;
                vTexCoord = aTexCoord;
                vChunkSlice = aChunkSlice;
                vTexIndices = aTexIndices;
                gl_Position = uProjection * uView * vec4(aPosition, 1.0);
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec3 vWorldPosition;
            in vec3 vNormal;
            in vec2 vTexCoord;
            flat in uint vChunkSlice;
            flat in uvec4 vTexIndices;

            uniform sampler2DArray uDiffuseArray;
            uniform sampler2DArray uAlphaShadowArray;
            uniform int uDiffuseLayerCount;
            uniform vec3 uLightDirection;
            uniform vec3 uLightColor;
            uniform vec3 uAmbientColor;

            out vec4 FragColor;

            bool HasLayer(uint idx)
            {
                return (idx != 65535u) && (int(idx) >= 0) && (int(idx) < uDiffuseLayerCount);
            }

            void main()
            {
                float texScale = 8.0 / 33.333;
                vec2 diffuseUV = vec2(-vWorldPosition.y, -vWorldPosition.x) * texScale;
                vec4 alphaShadow = texture(uAlphaShadowArray, vec3(vTexCoord, float(vChunkSlice)));

                bool has0 = vTexIndices.x != 65535u;
                bool has1 = HasLayer(vTexIndices.y);
                bool has2 = HasLayer(vTexIndices.z);
                bool has3 = HasLayer(vTexIndices.w);

                vec3 norm = normalize(vNormal);
                float diff = abs(dot(norm, normalize(uLightDirection)));
                vec3 lighting = uAmbientColor + (uLightColor * diff);

                vec3 result = vec3(1.0);
                if (has0)
                {
                    vec4 c0 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIndices.x)));
                    result = c0.rgb * lighting;
                }
                if (has1)
                {
                    vec4 c1 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIndices.y)));
                    result = mix(result, c1.rgb * lighting, alphaShadow.r);
                }
                if (has2)
                {
                    vec4 c2 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIndices.z)));
                    result = mix(result, c2.rgb * lighting, alphaShadow.g);
                }
                if (has3)
                {
                    vec4 c3 = texture(uDiffuseArray, vec3(diffuseUV, float(vTexIndices.w)));
                    result = mix(result, c3.rgb * lighting, alphaShadow.b);
                }

                float shadow = alphaShadow.a;
                result *= mix(1.0, 0.4, shadow);

                FragColor = vec4(result, 1.0);
            }
            """;

        _terrainProgram = CreateProgram(vertexSource, fragmentSource, "world terrain");
        _terrainViewLocation = _gl.GetUniformLocation(_terrainProgram, "uView");
        _terrainProjectionLocation = _gl.GetUniformLocation(_terrainProgram, "uProjection");
        _terrainLightDirectionLocation = _gl.GetUniformLocation(_terrainProgram, "uLightDirection");
        _terrainLightColorLocation = _gl.GetUniformLocation(_terrainProgram, "uLightColor");
        _terrainAmbientColorLocation = _gl.GetUniformLocation(_terrainProgram, "uAmbientColor");
        _terrainDiffuseLayerCountLocation = _gl.GetUniformLocation(_terrainProgram, "uDiffuseLayerCount");
        _gl.UseProgram(_terrainProgram);
        _gl.Uniform1(_gl.GetUniformLocation(_terrainProgram, "uDiffuseArray"), 0);
        _gl.Uniform1(_gl.GetUniformLocation(_terrainProgram, "uAlphaShadowArray"), 1);
        _gl.UseProgram(0);
    }

    private void InitializeSkyShader()
    {
        const string vertexSource = """
            #version 330 core
            out vec2 vClip;

            void main()
            {
                vec2 positions[3] = vec2[3](
                    vec2(-1.0, -1.0),
                    vec2( 3.0, -1.0),
                    vec2(-1.0,  3.0)
                );
                vClip = positions[gl_VertexID];
                gl_Position = vec4(vClip, 0.0, 1.0);
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec2 vClip;

            uniform mat4 uInverseViewProjection;
            uniform vec3 uCameraPosition;
            uniform vec3 uZenithColor;
            uniform vec3 uHorizonColor;
            uniform vec3 uFogColor;
            uniform float uBackdropStrength;
            uniform vec3 uBackdropTint;
            uniform float uBackdropSeed;

            out vec4 FragColor;

            float hash21(vec2 p)
            {
                p = fract(p * vec2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return fract(p.x * p.y);
            }

            void main()
            {
                vec4 farPoint = uInverseViewProjection * vec4(vClip, 1.0, 1.0);
                vec3 worldPoint = farPoint.xyz / farPoint.w;
                vec3 ray = normalize(worldPoint - uCameraPosition);
                float up = clamp(ray.z * 0.5 + 0.5, 0.0, 1.0);
                float dome = smoothstep(0.18, 0.96, up);
                float horizonBand = exp(-abs(ray.z) * 5.5);
                vec3 color = mix(uHorizonColor, uZenithColor, dome);
                color = mix(color, uFogColor, horizonBand * 0.34);
                if (uBackdropStrength > 0.0)
                {
                    float azimuth = atan(ray.y, ray.x) / 6.2831853 + 0.5 + (uBackdropSeed * 0.37);
                    float latitude = acos(clamp(ray.z, -1.0, 1.0)) / 3.1415926;
                    vec2 shellCell = floor(vec2(azimuth * 96.0, latitude * 42.0));
                    float star = step(0.988, hash21(shellCell + uBackdropSeed));
                    float zenithMask = smoothstep(0.30, 0.88, up);
                    float shellBand = smoothstep(0.04, 0.42, abs(ray.z)) * (1.0 - smoothstep(0.78, 1.0, abs(ray.z)));
                    vec3 shell = mix(uBackdropTint, vec3(0.86, 0.82, 0.66), star * zenithMask);
                    color = mix(color, shell, uBackdropStrength * (0.22 + shellBand * 0.38 + star * 0.65));
                }
                FragColor = vec4(color, 1.0);
            }
            """;

        _skyProgram = CreateProgram(vertexSource, fragmentSource, "world sky backdrop");
        _skyVao = _gl.GenVertexArray();
        _skyInverseViewProjectionLocation = _gl.GetUniformLocation(_skyProgram, "uInverseViewProjection");
        _skyCameraPositionLocation = _gl.GetUniformLocation(_skyProgram, "uCameraPosition");
        _skyZenithColorLocation = _gl.GetUniformLocation(_skyProgram, "uZenithColor");
        _skyHorizonColorLocation = _gl.GetUniformLocation(_skyProgram, "uHorizonColor");
        _skyFogColorLocation = _gl.GetUniformLocation(_skyProgram, "uFogColor");
        _skyBackdropStrengthLocation = _gl.GetUniformLocation(_skyProgram, "uBackdropStrength");
        _skyBackdropTintLocation = _gl.GetUniformLocation(_skyProgram, "uBackdropTint");
        _skyBackdropSeedLocation = _gl.GetUniformLocation(_skyProgram, "uBackdropSeed");
    }

    private void InitializeOverlayShader()
    {
        const string vertexSource = """
            #version 330 core
            layout (location = 0) in vec3 aPosition;
            layout (location = 1) in vec4 aColor;

            uniform mat4 uView;
            uniform mat4 uProjection;

            out vec4 vColor;

            void main()
            {
                vColor = aColor;
                gl_Position = uProjection * uView * vec4(aPosition, 1.0);
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec4 vColor;
            out vec4 FragColor;

            void main()
            {
                FragColor = vColor;
            }
            """;

        _overlayProgram = CreateProgram(vertexSource, fragmentSource, "world overlay");
        _overlayViewLocation = _gl.GetUniformLocation(_overlayProgram, "uView");
        _overlayProjectionLocation = _gl.GetUniformLocation(_overlayProgram, "uProjection");
    }

    private void InitializeMarkerShader()
    {
        const string vertexSource = """
            #version 330 core
            layout (location = 0) in vec3 aPosition;
            layout (location = 1) in vec4 aColor;

            uniform mat4 uView;
            uniform mat4 uProjection;

            out vec4 vColor;

            void main()
            {
                vColor = aColor;
                gl_PointSize = 6.0;
                gl_Position = uProjection * uView * vec4(aPosition, 1.0);
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec4 vColor;
            out vec4 FragColor;

            void main()
            {
                vec2 centered = gl_PointCoord - vec2(0.5, 0.5);
                if (dot(centered, centered) > 0.25)
                    discard;

                FragColor = vColor;
            }
            """;

        _markerProgram = CreateProgram(vertexSource, fragmentSource, "world marker");
        _markerViewLocation = _gl.GetUniformLocation(_markerProgram, "uView");
        _markerProjectionLocation = _gl.GetUniformLocation(_markerProgram, "uProjection");
    }

    private uint CreateProgram(string vertexSource, string fragmentSource, string label)
    {
        uint vertexShader = CompileShader(ShaderType.VertexShader, vertexSource, label);
        uint fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentSource, label);

        uint program = _gl.CreateProgram();
        _gl.AttachShader(program, vertexShader);
        _gl.AttachShader(program, fragmentShader);
        _gl.LinkProgram(program);
        _gl.GetProgram(program, ProgramPropertyARB.LinkStatus, out int status);
        if (status == 0)
        {
            string log = _gl.GetProgramInfoLog(program);
            _gl.DeleteShader(vertexShader);
            _gl.DeleteShader(fragmentShader);
            throw new InvalidOperationException($"Failed to link {label} shader: {log}");
        }

        _gl.DetachShader(program, vertexShader);
        _gl.DetachShader(program, fragmentShader);
        _gl.DeleteShader(vertexShader);
        _gl.DeleteShader(fragmentShader);
        return program;
    }

    private uint CompileShader(ShaderType type, string source, string label)
    {
        uint shader = _gl.CreateShader(type);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status == 0)
        {
            string log = _gl.GetShaderInfoLog(shader);
            _gl.DeleteShader(shader);
            throw new InvalidOperationException($"Failed to compile {label} shader ({type}): {log}");
        }

        return shader;
    }

    private unsafe void EnsureFramebuffer(int width, int height)
    {
        width = Math.Max(width, 64);
        height = Math.Max(height, 64);
        if (_framebuffer != 0 && _frameWidth == width && _frameHeight == height)
            return;

        DeleteFramebuffer();
        _frameWidth = width;
        _frameHeight = height;
        _framebuffer = _gl.GenFramebuffer();
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);

        _colorTexture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _colorTexture);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);
        _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)_frameWidth, (uint)_frameHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);
        _gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, _colorTexture, 0);

        _depthRenderbuffer = _gl.GenRenderbuffer();
        _gl.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _depthRenderbuffer);
        _gl.RenderbufferStorage(RenderbufferTarget.Renderbuffer, InternalFormat.DepthComponent24, (uint)_frameWidth, (uint)_frameHeight);
        _gl.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, RenderbufferTarget.Renderbuffer, _depthRenderbuffer);

        if (_gl.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != GLEnum.FramebufferComplete)
            throw new InvalidOperationException("Failed to create world GPU preview framebuffer.");

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    private void DeleteFramebuffer()
    {
        if (_colorTexture != 0)
        {
            _gl.DeleteTexture(_colorTexture);
            _colorTexture = 0;
        }

        if (_depthRenderbuffer != 0)
        {
            _gl.DeleteRenderbuffer(_depthRenderbuffer);
            _depthRenderbuffer = 0;
        }

        if (_framebuffer != 0)
        {
            _gl.DeleteFramebuffer(_framebuffer);
            _framebuffer = 0;
        }
    }

    private void DeleteTerrainBuffers()
    {
        foreach (TerrainTileMesh tileMesh in _terrainTiles)
            tileMesh.Dispose(_gl);

        _terrainTiles.Clear();
    }

    private void DeleteOverlayBuffers()
    {
        if (_overlayVbo != 0)
        {
            _gl.DeleteBuffer(_overlayVbo);
            _overlayVbo = 0;
        }

        if (_overlayVao != 0)
        {
            _gl.DeleteVertexArray(_overlayVao);
            _overlayVao = 0;
        }

        _overlayVertexCount = 0;
    }

    private void DeleteMarkerBuffers()
    {
        if (_markerVbo != 0)
        {
            _gl.DeleteBuffer(_markerVbo);
            _markerVbo = 0;
        }

        if (_markerVao != 0)
        {
            _gl.DeleteVertexArray(_markerVao);
            _markerVao = 0;
        }

        _markerVertexCount = 0;
    }

    private static readonly IReadOnlyDictionary<uint, string> DefaultReplaceableTextures = new Dictionary<uint, string>
    {
        [1] = @"Textures\ReplaceableTextures\CreatureSkin\CreatureSkin01.blp",
        [2] = @"Textures\ReplaceableTextures\ObjectSkin\ObjectSkin01.blp",
        [3] = @"Textures\ReplaceableTextures\WeaponBlade\WeaponBlade01.blp",
        [4] = @"Textures\ReplaceableTextures\WeaponHandle\WeaponHandle01.blp",
        [5] = @"Textures\ReplaceableTextures\Environment\Environment01.blp",
        [6] = @"Textures\ReplaceableTextures\CharHair\CharHair00_00.blp",
        [7] = @"Textures\ReplaceableTextures\CharFacialHair\CharFacialHair00_00.blp",
        [8] = @"Textures\ReplaceableTextures\SkinExtra\SkinExtra01.blp",
        [9] = @"Textures\ReplaceableTextures\UISkin\UISkin01.blp",
        [10] = @"Textures\ReplaceableTextures\TaurenMane\TaurenMane00_00.blp",
        [11] = @"Textures\ReplaceableTextures\Monster\Monster01_01.blp",
        [12] = @"Textures\ReplaceableTextures\Monster\Monster01_02.blp",
        [13] = @"Textures\ReplaceableTextures\Monster\Monster01_03.blp",
    };

    private static float Lerp(float start, float end, float amount) => start + ((end - start) * amount);

    private void InitializeWmoShader()
    {
        const string vertexShaderSource = """
            #version 330 core
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec3 aNormal;
            layout(location = 2) in vec2 aTexCoord;
            uniform mat4 uModel;
            uniform mat4 uView;
            uniform mat4 uProj;
            out vec3 vNormal;
            out vec2 vTexCoord;
            void main()
            {
                vNormal = normalize(mat3(uModel) * aNormal);
                vTexCoord = aTexCoord;
                gl_Position = uProj * uView * uModel * vec4(aPosition, 1.0);
            }
            """;
        const string fragmentShaderSource = """
            #version 330 core
            in vec3 vNormal;
            in vec2 vTexCoord;
            uniform vec3 uLightDir;
            uniform vec3 uAmbientColor;
            uniform vec3 uBaseColor;
            uniform bool uHasTexture;
            uniform sampler2D uTexture0;
            uniform float uAlphaTestThreshold;
            uniform bool uUseTextureAlpha;
            out vec4 fragColor;
            void main()
            {
                vec4 texel = uHasTexture ? texture(uTexture0, vTexCoord) : vec4(1.0, 1.0, 1.0, 1.0);
                float alpha = uUseTextureAlpha ? texel.a : 1.0;
                if (uAlphaTestThreshold > 0.0 && alpha < uAlphaTestThreshold)
                    discard;

                float light = max(dot(normalize(vNormal), normalize(uLightDir)), 0.18);
                vec3 shaded = texel.rgb * uBaseColor;
                shaded *= clamp(uAmbientColor + vec3(light), vec3(0.0), vec3(1.75));
                fragColor = vec4(shaded, alpha);
            }
            """;

        _wmoProgram = CreateProgram(vertexShaderSource, fragmentShaderSource, "world wmo");
        _uWmoView = _gl.GetUniformLocation(_wmoProgram, "uView");
        _uWmoProj = _gl.GetUniformLocation(_wmoProgram, "uProj");
        _uWmoModel = _gl.GetUniformLocation(_wmoProgram, "uModel");
        _uWmoLightDir = _gl.GetUniformLocation(_wmoProgram, "uLightDir");
        _uWmoAmbientColor = _gl.GetUniformLocation(_wmoProgram, "uAmbientColor");
        _uWmoBaseColor = _gl.GetUniformLocation(_wmoProgram, "uBaseColor");
        _uWmoHasTexture = _gl.GetUniformLocation(_wmoProgram, "uHasTexture");
        _uWmoTexture0 = _gl.GetUniformLocation(_wmoProgram, "uTexture0");
        _uWmoAlphaTestThreshold = _gl.GetUniformLocation(_wmoProgram, "uAlphaTestThreshold");
        _uWmoUseTextureAlpha = _gl.GetUniformLocation(_wmoProgram, "uUseTextureAlpha");
    }

    private void InitializeMdxShader()
    {
        const string vertexSource = """
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 aNormal;
            layout (location = 2) in vec2 aTexCoord;
            layout (location = 3) in vec4 aBoneIndices;
            layout (location = 4) in vec4 aBoneWeights;

            uniform mat4 uModel;
            uniform mat4 uView;
            uniform mat4 uProj;
            uniform bool uUseBoneSkinning;
            uniform int uBoneCount;
            uniform mat4 uBones[128];
            uniform bool uUseUvTransform;
            uniform vec2 uUvTranslation;
            uniform vec2 uUvScale;
            uniform vec2 uUvRotationRow0;
            uniform vec2 uUvRotationRow1;

            out vec3 vNormal;
            out vec3 vViewNormal;
            out vec2 vTexCoord;

            vec4 ApplySkinning(vec4 source, vec4 boneIndices, vec4 boneWeights)
            {
                if (!uUseBoneSkinning)
                    return source;

                float totalWeight = boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
                if (totalWeight <= 0.0001)
                    return source;

                vec4 skinned = vec4(0.0);
                bool applied = false;

                int index0 = int(aBoneIndices.x + 0.5);
                int index1 = int(aBoneIndices.y + 0.5);
                int index2 = int(aBoneIndices.z + 0.5);
                int index3 = int(aBoneIndices.w + 0.5);

                if (boneWeights.x > 0.0 && index0 >= 0 && index0 < uBoneCount && index0 < 128)
                {
                    skinned += (uBones[index0] * source) * (boneWeights.x / totalWeight);
                    applied = true;
                }

                if (boneWeights.y > 0.0 && index1 >= 0 && index1 < uBoneCount && index1 < 128)
                {
                    skinned += (uBones[index1] * source) * (boneWeights.y / totalWeight);
                    applied = true;
                }

                if (boneWeights.z > 0.0 && index2 >= 0 && index2 < uBoneCount && index2 < 128)
                {
                    skinned += (uBones[index2] * source) * (boneWeights.z / totalWeight);
                    applied = true;
                }

                if (boneWeights.w > 0.0 && index3 >= 0 && index3 < uBoneCount && index3 < 128)
                {
                    skinned += (uBones[index3] * source) * (boneWeights.w / totalWeight);
                    applied = true;
                }

                return applied ? skinned : source;
            }

            void main()
            {
                vec4 skinnedPosition = ApplySkinning(vec4(aPos, 1.0), aBoneIndices, aBoneWeights);
                vec3 skinnedNormal = ApplySkinning(vec4(aNormal, 0.0), aBoneIndices, aBoneWeights).xyz;
                gl_Position = uProj * uView * uModel * skinnedPosition;
                vNormal = normalize(mat3(uModel) * skinnedNormal);
                vViewNormal = mat3(uView * uModel) * skinnedNormal;
                vec2 texCoord = aTexCoord;
                if (uUseUvTransform)
                {
                    vec2 centered = (texCoord - vec2(0.5, 0.5)) * uUvScale;
                    texCoord = vec2(
                        dot(centered, uUvRotationRow0),
                        dot(centered, uUvRotationRow1)) + vec2(0.5, 0.5) + uUvTranslation;
                }

                vTexCoord = texCoord;
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec3 vNormal;
            in vec3 vViewNormal;
            in vec2 vTexCoord;

            uniform vec3 uLightDir;
            uniform vec3 uLightColor;
            uniform vec3 uAmbientColor;
            uniform vec3 uBaseColor;
            uniform vec3 uEmissiveColor;
            uniform float uAlpha;
            uniform bool uHasTexture;
            uniform sampler2D uTexture0;
            uniform bool uAlphaCutout;
            uniform float uAlphaThreshold;
            uniform bool uReceivesLighting;
            uniform bool uUseTextureAlpha;
            uniform bool uPremultiplyAlpha;
            uniform bool uSphereEnvMap;
            out vec4 FragColor;

            void main()
            {
                vec2 texCoord = vTexCoord;
                if (uSphereEnvMap)
                {
                    vec3 viewNormal = normalize(vViewNormal);
                    if (!gl_FrontFacing)
                        viewNormal = -viewNormal;

                    texCoord = viewNormal.xy * 0.5 + 0.5;
                }

                vec4 texel = uHasTexture ? texture(uTexture0, texCoord) : vec4(1.0, 1.0, 1.0, 1.0);
                vec3 texRgb = texel.rgb;
                if (uPremultiplyAlpha)
                    texRgb *= texel.a;

                float sampledAlpha = uUseTextureAlpha ? texel.a : 1.0;
                float finalAlpha = clamp(sampledAlpha * uAlpha, 0.0, 1.0);
                if ((uAlphaCutout || uAlphaThreshold > 0.0) && finalAlpha < uAlphaThreshold)
                    discard;

                vec3 shaded = texRgb * uBaseColor;
                if (uReceivesLighting)
                {
                    vec3 normal = normalize(vNormal);
                    vec3 lightDir = normalize(uLightDir);
                    float NdotL = dot(normal, lightDir);
                    float diffuse = NdotL * 0.5 + 0.5;
                    diffuse = diffuse * diffuse;
                    shaded *= clamp(uAmbientColor + (uLightColor * diffuse), vec3(0.0), vec3(1.75));
                }

                shaded += uEmissiveColor;
                FragColor = vec4(shaded, finalAlpha);
            }
            """;

        _mdxProgram = CreateProgram(vertexSource, fragmentSource, "world mdx");
        _uMdxView = _gl.GetUniformLocation(_mdxProgram, "uView");
        _uMdxProj = _gl.GetUniformLocation(_mdxProgram, "uProj");
        _uMdxModel = _gl.GetUniformLocation(_mdxProgram, "uModel");
        _uMdxLightDir = _gl.GetUniformLocation(_mdxProgram, "uLightDir");
        _uMdxLightColor = _gl.GetUniformLocation(_mdxProgram, "uLightColor");
        _uMdxAmbientColor = _gl.GetUniformLocation(_mdxProgram, "uAmbientColor");
        _uMdxBaseColor = _gl.GetUniformLocation(_mdxProgram, "uBaseColor");
        _uMdxEmissiveColor = _gl.GetUniformLocation(_mdxProgram, "uEmissiveColor");
        _uMdxAlpha = _gl.GetUniformLocation(_mdxProgram, "uAlpha");
        _uMdxHasTexture = _gl.GetUniformLocation(_mdxProgram, "uHasTexture");
        _uMdxTexture0 = _gl.GetUniformLocation(_mdxProgram, "uTexture0");
        _uMdxAlphaCutout = _gl.GetUniformLocation(_mdxProgram, "uAlphaCutout");
        _uMdxAlphaThreshold = _gl.GetUniformLocation(_mdxProgram, "uAlphaThreshold");
        _uMdxReceivesLighting = _gl.GetUniformLocation(_mdxProgram, "uReceivesLighting");
        _uMdxUseTextureAlpha = _gl.GetUniformLocation(_mdxProgram, "uUseTextureAlpha");
        _uMdxPremultiplyAlpha = _gl.GetUniformLocation(_mdxProgram, "uPremultiplyAlpha");
        _uMdxSphereEnvMap = _gl.GetUniformLocation(_mdxProgram, "uSphereEnvMap");
        _uMdxUseBoneSkinning = _gl.GetUniformLocation(_mdxProgram, "uUseBoneSkinning");
    }

    private CachedWmo? GetOrLoadWmo(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return null;

        string normalizedPath = modelPath.Replace('/', '\\').TrimStart('\\');
        string cacheKey = normalizedPath.ToLowerInvariant();
        if (_wmoCache.TryGetValue(cacheKey, out CachedWmo? cachedWmo))
            return cachedWmo;

        if (!_viewerIoService.TryReadVirtualFile(_sourceKey, normalizedPath, out byte[]? modelBytes, out _)
            || modelBytes is not { Length: > 0 })
        {
            _wmoCache[cacheKey] = null!;
            string msg = $"WMO not found: {modelPath}";
            Console.Error.WriteLine($"[WorldGpuPreviewRenderer] {msg}");
            try { File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}] {msg}{Environment.NewLine}"); } catch { }
            return null;
        }

        try
        {
            using MemoryStream stream = new(modelBytes, writable: false);
            WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, modelPath);

            CachedWmo cached = new();
            bool hasBounds = false;
            Vector3 boundsMin = new(float.MaxValue);
            Vector3 boundsMax = new(float.MinValue);

            foreach (WmoEmbeddedGroupMeshDetail group in document.Groups)
            {
                if (group.Mesh.Vertices.Count == 0 || group.Mesh.Indices.Count < 3)
                    continue;

                float[] interleaved = new float[group.Mesh.Vertices.Count * 8];
                for (int index = 0; index < group.Mesh.Vertices.Count; index++)
                {
                    Vector3 vertex = group.Mesh.Vertices[index];
                    Vector3 normal = index < group.Mesh.Normals.Count ? group.Mesh.Normals[index] : Vector3.UnitZ;
                    Vector2 uv = index < group.Mesh.PrimaryUvs.Count ? group.Mesh.PrimaryUvs[index] : Vector2.Zero;
                    int offset = index * 8;
                    interleaved[offset + 0] = vertex.X;
                    interleaved[offset + 1] = vertex.Y;
                    interleaved[offset + 2] = vertex.Z;
                    interleaved[offset + 3] = normal.X;
                    interleaved[offset + 4] = normal.Y;
                    interleaved[offset + 5] = normal.Z;
                    interleaved[offset + 6] = uv.X;
                    interleaved[offset + 7] = uv.Y;
                    boundsMin = Vector3.Min(boundsMin, vertex);
                    boundsMax = Vector3.Max(boundsMax, vertex);
                    hasBounds = true;
                }

                bool builtBatchCommand = false;
                foreach (WmoGroupBatchDetail batch in group.Mesh.Batches)
                {
                    if (batch.IndexCount < 3)
                        continue;

                    if (batch.FirstIndex < 0 || batch.FirstIndex + batch.IndexCount > group.Mesh.Indices.Count)
                        continue;

                    ushort[] batchIndices = group.Mesh.Indices.Skip(batch.FirstIndex).Take(batch.IndexCount).ToArray();
                    int materialIndex = ResolveBatchMaterialIndex(document, group.Mesh, batch);
                    uint loadedTextureId = 0;
                    bool hasTexture = materialIndex >= 0
                        && materialIndex < document.Materials.Count
                        && TryGetOrLoadMaterialTextureWmo(document.Materials[materialIndex], out loadedTextureId);

                    WmoMaterialDetail? material = materialIndex >= 0 && materialIndex < document.Materials.Count
                        ? document.Materials[materialIndex]
                        : null;

                    WmoPreviewBlendMode blendMode = ResolveBlendMode(material?.BlendMode ?? 0);

                    WmoCommandBuffers cmd = CreateWmoCommand(
                        interleaved,
                        batchIndices,
                        ComputeBatchCenter(group.Mesh.Vertices, batchIndices),
                        hasTexture ? loadedTextureId : _fallbackWhiteTexture,
                        hasTexture,
                        IsTransparentPass(blendMode),
                        GetAlphaTestThreshold(blendMode),
                        UsesTextureAlpha(blendMode),
                        GetSourceBlendFactor(blendMode),
                        GetDestinationBlendFactor(blendMode),
                        hasTexture ? Vector3.One : ComputeGroupColor(group.GroupIndex));

                    cached.Commands.Add(cmd);
                    builtBatchCommand = true;
                }

                if (!builtBatchCommand)
                {
                    WmoCommandBuffers cmd = CreateWmoCommand(
                        interleaved,
                        group.Mesh.Indices.ToArray(),
                        ComputeBoundsCenter(group.Mesh.Vertices),
                        _fallbackWhiteTexture,
                        hasTexture: false,
                        isTransparent: false,
                        alphaTestThreshold: 0.0f,
                        useTextureAlpha: false,
                        sourceBlendFactor: BlendingFactor.SrcAlpha,
                        destinationBlendFactor: BlendingFactor.OneMinusSrcAlpha,
                        ComputeGroupColor(group.GroupIndex));
                    cached.Commands.Add(cmd);
                }
            }

            if (hasBounds)
            {
                cached.BoundsMin = boundsMin;
                cached.BoundsMax = boundsMax;
            }

            _wmoCache[cacheKey] = cached;
            return cached;
        }
        catch (Exception ex)
        {
            _wmoCache[cacheKey] = null!;
            string msg = $"WMO parse error for {modelPath}: {ex.Message}";
            Console.Error.WriteLine($"[WorldGpuPreviewRenderer] {msg}");
            try { File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}] {msg}{Environment.NewLine}"); } catch { }
            return null;
        }
    }

    private unsafe WmoCommandBuffers CreateWmoCommand(
        float[] interleaved,
        ushort[] indices,
        Vector3 sortCenter,
        uint textureId,
        bool hasTexture,
        bool isTransparent,
        float alphaTestThreshold,
        bool useTextureAlpha,
        BlendingFactor sourceBlendFactor,
        BlendingFactor destinationBlendFactor,
        Vector3 baseColor)
    {
        uint vao = _gl.GenVertexArray();
        uint vbo = _gl.GenBuffer();
        uint ebo = _gl.GenBuffer();
        _gl.BindVertexArray(vao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* verticesPtr = interleaved)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(interleaved.Length * sizeof(float)), verticesPtr, BufferUsageARB.StaticDraw);
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        fixed (ushort* indicesPtr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), indicesPtr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, (uint)(8 * sizeof(float)), (void*)0);
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, (uint)(8 * sizeof(float)), (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(2);
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, (uint)(8 * sizeof(float)), (void*)(6 * sizeof(float)));
        _gl.BindVertexArray(0);
        return new WmoCommandBuffers(vao, vbo, ebo, (uint)indices.Length, sortCenter, textureId, hasTexture, isTransparent, alphaTestThreshold, useTextureAlpha, sourceBlendFactor, destinationBlendFactor, baseColor);
    }

    private bool TryGetOrLoadMaterialTextureWmo(WmoMaterialDetail material, out uint textureId)
    {
        textureId = 0;
        foreach (string candidate in EnumerateTextureCandidates(material))
        {
            if (!TryGetOrLoadTexture(candidate, out uint loadedTextureId))
                continue;

            textureId = loadedTextureId;
            return true;
        }

        return false;
    }

    private static IEnumerable<string> EnumerateTextureCandidates(WmoMaterialDetail material)
    {
        if (!string.IsNullOrWhiteSpace(material.Texture1Name))
            yield return EnsureBlpExtension(material.Texture1Name);

        if (!string.IsNullOrWhiteSpace(material.Texture2Name))
            yield return EnsureBlpExtension(material.Texture2Name);

        if (!string.IsNullOrWhiteSpace(material.Texture3Name))
            yield return EnsureBlpExtension(material.Texture3Name);
    }

    private static int ResolveBatchMaterialIndex(WmoRenderDocument document, WmoGroupMeshDetail mesh, WmoGroupBatchDetail batch)
    {
        if (batch.MaterialId is int directMaterialId && directMaterialId >= 0 && directMaterialId < document.Materials.Count)
            return directMaterialId;

        int firstTriangle = batch.FirstIndex / 3;
        if (firstTriangle >= 0 && firstTriangle < mesh.FaceMaterials.Count)
        {
            int faceMaterialId = mesh.FaceMaterials[firstTriangle].MaterialId;
            if (faceMaterialId >= 0 && faceMaterialId < document.Materials.Count)
                return faceMaterialId;
        }

        return document.Materials.Count > 0 ? 0 : -1;
    }

    private static WmoPreviewBlendMode ResolveBlendMode(uint rawBlendMode)
    {
        return rawBlendMode switch
        {
            0 => WmoPreviewBlendMode.Opaque,
            1 => WmoPreviewBlendMode.Blend,
            2 => WmoPreviewBlendMode.Add,
            3 => WmoPreviewBlendMode.AlphaKey,
            _ => WmoPreviewBlendMode.Blend,
        };
    }

    private static bool IsTransparentPass(WmoPreviewBlendMode blendMode)
    {
        return blendMode is WmoPreviewBlendMode.Blend or WmoPreviewBlendMode.Add;
    }

    private static float GetAlphaTestThreshold(WmoPreviewBlendMode blendMode)
    {
        return blendMode == WmoPreviewBlendMode.AlphaKey ? 0.5f : 0.0f;
    }

    private static bool UsesTextureAlpha(WmoPreviewBlendMode blendMode)
    {
        return blendMode is WmoPreviewBlendMode.Blend or WmoPreviewBlendMode.Add or WmoPreviewBlendMode.AlphaKey;
    }

    private static BlendingFactor GetSourceBlendFactor(WmoPreviewBlendMode blendMode)
    {
        return blendMode switch
        {
            WmoPreviewBlendMode.Add => BlendingFactor.SrcAlpha,
            _ => BlendingFactor.SrcAlpha,
        };
    }

    private static BlendingFactor GetDestinationBlendFactor(WmoPreviewBlendMode blendMode)
    {
        return blendMode switch
        {
            WmoPreviewBlendMode.Add => BlendingFactor.One,
            _ => BlendingFactor.OneMinusSrcAlpha,
        };
    }

    private static Vector3 ComputeBatchCenter(IReadOnlyList<Vector3> vertices, IReadOnlyList<ushort> batchIndices)
    {
        if (vertices.Count == 0 || batchIndices.Count == 0)
            return Vector3.Zero;

        Vector3 sum = Vector3.Zero;
        int count = 0;
        foreach (ushort index in batchIndices)
        {
            if ((uint)index >= (uint)vertices.Count)
                continue;

            sum += vertices[index];
            count++;
        }

        return count > 0 ? sum / count : ComputeBoundsCenter(vertices);
    }

    private static Vector3 ComputeBoundsCenter(IReadOnlyList<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return Vector3.Zero;

        Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);
        foreach (Vector3 vertex in vertices)
        {
            min = Vector3.Min(min, vertex);
            max = Vector3.Max(max, vertex);
        }

        return (min + max) * 0.5f;
    }

    private static Vector3 ComputeGroupColor(int groupIndex)
    {
        float red = ((groupIndex * 67 + 13) % 255) / 255f;
        float green = ((groupIndex * 131 + 7) % 255) / 255f;
        float blue = ((groupIndex * 43 + 29) % 255) / 255f;
        return new Vector3(red, green, blue);
    }

    private CachedMdx? GetOrLoadMdx(string modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return null;

        string normalizedPath = modelPath.Replace('/', '\\').TrimStart('\\');
        string cacheKey = normalizedPath.ToLowerInvariant();
        if (_mdxCache.TryGetValue(cacheKey, out CachedMdx? cachedMdx))
            return cachedMdx;

        if (!_viewerIoService.TryReadVirtualFile(_sourceKey, normalizedPath, out byte[]? modelBytes, out _)
            || modelBytes is not { Length: > 0 })
        {
            _mdxCache[cacheKey] = null!;
            string msg = $"MDX not found: {modelPath}";
            Console.Error.WriteLine($"[WorldGpuPreviewRenderer] {msg}");
            try { File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}] {msg}{Environment.NewLine}"); } catch { }
            return null;
        }

        try
        {
            using MemoryStream summaryStream = new(modelBytes, writable: false);
            MdxSummary summary = MdxSummaryReader.Read(summaryStream, modelPath);

            using MemoryStream geometryStream = new(modelBytes, writable: false);
            MdxGeometryFile geometry = MdxGeometryReader.Read(geometryStream, modelPath);

            using MemoryStream boneStream = new(modelBytes, writable: false);
            MdxBoneFile bones = MdxBoneReader.Read(boneStream, modelPath);

            using MemoryStream materialStream = new(modelBytes, writable: false);
            MdxMaterialFile materials = MdxMaterialReader.Read(materialStream, modelPath);

            CachedMdx cached = new();
            ResolveBoundsMdx(geometry, summary, out Vector3 initialMin, out Vector3 initialMax);
            cached.BoundsMin = initialMin;
            cached.BoundsMax = initialMax;

            foreach (MdxGeosetGeometry geoset in geometry.Geosets)
            {
                if (geoset.Vertices.Count == 0 || geoset.Indices.Count < 3)
                    continue;

                float[] vertexData = new float[geoset.Vertices.Count * 8];
                (Vector4[] boneIndices, Vector4[] boneWeights) = (Array.Empty<Vector4>(), Array.Empty<Vector4>());

                Vector3 geosetMin = new(float.MaxValue);
                Vector3 geosetMax = new(float.MinValue);
                for (int index = 0; index < geoset.Vertices.Count; index++)
                {
                    Vector3 position = geoset.Vertices[index];
                    Vector3 normal = index < geoset.Normals.Count ? geoset.Normals[index] : Vector3.UnitZ;

                    if (float.IsFinite(position.X) && float.IsFinite(position.Y) && float.IsFinite(position.Z))
                    {
                        geosetMin = Vector3.Min(geosetMin, position);
                        geosetMax = Vector3.Max(geosetMax, position);
                    }

                    int offset = index * 8;
                    vertexData[offset + 0] = position.X;
                    vertexData[offset + 1] = position.Y;
                    vertexData[offset + 2] = position.Z;
                    vertexData[offset + 3] = normal.X;
                    vertexData[offset + 4] = normal.Y;
                    vertexData[offset + 5] = normal.Z;
                    vertexData[offset + 6] = 0.0f;
                    vertexData[offset + 7] = 0.0f;
                }

                float[] skinningVertexData = MdxSkinningHelper.BuildSkinningVertexData(
                    boneIndices,
                    boneWeights,
                    geoset.Vertices.Count);
                ushort[] indices = geoset.Indices.ToArray();

                int layerCount = geoset.MaterialId >= 0 && geoset.MaterialId < summary.MaterialCount
                    ? summary.Materials[geoset.MaterialId].LayerCount
                    : 0;
                if (layerCount == 0)
                    layerCount = 1;

                for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                {
                    MdxResolvedMaterialState materialState = MdxRenderStateResolver.ResolveMaterial(summary, materials, geoset.MaterialId, layerIndex, 0, 0);
                    
                    MdxResolvedGeosetRenderState geosetState = new()
                    {
                        Alpha = 1.0f,
                        BaseColor = Vector3.One,
                        ReceivesLighting = true,
                        DepthTest = true,
                        DepthWrite = !materialState.IsTransparent,
                    };

                    if (geosetState.Alpha <= 0.001f)
                        continue;

                    uint textureId = _fallbackWhiteTexture;
                    bool hasTexture = false;
                    if (TryGetOrLoadMaterialTextureMdx(summary, materialState, out uint loadedTextureId))
                    {
                        textureId = loadedTextureId;
                        hasTexture = true;
                    }

                    float[] layeredVertexData = (float[])vertexData.Clone();
                    IReadOnlyList<Vector2> layerUvSet = materialState.CoordId >= 0 && materialState.CoordId < geoset.UvSetCount
                        ? geoset.UvSets[materialState.CoordId]
                        : geoset.PrimaryUvSet;
                    for (int vertexIndex = 0; vertexIndex < geoset.Vertices.Count; vertexIndex++)
                    {
                        Vector2 uv = vertexIndex < layerUvSet.Count ? layerUvSet[vertexIndex] : Vector2.Zero;
                        int offset = (vertexIndex * 8) + 6;
                        layeredVertexData[offset + 0] = uv.X;
                        layeredVertexData[offset + 1] = uv.Y;
                    }

                    bool usesTransform = false;
                    Vector2 uvTranslation = Vector2.Zero;
                    Vector2 uvScale = Vector2.One;
                    Vector2 uvRotationRow0 = new(1.0f, 0.0f);
                    Vector2 uvRotationRow1 = new(0.0f, 1.0f);

                    MdxCommandBuffers cmd = CreateMdxCommand(
                        layeredVertexData,
                        skinningVertexData,
                        indices,
                        textureId,
                        hasTexture,
                        materialState.IsTransparent,
                        materialState.IsAdditive,
                        geosetState.DepthTest,
                        geosetState.DepthWrite,
                        materialState.AlphaCutout,
                        geosetState.ReceivesLighting,
                        materialState.UsesSphereEnvMap,
                        usesBoneSkinning: false,
                        usesTransform,
                        uvTranslation,
                        uvScale,
                        uvRotationRow0,
                        uvRotationRow1,
                        geosetState.BaseColor,
                        new Vector3(materialState.EmissiveGain),
                        geosetState.Alpha,
                        materialState.BlendMode,
                        geoset.MaterialId >= 0 && geoset.MaterialId < summary.MaterialCount
                            ? summary.Materials[geoset.MaterialId].PriorityPlane
                            : 0,
                        ResolveBoundsCenter(geosetMin, geosetMax));

                    cached.Commands.Add(cmd);
                }
            }

            _mdxCache[cacheKey] = cached;
            return cached;
        }
        catch (Exception ex)
        {
            _mdxCache[cacheKey] = null!;
            string msg = $"MDX parse error for {modelPath}: {ex.Message}";
            Console.Error.WriteLine($"[WorldGpuPreviewRenderer] {msg}");
            try { File.AppendAllText(Path.Combine(Path.GetTempPath(), "wow_renderer_diag.txt"), $"[{DateTime.UtcNow:O}] {msg}{Environment.NewLine}"); } catch { }
            return null;
        }
    }

    private unsafe MdxCommandBuffers CreateMdxCommand(
        float[] interleaved,
        float[] skinningInterleaved,
        ushort[] indices,
        uint textureId,
        bool hasTexture,
        bool isTransparent,
        bool isAdditive,
        bool depthTest,
        bool depthWrite,
        bool alphaCutout,
        bool receivesLighting,
        bool usesSphereEnvMap,
        bool usesBoneSkinning,
        bool usesUvTransform,
        Vector2 uvTranslation,
        Vector2 uvScale,
        Vector2 uvRotationRow0,
        Vector2 uvRotationRow1,
        Vector3 baseColor,
        Vector3 emissiveColor,
        float alpha,
        uint blendMode,
        int transparentSortPriority,
        Vector3 transparentSortCenter)
    {
        uint vao = _gl.GenVertexArray();
        uint vbo = _gl.GenBuffer();
        uint skinningVbo = _gl.GenBuffer();
        uint ebo = _gl.GenBuffer();

        _gl.BindVertexArray(vao);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* vertexPtr = interleaved)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(interleaved.Length * sizeof(float)), vertexPtr, BufferUsageARB.StaticDraw);
        }

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        fixed (ushort* indexPtr = indices)
        {
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), indexPtr, BufferUsageARB.StaticDraw);
        }

        const uint stride = 8u * sizeof(float);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));
        _gl.EnableVertexAttribArray(2);

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, skinningVbo);
        fixed (float* skinningPtr = skinningInterleaved)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(skinningInterleaved.Length * sizeof(float)), skinningPtr, BufferUsageARB.StaticDraw);
        }

        _gl.VertexAttribPointer(3, 4, VertexAttribPointerType.Float, false, stride, (void*)0);
        _gl.EnableVertexAttribArray(3);
        _gl.VertexAttribPointer(4, 4, VertexAttribPointerType.Float, false, stride, (void*)(4 * sizeof(float)));
        _gl.EnableVertexAttribArray(4);

        _gl.BindVertexArray(0);

        return new MdxCommandBuffers(
            vao, vbo, skinningVbo, ebo, (uint)indices.Length,
            textureId, hasTexture, isTransparent, isAdditive,
            depthTest, depthWrite, alphaCutout, receivesLighting,
            usesSphereEnvMap, usesBoneSkinning, usesUvTransform,
            uvTranslation, uvScale, uvRotationRow0, uvRotationRow1,
            baseColor, emissiveColor, alpha, blendMode,
            transparentSortPriority, transparentSortCenter);
    }

    private bool TryGetOrLoadMaterialTextureMdx(MdxSummary summary, MdxResolvedMaterialState material, out uint textureId)
    {
        textureId = 0;
        foreach (string candidate in EnumerateTextureCandidatesMdx(material))
        {
            if (!TryGetOrLoadTexture(candidate, out uint loadedTextureId))
                continue;

            textureId = loadedTextureId;
            return true;
        }

        return false;
    }

    private static IEnumerable<string> EnumerateTextureCandidatesMdx(MdxResolvedMaterialState material)
    {
        if (!string.IsNullOrWhiteSpace(material.TexturePath))
            yield return material.TexturePath;

        if (material.ReplaceableId != 0)
        {
            if (DefaultReplaceableTextures.TryGetValue(material.ReplaceableId, out string? replaceablePath))
                yield return replaceablePath;
        }
    }

    private uint CreateFallbackWhiteTexture()
    {
        byte[] whitePixel = [255, 255, 255, 255];
        uint textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, textureId);
        unsafe
        {
            fixed (byte* pixelPtr = whitePixel)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, 1, 1, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.Repeat);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.Repeat);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _ownedTextureIds.Add(textureId);
        return textureId;
    }

    private static Vector3 ResolveBoundsCenter(Vector3 min, Vector3 max)
    {
        if (!float.IsFinite(min.X) || !float.IsFinite(min.Y) || !float.IsFinite(min.Z)
            || !float.IsFinite(max.X) || !float.IsFinite(max.Y) || !float.IsFinite(max.Z))
            return Vector3.Zero;

        return (min + max) * 0.5f;
    }

    private static void ResolveBoundsMdx(MdxGeometryFile geometry, MdxSummary summary, out Vector3 min, out Vector3 max)
    {
        if (summary.BoundsMin is Vector3 summaryMin && summary.BoundsMax is Vector3 summaryMax)
        {
            Vector3 summaryExtent = summaryMax - summaryMin;
            if (float.IsFinite(summaryExtent.X) && float.IsFinite(summaryExtent.Y) && float.IsFinite(summaryExtent.Z)
                && summaryExtent.LengthSquared() > 0.0001f)
            {
                min = summaryMin;
                max = summaryMax;
                return;
            }
        }

        bool found = false;
        min = new Vector3(float.MaxValue);
        max = new Vector3(float.MinValue);
        foreach (MdxGeosetGeometry geoset in geometry.Geosets)
        {
            if (geoset.BoundsMin is Vector3 geosetMin && geoset.BoundsMax is Vector3 geosetMax)
            {
                min = Vector3.Min(min, geosetMin);
                max = Vector3.Max(max, geosetMax);
                found = true;
            }
        }

        if (!found)
        {
            min = new Vector3(-1.0f, -1.0f, -1.0f);
            max = new Vector3(1.0f, 1.0f, 1.0f);
        }
    }

    private static float[] BuildZeroSkinningVertexData(int vertexCount)
    {
        return new float[vertexCount * 8];
    }

    private bool TryGetOrLoadTexture(string texturePath, out uint textureId)
    {
        string cacheKey = texturePath.Replace('/', '\\').ToLowerInvariant();
        if (_loadedTextureCache.TryGetValue(cacheKey, out textureId))
            return textureId != 0;

        string normalized = EnsureBlpExtension(texturePath).Replace('/', '\\').TrimStart('\\');
        if (!_viewerIoService.TryReadVirtualFile(_sourceKey, normalized, out byte[]? bytes, out _) || bytes == null || bytes.Length == 0)
        {
            _loadedTextureCache[cacheKey] = 0;
            textureId = 0;
            return false;
        }

        try
        {
            using MemoryStream stream = new(bytes, writable: false);
            using BlpFile blp = new(stream);
            byte[] rgbaPixels = blp.GetPixels(0, out int width, out int height, bgra: false);
            textureId = UploadTexture(rgbaPixels, width, height);
            _loadedTextureCache[cacheKey] = textureId;
            _ownedTextureIds.Add(textureId);
            return textureId != 0;
        }
        catch
        {
            _loadedTextureCache[cacheKey] = 0;
            textureId = 0;
            return false;
        }
    }

    private uint UploadTexture(byte[] rgbaPixels, int width, int height)
    {
        uint textureId = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, textureId);
        unsafe
        {
            fixed (byte* pixelPtr = rgbaPixels)
            {
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)width, (uint)height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.Repeat);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.Repeat);
        _gl.GenerateMipmap(TextureTarget.Texture2D);
        _gl.BindTexture(TextureTarget.Texture2D, 0);
        return textureId;
    }

    private void ConfigureBlendMode(bool isAdditive, uint blendMode)
    {
        if (isAdditive || blendMode is MdxBlendModeAdditive or MdxBlendModeAddAlpha)
        {
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
            return;
        }

        if (blendMode is MdxBlendModeModulate or MdxBlendModeModulate2X)
        {
            _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
            return;
        }

        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
    }

    private static void ResolveAlphaHandling(MdxCommandBuffers command, out bool useTextureAlpha, out bool premultiplyAlpha, out float alphaThreshold)
    {
        if (command.AlphaCutout)
        {
            useTextureAlpha = true;
            premultiplyAlpha = false;
            alphaThreshold = 0.75f;
            return;
        }

        if (!command.IsTransparent)
        {
            useTextureAlpha = false;
            premultiplyAlpha = false;
            alphaThreshold = 0.0f;
            return;
        }

        switch (command.BlendMode)
        {
            case MdxBlendModeBlend:
                useTextureAlpha = true;
                premultiplyAlpha = true;
                alphaThreshold = 0.0f;
                return;
            case MdxBlendModeAdditive:
            case MdxBlendModeAddAlpha:
                useTextureAlpha = true;
                premultiplyAlpha = false;
                alphaThreshold = 0.0f;
                return;
            case MdxBlendModeModulate:
            case MdxBlendModeModulate2X:
                useTextureAlpha = false;
                premultiplyAlpha = false;
                alphaThreshold = 0.0f;
                return;
            case 0:
                useTextureAlpha = false;
                premultiplyAlpha = false;
                alphaThreshold = 0.05f;
                return;
            default:
                useTextureAlpha = true;
                premultiplyAlpha = false;
                alphaThreshold = 0.05f;
                return;
        }
    }

    private sealed class CachedWmo
    {
        public List<WmoCommandBuffers> Commands { get; } = [];
        public Vector3 BoundsMin { get; set; }
        public Vector3 BoundsMax { get; set; }
    }

    private sealed class WmoCommandBuffers
    {
        public WmoCommandBuffers(
            uint vao,
            uint vbo,
            uint ebo,
            uint indexCount,
            Vector3 sortCenter,
            uint textureId,
            bool hasTexture,
            bool isTransparent,
            float alphaTestThreshold,
            bool useTextureAlpha,
            BlendingFactor sourceBlendFactor,
            BlendingFactor destinationBlendFactor,
            Vector3 baseColor)
        {
            Vao = vao;
            Vbo = vbo;
            Ebo = ebo;
            IndexCount = indexCount;
            SortCenter = sortCenter;
            TextureId = textureId;
            HasTexture = hasTexture;
            IsTransparent = isTransparent;
            AlphaTestThreshold = alphaTestThreshold;
            UseTextureAlpha = useTextureAlpha;
            SourceBlendFactor = sourceBlendFactor;
            DestinationBlendFactor = destinationBlendFactor;
            BaseColor = baseColor;
        }

        public uint Vao { get; }
        public uint Vbo { get; }
        public uint Ebo { get; }
        public uint IndexCount { get; }
        public Vector3 SortCenter { get; }
        public uint TextureId { get; }
        public bool HasTexture { get; }
        public bool IsTransparent { get; }
        public float AlphaTestThreshold { get; }
        public bool UseTextureAlpha { get; }
        public BlendingFactor SourceBlendFactor { get; }
        public BlendingFactor DestinationBlendFactor { get; }
        public Vector3 BaseColor { get; }

        public void Dispose(GL gl)
        {
            if (Vbo != 0) gl.DeleteBuffer(Vbo);
            if (Ebo != 0) gl.DeleteBuffer(Ebo);
            if (Vao != 0) gl.DeleteVertexArray(Vao);
        }
    }

    private sealed class CachedMdx
    {
        public List<MdxCommandBuffers> Commands { get; } = [];
        public Vector3 BoundsMin { get; set; }
        public Vector3 BoundsMax { get; set; }
    }

    private sealed class MdxCommandBuffers
    {
        public MdxCommandBuffers(
            uint vao,
            uint vbo,
            uint skinningVbo,
            uint ebo,
            uint indexCount,
            uint textureId,
            bool hasTexture,
            bool isTransparent,
            bool isAdditive,
            bool depthTest,
            bool depthWrite,
            bool alphaCutout,
            bool receivesLighting,
            bool usesSphereEnvMap,
            bool usesBoneSkinning,
            bool usesUvTransform,
            Vector2 uvTranslation,
            Vector2 uvScale,
            Vector2 uvRotationRow0,
            Vector2 uvRotationRow1,
            Vector3 baseColor,
            Vector3 emissiveColor,
            float alpha,
            uint blendMode,
            int transparentSortPriority,
            Vector3 transparentSortCenter)
        {
            Vao = vao;
            Vbo = vbo;
            SkinningVbo = skinningVbo;
            Ebo = ebo;
            IndexCount = indexCount;
            TextureId = textureId;
            HasTexture = hasTexture;
            IsTransparent = isTransparent;
            IsAdditive = isAdditive;
            DepthTest = depthTest;
            DepthWrite = depthWrite;
            AlphaCutout = alphaCutout;
            ReceivesLighting = receivesLighting;
            UsesSphereEnvMap = usesSphereEnvMap;
            UsesBoneSkinning = usesBoneSkinning;
            UsesUvTransform = usesUvTransform;
            UvTranslation = uvTranslation;
            UvScale = uvScale;
            UvRotationRow0 = uvRotationRow0;
            UvRotationRow1 = uvRotationRow1;
            BaseColor = baseColor;
            EmissiveColor = emissiveColor;
            Alpha = alpha;
            BlendMode = blendMode;
            TransparentSortPriority = transparentSortPriority;
            TransparentSortCenter = transparentSortCenter;
        }

        public uint Vao { get; }
        public uint Vbo { get; }
        public uint SkinningVbo { get; }
        public uint Ebo { get; }
        public uint IndexCount { get; }
        public uint TextureId { get; }
        public bool HasTexture { get; }
        public bool IsTransparent { get; }
        public bool IsAdditive { get; }
        public bool DepthTest { get; }
        public bool DepthWrite { get; }
        public bool AlphaCutout { get; }
        public bool ReceivesLighting { get; }
        public bool UsesSphereEnvMap { get; }
        public bool UsesBoneSkinning { get; }
        public bool UsesUvTransform { get; }
        public Vector2 UvTranslation { get; }
        public Vector2 UvScale { get; }
        public Vector2 UvRotationRow0 { get; }
        public Vector2 UvRotationRow1 { get; }
        public Vector3 BaseColor { get; }
        public Vector3 EmissiveColor { get; }
        public float Alpha { get; }
        public uint BlendMode { get; }
        public int TransparentSortPriority { get; }
        public Vector3 TransparentSortCenter { get; }

        public void Dispose(GL gl)
        {
            if (Vbo != 0) gl.DeleteBuffer(Vbo);
            if (SkinningVbo != 0) gl.DeleteBuffer(SkinningVbo);
            if (Ebo != 0) gl.DeleteBuffer(Ebo);
            if (Vao != 0) gl.DeleteVertexArray(Vao);
        }
    }

    private struct TransparentDrawEntry
    {
        public bool IsWmo;
        public WmoCommandBuffers WmoCommand;
        public MdxCommandBuffers MdxCommand;
        public Matrix4x4 Transform;
        public float DistanceSq;
    }

    private enum WmoPreviewBlendMode
    {
        Opaque,
        Blend,
        Add,
        AlphaKey,
    }

}
