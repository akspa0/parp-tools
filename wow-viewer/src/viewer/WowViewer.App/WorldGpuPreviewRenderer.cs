using System.IO;
using System.Numerics;
using SereniaBLPLib;
using Silk.NET.OpenGL;

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
    }

    public uint PreviewTextureHandle => _colorTexture;

    public bool HasRenderableGeometry => _showSky || _terrainTiles.Count > 0 || _overlayVertexCount > 0 || _markerVertexCount > 0;

    public int TerrainTriangleCount => _terrainTiles.Sum(static tile => checked((int)(tile.IndexCount / 3)));

    public int MarkerCount => checked((int)_markerVertexCount);

    public float SceneScale => MathF.Max((_boundsMax - _boundsMin).Length(), 128f);

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        ClearPreview();

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
    }

    public void LoadPreview(WowViewerWorldRuntimeFrameResult frame, WorldViewCamera camera, bool ignoreTerrainHoles = false, bool showHoleOverlay = false)
    {
        ArgumentNullException.ThrowIfNull(frame);
        ArgumentNullException.ThrowIfNull(camera);

        ClearPreview();
        _showSky = frame.PassOptions.SkyVisible;
        ConfigureSkyColors(frame);
        BuildTerrainBuffers(frame, ignoreTerrainHoles);
        if (showHoleOverlay)
            BuildHoleOverlayBuffers(frame);
        BuildMarkerBuffers(frame);
        BuildCamera(frame, camera);
    }

    public unsafe void Render(int width, int height, WorldViewCamera camera)
    {
        ArgumentNullException.ThrowIfNull(camera);

        if (!HasRenderableGeometry)
            return;

        EnsureFramebuffer(width, height);
        BuildMatrices(width, height, camera, out Matrix4x4 view, out Matrix4x4 projection);
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
            _gl.Uniform3(_skyCameraPositionLocation, camera.Position.X, camera.Position.Y, camera.Position.Z);
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
            _gl.Uniform3(_terrainLightDirectionLocation, -0.45f, -0.55f, 0.70f);
            _gl.Uniform3(_terrainLightColorLocation, 0.80f, 0.82f, 0.78f);
            _gl.Uniform3(_terrainAmbientColorLocation, 0.28f, 0.30f, 0.34f);
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

        if (_markerVertexCount > 0)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.UseProgram(_markerProgram);
            _gl.UniformMatrix4(_markerViewLocation, 1, false, (float*)&view.M11);
            _gl.UniformMatrix4(_markerProjectionLocation, 1, false, (float*)&projection.M11);
            _gl.BindVertexArray(_markerVao);
            _gl.DrawArrays(PrimitiveType.Points, 0, _markerVertexCount);
            _gl.Disable(EnableCap.Blend);
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
        const int chunkAlphaSliceCount = 256;
        byte[] alphaShadow = new byte[ChunkAlphaSize * ChunkAlphaSize * 4 * chunkAlphaSliceCount];
        List<float> vertexData = [];
        List<byte> chunkSliceData = [];
        List<ushort> texIndexData = [];
        List<uint> indexData = [];

        foreach (var chunk in tile.TerrainTileData.Chunks)
        {
            if (!chunk.HasHeights || chunk.Heights is null)
                continue;

            Vector3[] positions = BuildChunkPositions(tile.TileX, tile.TileY, chunk);
            int[] chunkIndices = BuildChunkIndices(chunk.HoleMask, ignoreTerrainHoles);
            if (chunkIndices.Length == 0)
                continue;

            Vector3[] normals = BuildChunkNormals(chunkIndices, positions);
            int baseVertex = vertexData.Count / 11;
            byte chunkSlice = GetChunkSlice(chunk);
            ushort[] chunkTextureIndices = BuildChunkTextureIndices(chunk, textureIndices);

            for (int index = 0; index < positions.Length; index++)
            {
                Vector3 position = positions[index];
                Vector3 normal = normals[index];
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
                vertexData.Add(fallbackColor.X);
                vertexData.Add(fallbackColor.Y);
                vertexData.Add(fallbackColor.Z);

                chunkSliceData.Add(chunkSlice);
                texIndexData.Add(chunkTextureIndices[0]);
                texIndexData.Add(chunkTextureIndices[1]);
                texIndexData.Add(chunkTextureIndices[2]);
                texIndexData.Add(chunkTextureIndices[3]);
                ExpandBounds(position);
            }

            foreach (int localIndex in chunkIndices)
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

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, terrainTile.VboChunkSlice);
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

    private void BuildCamera(WowViewerWorldRuntimeFrameResult frame, WorldViewCamera camera)
    {
        Vector3 boundsCenter;
        if (_boundsMin.X == float.MaxValue || _boundsMax.X == float.MinValue)
        {
            Vector2 planarCenter = (frame.PlanarMin + frame.PlanarMax) * 0.5f;
            float centerHeight = frame.TerrainTileData.Heightmap?.CenterHeight ?? 0f;
            boundsCenter = new Vector3(planarCenter.X, planarCenter.Y, centerHeight);
            _boundsMin = new Vector3(frame.PlanarMin.X, frame.PlanarMin.Y, centerHeight - 32f);
            _boundsMax = new Vector3(frame.PlanarMax.X, frame.PlanarMax.Y, centerHeight + 32f);
        }
        else
        {
            boundsCenter = (_boundsMin + _boundsMax) * 0.5f;
        }

        Vector3 cameraTarget = frame.CameraTarget;
        if (cameraTarget.LengthSquared() <= 0.0001f)
            cameraTarget = boundsCenter;

        Vector3 cameraPosition;
        if (frame.CameraForward.LengthSquared() > 0.0001f)
        {
            Vector3 offset = frame.CameraPosition - cameraTarget;
            cameraPosition = offset.LengthSquared() > 1f
                ? frame.CameraPosition
                : cameraTarget - (frame.CameraForward * 900f) + new Vector3(0f, 0f, 220f);
        }
        else
        {
            Vector3 extent = _boundsMax - _boundsMin;
            float radius = MathF.Max(extent.Length() * 0.5f, 128f);
            cameraPosition = cameraTarget + new Vector3(-radius * 1.15f, -radius * 1.15f, radius * 0.60f);
        }

        camera.SetPose(cameraPosition, cameraTarget, saveAsDefault: true);
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
        float chunkWorldX = MapOrigin - (tileY * TileSize) - (chunk.IndexY * ChunkSize);
        float chunkWorldY = MapOrigin - (tileX * TileSize) - (chunk.IndexX * ChunkSize);
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
            return null;

        try
        {
            using MemoryStream stream = new(data, writable: false);
            using BlpFile blp = new(stream);
            byte[] rgbaPixels = blp.GetPixels(0, out int width, out int height, bgra: false);
            return new TerrainTextureSample(width, height, rgbaPixels);
        }
        catch
        {
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

    private static void FillAlphaShadowSlice(byte[] alphaShadow, byte slice, WowViewer.Core.Runtime.World.Terrain.WorldTerrainChunkData chunk)
    {
        int sliceBase = slice * ChunkAlphaSize * ChunkAlphaSize * 4;
        for (int layerIndex = 1; layerIndex <= 3; layerIndex++)
        {
            byte[]? alphaMap = layerIndex < chunk.TextureLayers.Count
                ? chunk.TextureLayers[layerIndex].DecodedAlpha?.AlphaMap
                : null;
            if (alphaMap is not { Length: ChunkAlphaSize * ChunkAlphaSize })
                continue;

            int channel = layerIndex - 1;
            for (int sampleIndex = 0; sampleIndex < alphaMap.Length; sampleIndex++)
                alphaShadow[sliceBase + (sampleIndex * 4) + channel] = alphaMap[sampleIndex];
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
            out vec3 vFallbackColor;

            void main()
            {
                vWorldPosition = aPosition;
                vNormal = aNormal;
                vTexCoord = aTexCoord;
                vChunkSlice = aChunkSlice;
                vTexIndices = aTexIndices;
                vFallbackColor = aFallbackColor;
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
            in vec3 vFallbackColor;

            uniform sampler2DArray uDiffuseArray;
            uniform sampler2DArray uAlphaShadowArray;
            uniform int uDiffuseLayerCount;
            uniform vec3 uLightDirection;
            uniform vec3 uLightColor;
            uniform vec3 uAmbientColor;

            out vec4 FragColor;

            bool HasLayer(uint textureIndex)
            {
                return textureIndex != 65535u && int(textureIndex) < uDiffuseLayerCount;
            }

            void main()
            {
                vec3 normal = normalize(vNormal);
                float ndotl = max(dot(normal, normalize(uLightDirection)), 0.0);
                vec3 lighting = uAmbientColor + (uLightColor * ndotl);
                float texScale = 8.0 / 33.333;
                vec2 diffuseUv = vec2(-vWorldPosition.y, -vWorldPosition.x) * texScale;
                vec4 alphaShadow = texture(uAlphaShadowArray, vec3(vTexCoord, float(vChunkSlice)));

                bool has0 = HasLayer(vTexIndices.x);
                bool has1 = HasLayer(vTexIndices.y);
                bool has2 = HasLayer(vTexIndices.z);
                bool has3 = HasLayer(vTexIndices.w);

                vec3 result = vFallbackColor * lighting;
                if (has0)
                    result = texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.x))).rgb * lighting;
                if (has1)
                    result = mix(result, texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.y))).rgb * lighting, alphaShadow.r);
                if (has2)
                    result = mix(result, texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.z))).rgb * lighting, alphaShadow.g);
                if (has3)
                    result = mix(result, texture(uDiffuseArray, vec3(diffuseUv, float(vTexIndices.w))).rgb * lighting, alphaShadow.b);

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

}
