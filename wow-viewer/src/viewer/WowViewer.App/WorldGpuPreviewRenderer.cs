using System.Numerics;
using Silk.NET.OpenGL;

namespace WowViewer.App;

internal sealed class WorldGpuPreviewRenderer : IDisposable
{
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16.0f;
    private const float ChunkSubCellSize = ChunkSize / 8.0f;
    private const float MapOrigin = 32.0f * TileSize;

    private readonly GL _gl;
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
    private uint _terrainVao;
    private uint _terrainVbo;
    private uint _terrainEbo;
    private uint _terrainIndexCount;
    private uint _overlayVao;
    private uint _overlayVbo;
    private uint _overlayVertexCount;
    private uint _markerVao;
    private uint _markerVbo;
    private uint _markerVertexCount;
    private Vector3 _boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
    private Vector3 _boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
    private readonly WorldPreviewCameraState _camera = new();
    private bool _disposed;

    public WorldGpuPreviewRenderer(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        InitializeTerrainShader();
        InitializeOverlayShader();
        InitializeMarkerShader();
    }

    public uint PreviewTextureHandle => _colorTexture;

    public bool HasRenderableGeometry => _terrainIndexCount > 0 || _overlayVertexCount > 0 || _markerVertexCount > 0;

    public int TerrainTriangleCount => checked((int)(_terrainIndexCount / 3));

    public int MarkerCount => checked((int)_markerVertexCount);

    public float SceneScale => MathF.Max((_boundsMax - _boundsMin).Length(), 128f);

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        ClearPreview();

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
        _camera.ResetToIdentity();
    }

    public void LoadPreview(WowViewerWorldRuntimeFrameResult frame, bool ignoreTerrainHoles = false, bool showHoleOverlay = false)
    {
        ArgumentNullException.ThrowIfNull(frame);

        ClearPreview();
        BuildTerrainBuffers(frame, ignoreTerrainHoles);
        if (showHoleOverlay)
            BuildHoleOverlayBuffers(frame);
        BuildMarkerBuffers(frame);
        BuildCamera(frame);
    }

    public void ResetCamera()
    {
        _camera.Reset();
    }

    public void RotateCamera(float azimuthDeltaDegrees, float elevationDeltaDegrees)
    {
        _camera.RotateLook(azimuthDeltaDegrees, elevationDeltaDegrees);
    }

    public void TranslateCamera(float forwardDistance, float strafeDistance, float verticalDistance)
    {
        _camera.Translate(forwardDistance, strafeDistance, verticalDistance);
    }

    public unsafe void Render(int width, int height)
    {
        if (!HasRenderableGeometry)
            return;

        EnsureFramebuffer(width, height);
        BuildMatrices(width, height, out Matrix4x4 view, out Matrix4x4 projection);

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Disable(EnableCap.CullFace);
        _gl.ClearColor(0.07f, 0.09f, 0.12f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        if (_terrainIndexCount > 0)
        {
            _gl.UseProgram(_terrainProgram);
            _gl.UniformMatrix4(_terrainViewLocation, 1, false, (float*)&view.M11);
            _gl.UniformMatrix4(_terrainProjectionLocation, 1, false, (float*)&projection.M11);
            _gl.Uniform3(_terrainLightDirectionLocation, -0.45f, -0.55f, 0.70f);
            _gl.Uniform3(_terrainLightColorLocation, 0.80f, 0.82f, 0.78f);
            _gl.Uniform3(_terrainAmbientColorLocation, 0.28f, 0.30f, 0.34f);
            _gl.BindVertexArray(_terrainVao);
            _gl.DrawElements(PrimitiveType.Triangles, _terrainIndexCount, DrawElementsType.UnsignedInt, null);
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
        float minHeight = frame.TerrainTileData.Heightmap?.MinHeight ?? 0f;
        float maxHeight = frame.TerrainTileData.Heightmap?.MaxHeight ?? 0f;
        float heightRange = MathF.Max(maxHeight - minHeight, 1.0f);

        List<float> vertexData = [];
        List<uint> indexData = [];

        foreach (var chunk in frame.TerrainTileData.Chunks)
        {
            if (!chunk.HasHeights || chunk.Heights is null)
                continue;

            Vector3[] positions = BuildChunkPositions(frame.SelectedTileX, frame.SelectedTileY, chunk);
            int[] chunkIndices = BuildChunkIndices(chunk.HoleMask, ignoreTerrainHoles);
            Vector3[] normals = BuildChunkNormals(chunkIndices, positions);
            int baseVertex = vertexData.Count / 9;
            for (int index = 0; index < positions.Length; index++)
            {
                Vector3 position = positions[index];
                Vector3 normal = normals[index];
                Vector3 color = ComputeTerrainColor((position.Z - minHeight) / heightRange, Math.Clamp(1.0f - normal.Z, 0.0f, 1.0f));

                vertexData.Add(position.X);
                vertexData.Add(position.Y);
                vertexData.Add(position.Z);
                vertexData.Add(normal.X);
                vertexData.Add(normal.Y);
                vertexData.Add(normal.Z);
                vertexData.Add(color.X);
                vertexData.Add(color.Y);
                vertexData.Add(color.Z);

                ExpandBounds(position);
            }

            foreach (int localIndex in chunkIndices)
                indexData.Add((uint)(baseVertex + localIndex));
        }

        if (indexData.Count == 0)
            return;

        _terrainIndexCount = (uint)indexData.Count;
        _terrainVao = _gl.GenVertexArray();
        _terrainVbo = _gl.GenBuffer();
        _terrainEbo = _gl.GenBuffer();

        _gl.BindVertexArray(_terrainVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _terrainVbo);
        float[] vertexArray = vertexData.ToArray();
        fixed (float* vertexPtr = vertexArray)
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertexArray.Length * sizeof(float)), vertexPtr, BufferUsageARB.StaticDraw);
        }

        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, _terrainEbo);
        uint[] indexArray = indexData.ToArray();
        fixed (uint* indexPtr = indexArray)
        {
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indexArray.Length * sizeof(uint)), indexPtr, BufferUsageARB.StaticDraw);
        }

        const uint stride = 9u * sizeof(float);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(2, 3, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));
        _gl.EnableVertexAttribArray(2);
        _gl.BindVertexArray(0);
    }

    private unsafe void BuildHoleOverlayBuffers(WowViewerWorldRuntimeFrameResult frame)
    {
        List<float> overlayData = [];

        foreach (var chunk in frame.TerrainTileData.Chunks)
        {
            if (!chunk.HasHeights || chunk.Heights is null || !chunk.HasHoles)
                continue;

            Vector3[] positions = BuildChunkPositions(frame.SelectedTileX, frame.SelectedTileY, chunk);
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

    private void BuildCamera(WowViewerWorldRuntimeFrameResult frame)
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

        _camera.SetPose(cameraPosition, cameraTarget, saveAsDefault: true);
    }

    private void BuildMatrices(int width, int height, out Matrix4x4 view, out Matrix4x4 projection)
    {
        Vector3 forward = _camera.GetForwardVector();
        Vector3 worldUp = MathF.Abs(Vector3.Dot(forward, Vector3.UnitZ)) > 0.98f
            ? Vector3.UnitY
            : Vector3.UnitZ;
        view = Matrix4x4.CreateLookAt(_camera.Position, _camera.Target, worldUp);

        Vector3 extent = _boundsMax - _boundsMin;
        float radius = MathF.Max(extent.Length() * 0.5f, 128f);
        float distance = Vector3.Distance(_camera.Position, _camera.Target);
        float aspect = Math.Max(width, 1) / (float)Math.Max(height, 1);
        float farPlane = MathF.Max(2048f, distance + (radius * 4.0f));
        projection = Matrix4x4.CreatePerspectiveFieldOfView(50.0f * MathF.PI / 180.0f, aspect, 1.0f, farPlane);
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

    private void ExpandBounds(Vector3 position)
    {
        _boundsMin = Vector3.Min(_boundsMin, position);
        _boundsMax = Vector3.Max(_boundsMax, position);
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
            layout (location = 2) in vec3 aColor;

            uniform mat4 uView;
            uniform mat4 uProjection;

            out vec3 vNormal;
            out vec3 vColor;

            void main()
            {
                vNormal = aNormal;
                vColor = aColor;
                gl_Position = uProjection * uView * vec4(aPosition, 1.0);
            }
            """;

        const string fragmentSource = """
            #version 330 core
            in vec3 vNormal;
            in vec3 vColor;

            uniform vec3 uLightDirection;
            uniform vec3 uLightColor;
            uniform vec3 uAmbientColor;

            out vec4 FragColor;

            void main()
            {
                vec3 normal = normalize(vNormal);
                float ndotl = max(dot(normal, normalize(uLightDirection)), 0.0);
                vec3 litColor = vColor * (uAmbientColor + (uLightColor * ndotl));
                FragColor = vec4(litColor, 1.0);
            }
            """;

        _terrainProgram = CreateProgram(vertexSource, fragmentSource, "world terrain");
        _terrainViewLocation = _gl.GetUniformLocation(_terrainProgram, "uView");
        _terrainProjectionLocation = _gl.GetUniformLocation(_terrainProgram, "uProjection");
        _terrainLightDirectionLocation = _gl.GetUniformLocation(_terrainProgram, "uLightDirection");
        _terrainLightColorLocation = _gl.GetUniformLocation(_terrainProgram, "uLightColor");
        _terrainAmbientColorLocation = _gl.GetUniformLocation(_terrainProgram, "uAmbientColor");
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
        if (_terrainEbo != 0)
        {
            _gl.DeleteBuffer(_terrainEbo);
            _terrainEbo = 0;
        }

        if (_terrainVbo != 0)
        {
            _gl.DeleteBuffer(_terrainVbo);
            _terrainVbo = 0;
        }

        if (_terrainVao != 0)
        {
            _gl.DeleteVertexArray(_terrainVao);
            _terrainVao = 0;
        }

        _terrainIndexCount = 0;
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

    private sealed class WorldPreviewCameraState
    {
        private static readonly Vector3 IdentityPosition = new(0f, 0f, 1f);
        private static readonly Vector3 IdentityTarget = Vector3.Zero;

        public Vector3 Position { get; private set; } = IdentityPosition;

        public Vector3 Target { get; private set; } = IdentityTarget;

        private Vector3 DefaultPosition { get; set; } = IdentityPosition;

        private Vector3 DefaultTarget { get; set; } = IdentityTarget;

        public void ResetToIdentity()
        {
            Position = IdentityPosition;
            Target = IdentityTarget;
            DefaultPosition = IdentityPosition;
            DefaultTarget = IdentityTarget;
        }

        public void SetPose(Vector3 position, Vector3 target, bool saveAsDefault)
        {
            Position = position;
            Target = target;
            if (saveAsDefault)
            {
                DefaultPosition = position;
                DefaultTarget = target;
            }
        }

        public void Reset()
        {
            Position = DefaultPosition;
            Target = DefaultTarget;
        }

        public Vector3 GetForwardVector()
        {
            Vector3 forward = Target - Position;
            if (forward.LengthSquared() <= 1e-6f)
                return Vector3.Normalize(new Vector3(1f, 1f, -0.3f));

            return Vector3.Normalize(forward);
        }

        public void RotateLook(float azimuthDeltaDegrees, float elevationDeltaDegrees)
        {
            Vector3 forward = GetForwardVector();
            float distance = MathF.Max(Vector3.Distance(Position, Target), 1.0f);
            GetCameraAngles(forward, out float azimuthDegrees, out float elevationDegrees);
            azimuthDegrees -= azimuthDeltaDegrees;
            elevationDegrees = Math.Clamp(elevationDegrees + elevationDeltaDegrees, -89.0f, 89.0f);
            Vector3 rotatedForward = ComputeForwardVector(azimuthDegrees, elevationDegrees);
            Target = Position + (rotatedForward * distance);
        }

        public void Translate(float forwardDistance, float strafeDistance, float verticalDistance)
        {
            BuildBasis(out Vector3 forward, out Vector3 right, out _);
            Vector3 delta = (forward * forwardDistance) + (right * strafeDistance) + (Vector3.UnitZ * verticalDistance);
            Position += delta;
            Target += delta;
        }

        private void BuildBasis(out Vector3 forward, out Vector3 right, out Vector3 up)
        {
            forward = GetForwardVector();
            Vector3 worldUp = MathF.Abs(Vector3.Dot(forward, Vector3.UnitZ)) > 0.98f
                ? Vector3.UnitY
                : Vector3.UnitZ;

            right = Vector3.Cross(worldUp, forward);
            if (right.LengthSquared() <= 1e-6f)
                right = Vector3.UnitX;
            else
                right = Vector3.Normalize(right);

            up = Vector3.Cross(forward, right);
            if (up.LengthSquared() <= 1e-6f)
                up = worldUp;
            else
                up = Vector3.Normalize(up);
        }
    }
}