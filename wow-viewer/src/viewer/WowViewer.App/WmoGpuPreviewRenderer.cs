using System.Numerics;
using SereniaBLPLib;
using Silk.NET.OpenGL;
using WowViewer.Core.Wmo;

namespace WowViewer.App;

internal sealed class WmoGpuPreviewRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly List<CommandBuffers> _commands = [];
    private readonly List<(int CommandIndex, float DistanceSquared)> _transparentCommandSortScratch = [];
    private readonly Dictionary<string, uint> _loadedTextureCache = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<uint> _ownedTextureIds = [];
    private uint _shaderProgram;
    private int _uView;
    private int _uProj;
    private int _uLightDir;
    private int _uAmbientColor;
    private int _uBaseColor;
    private int _uHasTexture;
    private int _uTexture0;
    private int _uAlphaTestThreshold;
    private int _uUseTextureAlpha;
    private uint _framebuffer;
    private uint _colorTexture;
    private uint _depthRenderbuffer;
    private uint _fallbackWhiteTexture;
    private int _frameWidth;
    private int _frameHeight;
    private Vector3 _boundsMin = new(-1f, -1f, -1f);
    private Vector3 _boundsMax = new(1f, 1f, 1f);
    private Vector3 _ambientColor = new(0.30f, 0.30f, 0.34f);
    private readonly Vector3 _lightDir = Vector3.Normalize(new Vector3(0.35f, 0.45f, 1.0f));
    private PreviewCameraSettings _cameraSettings = new()
    {
        Mode = PreviewCameraMode.Orbit,
        PresetName = null,
        ZoomFactor = 0.9f,
    };

    public WmoGpuPreviewRenderer(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
        InitializeShader();
        _fallbackWhiteTexture = CreateFallbackWhiteTexture();
    }

    public uint PreviewTextureHandle => _colorTexture;

    public bool HasRenderableGeometry => _commands.Count > 0;

    public int CommandCount => _commands.Count;

    public Vector3 BoundsMin => _boundsMin;

    public Vector3 BoundsMax => _boundsMax;

    public void Dispose()
    {
        ClearPreview();
        DeleteFramebuffer();
        foreach (uint textureId in _ownedTextureIds)
            _gl.DeleteTexture(textureId);

        _ownedTextureIds.Clear();
        _loadedTextureCache.Clear();

        if (_fallbackWhiteTexture != 0)
            _gl.DeleteTexture(_fallbackWhiteTexture);

        if (_shaderProgram != 0)
            _gl.DeleteProgram(_shaderProgram);
    }

    public void LoadPreview(WmoPreviewLoadResult preview)
    {
        ArgumentNullException.ThrowIfNull(preview);

        ClearPreview();
        _cameraSettings = preview.Request.Camera;
        BuildBuffers(preview);
    }

    public void SetCameraSettings(PreviewCameraSettings settings)
    {
        _cameraSettings = settings ?? throw new ArgumentNullException(nameof(settings));
    }

    public unsafe void Render(int width, int height)
    {
        if (!HasRenderableGeometry)
            return;

        EnsureFramebuffer(width, height);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);
        _gl.Viewport(0, 0, (uint)_frameWidth, (uint)_frameHeight);
        _gl.Enable(EnableCap.DepthTest);
        _gl.Disable(EnableCap.CullFace);
        _gl.ClearColor(0.07f, 0.08f, 0.10f, 1.0f);
        _gl.Clear((uint)(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit));

        PreviewCameraPose pose = PreviewCameraPlanner.CreatePose(_boundsMin, _boundsMax, _cameraSettings, null, null, 0, 0, _frameWidth, _frameHeight);
        _gl.UseProgram(_shaderProgram);
        Matrix4x4 view = pose.View;
        Matrix4x4 projection = pose.Projection;
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view.M11);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&projection.M11);
        _gl.Uniform3(_uLightDir, _lightDir.X, _lightDir.Y, _lightDir.Z);
        _gl.Uniform3(_uAmbientColor, _ambientColor.X, _ambientColor.Y, _ambientColor.Z);
        _gl.Uniform1(_uTexture0, 0);

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);

        RenderPass(transparentPass: false, pose.CameraPosition);
        RenderPass(transparentPass: true, pose.CameraPosition);

        _gl.BindTexture(TextureTarget.Texture2D, 0);
        _gl.BindVertexArray(0);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    public void CaptureBmp(string outputPath, int width, int height)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        Render(width, height);

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);
        byte[] rgbaPixels = new byte[_frameWidth * _frameHeight * 4];
        unsafe
        {
            fixed (byte* pixelPtr = rgbaPixels)
            {
                _gl.ReadPixels(0, 0, (uint)_frameWidth, (uint)_frameHeight, PixelFormat.Rgba, PixelType.UnsignedByte, pixelPtr);
            }
        }

        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
        ImageOutputWriter.WriteRgbaImage(outputPath, _frameWidth, _frameHeight, rgbaPixels, sourceOriginBottomLeft: true);
    }

    private unsafe void RenderPass(bool transparentPass, Vector3 cameraPosition)
    {
        if (transparentPass)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.DepthMask(false);
        }
        else
        {
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
        }

        if (transparentPass)
        {
            _transparentCommandSortScratch.Clear();
            for (int index = 0; index < _commands.Count; index++)
            {
                CommandBuffers command = _commands[index];
                if (!command.IsTransparent)
                    continue;

                float distanceSquared = Vector3.DistanceSquared(cameraPosition, command.SortCenter);
                _transparentCommandSortScratch.Add((index, distanceSquared));
            }

            _transparentCommandSortScratch.Sort(static (left, right) => right.DistanceSquared.CompareTo(left.DistanceSquared));

            foreach ((int commandIndex, _) in _transparentCommandSortScratch)
                RenderCommand(_commands[commandIndex], transparentPass);

            return;
        }

        foreach (CommandBuffers command in _commands)
        {
            if (command.IsTransparent)
                continue;

            RenderCommand(command, transparentPass);
        }
    }

    private unsafe void RenderCommand(CommandBuffers command, bool transparentPass)
    {
        _gl.Uniform3(_uBaseColor, command.BaseColor.X, command.BaseColor.Y, command.BaseColor.Z);
        _gl.Uniform1(_uHasTexture, command.HasTexture ? 1 : 0);
        _gl.Uniform1(_uAlphaTestThreshold, command.AlphaTestThreshold);
        _gl.Uniform1(_uUseTextureAlpha, command.UseTextureAlpha ? 1 : 0);
        if (transparentPass)
        {
            _gl.BlendFunc(command.SourceBlendFactor, command.DestinationBlendFactor);
        }

        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindTexture(TextureTarget.Texture2D, command.TextureId);
        _gl.BindVertexArray(command.Vao);
        _gl.DrawElements(PrimitiveType.Triangles, command.IndexCount, DrawElementsType.UnsignedShort, null);
    }

    public void ClearPreview()
    {
        foreach (CommandBuffers command in _commands)
        {
            _gl.DeleteBuffer(command.Vbo);
            _gl.DeleteBuffer(command.Ebo);
            _gl.DeleteVertexArray(command.Vao);
        }

        _commands.Clear();
        _boundsMin = new(-1f, -1f, -1f);
        _boundsMax = new(1f, 1f, 1f);
    }

    private unsafe void BuildBuffers(WmoPreviewLoadResult preview)
    {
        WmoRenderDocument document = preview.Document;
        bool hasBounds = false;
        Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);

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
                    && TryGetOrLoadMaterialTexture(preview.Request, document.Materials[materialIndex], out loadedTextureId);

                WmoMaterialDetail? material = materialIndex >= 0 && materialIndex < document.Materials.Count
                    ? document.Materials[materialIndex]
                    : null;

                WmoPreviewBlendMode blendMode = ResolveBlendMode(material?.BlendMode ?? 0);

                CreateCommand(
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

                builtBatchCommand = true;
            }

            if (!builtBatchCommand)
            {
                CreateCommand(
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
            }
        }

        if (hasBounds)
        {
            _boundsMin = boundsMin;
            _boundsMax = boundsMax;
        }
    }

    private unsafe void CreateCommand(
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
        if (indices.Length < 3)
            return;

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
        _commands.Add(new CommandBuffers(vao, vbo, ebo, (uint)indices.Length, sortCenter, textureId, hasTexture, isTransparent, alphaTestThreshold, useTextureAlpha, sourceBlendFactor, destinationBlendFactor, baseColor));
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

    private bool TryGetOrLoadMaterialTexture(WmoPreviewLoadRequest request, WmoMaterialDetail material, out uint textureId)
    {
        textureId = 0;
        foreach (string candidate in EnumerateTextureCandidates(material))
        {
            if (!TryGetOrLoadTexture(request, candidate, out uint loadedTextureId))
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

    private static string EnsureBlpExtension(string texturePath)
    {
        return texturePath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase)
            ? texturePath
            : $"{texturePath}.blp";
    }

    private bool TryGetOrLoadTexture(WmoPreviewLoadRequest request, string texturePath, out uint textureId)
    {
        string cacheKey = texturePath.Replace('/', '\\').ToLowerInvariant();
        if (_loadedTextureCache.TryGetValue(cacheKey, out textureId))
            return textureId != 0;

        if (!TryReadTextureBytes(request, texturePath, out byte[]? bytes) || bytes == null || bytes.Length == 0)
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

    private static bool TryReadTextureBytes(WmoPreviewLoadRequest request, string texturePath, out byte[]? bytes)
    {
        bytes = null;

        if (request.UsesArchiveSource)
        {
            try
            {
                bytes = VirtualAssetOverlayResolver.ReadVirtualFilePreferLoose(texturePath, request.ArchiveRoot!, request.LooseOverlayRoot, request.BuildLabel);
                return bytes.Length > 0;
            }
            catch
            {
                return false;
            }
        }

        if (string.IsNullOrWhiteSpace(request.InputPath))
            return false;

        string normalized = texturePath.Replace('/', '\\').TrimStart('\\');
        string? modelDirectory = Path.GetDirectoryName(Path.GetFullPath(request.InputPath));
        string[] candidates =
        [
            normalized,
            modelDirectory == null ? normalized : Path.Combine(modelDirectory, Path.GetFileName(normalized)),
            modelDirectory == null ? normalized : Path.Combine(modelDirectory, normalized),
        ];

        foreach (string candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!File.Exists(candidate))
                continue;

            bytes = File.ReadAllBytes(candidate);
            return bytes.Length > 0;
        }

        return false;
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

    private unsafe void EnsureFramebuffer(int width, int height)
    {
        width = Math.Max(width, 16);
        height = Math.Max(height, 16);
        if (_framebuffer != 0 && _frameWidth == width && _frameHeight == height)
            return;

        DeleteFramebuffer();
        _frameWidth = width;
        _frameHeight = height;
        _framebuffer = _gl.GenFramebuffer();
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, _framebuffer);

        _colorTexture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2D, _colorTexture);
        _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba8, (uint)_frameWidth, (uint)_frameHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);
        _gl.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, TextureTarget.Texture2D, _colorTexture, 0);

        _depthRenderbuffer = _gl.GenRenderbuffer();
        _gl.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _depthRenderbuffer);
        _gl.RenderbufferStorage(RenderbufferTarget.Renderbuffer, InternalFormat.DepthComponent24, (uint)_frameWidth, (uint)_frameHeight);
        _gl.FramebufferRenderbuffer(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, RenderbufferTarget.Renderbuffer, _depthRenderbuffer);
        _gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    private void DeleteFramebuffer()
    {
        if (_depthRenderbuffer != 0)
            _gl.DeleteRenderbuffer(_depthRenderbuffer);
        if (_colorTexture != 0)
            _gl.DeleteTexture(_colorTexture);
        if (_framebuffer != 0)
            _gl.DeleteFramebuffer(_framebuffer);

        _depthRenderbuffer = 0;
        _colorTexture = 0;
        _framebuffer = 0;
        _frameWidth = 0;
        _frameHeight = 0;
    }

    private void InitializeShader()
    {
        const string vertexShaderSource = """
            #version 330 core
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec3 aNormal;
            layout(location = 2) in vec2 aTexCoord;
            uniform mat4 uView;
            uniform mat4 uProj;
            out vec3 vNormal;
            out vec2 vTexCoord;
            void main()
            {
                vNormal = aNormal;
                vTexCoord = aTexCoord;
                gl_Position = uProj * uView * vec4(aPosition, 1.0);
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

        uint vertexShader = CompileShader(ShaderType.VertexShader, vertexShaderSource);
        uint fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentShaderSource);
        _shaderProgram = _gl.CreateProgram();
        _gl.AttachShader(_shaderProgram, vertexShader);
        _gl.AttachShader(_shaderProgram, fragmentShader);
        _gl.LinkProgram(_shaderProgram);
        _gl.GetProgram(_shaderProgram, ProgramPropertyARB.LinkStatus, out int linked);
        if (linked == 0)
        {
            string info = _gl.GetProgramInfoLog(_shaderProgram);
            throw new InvalidOperationException($"WMO preview shader link failed: {info}");
        }

        _gl.DeleteShader(vertexShader);
        _gl.DeleteShader(fragmentShader);
        _uView = _gl.GetUniformLocation(_shaderProgram, "uView");
        _uProj = _gl.GetUniformLocation(_shaderProgram, "uProj");
        _uLightDir = _gl.GetUniformLocation(_shaderProgram, "uLightDir");
        _uAmbientColor = _gl.GetUniformLocation(_shaderProgram, "uAmbientColor");
        _uBaseColor = _gl.GetUniformLocation(_shaderProgram, "uBaseColor");
        _uHasTexture = _gl.GetUniformLocation(_shaderProgram, "uHasTexture");
        _uTexture0 = _gl.GetUniformLocation(_shaderProgram, "uTexture0");
        _uAlphaTestThreshold = _gl.GetUniformLocation(_shaderProgram, "uAlphaTestThreshold");
        _uUseTextureAlpha = _gl.GetUniformLocation(_shaderProgram, "uUseTextureAlpha");
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
        return textureId;
    }

    private uint CompileShader(ShaderType shaderType, string source)
    {
        uint shader = _gl.CreateShader(shaderType);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int compiled);
        if (compiled == 0)
        {
            string info = _gl.GetShaderInfoLog(shader);
            throw new InvalidOperationException($"WMO preview shader compile failed: {info}");
        }

        return shader;
    }

    private static Vector3 ComputeGroupColor(int groupIndex)
    {
        float red = ((groupIndex * 67 + 13) % 255) / 255f;
        float green = ((groupIndex * 131 + 7) % 255) / 255f;
        float blue = ((groupIndex * 43 + 29) % 255) / 255f;
        return new Vector3(red, green, blue);
    }

    private sealed record CommandBuffers(
        uint Vao,
        uint Vbo,
        uint Ebo,
        uint IndexCount,
        Vector3 SortCenter,
        uint TextureId,
        bool HasTexture,
        bool IsTransparent,
        float AlphaTestThreshold,
        bool UseTextureAlpha,
        BlendingFactor SourceBlendFactor,
        BlendingFactor DestinationBlendFactor,
        Vector3 BaseColor);

    private enum WmoPreviewBlendMode
    {
        Opaque,
        Blend,
        Add,
        AlphaKey,
    }
}