using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Rendering;
using SereniaBLPLib;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders terrain chunks using multi-pass texture layering with alpha blending.
/// Base layer is rendered opaque, subsequent layers blend using alpha maps from MCAL.
/// </summary>
public class TerrainRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly ShaderProgram _shader;
    private readonly IDataSource? _dataSource;
    private readonly TerrainLighting _lighting;

    // Texture cache: texture name → GL handle
    private readonly Dictionary<string, uint> _textureCache = new(StringComparer.OrdinalIgnoreCase);

    // Loaded chunk meshes
    private readonly List<TerrainChunkMesh> _chunks = new();

    // Tile texture name tables
    private readonly Dictionary<(int, int), List<string>> _tileTextures = new();

    private bool _wireframe;

    // Layer visibility toggles (exposed for UI)
    public bool ShowLayer0 { get; set; } = true;
    public bool ShowLayer1 { get; set; } = true;
    public bool ShowLayer2 { get; set; } = true;
    public bool ShowLayer3 { get; set; } = true;

    // Grid overlay
    public bool ShowChunkGrid { get; set; } = false;
    public bool ShowTileGrid { get; set; } = false;

    public int LoadedChunkCount => _chunks.Count;
    public TerrainLighting Lighting => _lighting;

    public TerrainRenderer(GL gl, IDataSource? dataSource, TerrainLighting lighting)
    {
        _gl = gl;
        _dataSource = dataSource;
        _lighting = lighting;
        _shader = CreateTerrainShader();
    }

    /// <summary>
    /// Add chunk meshes and their texture tables for rendering.
    /// </summary>
    public void AddChunks(List<TerrainChunkMesh> chunks, Dictionary<(int, int), List<string>> tileTextures)
    {
        foreach (var chunk in chunks)
        {
            chunk.Gl = _gl;
            UploadAlphaTextures(chunk);
        }
        _chunks.AddRange(chunks);

        foreach (var kvp in tileTextures)
        {
            _tileTextures[kvp.Key] = kvp.Value;
            // Pre-load textures for this tile
            foreach (var texName in kvp.Value)
                GetOrLoadTexture(texName);
        }

        Console.WriteLine($"[TerrainRenderer] Now rendering {_chunks.Count} chunks, {_textureCache.Count} textures cached");
    }

    /// <summary>
    /// Render all loaded terrain chunks.
    /// </summary>
    public unsafe void Render(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos)
    {
        if (_chunks.Count == 0) return;

        _lighting.Update();

        _shader.Use();
        _shader.SetMat4("uView", view);
        _shader.SetMat4("uProj", proj);
        _shader.SetMat4("uModel", Matrix4x4.Identity);
        _shader.SetVec3("uLightDir", _lighting.LightDirection);
        _shader.SetVec3("uLightColor", _lighting.LightColor);
        _shader.SetVec3("uAmbientColor", _lighting.AmbientColor);
        _shader.SetVec3("uFogColor", _lighting.FogColor);
        _shader.SetFloat("uFogStart", _lighting.FogStart);
        _shader.SetFloat("uFogEnd", _lighting.FogEnd);
        _shader.SetVec3("uCameraPos", cameraPos);

        // Disable face culling for terrain (winding varies with coordinate system)
        _gl.Disable(EnableCap.CullFace);

        if (_wireframe)
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Line);
        else
            _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);

        // Pass grid uniforms
        _shader.SetInt("uShowChunkGrid", ShowChunkGrid ? 1 : 0);
        _shader.SetInt("uShowTileGrid", ShowTileGrid ? 1 : 0);

        // Render each chunk
        foreach (var chunk in _chunks)
        {
            RenderChunk(chunk);
        }

        _gl.PolygonMode(TriangleFace.FrontAndBack, PolygonMode.Fill);
        _gl.Enable(EnableCap.CullFace);
    }

    private void RenderChunk(TerrainChunkMesh chunk)
    {
        var texNames = _tileTextures.GetValueOrDefault((chunk.TileX, chunk.TileY));
        if (texNames == null || texNames.Count == 0 || chunk.Layers.Length == 0)
        {
            // No textures — render with flat color
            RenderChunkPass(chunk, 0, isBaseLayer: true);
            return;
        }

        bool[] layerVisible = { ShowLayer0, ShowLayer1, ShowLayer2, ShowLayer3 };

        // Base layer (opaque, no alpha map)
        if (layerVisible[0])
        {
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
            RenderChunkPass(chunk, 0, isBaseLayer: true);
        }

        // Alpha-blended overlay layers
        if (chunk.Layers.Length > 1)
        {
            _gl.Enable(EnableCap.Blend);
            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            _gl.DepthMask(false);
            _gl.DepthFunc(DepthFunction.Lequal);

            for (int layer = 1; layer < chunk.Layers.Length && layer < 4; layer++)
            {
                if (!layerVisible[layer]) continue;
                RenderChunkPass(chunk, layer, isBaseLayer: false);
            }

            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);
        }
    }

    private unsafe void RenderChunkPass(TerrainChunkMesh chunk, int layerIndex, bool isBaseLayer)
    {
        _shader.SetInt("uIsBaseLayer", isBaseLayer ? 1 : 0);

        // Bind diffuse texture for this layer
        bool hasTex = false;
        if (layerIndex < chunk.Layers.Length)
        {
            var texNames = _tileTextures.GetValueOrDefault((chunk.TileX, chunk.TileY));
            int texIdx = chunk.Layers[layerIndex].TextureIndex;
            if (texNames != null && texIdx >= 0 && texIdx < texNames.Count)
            {
                uint glTex = GetOrLoadTexture(texNames[texIdx]);
                if (glTex != 0)
                {
                    _gl.ActiveTexture(TextureUnit.Texture0);
                    _gl.BindTexture(TextureTarget.Texture2D, glTex);
                    hasTex = true;
                }
            }
        }
        _shader.SetInt("uHasTexture", hasTex ? 1 : 0);
        _shader.SetInt("uDiffuseSampler", 0);

        // Bind alpha map for overlay layers
        bool hasAlpha = false;
        if (!isBaseLayer && chunk.AlphaTextures.TryGetValue(layerIndex, out uint alphaTex))
        {
            _gl.ActiveTexture(TextureUnit.Texture1);
            _gl.BindTexture(TextureTarget.Texture2D, alphaTex);
            hasAlpha = true;
        }
        _shader.SetInt("uHasAlphaMap", hasAlpha ? 1 : 0);
        _shader.SetInt("uAlphaSampler", 1);

        // Draw
        _gl.BindVertexArray(chunk.Vao);
        _gl.DrawElements(PrimitiveType.Triangles, chunk.IndexCount, DrawElementsType.UnsignedShort, null);
        _gl.BindVertexArray(0);
    }

    public void ToggleWireframe()
    {
        _wireframe = !_wireframe;
    }

    private uint GetOrLoadTexture(string textureName)
    {
        if (string.IsNullOrEmpty(textureName)) return 0;
        if (_textureCache.TryGetValue(textureName, out uint cached)) return cached;

        uint glTex = LoadTerrainTexture(textureName);
        _textureCache[textureName] = glTex;
        return glTex;
    }

    private unsafe uint LoadTerrainTexture(string textureName)
    {
        byte[]? blpData = null;

        // Try data source (MPQ)
        if (_dataSource != null)
        {
            blpData = _dataSource.ReadFile(textureName);
            if (blpData == null)
            {
                // Try with normalized path
                blpData = _dataSource.ReadFile(textureName.Replace('/', '\\'));
            }
        }

        if (blpData == null || blpData.Length == 0)
        {
            // Return 0 — will render as untextured
            return 0;
        }

        try
        {
            using var ms = new MemoryStream(blpData);
            using var blp = new BlpFile(ms);
            var bmp = blp.GetBitmap(0);

            int w = bmp.Width, h = bmp.Height;
            var pixels = new byte[w * h * 4];
            var rect = new System.Drawing.Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            try
            {
                var srcBytes = new byte[data.Stride * h];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, srcBytes, 0, srcBytes.Length);

                // BGRA → RGBA
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int srcIdx = y * data.Stride + x * 4;
                        int dstIdx = (y * w + x) * 4;
                        pixels[dstIdx + 0] = srcBytes[srcIdx + 2]; // R
                        pixels[dstIdx + 1] = srcBytes[srcIdx + 1]; // G
                        pixels[dstIdx + 2] = srcBytes[srcIdx + 0]; // B
                        pixels[dstIdx + 3] = srcBytes[srcIdx + 3]; // A
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(data);
            }
            bmp.Dispose();

            // Upload to GPU with repeat wrapping (terrain tiles)
            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            _gl.GenerateMipmap(TextureTarget.Texture2D);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            return tex;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[TerrainRenderer] Failed to load texture {textureName}: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Upload alpha map byte arrays as GL textures (R8, 64×64).
    /// </summary>
    private unsafe void UploadAlphaTextures(TerrainChunkMesh chunk)
    {
        foreach (var kvp in chunk.AlphaMaps)
        {
            int layer = kvp.Key;
            byte[] alphaData = kvp.Value;

            // Alpha maps are 64×64 (4096 bytes after expansion)
            int size = 64;
            if (alphaData.Length < size * size)
                continue;

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);

            fixed (byte* ptr = alphaData)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.R8,
                    (uint)size, (uint)size, 0, PixelFormat.Red, PixelType.UnsignedByte, ptr);

            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            chunk.AlphaTextures[layer] = tex;
        }
    }

    private ShaderProgram CreateTerrainShader()
    {
        string vertSrc = @"
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = mat3(uModel) * aNormal;
    vTexCoord = aTexCoord;
    gl_Position = uProj * uView * worldPos;
}
";

        string fragSrc = @"
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;

uniform sampler2D uDiffuseSampler;
uniform sampler2D uAlphaSampler;
uniform int uHasTexture;
uniform int uHasAlphaMap;
uniform int uIsBaseLayer;
uniform int uShowChunkGrid;
uniform int uShowTileGrid;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform vec3 uFogColor;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec3 uCameraPos;

out vec4 FragColor;

void main() {
    // Diffuse texture: use WORLD-SPACE UVs for seamless tiling across chunks.
    // WoW terrain textures tile ~8 times per chunk (chunk = ChunkSize/16 = 33.333 units).
    // So texture repeats every 33.333/8 = ~4.167 units.
    float texScale = 8.0 / 33.333;
    vec2 worldUV = vWorldPos.xy * texScale;

    vec4 texColor;
    if (uHasTexture == 1) {
        texColor = texture(uDiffuseSampler, worldUV);
    } else {
        texColor = vec4(0.3, 0.5, 0.2, 1.0);
    }

    // Alpha from alpha map (overlay layers only)
    // Alpha maps use per-chunk 0-1 UVs (vTexCoord), which is correct.
    float alpha = 1.0;
    if (uIsBaseLayer == 0 && uHasAlphaMap == 1) {
        float halfTexel = 0.5 / 64.0;
        vec2 alphaUV = vTexCoord * (1.0 - 2.0 * halfTexel) + halfTexel;
        alpha = texture(uAlphaSampler, alphaUV).r;
        if (alpha < 0.004) discard;
    }

    // Lighting
    vec3 norm = normalize(vNormal);
    float diff = abs(dot(norm, normalize(uLightDir)));
    vec3 lighting = uAmbientColor + uLightColor * diff;
    vec3 litColor = texColor.rgb * lighting;

    // Fog
    float dist = length(vWorldPos - uCameraPos);
    float fogFactor = clamp((uFogEnd - dist) / (uFogEnd - uFogStart), 0.0, 1.0);
    vec3 finalColor = mix(uFogColor, litColor, fogFactor);

    // Grid overlays (drawn on base layer only to avoid double-drawing)
    if (uIsBaseLayer == 1) {
        // Chunk grid: chunk size = 33.333 units
        if (uShowChunkGrid == 1) {
            float chunkSize = 33.333;
            vec2 chunkFrac = fract(vWorldPos.xy / chunkSize);
            float chunkLine = step(chunkFrac.x, 0.005) + step(1.0 - chunkFrac.x, 0.005)
                            + step(chunkFrac.y, 0.005) + step(1.0 - chunkFrac.y, 0.005);
            chunkLine = clamp(chunkLine, 0.0, 1.0);
            finalColor = mix(finalColor, vec3(0.0, 1.0, 1.0), chunkLine * 0.6);
        }
        // Tile grid: tile size = 533.333 units
        if (uShowTileGrid == 1) {
            float tileSize = 533.333;
            vec2 tileFrac = fract(vWorldPos.xy / tileSize);
            float tileLine = step(tileFrac.x, 0.001) + step(1.0 - tileFrac.x, 0.001)
                           + step(tileFrac.y, 0.001) + step(1.0 - tileFrac.y, 0.001);
            tileLine = clamp(tileLine, 0.0, 1.0);
            finalColor = mix(finalColor, vec3(1.0, 0.3, 0.0), tileLine * 0.8);
        }
    }

    FragColor = vec4(finalColor, alpha);
}
";

        return ShaderProgram.Create(_gl, vertSrc, fragSrc);
    }

    public void Dispose()
    {
        foreach (var chunk in _chunks)
            chunk.Dispose();
        _chunks.Clear();

        foreach (var tex in _textureCache.Values)
        {
            if (tex != 0) _gl.DeleteTexture(tex);
        }
        _textureCache.Clear();

        _shader.Dispose();
    }
}
