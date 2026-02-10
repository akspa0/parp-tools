using Silk.NET.OpenGL;
using SereniaBLPLib;
using MdxViewer.DataSources;

namespace MdxViewer.Rendering;

/// <summary>
/// Renders the Alpha 0.5.3 loading screen: a fullscreen background image with a progress bar.
/// Based on RE analysis of the original client (specifications/outputs/053/LoadingScreens/).
///
/// The Alpha client uses 3 layers:
///   1. Background: Interface\Glues\loading.blp
///   2. Progress bar border: Interface\Glues\LoadingBar\Loading-BarBorder.blp
///   3. Progress bar fill: Interface\Glues\LoadingBar\Loading-BarFill.blp
///
/// Progress formula: ratio = (current/max) * 0.75, then +0.25 when world finishes = 1.0.
/// The fill quad's width is scaled by the progress ratio.
///
/// The original client blocks the main thread during CWorld::LoadMap but force-presents
/// frames via UpdateProgressBar → GxScenePresent. We replicate this by calling
/// Present() from the onStatus callback during WorldScene construction.
/// </summary>
public class LoadingScreen : IDisposable
{
    private readonly GL _gl;
    private uint _bgTexture;
    private uint _borderTexture;
    private uint _fillTexture;
    private uint _blackTexture;
    private uint _vao;
    private uint _vbo;
    private uint _shader;
    private float _progress;
    private bool _active;
    private bool _disposed;

    // Progress bar position/size in normalized coords relative to the 4:3 content area.
    // Based on the Alpha client's LoadingBar placement: centered horizontally, near bottom.
    // The original 800x600 client places the bar at roughly y=72px from bottom, 480px wide, 20px tall.
    private const float BarRelX = 0.20f;   // 160/800
    private const float BarRelY = 0.10f;   // ~60/600 from bottom
    private const float BarRelW = 0.60f;   // 480/800
    private const float BarRelH = 0.033f;  // 20/600

    // Border is slightly larger than fill
    private const float BorderPad = 0.006f;

    public bool IsActive => _active;
    public float Progress => _progress;

    public LoadingScreen(GL gl)
    {
        _gl = gl;
        CreateShader();
        CreateQuadVao();
    }

    /// <summary>
    /// Load the 3 BLP textures from the data source and activate the loading screen.
    /// </summary>
    public void Enable(IDataSource? dataSource)
    {
        _progress = 0f;
        _active = true;

        // Try to load textures from MPQ
        _bgTexture = TryLoadBlp(dataSource, "Interface\\Glues\\loading.blp");
        _borderTexture = TryLoadBlp(dataSource, "Interface\\Glues\\LoadingBar\\Loading-BarBorder.blp");
        _fillTexture = TryLoadBlp(dataSource, "Interface\\Glues\\LoadingBar\\Loading-BarFill.blp");

        if (_bgTexture == 0)
            Console.WriteLine("[LoadingScreen] Warning: background texture not found");
    }

    /// <summary>
    /// Update progress ratio. Call from the loading callback.
    /// currentStep/totalSteps drives 0-75%, then call SetWorldLoaded() for final 25%.
    /// </summary>
    public void UpdateProgress(int currentStep, int totalSteps)
    {
        if (totalSteps > 0)
            _progress = ((float)currentStep / totalSteps) * 0.75f;
        _progress = Math.Clamp(_progress, 0f, 1f);
    }

    /// <summary>
    /// Signal that the world has finished loading — pushes progress to 100%.
    /// </summary>
    public void SetWorldLoaded()
    {
        _progress = 1.0f;
    }

    /// <summary>
    /// Disable the loading screen and free textures.
    /// </summary>
    public void Disable()
    {
        _active = false;
        if (_bgTexture != 0) { _gl.DeleteTexture(_bgTexture); _bgTexture = 0; }
        if (_borderTexture != 0) { _gl.DeleteTexture(_borderTexture); _borderTexture = 0; }
        if (_fillTexture != 0) { _gl.DeleteTexture(_fillTexture); _fillTexture = 0; }
        if (_blackTexture != 0) { _gl.DeleteTexture(_blackTexture); _blackTexture = 0; }
    }

    /// <summary>
    /// Render the loading screen with 4:3 letterboxing.
    /// The background is displayed at 4:3 aspect ratio, centered in the viewport
    /// with black bars on the sides for widescreen displays.
    /// </summary>
    public unsafe void Render(int viewportW = 0, int viewportH = 0)
    {
        if (!_active) return;

        // Save GL state
        _gl.Disable(EnableCap.DepthTest);
        _gl.Disable(EnableCap.CullFace);
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

        _gl.UseProgram(_shader);
        _gl.BindVertexArray(_vao);

        // Compute 4:3 letterbox region in normalized [0,1] coords.
        // If the viewport is wider than 4:3, we get pillarboxing (black bars on sides).
        // If narrower, we get letterboxing (black bars top/bottom).
        const float targetAspect = 4f / 3f;
        float vpAspect = viewportW > 0 && viewportH > 0
            ? (float)viewportW / viewportH
            : targetAspect; // default to 4:3 if unknown

        float contentX, contentY, contentW, contentH;
        if (vpAspect > targetAspect)
        {
            // Wider than 4:3 — pillarbox (black bars on sides)
            contentH = 1f;
            contentW = targetAspect / vpAspect;
            contentX = (1f - contentW) * 0.5f;
            contentY = 0f;
        }
        else
        {
            // Taller than 4:3 — letterbox (black bars top/bottom)
            contentW = 1f;
            contentH = vpAspect / targetAspect;
            contentX = 0f;
            contentY = (1f - contentH) * 0.5f;
        }

        // 0. Black background (fullscreen) — clears any area outside the 4:3 region
        DrawColorQuad(0f, 0f, 1f, 1f, 0f, 0f, 0f, 1f);

        // 1. Background image (within 4:3 region)
        if (_bgTexture != 0)
            DrawQuad(_bgTexture, contentX, contentY, contentW, contentH);

        // 2. Progress bar border (positioned relative to 4:3 content area)
        float barX = contentX + BarRelX * contentW;
        float barY = contentY + BarRelY * contentH;
        float barW = BarRelW * contentW;
        float barH = BarRelH * contentH;
        float borderPadX = BorderPad * contentW;
        float borderPadY = BorderPad * contentH;

        if (_borderTexture != 0)
            DrawQuad(_borderTexture,
                barX - borderPadX, barY - borderPadY,
                barW + borderPadX * 2, barH + borderPadY * 2);

        // 3. Progress bar fill (width scaled by progress)
        if (_fillTexture != 0 && _progress > 0.001f)
            DrawQuad(_fillTexture, barX, barY, barW * _progress, barH);

        _gl.BindVertexArray(0);
        _gl.UseProgram(0);

        // Restore GL state
        _gl.Enable(EnableCap.DepthTest);
        _gl.Disable(EnableCap.Blend);
    }

    private unsafe void DrawQuad(uint texture, float x, float y, float w, float h)
    {
        // Convert from (0,0)=bottom-left to NDC (-1,-1) to (1,1)
        float x0 = x * 2f - 1f;
        float y0 = y * 2f - 1f;
        float x1 = (x + w) * 2f - 1f;
        float y1 = (y + h) * 2f - 1f;

        // Triangle strip: 4 vertices (pos.xy + uv)
        float[] verts = {
            x0, y0, 0f, 1f,  // bottom-left
            x1, y0, 1f, 1f,  // bottom-right
            x0, y1, 0f, 0f,  // top-left
            x1, y1, 1f, 0f,  // top-right
        };

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* ptr = verts)
            _gl.BufferSubData(BufferTargetARB.ArrayBuffer, 0, (nuint)(verts.Length * sizeof(float)), ptr);

        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindTexture(TextureTarget.Texture2D, texture);

        _gl.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);
    }

    /// <summary>
    /// Draw a solid-color quad (no texture). Used for black letterbox bars.
    /// </summary>
    private unsafe void DrawColorQuad(float x, float y, float w, float h, float r, float g, float b, float a)
    {
        float x0 = x * 2f - 1f;
        float y0 = y * 2f - 1f;
        float x1 = (x + w) * 2f - 1f;
        float y1 = (y + h) * 2f - 1f;

        float[] verts = {
            x0, y0, 0f, 0f,
            x1, y0, 0f, 0f,
            x0, y1, 0f, 0f,
            x1, y1, 0f, 0f,
        };

        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        fixed (float* ptr = verts)
            _gl.BufferSubData(BufferTargetARB.ArrayBuffer, 0, (nuint)(verts.Length * sizeof(float)), ptr);

        // Bind a 1x1 black pixel texture (or just use the shader with a black texture)
        // Simpler: create a tiny 1x1 texture on first use
        if (_blackTexture == 0)
        {
            _blackTexture = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, _blackTexture);
            byte[] black = { 0, 0, 0, 255 };
            fixed (byte* ptr = black)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, 1, 1, 0,
                    PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
        }

        _gl.ActiveTexture(TextureUnit.Texture0);
        _gl.BindTexture(TextureTarget.Texture2D, _blackTexture);
        _gl.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);
    }

    private uint TryLoadBlp(IDataSource? dataSource, string path)
    {
        if (dataSource == null) return 0;

        byte[]? data = dataSource.ReadFile(path);
        if (data == null)
        {
            // Try lowercase
            data = dataSource.ReadFile(path.ToLowerInvariant());
        }
        if (data == null)
        {
            Console.WriteLine($"[LoadingScreen] BLP not found: {path}");
            return 0;
        }

        return LoadBlpToTexture(data, path);
    }

    private unsafe uint LoadBlpToTexture(byte[] blpData, string name)
    {
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

                // BGRA -> RGBA
                for (int y = 0; y < h; y++)
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
            finally { bmp.UnlockBits(data); }
            bmp.Dispose();

            uint tex = _gl.GenTexture();
            _gl.BindTexture(TextureTarget.Texture2D, tex);
            fixed (byte* ptr = pixels)
                _gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba,
                    (uint)w, (uint)h, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            _gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
            _gl.BindTexture(TextureTarget.Texture2D, 0);

            Console.WriteLine($"[LoadingScreen] Loaded {name} ({w}x{h})");
            return tex;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[LoadingScreen] Failed to load {name}: {ex.Message}");
            return 0;
        }
    }

    private unsafe void CreateQuadVao()
    {
        _vao = _gl.GenVertexArray();
        _gl.BindVertexArray(_vao);

        _vbo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);

        // Allocate space for 4 vertices * 4 floats (pos.xy + uv)
        _gl.BufferData(BufferTargetARB.ArrayBuffer, 4 * 4 * sizeof(float), null, BufferUsageARB.DynamicDraw);

        // Position (location 0): vec2
        _gl.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);

        // UV (location 1): vec2
        _gl.VertexAttribPointer(1, 2, VertexAttribPointerType.Float, false, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);

        _gl.BindVertexArray(0);
    }

    private void CreateShader()
    {
        const string vertSrc = @"#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}";
        const string fragSrc = @"#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTexture;
void main() {
    FragColor = texture(uTexture, vUV);
}";

        uint vs = _gl.CreateShader(ShaderType.VertexShader);
        _gl.ShaderSource(vs, vertSrc);
        _gl.CompileShader(vs);
        CheckShader(vs, "vertex");

        uint fs = _gl.CreateShader(ShaderType.FragmentShader);
        _gl.ShaderSource(fs, fragSrc);
        _gl.CompileShader(fs);
        CheckShader(fs, "fragment");

        _shader = _gl.CreateProgram();
        _gl.AttachShader(_shader, vs);
        _gl.AttachShader(_shader, fs);
        _gl.LinkProgram(_shader);

        _gl.GetProgram(_shader, ProgramPropertyARB.LinkStatus, out int linked);
        if (linked == 0)
            Console.WriteLine($"[LoadingScreen] Shader link error: {_gl.GetProgramInfoLog(_shader)}");

        _gl.DeleteShader(vs);
        _gl.DeleteShader(fs);

        // Set texture uniform
        _gl.UseProgram(_shader);
        int loc = _gl.GetUniformLocation(_shader, "uTexture");
        if (loc >= 0) _gl.Uniform1(loc, 0);
        _gl.UseProgram(0);
    }

    private void CheckShader(uint shader, string type)
    {
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int ok);
        if (ok == 0)
            Console.WriteLine($"[LoadingScreen] {type} shader error: {_gl.GetShaderInfoLog(shader)}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Disable();
        if (_vao != 0) _gl.DeleteVertexArray(_vao);
        if (_vbo != 0) _gl.DeleteBuffer(_vbo);
        if (_shader != 0) _gl.DeleteProgram(_shader);
    }
}
