using System.Numerics;
using MdxViewer.Logging;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Renders wireframe bounding boxes, lines, and pin markers for debug visualization.
/// Uses batched rendering: collect geometry with AddLine/AddPin, then flush with a single draw call.
/// Legacy per-call methods (DrawBox, DrawBoxMinMax) remain for low-count debug boxes.
/// </summary>
public class BoundingBoxRenderer : IDisposable
{
    private readonly GL _gl;

    // Per-object cube rendering (legacy, for bounding boxes — low count)
    private uint _cubeVao, _cubeVbo, _cubeEbo, _shader;
    private int _uMvp, _uColor;
    private bool _initialized;

    // Batched line rendering (for taxi paths, pins — high count)
    private uint _batchVao, _batchVbo;
    private uint _batchShader;
    private int _batchUVP, _batchInitialized;
    private readonly List<float> _batchVertices = new(4096); // pos(3) + color(3) per vertex
    private int _batchLineVertexCount;

    // Batched solid-triangle rendering (PM4 filled overlay)
    private uint _solidBatchVao, _solidBatchVbo;
    private uint _solidBatchShader;
    private int _solidBatchUVP, _solidBatchInitialized;
    private readonly List<float> _solidBatchVertices = new(4096); // pos(3) + color(4) per vertex
    private int _solidBatchVertexCount;

    // 8 vertices of a unit cube (0..1), scaled/translated per box
    private static readonly float[] CubeVertices =
    {
        0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0, // bottom face
        0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1  // top face
    };

    // 24 indices forming 12 line segments
    private static readonly ushort[] CubeIndices =
    {
        0,1, 1,2, 2,3, 3,0, // bottom
        4,5, 5,6, 6,7, 7,4, // top
        0,4, 1,5, 2,6, 3,7  // verticals
    };

    public BoundingBoxRenderer(GL gl)
    {
        _gl = gl;
        Initialize();
    }

    private unsafe void Initialize()
    {
        // ── Shader for per-object cube draws (uniform color) ──
        string vertSrc = @"#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }";

        string fragSrc = @"#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() { FragColor = vec4(uColor, 1.0); }";

        uint vs = CompileShader(ShaderType.VertexShader, vertSrc);
        uint fs = CompileShader(ShaderType.FragmentShader, fragSrc);
        _shader = _gl.CreateProgram();
        _gl.AttachShader(_shader, vs);
        _gl.AttachShader(_shader, fs);
        _gl.LinkProgram(_shader);
        _gl.GetProgram(_shader, ProgramPropertyARB.LinkStatus, out int linkStatus);
        if (linkStatus == 0)
            ViewerLog.Trace($"[BoundingBoxRenderer] Program link error: {_gl.GetProgramInfoLog(_shader)}");
        _gl.DeleteShader(vs);
        _gl.DeleteShader(fs);

        _uMvp = _gl.GetUniformLocation(_shader, "uMVP");
        _uColor = _gl.GetUniformLocation(_shader, "uColor");

        // Cube VAO/VBO/EBO
        _cubeVao = _gl.GenVertexArray();
        _cubeVbo = _gl.GenBuffer();
        _cubeEbo = _gl.GenBuffer();

        _gl.BindVertexArray(_cubeVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _cubeVbo);
        fixed (float* p = CubeVertices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(CubeVertices.Length * sizeof(float)), p, BufferUsageARB.StaticDraw);
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, _cubeEbo);
        fixed (ushort* p = CubeIndices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(CubeIndices.Length * sizeof(ushort)), p, BufferUsageARB.StaticDraw);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.BindVertexArray(0);

        // ── Shader for batched lines (per-vertex color) ──
        string batchVertSrc = @"#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 uVP;
out vec3 vColor;
void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
    vColor = aColor;
}";
        string batchFragSrc = @"#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }";

        uint bvs = CompileShader(ShaderType.VertexShader, batchVertSrc);
        uint bfs = CompileShader(ShaderType.FragmentShader, batchFragSrc);
        _batchShader = _gl.CreateProgram();
        _gl.AttachShader(_batchShader, bvs);
        _gl.AttachShader(_batchShader, bfs);
        _gl.LinkProgram(_batchShader);
        _gl.GetProgram(_batchShader, ProgramPropertyARB.LinkStatus, out int batchLink);
        if (batchLink == 0)
            ViewerLog.Trace($"[BoundingBoxRenderer] Batch program link error: {_gl.GetProgramInfoLog(_batchShader)}");
        _gl.DeleteShader(bvs);
        _gl.DeleteShader(bfs);

        _batchUVP = _gl.GetUniformLocation(_batchShader, "uVP");

        // Batch VAO/VBO (dynamic)
        _batchVao = _gl.GenVertexArray();
        _batchVbo = _gl.GenBuffer();
        _gl.BindVertexArray(_batchVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _batchVbo);
        // pos(3) + color(3) = 6 floats per vertex, stride = 24 bytes
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.BindVertexArray(0);

        // ── Shader for batched solid triangles (per-vertex color + alpha) ──
        string solidVertSrc = @"#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 uVP;
out vec4 vColor;
void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
    vColor = aColor;
}";
        string solidFragSrc = @"#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main() { FragColor = vColor; }";

        uint svs = CompileShader(ShaderType.VertexShader, solidVertSrc);
        uint sfs = CompileShader(ShaderType.FragmentShader, solidFragSrc);
        _solidBatchShader = _gl.CreateProgram();
        _gl.AttachShader(_solidBatchShader, svs);
        _gl.AttachShader(_solidBatchShader, sfs);
        _gl.LinkProgram(_solidBatchShader);
        _gl.GetProgram(_solidBatchShader, ProgramPropertyARB.LinkStatus, out int solidLink);
        if (solidLink == 0)
            ViewerLog.Trace($"[BoundingBoxRenderer] Solid batch program link error: {_gl.GetProgramInfoLog(_solidBatchShader)}");
        _gl.DeleteShader(svs);
        _gl.DeleteShader(sfs);

        _solidBatchUVP = _gl.GetUniformLocation(_solidBatchShader, "uVP");

        _solidBatchVao = _gl.GenVertexArray();
        _solidBatchVbo = _gl.GenBuffer();
        _gl.BindVertexArray(_solidBatchVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _solidBatchVbo);
        // pos(3) + color(4) = 7 floats per vertex, stride = 28 bytes
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 7 * sizeof(float), (void*)0);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(1, 4, VertexAttribPointerType.Float, false, 7 * sizeof(float), (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        _gl.BindVertexArray(0);

        _batchInitialized = 1;
        _solidBatchInitialized = 1;
        _initialized = true;
        ViewerLog.Trace($"[BoundingBoxRenderer] Init: cube={_shader} batch={_batchShader}");
    }

    // ═══════════════════════════════════════════════════════════════════
    // BATCHED API — collect geometry, then flush with 1 draw call
    // ═══════════════════════════════════════════════════════════════════

    /// <summary>
    /// Clear the batch buffer. Call at the start of each frame before adding geometry.
    /// </summary>
    public void BeginBatch()
    {
        _batchVertices.Clear();
        _batchLineVertexCount = 0;
    }

    /// <summary>
    /// Clear the solid-triangle batch buffer.
    /// </summary>
    public void BeginSolidBatch()
    {
        _solidBatchVertices.Clear();
        _solidBatchVertexCount = 0;
    }

    /// <summary>
    /// Add a line segment to the batch (2 vertices).
    /// </summary>
    public void BatchLine(Vector3 from, Vector3 to, Vector3 color)
    {
        _batchVertices.Add(from.X); _batchVertices.Add(from.Y); _batchVertices.Add(from.Z);
        _batchVertices.Add(color.X); _batchVertices.Add(color.Y); _batchVertices.Add(color.Z);
        _batchVertices.Add(to.X); _batchVertices.Add(to.Y); _batchVertices.Add(to.Z);
        _batchVertices.Add(color.X); _batchVertices.Add(color.Y); _batchVertices.Add(color.Z);
        _batchLineVertexCount += 2;
    }

    public void BatchBoxMinMax(Vector3 min, Vector3 max, Vector3 color)
    {
        GetSanitizedBoxCorners(min, max,
            out Vector3 v0, out Vector3 v1, out Vector3 v2, out Vector3 v3,
            out Vector3 v4, out Vector3 v5, out Vector3 v6, out Vector3 v7);

        BatchLine(v0, v1, color); BatchLine(v1, v2, color);
        BatchLine(v2, v3, color); BatchLine(v3, v0, color);
        BatchLine(v4, v5, color); BatchLine(v5, v6, color);
        BatchLine(v6, v7, color); BatchLine(v7, v4, color);
        BatchLine(v0, v4, color); BatchLine(v1, v5, color);
        BatchLine(v2, v6, color); BatchLine(v3, v7, color);
    }

    public void BatchHighlightedBoxMinMax(Vector3 min, Vector3 max, float timeSeconds, Vector3 innerColor, Vector3 accentColorA, Vector3 accentColorB)
    {
        BatchBoxMinMax(min, max, innerColor);

        Vector3 actualMin = Vector3.Min(min, max);
        Vector3 actualMax = Vector3.Max(min, max);
        Vector3 size = actualMax - actualMin;
        float maxDimension = MathF.Max(size.X, MathF.Max(size.Y, size.Z));
        float pulse = 0.85f + 0.35f * (0.5f + 0.5f * MathF.Sin(timeSeconds * 4.0f));
        float inflate = Math.Clamp(maxDimension * 0.015f, 0.75f, 6.0f) * pulse;
        Vector3 accentMin = actualMin - new Vector3(inflate);
        Vector3 accentMax = actualMax + new Vector3(inflate);
        float segmentLength = Math.Clamp(maxDimension * 0.08f, 6.0f, 22.0f);

        BatchAlternatingBoxMinMax(accentMin, accentMax, accentColorA, accentColorB, timeSeconds, segmentLength);
    }

    /// <summary>
    /// Add a pin marker to the batch: vertical line + diamond head wireframe.
    /// Adds 28 vertices (14 line segments) per pin.
    /// </summary>
    public void BatchPin(Vector3 position, float height, float headSize, Vector3 color)
    {
        // Vertical line (4 edges of a thin column)
        var bot = position;
        var top = position + new Vector3(0, 0, height);
        float t = 0.5f;
        BatchLine(bot + new Vector3(-t, 0, 0), top + new Vector3(-t, 0, 0), color);
        BatchLine(bot + new Vector3(t, 0, 0), top + new Vector3(t, 0, 0), color);
        BatchLine(bot + new Vector3(0, -t, 0), top + new Vector3(0, -t, 0), color);
        BatchLine(bot + new Vector3(0, t, 0), top + new Vector3(0, t, 0), color);

        // Diamond head at top: 12 edges of a wireframe cube
        var hc = position + new Vector3(0, 0, height);
        var hs = headSize;
        var v0 = hc + new Vector3(-hs, -hs, -hs);
        var v1 = hc + new Vector3( hs, -hs, -hs);
        var v2 = hc + new Vector3( hs,  hs, -hs);
        var v3 = hc + new Vector3(-hs,  hs, -hs);
        var v4 = hc + new Vector3(-hs, -hs,  hs);
        var v5 = hc + new Vector3( hs, -hs,  hs);
        var v6 = hc + new Vector3( hs,  hs,  hs);
        var v7 = hc + new Vector3(-hs,  hs,  hs);
        // Bottom face
        BatchLine(v0, v1, color); BatchLine(v1, v2, color);
        BatchLine(v2, v3, color); BatchLine(v3, v0, color);
        // Top face
        BatchLine(v4, v5, color); BatchLine(v5, v6, color);
        BatchLine(v6, v7, color); BatchLine(v7, v4, color);
        // Verticals
        BatchLine(v0, v4, color); BatchLine(v1, v5, color);
        BatchLine(v2, v6, color); BatchLine(v3, v7, color);
    }

    private void BatchAlternatingBoxMinMax(Vector3 min, Vector3 max, Vector3 colorA, Vector3 colorB, float timeSeconds, float segmentLength)
    {
        GetSanitizedBoxCorners(min, max,
            out Vector3 v0, out Vector3 v1, out Vector3 v2, out Vector3 v3,
            out Vector3 v4, out Vector3 v5, out Vector3 v6, out Vector3 v7);

        BatchAlternatingLine(v0, v1, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v1, v2, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v2, v3, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v3, v0, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v4, v5, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v5, v6, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v6, v7, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v7, v4, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v0, v4, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v1, v5, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v2, v6, colorA, colorB, timeSeconds, segmentLength);
        BatchAlternatingLine(v3, v7, colorA, colorB, timeSeconds, segmentLength);
    }

    private void BatchAlternatingLine(Vector3 from, Vector3 to, Vector3 colorA, Vector3 colorB, float timeSeconds, float segmentLength)
    {
        Vector3 delta = to - from;
        float lineLength = delta.Length();
        if (lineLength <= 0.001f)
        {
            BatchLine(from, to, colorA);
            return;
        }

        int segmentCount = Math.Clamp((int)MathF.Ceiling(lineLength / MathF.Max(segmentLength, 1.0f)), 2, 32);
        float phase = timeSeconds * 5.0f;
        for (int i = 0; i < segmentCount; i++)
        {
            float t0 = i / (float)segmentCount;
            float t1 = (i + 1) / (float)segmentCount;
            Vector3 segmentFrom = Vector3.Lerp(from, to, t0);
            Vector3 segmentTo = Vector3.Lerp(from, to, t1);
            float mix = 0.5f + 0.5f * MathF.Sin(i * MathF.PI + phase);
            Vector3 segmentColor = Vector3.Lerp(colorA, colorB, mix);
            BatchLine(segmentFrom, segmentTo, segmentColor);
        }
    }

    private static void GetSanitizedBoxCorners(
        Vector3 min,
        Vector3 max,
        out Vector3 v0,
        out Vector3 v1,
        out Vector3 v2,
        out Vector3 v3,
        out Vector3 v4,
        out Vector3 v5,
        out Vector3 v6,
        out Vector3 v7)
    {
        SanitizeBox(ref min, ref max);

        v0 = new Vector3(min.X, min.Y, min.Z);
        v1 = new Vector3(max.X, min.Y, min.Z);
        v2 = new Vector3(max.X, max.Y, min.Z);
        v3 = new Vector3(min.X, max.Y, min.Z);
        v4 = new Vector3(min.X, min.Y, max.Z);
        v5 = new Vector3(max.X, min.Y, max.Z);
        v6 = new Vector3(max.X, max.Y, max.Z);
        v7 = new Vector3(min.X, max.Y, max.Z);
    }

    private static void SanitizeBox(ref Vector3 min, ref Vector3 max)
    {
        Vector3 actualMin = Vector3.Min(min, max);
        Vector3 actualMax = Vector3.Max(min, max);
        Vector3 size = actualMax - actualMin;
        const float minimumAxis = 0.1f;

        if (size.LengthSquared() < 0.0001f)
        {
            Vector3 center = (actualMin + actualMax) * 0.5f;
            actualMin = center - new Vector3(5f, 5f, 5f);
            actualMax = center + new Vector3(5f, 5f, 5f);
        }
        else
        {
            if (size.X < minimumAxis)
            {
                float pad = (minimumAxis - size.X) * 0.5f;
                actualMin.X -= pad;
                actualMax.X += pad;
            }

            if (size.Y < minimumAxis)
            {
                float pad = (minimumAxis - size.Y) * 0.5f;
                actualMin.Y -= pad;
                actualMax.Y += pad;
            }

            if (size.Z < minimumAxis)
            {
                float pad = (minimumAxis - size.Z) * 0.5f;
                actualMin.Z -= pad;
                actualMax.Z += pad;
            }
        }

        min = actualMin;
        max = actualMax;
    }

    /// <summary>
    /// Add a solid triangle to the batch (3 vertices).
    /// </summary>
    public void BatchTriangle(Vector3 a, Vector3 b, Vector3 c, Vector3 color, float alpha)
    {
        AddSolidVertex(a, color, alpha);
        AddSolidVertex(b, color, alpha);
        AddSolidVertex(c, color, alpha);
        _solidBatchVertexCount += 3;
    }

    private void AddSolidVertex(Vector3 pos, Vector3 color, float alpha)
    {
        _solidBatchVertices.Add(pos.X); _solidBatchVertices.Add(pos.Y); _solidBatchVertices.Add(pos.Z);
        _solidBatchVertices.Add(color.X); _solidBatchVertices.Add(color.Y); _solidBatchVertices.Add(color.Z); _solidBatchVertices.Add(alpha);
    }

    /// <summary>
    /// Upload and draw all batched lines in a single draw call.
    /// </summary>
    public unsafe void FlushBatch(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_batchInitialized == 0 || _batchLineVertexCount == 0) return;

        var vp = view * proj;

        _gl.UseProgram(_batchShader);
        _gl.UniformMatrix4(_batchUVP, 1, false, (float*)&vp);

        _gl.BindVertexArray(_batchVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _batchVbo);

        // Upload vertex data
        var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_batchVertices);
        fixed (float* p = span)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(span.Length * sizeof(float)), p, BufferUsageARB.StreamDraw);

        _gl.DrawArrays(PrimitiveType.Lines, 0, (uint)_batchLineVertexCount);
        _gl.BindVertexArray(0);
    }

    /// <summary>
    /// Upload and draw all batched solid triangles in a single draw call.
    /// Caller is responsible for blend/depth state configuration.
    /// </summary>
    public unsafe void FlushSolidBatch(Matrix4x4 view, Matrix4x4 proj)
    {
        if (_solidBatchInitialized == 0 || _solidBatchVertexCount == 0) return;

        var vp = view * proj;

        _gl.UseProgram(_solidBatchShader);
        _gl.UniformMatrix4(_solidBatchUVP, 1, false, (float*)&vp);

        _gl.BindVertexArray(_solidBatchVao);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _solidBatchVbo);

        var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_solidBatchVertices);
        fixed (float* p = span)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(span.Length * sizeof(float)), p, BufferUsageARB.StreamDraw);

        _gl.DrawArrays(PrimitiveType.Triangles, 0, (uint)_solidBatchVertexCount);
        _gl.BindVertexArray(0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // LEGACY PER-OBJECT API — for bounding boxes (low count)
    // ═══════════════════════════════════════════════════════════════════

    /// <summary>
    /// Draw a wireframe box at the given position with the given size.
    /// </summary>
    public unsafe void DrawBox(Vector3 center, Vector3 halfExtents, Matrix4x4 view, Matrix4x4 proj, Vector3 color)
    {
        if (!_initialized) return;

        var model = Matrix4x4.CreateScale(halfExtents.X * 2, halfExtents.Y * 2, halfExtents.Z * 2)
            * Matrix4x4.CreateTranslation(center - halfExtents);
        var mvp = model * view * proj;

        _gl.UseProgram(_shader);
        _gl.UniformMatrix4(_uMvp, 1, false, (float*)&mvp);
        _gl.Uniform3(_uColor, color.X, color.Y, color.Z);

        _gl.BindVertexArray(_cubeVao);
        _gl.DrawElements(PrimitiveType.Lines, (uint)CubeIndices.Length, DrawElementsType.UnsignedShort, (void*)0);
        _gl.BindVertexArray(0);
    }

    /// <summary>
    /// Draw a wireframe box from min/max corners.
    /// </summary>
    public void DrawBoxMinMax(Vector3 min, Vector3 max, Matrix4x4 view, Matrix4x4 proj, Vector3 color)
    {
        SanitizeBox(ref min, ref max);
        var center = (min + max) * 0.5f;
        var halfExtents = (max - min) * 0.5f;
        halfExtents = new Vector3(MathF.Abs(halfExtents.X), MathF.Abs(halfExtents.Y), MathF.Abs(halfExtents.Z));
        DrawBox(center, halfExtents, view, proj, color);
    }

    /// <summary>
    /// Draw a small marker box at a point position.
    /// </summary>
    public void DrawMarker(Vector3 position, float size, Matrix4x4 view, Matrix4x4 proj, Vector3 color)
    {
        var half = new Vector3(size, size, size);
        DrawBox(position, half, view, proj, color);
    }

    private uint CompileShader(ShaderType type, string source)
    {
        uint shader = _gl.CreateShader(type);
        _gl.ShaderSource(shader, source);
        _gl.CompileShader(shader);
        _gl.GetShader(shader, ShaderParameterName.CompileStatus, out int status);
        if (status == 0)
            ViewerLog.Trace($"[BoundingBoxRenderer] Shader compile error: {_gl.GetShaderInfoLog(shader)}");
        return shader;
    }

    public void Dispose()
    {
        if (_initialized)
        {
            _gl.DeleteVertexArray(_cubeVao);
            _gl.DeleteBuffer(_cubeVbo);
            _gl.DeleteBuffer(_cubeEbo);
            _gl.DeleteProgram(_shader);
            _gl.DeleteVertexArray(_batchVao);
            _gl.DeleteBuffer(_batchVbo);
            _gl.DeleteProgram(_batchShader);
            _gl.DeleteVertexArray(_solidBatchVao);
            _gl.DeleteBuffer(_solidBatchVbo);
            _gl.DeleteProgram(_solidBatchShader);
            _initialized = false;
        }
    }
}
