using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.Wmo;
using WoWViewer.Logging;

namespace WoWViewer.Rendering;

/// <summary>
/// Handles OpenGL shader compilation, mesh generation (from MLIQ chunks),
/// orientation selection, and rendering of WMO liquid surfaces.
/// </summary>
internal sealed class WmoLiquidRenderer : IDisposable
{
    private readonly GL _gl;
    private readonly WmoV14ToV17Converter.WmoV14Data _wmo;
    private readonly string? _buildVersion;

    private readonly List<LiquidMeshData> _liquidMeshes = new();
    private static uint _liquidShader;
    private static int _uLiqModel, _uLiqView, _uLiqProj, _uLiqColor;
    private static int _liquidShaderRefCount;
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

    public int LiquidMeshCount => _liquidMeshes.Count;

    public WmoLiquidRenderer(GL gl, WmoV14ToV17Converter.WmoV14Data wmo, string? buildVersion)
    {
        _gl = gl;
        _wmo = wmo;
        _buildVersion = buildVersion;

        InitLiquidShader();
        BuildLiquidMeshes();
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

    public void EnsureLiquidMeshesUpToDate()
    {
        if (_builtMliqRotationRevision == _mliqRotationRevision)
            return;

        DisposeLiquidMeshes();
        BuildLiquidMeshes();
    }

    public unsafe void Render(
        Matrix4x4 model,
        Matrix4x4 view,
        Matrix4x4 proj,
        bool[] runtimeVisibleGroups,
        ref int currentDrawCalls,
        ref int currentLiquidDrawCalls,
        ref int currentVisibleLiquidMeshes)
    {
        if (_liquidMeshes.Count == 0)
            return;

        _gl.UseProgram(_liquidShader);
        _gl.UniformMatrix4(_uLiqModel, 1, false, (float*)&model);
        _gl.UniformMatrix4(_uLiqView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uLiqProj, 1, false, (float*)&proj);

        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
        _gl.DepthMask(false);

        foreach (var liq in _liquidMeshes)
        {
            if (liq.GroupIndex >= 0 && liq.GroupIndex < runtimeVisibleGroups.Length && !runtimeVisibleGroups[liq.GroupIndex])
                continue;

            currentDrawCalls++;
            currentLiquidDrawCalls++;
            currentVisibleLiquidMeshes++;
            _gl.Uniform4(_uLiqColor, liq.ColorR, liq.ColorG, liq.ColorB, liq.ColorA);
            _gl.BindVertexArray(liq.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, liq.IndexCount, DrawElementsType.UnsignedShort, null);
        }

        _gl.BindVertexArray(0);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
    }

    private int GetBaselineMliqRotationQuarterTurns()
    {
        return WmoLiquidLayoutResolver.GetBaselineRotationQuarterTurns(_wmo.Version, _buildVersion);
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
        DisposeLiquidMeshes();

        _liquidShaderRefCount--;
        if (_liquidShaderRefCount <= 0 && _liquidShader != 0)
        {
            _gl.DeleteProgram(_liquidShader);
            _liquidShader = 0;
            _liquidShaderRefCount = 0;
        }
    }

    private class LiquidMeshData
    {
        public int GroupIndex;
        public uint Vao, Vbo, Ebo;
        public uint IndexCount;
        public float ColorR, ColorG, ColorB, ColorA;
    }
}
