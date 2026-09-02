using System;
using System.Collections.Generic;
using System.Numerics;
using ImGuiNET;
using Silk.NET.OpenGL;
using WoWViewer.Terrain;

namespace WoWViewer.Rendering;

public sealed record ClusterItem(
    string DedupKey,
    string Title,
    string Detail,
    string SecondaryDetail,
    Vector3 WorldPosition,
    float? Distance,
    Action Apply);

/// <summary>
/// In-scene 3D radial cluster disambiguator for dense object clusters.
/// Replaces intrusive 2D modal tooltips with an in-world 3D radial ring and compact world-anchored target pins.
/// </summary>
public sealed class SceneClusterSelector3D : IDisposable
{
    private readonly GL _gl;
    private ShaderProgram? _shader;
    private ProceduralMesh? _ringMesh;
    private ProceduralMesh? _pinMesh;

    private readonly List<ClusterItem> _candidates = new();
    private Vector3 _centerWorldPos;
    private bool _active;
    private int _hoveredIndex = -1;
    private bool _disposed;

    public bool IsActive => _active;
    public int CandidateCount => _candidates.Count;

    public SceneClusterSelector3D(GL gl)
    {
        _gl = gl;
        InitializeGpuResources();
    }

    private void InitializeGpuResources()
    {
        try
        {
            const string vertexShader = @"#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;

void main()
{
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    gl_Position = uProj * uView * uModel * vec4(aPos, 1.0);
}";

            const string fragmentShader = @"#version 330 core
in vec3 vNormal;
uniform vec4 uColor;

out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal);
    float diff = max(abs(n.z), 0.3);
    FragColor = vec4(uColor.rgb * diff + uColor.rgb * 0.4, uColor.a);
}";

            _shader = ShaderProgram.Create(_gl, vertexShader, fragmentShader);
            _ringMesh = ProceduralMeshLoader.CreateOrbitalRing(_gl, 1.2f, 0.035f, 36, 8);
            _pinMesh = OpenScadAssetResolver.TryLoadMesh(_gl, "cluster_pin.off")
                ?? ProceduralMeshLoader.CreatePointerArrow(_gl);
        }
        catch
        {
            // Fallback handled gracefully
        }
    }

    public void Open(Vector3 centerWorldPos, IEnumerable<ClusterItem> items)
    {
        _candidates.Clear();
        _candidates.AddRange(items);
        _centerWorldPos = centerWorldPos;
        _active = _candidates.Count > 1;
        _hoveredIndex = -1;
    }

    public void Close()
    {
        _active = false;
        _candidates.Clear();
        _hoveredIndex = -1;
    }

    /// <summary>
    /// Checks keyboard shortcuts (1-9 and Escape) when cluster disambiguator is open.
    /// </summary>
    public bool HandleInput()
    {
        if (!_active || _candidates.Count == 0)
            return false;

        if (ImGui.IsKeyPressed(ImGuiKey.Escape))
        {
            Close();
            return true;
        }

        for (int i = 0; i < Math.Min(_candidates.Count, 9); i++)
        {
            ImGuiKey key = (ImGuiKey)((int)ImGuiKey._1 + i);
            if (ImGui.IsKeyPressed(key))
            {
                Action apply = _candidates[i].Apply;
                Close();
                apply();
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Renders the 3D in-world orbital ring and candidate markers on the contact surface.
    /// </summary>
    public void RenderWorld3D(Camera camera, Matrix4x4 proj)
    {
        if (!_active || _shader == null || _ringMesh == null || _disposed)
            return;

        Matrix4x4 view = camera.GetViewMatrix();

        _gl.DepthRange(0.0f, 0.05f); // Always visible over contact terrain/doodads
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.Enable(EnableCap.Blend);
        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

        _shader.Use();
        _shader.SetMat4("uView", view);
        _shader.SetMat4("uProj", proj);

        // 1. Draw central orbital ring at contact point
        Matrix4x4 ringModel = Matrix4x4.CreateTranslation(_centerWorldPos);
        _shader.SetMat4("uModel", ringModel);
        _shader.SetVec4("uColor", new Vector4(0.95f, 0.78f, 0.28f, 0.85f)); // WoW Gold
        _ringMesh.Draw();

        // 2. Draw radial candidate pins
        float radius = 1.2f;
        int count = _candidates.Count;
        for (int i = 0; i < count; i++)
        {
            float angle = (float)i / count * MathF.PI * 2f;
            Vector3 pinPos = _centerWorldPos + new Vector3(MathF.Cos(angle) * radius, MathF.Sin(angle) * radius, 0.2f);

            bool isHovered = i == _hoveredIndex;
            Vector4 pinColor = isHovered
                ? new Vector4(0.2f, 0.95f, 0.35f, 0.95f) // Glowing green when hovered
                : new Vector4(0.35f, 0.65f, 0.95f, 0.85f); // Tech blue

            Matrix4x4 pinModel = Matrix4x4.CreateScale(0.4f) * Matrix4x4.CreateTranslation(pinPos);
            _shader.SetMat4("uModel", pinModel);
            _shader.SetVec4("uColor", pinColor);
            _pinMesh?.Draw();
        }

        _gl.DepthRange(0.0f, 1.0f);
    }

    /// <summary>
    /// Renders a single consolidated in-scene disambiguator card anchored directly above the 3D orbital ring.
    /// </summary>
    public void RenderScreenOverlay(
        Matrix4x4 view,
        Matrix4x4 proj,
        float vpX,
        float vpY,
        float vpW,
        float vpH)
    {
        if (!_active || _candidates.Count == 0)
            return;

        // Project cluster datum point to viewport screen coordinates
        Vector4 centerClip = Vector4.Transform(new Vector4(_centerWorldPos + new Vector3(0, 0, 1.4f), 1.0f), view * proj);
        if (centerClip.W <= 0.01f) return; // Behind camera

        float ndcX = centerClip.X / centerClip.W;
        float ndcY = centerClip.Y / centerClip.W;
        float screenX = vpX + (ndcX + 1.0f) * 0.5f * vpW;
        float screenY = vpY + (1.0f - ndcY) * 0.5f * vpH;

        // Position single consolidated card above cluster datum
        Vector2 cardPos = new(
            Math.Clamp(screenX - 160f, vpX + 10f, vpX + vpW - 360f),
            Math.Clamp(screenY - 100f, vpY + 10f, vpY + vpH - 240f));

        ImGui.SetNextWindowPos(cardPos, ImGuiCond.Always);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(14f, 10f));
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 2f);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, 6f);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, new Vector4(0.04f, 0.06f, 0.11f, 0.96f));
        ImGui.PushStyleColor(ImGuiCol.Border, new Vector4(0.95f, 0.79f, 0.28f, 0.95f)); // Warcraft gold
        ImGui.PushStyleColor(ImGuiCol.Separator, new Vector4(0.88f, 0.73f, 0.22f, 0.80f));

        ImGuiWindowFlags flags = ImGuiWindowFlags.NoDecoration
            | ImGuiWindowFlags.AlwaysAutoResize
            | ImGuiWindowFlags.NoDocking
            | ImGuiWindowFlags.NoSavedSettings
            | ImGuiWindowFlags.NoMove;

        Action? pendingApply = null;

        if (ImGui.Begin("##SceneClusterDisambiguator", flags))
        {
            ImGui.TextColored(new Vector4(1.0f, 0.88f, 0.35f, 1.0f), "✦ Select Clustered Target");
            ImGui.TextDisabled("Click candidate or press 1-9 (Esc to close):");
            ImGui.Separator();

            for (int i = 0; i < _candidates.Count; i++)
            {
                var cand = _candidates[i];
                string prefix = i < 9 ? $"[{i + 1}] " : "";
                bool isHovered = i == _hoveredIndex;

                if (isHovered)
                {
                    ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.18f, 0.36f, 0.58f, 0.90f));
                }

                if (ImGui.Button($"{prefix}{cand.Title}##cluster_pick_{i}", new Vector2(300f, 0f)))
                {
                    pendingApply = cand.Apply;
                }

                if (ImGui.IsItemHovered())
                {
                    _hoveredIndex = i;
                }

                if (isHovered)
                {
                    ImGui.PopStyleColor();
                }

                if (!string.IsNullOrWhiteSpace(cand.Detail))
                {
                    ImGui.SameLine();
                    ImGui.TextDisabled(cand.Detail);
                }
            }

            ImGui.End();
        }

        ImGui.PopStyleColor(3);
        ImGui.PopStyleVar(3);

        if (pendingApply != null)
        {
            Close();
            pendingApply();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _ringMesh?.Dispose();
        _ringMesh = null;
        _pinMesh?.Dispose();
        _pinMesh = null;
        _shader?.Dispose();
        _shader = null;
    }
}
