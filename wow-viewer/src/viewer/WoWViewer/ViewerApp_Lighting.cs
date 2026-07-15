using System;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Read-only lighting status panel: shows whether a LIT file resolved, which one, and the
/// colours/fog actually reaching the renderer. Exists so a synthetic-minimap capture can be
/// confirmed to be lit as intended before it becomes training data.
/// </summary>
public partial class ViewerApp
{
    private void DrawLightingContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to inspect lighting.");
            return;
        }

        DrawLitStatusSection(_worldScene);
        ImGui.Separator();
        DrawLitSampleSection(_worldScene);
        ImGui.Separator();
        DrawEffectiveLightingSection();
    }

    private void DrawLitStatusSection(WorldScene scene)
    {
        ImGui.SeparatorText("LIT source");

        bool loaded = scene.LitLoader != null;
        ImGui.TextColored(
            loaded ? new Vector4(0.45f, 0.85f, 0.45f, 1f) : new Vector4(0.85f, 0.55f, 0.35f, 1f),
            loaded ? "LOADED" : (scene.LitLoadAttempted ? "NOT LOADED" : "NOT ATTEMPTED"));

        ImGui.TextWrapped(scene.LitStatus);

        if (scene.LitLoader is { } lit)
        {
            ImGui.Text($"Version: {DescribeLitVersion(lit.Version)}");
            ImGui.Text($"Lights: {lit.Lights.Count}");
        }

        string? source = scene.SelectedLitSourcePath;
        ImGui.Text("Source:");
        ImGui.SameLine();
        if (string.IsNullOrWhiteSpace(source))
            ImGui.TextDisabled("(none)");
        else
            ImGui.TextWrapped(source);

        var available = scene.AvailableLitSourcePaths;
        if (available.Count > 0 && ImGui.TreeNode($"Available LIT files ({available.Count})###LitAvailable"))
        {
            foreach (string path in available)
                ImGui.BulletText(path);
            ImGui.TreePop();
        }

        // These toggles lazy-load the LIT, so they double as the way to force a load attempt.
        bool showLitLights = scene.ShowLitLights;
        if (ImGui.Checkbox("Show LIT lights", ref showLitLights))
            scene.ShowLitLights = showLitLights;

        bool useLitFog = scene.UseLitFogOverride;
        if (ImGui.Checkbox("Use LIT fog override", ref useLitFog))
            scene.UseLitFogOverride = useLitFog;
    }

    private static void DrawLitSampleSection(WorldScene scene)
    {
        ImGui.SeparatorText("Last LIT sample");

        if (scene.LastLitSample is not { } sample)
        {
            ImGui.TextDisabled("No LIT sample evaluated yet.");
            ImGui.TextDisabled("Enable a LIT toggle above and move the camera in a lit world.");
            return;
        }

        ImGui.Text($"Dominant light: [{sample.DominantLightIndex}] {sample.DominantLightName}");
        ImGui.Text($"Weight: {sample.DominantWeight:F3}");
        ImGui.Text($"Time of day: {sample.TimeOfDay:F3} ({DescribeTimeOfDay(sample.TimeOfDay)})");

        DrawColorRow("Direct", sample.DirectColor);
        DrawColorRow("Ambient", sample.AmbientColor);
        DrawColorRow("Fog", sample.FogColor);
        DrawColorRow("Sky top", sample.SkyTopColor);
        DrawColorRow("Sky horizon", sample.SkyHorizonColor);

        ImGui.Text($"Fog start: {sample.FogStart:F1}   end: {sample.FogEnd:F1}   startScalar: {sample.FogStartScalar:F3}");
    }

    private void DrawEffectiveLightingSection()
    {
        ImGui.SeparatorText("Effective terrain lighting");

        TerrainLighting? lighting = _terrainManager?.Lighting ?? _vlmTerrainManager?.Lighting;
        if (lighting == null)
        {
            ImGui.TextDisabled("No terrain lighting active.");
            return;
        }

        ImGui.Text($"Game time: {lighting.GameTime:F3} ({DescribeTimeOfDay(lighting.GameTime)})");

        Vector3 dir = lighting.LightDirection;
        ImGui.Text($"Light dir: ({dir.X:F3}, {dir.Y:F3}, {dir.Z:F3})");
        if (lighting.HasExternalLightDirectionOverride)
        {
            Vector3 ext = lighting.ExternalLightDirection;
            ImGui.TextColored(new Vector4(0.55f, 0.75f, 1f, 1f),
                $"  overridden externally: ({ext.X:F3}, {ext.Y:F3}, {ext.Z:F3})");
        }

        DrawColorRow("Light", lighting.LightColor);
        DrawColorRow("Ambient", lighting.AmbientColor);
        DrawColorRow("Fog", lighting.FogColor);
        ImGui.Text($"Fog start: {lighting.FogStart:F1}   end: {lighting.FogEnd:F1}");
    }

    private static void DrawColorRow(string label, Vector3 color)
    {
        Vector4 swatch = new(color.X, color.Y, color.Z, 1f);
        ImGui.ColorButton($"##{label}Swatch", swatch, ImGuiColorEditFlags.NoTooltip, new Vector2(16, 16));
        ImGui.SameLine();
        ImGui.Text($"{label}: ({color.X:F3}, {color.Y:F3}, {color.Z:F3})");
    }

    private static string DescribeLitVersion(uint version) => version switch
    {
        LitLoader.Version83 => "v83 (0x80000003)",
        LitLoader.Version84 => "v84 (0x80000004)",
        LitLoader.Version85 => "v85 (0x80000005)",
        LitLoader.Version02Test => "v2 test (0x00000002)",
        _ => $"unknown (0x{version:X8})",
    };

    private static string DescribeTimeOfDay(float normalized)
    {
        int minutes = (int)MathF.Round(Math.Clamp(normalized, 0f, 1f) * 1440f) % 1440;
        return $"{minutes / 60:D2}:{minutes % 60:D2}";
    }
}
