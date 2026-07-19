using System;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Lighting inspection panel: shows the resolved source, active colors/fog, and spatial LIT entry
/// diagnostics without allowing the inspection surfaces to alter lighting selection.
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
        DrawLitEntriesSection(_worldScene);
        ImGui.Separator();
        DrawLitSampleSection(_worldScene);
        ImGui.Separator();
        DrawEffectiveLightingSection(_worldScene);
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

        bool showLitMinimapMarkers = scene.ShowLitMinimapMarkers;
        if (ImGui.Checkbox("Show LIT minimap markers", ref showLitMinimapMarkers))
            scene.ShowLitMinimapMarkers = showLitMinimapMarkers;
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Shows positional LIT entries in the regular and full-screen minimaps. This does not change lighting or fog.");

        bool useLitFog = scene.UseLitFogOverride;
        if (ImGui.Checkbox("Use LIT fog override", ref useLitFog))
            scene.UseLitFogOverride = useLitFog;
    }

    private void DrawLitEntriesSection(WorldScene scene)
    {
        ImGui.SeparatorText("LIT entries");
        if (scene.LitLoader is not { HasData: true } lit)
        {
            ImGui.TextDisabled("No loaded LIT entries to map.");
            ImGui.TextDisabled("Enable a LIT visualization above to request the existing lazy load.");
            return;
        }

        ImGui.TextDisabled("Select an entry to highlight it on both minimaps. Double-click a positional entry to focus the 3D camera.");
        if (!ImGui.BeginChild("##LitEntries", new Vector2(0, 220f), true))
            return;

        float rowHeight = GetUniformListRowHeight();
        GetVisibleListRange(lit.Lights.Count, rowHeight, out int startIndex, out int endIndex);
        if (startIndex > 0)
            ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

        for (int index = startIndex; index < endIndex; index++)
        {
            LitLoader.LitLight light = lit.Lights[index];
            string label = $"[{index}] {light.DisplayName}";
            if (!light.IsNavigable)
                label += " (not mappable)";

            bool selected = scene.SelectedLitLightIndex == index;
            if (ImGui.Selectable(label, selected, ImGuiSelectableFlags.AllowDoubleClick))
            {
                scene.SelectedLitLightIndex = index;
                if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                    FocusCameraOnLitLight(index, closeFullscreenAfterFocus: false);
            }

            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.Text($"Position: ({light.Position.X:F1}, {light.Position.Y:F1}, {light.Position.Z:F1})");
                ImGui.Text($"Radius: {light.Radius:F1}  Dropoff: {light.Dropoff:F1}");
                ImGui.Text($"Chunk: ({light.ChunkX}, {light.ChunkY}) r={light.ChunkRadius}");
                ImGui.TextDisabled(light.IsNavigable
                    ? "Double-click to focus the 3D camera."
                    : "Default or invalid-position LIT entry; no minimap marker or camera target.");
                ImGui.EndTooltip();
            }
        }

        if (endIndex < lit.Lights.Count)
            ImGui.Dummy(new Vector2(0, (lit.Lights.Count - endIndex) * rowHeight));

        ImGui.EndChild();
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

    private void DrawEffectiveLightingSection(WorldScene scene)
    {
        ImGui.SeparatorText("Effective terrain lighting");

        if (scene.LightService is { } dbcLighting)
        {
            Vector4 statusColor = dbcLighting.ActiveLightId >= 0
                ? new Vector4(0.45f, 0.85f, 0.45f, 1f)
                : new Vector4(0.95f, 0.55f, 0.35f, 1f);
            ImGui.TextColored(statusColor, dbcLighting.ActiveLightId >= 0 ? "DBC ACTIVE" : "DBC INACTIVE");
            ImGui.TextWrapped($"Source: {dbcLighting.Source}");
            ImGui.TextWrapped(dbcLighting.Status);
        }
        else
        {
            ImGui.TextColored(new Vector4(0.95f, 0.55f, 0.35f, 1f), "DBC NOT LOADED");
            ImGui.TextDisabled("The loaded world has no exact-build Light DBC service.");
        }

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
        ImGui.Text($"Active fog start: {scene.ActiveFogStart:F1}   end: {scene.ActiveFogEnd:F1}");
        ImGui.TextDisabled($"Source: {scene.ActiveFogRangeSource}{(scene.ActiveFogRangeAdjusted ? " (normalized to stay visible)" : string.Empty)}");

        ImGui.SeparatorText("Active fog range");
        ImGui.TextDisabled("This affects the loaded world now. Settings contains load defaults only.");

        float fogStart = scene.HasUserFogRangeOverride ? scene.UserFogStart : scene.ActiveFogStart;
        float fogEnd = scene.HasUserFogRangeOverride ? scene.UserFogEnd : scene.ActiveFogEnd;
        bool fogStartChanged = ImGui.SliderFloat("Fog Start", ref fogStart, 0f, MaxTerrainFogDistance - 1f, "%.0f");
        bool fogEndChanged = ImGui.SliderFloat("Fog End", ref fogEnd, 1f, MaxTerrainFogDistance, "%.0f");
        if (fogStartChanged || fogEndChanged)
        {
            if (fogEnd <= fogStart)
            {
                if (fogEndChanged && !fogStartChanged)
                    fogStart = Math.Max(0f, fogEnd - 1f);
                else
                    fogEnd = Math.Min(MaxTerrainFogDistance, fogStart + 1f);
            }

            scene.SetUserFogRangeOverride(fogStart, fogEnd);
        }

        if (scene.HasUserFogRangeOverride)
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("Reset to lighting recommendation"))
                scene.ClearUserFogRangeOverride();
        }
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
