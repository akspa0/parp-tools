using System;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;

namespace WoWViewer;

/// <summary>
/// Sky controls service: on/off, source selector, time-of-day slider + lock,
/// custom color overrides, and celestial (sun, moon, stars, clouds) toggles (Spec 292).
/// </summary>
internal sealed class SkyControlsService
{
    private readonly IViewerAppHost _host;

    internal SkyControlsService(IViewerAppHost host)
    {
        _host = host;
    }

    internal void DrawSkyControlsContent()
    {
        WorldScene? scene = _host.WorldScene;
        TerrainLighting? lighting = _host.TerrainManager?.Lighting;

        if (scene == null)
        {
            SharedUiWidgets.DrawEmptyState("No world loaded", "Load a world to configure atmosphere and sky settings.");
            return;
        }

        SkyDomeRenderer skyDome = scene.SkyDome;
        SceneAtmosphere atmosphere = scene.Atmosphere;

        // 1. Sky Enable Toggle
        bool skyEnabled = scene.ShowSky;
        if (ImGui.Checkbox("Enable Sky##SkyEnabled", ref skyEnabled))
        {
            scene.ShowSky = skyEnabled;
            _host.UiManager.BottomBar.BottomToggleSky = skyEnabled;
            _host.Settings.SaveViewerSettings();
        }

        // 2. Sky Source Selector
        int currentSource = (int)skyDome.SkySource;
        string[] sourceNames = ["Auto (Atmosphere + Dome)", "Procedural Dome Only", "Client Sky Model Only"];
        if (ImGui.Combo("Sky Source##SkySource", ref currentSource, sourceNames, sourceNames.Length))
        {
            skyDome.SkySource = (SkySource)currentSource;
            _host.Settings.SaveViewerSettings();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Auto blends procedural dome with client LightSkybox/Stars models. Select a mode to isolate.");

        // Report active client sky model if any
        if (atmosphere.HasActiveLightSkybox)
            ImGui.TextDisabled($"Client model: {atmosphere.ActiveLightSkyboxSourcePath}");
        else
            ImGui.TextDisabled("Client model: Procedural fallback");

        ImGui.Separator();
        ImGui.Text("Time of Day & Sun Cycle");

        // 3. Time-of-day slider (0 - 24 hours)
        float gameTime = lighting?.GameTime ?? 0.5f;
        float hours = gameTime * 24f;
        if (ImGui.SliderFloat("Hour##SkyHour", ref hours, 0f, 24f, "%.1f h"))
        {
            if (lighting != null)
            {
                lighting.GameTime = Math.Clamp(hours / 24f, 0f, 1f);
                lighting.HasManualGameTimeOverride = true;
            }
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Scrub through the 24-hour day/night lighting and solar cycle.");

        // Lock time toggle
        bool isLocked = lighting?.HasManualGameTimeOverride ?? false;
        if (ImGui.Checkbox("Lock Time of Day##SkyLockTime", ref isLocked))
        {
            if (lighting != null)
                lighting.HasManualGameTimeOverride = isLocked;
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Prevents automatic time progression so you can inspect fixed lighting conditions.");

        ImGui.SameLine();
        if (ImGui.SmallButton("Noon (12:00)##SkyNoon"))
        {
            if (lighting != null)
            {
                lighting.GameTime = 0.5f;
                lighting.HasManualGameTimeOverride = true;
            }
        }
        ImGui.SameLine();
        if (ImGui.SmallButton("Midnight (00:00)##SkyMidnight"))
        {
            if (lighting != null)
            {
                lighting.GameTime = 0.0f;
                lighting.HasManualGameTimeOverride = true;
            }
        }

        ImGui.Separator();
        ImGui.Text("Celestial Objects");

        // 4. Sun / Moon / Stars / Clouds toggles
        bool showSun = skyDome.ShowSun;
        if (ImGui.Checkbox("Sun##SkySun", ref showSun))
        {
            skyDome.ShowSun = showSun;
            _host.Settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        bool showMoon = skyDome.ShowMoon;
        if (ImGui.Checkbox("Moon##SkyMoon", ref showMoon))
        {
            skyDome.ShowMoon = showMoon;
            _host.Settings.SaveViewerSettings();
        }

        bool showStars = skyDome.ShowStars && atmosphere.ShowStars;
        if (ImGui.Checkbox("Stars##SkyStars", ref showStars))
        {
            skyDome.ShowStars = showStars;
            atmosphere.ShowStars = showStars;
            _host.Settings.SaveViewerSettings();
        }
        ImGui.SameLine();
        bool showClouds = skyDome.ShowClouds && atmosphere.ShowClouds;
        if (ImGui.Checkbox("Clouds##SkyClouds", ref showClouds))
        {
            skyDome.ShowClouds = showClouds;
            atmosphere.ShowClouds = showClouds;
            _host.Settings.SaveViewerSettings();
        }

        ImGui.Separator();
        ImGui.Text("Color Overrides");

        bool customColors = skyDome.CustomColorOverrides;
        if (ImGui.Checkbox("Override Sky Colors##SkyOverrideColors", ref customColors))
        {
            skyDome.CustomColorOverrides = customColors;
            _host.Settings.SaveViewerSettings();
        }

        if (customColors)
        {
            Vector3 zenith = skyDome.ZenithColor;
            if (ImGui.ColorEdit3("Zenith Color##SkyZenith", ref zenith))
            {
                skyDome.ZenithColor = zenith;
                _host.Settings.SaveViewerSettings();
            }

            Vector3 horizon = skyDome.HorizonColor;
            if (ImGui.ColorEdit3("Horizon Color##SkyHorizon", ref horizon))
            {
                skyDome.HorizonColor = horizon;
                _host.Settings.SaveViewerSettings();
            }

            Vector3 fog = skyDome.SkyFogColor;
            if (ImGui.ColorEdit3("Sky Fog Color##SkyFog", ref fog))
            {
                skyDome.SkyFogColor = fog;
                _host.Settings.SaveViewerSettings();
            }

            if (ImGui.SmallButton("Daytime Blue##SkyResetDay"))
            {
                skyDome.ZenithColor = new Vector3(0.25f, 0.45f, 0.85f);
                skyDome.HorizonColor = new Vector3(0.65f, 0.75f, 0.90f);
                skyDome.SkyFogColor = skyDome.HorizonColor;
                _host.Settings.SaveViewerSettings();
            }
            ImGui.SameLine();
            if (ImGui.SmallButton("Dusk Warm##SkyResetDusk"))
            {
                skyDome.ZenithColor = new Vector3(0.3f, 0.2f, 0.5f);
                skyDome.HorizonColor = new Vector3(0.9f, 0.5f, 0.3f);
                skyDome.SkyFogColor = skyDome.HorizonColor;
                _host.Settings.SaveViewerSettings();
            }
        }
        else
        {
            ImGui.TextDisabled("Colors dynamically driven by TerrainLighting & Light.dbc / LIT tables.");
        }
    }
}

