using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.UI;

namespace WoWViewer;

internal sealed partial class ViewerChromeService
{
    private void DrawObjectTransparencyContent(string idSuffix, bool hasWmo, bool hasM2, float sliderWidth = 180f)
    {
        SharedUiWidgets.SectionTitle("Object Transparency & Visibility");
        SharedUiWidgets.Divider();

        if (hasWmo)
        {
            SharedUiWidgets.SectionTitle("WMO Buildings & Structures");
            bool wmoTex = WmoRenderer.TexturesEnabled;
            if (ImGui.Checkbox($"Enable WMO Textures##pop_tex_{idSuffix}", ref wmoTex))
            {
                WmoRenderer.TexturesEnabled = wmoTex;
                _settings.SaveViewerSettings();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Toggle textures for WMO objects. Uncheck to render untextured clay geometry.");

            float wmoOpacity = WmoRenderer.GlobalOpacity * 100f;
            if (sliderWidth > 0f) ImGui.SetNextItemWidth(sliderWidth);
            else ImGui.SetNextItemWidth(-1f);
            if (SharedUiWidgets.DrawCompactSlider($"WMO Opacity##pop_wmo_{idSuffix}", ref wmoOpacity, 0f, 100f, "%.0f%%"))
                WmoRenderer.GlobalOpacity = Math.Clamp(wmoOpacity / 100f, 0f, 1f);
            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();

            SharedUiWidgets.DrawButtonRow(
                SharedUiWidgets.Btn($"0%##wmo0_{idSuffix}", () => { WmoRenderer.GlobalOpacity = 0f; _settings.SaveViewerSettings(); }, help: "Completely invisible WMOs", small: true),
                SharedUiWidgets.Btn($"25%##wmo25_{idSuffix}", () => { WmoRenderer.GlobalOpacity = 0.25f; _settings.SaveViewerSettings(); }, small: true),
                SharedUiWidgets.Btn($"50%##wmo50_{idSuffix}", () => { WmoRenderer.GlobalOpacity = 0.5f; _settings.SaveViewerSettings(); }, small: true),
                SharedUiWidgets.Btn($"75% See-Thru##wmo75_{idSuffix}", () => { WmoRenderer.GlobalOpacity = 0.75f; _settings.SaveViewerSettings(); }, help: "See inside WMO buildings while preserving exterior cues", small: true),
                SharedUiWidgets.Btn($"100% Solid##wmo100_{idSuffix}", () => { WmoRenderer.GlobalOpacity = 1f; _settings.SaveViewerSettings(); }, small: true));

            ImGui.Spacing();
            SharedUiWidgets.SectionTitle("Collision Walls");
            float wallOpacity = WmoRenderer.CollisionWallOpacity * 100f;
            if (sliderWidth > 0f) ImGui.SetNextItemWidth(sliderWidth);
            else ImGui.SetNextItemWidth(-1f);
            if (SharedUiWidgets.DrawCompactSlider($"Wall Opacity##pop_wall_{idSuffix}", ref wallOpacity, 0f, 100f, "%.0f%%"))
                WmoRenderer.CollisionWallOpacity = Math.Clamp(wallOpacity / 100f, 0f, 1f);
            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();

            SharedUiWidgets.DrawButtonRow(
                SharedUiWidgets.Btn($"0% Hidden##wall0_{idSuffix}", () => { WmoRenderer.CollisionWallOpacity = 0f; _settings.SaveViewerSettings(); }, small: true),
                SharedUiWidgets.Btn($"50% Translucent##wall50_{idSuffix}", () => { WmoRenderer.CollisionWallOpacity = 0.5f; _settings.SaveViewerSettings(); }, small: true),
                SharedUiWidgets.Btn($"100% Solid##wall100_{idSuffix}", () => { WmoRenderer.CollisionWallOpacity = 1f; _settings.SaveViewerSettings(); }, small: true));

            if (hasM2)
                SharedUiWidgets.Divider();
        }

        if (hasM2)
        {
            SharedUiWidgets.SectionTitle("M2 & MDX Doodads / Models");
            float m2Opacity = M2Renderer.GlobalOpacity * 100f;
            if (sliderWidth > 0f) ImGui.SetNextItemWidth(sliderWidth);
            else ImGui.SetNextItemWidth(-1f);
            if (SharedUiWidgets.DrawCompactSlider($"M2 Opacity##pop_m2_{idSuffix}", ref m2Opacity, 0f, 100f, "%.0f%%"))
            {
                M2Renderer.GlobalOpacity = Math.Clamp(m2Opacity / 100f, 0f, 1f);
                MdxRenderer.GlobalOpacity = M2Renderer.GlobalOpacity;
            }
            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();

            SharedUiWidgets.DrawButtonRow(
                SharedUiWidgets.Btn($"0% Invisible##m2_0_{idSuffix}", () => { M2Renderer.GlobalOpacity = 0f; MdxRenderer.GlobalOpacity = 0f; _settings.SaveViewerSettings(); }, small: true),
                SharedUiWidgets.Btn($"50% Translucent##m2_50_{idSuffix}", () => { M2Renderer.GlobalOpacity = 0.5f; MdxRenderer.GlobalOpacity = 0.5f; _settings.SaveViewerSettings(); }, small: true),
                SharedUiWidgets.Btn($"75% See-Thru##m2_75_{idSuffix}", () => { M2Renderer.GlobalOpacity = 0.75f; MdxRenderer.GlobalOpacity = 0.75f; _settings.SaveViewerSettings(); }, small: true),
                SharedUiWidgets.Btn($"100% Solid##m2_100_{idSuffix}", () => { M2Renderer.GlobalOpacity = 1f; MdxRenderer.GlobalOpacity = 1f; _settings.SaveViewerSettings(); }, small: true));
        }

        SharedUiWidgets.Divider();
        if (SharedUiWidgets.ActionButton($"Reset All to Solid (100%)##reset_{idSuffix}"))
        {
            WmoRenderer.GlobalOpacity = 1f;
            M2Renderer.GlobalOpacity = 1f;
            MdxRenderer.GlobalOpacity = 1f;
            WmoRenderer.CollisionWallOpacity = 1f;
            WmoRenderer.TexturesEnabled = true;
            _settings.SaveViewerSettings();
        }
    }

}
