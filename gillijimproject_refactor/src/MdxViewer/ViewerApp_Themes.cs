using System.Numerics;
using ImGuiNET;

namespace MdxViewer;

public partial class ViewerApp
{
    private enum UiThemeKind
    {
        ModernSlate = 0,
        PreAlphaBrass = 1,
    }

    private static readonly UiThemeKind[] UiThemeOptions =
    {
        UiThemeKind.ModernSlate,
        UiThemeKind.PreAlphaBrass,
    };

    private UiThemeKind _uiTheme = UiThemeKind.ModernSlate;

    private void ApplyActiveUiTheme()
    {
        ApplyUiTheme(_uiTheme, persist: false);
    }

    private void SetUiTheme(UiThemeKind theme)
    {
        if (_uiTheme == theme)
            return;

        ApplyUiTheme(theme, persist: true);
    }

    private void ResetActiveUiThemeToDefaults()
    {
        ApplyUiTheme(_uiTheme, persist: true);
    }

    private void ApplyUiTheme(UiThemeKind theme, bool persist)
    {
        _uiTheme = theme;

        ImGuiStylePtr style = ImGui.GetStyle();
        ImGui.StyleColorsDark();

        Vector4 clearColor;
        switch (theme)
        {
            case UiThemeKind.PreAlphaBrass:
                clearColor = new Vector4(0.05f, 0.07f, 0.10f, 1f);
                ApplyPreAlphaBrassTheme(style);
                break;

            default:
                clearColor = new Vector4(0.05f, 0.05f, 0.10f, 1f);
                ApplyModernSlateTheme(style);
                break;
        }

        _gl.ClearColor(clearColor.X, clearColor.Y, clearColor.Z, clearColor.W);

        if (persist)
            SaveViewerSettings();
    }

    private void DrawUiThemeSettingsContent()
    {
        ImGui.Text("UI Theme");
        ImGui.TextDisabled("Theme changes chrome and palette first; shell layout changes come later.");

        if (ImGui.BeginCombo("Style", GetUiThemeLabel(_uiTheme)))
        {
            foreach (UiThemeKind option in UiThemeOptions)
            {
                bool selected = option == _uiTheme;
                if (ImGui.Selectable(GetUiThemeLabel(option), selected))
                    SetUiTheme(option);

                if (selected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        ImGui.TextWrapped(GetUiThemeDescription(_uiTheme));

        if (ImGui.SmallButton("Restore Theme Defaults"))
            ResetActiveUiThemeToDefaults();
    }

    private static string GetUiThemeLabel(UiThemeKind theme)
    {
        return theme switch
        {
            UiThemeKind.PreAlphaBrass => "Pre-Alpha Brass",
            _ => "Modern Slate"
        };
    }

    private static string GetUiThemeDescription(UiThemeKind theme)
    {
        return theme switch
        {
            UiThemeKind.PreAlphaBrass => "Square brass-framed chrome with a muted parchment and navy palette inspired by the early 2001-era UI direction.",
            _ => "The current slate viewer chrome with restrained rounding and neutral dark panels."
        };
    }

    private static void ApplyModernSlateTheme(ImGuiStylePtr style)
    {
        style.WindowRounding = 4f;
        style.ChildRounding = 3f;
        style.FrameRounding = 2f;
        style.PopupRounding = 3f;
        style.ScrollbarRounding = 3f;
        style.GrabRounding = 2f;
        style.TabRounding = 3f;
        style.WindowBorderSize = 1f;
        style.ChildBorderSize = 1f;
        style.FrameBorderSize = 0f;
        style.PopupBorderSize = 1f;
        style.TabBorderSize = 0f;
        style.WindowPadding = new Vector2(10f, 10f);
        style.FramePadding = new Vector2(8f, 4f);
        style.ItemSpacing = new Vector2(8f, 6f);

        style.Colors[(int)ImGuiCol.Text] = Rgba(231, 235, 242);
        style.Colors[(int)ImGuiCol.TextDisabled] = Rgba(143, 152, 168);
        style.Colors[(int)ImGuiCol.WindowBg] = Rgba(31, 33, 39, 242);
        style.Colors[(int)ImGuiCol.ChildBg] = Rgba(24, 26, 31, 212);
        style.Colors[(int)ImGuiCol.PopupBg] = Rgba(22, 24, 29, 245);
        style.Colors[(int)ImGuiCol.Border] = Rgba(72, 79, 93, 196);
        style.Colors[(int)ImGuiCol.BorderShadow] = Rgba(0, 0, 0, 0);
        style.Colors[(int)ImGuiCol.FrameBg] = Rgba(48, 53, 64, 196);
        style.Colors[(int)ImGuiCol.FrameBgHovered] = Rgba(67, 83, 108, 220);
        style.Colors[(int)ImGuiCol.FrameBgActive] = Rgba(78, 96, 125, 235);
        style.Colors[(int)ImGuiCol.TitleBg] = Rgba(33, 39, 52, 255);
        style.Colors[(int)ImGuiCol.TitleBgActive] = Rgba(46, 57, 79, 255);
        style.Colors[(int)ImGuiCol.MenuBarBg] = Rgba(38, 42, 50, 255);
        style.Colors[(int)ImGuiCol.ScrollbarBg] = Rgba(18, 20, 24, 160);
        style.Colors[(int)ImGuiCol.ScrollbarGrab] = Rgba(78, 85, 98, 180);
        style.Colors[(int)ImGuiCol.ScrollbarGrabHovered] = Rgba(96, 106, 124, 210);
        style.Colors[(int)ImGuiCol.ScrollbarGrabActive] = Rgba(118, 130, 150, 230);
        style.Colors[(int)ImGuiCol.CheckMark] = Rgba(187, 209, 255);
        style.Colors[(int)ImGuiCol.SliderGrab] = Rgba(115, 137, 176, 190);
        style.Colors[(int)ImGuiCol.SliderGrabActive] = Rgba(153, 180, 227, 235);
        style.Colors[(int)ImGuiCol.Button] = Rgba(61, 69, 84, 215);
        style.Colors[(int)ImGuiCol.ButtonHovered] = Rgba(81, 97, 126, 232);
        style.Colors[(int)ImGuiCol.ButtonActive] = Rgba(96, 117, 152, 245);
        style.Colors[(int)ImGuiCol.Header] = Rgba(56, 68, 87, 210);
        style.Colors[(int)ImGuiCol.HeaderHovered] = Rgba(77, 95, 121, 228);
        style.Colors[(int)ImGuiCol.HeaderActive] = Rgba(91, 112, 145, 240);
        style.Colors[(int)ImGuiCol.Separator] = Rgba(80, 88, 105, 186);
        style.Colors[(int)ImGuiCol.SeparatorHovered] = Rgba(118, 130, 153, 218);
        style.Colors[(int)ImGuiCol.SeparatorActive] = Rgba(153, 168, 196, 232);
        style.Colors[(int)ImGuiCol.ResizeGrip] = Rgba(80, 94, 122, 140);
        style.Colors[(int)ImGuiCol.ResizeGripHovered] = Rgba(116, 137, 176, 204);
        style.Colors[(int)ImGuiCol.ResizeGripActive] = Rgba(149, 176, 224, 224);
        style.Colors[(int)ImGuiCol.Tab] = Rgba(40, 47, 59, 255);
        style.Colors[(int)ImGuiCol.TabHovered] = Rgba(74, 89, 115, 255);
        style.Colors[(int)ImGuiCol.TabActive] = Rgba(58, 69, 89, 255);
        style.Colors[(int)ImGuiCol.DockingPreview] = Rgba(106, 141, 214, 180);
    }

    private static void ApplyPreAlphaBrassTheme(ImGuiStylePtr style)
    {
        style.WindowRounding = 0f;
        style.ChildRounding = 0f;
        style.FrameRounding = 0f;
        style.PopupRounding = 0f;
        style.ScrollbarRounding = 0f;
        style.GrabRounding = 0f;
        style.TabRounding = 0f;
        style.WindowBorderSize = 1.5f;
        style.ChildBorderSize = 1f;
        style.FrameBorderSize = 1f;
        style.PopupBorderSize = 1.5f;
        style.TabBorderSize = 1f;
        style.WindowPadding = new Vector2(11f, 10f);
        style.FramePadding = new Vector2(9f, 5f);
        style.ItemSpacing = new Vector2(8f, 7f);
        style.ItemInnerSpacing = new Vector2(6f, 4f);

        style.Colors[(int)ImGuiCol.Text] = Rgba(229, 214, 170);
        style.Colors[(int)ImGuiCol.TextDisabled] = Rgba(151, 140, 112);
        style.Colors[(int)ImGuiCol.WindowBg] = Rgba(17, 24, 34, 244);
        style.Colors[(int)ImGuiCol.ChildBg] = Rgba(21, 31, 44, 222);
        style.Colors[(int)ImGuiCol.PopupBg] = Rgba(16, 22, 31, 248);
        style.Colors[(int)ImGuiCol.Border] = Rgba(162, 128, 70, 224);
        style.Colors[(int)ImGuiCol.BorderShadow] = Rgba(48, 30, 8, 96);
        style.Colors[(int)ImGuiCol.FrameBg] = Rgba(57, 64, 54, 208);
        style.Colors[(int)ImGuiCol.FrameBgHovered] = Rgba(82, 93, 73, 226);
        style.Colors[(int)ImGuiCol.FrameBgActive] = Rgba(112, 96, 58, 236);
        style.Colors[(int)ImGuiCol.TitleBg] = Rgba(35, 47, 63, 255);
        style.Colors[(int)ImGuiCol.TitleBgActive] = Rgba(67, 53, 28, 255);
        style.Colors[(int)ImGuiCol.MenuBarBg] = Rgba(40, 51, 64, 255);
        style.Colors[(int)ImGuiCol.ScrollbarBg] = Rgba(12, 17, 24, 170);
        style.Colors[(int)ImGuiCol.ScrollbarGrab] = Rgba(120, 96, 54, 204);
        style.Colors[(int)ImGuiCol.ScrollbarGrabHovered] = Rgba(151, 124, 73, 222);
        style.Colors[(int)ImGuiCol.ScrollbarGrabActive] = Rgba(184, 152, 98, 236);
        style.Colors[(int)ImGuiCol.CheckMark] = Rgba(212, 180, 105);
        style.Colors[(int)ImGuiCol.SliderGrab] = Rgba(163, 127, 69, 214);
        style.Colors[(int)ImGuiCol.SliderGrabActive] = Rgba(208, 171, 103, 236);
        style.Colors[(int)ImGuiCol.Button] = Rgba(64, 71, 54, 216);
        style.Colors[(int)ImGuiCol.ButtonHovered] = Rgba(99, 87, 53, 232);
        style.Colors[(int)ImGuiCol.ButtonActive] = Rgba(128, 104, 58, 242);
        style.Colors[(int)ImGuiCol.Header] = Rgba(54, 63, 78, 218);
        style.Colors[(int)ImGuiCol.HeaderHovered] = Rgba(85, 76, 52, 232);
        style.Colors[(int)ImGuiCol.HeaderActive] = Rgba(117, 96, 58, 242);
        style.Colors[(int)ImGuiCol.Separator] = Rgba(132, 104, 63, 188);
        style.Colors[(int)ImGuiCol.SeparatorHovered] = Rgba(180, 146, 90, 224);
        style.Colors[(int)ImGuiCol.SeparatorActive] = Rgba(209, 176, 116, 236);
        style.Colors[(int)ImGuiCol.ResizeGrip] = Rgba(109, 85, 47, 150);
        style.Colors[(int)ImGuiCol.ResizeGripHovered] = Rgba(160, 129, 77, 212);
        style.Colors[(int)ImGuiCol.ResizeGripActive] = Rgba(208, 171, 103, 226);
        style.Colors[(int)ImGuiCol.Tab] = Rgba(28, 39, 53, 255);
        style.Colors[(int)ImGuiCol.TabHovered] = Rgba(93, 78, 48, 255);
        style.Colors[(int)ImGuiCol.TabActive] = Rgba(67, 52, 30, 255);
        style.Colors[(int)ImGuiCol.DockingPreview] = Rgba(180, 146, 90, 164);
    }

    private static Vector4 Rgba(byte r, byte g, byte b, byte a = 255)
    {
        const float scale = 1f / 255f;
        return new Vector4(r * scale, g * scale, b * scale, a * scale);
    }
}