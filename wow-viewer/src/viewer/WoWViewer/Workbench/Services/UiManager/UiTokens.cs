using System.Numerics;

namespace WoWViewer;

/// <summary>
/// The single design-token set for the viewer shell (Spec 280 layout.md §8).
///
/// Providers ask for a token; they never hard-code a colour or a font size. The
/// viewer theme still owns the active palette, so the tokens here are the
/// structural values (spacing, sizes) plus the neutral/brass accents the
/// specification fixes for the 2026 surface treatment.
/// </summary>
internal static class UiTokens
{
    // Colour — surface treatment from layout.md §8.
    public static readonly Vector4 Surface0 = new(0.071f, 0.063f, 0.055f, 1f);
    public static readonly Vector4 Surface1 = new(0.106f, 0.094f, 0.082f, 1f);
    public static readonly Vector4 Surface2 = new(0.141f, 0.125f, 0.110f, 1f);
    public static readonly Vector4 Border = new(0.227f, 0.200f, 0.169f, 1f);
    public static readonly Vector4 Text = new(0.910f, 0.875f, 0.816f, 1f);
    public static readonly Vector4 TextMuted = new(0.604f, 0.561f, 0.494f, 1f);
    public static readonly Vector4 Brass = new(0.788f, 0.635f, 0.153f, 1f);
    public static readonly Vector4 BrassDim = new(0.541f, 0.435f, 0.118f, 1f);
    public static readonly Vector4 Danger = new(0.769f, 0.333f, 0.231f, 1f);
    public static readonly Vector4 Success = new(0.431f, 0.620f, 0.353f, 1f);

    // Type — one scale, one family.
    public const float Caption = 12f;
    public const float Body = 14f;
    public const float Label = 14f;
    public const float Title = 20f;

    // Spacing — 8 px grid (4 px steps allowed).
    public const float Space4 = 4f;
    public const float Space8 = 8f;
    public const float Space12 = 12f;
    public const float Space16 = 16f;
    public const float Space24 = 24f;
    public const float Space32 = 32f;
    public const float ControlHeight = 24f;
    public const float RowHeight = 28f;
    public const float SectionHeaderHeight = 28f;

    // Surface treatment — one window look for every manager-owned region (Spec 280 P8).
    // The theme service still owns the active palette; these are the structural
    // values that make every chrome region read as the same surface.
    public const float WindowRounding = 4f;
    public const float WindowBorderSize = 1f;
    public static readonly Vector2 WindowPadding = new(Space8, Space8);

    // Region metrics.
    public const float TopBarHeight = 28f;
    public const float BottomBarHeight = 32f;
    public const float StatusBarHeight = 22f;
    public const float SidebarWidth = 280f;
    public const float InspectorWidth = 360f;
    public const float MinimapSize = 180f;
    // The chat is a docked panel spanning the scene between the sidebars; only its height is a
    // token. It has no width token because it is never a floating rect.
    public const float ChatDockHeight = 180f;
    public const float OptionsWidth = 320f;
}
