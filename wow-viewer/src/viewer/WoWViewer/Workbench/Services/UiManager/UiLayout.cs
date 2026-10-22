using System.Numerics;

namespace WoWViewer;

internal enum UiRegion { Scene, TopBar, BottomBar, StatusBar, LeftSidebar, RightSidebar, RightDock, Options, Minimap, Chat }

internal readonly record struct UiRect(float X, float Y, float Width, float Height)
{
    public Vector2 Position => new(X, Y);
    public Vector2 Size => new(Width, Height);
    public bool Contains(float x, float y) => Width > 0 && Height > 0 && x >= X && y >= Y && x < X + Width && y < Y + Height;
}

internal readonly record struct UiRegionState(UiRect Rect, bool Visible, bool BlocksSceneInput = true);

/// <summary>
/// Pure logical-pixel layout. Sidebars overlay the scene; they never change its projection.
///
/// All metrics are expressed in 1080p logical pixels and multiplied by <c>scale</c>. The scale is
/// the larger of the width and height ratios, not the height alone: an ultrawide 3840x1080 window
/// has a 1080p height but a 4K width, and scaling by height alone left 280 px sidebars and a 12 px
/// font on a 3840 px-wide display.
/// </summary>
internal sealed class UiLayout
{
    public IReadOnlyDictionary<UiRegion, UiRegionState> Regions { get; }
    public UiRegionState this[UiRegion region] => Regions[region];

    /// <summary>Scale factor applied to every metric (1.0 at 1080p, 2.0 at 2160p).</summary>
    public float Scale { get; }

    public UiLayout(Vector2 display, bool hidden, bool left, bool right, float leftWidth, float rightWidth,
        bool options = false, bool minimapVisible = true, bool minimapCollapsed = false,
        bool chatVisible = true, bool chatCollapsed = false, float scale = 1f,
        bool rightDockVisible = false, float rightDockWidth = 360f)
    {
        float w = float.IsFinite(display.X) ? Math.Max(0, display.X) : 0;
        float h = float.IsFinite(display.Y) ? Math.Max(0, display.Y) : 0;
        float s = float.IsFinite(scale) ? Math.Clamp(scale, 1f, 2.5f) : 1f;
        Scale = s;

        float top = hidden ? 0 : Math.Min(28f * s, h);
        float status = Math.Min(22f * s, Math.Max(0, h - top));
        float bottom = hidden ? 0 : Math.Min(52f * s, Math.Max(0, h - top - status));
        float sceneHeight = Math.Max(0, h - top - bottom - status);

        // Edge regions are compact action rails, never full-width data sidebars. Their tool content
        // is opened in independent windows, so persisted sidebar widths cannot steal scene space.
        float railWidth = 44f * s;
        float lw = !hidden && left ? Math.Min(railWidth, w * 0.12f) : 0f;
        float rw = !hidden && right ? Math.Min(railWidth, w * 0.12f) : 0f;

        // A real minimap footprint: square, roughly 140 px at 1080p. The old docked height used
        // rw * 0.8f, which turned the minimap into a half-screen panel on a 1920 px window.
        float mapSize = (w >= 1920 ? 150f : w >= 1600 ? 140f : w >= 1280 ? 130f : 120f) * s;
        mapSize = Math.Min(mapSize, Math.Min(w, sceneHeight));
        if (minimapCollapsed) mapSize = Math.Min(32f * s, mapSize);

        // The minimap is screen chrome, never hosted in (or positioned relative to) a data sidebar.
        float mapPadding = 12f * s;
        var minimapRect = new UiRect(
            Math.Max(0f, w - mapSize - mapPadding),
            top + mapPadding,
            mapSize,
            mapSize);
        float rightTop = top + (minimapVisible ? mapSize + mapPadding * 2f : 0f);
        float rightHeight = Math.Max(0f, sceneHeight - (rightTop - top));

        // The right dock is the inspector/world/model/PM4 home: below the minimap, above the
        // bottom bars, and to the left of the icon rail. The user can drag its left edge.
        float minDock = 240f * s;
        float maxDock = MathF.Max(minDock, MathF.Min(w * 0.45f, w - rw - 80f * s));
        float dockWidth = Math.Clamp(rightDockWidth * s, minDock, maxDock);
        var rightDockRect = new UiRect(Math.Max(0f, w - rw - dockWidth), rightTop, dockWidth, rightHeight);

        // Chat is lower-left game chrome. It is independent of the sidebars and remains readable
        // when either data panel is opened. It overlays the scene instead of stealing its columns.
        float chatDockHeight = chatCollapsed
            ? Math.Min(28f * s, sceneHeight)
            : Math.Min(176f * s, Math.Max(0, sceneHeight * 0.28f));
        float chatDockX = lw + 10f * s;
        float chatDockWidth = Math.Min(420f * s, Math.Max(0f, Math.Min(w * 0.34f, w - lw - rw - 20f * s)));
        float chatDockY = top + Math.Max(0, sceneHeight - chatDockHeight);
        var chatRect = new UiRect(chatDockX, chatDockY, chatDockWidth, chatDockHeight);

        // Options anchors to the right sidebar's left edge, independent of the chat dock.
        float ow = Math.Min(420f * s, w);
        float oh = Math.Min(520f * s, sceneHeight);
        float ox = Math.Clamp(w - rw - ow - 8f * s, 0, Math.Max(0, w - ow));
        Regions = new Dictionary<UiRegion, UiRegionState>
        {
            [UiRegion.Scene] = new(new(0, top, w, sceneHeight), true, false),
            [UiRegion.TopBar] = new(new(0, 0, w, top), !hidden),
            [UiRegion.StatusBar] = new(new(0, h - status, w, status), true),
            [UiRegion.BottomBar] = new(new(0, h - status - bottom, w, bottom), !hidden),
            [UiRegion.LeftSidebar] = new(new(0, top, lw, sceneHeight), !hidden && left),
            [UiRegion.RightSidebar] = new(new(w - rw, rightTop, rw, rightHeight), !hidden && right),
            [UiRegion.RightDock] = new(rightDockRect, !hidden && right && rightDockVisible),
            [UiRegion.Options] = new(new(ox, Math.Max(top, h - status - bottom - oh), ow, oh), !hidden && options),
            [UiRegion.Minimap] = new(minimapRect, !hidden && minimapVisible),
            [UiRegion.Chat] = new(chatRect, !hidden && chatVisible),
        };
    }

    public static (bool Left, bool Right, float LeftWidth, float RightWidth) Defaults(float width)
    {
        _ = width;
        // Left/right are persistent icon rails. Their content opens in independent tool windows.
        return (true, true, 44f, 44f);
    }

    /// <summary>Initial bounds for opt-in tool windows, including bottom/right-screen clearance.</summary>
    internal static UiRect ToolWindowRect(Vector2 display, Vector2 preferredPosition, Vector2 logicalSize, float scale)
    {
        float s = float.IsFinite(scale) ? Math.Clamp(scale, 1f, 2.5f) : 1f;
        Vector2 bounds = new(
            float.IsFinite(display.X) ? MathF.Max(0f, display.X) : 0f,
            float.IsFinite(display.Y) ? MathF.Max(0f, display.Y) : 0f);
        Vector2 margin = Vector2.Min(new Vector2(8f * s), bounds * 0.1f);
        Vector2 size = Vector2.Min(logicalSize * s, Vector2.Max(Vector2.Zero, bounds - margin * 2f));
        size = Vector2.Min(size, bounds * new Vector2(0.62f, 0.78f));
        Vector2 position = Vector2.Clamp(preferredPosition, margin, Vector2.Max(margin, bounds - size - margin));
        return new(position.X, position.Y, size.X, size.Y);
    }

    /// <summary>
    /// UI scale for a display: 1.0 at 1920x1080, 2.0 at 3840x2160, clamped to 1.0–2.5. Uses the
    /// larger of the width and height ratios so ultrawide and short windows are not left at 1x.
    /// </summary>
    public static float ScaleFor(Vector2 display)
    {
        if (!float.IsFinite(display.X) || !float.IsFinite(display.Y) || display.X <= 0 || display.Y <= 0)
            return 1f;
        float byWidth = display.X / 1920f;
        float byHeight = display.Y / 1080f;
        return Math.Clamp(MathF.Max(byWidth, byHeight), 1f, 2.5f);
    }
}
