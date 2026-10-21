using System.Numerics;

namespace WoWViewer;

internal enum UiRegion { Scene, TopBar, BottomBar, StatusBar, LeftSidebar, RightSidebar, Options, Minimap, Chat }

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
/// All metrics are expressed in 1080p logical pixels and multiplied by <c>scale</c>, which the
/// manager derives from the display height. Without that, a 4K window renders 280 px sidebars and
/// a 12 px font, which is unusable.
/// </summary>
internal sealed class UiLayout
{
    public IReadOnlyDictionary<UiRegion, UiRegionState> Regions { get; }
    public UiRegionState this[UiRegion region] => Regions[region];

    /// <summary>Scale factor applied to every metric (1.0 at 1080p, 2.0 at 2160p).</summary>
    public float Scale { get; }

    public UiLayout(Vector2 display, bool hidden, bool left, bool right, float leftWidth, float rightWidth,
        bool options = false, UiRect? chat = null, bool minimapVisible = true, bool minimapCollapsed = false,
        bool minimapDocked = false, bool chatVisible = true, bool chatCollapsed = false, float scale = 1f)
    {
        float w = float.IsFinite(display.X) ? Math.Max(0, display.X) : 0;
        float h = float.IsFinite(display.Y) ? Math.Max(0, display.Y) : 0;
        float s = float.IsFinite(scale) ? Math.Clamp(scale, 1f, 2.5f) : 1f;
        Scale = s;

        float top = hidden ? 0 : Math.Min(28f * s, h);
        float status = Math.Min(22f * s, Math.Max(0, h - top));
        float bottom = hidden ? 0 : Math.Min(32f * s, Math.Max(0, h - top - status));
        float sceneHeight = Math.Max(0, h - top - bottom - status);

        float lw = hidden ? 0 : left ? SafeWidth(leftWidth * s, w) : Math.Min(24f * s, w / 2);
        float rw = hidden ? 0 : right ? SafeWidth(rightWidth * s, w) : Math.Min(24f * s, w / 2);

        float mapSize = (w >= 1920 ? 180f : w >= 1600 ? 160f : w >= 1280 ? 140f : 120f) * s;
        mapSize = Math.Min(mapSize, Math.Min(w, sceneHeight));
        if (minimapCollapsed) mapSize = Math.Min(32f * s, mapSize);

        bool docked = minimapDocked && right && !hidden;
        float rightTop = top;
        float rightHeight = sceneHeight;
        UiRect minimapRect;
        if (docked)
        {
            // Docked: a first-class panel at the top of the right sidebar, as wide as the
            // sidebar and tall enough to be a usable map (not a thumbnail).
            float dockHeight = Math.Min(sceneHeight, Math.Max(mapSize, rw * 0.8f));
            minimapRect = new UiRect(w - rw, top, rw, dockHeight);
            rightTop = top + dockHeight;
            rightHeight = Math.Max(0, sceneHeight - dockHeight);
        }
        else
        {
            float rightOffset = right ? rw : 0;
            minimapRect = new UiRect(Math.Max(0, w - rightOffset - mapSize - 8f * s), top + 8f * s, mapSize, mapSize);
        }

        // The chat is a docked panel, not a floating window. It spans the scene between the
        // sidebars and sits on the bottom edge of the scene area, so it can never cover a
        // sidebar, the bottom bar or the status bar, and there is no drag/resize handle to lose.
        float chatDockHeight = chatCollapsed
            ? Math.Min(28f * s, sceneHeight)
            : Math.Min(180f * s, Math.Max(0, sceneHeight * 0.28f));
        float chatDockX = lw;
        float chatDockWidth = Math.Max(0, w - lw - rw);
        float chatDockY = top + Math.Max(0, sceneHeight - chatDockHeight);
        var chatRect = chat ?? new UiRect(chatDockX, chatDockY, chatDockWidth, chatDockHeight);

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
            [UiRegion.LeftSidebar] = new(new(0, top, lw, sceneHeight), !hidden),
            [UiRegion.RightSidebar] = new(new(w - rw, rightTop, rw, rightHeight), !hidden),
            [UiRegion.Options] = new(new(ox, Math.Max(top, h - status - bottom - oh), ow, oh), !hidden && options),
            [UiRegion.Minimap] = new(minimapRect, !hidden && minimapVisible),
            [UiRegion.Chat] = new(chatRect, !hidden && chatVisible),
        };
    }

    private static float SafeWidth(float value, float display) =>
        Math.Clamp(float.IsFinite(value) ? value : 280, Math.Min(180, display / 2), Math.Max(0, display / 2));

    public static (bool Left, bool Right, float LeftWidth, float RightWidth) Defaults(float width) =>
        (width >= 1600, width >= 1280, width >= 1920 ? 280 : 240, width >= 1920 ? 360 : 320);

    /// <summary>UI scale for a display: 1.0 at 1080p, 2.0 at 2160p, clamped to 1.0–2.5.</summary>
    public static float ScaleFor(Vector2 display) =>
        float.IsFinite(display.Y) && display.Y > 0 ? Math.Clamp(display.Y / 1080f, 1f, 2.5f) : 1f;
}
