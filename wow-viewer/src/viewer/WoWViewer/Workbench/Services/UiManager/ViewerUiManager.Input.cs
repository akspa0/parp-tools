using System.Numerics;
using System.Reflection;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL.Extensions.ImGui;
namespace WoWViewer;
internal sealed partial class ViewerUiManager
{
    private static readonly MethodInfo? ImGuiControllerWindowResizedMethod = typeof(ImGuiController).GetMethod("WindowResized", BindingFlags.Instance | BindingFlags.NonPublic);
    private readonly Lock _pendingImGuiMouseEventLock = new();
    private readonly Queue<(int ButtonIndex, bool Down)> _pendingImGuiMouseButtonEvents = new();
    private Vector2D<int> _lastSyncedImGuiWindowSize;
    private Vector2D<int> _lastSyncedImGuiFramebufferSize;
    internal static bool HasImGuiContext() => ImGui.GetCurrentContext() != IntPtr.Zero;
    internal bool CanSceneConsumeMouse(float x, float y)
    {
        if (!HasImGuiContext()) return false;
        UpdateLayout(DisplaySize);
        if (!Layout[UiRegion.Scene].Rect.Contains(x, y) || _popups.IsOpen || ImGui.GetIO().WantCaptureMouse) return false;
        foreach (var state in Layout.Regions.Values)
            if (state.Visible && state.BlocksSceneInput && state.Rect.Contains(x, y)) return false;
        foreach (var state in _hosted.Values)
            if (state.Visible && state.BlocksSceneInput && state.Rect.Contains(x, y)) return false;
        return !_host.FullscreenMinimap;
    }
    internal bool IsSceneMouseCaptureBlocked(float x, float y) => !CanSceneConsumeMouse(x, y);
    internal bool IsSceneKeyboardCaptureBlocked() => !HasImGuiContext() || _popups.IsOpen || ImGui.GetIO().WantCaptureKeyboard || ImGui.GetIO().WantTextInput;
    internal bool TryGetSceneViewportRect(out float x, out float y, out float width, out float height)
    {
        UpdateLayout(HasImGuiContext() ? DisplaySize : new Vector2(_host.Window.Size.X, _host.Window.Size.Y));
        var rect = Layout[UiRegion.Scene].Rect;
        x = rect.X; y = rect.Y; width = rect.Width; height = rect.Height;
        return width > 0 && height > 0;
    }
    internal void HandleChromeShortcuts()
    {
        if (ImGui.GetIO().WantTextInput) return;
        if (ImGui.IsKeyPressed(ImGuiKey.F1)) _host.MainMenuBar.RequestKeyboardShortcuts();
        if (ImGui.IsKeyPressed(ImGuiKey.LeftBracket)) { _host.ShowLeftSidebar = !_host.ShowLeftSidebar; SaveLayout(); }
        if (ImGui.IsKeyPressed(ImGuiKey.RightBracket)) { _host.ShowRightSidebar = !_host.ShowRightSidebar; SaveLayout(); }
        if (ImGui.IsKeyPressed(ImGuiKey.Tab)) _host.HideUiChrome = !_host.HideUiChrome;
        if (ImGui.IsKeyPressed(ImGuiKey.M)) { _minimapVisible = !_minimapVisible; SaveLayout(); }
        if (ImGui.IsKeyPressed(ImGuiKey.I)) OpenViewer(Workbench.ViewerPanel.Inspector);
        if (ImGui.IsKeyPressed(ImGuiKey.P)) OpenViewer(Workbench.ViewerPanel.PM4);
        if (ImGui.IsKeyPressed(ImGuiKey.Escape)) _popups.CloseTopmost();
    }
    internal void QueueImGuiMouseButtonEvent(MouseButton button, bool down)
    {
        int? buttonIndex = button switch
        {
            MouseButton.Left => 0,
            MouseButton.Right => 1,
            MouseButton.Middle => 2,
            _ => null,
        };

        if (!buttonIndex.HasValue)
            return;

        lock (_pendingImGuiMouseEventLock)
        {
            _pendingImGuiMouseButtonEvents.Enqueue((buttonIndex.Value, down));
        }
    }

    internal void FlushPendingImGuiMouseButtonEvents()
    {
        lock (_pendingImGuiMouseEventLock)
        {
            if (_pendingImGuiMouseButtonEvents.Count == 0)
                return;

            var io = ImGui.GetIO();
            while (_pendingImGuiMouseButtonEvents.Count > 0)
            {
                var (buttonIndex, down) = _pendingImGuiMouseButtonEvents.Dequeue();
                io.AddMouseButtonEvent(buttonIndex, down);
            }
        }
    }

    internal bool TryGetSceneFramebufferViewport(out int x, out int y, out uint width, out uint height)
    {
        x = y = 0;
        width = height = 0;

        if (!TryGetSceneViewportRect(out float viewportX, out float viewportY, out float viewportWidth, out float viewportHeight))
            return false;

        Vector2D<int> windowSize = _host.Window.Size;
        Vector2D<int> framebufferSize = _host.Window.FramebufferSize;
        if (windowSize.X <= 0 || windowSize.Y <= 0 || framebufferSize.X <= 0 || framebufferSize.Y <= 0)
            return false;

        float scaleX = (float)framebufferSize.X / windowSize.X;
        float scaleY = (float)framebufferSize.Y / windowSize.Y;

        int viewportLeft = (int)MathF.Round(viewportX * scaleX);
        int viewportTop = (int)MathF.Round(viewportY * scaleY);
        int viewportRight = (int)MathF.Round((viewportX + viewportWidth) * scaleX);
        int viewportBottom = (int)MathF.Round((viewportY + viewportHeight) * scaleY);

        viewportLeft = Math.Clamp(viewportLeft, 0, framebufferSize.X);
        viewportRight = Math.Clamp(viewportRight, viewportLeft, framebufferSize.X);
        viewportTop = Math.Clamp(viewportTop, 0, framebufferSize.Y);
        viewportBottom = Math.Clamp(viewportBottom, viewportTop, framebufferSize.Y);

        x = viewportLeft;
        y = framebufferSize.Y - viewportBottom;
        width = (uint)Math.Max(1, viewportRight - viewportLeft);
        height = (uint)Math.Max(1, viewportBottom - viewportTop);
        return true;
    }

    internal void SyncImGuiWindowMetrics(Vector2D<int> windowSize, Vector2D<int> framebufferSize)
    {
        if (_host.ImGui == null || !HasImGuiContext())
            return;

        if (windowSize.X <= 0 || windowSize.Y <= 0 || framebufferSize.X <= 0 || framebufferSize.Y <= 0)
            return;

        bool windowSizeChanged = !windowSize.Equals(_lastSyncedImGuiWindowSize);
        bool framebufferSizeChanged = !framebufferSize.Equals(_lastSyncedImGuiFramebufferSize);
        if (!windowSizeChanged && !framebufferSizeChanged)
            return;

        if (windowSizeChanged)
            ImGuiControllerWindowResizedMethod?.Invoke(_host.ImGui, new object[] { windowSize });

        ImGuiIOPtr io = ImGui.GetIO();
        io.DisplaySize = new Vector2(windowSize.X, windowSize.Y);
        io.DisplayFramebufferScale = new Vector2(
            windowSize.X > 0 ? (float)framebufferSize.X / windowSize.X : 1f,
            windowSize.Y > 0 ? (float)framebufferSize.Y / windowSize.Y : 1f);

        _lastSyncedImGuiWindowSize = windowSize;
        _lastSyncedImGuiFramebufferSize = framebufferSize;
    }
}
