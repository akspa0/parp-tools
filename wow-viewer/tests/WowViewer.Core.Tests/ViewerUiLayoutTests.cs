using System.Numerics;
using WoWViewer;

namespace WowViewer.Core.Tests;

public sealed class ViewerUiLayoutTests
{
    [Fact]
    public void DefaultLayout_UsesNarrowVisibleActionRails()
    {
        var defaults = UiLayout.Defaults(1688f);

        Assert.True(defaults.Left);
        Assert.True(defaults.Right);
    }

    [Fact]
    public void SceneMinimapAndChat_DoNotInheritPersistedSidebarWidths()
    {
        Vector2 display = new(1688f, 1452f);
        float scale = UiLayout.ScaleFor(display);
        var oversized = new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 516f, rightWidth: 720f, scale: scale);
        var reset = new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 280f, rightWidth: 360f, scale: scale);

        Assert.Equal(display.X, oversized[UiRegion.Scene].Rect.Width);
        Assert.InRange(oversized[UiRegion.LeftSidebar].Rect.Width, 0f, 44f * scale + 0.1f);
        Assert.InRange(oversized[UiRegion.RightSidebar].Rect.Width, 0f, 44f * scale + 0.1f);
        Assert.Equal(reset[UiRegion.Chat].Rect, oversized[UiRegion.Chat].Rect);
        Assert.Equal(reset[UiRegion.Minimap].Rect, oversized[UiRegion.Minimap].Rect);
        Assert.True(oversized[UiRegion.Chat].Rect.X < display.X * 0.1f);
        Assert.True(oversized[UiRegion.Chat].Rect.Width >= 400f);
        Assert.True(oversized[UiRegion.Minimap].Rect.X > display.X * 0.7f);
        Assert.True(oversized[UiRegion.Minimap].Rect.Y < display.Y * 0.1f);
    }

    [Theory]
    [InlineData(800, 600)]
    [InlineData(1280, 720)]
    [InlineData(1688, 1452)]
    [InlineData(1920, 1080)]
    [InlineData(2560, 1440)]
    [InlineData(3840, 2160)]
    public void ToolWindow_OpensInsideTheClientArea(int width, int height)
    {
        var display = new Vector2(width, height);
        float scale = UiLayout.ScaleFor(display);
        UiRect rect = UiLayout.ToolWindowRect(display, new Vector2(width - 40, height - 30), new(520, 620), scale);

        Assert.True(rect.Width > 0 && rect.Height > 0);
        Assert.True(rect.X >= 0 && rect.Y >= 0);
        Assert.True(rect.X + rect.Width <= width);
        Assert.True(rect.Y + rect.Height <= height);
        Assert.True(rect.Width <= width * 0.62f);
        Assert.True(rect.Height <= height * 0.78f);
    }
}
