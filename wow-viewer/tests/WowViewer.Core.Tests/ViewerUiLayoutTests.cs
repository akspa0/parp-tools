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
    [InlineData(1920, 1080)]
    [InlineData(2560, 1440)]
    [InlineData(3840, 2160)]
    public void FooterRegions_AreNonEmptyAndInsideTheClient(int width, int height)
    {
        var display = new Vector2(width, height);
        float scale = UiLayout.ScaleFor(display);
        var layout = new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 280f, rightWidth: 360f, scale: scale);

        UiRect bottom = layout[UiRegion.BottomBar].Rect;
        UiRect status = layout[UiRegion.StatusBar].Rect;

        Assert.True(bottom.Width > 0f && bottom.Height > 0f);
        Assert.True(status.Width > 0f && status.Height > 0f);
        Assert.True(bottom.X >= 0f && bottom.Y >= 0f);
        Assert.True(bottom.X + bottom.Width <= width + 0.5f);
        Assert.True(bottom.Y + bottom.Height <= height + 0.5f);
        Assert.True(status.X + status.Width <= width + 0.5f);
        Assert.True(status.Y + status.Height <= height + 0.5f);
        Assert.True(status.Y >= bottom.Y + bottom.Height - 0.5f);
    }

    // Measured button-row height at a text scale: the base font is 14 px, frame padding 3 px per side.
    private static float RowAt(float displayScale, float userScale) =>
        14f * userScale * displayScale + 2f * 3f * displayScale;

    [Theory]
    [InlineData(1920, 1080, 1.0f)]
    [InlineData(1920, 1080, 2.2f)]
    [InlineData(3440, 1392, 1.0f)]   // the operator's ultrawide run: status text was clipped here
    [InlineData(3440, 1392, 2.2f)]
    [InlineData(3840, 2160, 1.0f)]
    [InlineData(3840, 2160, 2.2f)]
    public void FooterBars_AreNeverShorterThanTheirButtonRow(int width, int height, float userScale)
    {
        var display = new Vector2(width, height);
        float scale = UiLayout.ScaleFor(display);
        float row = RowAt(scale, userScale);
        var layout = new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 280f, rightWidth: 360f, scale: scale, footerRowHeight: row);

        UiRect bottom = layout[UiRegion.BottomBar].Rect;
        UiRect status = layout[UiRegion.StatusBar].Rect;

        // The content a bar holds is one row plus its own vertical window padding top and bottom.
        Assert.True(bottom.Height >= row + 2f * UiLayout.BottomBarPadY * scale - 0.01f);
        Assert.True(status.Height >= row + 2f * UiLayout.StatusBarPadY * scale - 0.01f);
        Assert.True(status.Y + status.Height <= height + 0.5f);
        Assert.True(bottom.Y + bottom.Height <= status.Y + 0.5f);
    }

    [Fact]
    public void FooterBars_GrowWithTextSizeAndShrinkTheSceneInstead()
    {
        var display = new Vector2(1920f, 1080f);
        var normal = new UiLayout(display, false, true, true, 280f, 360f, scale: 1f, footerRowHeight: RowAt(1f, 1.0f));
        var large = new UiLayout(display, false, true, true, 280f, 360f, scale: 1f, footerRowHeight: RowAt(1f, 2.2f));

        Assert.True(large[UiRegion.StatusBar].Rect.Height > normal[UiRegion.StatusBar].Rect.Height);
        Assert.True(large[UiRegion.BottomBar].Rect.Height > normal[UiRegion.BottomBar].Rect.Height);
        Assert.True(large[UiRegion.Scene].Rect.Height < normal[UiRegion.Scene].Rect.Height);
        // The bars plus the scene still tile the client exactly.
        float total = large[UiRegion.TopBar].Rect.Height + large[UiRegion.Scene].Rect.Height
            + large[UiRegion.BottomBar].Rect.Height + large[UiRegion.StatusBar].Rect.Height;
        Assert.Equal(display.Y, total, 1);
    }

    [Fact]
    public void FooterBars_FallBackToTheFloorBeforeTheFontIsMeasured()
    {
        var display = new Vector2(1920f, 1080f);
        var unmeasured = new UiLayout(display, false, true, true, 280f, 360f, scale: 1f);
        var invalid = new UiLayout(display, false, true, true, 280f, 360f, scale: 1f, footerRowHeight: float.NaN);

        Assert.Equal(UiLayout.MinFooterRow + 2f * UiLayout.StatusBarPadY, unmeasured[UiRegion.StatusBar].Rect.Height, 2);
        Assert.Equal(unmeasured[UiRegion.StatusBar].Rect, invalid[UiRegion.StatusBar].Rect);
        Assert.Equal(unmeasured[UiRegion.BottomBar].Rect, invalid[UiRegion.BottomBar].Rect);
    }

    [Theory]
    [InlineData(1920, 1080)]
    [InlineData(2560, 1440)]
    [InlineData(3840, 2160)]
    public void RightDock_SitsBelowMinimapAndAboveBottomBar(int width, int height)
    {
        var display = new Vector2(width, height);
        float scale = UiLayout.ScaleFor(display);
        var layout = new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 280f, rightWidth: 360f, scale: scale,
            rightDockVisible: true, rightDockWidth: 360f);

        UiRect dock = layout[UiRegion.RightDock].Rect;
        UiRect minimap = layout[UiRegion.Minimap].Rect;
        UiRect bottom = layout[UiRegion.BottomBar].Rect;
        UiRect rail = layout[UiRegion.RightSidebar].Rect;

        Assert.True(layout[UiRegion.RightDock].Visible);
        Assert.True(dock.Width > 0f && dock.Height > 0f);
        Assert.True(dock.Y >= minimap.Y + minimap.Height - 0.5f);
        Assert.True(dock.Y + dock.Height <= bottom.Y + 0.5f);
        Assert.True(dock.X + dock.Width <= rail.X + 0.5f);
        Assert.True(dock.X >= 0f);
    }

    [Fact]
    public void RightDock_IsHiddenWhenNotRequested()
    {
        var display = new Vector2(1920f, 1080f);
        var layout = new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 280f, rightWidth: 360f, scale: 1f);

        Assert.False(layout[UiRegion.RightDock].Visible);
    }

    [Theory]
    [InlineData(360f, 360f)]
    [InlineData(100f, 240f)]
    [InlineData(2000f, 864f)]
    public void RightDock_ClampsRequestedWidth(float requested, float expected)
    {
        var display = new Vector2(1920f, 1080f);
        var layout = new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 280f, rightWidth: 360f, scale: 1f,
            rightDockVisible: true, rightDockWidth: requested);

        Assert.Equal(expected, layout[UiRegion.RightDock].Rect.Width, 1);
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
