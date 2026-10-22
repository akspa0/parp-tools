using System.Numerics;
using WoWViewer;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 286 U0 regressions: the footer must fit the actual client region, keep essential
/// controls reachable, and disclose detail on hover instead of hiding it behind a scrollbar.
/// </summary>
public sealed class ViewerFooterLayoutTests
{
    private static UiLayout LayoutFor(int width, int height, float? scale = null)
    {
        var display = new Vector2(width, height);
        float effective = scale ?? UiLayout.ScaleFor(display);
        return new UiLayout(display, hidden: false, left: true, right: true,
            leftWidth: 280f, rightWidth: 360f, scale: effective);
    }

    private static List<FooterItemMeasure> BottomItems(float scale) => new()
    {
        new FooterItemMeasure("preset-Viewer", 82f * scale, 64f * scale, FooterItemPriority.Essential),
        new FooterItemMeasure("preset-Editor", 82f * scale, 64f * scale, FooterItemPriority.Essential),
        new FooterItemMeasure("preset-Archaeology", 120f * scale, 64f * scale, FooterItemPriority.Essential),
        new FooterItemMeasure("bottom-spawns", 78f * scale, 56f * scale, FooterItemPriority.Essential),
        new FooterItemMeasure("bottom-pages", 78f * scale, 56f * scale, FooterItemPriority.Essential),
        new FooterItemMeasure("bottom-options", 76f * scale, 56f * scale, FooterItemPriority.Essential),
    };

    [Theory]
    [InlineData(1920, 1080)]
    [InlineData(2560, 1440)]
    [InlineData(3840, 2160)]
    public void BottomBar_FitsInsideItsRegionAtCommonResolutions(int width, int height)
    {
        UiLayout layout = LayoutFor(width, height);
        UiRect region = layout[UiRegion.BottomBar].Rect;
        float scale = layout.Scale;
        float available = ViewerFooterLayout.AvailableWidth(region, scale);

        FooterFitResult fit = ViewerFooterLayout.Fit(region, scale, BottomItems(scale));

        Assert.True(fit.ContentWidth <= available + 0.5f);
        foreach (FooterItemPlacement placement in fit.Placements)
        {
            Assert.True(placement.Visible);
            Assert.True(placement.X >= 0f);
            Assert.True(placement.X + placement.Width <= available + 0.5f);
        }
    }

    [Theory]
    [InlineData(1920, 1080, 1.0f)]
    [InlineData(1920, 1080, 1.5f)]
    [InlineData(2560, 1440, 2.0f)]
    [InlineData(3840, 2160, 2.5f)]
    public void BottomBar_FitsAtDefaultAndCustomEffectiveScales(int width, int height, float scale)
    {
        UiLayout layout = LayoutFor(width, height, scale);
        UiRect region = layout[UiRegion.BottomBar].Rect;
        float available = ViewerFooterLayout.AvailableWidth(region, scale);

        FooterFitResult fit = ViewerFooterLayout.Fit(region, scale, BottomItems(scale));

        Assert.True(fit.ContentWidth <= available + 0.5f);
        Assert.All(fit.Placements, placement => Assert.True(placement.Visible));
    }

    [Fact]
    public void NarrowRegion_CompactsBeforeHidingAndKeepsEssentials()
    {
        var region = new UiRect(0f, 0f, 320f, 52f);
        var items = new List<FooterItemMeasure>
        {
            new("essential-a", 200f, 80f, FooterItemPriority.Essential),
            new("essential-b", 200f, 80f, FooterItemPriority.Essential),
            new("secondary-a", 200f, 80f, FooterItemPriority.Secondary),
        };

        FooterFitResult fit = ViewerFooterLayout.Fit(region, 1f, items);

        Assert.True(fit.Get("essential-a").Visible);
        Assert.True(fit.Get("essential-b").Visible);
        Assert.True(fit.Compacted);
        Assert.True(fit.ContentWidth <= ViewerFooterLayout.AvailableWidth(region, 1f) + 0.5f);
    }

    [Fact]
    public void SecondaryItems_AreHiddenBeforeEssentials()
    {
        var region = new UiRect(0f, 0f, 200f, 52f);
        var items = new List<FooterItemMeasure>
        {
            new("essential-a", 120f, 90f, FooterItemPriority.Essential),
            new("essential-b", 120f, 90f, FooterItemPriority.Essential),
            new("secondary-a", 120f, 90f, FooterItemPriority.Secondary),
        };

        FooterFitResult fit = ViewerFooterLayout.Fit(region, 1f, items);

        Assert.True(fit.Get("essential-a").Visible);
        Assert.True(fit.Get("essential-b").Visible);
        Assert.False(fit.Get("secondary-a").Visible);
        Assert.True(fit.Overflowed);
        Assert.True(fit.ContentWidth <= ViewerFooterLayout.AvailableWidth(region, 1f) + 0.5f);
    }

    [Fact]
    public void StatusFit_RightAlignsInsideContentWidth()
    {
        FooterStatusFit fit = ViewerFooterLayout.FitStatus(contentWidth: 1000f, scale: 1f, leftEnd: 200f, rightWidth: 300f);

        Assert.True(fit.RightVisible);
        Assert.False(fit.Compacted);
        Assert.Equal(700f, fit.RightX, 3);
        Assert.Equal(300f, fit.RightWidth, 3);
    }

    [Fact]
    public void StatusFit_CompactsWhenLeftAndRightCollide()
    {
        FooterStatusFit fit = ViewerFooterLayout.FitStatus(contentWidth: 500f, scale: 1f, leftEnd: 300f, rightWidth: 400f);

        Assert.True(fit.RightVisible);
        Assert.True(fit.Compacted);
        Assert.True(fit.RightWidth < 400f);
        Assert.True(fit.RightX + fit.RightWidth <= 500f + 0.5f);
    }

    [Fact]
    public void StatusFit_HidesRightReadoutWhenNoRoomRemains()
    {
        FooterStatusFit fit = ViewerFooterLayout.FitStatus(contentWidth: 200f, scale: 1f, leftEnd: 190f, rightWidth: 100f);

        Assert.False(fit.RightVisible);
        Assert.True(fit.Compacted);
    }

    [Theory]
    [InlineData(1920, 1080)]
    [InlineData(2560, 1440)]
    [InlineData(3840, 2160)]
    public void DisclosureRect_StaysInsideTheClientArea(int width, int height)
    {
        var display = new Vector2(width, height);
        float scale = UiLayout.ScaleFor(display);

        UiRect rect = ViewerFooterLayout.DisclosureRect(
            display,
            new Vector2(width - 4f, height - 4f),
            new Vector2(2000f, 2000f),
            scale);

        Assert.True(rect.X >= 0f && rect.Y >= 0f);
        Assert.True(rect.X + rect.Width <= width + 0.5f);
        Assert.True(rect.Y + rect.Height <= height + 0.5f);
        Assert.True(rect.Width <= ViewerFooterLayout.TooltipMaxWidth * scale + 0.5f);
        Assert.True(rect.Height <= ViewerFooterLayout.TooltipMaxHeight * scale + 0.5f);
    }

    [Fact]
    public void Snapshot_CompactLineKeepsFpsAreaAndPending()
    {
        var snapshot = new FooterDiagnosticsSnapshot(
            HasRuntimeScene: true,
            Fps: 59.6,
            Area: "Area: Elwynn Forest",
            CpuMs: 4.2,
            LoadedTiles: 9,
            ChunksRendered: 120,
            ChunksCulled: 30,
            VisibleWmo: 4,
            TotalWmo: 6,
            VisibleMdx: 12,
            TotalMdx: 20,
            MdxLoaded: 18,
            MdxFailed: 2,
            ActiveAudio: 3,
            ResidentAudio: 5,
            PendingAssetLoads: 7,
            Coordinates: "Local: (1, 2, 3)  WoW: (4, 5, 6)",
            Tile: "Tile: 32_48",
            Facing: "Facing: 90.0° East",
            StatusMessage: string.Empty);

        Assert.Equal("FPS 60", snapshot.FpsText);
        Assert.Contains("Area: Elwynn Forest", snapshot.CompactSuffix);
        Assert.Contains("Pending 7", snapshot.CompactSuffix);
        Assert.DoesNotContain("CPU", snapshot.CompactSuffix);
        Assert.Contains(snapshot.DetailLines, line => line.Contains("CPU: 4.2 ms"));
        Assert.Contains(snapshot.DetailLines, line => line.Contains("Pending asset loads: 7"));
        Assert.Contains(snapshot.DetailLines, line => line.Contains("MDX: 12 visible / 20 total"));
    }

    [Fact]
    public void Snapshot_WithoutSceneKeepsFpsReachableAndShowsStatusMessage()
    {
        FooterDiagnosticsSnapshot snapshot = FooterDiagnosticsSnapshot.Empty with
        {
            Fps = 0d,
            StatusMessage = "Audio output muted.",
        };

        Assert.Equal("FPS --", snapshot.FpsText);
        Assert.Contains("Audio output muted.", snapshot.CompactSuffix);
        Assert.Empty(snapshot.LeftText);
    }
}
