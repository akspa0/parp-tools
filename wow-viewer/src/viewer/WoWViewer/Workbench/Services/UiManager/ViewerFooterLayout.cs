using System.Numerics;

namespace WoWViewer;

/// <summary>
/// Spec 286 U0: pure footer fit policy.
/// <para>
/// The bottom action bar and the status bar are drawn into fixed client regions. Their content
/// used to be laid out with display-global coordinates and a horizontal scrollbar, so long labels
/// and wide counters pushed essential controls off-screen. This policy measures the real region,
/// keeps every essential item reachable, compacts before it hides anything, and bounds the hover
/// disclosure. It touches no ImGui state, so it is source-linked into the headless test project.
/// </para>
/// </summary>
internal static class ViewerFooterLayout
{
    /// <summary>Horizontal padding kept clear inside a footer region.</summary>
    public const float RegionPadding = 8f;

    /// <summary>Gap between adjacent footer items at scale 1.</summary>
    public const float ItemGap = 10f;

    /// <summary>Floor for a compacted item so it stays clickable.</summary>
    public const float MinItemWidth = 24f;

    /// <summary>Maximum width of a hover disclosure at scale 1.</summary>
    public const float TooltipMaxWidth = 420f;

    /// <summary>Maximum height of a hover disclosure at scale 1.</summary>
    public const float TooltipMaxHeight = 320f;

    /// <summary>Usable content width inside a footer region.</summary>
    public static float AvailableWidth(UiRect region, float scale)
    {
        float s = MathF.Max(1f, scale);
        return MathF.Max(0f, region.Width - 2f * RegionPadding * s);
    }

    /// <summary>
    /// Fit a row of footer items into <paramref name="region"/>.
    /// <para>
    /// Order of preference: natural widths, compact secondary items, compact everything, hide
    /// secondary items from the end, then shrink the remaining essentials proportionally. Essential
    /// items are never hidden, so a reachable action never disappears behind a scrollbar.
    /// </para>
    /// </summary>
    public static FooterFitResult Fit(
        UiRect region,
        float scale,
        IReadOnlyList<FooterItemMeasure> items,
        float gap = ItemGap)
    {
        float available = AvailableWidth(region, scale);
        float effectiveGap = MathF.Max(0f, gap) * MathF.Max(1f, scale);

        if (items.Count == 0)
            return new FooterFitResult(Array.Empty<FooterItemPlacement>(), 0f, false, 0);

        if (TotalWidth(items, effectiveGap, compact: false) <= available)
            return Place(items, effectiveGap, compact: false);

        if (TotalWidth(items, effectiveGap, compact: true, compactOnlySecondary: true) <= available)
            return Place(items, effectiveGap, compact: true, compactOnlySecondary: true);

        if (TotalWidth(items, effectiveGap, compact: true) <= available)
            return Place(items, effectiveGap, compact: true);

        var hidden = new HashSet<string>(StringComparer.Ordinal);
        for (int i = items.Count - 1; i >= 0; i--)
        {
            if (items[i].Priority != FooterItemPriority.Secondary) continue;
            hidden.Add(items[i].Id);
            if (TotalWidth(items, effectiveGap, compact: true, hidden: hidden) <= available)
                return Place(items, effectiveGap, compact: true, hidden: hidden);
        }

        return PlaceShrunk(items, effectiveGap, available, hidden);
    }

    /// <summary>
    /// Right-align a status readout inside the actual content width. When the left readout and the
    /// right readout cannot coexist, the right readout is compacted to the remaining space instead
    /// of being positioned with the display-global width and clipped off-screen.
    /// </summary>
    public static FooterStatusFit FitStatus(
        float contentWidth,
        float scale,
        float leftEnd,
        float rightWidth,
        float gap = ItemGap)
    {
        float width = MathF.Max(0f, contentWidth);
        float effectiveGap = MathF.Max(0f, gap) * MathF.Max(1f, scale);
        float right = MathF.Max(0f, rightWidth);
        float left = MathF.Max(0f, leftEnd);

        float rightX = MathF.Max(left + effectiveGap, width - right);
        if (rightX + right <= width)
            return new FooterStatusFit(rightX, right, true, false);

        float remaining = MathF.Max(0f, width - left - effectiveGap);
        if (remaining < MinItemWidth)
            return new FooterStatusFit(width, 0f, false, true);

        return new FooterStatusFit(width - remaining, remaining, true, true);
    }

    /// <summary>Clamp a hover disclosure rectangle inside the client area.</summary>
    public static UiRect DisclosureRect(Vector2 display, Vector2 anchor, Vector2 desiredSize, float scale)
    {
        float s = MathF.Max(1f, scale);
        float width = MathF.Min(MathF.Max(0f, desiredSize.X), MathF.Min(TooltipMaxWidth * s, MathF.Max(0f, display.X)));
        float height = MathF.Min(MathF.Max(0f, desiredSize.Y), MathF.Min(TooltipMaxHeight * s, MathF.Max(0f, display.Y)));
        float x = Math.Clamp(anchor.X, 0f, MathF.Max(0f, display.X - width));
        float y = Math.Clamp(anchor.Y, 0f, MathF.Max(0f, display.Y - height));
        return new UiRect(x, y, width, height);
    }

    private static float TotalWidth(
        IReadOnlyList<FooterItemMeasure> items,
        float gap,
        bool compact,
        bool compactOnlySecondary = false,
        HashSet<string>? hidden = null)
    {
        float total = 0f;
        int count = 0;
        foreach (FooterItemMeasure item in items)
        {
            if (hidden != null && hidden.Contains(item.Id)) continue;
            total += WidthOf(item, compact, compactOnlySecondary);
            count++;
        }

        return count == 0 ? 0f : total + gap * (count - 1);
    }

    private static FooterFitResult Place(
        IReadOnlyList<FooterItemMeasure> items,
        float gap,
        bool compact,
        bool compactOnlySecondary = false,
        HashSet<string>? hidden = null)
    {
        var placements = new List<FooterItemPlacement>(items.Count);
        float x = 0f;
        int hiddenCount = 0;
        foreach (FooterItemMeasure item in items)
        {
            if (hidden != null && hidden.Contains(item.Id))
            {
                placements.Add(new FooterItemPlacement(item.Id, x, 0f, false, false));
                hiddenCount++;
                continue;
            }

            bool isCompact = compact && (compactOnlySecondary ? item.Priority == FooterItemPriority.Secondary : true);
            float width = WidthOf(item, compact, compactOnlySecondary);
            placements.Add(new FooterItemPlacement(item.Id, x, width, true, isCompact));
            x += width + gap;
        }

        float content = ContentWidth(placements, gap);
        return new FooterFitResult(placements, content, compact, hiddenCount);
    }

    private static FooterFitResult PlaceShrunk(
        IReadOnlyList<FooterItemMeasure> items,
        float gap,
        float available,
        HashSet<string> hidden)
    {
        var visible = new List<FooterItemMeasure>(items.Count);
        foreach (FooterItemMeasure item in items)
        {
            if (!hidden.Contains(item.Id)) visible.Add(item);
        }

        if (visible.Count == 0)
            return new FooterFitResult(Array.Empty<FooterItemPlacement>(), 0f, true, items.Count);

        float gaps = gap * MathF.Max(0, visible.Count - 1);
        float budget = MathF.Max(0f, available - gaps);
        float natural = 0f;
        foreach (FooterItemMeasure item in visible)
            natural += MathF.Max(MinItemWidth, item.CompactWidth);

        float factor = natural > 0f ? MathF.Min(1f, budget / natural) : 1f;
        var placements = new List<FooterItemPlacement>(visible.Count);
        float x = 0f;
        foreach (FooterItemMeasure item in visible)
        {
            float width = MathF.Max(1f, MathF.Max(MinItemWidth, item.CompactWidth) * factor);
            placements.Add(new FooterItemPlacement(item.Id, x, width, true, true));
            x += width + gap;
        }

        return new FooterFitResult(placements, ContentWidth(placements, gap), true, items.Count - visible.Count);
    }

    private static float WidthOf(FooterItemMeasure item, bool compact, bool compactOnlySecondary)
    {
        bool isCompact = compact && (compactOnlySecondary ? item.Priority == FooterItemPriority.Secondary : true);
        return isCompact
            ? MathF.Max(MinItemWidth, item.CompactWidth)
            : MathF.Max(MinItemWidth, item.NaturalWidth);
    }

    private static float ContentWidth(IReadOnlyList<FooterItemPlacement> placements, float gap)
    {
        float total = 0f;
        int count = 0;
        foreach (FooterItemPlacement placement in placements)
        {
            if (!placement.Visible) continue;
            total += placement.Width;
            count++;
        }

        return count == 0 ? 0f : total + gap * (count - 1);
    }
}

/// <summary>How important a footer item is when space runs out.</summary>
internal enum FooterItemPriority
{
    /// <summary>Never hidden; compacted or shrunk instead.</summary>
    Essential = 0,

    /// <summary>Compacted first, then hidden behind the disclosure.</summary>
    Secondary = 1,
}

/// <summary>Measured widths for one footer item.</summary>
internal readonly record struct FooterItemMeasure(
    string Id,
    float NaturalWidth,
    float CompactWidth,
    FooterItemPriority Priority = FooterItemPriority.Secondary);

/// <summary>Where one footer item landed inside its region.</summary>
internal readonly record struct FooterItemPlacement(
    string Id,
    float X,
    float Width,
    bool Visible,
    bool Compact);

/// <summary>Result of fitting a footer row.</summary>
internal readonly record struct FooterFitResult(
    IReadOnlyList<FooterItemPlacement> Placements,
    float ContentWidth,
    bool Compacted,
    int HiddenCount)
{
    /// <summary>True when at least one item could not be shown.</summary>
    public bool Overflowed => HiddenCount > 0;

    /// <summary>Placement for <paramref name="id"/>, or a hidden zero-width placement.</summary>
    public FooterItemPlacement Get(string id)
    {
        foreach (FooterItemPlacement placement in Placements)
        {
            if (string.Equals(placement.Id, id, StringComparison.Ordinal)) return placement;
        }

        return new FooterItemPlacement(id, 0f, 0f, false, false);
    }
}

/// <summary>Result of fitting the status readouts.</summary>
internal readonly record struct FooterStatusFit(
    float RightX,
    float RightWidth,
    bool RightVisible,
    bool Compacted);
