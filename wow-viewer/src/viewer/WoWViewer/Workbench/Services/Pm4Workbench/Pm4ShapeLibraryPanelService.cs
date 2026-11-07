using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.UI;

namespace WoWViewer.Workbench.Services.Pm4Workbench;

/// <summary>
/// Spec 295 Phase 3 (T011): Visual PM4 shape library panel using <see cref="SharedUiWidgets"/>.
/// Renders 2D footprint and elevation profile previews, relative proportions, associated asset paths,
/// search/filter controls with selection retention, and side-by-side shape comparison.
/// </summary>
internal sealed class Pm4ShapeLibraryPanelService
{
    private string _searchBuffer = string.Empty;
    private int _selectedFamilyIndex;
    private int _selectedAssetKindIndex;

    public void Draw(Pm4ShapeMatchWorkbenchService workbench, WorldScene? worldScene)
    {
        workbench.EnsureLibraryInitialized(worldScene);
        Pm4ShapeLibraryVisualService library = workbench.Library;

        if (!SharedUiWidgets.SectionHeader("Visual PM4 Shape Library", defaultOpen: false, id: "Pm4VisualShapeLibrarySection"))
            return;

        ImGui.TextDisabled(
            $"Entries: {library.FilteredEntries.Count} / {library.AllEntries.Count} (Healthy: {library.HealthyCount}, Unhealthy: {library.UnhealthyCount})");

        if (SharedUiWidgets.ActionButton("Rebuild Shape Library##Pm4ShapeLibRebuild"))
        {
            workbench.EnsureLibraryInitialized(worldScene, forceRebuild: true);
        }

        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Clear Library Filter##Pm4ShapeLibClearFilter"))
        {
            _searchBuffer = string.Empty;
            _selectedFamilyIndex = 0;
            _selectedAssetKindIndex = 0;
            library.ApplyFilter(new Pm4ShapeLibraryFilter());
        }

        bool filterChanged = false;
        if (ImGui.InputText("Search Shapes##Pm4ShapeLibSearch", ref _searchBuffer, 128))
            filterChanged = true;

        var families = new List<string>(library.AvailableFamilies.Count + 1) { "All" };
        families.AddRange(library.AvailableFamilies);
        _selectedFamilyIndex = Math.Clamp(_selectedFamilyIndex, 0, families.Count - 1);
        string[] familyItems = families.ToArray();
        if (ImGui.Combo("Shape Family##Pm4ShapeLibFamily", ref _selectedFamilyIndex, familyItems, familyItems.Length))
            filterChanged = true;

        var kinds = new List<string>(library.AvailableAssetKinds.Count + 1) { "All" };
        kinds.AddRange(library.AvailableAssetKinds);
        _selectedAssetKindIndex = Math.Clamp(_selectedAssetKindIndex, 0, kinds.Count - 1);
        string[] kindItems = kinds.ToArray();
        if (ImGui.Combo("Shape Asset Kind##Pm4ShapeLibKind", ref _selectedAssetKindIndex, kindItems, kindItems.Length))
            filterChanged = true;

        bool includeUnhealthy = library.CurrentFilter.IncludeUnhealthy;
        if (ImGui.Checkbox("Include Unhealthy Entries##Pm4ShapeLibIncludeUnhealthy", ref includeUnhealthy))
            filterChanged = true;

        ImGui.SameLine();
        bool curatedOnly = library.CurrentFilter.CuratedOnly;
        if (ImGui.Checkbox("Curated Shapes Only##Pm4ShapeLibCuratedOnly", ref curatedOnly))
            filterChanged = true;

        if (filterChanged)
        {
            library.ApplyFilter(new Pm4ShapeLibraryFilter(
                SearchText: _searchBuffer,
                FamilyFilter: familyItems[_selectedFamilyIndex],
                AssetKindFilter: kindItems[_selectedAssetKindIndex],
                IncludeUnhealthy: includeUnhealthy,
                CuratedOnly: curatedOnly));
        }

        ImGui.Spacing();
        if (ImGui.BeginChild("Pm4ShapeLibraryCatalogList", new Vector2(0f, 190f), true))
        {
            int maxVisible = Math.Min(library.FilteredEntries.Count, 250);
            for (int i = 0; i < maxVisible; i++)
            {
                Pm4ShapeLibraryVisualEntry entry = library.FilteredEntries[i];
                bool isSelected = string.Equals(entry.EntryKey, library.SelectedEntryKey, StringComparison.OrdinalIgnoreCase);
                string healthBadge = entry.IsHealthy ? string.Empty : " [UNHEALTHY]";
                string curatedBadge = entry.IsCurated ? "★ " : string.Empty;
                string rowLabel = $"{curatedBadge}{entry.DisplayName} [{entry.AssetKind} · {entry.Family}] — {entry.Preview.ProportionSummary}{healthBadge}##lib_row_{i}";

                if (ImGui.Selectable(rowLabel, isSelected))
                {
                    library.SelectEntry(entry.EntryKey);
                }
            }
        }
        ImGui.EndChild();

        if (library.SelectedEntry is { } selected)
        {
            ImGui.Spacing();
            DrawEntryInspectorCard(selected, "Selected Library Shape");

            if (SharedUiWidgets.ActionButton("Pin Selected for Compare##Pm4ShapeLibPinCompare"))
            {
                library.SetComparisonEntry(selected.EntryKey);
            }

            if (library.ComparisonEntry != null)
            {
                ImGui.SameLine();
                if (SharedUiWidgets.ActionButton("Clear Shape Comparison##Pm4ShapeLibClearCompare"))
                {
                    library.SetComparisonEntry(null);
                }
            }
        }

        if (library.GetActiveComparison() is { } comparison)
        {
            ImGui.Spacing();
            ImGui.Separator();
            ImGui.TextColored(UiTokens.Brass, "Side-by-Side Shape Comparison");
            ImGui.TextWrapped(comparison.Summary);
            DrawSideBySideComparisonCards(comparison);
        }
    }

    public static void DrawEntryInspectorCard(Pm4ShapeLibraryVisualEntry entry, string heading)
    {
        ImGui.TextColored(UiTokens.Brass, $"{heading}: {entry.DisplayName} ({entry.AssetKind})");
        ImGui.TextDisabled($"Family: {entry.Family} | Proportions: {entry.Preview.ProportionSummary}");
        if (!entry.IsHealthy)
        {
            ImGui.TextColored(UiTokens.Danger, $"Health: Unhealthy ({entry.HealthReason ?? "missing or unreadable"}) — excluded from automatic proposals");
        }

        DrawDualPreviewBox($"##preview_{entry.EntryKey}", entry.Preview, new Vector2(220f, 84f));

        ImGui.TextDisabled($"Primary Asset: {entry.PrimaryModelPath}");
        if (entry.AssociatedAssetPaths.Count > 1)
        {
            ImGui.TextDisabled($"Shared by {entry.AssociatedAssetPaths.Count} assets:");
            for (int i = 0; i < Math.Min(entry.AssociatedAssetPaths.Count, 4); i++)
            {
                ImGui.BulletText(entry.AssociatedAssetPaths[i]);
            }
        }
    }

    private static void DrawSideBySideComparisonCards(Pm4ShapeLibraryComparison comparison)
    {
        if (ImGui.BeginTable("Pm4ShapeCompareTable", 2, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.SizingStretchSame))
        {
            ImGui.TableNextRow();
            ImGui.TableSetColumnIndex(0);
            ImGui.TextColored(UiTokens.Success, $"Left: {comparison.Left.DisplayName}");
            ImGui.TextDisabled(comparison.Left.Preview.ProportionSummary);
            DrawDualPreviewBox("##cmp_left", comparison.Left.Preview, new Vector2(160f, 72f));

            ImGui.TableSetColumnIndex(1);
            ImGui.TextColored(UiTokens.Brass, $"Right: {comparison.Right.DisplayName}");
            ImGui.TextDisabled(comparison.Right.Preview.ProportionSummary);
            DrawDualPreviewBox("##cmp_right", comparison.Right.Preview, new Vector2(160f, 72f));

            ImGui.EndTable();
        }
    }

    public static void DrawDualPreviewBox(string id, Pm4ShapeVisualPreview preview, Vector2 size)
    {
        Vector2 cursorScreen = ImGui.GetCursorScreenPos();
        ImGui.InvisibleButton(id, size);
        ImDrawListPtr drawList = ImGui.GetWindowDrawList();

        uint bgCol = ImGui.GetColorU32(new Vector4(0.08f, 0.09f, 0.11f, 0.92f));
        uint borderCol = ImGui.GetColorU32(new Vector4(0.32f, 0.35f, 0.42f, 0.85f));
        uint footStroke = ImGui.GetColorU32(new Vector4(0.28f, 0.78f, 0.95f, 0.95f));
        uint elevStroke = ImGui.GetColorU32(new Vector4(0.92f, 0.72f, 0.28f, 0.95f));

        drawList.AddRectFilled(cursorScreen, cursorScreen + size, bgCol, 4f);
        drawList.AddRect(cursorScreen, cursorScreen + size, borderCol, 4f);

        float halfWidth = size.X * 0.5f;
        Vector2 midTop = new(cursorScreen.X + halfWidth, cursorScreen.Y + 4f);
        Vector2 midBottom = new(cursorScreen.X + halfWidth, cursorScreen.Y + size.Y - 4f);
        drawList.AddLine(midTop, midBottom, borderCol, 1f);

        Vector2 leftCenter = new(cursorScreen.X + halfWidth * 0.5f, cursorScreen.Y + size.Y * 0.5f);
        Vector2 rightCenter = new(cursorScreen.X + halfWidth * 1.5f, cursorScreen.Y + size.Y * 0.5f);
        float radius = MathF.Min(halfWidth, size.Y) * 0.36f;

        DrawNormalizedPolygon(drawList, preview.NormalizedFootprintPolygon, leftCenter, radius, footStroke);
        DrawNormalizedPolygon(drawList, preview.NormalizedElevationProfile, rightCenter, radius, elevStroke);
    }

    private static void DrawNormalizedPolygon(
        ImDrawListPtr drawList,
        IReadOnlyList<Vector2> points,
        Vector2 center,
        float scaleRadius,
        uint strokeColor)
    {
        if (points == null || points.Count < 2)
            return;

        for (int i = 0; i < points.Count; i++)
        {
            Vector2 a = points[i];
            Vector2 b = points[(i + 1) % points.Count];
            Vector2 pa = new(center.X + a.X * scaleRadius, center.Y - a.Y * scaleRadius);
            Vector2 pb = new(center.X + b.X * scaleRadius, center.Y - b.Y * scaleRadius);
            drawList.AddLine(pa, pb, strokeColor, 1.75f);
        }
    }
}

