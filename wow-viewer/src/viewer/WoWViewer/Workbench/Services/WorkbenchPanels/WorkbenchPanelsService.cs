using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;
using WoWViewer.Workbench;
using static WoWViewer.ViewerApp;
using static WoWViewer.ImGuiListLayout;
namespace WoWViewer;
internal sealed partial class WorkbenchPanelsService
{
    private readonly IViewerAppHost _host;
    internal WorkbenchPanelsService(IViewerAppHost host) => _host = host;
    internal void DrawDestinationContent(WorkbenchTab destination)
    {
        if (destination == WorkbenchTab.Editor) { EnsureEditorHost(); EnsureEditorPages().Draw(_activeBottomTabIndex); }
        else if (destination == WorkbenchTab.Archaeology) _archaeologyPanel.DrawArchaeologyWorkbenchSubTabContent();
    }
    internal void DrawViewerContent(ViewerPanel panel)
    {
        var ui = _host.UiManager;
        switch (panel)
        {
            case ViewerPanel.Inspector:
                int inspector = DrawPageCombo("##InspectorPage", ["Context", "Scene Investigation", "MCNK Flags", "World Context"], ui.InspectorPage);
                if (inspector != ui.InspectorPage) ui.OpenViewer(panel, inspector);
                switch (inspector)
                {
                    case 0: DrawUnifiedInspectorContent(); break;
                    case 1: _investigation.DrawVisualInvestigationToolbox(showWorldObjectRangeControls: _worldScene != null); break;
                    case 2: _investigation.DrawMcnkFlagOverlayControls(); break;
                    case 3: DrawInspectorWorldContextContent(); break;
                }
                break;
            case ViewerPanel.World:
                int world = DrawPageCombo("##WorldPage", ["Objects", "Placements", "LOD & Budget", "Exports"], ui.WorldPage);
                if (world != ui.WorldPage) ui.OpenViewer(panel, world);
                if (_worldScene == null)
                {
                    SharedUiWidgets.DrawEmptyState("No world loaded", "Load a world to inspect its objects, placements, LOD state and exports.");
                    break;
                }
                switch (world)
                {
                    case 0: _worldObjectsPanel.DrawWorldObjectsPanelContent(); break;
                    case 1: _worldObjectsPanel.DrawWorldPlacementsSubTab(); break;
                    case 2: DrawWorldLodSubTab(); break;
                    case 3: _mainMenuBar.DrawWorldExportContent(); break;
                }
                break;
            case ViewerPanel.Model:
                int model = DrawPageCombo("##ModelPage", ["Info", "Animations", "Actions & Exports"], ui.ModelPage);
                if (model != ui.ModelPage) ui.OpenViewer(panel, model);
                switch (model)
                {
                    case 0: _modelInspector.DrawModelInfoPanelContent(); break;
                    case 1: _modelInspector.DrawModelAnimationsSubTab(); break;
                    case 2: _modelInspector.DrawModelActionsSubTab(); _mainMenuBar.DrawModelExportContent(); break;
                }
                break;
            case ViewerPanel.PM4:
                if (_worldScene != null) _pm4Workbench.DrawPm4WorkbenchInspector();
                else SharedUiWidgets.DrawEmptyState("No world loaded", "Load a world to inspect its PM4 navigation surface overlay.");
                break;
        }
    }
    internal void DrawConvertersSubTabContent() => _mainMenuBar.DrawConvertersContent();
    internal void DrawCameraContent() => DrawCameraControlsContent();
    internal void DrawSceneToolsContent()
    {
        var ui = _host.UiManager;
        int page = DrawPageCombo("##SceneToolsPage", ["Taxi", "Capture"], ui.SceneToolsPage);
        if (page != ui.SceneToolsPage) ui.OpenSceneTools(page);
        if (page == 0)
        {
            if (_worldScene != null) _taxiPanel.DrawTaxiContent();
            else SharedUiWidgets.DrawEmptyState("No world loaded", "Load a world to use the taxi route tools.");
        }
        else DrawCapturePanelContent();
    }
    internal void DrawAssetsContent()
    {
        if (_catalogView == null)
        {
            _catalogView = new Catalog.AssetCatalogView(_gl);
            _catalogView.SetDataSource(_dataSource);
            _catalogView.OnLoadModelRequested = _modelLoader.OnCatalogLoadModel;
        }
        _catalogView.DrawContent();
    }
    internal void DrawUnifiedInspectorContent()
    {
        var content = _inspectorPayloads.BuildInspectorContent();
        if (content.HasContent)
        {
            InspectorContentHost.Draw(content, _inspectorPayloads.HandleInspectorAction);
        }
        else
        {
            SharedUiWidgets.DrawEmptyState(
                "Nothing selected",
                "Move the camera over a loaded ADT/MCNK, or select a model, world object, or PM4 surface.");
        }

        if (_worldScene == null && (_renderer is IModelRenderer || _renderer is WmoRenderer))
        {
            ImGui.Separator();
            _modelInspector.DrawModelInfoContent();
        }
    }

    private void DrawInspectorWorldContextContent()
    {
        if (_worldScene == null)
            return;

        bool objectFogEnabled = _worldScene.ObjectFogEnabled;
        if (ImGui.Checkbox("Fog Objects", ref objectFogEnabled))
            _worldScene.ObjectFogEnabled = objectFogEnabled;

        bool showHoverTooltips = _worldScene.HoverPick.ShowHoveredAssetTooltips;
        if (ImGui.Checkbox("Hover Tooltips", ref showHoverTooltips))
            _worldScene.HoverPick.ShowHoveredAssetTooltips = showHoverTooltips;

        bool limitHoverPickRange = _worldScene.HoverPick.LimitHoveredAssetRange;
        if (ImGui.Checkbox("Limit Hover/Pick Range", ref limitHoverPickRange))
            _worldScene.HoverPick.LimitHoveredAssetRange = limitHoverPickRange;

        if (_worldScene.HoverPick.LimitHoveredAssetRange)
        {
            bool useDynamicHoverRange = _worldScene.HoverPick.UseDynamicHoveredAssetRange;
            if (ImGui.Checkbox("Dynamic Hover Range", ref useDynamicHoverRange))
                _worldScene.HoverPick.UseDynamicHoveredAssetRange = useDynamicHoverRange;

            float hoverPickRange = _worldScene.HoverPick.HoveredAssetMaxDistance;
            if (ImGui.SliderFloat("Hover/Pick Range", ref hoverPickRange, 100f, MaxTerrainFogDistance, "%.2f yd"))
                _worldScene.HoverPick.HoveredAssetMaxDistance = hoverPickRange;

            ImGui.TextDisabled($"Effective range: {_worldScene.HoverPick.EffectiveHoveredAssetMaxDistance:F2} yd");
        }

        bool showSelectedObjectBounds = _worldScene.ShowSelectedObjectBounds;
        if (ImGui.Checkbox("Show Selected Object Bounds", ref showSelectedObjectBounds))
            _worldScene.ShowSelectedObjectBounds = showSelectedObjectBounds;

        _worldObjectsPanel.DrawObjectPathFilterControls();

        ImGui.Separator();
        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0)
        {
            bool showPoi = _worldScene.ShowPoi;
            if (ImGui.Checkbox($"Area POIs ({_worldScene.PoiLoader.Entries.Count})", ref showPoi))
                _worldScene.ShowPoi = showPoi;
        }
        else if (!_worldScene.PoiLoadAttempted)
        {
            if (SharedUiWidgets.ActionButton("Load Area POIs"))
                _worldScene.ShowPoi = true;
        }
        else
        {
            ImGui.TextDisabled("Area POIs: none found");
        }

        if (_worldScene.PoiLoader != null && _worldScene.PoiLoader.Entries.Count > 0
            && ImGui.TreeNode($"Area POI Details ({_worldScene.PoiLoader.Entries.Count})"))
        {
            int poiCount = _worldScene.PoiLoader.Entries.Count;
            if (ImGui.BeginChild("##InspectAreaPoiList", new Vector2(0, 200f), true))
            {
                float rowHeight = GetUniformListRowHeight();
                GetVisibleListRange(poiCount, rowHeight, out int startIndex, out int endIndex);
                if (startIndex > 0)
                    ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

                for (int i = startIndex; i < endIndex; i++)
                {
                    var poi = _worldScene.PoiLoader.Entries[i];
                    bool selected = _selectedAreaPoiId == poi.Id;
                    if (ImGui.Selectable($"[{poi.Id}] {poi.Name}", selected, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        _taxiAndAreaPoi.SelectAreaPoi(poi.Id, toggle: false);
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        {
                            _camera.Position = poi.Position + new Vector3(0, 0, 50);
                            _camera.Pitch = -30f;
                        }
                    }
                }

                if (endIndex < poiCount)
                    ImGui.Dummy(new Vector2(0, (poiCount - endIndex) * rowHeight));
                ImGui.EndChild();
            }
            ImGui.TreePop();
        }

        ImGui.Separator();
        if (_worldScene.AreaTriggerLoader != null && _worldScene.AreaTriggerLoader.Count > 0)
        {
            bool showTriggers = _worldScene.ShowAreaTriggers;
            if (ImGui.Checkbox($"AreaTriggers ({_worldScene.AreaTriggerLoader.Count})", ref showTriggers))
                _worldScene.ShowAreaTriggers = showTriggers;
            if (_worldScene.ShowAreaTriggers && ImGui.IsItemHovered())
                ImGui.SetTooltip("Instance portals, event markers, and script triggers.\nGreen spheres/boxes from AreaTrigger.dbc");
        }
        else if (!_worldScene.AreaTriggerLoadAttempted)
        {
            if (SharedUiWidgets.ActionButton("Load AreaTriggers"))
                _worldScene.ShowAreaTriggers = true;
        }
        else
        {
            ImGui.TextDisabled("AreaTriggers: none found");
        }
    }

    private void DrawCameraControlsContent()
    {
        ImGui.SliderFloat("Camera Speed", ref _cameraSpeed, 1f, 500f, "%.0f");
        ImGui.Text("Hold Shift for 5x boost");
        ImGui.SliderFloat("FOV", ref _fovDegrees, 20f, 90f, "%.0f°");

        if (_terrainManager != null && !_terrainManager.Adapter.IsWmoBased)
        {
            ImGui.Separator();

            bool autoAdtBudget = _terrainManager.DetailedTileCountOverride <= 0;
            int adtDetailTiles = autoAdtBudget
                ? _terrainManager.EffectiveDetailedTileCount
                : _terrainManager.DetailedTileCountOverride;

            if (ImGui.SliderInt("ADT Detail Tiles", ref adtDetailTiles, 1, TerrainManager.MaxManualDetailedTileCount))
            {
                _terrainManager.DetailedTileCountOverride = adtDetailTiles;
                _savedDetailedAdtTileCountOverride = _terrainManager.DetailedTileCountOverride;
            }

            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();

            ImGui.SameLine();
            if (ImGui.SmallButton("Auto"))
            {
                _terrainManager.DetailedTileCountOverride = 0;
                _savedDetailedAdtTileCountOverride = 0;
                _settings.SaveViewerSettings();
            }

            int retainedTileRadius = _terrainManager.RetainedTileRadius;
            if (ImGui.SliderInt("ADT Retain Radius", ref retainedTileRadius, TerrainManager.MinRetainedTileRadius, TerrainManager.MaxRetainedTileRadius))
                _terrainManager.RetainedTileRadius = retainedTileRadius;

            ImGui.TextDisabled(autoAdtBudget
                ? $"Active submission: {_terrainManager.EffectiveDetailedTileCount} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})"
                : $"Active submission: {_terrainManager.DetailedTileCountOverride} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})");
        }
    }

    private bool DrawSelectedObjectSummaryContent()
    {
        bool hasSelectedPm4 = _worldScene?.Pm4Overlay.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo)
            || hasSelectedPm4
            || _selectedObjectType.StartsWith("Taxi", StringComparison.OrdinalIgnoreCase))
            return false;

        ImGui.TextWrapped(_selectedObjectInfo);
        if (_worldObjectsPanel.TryGetSelectedWorldObjectModelPath(out string selectedModelPath, out _))
        {
            ImGui.Separator();
            _navigatorPanel.DrawAssetPathActions("Selected Asset", selectedModelPath, "SelectedWorldObject");
        }

        _modelInspector.DrawSelectedWmoControls();
        _sqlSpawnStreaming.DrawSelectedSqlGameObjectAnimationControls();
        return true;
    }

    internal bool DrawSelectedObjectInspectorSection(bool defaultOpen = true)
    {
        bool hasSelectedPm4 = _worldScene?.Pm4Overlay.HasSelectedPm4Object == true;
        if (string.IsNullOrEmpty(_selectedObjectInfo) || hasSelectedPm4)
            return false;

        if (!SharedUiWidgets.SectionHeader("Selected Object", defaultOpen: defaultOpen, id: "SelectedObject"))
            return true;

        DrawSelectedObjectSummaryContent();
        return true;
    }

    private void DrawWorldLodSubTab()
    {
        ImGui.TextDisabled("WDL visibility, detailed ADT budget, and distance LOD state.");
        ImGui.Separator();

        if (_worldScene == null && _terrainManager == null && _vlmTerrainManager == null)
        {
            ImGui.TextDisabled("Load a world map to inspect World LOD state.");
            return;
        }

        if (_worldScene != null)
        {
            bool showWdl = _worldScene.ShowWdlTerrain;
            if (ImGui.Checkbox("Show WDL Terrain", ref showWdl))
                _worldScene.ShowWdlTerrain = showWdl;

            bool showBoundingBoxes = _worldScene.ShowBoundingBoxes;
            if (ImGui.Checkbox("World Bounding Boxes", ref showBoundingBoxes))
                _worldScene.ShowBoundingBoxes = showBoundingBoxes;

            bool colorSpawns = _worldScene.ExternalSpawns.ColorSpawnBoundingBoxes;
            if (ImGui.Checkbox("Color NPC/GO BBs", ref colorSpawns))
                _worldScene.ExternalSpawns.ColorSpawnBoundingBoxes = colorSpawns;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Colors NPC spawn bounding boxes lime green and GameObject spawns warm amber.");

            ImGui.SameLine();
            bool spawnsOnly = _worldScene.ExternalSpawns.ShowSpawnBoundingBoxesOnly;
            if (ImGui.Checkbox("Spawn BBs Only", ref spawnsOnly))
                _worldScene.ExternalSpawns.ShowSpawnBoundingBoxesOnly = spawnsOnly;
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Only render bounding boxes for SQL-spawned NPCs and GameObjects, hiding static MDDF/MODF boxes.");

            bool showPm4Overlay = _worldScene.Pm4Overlay.ShowPm4Overlay;
            if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
                _worldScene.Pm4Overlay.ShowPm4Overlay = showPm4Overlay;
            if (_worldScene.Pm4Overlay.ShowPm4Overlay && ImGui.IsItemHovered())
                ImGui.SetTooltip(_worldScene.Pm4Overlay.Pm4Status);
        }

        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer != null)
        {
            int loadedTiles = _terrainManager?.LoadedTileCount ?? _vlmTerrainManager?.LoadedTileCount ?? 0;
            ImGui.Text($"Loaded tiles: {loadedTiles}");
            ImGui.Text($"Terrain chunks: {renderer.ChunksRendered} rendered / {renderer.ChunksCulled} culled");
        }

        if (_terrainManager != null)
        {
            int adtDetailTiles = _terrainManager.DetailedTileCountOverride <= 0
                ? _terrainManager.EffectiveDetailedTileCount
                : _terrainManager.DetailedTileCountOverride;
            if (ImGui.SliderInt("ADT Detail Tiles", ref adtDetailTiles, 1, TerrainManager.MaxManualDetailedTileCount))
            {
                _terrainManager.DetailedTileCountOverride = adtDetailTiles;
                _savedDetailedAdtTileCountOverride = _terrainManager.DetailedTileCountOverride;
            }
            if (ImGui.IsItemDeactivatedAfterEdit())
                _settings.SaveViewerSettings();

            ImGui.SameLine();
            if (ImGui.SmallButton("Auto##WorldLodAdtDetail"))
            {
                _terrainManager.DetailedTileCountOverride = 0;
                _savedDetailedAdtTileCountOverride = 0;
                _settings.SaveViewerSettings();
            }

            int retainedTileRadius = _terrainManager.RetainedTileRadius;
            if (ImGui.SliderInt("ADT Retain Radius##WorldLod", ref retainedTileRadius, TerrainManager.MinRetainedTileRadius, TerrainManager.MaxRetainedTileRadius))
                _terrainManager.RetainedTileRadius = retainedTileRadius;

            ImGui.TextDisabled(_terrainManager.DetailedTileCountOverride <= 0
                ? $"Active submission: {_terrainManager.EffectiveDetailedTileCount} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})"
                : $"Active submission: {_terrainManager.DetailedTileCountOverride} / retained window: {_terrainManager.EffectiveRetainedTileCount} (radius {_terrainManager.EffectiveRetainedTileRadius})");
        }

        ImGui.Separator();
        ImGui.TextDisabled("More World LOD facts belong here after the right-sidebar audit identifies the WDL data owner.");
    }

    internal static int DrawPageCombo(
        string id,
        string[] labels,
        int activeIndex)
    {
        if (labels.Length == 0)
            return 0;

        int selected = Math.Clamp(activeIndex, 0, labels.Length - 1);
        ImGui.SetNextItemWidth(-1f);
        if (ImGui.BeginCombo(id, labels[selected]))
        {
            for (int i = 0; i < labels.Length; i++)
            {
                bool isSelected = selected == i;
                if (ImGui.Selectable(labels[i], isSelected))
                    selected = i;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }
        return selected;
    }
}
