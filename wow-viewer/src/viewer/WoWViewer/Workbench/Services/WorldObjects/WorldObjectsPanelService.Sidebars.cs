using System.Numerics;
using System.Diagnostics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Workbench;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;
using WoWViewer.Population;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// WorldObjectsPanelService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class WorldObjectsPanelService
{

    internal void DrawObjectPathFilterControls()
    {
        if (_worldScene == null)
            return;

        ImGui.Separator();
        ImGui.Text("Object Path Filters");

        bool filtersEnabled = _worldScene.ObjectPathFiltersEnabled;
        if (ImGui.Checkbox("Enable Path Filters", ref filtersEnabled))
        {
            _worldScene.ObjectPathFiltersEnabled = filtersEnabled;
            PersistObjectPathFiltersForCurrentMap();
        }

        ImGui.SameLine();
        if (_worldScene.ObjectPathFilters.Count == 0)
            ImGui.BeginDisabled();
        if (ImGui.SmallButton("Clear All"))
        {
            _worldScene.ClearObjectPathFilters();
            PersistObjectPathFiltersForCurrentMap();
            _statusMessage = "Cleared object path filters for the current map.";
        }
        if (_worldScene.ObjectPathFilters.Count == 0)
            ImGui.EndDisabled();

        string filterInput = _objectPathFilterInput;
        if (ImGui.InputTextWithHint("Path Prefix", "World\\...", ref filterInput, 512))
            _objectPathFilterInput = filterInput;

        bool appliesToWmo = _objectPathFilterInputAppliesToWmo;
        if (ImGui.Checkbox("WMO##ObjectPathFilterWmo", ref appliesToWmo))
            _objectPathFilterInputAppliesToWmo = appliesToWmo;

        ImGui.SameLine();
        bool appliesToMdx = _objectPathFilterInputAppliesToMdx;
        if (ImGui.Checkbox("MDX##ObjectPathFilterMdx", ref appliesToMdx))
            _objectPathFilterInputAppliesToMdx = appliesToMdx;

        bool canAddFilter = !string.IsNullOrWhiteSpace(_objectPathFilterInput)
            && (_objectPathFilterInputAppliesToWmo || _objectPathFilterInputAppliesToMdx);
        if (!canAddFilter)
            ImGui.BeginDisabled();
        if (ImGui.Button("Add Filter"))
        {
            if (_worldScene.AddObjectPathFilter(_objectPathFilterInput, _objectPathFilterInputAppliesToWmo, _objectPathFilterInputAppliesToMdx))
            {
                PersistObjectPathFiltersForCurrentMap();
                _statusMessage = $"Added object path filter: {_objectPathFilterInput.Trim()}";
            }
            else
            {
                _statusMessage = "Object path filter was empty, duplicated, or had no enabled asset family.";
            }
        }
        if (!canAddFilter)
            ImGui.EndDisabled();

        if (TryGetSelectedWorldObjectModelPath(out string selectedModelPath, out bool selectedIsWmo))
        {
            ImGui.TextDisabled($"Selected: {selectedModelPath}");

            List<string> prefixCandidates = BuildObjectPathFilterPrefixCandidates(selectedModelPath);
            if (prefixCandidates.Count > 0 && ImGui.TreeNode("Quick Add From Selected Object"))
            {
                for (int i = 0; i < prefixCandidates.Count; i++)
                {
                    string prefix = prefixCandidates[i];
                    bool alreadyExists = _worldScene.ObjectPathFilters.Any(entry =>
                        string.Equals(entry.PathPrefix, prefix, StringComparison.OrdinalIgnoreCase)
                        && entry.AppliesToWmo == selectedIsWmo
                        && entry.AppliesToMdx == !selectedIsWmo);

                    if (alreadyExists)
                        ImGui.BeginDisabled();

                    if (ImGui.SmallButton($"{prefix}##QuickObjectPathFilter{i}")
                        && _worldScene.AddObjectPathFilter(prefix, selectedIsWmo, !selectedIsWmo))
                    {
                        PersistObjectPathFiltersForCurrentMap();
                        _statusMessage = $"Added {(selectedIsWmo ? "WMO" : "MDX")} family filter: {prefix}";
                    }

                    if (alreadyExists)
                        ImGui.EndDisabled();
                }

                ImGui.TreePop();
            }
        }

        if (_worldScene.ObjectPathFilters.Count == 0)
        {
            ImGui.TextDisabled("No path filters are saved for the current map.");
            return;
        }

        ImGui.TextDisabled($"Current map filters: {_worldScene.ObjectPathFilters.Count}");
        if (!ImGui.BeginTable("ObjectPathFiltersTable", 3, ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp))
            return;

        ImGui.TableSetupColumn("Family", ImGuiTableColumnFlags.WidthFixed, 84f);
        ImGui.TableSetupColumn("Prefix", ImGuiTableColumnFlags.WidthStretch);
        ImGui.TableSetupColumn("Action", ImGuiTableColumnFlags.WidthFixed, 72f);
        ImGui.TableHeadersRow();

        for (int i = 0; i < _worldScene.ObjectPathFilters.Count; i++)
        {
            ObjectPathFilterEntry entry = _worldScene.ObjectPathFilters[i];
            string familyLabel = entry.AppliesToWmo && entry.AppliesToMdx
                ? "WMO+MDX"
                : entry.AppliesToWmo
                    ? "WMO"
                    : "MDX";

            ImGui.TableNextRow();

            ImGui.TableSetColumnIndex(0);
            ImGui.TextUnformatted(familyLabel);

            ImGui.TableSetColumnIndex(1);
            ImGui.TextUnformatted(entry.PathPrefix);

            ImGui.TableSetColumnIndex(2);
            if (ImGui.SmallButton($"Remove##ObjectPathFilter{i}"))
            {
                _worldScene.RemoveObjectPathFilter(entry.PathPrefix, entry.AppliesToWmo, entry.AppliesToMdx);
                PersistObjectPathFiltersForCurrentMap();
                _statusMessage = $"Removed object path filter: {entry.PathPrefix}";
            }
        }

        ImGui.EndTable();
    }

    internal void DrawWorldObjectsPanelContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextWrapped("Load a world scene to inspect object, SQL population, POI, taxi, and PM4 overlay workflows.");
            return;
        }

        DrawWorldObjectsContentCore();
    }


    internal void DrawWorldPlacementsSubTab()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to inspect WMO and MDX placements.");
            return;
        }

        ImGui.TextDisabled("WMO and MDX placements. Double-click a row to focus the camera.");
        DrawPlacementListsContent();
    }
}
