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
using static WoWViewer.CaptureAutomationService;

namespace WoWViewer;

// TaxiPanelService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class TaxiPanelService
{

    private void DrawSelectedTaxiControls()
    {
        if (_worldScene == null)
            return;

        if (_worldScene.TaxiLoader != null && _worldScene.TaxiLoader.Routes.Count > 0)
        {
            bool showTaxi = _worldScene.ShowTaxi;
            if (ImGui.Checkbox($"Show Taxi Paths ({_worldScene.TaxiLoader.Routes.Count})", ref showTaxi))
                _worldScene.ShowTaxi = showTaxi;

            if (_worldScene.ShowTaxi && (_worldScene.SelectedTaxiNodeId >= 0 || _worldScene.SelectedTaxiRouteId >= 0))
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Show All"))
                {
                    _worldScene.ClearTaxiSelection();
                    _taxiAndAreaPoi.ClearSelectedTaxiInfo();
                }
            }
        }
        else if (!_worldScene.TaxiLoadAttempted)
        {
            if (ImGui.Button("Load Taxi Paths"))
                _worldScene.ShowTaxi = true;

            ImGui.TextDisabled("Load taxi paths to enable viewport picking, route browsing, and actor overrides.");
            return;
        }
        else
        {
            ImGui.TextDisabled("Taxi Paths: none found");
            return;
        }

        bool hasTaxiSelection = _worldScene.SelectedTaxiNodeId >= 0 || _worldScene.SelectedTaxiRouteId >= 0;
        if (hasTaxiSelection && !string.IsNullOrWhiteSpace(_selectedObjectInfo))
        {
            ImGui.Separator();
            ImGui.TextWrapped(_selectedObjectInfo);
        }

        ImGui.Separator();
        ImGui.Text("Taxi Controls");

        if (!hasTaxiSelection)
            ImGui.BeginDisabled();
        if (ImGui.Button("Focus Selected Taxi"))
            _taxiAndAreaPoi.FocusSelectedTaxi();
        if (!hasTaxiSelection)
            ImGui.EndDisabled();

        bool hasSelectedTaxiRoute = _worldScene.SelectedTaxiRouteId >= 0;
        bool rideCameraAttachedToSelection = _taxiRideCameraEnabled
            && hasSelectedTaxiRoute
            && _taxiRideCameraRouteId == _worldScene.SelectedTaxiRouteId;
        bool rideCameraActive = _taxiRideCameraEnabled && _taxiRideCameraRouteId >= 0;

        bool canToggleRideCamera = hasSelectedTaxiRoute || _taxiRideCameraEnabled;
        if (!canToggleRideCamera)
            ImGui.BeginDisabled();
        if (ImGui.Button(rideCameraActive ? "Detach Ride Camera" : "Ride Selected Route"))
        {
            if (rideCameraActive)
                StopTaxiRideCamera("Ride camera detached.");
            else
                TryAttachTaxiRideCameraToSelectedRoute();
        }
        if (!canToggleRideCamera)
            ImGui.EndDisabled();

        if (_taxiRideCameraEnabled)
            ImGui.TextDisabled($"Ride Camera: {_taxiAndAreaPoi.GetTaxiRouteDisplayLabel(_taxiRideCameraRouteId)}");

        int taxiRideCameraMode = (int)_taxiRideCameraMode;
        string[] taxiRideCameraLabels = { "Cockpit", "Chase" };
        if (ImGui.Combo("Ride Camera Mode", ref taxiRideCameraMode, taxiRideCameraLabels, taxiRideCameraLabels.Length))
            _taxiRideCameraMode = (TaxiRideCameraMode)taxiRideCameraMode;

        if (_taxiRideCameraMode == TaxiRideCameraMode.Cockpit)
        {
            float cockpitHeight = _taxiRideCockpitHeight;
            if (ImGui.SliderFloat("Ride Camera Height", ref cockpitHeight, 2f, 30f, "%.1f"))
                _taxiRideCockpitHeight = cockpitHeight;
        }
        else
        {
            float chaseDistance = _taxiRideChaseDistance;
            if (ImGui.SliderFloat("Ride Chase Distance", ref chaseDistance, 8f, 120f, "%.1f"))
                _taxiRideChaseDistance = chaseDistance;

            float chaseHeight = _taxiRideChaseHeight;
            if (ImGui.SliderFloat("Ride Chase Height", ref chaseHeight, 2f, 40f, "%.1f"))
                _taxiRideChaseHeight = chaseHeight;
        }

        float rideLookAhead = _taxiRideLookAhead;
        if (ImGui.SliderFloat("Ride Look Ahead", ref rideLookAhead, 8f, 80f, "%.1f"))
            _taxiRideLookAhead = rideLookAhead;

        int videoFps = _videoCaptureFps;
        if (ImGui.SliderInt("Ride Video FPS", ref videoFps, 12, 60))
            _videoCaptureFps = videoFps;

        bool videoIncludeUi = _videoCaptureIncludeUi;
        if (ImGui.Checkbox("Ride Video Includes UI", ref videoIncludeUi))
            _videoCaptureIncludeUi = videoIncludeUi;

        if (_activeVideoRecording == null)
        {
            if (!hasSelectedTaxiRoute)
                ImGui.BeginDisabled();
            if (ImGui.Button("Record Selected Route Video"))
                TryStartTaxiRideVideoCapture();
            if (!hasSelectedTaxiRoute)
                ImGui.EndDisabled();
        }
        else
        {
            if (ImGui.Button("Stop Route Video"))
                StopVideoRecording();

            ImGui.SameLine();
            ImGui.TextDisabled(Path.GetFileName(_activeVideoRecording.OutputPath));
        }

        bool showTaxiActors = _worldScene.ShowTaxiActors;
        if (ImGui.Checkbox("Show Animated Taxi Actor", ref showTaxiActors))
            _worldScene.ShowTaxiActors = showTaxiActors;

        float speedMultiplier = _worldScene.TaxiActorSpeedMultiplier;
        if (ImGui.SliderFloat("Taxi Speed", ref speedMultiplier, WorldScene.TaxiActorMinSpeedSetting, WorldScene.TaxiActorMaxSpeedSetting, "%.2f"))
            _worldScene.TaxiActorSpeedMultiplier = speedMultiplier;
        ImGui.TextDisabled("0.10 = 100% speed, 0.01 = 10%, 0.50 = 500%.");

        float scaleMultiplier = _worldScene.TaxiActorScaleMultiplier;
        if (ImGui.SliderFloat("Taxi Actor Scale", ref scaleMultiplier, 0.05f, 5f, "%.2fx"))
            _worldScene.TaxiActorScaleMultiplier = scaleMultiplier;

        ImGui.Separator();
        string[] taxiGroupingLabels = { "None", "From Node", "To Node" };
        ImGui.Text($"Routes ({_worldScene.TaxiLoader.Routes.Count})");
        ImGui.SetNextItemWidth(140f);
        ImGui.Combo("Group By", ref _taxiRouteListGroupingMode, taxiGroupingLabels, taxiGroupingLabels.Length);

        string taxiRouteFilter = _taxiRouteFilter;
        if (ImGui.InputText("Search Routes", ref taxiRouteFilter, 256))
            _taxiRouteFilter = taxiRouteFilter;

        var routeEntries = new List<(TaxiPathLoader.TaxiRoute Route, string FromName, string ToName, string Label, string GroupKey)>();
        foreach (TaxiPathLoader.TaxiRoute route in _worldScene.TaxiLoader.Routes)
        {
            string fromName = _worldScene.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
            string toName = _worldScene.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
            string label = $"{_taxiAndAreaPoi.GetTaxiRouteDisplayLabel(route.PathId)} ({route.Waypoints.Count} pts)";
            string searchText = $"{route.PathId} {fromName} {toName} {label}";
            if (!string.IsNullOrWhiteSpace(_taxiRouteFilter)
                && !searchText.Contains(_taxiRouteFilter, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string groupKey = _taxiRouteListGroupingMode switch
            {
                1 => fromName,
                2 => toName,
                _ => string.Empty,
            };

            routeEntries.Add((route, fromName, toName, label, groupKey));
        }

        routeEntries.Sort((left, right) =>
        {
            if (_taxiRouteListGroupingMode != 0)
            {
                int groupCompare = StringComparer.OrdinalIgnoreCase.Compare(left.GroupKey, right.GroupKey);
                if (groupCompare != 0)
                    return groupCompare;
            }

            int primaryCompare = _taxiRouteListGroupingMode switch
            {
                1 => StringComparer.OrdinalIgnoreCase.Compare(left.ToName, right.ToName),
                2 => StringComparer.OrdinalIgnoreCase.Compare(left.FromName, right.FromName),
                _ => 0,
            };
            if (primaryCompare != 0)
                return primaryCompare;

            return left.Route.PathId.CompareTo(right.Route.PathId);
        });

        if (routeEntries.Count != _worldScene.TaxiLoader.Routes.Count)
            ImGui.TextDisabled($"Showing {routeEntries.Count} of {_worldScene.TaxiLoader.Routes.Count} routes");

        if (ImGui.BeginChild("##TaxiRouteSidebarList", new Vector2(0, 220f), true))
        {
            if (routeEntries.Count == 0)
            {
                ImGui.TextDisabled(string.IsNullOrWhiteSpace(_taxiRouteFilter)
                    ? "No taxi routes are available."
                    : "No taxi routes match the current search.");
            }
            else
            {
                Dictionary<string, int>? groupCounts = null;
                string currentGroupKey = string.Empty;
                if (_taxiRouteListGroupingMode != 0)
                {
                    groupCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                    foreach (var entry in routeEntries)
                        groupCounts[entry.GroupKey] = groupCounts.TryGetValue(entry.GroupKey, out int count) ? count + 1 : 1;
                }

                for (int i = 0; i < routeEntries.Count; i++)
                {
                    var entry = routeEntries[i];
                    if (_taxiRouteListGroupingMode != 0 && !string.Equals(currentGroupKey, entry.GroupKey, StringComparison.OrdinalIgnoreCase))
                    {
                        currentGroupKey = entry.GroupKey;
                        if (i > 0)
                            ImGui.Separator();
                        ImGui.TextDisabled($"{currentGroupKey} ({groupCounts![currentGroupKey]})");
                    }

                    bool isSelected = _worldScene.SelectedTaxiRouteId == entry.Route.PathId;
                    if (ImGui.Selectable(entry.Label, isSelected, ImGuiSelectableFlags.AllowDoubleClick))
                    {
                        _taxiAndAreaPoi.SelectTaxiRoute(entry.Route.PathId, toggle: true);
                        if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                            _taxiAndAreaPoi.FocusSelectedTaxi();
                    }

                    if (ImGui.IsItemHovered())
                    {
                        ImGui.BeginTooltip();
                        ImGui.Text($"Cost: {entry.Route.Cost}");
                        ImGui.Text($"From: {entry.FromName}");
                        ImGui.Text($"To: {entry.ToName}");
                        ImGui.Text($"Waypoints: {entry.Route.Waypoints.Count}");
                        ImGui.Text("Single-click selects the route. Double-click focuses the camera.");
                        ImGui.EndTooltip();
                    }
                }
            }

            ImGui.EndChild();
        }

        if (_worldScene.SelectedTaxiNodeId >= 0)
            ImGui.TextDisabled($"Selected taxi node: {_worldScene.SelectedTaxiNodeId}");
        else if (_worldScene.SelectedTaxiRouteId >= 0)
            ImGui.TextDisabled($"Selected taxi route: {_worldScene.SelectedTaxiRouteId}");

        if (_taxiAndAreaPoi.TryGetTaxiActorOverrideRouteId(out int routeId))
        {
            IReadOnlyList<TaxiPathLoader.TaxiRoute> candidateRoutes = _taxiAndAreaPoi.GetTaxiActorOverrideCandidateRoutes();

            if (_worldScene.SelectedTaxiNodeId >= 0)
            {
                ImGui.TextDisabled($"Selected taxi node: {_worldScene.SelectedTaxiNodeId}");

                string previewLabel = _taxiAndAreaPoi.GetTaxiRouteDisplayLabel(routeId);
                if (ImGui.BeginCombo("Override Target Route", previewLabel))
                {
                    foreach (TaxiPathLoader.TaxiRoute candidateRoute in candidateRoutes)
                    {
                        bool isSelected = candidateRoute.PathId == routeId;
                        if (ImGui.Selectable(_taxiAndAreaPoi.GetTaxiRouteDisplayLabel(candidateRoute.PathId), isSelected))
                        {
                            _taxiActorModelOverrideTargetRouteId = candidateRoute.PathId;
                            _taxiAndAreaPoi.SyncTaxiActorModelOverrideInput(candidateRoute.PathId);
                        }

                        if (isSelected)
                            ImGui.SetItemDefaultFocus();
                    }

                    ImGui.EndCombo();
                }
            }
            else if (_worldScene.SelectedTaxiRouteId >= 0)
            {
                ImGui.TextDisabled($"Selected taxi route: {_worldScene.SelectedTaxiRouteId}");
            }

            _taxiAndAreaPoi.SyncTaxiActorModelOverrideInput(routeId);

            string resolvedActorModelPath = _worldScene.GetResolvedTaxiActorModelPath(routeId) ?? "not found";
            string? actorOverridePath = _worldScene.GetTaxiActorModelOverride(routeId);
            IReadOnlyList<string> defaultTaxiActorModels = WorldScene.DefaultTaxiActorModelPaths;
            ImGui.TextWrapped($"Override Route: {_taxiAndAreaPoi.GetTaxiRouteDisplayLabel(routeId)}");
            ImGui.TextWrapped($"Resolved Actor Model: {resolvedActorModelPath}");
            ImGui.TextDisabled($"Override: {actorOverridePath ?? "auto"}");

            string actorModelPath = _taxiActorModelOverrideInput;
            if (ImGui.InputText("Actor Model Path", ref actorModelPath, 512))
                _taxiActorModelOverrideInput = actorModelPath;

            if (defaultTaxiActorModels.Count > 0)
            {
                if (ImGui.Button("Use Gryphon Default"))
                {
                    _taxiActorModelOverrideInput = defaultTaxiActorModels[0];
                    _taxiActorModelOverrideInputRouteId = routeId;
                    _taxiAndAreaPoi.ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                    _taxiAndAreaPoi.RefreshSelectedTaxiInfo();
                }
            }

            if (defaultTaxiActorModels.Count > 1)
            {
                ImGui.SameLine();
                if (ImGui.Button("Use FelBat Default"))
                {
                    _taxiActorModelOverrideInput = defaultTaxiActorModels[1];
                    _taxiActorModelOverrideInputRouteId = routeId;
                    _taxiAndAreaPoi.ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                    _taxiAndAreaPoi.RefreshSelectedTaxiInfo();
                }
            }

            if (ImGui.Button("Apply Model Override"))
            {
                _taxiAndAreaPoi.ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                _taxiAndAreaPoi.SyncTaxiActorModelOverrideInput(routeId);
                _taxiAndAreaPoi.RefreshSelectedTaxiInfo();
            }

            ImGui.SameLine();
            if (ImGui.Button("Clear Override"))
            {
                _taxiAndAreaPoi.ApplyTaxiActorModelOverride(routeId, null);
                _taxiAndAreaPoi.SyncTaxiActorModelOverrideInput(routeId);
                _taxiAndAreaPoi.RefreshSelectedTaxiInfo();
            }

            if (TryGetSelectedBrowserModelPath(out string selectedBrowserModelPath))
            {
                if (ImGui.Button("Use Selected Browser Asset"))
                {
                    _taxiActorModelOverrideInput = selectedBrowserModelPath.Replace('/', '\\');
                    _taxiActorModelOverrideInputRouteId = routeId;
                    _taxiAndAreaPoi.ApplyTaxiActorModelOverride(routeId, _taxiActorModelOverrideInput);
                    _taxiAndAreaPoi.RefreshSelectedTaxiInfo();
                }

                ImGui.SameLine();
                ImGui.TextDisabled(Path.GetFileName(selectedBrowserModelPath));
            }

            if (_taxiAndAreaPoi.TryGetLoadedTaxiActorModelPath(out string loadedModelPath))
            {
                if (ImGui.Button("Use Loaded Model"))
                {
                    _taxiActorModelOverrideInput = loadedModelPath;
                    _taxiActorModelOverrideInputRouteId = routeId;
                    _taxiAndAreaPoi.ApplyTaxiActorModelOverride(routeId, loadedModelPath);
                    _taxiAndAreaPoi.RefreshSelectedTaxiInfo();
                }

                ImGui.SameLine();
                ImGui.TextDisabled(Path.GetFileName(loadedModelPath));
            }

            if (!string.IsNullOrWhiteSpace(actorOverridePath))
            {
                if (ImGui.Button("Copy Override Path"))
                    CopyTextToClipboard(actorOverridePath, "override path");

                ImGui.SameLine();
                if (ImGui.Button("Open Override Asset"))
                    _modelLoader.LoadFileFromDataSource(actorOverridePath);

                if (_dataSourceSession.HasWorldReturnTarget() && _worldScene == null)
                {
                    ImGui.SameLine();
                    if (ImGui.Button("Return To Last World"))
                        _dataSourceSession.ReturnToLastWorldScene();
                }
            }
        }
        else if (_worldScene.SelectedTaxiNodeId >= 0)
            ImGui.TextDisabled("No connected routes were found for this taxi node.");
        else
            ImGui.TextDisabled("Select a taxi route from the list or click one in the viewport to configure the animated actor.");
    }

    internal void DrawTaxiContent()
    {
        DrawSelectedTaxiControls();
    }
}
