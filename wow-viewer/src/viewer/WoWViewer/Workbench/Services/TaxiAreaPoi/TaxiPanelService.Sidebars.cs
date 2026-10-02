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

        if (_worldScene.TaxiActors.TaxiLoader != null && _worldScene.TaxiActors.TaxiLoader.Routes.Count > 0)
        {
            bool showTaxi = _worldScene.TaxiActors.ShowTaxi;
            if (ImGui.Checkbox($"Show Taxi Paths ({_worldScene.TaxiActors.TaxiLoader.Routes.Count})", ref showTaxi))
                _worldScene.TaxiActors.ShowTaxi = showTaxi;

            if (_worldScene.TaxiActors.ShowTaxi && (_worldScene.TaxiActors.SelectedTaxiNodeId >= 0 || _worldScene.TaxiActors.SelectedTaxiRouteId >= 0))
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Show All"))
                {
                    _worldScene.TaxiActors.ClearTaxiSelection();
                    _taxiAndAreaPoi.ClearSelectedTaxiInfo();
                }
            }
        }
        else if (!_worldScene.TaxiActors.TaxiLoadAttempted)
        {
            if (ImGui.Button("Load Taxi Paths"))
                _worldScene.TaxiActors.ShowTaxi = true;

            ImGui.TextDisabled("Load taxi paths to enable viewport picking, route browsing, and actor overrides.");
            return;
        }
        else
        {
            ImGui.TextDisabled("Taxi Paths: none found");
            return;
        }

        bool hasTaxiSelection = _worldScene.TaxiActors.SelectedTaxiNodeId >= 0 || _worldScene.TaxiActors.SelectedTaxiRouteId >= 0;
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

        bool hasSelectedTaxiRoute = _worldScene.TaxiActors.SelectedTaxiRouteId >= 0;
        bool rideCameraAttachedToSelection = _taxiRideCameraEnabled
            && hasSelectedTaxiRoute
            && _taxiRideCameraRouteId == _worldScene.TaxiActors.SelectedTaxiRouteId;
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

        ImGui.Checkbox("Auto-stop at destination", ref _taxiAutoStopOnRouteArrival);
        ImGui.SameLine();
        ImGui.Checkbox("Route Tour Callouts", ref _taxiRecordWithFeatureTour);

        bool isAnyRecordingActive = _recordingCoordinator.IsRecording;

        if (!isAnyRecordingActive)
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
            {
                _recordingCoordinator.StopRecording("Taxi route video recording stopped by user.");
            }

            if (_recordingCoordinator.ActiveSession is { } active)
            {
                ImGui.SameLine();
                ImGui.TextDisabled($"{Path.GetFileName(active.OutputPath)} ({active.ElapsedSeconds:F1}s)");
            }
        }

        if (!isAnyRecordingActive && _recordingCoordinator.LastCompletedSession is { } lastCompleted
            && lastCompleted.SourceKind == RecordingSourceKind.TaxiRoute)
        {
            ImGui.TextDisabled($"Saved: {Path.GetFileName(lastCompleted.OutputPath)} ({lastCompleted.DurationSeconds:F1}s)");
        }

        if (ImGui.CollapsingHeader("Flight Playlist & Promo Video Showreel", ImGuiTreeNodeFlags.DefaultOpen))
        {
            TaxiPlaylistService playlist = _host.TaxiPlaylist;
            ShowreelOverlayService showreel = _host.ShowreelOverlay;

            bool showreelOn = showreel.Config.EnableOverlay;
            if (ImGui.Checkbox("Tour HUD Overlay in Video", ref showreelOn))
                showreel.Config.EnableOverlay = showreelOn;

            ImGui.SameLine();
            bool previewOn = showreel.Config.PreviewInViewport;
            if (ImGui.Checkbox("Preview in Viewport", ref previewOn))
                showreel.Config.PreviewInViewport = previewOn;

            if (showreel.Config.EnableOverlay || showreel.Config.PreviewInViewport)
            {
                bool showTele = showreel.Config.ShowLiveTelemetry;
                if (ImGui.Checkbox("Live Telemetry & Coordinates", ref showTele))
                    showreel.Config.ShowLiveTelemetry = showTele;
                ImGui.SameLine();
                bool showZone = showreel.Config.ShowZoneBanners;
                if (ImGui.Checkbox("Zone Banners", ref showZone))
                    showreel.Config.ShowZoneBanners = showZone;

                bool showLand = showreel.Config.ShowLandmarkCallouts;
                if (ImGui.Checkbox("Landmark Proximity Callouts", ref showLand))
                    showreel.Config.ShowLandmarkCallouts = showLand;
                ImGui.SameLine();
                bool showBadges = showreel.Config.ShowEngineBadges;
                if (ImGui.Checkbox("Engine Feature Badges", ref showBadges))
                    showreel.Config.ShowEngineBadges = showBadges;

                bool showPerf = showreel.Config.ShowPerformanceTelemetry;
                if (ImGui.Checkbox("Performance & Hitch Diagnostics", ref showPerf))
                    showreel.Config.ShowPerformanceTelemetry = showPerf;
                ImGui.SameLine();
                bool showPipe = showreel.Config.ShowPipelineTelemetry;
                if (ImGui.Checkbox("Pipeline & Raw Data Metrics", ref showPipe))
                    showreel.Config.ShowPipelineTelemetry = showPipe;

                bool showHitch = showreel.Config.ShowHitchAlerts;
                if (ImGui.Checkbox("Visual Hitch Alert Badges", ref showHitch))
                    showreel.Config.ShowHitchAlerts = showHitch;
                ImGui.SameLine();
                bool expDiag = showreel.Config.ExpandedDiagnostics;
                if (ImGui.Checkbox("Expanded Stage Breakdown", ref expDiag))
                    showreel.Config.ExpandedDiagnostics = expDiag;

                if (showreel.Config.ShowHitchAlerts)
                {
                    float threshold = showreel.Config.HitchThresholdMs;
                    if (ImGui.SliderFloat("Hitch Spike Threshold", ref threshold, 16.6f, 100.0f, "%.1f ms"))
                        showreel.Config.HitchThresholdMs = threshold;
                }
            }

            ImGui.Separator();

            ImGui.Text($"Playlist Segments ({playlist.Items.Count}):");
            if (playlist.Items.Count == 0)
            {
                ImGui.TextDisabled("Playlist is empty. Select a route and click 'Add Selected Route' or 'Auto-Chain'.");
            }
            else
            {
                for (int i = 0; i < playlist.Items.Count; i++)
                {
                    var item = playlist.Items[i];
                    bool isCurrent = playlist.IsPlaying && playlist.CurrentIndex == i;
                    string prefix = isCurrent ? ">> " : $"{i + 1}. ";
                    ImGui.TextUnformatted($"{prefix}{item.DisplayLabel} ({item.RouteLength:F0} yd)");

                    ImGui.SameLine();
                    if (ImGui.SmallButton($"^##up_{i}") && i > 0)
                        playlist.MoveUp(i);

                    ImGui.SameLine();
                    if (ImGui.SmallButton($"v##dn_{i}") && i < playlist.Items.Count - 1)
                        playlist.MoveDown(i);

                    ImGui.SameLine();
                    if (ImGui.SmallButton($"X##del_{i}"))
                        playlist.RemoveAt(i);
                }
            }

            if (hasSelectedTaxiRoute)
            {
                if (ImGui.Button("Add Selected Route"))
                    playlist.AddRoute(_worldScene.TaxiActors.SelectedTaxiRouteId);

                ImGui.SameLine();
                if (ImGui.Button("Auto-Chain 4 Hops"))
                {
                    var r = _worldScene.TaxiActors.GetTaxiRoute(_worldScene.TaxiActors.SelectedTaxiRouteId);
                    if (r != null)
                        playlist.BuildAutoChain(r.FromNodeId, 4);
                }
            }

            if (playlist.Items.Count > 0)
            {
                ImGui.SameLine();
                if (ImGui.Button("Clear"))
                    playlist.Clear();

                bool loop = playlist.Loop;
                if (ImGui.Checkbox("Loop Playlist", ref loop))
                    playlist.Loop = loop;

                if (!playlist.IsPlaying)
                {
                    if (ImGui.Button("Play Playlist"))
                        playlist.StartPlaylist(recordVideo: false);

                    ImGui.SameLine();
                    if (ImGui.Button("Record Playlist Video"))
                    {
                        playlist.StartPlaylist(
                            recordVideo: true,
                            videoFps: _videoCaptureFps,
                            includeUi: _videoCaptureIncludeUi,
                            includeShowreelOverlay: showreel.Config.EnableOverlay);
                    }
                }
                else
                {
                    if (ImGui.Button("Stop Playlist"))
                        playlist.StopPlaylist("Stopped by user.");

                    ImGui.SameLine();
                    ImGui.TextDisabled(playlist.StatusText);
                }
            }
        }

        bool showTaxiActors = _worldScene.TaxiActors.ShowTaxiActors;
        if (ImGui.Checkbox("Show Animated Taxi Actor", ref showTaxiActors))
            _worldScene.TaxiActors.ShowTaxiActors = showTaxiActors;

        float speedMultiplier = _worldScene.TaxiActors.TaxiActorSpeedMultiplier;
        if (ImGui.SliderFloat("Taxi Speed", ref speedMultiplier, TaxiActorScene.TaxiActorMinSpeedSetting, TaxiActorScene.TaxiActorMaxSpeedSetting, "%.2f"))
            _worldScene.TaxiActors.TaxiActorSpeedMultiplier = speedMultiplier;
        ImGui.TextDisabled("0.10 = 100% speed, 0.01 = 10%, 0.50 = 500%.");

        float scaleMultiplier = _worldScene.TaxiActors.TaxiActorScaleMultiplier;
        if (ImGui.SliderFloat("Taxi Actor Scale", ref scaleMultiplier, 0.05f, 5f, "%.2fx"))
            _worldScene.TaxiActors.TaxiActorScaleMultiplier = scaleMultiplier;

        ImGui.Separator();
        string[] taxiGroupingLabels = { "None", "From Node", "To Node" };
        ImGui.Text($"Routes ({_worldScene.TaxiActors.TaxiLoader.Routes.Count})");
        ImGui.SetNextItemWidth(140f);
        ImGui.Combo("Group By", ref _taxiRouteListGroupingMode, taxiGroupingLabels, taxiGroupingLabels.Length);

        string taxiRouteFilter = _taxiRouteFilter;
        if (ImGui.InputText("Search Routes", ref taxiRouteFilter, 256))
            _taxiRouteFilter = taxiRouteFilter;

        var routeEntries = new List<(TaxiPathLoader.TaxiRoute Route, string FromName, string ToName, string Label, string GroupKey)>();
        foreach (TaxiPathLoader.TaxiRoute route in _worldScene.TaxiActors.TaxiLoader.Routes)
        {
            string fromName = _worldScene.TaxiActors.GetTaxiNode(route.FromNodeId)?.Name ?? $"#{route.FromNodeId}";
            string toName = _worldScene.TaxiActors.GetTaxiNode(route.ToNodeId)?.Name ?? $"#{route.ToNodeId}";
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

        if (routeEntries.Count != _worldScene.TaxiActors.TaxiLoader.Routes.Count)
            ImGui.TextDisabled($"Showing {routeEntries.Count} of {_worldScene.TaxiActors.TaxiLoader.Routes.Count} routes");

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

                    bool isSelected = _worldScene.TaxiActors.SelectedTaxiRouteId == entry.Route.PathId;
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

        if (_worldScene.TaxiActors.SelectedTaxiNodeId >= 0)
            ImGui.TextDisabled($"Selected taxi node: {_worldScene.TaxiActors.SelectedTaxiNodeId}");
        else if (_worldScene.TaxiActors.SelectedTaxiRouteId >= 0)
            ImGui.TextDisabled($"Selected taxi route: {_worldScene.TaxiActors.SelectedTaxiRouteId}");

        if (_taxiAndAreaPoi.TryGetTaxiActorOverrideRouteId(out int routeId))
        {
            IReadOnlyList<TaxiPathLoader.TaxiRoute> candidateRoutes = _taxiAndAreaPoi.GetTaxiActorOverrideCandidateRoutes();

            if (_worldScene.TaxiActors.SelectedTaxiNodeId >= 0)
            {
                ImGui.TextDisabled($"Selected taxi node: {_worldScene.TaxiActors.SelectedTaxiNodeId}");

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
            else if (_worldScene.TaxiActors.SelectedTaxiRouteId >= 0)
            {
                ImGui.TextDisabled($"Selected taxi route: {_worldScene.TaxiActors.SelectedTaxiRouteId}");
            }

            _taxiAndAreaPoi.SyncTaxiActorModelOverrideInput(routeId);

            string resolvedActorModelPath = _worldScene.TaxiActors.GetResolvedTaxiActorModelPath(routeId) ?? "not found";
            string? actorOverridePath = _worldScene.TaxiActors.GetTaxiActorModelOverride(routeId);
            IReadOnlyList<string> defaultTaxiActorModels = TaxiActorScene.DefaultTaxiActorModelPaths;
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
        else if (_worldScene.TaxiActors.SelectedTaxiNodeId >= 0)
            ImGui.TextDisabled("No connected routes were found for this taxi node.");
        else
            ImGui.TextDisabled("Select a taxi route from the list or click one in the viewport to configure the animated actor.");
    }

    internal void DrawTaxiContent()
    {
        DrawSelectedTaxiControls();
    }
}
