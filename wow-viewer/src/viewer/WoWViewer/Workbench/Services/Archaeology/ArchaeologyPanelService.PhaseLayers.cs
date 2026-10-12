using System.IO;
using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.Maps;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// ArchaeologyPanelService: members moved from ViewerApp_PhaseLayers.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ArchaeologyPanelService
{

    private DbcMapPhaseTable? _mapPhaseTable;
    private string? _mapPhaseTableBuild;

    /// <summary>
    /// Offer the current map's child maps straight from <c>Map.dbc</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>Map.dbc.ParentMapID</c> is the relationship that says which maps are phases of which.
    /// MEASURED on 5.0.1.15464: 18 of 239 map rows carry a parent, e.g.
    /// <c>JadeForestAllianceHubPhase</c> and <c>JadeForestBattlefieldPhase</c> both name
    /// <c>HawaiiMainLand</c> (Pandaria).
    /// </para>
    /// <para>
    /// <c>Phase.dbc</c> deliberately plays no part here: its 5.0.1 layout is <c>ID, Name, Flags</c>
    /// with <b>no map reference at all</b>. The <c>MapID</c> column some documentation shows belongs
    /// to the 4.0.0.11927-12539 layout and was removed by 4.0.0.12911, so driving discovery from it
    /// would silently find nothing on this client.
    /// </para>
    /// </remarks>
    private bool DrawDbcPhaseDiscovery(TerrainManager terrain, IList<PhaseLayerSettings> layers)
    {
        if (_dbcProvider == null || string.IsNullOrWhiteSpace(_dbdDir) || string.IsNullOrWhiteSpace(_dbcBuild))
            return false;

        if (_mapPhaseTable == null || !string.Equals(_mapPhaseTableBuild, _dbcBuild, StringComparison.Ordinal))
        {
            try
            {
                _mapPhaseTable = DbcMapPhaseTable.Load(_dbcProvider, _dbdDir!, _dbcBuild!);
            }
            catch (Exception ex)
            {
                _mapPhaseTable = DbcMapPhaseTable.Empty;
                ViewerLog.Important(ViewerLog.Category.Dbc, $"[Phase] Map.dbc phase relationships unavailable: {ex.Message}");
            }

            _mapPhaseTableBuild = _dbcBuild;
        }

        string baseMap = terrain.MapName ?? string.Empty;
        IReadOnlyList<MapPhaseRecord> children = _mapPhaseTable.GetChildMapsOf(baseMap);

        if (children.Count == 0)
        {
            ImGui.TextDisabled($"Map.dbc lists no child maps for '{baseMap}'.");
            return false;
        }

        ImGui.TextWrapped($"Map.dbc lists {children.Count} child map(s) for '{baseMap}':");

        bool dirty = false;
        foreach (MapPhaseRecord child in children)
        {
            bool alreadyAdded = layers.Any(l => string.Equals(l.MapName, child.Directory, StringComparison.OrdinalIgnoreCase));

            ImGui.PushID(child.MapId);
            if (alreadyAdded)
            {
                ImGui.TextDisabled($"  [{child.MapId}] {child.Directory} - added");
            }
            else
            {
                if (ImGui.SmallButton("Add"))
                {
                    layers.Add(new PhaseLayerSettings { MapName = child.Directory });
                    dirty = true;
                }

                ImGui.SameLine();
                ImGui.Text($"[{child.MapId}] {child.Directory}");
                if (!string.Equals(child.DisplayName, child.Directory, StringComparison.OrdinalIgnoreCase))
                {
                    ImGui.SameLine();
                    ImGui.TextDisabled($"- {child.DisplayName}");
                }
            }

            ImGui.PopID();
        }

        if (children.Any(c => !layers.Any(l => string.Equals(l.MapName, c.Directory, StringComparison.OrdinalIgnoreCase)))
            && ImGui.Button("Add all child maps"))
        {
            foreach (MapPhaseRecord child in children)
            {
                if (!layers.Any(l => string.Equals(l.MapName, child.Directory, StringComparison.OrdinalIgnoreCase)))
                {
                    layers.Add(new PhaseLayerSettings { MapName = child.Directory });
                    dirty = true;
                }
            }
        }

        ImGui.Separator();
        return dirty;
    }

    private bool DrawPhaseLayerAddControls(TerrainManager terrain, IList<PhaseLayerSettings> layers)
    {
        bool dirty = false;
        string currentBaseMap = terrain.MapName ?? string.Empty;

        var available = MapListSorting.Sort(
                _discoveredMaps
                    .Where(m => !string.Equals(m.Directory, currentBaseMap, StringComparison.OrdinalIgnoreCase))
                    .Where(m => !layers.Any(l => string.Equals(l.MapName, m.Directory, StringComparison.OrdinalIgnoreCase))),
                _mapListSortMode)
            .ToList();

        DrawMapSortModeSelector("##phaseSort");

        if (available.Count > 0 && ImGui.BeginCombo("##addPhaseLayer", "Add phase map..."))
        {
            ImGui.InputTextWithHint("##phaseFilter", "Filter maps...", ref _secondaryOverlaySearchFilter, 128);
            foreach (var map in available)
            {
                string displayName = map.HasDbcEntry
                    ? $"[{map.Id:D3}] {map.Name} ({map.Directory})"
                    : $"[custom] {map.Name} ({map.Directory})";

                if (!string.IsNullOrEmpty(_secondaryOverlaySearchFilter)
                    && displayName.IndexOf(_secondaryOverlaySearchFilter, StringComparison.OrdinalIgnoreCase) < 0
                    && map.Directory.IndexOf(_secondaryOverlaySearchFilter, StringComparison.OrdinalIgnoreCase) < 0)
                {
                    continue;
                }

                if (ImGui.Selectable(displayName))
                {
                    layers.Add(new PhaseLayerSettings { MapName = map.Directory });
                    dirty = true;
                }
            }

            ImGui.EndCombo();
        }

        ImGui.InputTextWithHint("##manualPhaseMap", "Manual map directory (e.g. Gilneas)", ref _secondaryOverlayMapInput, 256);
        ImGui.SameLine();
        if (ImGui.Button("Add") && !string.IsNullOrWhiteSpace(_secondaryOverlayMapInput))
        {
            string name = _secondaryOverlayMapInput.Trim();
            if (!layers.Any(l => string.Equals(l.MapName, name, StringComparison.OrdinalIgnoreCase)))
            {
                layers.Add(new PhaseLayerSettings { MapName = name });
                dirty = true;
            }

            _secondaryOverlayMapInput = string.Empty;
        }

        // Spec 247 US5: a folder of DAT files as a donor. DAT files are the terrain project files the client's
        // ADTs were built from, so this overlays the source data on the shipped tiles. The picker is opened
        // directly (it defers its own draw), so this adds no ViewerApp state -- AGENTS.md section 10.
        ImGui.SameLine();
        if (ImGui.Button("Add DAT folder..."))
        {
            ImGuiPathPicker.Instance.Open(
                "Select a folder of DAT terrain files to overlay (v22/23/26; any file name or extension)",
                pickFolder: true,
                initialPath: _cascAhdrSource._lastAhdrTerrainFolder,
                filterExtension: null,
                path =>
                {
                    if (string.IsNullOrEmpty(path) || !Directory.Exists(path))
                        return;

                    string locator = DatLayerSource.ForFolder(path);
                    if (layers.Any(l => string.Equals(l.MapName, locator, StringComparison.OrdinalIgnoreCase)))
                        return;

                    DatLayerSource.Forget(locator);
                    layers.Add(new PhaseLayerSettings { MapName = locator });
                    terrain.RefreshPhaseLayers();
                });
        }

        if (ImGui.IsItemHovered())
        {
            ImGui.SetTooltip(
                "Overlay a folder of DAT terrain files as a Cartography layer. " +
                "Heights are converted from inches to yards. v22 donors show layer 0 only " +
                "until the AMAP codec is decoded. Use the layer's offset/rotation/mirror " +
                "controls to align it against the shipped tiles.");
        }

        return dirty;
    }

    private bool DrawPhaseLayerStack(TerrainManager terrain, IList<PhaseLayerSettings> layers)
    {
        bool dirty = false;
        int? removeAt = null;
        int? moveUp = null;
        int? moveDown = null;

        // Spec 232 FR-11: per-channel gates for the BASE map — drop what you don't want to keep
        // (liquids, shadows, objects, ...) instead of being forced to keep everything it carries.
        if (ImGui.CollapsingHeader("Base map channels", ImGuiTreeNodeFlags.DefaultOpen))
        {
            (string Label, PhaseDataChannel Channel)[] baseChannels =
            {
                ("Liquid", PhaseDataChannel.Liquid),
                ("Baked shadows", PhaseDataChannel.Shadows),
                ("Doodads (MDDF)", PhaseDataChannel.Doodads),
                ("World objects (MODF)", PhaseDataChannel.WorldObjects),
                ("Area ID", PhaseDataChannel.AreaId),
                ("Texturing (MCLY + MCAL)", PhaseDataChannel.TextureLayers),
                ("Vertex colours (MCCV)", PhaseDataChannel.VertexColors),
                ("Holes", PhaseDataChannel.Holes),
            };
            foreach ((string label, PhaseDataChannel channel) in baseChannels)
            {
                bool kept = terrain.BaseChannelKeep.HasFlag(channel);
                if (ImGui.Checkbox(label, ref kept))
                {
                    terrain.BaseChannelKeep = kept
                        ? terrain.BaseChannelKeep | channel
                        : terrain.BaseChannelKeep & ~channel;
                    dirty = true;
                }
            }

            if (dirty)
                ImGui.Separator();
        }

        for (int i = 0; i < layers.Count; i++)
        {
            PhaseLayerSettings layer = layers[i];
            ImGui.PushID(i);

            bool enabled = layer.Enabled;
            if (ImGui.Checkbox("##enabled", ref enabled))
            {
                layer.Enabled = enabled;
                dirty = true;
            }

            // Spec 232 FR-2: lock toggle — a locked layer keeps its alignment against accidents.
            ImGui.SameLine();
            bool locked = layer.Locked;
            if (ImGui.Checkbox("##lock", ref locked))
            {
                layer.Locked = locked;
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip(locked
                    ? "Alignment LOCKED — unlock to edit this layer."
                    : "Lock this layer's alignment against accidental edits.");

            ImGui.SameLine();
            bool expanded = ImGui.CollapsingHeader($"{i + 1}. {layer.MapName}###phaseLayerHeader");

            // Cartography (Spec 222): the expanded row is the selected row — its minimap footprint
            // draws highlighted. Keeping this on the header (not hover) makes the selection stable
            // for the drag interaction that lands next.
            int selectedIndex = _selectedPhaseLayerIndex;
            if (expanded && selectedIndex != i)
            {
                _selectedPhaseLayerIndex = i;
                if (_worldScene != null)
                    _worldScene.SelectedPhaseLayerIndex = i;
            }

            DrawPhaseLayerStatusBadge(terrain, layer);

            if (!expanded)
            {
                ImGui.PopID();
                continue;
            }

            ImGui.Indent();

            if (layer.Locked)
                ImGui.BeginDisabled();

            if (ImGui.SmallButton("Up") && i > 0)
                moveUp = i;
            ImGui.SameLine();
            if (ImGui.SmallButton("Down") && i < layers.Count - 1)
                moveDown = i;
            ImGui.SameLine();
            // Spec 232 FR-15: locked layers are protected from deletion.
            if (layer.Locked)
            {
                ImGui.BeginDisabled();
                ImGui.SmallButton("Remove");
                ImGui.EndDisabled();
                if (ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled))
                    ImGui.SetTooltip("This layer is LOCKED — unlock it before removing.");
            }
            else if (ImGui.SmallButton("Remove"))
                removeAt = i;

            bool gated = layer.OnlyTakeWhatThePhaseCarries;
            if (ImGui.Checkbox("Only take what this phase actually has", ref gated))
            {
                layer.OnlyTakeWhatThePhaseCarries = gated;
                dirty = true;
            }

            if (ImGui.IsItemHovered())
                ImGui.SetTooltip(PresenceGateTooltip);

            bool usePlacedTilesOnly = layer.UsePlacedTilesOnly;
            if (ImGui.Checkbox("Compose placed tiles only", ref usePlacedTilesOnly))
            {
                layer.UsePlacedTilesOnly = usePlacedTilesOnly;
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip(usePlacedTilesOnly
                    ? "This layer contributes only at targets selected in the donor tile picker. Turn this off to compose its whole donor map through the tile offset."
                    : "This layer composes through its tile offset except where an explicit donor-tile placement overrides a target.");

            if (layer.TilePlacements.Count > 0)
            {
                ImGui.TextUnformatted("Placed target locks");
                for (int placementIndex = 0; placementIndex < layer.TilePlacements.Count; placementIndex++)
                {
                    PhaseTilePlacement placement = layer.TilePlacements[placementIndex];
                    bool placementLocked = placement.Locked;
                    // Labels print in ADT-name order (xx_yy = column_row) to match the status bar.
                    if (ImGui.Checkbox($"Lock target {placement.TargetTileY:D2}_{placement.TargetTileX:D2}##placedLock{placementIndex}", ref placementLocked))
                    {
                        layer.TilePlacements[placementIndex] = placement with { Locked = placementLocked };
                        dirty = true;
                    }
                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("When locked, later layers cannot replace this target tile. The lock is saved in the layer project and marked L on the minimap.");
                    ImGui.SameLine();
                    ImGui.TextDisabled($"donor {placement.DonorTileY:D2}_{placement.DonorTileX:D2}");
                }
            }

            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.Heightmap, "Heightmap (MCVT)");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.Normals, "Normals (MCNR)");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.Holes, "Holes");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.TextureLayers, "Texturing (MCLY + MCAL)");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.VertexColors, "Vertex colours (MCCV)");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.Shadows, "Baked shadows (MCSH)");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.Liquid, "Liquid");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.Doodads, "Doodads (MDDF)");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.WorldObjects, "World objects (MODF)");
            dirty |= PhaseChannelCheckbox(layer, PhaseDataChannel.AreaId, "Area ID");

            if (ImGui.SmallButton("All"))
            {
                layer.Channels = PhaseDataChannel.All;
                dirty = true;
            }

            ImGui.SameLine();
            if (ImGui.SmallButton("None"))
            {
                layer.Channels = PhaseDataChannel.None;
                dirty = true;
            }

            ImGui.SameLine();
            if (ImGui.SmallButton("Objects only"))
            {
                layer.Channels = PhaseDataChannel.Objects;
                dirty = true;
            }

            ImGui.SameLine();
            if (ImGui.SmallButton("Terrain only"))
            {
                layer.Channels = PhaseDataChannel.Terrain | PhaseDataChannel.Texturing;
                dirty = true;
            }

            ImGui.Spacing();
            int offsetX = layer.TileOffsetX;
            int offsetY = layer.TileOffsetY;

            // A fixed 80px width clips the digits at larger font scales (the operator could not
            // read the values at any font size). Size the field from the actual text so the
            // numbers stay legible at every font scale, and put the label on its own line so
            // the input gets the full row width.
            float inputWidth = MathF.Max(
                ImGui.CalcTextSize("-63").X + ImGui.GetStyle().FramePadding.X * 4f
                    + ImGui.GetFontSize() * 1.5f,
                ImGui.GetFontSize() * 4f);

            ImGui.TextUnformatted("Tile offset X (row, N-S)");
            ImGui.SetNextItemWidth(inputWidth);
            if (ImGui.InputInt("##tileOffsetX", ref offsetX, 1))
            {
                layer.TileOffsetX = Math.Clamp(offsetX, -63, 63);
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Row offset (North-South, engine tileX). Positive moves the layer's content toward higher tile X.");

            ImGui.TextUnformatted("Tile offset Y (column, W-E)");
            ImGui.SetNextItemWidth(inputWidth);
            if (ImGui.InputInt("##tileOffsetY", ref offsetY, 1))
            {
                layer.TileOffsetY = Math.Clamp(offsetY, -63, 63);
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Column offset (West-East, engine tileY). Positive moves the layer's content toward higher tile Y.");

            if (layer.HasTileOffset)
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Reset offset"))
                {
                    layer.TileOffsetX = 0;
                    layer.TileOffsetY = 0;
                    dirty = true;
                }
            }

            // Spec 232 FR-1: cell-level fine-tune — 1 cell = one MCNK = 1/16 tile, for aligning
            // roadways and other content that is close but not exactly aligned after the tile
            // offset/rotation.
            ImGui.TextUnformatted("Cell fine-tune (1 cell = 1/16 tile)");
            int cellOffsetX = layer.CellOffsetX;
            ImGui.SetNextItemWidth(inputWidth);
            if (ImGui.InputInt("##cellOffsetX", ref cellOffsetX, 1))
            {
                layer.CellOffsetX = Math.Clamp(cellOffsetX, -15, 15);
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Fine nudge along the north-south axis, in terrain cells. Moves the composed content without changing which donor tile the bulk of the tile comes from.");
            ImGui.SameLine();
            int cellOffsetY = layer.CellOffsetY;
            ImGui.SetNextItemWidth(inputWidth);
            if (ImGui.InputInt("##cellOffsetY", ref cellOffsetY, 1))
            {
                layer.CellOffsetY = Math.Clamp(cellOffsetY, -15, 15);
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Fine nudge along the west-east axis, in terrain cells.");
            ImGui.SameLine();
            if (layer.HasCellOffset && ImGui.SmallButton("Reset cells"))
            {
                layer.CellOffsetX = 0;
                layer.CellOffsetY = 0;
                dirty = true;
            }

            ImGui.TextDisabled(TileOffsetNote);
            ImGui.TextDisabled(ObjectOwnershipNote);

            // Operator directive 2026-09-09: per-layer world-Z control — raise/lower (or scale)
            // a cut tile so it meets the terrain it lands beside (e.g. Teldrassil on Kalimdor).
            ImGui.TextUnformatted("Z offset / scale");
            float zOffset = layer.ZOffset;
            ImGui.SetNextItemWidth(inputWidth);
            if (ImGui.InputFloat("##layerZOffset", ref zOffset, 1f, 10f, "%.1f"))
            {
                layer.ZOffset = zOffset;
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("World-Z translation (yards) applied to this layer's terrain, liquids, and object heights. Positive raises the tile.");
            ImGui.SameLine();
            float zScale = layer.ZScale;
            ImGui.SetNextItemWidth(inputWidth);
            if (ImGui.InputFloat("##layerZScale", ref zScale, 0.01f, 0.1f, "%.3f"))
            {
                layer.ZScale = Math.Clamp(zScale, 0.01f, 100f);
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Multiplier on the layer's world-Z (1 = unchanged). Scales the terrain relief and object heights together.");
            ImGui.SameLine();
            if ((layer.ZOffset != 0f || layer.ZScale != 1f) && ImGui.SmallButton("Reset Z"))
            {
                layer.ZOffset = 0f;
                layer.ZScale = 1f;
                dirty = true;
            }

            // Spec 232 T064: magnetic WDL edge-snap — blend footprint-boundary tile edges toward
            // the base map's WDL macro lattice so a cut tile meets its neighbors without a manual
            // Z nudge.
            ImGui.TextUnformatted("WDL edge-snap (magnetic)");
            float edgeBlend = layer.EdgeBlendWdl;
            ImGui.SetNextItemWidth(inputWidth);
            if (ImGui.SliderFloat("##layerEdgeBlend", ref edgeBlend, 0f, 1f, "%.2f"))
            {
                layer.EdgeBlendWdl = edgeBlend;
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Blends this layer's tile-boundary heights toward the base map's WDL macro lattice at footprint edges (0 = off). Interior seams are untouched.");
            ImGui.SameLine();
            if (layer.EdgeBlendWdl != 0f && ImGui.SmallButton("Reset edge-snap"))
            {
                layer.EdgeBlendWdl = 0f;
                dirty = true;
            }

            // Spec 231 Phase 7: whole-layer rotation and mirror, wired through
            // PhaseComposition.ResolveTileSource + TileContentTransform by the terrain adapters.
            ImGui.TextUnformatted("Layer rotation");
            int rotationQuarter = layer.RotationDegrees switch
            {
                90f => 1,
                180f => 2,
                270f => 3,
                0f => 0,
                _ => -1,
            };
            string rotationLabel = rotationQuarter switch
            {
                1 => "90° clockwise",
                2 => "180°",
                3 => "90° counter-clockwise",
                0 => "No rotation",
                _ => $"Custom ({layer.RotationDegrees:0.#}°)",
            };
            if (ImGui.BeginCombo("##phaseLayerRotation", rotationLabel))
            {
                string[] rotationChoices = ["No rotation", "90° clockwise", "180°", "90° counter-clockwise"];
                for (int choice = 0; choice < rotationChoices.Length; choice++)
                {
                    bool selected = rotationQuarter == choice;
                    if (ImGui.Selectable(rotationChoices[choice], selected))
                    {
                        layer.RotationDegrees = choice * 90f;
                        if (choice != 0 && layer.RotationOriginTileX <= 0f && layer.RotationOriginTileY <= 0f)
                        {
                            // First transform on this layer: pivot on the donor footprint so a
                            // quarter turn about (0,0) cannot fling the layer outside the grid.
                            IReadOnlyList<(int TileX, int TileY)> rotationDonorTiles = terrain.GetLayerFootprint(layer);
                            if (rotationDonorTiles.Count > 0)
                            {
                                double avgX = 0, avgY = 0;
                                foreach ((int tileX, int tileY) in rotationDonorTiles)
                                {
                                    avgX += tileX;
                                    avgY += tileY;
                                }
                                layer.RotationOriginTileX = (float)(avgX / rotationDonorTiles.Count);
                                layer.RotationOriginTileY = (float)(avgY / rotationDonorTiles.Count);
                            }
                        }
                        if (choice == 0)
                        {
                            layer.RotationOriginTileX = 0;
                            layer.RotationOriginTileY = 0;
                        }
                        dirty = true;
                    }
                    if (selected)
                        ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Rotates this layer's whole tile grid around the rotation origin below. "
                    + "Quarter turns are exact-grid: terrain, texturing, liquids, and placements all rotate together.");

            bool mirrorLeftRight = layer.MirrorHorizontal;
            if (ImGui.Checkbox("Mirror left-right", ref mirrorLeftRight))
            {
                layer.MirrorHorizontal = mirrorLeftRight;
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Flips this layer's content along the west-east (tile Y) axis.");
            ImGui.SameLine();
            bool mirrorUpDown = layer.MirrorVertical;
            if (ImGui.Checkbox("Mirror up-down", ref mirrorUpDown))
            {
                layer.MirrorVertical = mirrorUpDown;
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Flips this layer's content along the north-south (tile X) axis.");

            if (layer.RotationDegrees != 0f || layer.MirrorHorizontal || layer.MirrorVertical)
            {
                ImGui.TextDisabled($"Rotation origin: tile ({layer.RotationOriginTileX:0.#}, {layer.RotationOriginTileY:0.#})");
                if (ImGui.SmallButton("Center origin on layer"))
                {
                    IReadOnlyList<(int TileX, int TileY)> rotationDonorTiles = terrain.GetLayerFootprint(layer);
                    if (rotationDonorTiles.Count > 0)
                    {
                        double avgX = 0, avgY = 0;
                        foreach ((int tileX, int tileY) in rotationDonorTiles)
                        {
                            avgX += tileX;
                            avgY += tileY;
                        }
                        layer.RotationOriginTileX = (float)(avgX / rotationDonorTiles.Count);
                        layer.RotationOriginTileY = (float)(avgY / rotationDonorTiles.Count);
                        dirty = true;
                    }
                }
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("Rotations pivot on this donor-map tile. Centering on the footprint keeps the layer in place while it turns.");
                ImGui.SameLine();
                if (ImGui.SmallButton("Origin at 0,0"))
                {
                    layer.RotationOriginTileX = 0;
                    layer.RotationOriginTileY = 0;
                    dirty = true;
                }
            }

            ImGui.TextDisabled(ObjectOwnershipNote);

            if (layer.Locked)
                ImGui.EndDisabled();

            ImGui.Unindent();
            ImGui.PopID();
        }

        if (moveUp is int up)
        {
            (layers[up - 1], layers[up]) = (layers[up], layers[up - 1]);
            dirty = true;
        }
        else if (moveDown is int down)
        {
            (layers[down + 1], layers[down]) = (layers[down], layers[down + 1]);
            dirty = true;
        }
        else if (removeAt is int remove)
        {
            layers.RemoveAt(remove);
            dirty = true;
        }

        if (dirty)
        {
            // Removals/reorders invalidate the index-based selection; re-sync before the minimap
            // draws the next frame so the highlight follows the right layer.
            _selectedPhaseLayerIndex = -1;
            if (_worldScene != null)
                _worldScene.SelectedPhaseLayerIndex = -1;
        }

        return dirty;
    }

    /// <summary>Cartography (Spec 222): the currently selected layer row; -1 = none.</summary>
    private int _selectedPhaseLayerIndex = -1;

    /// <summary>
    /// Cartography (Spec 222): per-row status — footprint swatch, resolution outcome, and the
    /// non-overlap warning. This is the inline state the old panel never had: a layer that
    /// contributes nothing now says so, in the row, every frame.
    /// </summary>
    private static void DrawPhaseLayerStatusBadge(TerrainManager terrain, PhaseLayerSettings layer)
    {
        int paletteIndex = layer.FootprintColorIndex >= 0
            ? layer.FootprintColorIndex % TerrainManager.FootprintPalette.Count
            : 0;
        (float r, float g, float b) = TerrainManager.FootprintPalette[paletteIndex];
        ImGui.ColorButton("##footprintSwatch", new Vector4(r, g, b, 1f), ImGuiColorEditFlags.None, new Vector2(11, 11));
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("This layer's minimap footprint color.");
        ImGui.SameLine();

        switch (layer.Resolution)
        {
            case PhaseLayerResolution.NotYetChecked:
                ImGui.TextDisabled("resolving…");
                break;

            case PhaseLayerResolution.Unresolved:
                ImGui.PushStyleColor(ImGuiCol.Text, 0xFF1A8CFF); // orange
                ImGui.TextUnformatted("map not found — the layer contributes nothing");
                ImGui.PopStyleColor();
                break;

            case PhaseLayerResolution.Resolved when terrain.IsMapWmoBased(layer.MapName):
                ImGui.TextDisabled("WMO-only map (dungeon) — no terrain tiles to compose");
                break;

            case PhaseLayerResolution.Resolved:
                IReadOnlyList<(int TileX, int TileY)> donor = terrain.GetLayerFootprint(layer);
                IReadOnlyList<(int TileX, int TileY)> baseTiles = terrain.GetBaseFootprint();

                if (layer.UsePlacedTilesOnly)
                {
                    var placedTargets = layer.TilePlacements
                        .Where(static placement => placement.IsValid)
                        .Select(static placement => (placement.TargetTileX, placement.TargetTileY))
                        .Distinct()
                        .ToList();
                    if (MapFootprint.OverlapsBase(baseTiles, placedTargets, 0, 0))
                        ImGui.TextDisabled($"{placedTargets.Count} explicitly placed tile(s) — overlapping the base map");
                    else
                        ImGui.TextDisabled($"{placedTargets.Count} explicitly placed tile(s) — none overlap the base map");
                    break;
                }

                // Spec 231 Phase 7: compose the footprint through rotation/mirror + offset so the
                // overlap check matches where the layer's tiles actually land on the base map.
                // Targets outside the 64x64 grid are dropped (grid confinement).
                var donorTarget = new List<(int TileX, int TileY)>(donor.Count);
                foreach ((int donorTileX, int donorTileY) in donor)
                {
                    (int tx, int ty) = PhaseCompositionPolicy.ForwardTransformTile(donorTileX, donorTileY, layer);
                    tx += layer.TileOffsetX;
                    ty += layer.TileOffsetY;
                    if (tx < 0 || tx > 63 || ty < 0 || ty > 63)
                        continue;
                    donorTarget.Add((tx, ty));
                }

                if (MapFootprint.OverlapsBase(baseTiles, donorTarget, 0, 0))
                {
                    ImGui.TextDisabled($"{donor.Count} tiles — overlapping the base map");
                }
                else
                {
                    ImGui.PushStyleColor(ImGuiCol.Text, 0xFF1A8CFF); // orange
                    ImGui.TextUnformatted(
                        $"{donor.Count} tiles — none overlap the base map; set an offset, rotate, or align to see anything");
                    ImGui.PopStyleColor();
                }
 
                break;
        }
    }

    private const string PresenceGateTooltip =
        "On (recommended): a channel is taken only where the phase really carries it, so the phase's "
        + "blank chunks cannot erase good base terrain or texturing.\n\n"
        + "Off: this phase is authoritative for every channel ticked below, including ones it is "
        + "empty for. That is how you make a phase deliberately CLEAR something, and is destructive "
        + "otherwise.";

    private const string TileOffsetNote =
        "Shifts this layer in tile space, for maps authored at different coordinates than the map "
        + "they overlay - instance and dungeon maps are often copies of an earlier revision of a "
        + "zone stored elsewhere in the grid. Terrain and its placements move together.";

    private const string ObjectOwnershipNote =
        "Objects are presence-gated replace: a phase that ships placements owns them for that tile; "
        + "one that ships none leaves the base map's alone.";

    /// <summary>One channel checkbox. Returns true when the selection changed.</summary>
    private static bool PhaseChannelCheckbox(PhaseLayerSettings layer, PhaseDataChannel channel, string label)
    {
        bool on = (layer.Channels & channel) != 0;
        if (!ImGui.Checkbox(label, ref on))
            return false;

        layer.Channels = on ? layer.Channels | channel : layer.Channels & ~channel;
        return true;
    }

    // ── Cartography destination under Archaeology (Spec 223 / Spec 222) ───────

    private int _cartographySubTab = 0;
    private string _cartographyDonorMap = string.Empty;
    private int _cartographyDonorTileX = 32;
    private int _cartographyDonorTileY = 32;
    private int _cartographyTargetTileX = 32;
    private int _cartographyTargetTileY = 32;
    private string? _cartographySaveStatus = null;

    /// <summary>
    /// Cartography workbench destination under Archaeology (Spec 223-T302 / Spec 222).
    /// Houses the phase map layer stack, synthesized minimap export, and save-merged-output controls.
    /// </summary>
    private void DrawArchaeologyCartographyContent()
    {
        TerrainManager? terrain = _terrainManager;
        if (terrain == null)
        {
            ImGui.TextDisabled("Load a terrain-backed map to use Cartography.");
            return;
        }

        string[] subTabs = ["Layers", "Synthesized Minimap", "Save Merged Output"];
        for (int i = 0; i < subTabs.Length; i++)
        {
            if (i > 0)
                ImGui.SameLine();

            bool isSelected = _cartographySubTab == i;
            if (isSelected)
                ImGui.PushStyleColor(ImGuiCol.Button, new Vector4(0.26f, 0.59f, 0.98f, 0.8f));

            if (ImGui.Button(subTabs[i]))
                _cartographySubTab = i;

            if (isSelected)
                ImGui.PopStyleColor();
        }

        ImGui.Separator();

        switch (_cartographySubTab)
        {
            case 0:
                DrawCartographyLayersSubTab(terrain);
                break;
            case 1:
                DrawSynthesizedMinimapExportContent(showCloseButton: false);
                break;
            case 2:
                DrawCartographySaveMergedSubTab(terrain);
                break;
        }
    }

    private void DrawCartographyLayersSubTab(TerrainManager terrain)
    {
        ImGui.TextWrapped(
            "Stack phase maps and donor tiles over the base map. Layers apply in order from top to bottom; later layers win on shared channels.");

        IList<PhaseLayerSettings> layers = terrain.PhaseLayers;
        bool dirty = false;

        // 1. Layer Stack (Spec 222-T109)
        if (layers.Count > 0)
        {
            dirty |= DrawPhaseLayerStack(terrain, layers);
            if (ImGui.Button("Clear all layers"))
            {
                // Spec 232 FR-15: locked layers are protected from deletion — clear everything
                // else and tell the operator what was kept.
                int removed = 0;
                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    if (layers[i].Locked)
                        continue;
                    layers.RemoveAt(i);
                    removed++;
                }
                dirty |= removed > 0;
                if (removed > 0)
                    _statusMessage = $"Cleared {removed} unlocked layer(s); locked layer(s) were kept — unlock them to remove.";
            }

            // Spec 232 FR-7: tiles do not always load everything after a placement or alignment
            // change — this forces every tile to evict and re-stream under the current stack.
            ImGui.SameLine();
            if (ImGui.Button("Force reload tiles"))
            {
                terrain.RefreshPhaseLayers();
                dirty = true;
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Evicts and re-streams every tile under the current layer stack. Use when terrain, objects, or textures did not load after a change.");

            ImGui.Separator();
        }

        // Spec 232 Phase 2 (FR-2/FR-8): layer-project persistence — Save writes this map's stack
        // (auto-loaded on launch), Load replaces the live stack with the saved one.
        ImGui.TextUnformatted("Layer project");
        if (ImGui.Button("Save project"))
        {
            try
            {
                terrain.SaveLayerProject();
                _statusMessage = $"Layer project saved for '{terrain.MapName}'. It reloads automatically on launch.";
            }
            catch (Exception ex)
            {
                _statusMessage = $"Layer project save failed: {ex.Message}";
            }
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Writes the current layer stack (offsets, rotation, mirrors, channels, locks) as this map's layer project. Auto-loads on every launch.");
        ImGui.SameLine();
        if (ImGui.Button("Load project"))
        {
            try
            {
                bool loaded = terrain.LoadLayerProject();
                _statusMessage = loaded
                    ? $"Layer project loaded for '{terrain.MapName}'."
                    : $"No saved layer project exists for '{terrain.MapName}'.";
                _selectedPhaseLayerIndex = -1;
                if (_worldScene != null)
                    _worldScene.SelectedPhaseLayerIndex = -1;
            }
            catch (Exception ex)
            {
                _statusMessage = $"Layer project load failed: {ex.Message}";
            }
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Replaces the live layer stack with this map's saved project.");
        ImGui.Separator();

        // 2. DBC Phase Discovery (Child maps from Map.dbc) - Spec 222-T110
        dirty |= DrawDbcPhaseDiscovery(terrain, layers);

        // 3. Add Phase Map Controls
        dirty |= DrawPhaseLayerAddControls(terrain, layers);

        // 4. Donor Tile-Grid Picker (Single-Tile Placement) - Spec 222-T108
        dirty |= DrawDonorTileGridPicker(terrain, layers);

        if (dirty)
            terrain.RefreshPhaseLayers();
    }

    private bool DrawDonorTileGridPicker(TerrainManager terrain, IList<PhaseLayerSettings> layers)
    {
        if (!ImGui.CollapsingHeader("Donor Tile Grid Picker (Single-Tile Placement)"))
            return false;

        ImGui.TextWrapped("Browse a donor map's 64x64 grid and place a specific tile at a chosen target position (Spec 222-T108).");

        bool dirty = false;
        var availableMaps = _discoveredMaps;

        if (availableMaps.Count > 0 && ImGui.BeginCombo("Donor Map##donorGrid", string.IsNullOrWhiteSpace(_cartographyDonorMap) ? "Select donor map..." : _cartographyDonorMap))
        {
            foreach (var m in availableMaps)
            {
                bool isSelected = string.Equals(m.Directory, _cartographyDonorMap, StringComparison.OrdinalIgnoreCase);
                string label = m.HasDbcEntry ? $"[{m.Id:D3}] {m.Name} ({m.Directory})" : $"[custom] {m.Name} ({m.Directory})";
                if (ImGui.Selectable(label, isSelected))
                    _cartographyDonorMap = m.Directory;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        if (!string.IsNullOrWhiteSpace(_cartographyDonorMap))
        {
            // Coordinate order (2026-09-09 operator report: "it doesn't put the tile at the target
            // xx_yy location"): the operator enters tiles in ADT-NAME order (xx = column W-E, first
            // number of e.g. "Azeroth_28_50"; yy = row N-S). The composition grid is (tileX = row,
            // tileY = column), so the picker maps xx -> tileY and yy -> tileX. Entering both numbers
            // swapped used to land the tile on the diagonal mirror of the requested location.
            ImGui.TextWrapped("Tiles are named xx_yy (xx = column W-E, yy = row N-S), matching the status bar.");
            ImGui.Text("Donor Tile:");
            ImGui.SetNextItemWidth(100);
            ImGui.InputInt("xx (col)##donorX", ref _cartographyDonorTileX);
            ImGui.SameLine();
            ImGui.SetNextItemWidth(100);
            ImGui.InputInt("yy (row)##donorY", ref _cartographyDonorTileY);
            _cartographyDonorTileX = Math.Clamp(_cartographyDonorTileX, 0, 63);
            _cartographyDonorTileY = Math.Clamp(_cartographyDonorTileY, 0, 63);

            ImGui.Text("Target Tile:");
            ImGui.SetNextItemWidth(100);
            ImGui.InputInt("xx (col)##targetX", ref _cartographyTargetTileX);
            ImGui.SameLine();
            ImGui.SetNextItemWidth(100);
            ImGui.InputInt("yy (row)##targetY", ref _cartographyTargetTileY);
            _cartographyTargetTileX = Math.Clamp(_cartographyTargetTileX, 0, 63);
            _cartographyTargetTileY = Math.Clamp(_cartographyTargetTileY, 0, 63);

            ImGui.TextDisabled($"Placement: {_cartographyDonorTileX:D2}_{_cartographyDonorTileY:D2} -> "
                + $"{_cartographyTargetTileX:D2}_{_cartographyTargetTileY:D2}");

            if (ImGui.Button("Add Single Tile Placement"))
            {
                var layer = new PhaseLayerSettings
                {
                    MapName = _cartographyDonorMap,
                    // Internal offsets follow the (tileX = row, tileY = column) grid.
                    TileOffsetX = _cartographyTargetTileY - _cartographyDonorTileY,
                    TileOffsetY = _cartographyTargetTileX - _cartographyDonorTileX,
                    UsePlacedTilesOnly = true,
                    Channels = PhaseDataChannel.All,
                    OnlyTakeWhatThePhaseCarries = true,
                };
                layer.TilePlacements.Add(new PhaseTilePlacement(
                    _cartographyDonorTileY,
                    _cartographyDonorTileX,
                    _cartographyTargetTileY,
                    _cartographyTargetTileX));
                layers.Add(layer);
                dirty = true;
            }
        }

        ImGui.Separator();
        return dirty;
    }

    private void DrawCartographySaveMergedSubTab(TerrainManager terrain)
    {
        ImGui.TextWrapped("Save and export merged composition data, phase manifests, and terrain outputs.");
        ImGui.Separator();

        ImGui.Text("Composition Summary");
        ImGui.TextDisabled($"Base Map: {terrain.MapName ?? "none"}");
        ImGui.TextDisabled($"Active Phase Layers: {terrain.PhaseLayers.Count(l => l.Enabled)} of {terrain.PhaseLayers.Count}");

        foreach (var layer in terrain.PhaseLayers)
        {
            if (!layer.Enabled)
                continue;
            ImGui.BulletText($"{layer.MapName} (offset X:{layer.TileOffsetX}, Y:{layer.TileOffsetY}) - channels: {layer.Channels}");
        }

        ImGui.Separator();
        ImGui.Text("Project Output");
        ImGui.TextWrapped($"Folder: {_projectOutput.DescribeEditorProjectOutputDirectory()}");
        if (ImGui.Button("New Project Folder##cartography"))
            _projectOutput.StartNewEditorProjectOutputDirectory();

        ImGui.SameLine();
        if (ImGui.Button("Save Composition Manifest"))
        {
            try
            {
                string dir = _projectOutput.EnsureEditorProjectOutputDirectory();
                string manifestPath = Path.Combine(dir, "cartography-manifest.json");
                var manifestData = new
                {
                    baseMap = terrain.MapName,
                    createdAt = DateTime.UtcNow.ToString("o"),
                    layers = terrain.PhaseLayers.Select(l => new
                    {
                        mapName = l.MapName,
                        enabled = l.Enabled,
                        offsetX = l.TileOffsetX,
                        offsetY = l.TileOffsetY,
                        channels = l.Channels.ToString(),
                        onlyTakeWhatPhaseCarries = l.OnlyTakeWhatThePhaseCarries
                    })
                };
                File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifestData, new JsonSerializerOptions { WriteIndented = true }));
                _cartographySaveStatus = $"Manifest saved: {manifestPath}";
            }
            catch (Exception ex)
            {
                _cartographySaveStatus = $"Error saving manifest: {ex.Message}";
            }
        }

        if (!string.IsNullOrEmpty(_cartographySaveStatus))
        {
            ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(0.35f, 0.9f, 0.45f, 1f));
            ImGui.TextWrapped(_cartographySaveStatus);
            ImGui.PopStyleColor();
        }

        ImGui.Separator();
        ImGui.Text("Export Composed Outputs");
        if (ImGui.Button("Export Composed Terrain Alpha"))
        {
            _terrainExportKind = TerrainExportKind.AlphaCurrentTileAtlas;
            _wantTerrainExport = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Export Composed Heightmaps"))
        {
            _terrainExportKind = TerrainExportKind.Heightmap257CurrentTilePerTile;
            _wantTerrainExport = true;
        }

        if (ImGui.Button("Export Composed GLB Scene"))
            _wantExportGlb = true;
        ImGui.SameLine();
        if (ImGui.Button("Export Composed GLB Collision"))
            _wantExportGlbCollision = true;
    }
}
