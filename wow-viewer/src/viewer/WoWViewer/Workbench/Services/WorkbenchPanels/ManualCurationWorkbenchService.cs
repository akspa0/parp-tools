using System.Security.Cryptography;
using System.Numerics;
using System.Text;
using ImGuiNET;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.Runtime.World;
using WoWViewer;
using WoWViewer.Catalog;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer.Workbench.Services.WorkbenchPanels;

/// <summary>
/// In-memory owner for manual placement snapshots and explicit object↔PM4 associations.
/// Persistence is provided by the curation-project store in the following implementation phase.
/// </summary>
internal sealed class ManualCurationWorkbenchService
{
    private readonly Dictionary<string, ManualCurationProject> _projects = new(StringComparer.OrdinalIgnoreCase);

    public static string NormalizeMapName(string? mapName)
    {
        if (string.IsNullOrWhiteSpace(mapName))
            return "development";
        string clean = Path.GetFileNameWithoutExtension(mapName).Trim();
        return string.IsNullOrWhiteSpace(clean) ? "development" : clean;
    }

    public ManualCurationProject GetProject(string? mapName)
    {
        string key = NormalizeMapName(mapName);
        if (!_projects.TryGetValue(key, out ManualCurationProject? project))
        {
            string path = ProjectPath(key);
            project = File.Exists(path)
                ? ManualCurationProjectFile.Load(path)
                : new ManualCurationProject { MapName = key };
            _projects.Add(key, project);
        }
        return project;
    }

    public string SaveProject(string? mapName)
    {
        string key = NormalizeMapName(mapName);
        string path = ProjectPath(key);
        ManualCurationProjectFile.Save(path, GetProject(key));
        return path;
    }

    public ManualCurationProject ReloadProject(string? mapName)
    {
        string key = NormalizeMapName(mapName);
        string path = ProjectPath(key);
        ManualCurationProject project = File.Exists(path)
            ? ManualCurationProjectFile.Load(path)
            : new ManualCurationProject { MapName = key };
        _projects[key] = project;
        return project;
    }

    public bool TryRecordPlacement(
        string? mapName,
        ObjectInstance instance,
        bool createdFromAssetCatalog,
        out Guid placementId)
    {
        placementId = Guid.Empty;
        if (!instance.HasTileCoordinate || instance.TileX is < 0 or >= 64 || instance.TileY is < 0 or >= 64 ||
            string.IsNullOrWhiteSpace(instance.ModelPath))
        {
            return false;
        }

        string normalizedMap = NormalizeMapName(mapName);
        createdFromAssetCatalog |= instance.UniqueId >= Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase;
        ManualCurationProject project = GetProject(normalizedMap);
        ManualObjectPlacement? catalogMatch = createdFromAssetCatalog
            ? project.Placements.FirstOrDefault(placement => placement.CreatedFromAssetCatalog &&
                placement.TileX == instance.TileX && placement.TileY == instance.TileY &&
                placement.PlacementEntryIndex == instance.PlacementEntryIndex &&
                string.Equals(placement.ModelPath, instance.ModelPath, StringComparison.OrdinalIgnoreCase))
            : null;
        Guid stablePlacementId = catalogMatch?.PlacementId ?? CreatePlacementId(normalizedMap, instance);
        placementId = stablePlacementId;
        ManualObjectPlacement snapshot = ToSnapshot(stablePlacementId, instance, createdFromAssetCatalog);
        int existingIndex = project.Placements.FindIndex(placement => placement.PlacementId == stablePlacementId);
        if (existingIndex >= 0)
            ManualCurationProjectPolicy.TryUpdatePlacement(project, snapshot);
        else if (!ManualCurationProjectPolicy.TryAddPlacement(project, snapshot))
            return false;

        return true;
    }

    public ManualAssociationResult TryPair(
        string? mapName,
        ObjectInstance instance,
        Pm4HullIdentity hull,
        bool createdFromAssetCatalog = false)
    {
        if (!TryRecordPlacement(mapName, instance, createdFromAssetCatalog, out Guid placementId))
            return ManualAssociationResult.PlacementNotFound;

        return ManualCurationProjectPolicy.TryAddAssociation(GetProject(mapName), placementId, hull);
    }

    public bool TryPlaceCatalogAsset(
        string? mapName,
        AssetCatalogEntry catalogEntry,
        Pm4HullIdentity hull,
        Vector3 position,
        out Guid placementId)
    {
        ArgumentNullException.ThrowIfNull(catalogEntry);
        placementId = Guid.Empty;
        if (!hull.IsValid || string.IsNullOrWhiteSpace(catalogEntry.ModelPath) ||
            !float.IsFinite(position.X) || !float.IsFinite(position.Y) || !float.IsFinite(position.Z))
        {
            return false;
        }

        string normalizedMap = NormalizeMapName(mapName);
        ManualCurationProject project = GetProject(normalizedMap);
        int syntheticId = Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase + project.Placements.Count + 1;
        while (project.Placements.Any(placement => placement.UniqueId == syntheticId))
            syntheticId++;

        string assetKind = catalogEntry.IsWmo
            ? "WMO"
            : (Path.GetExtension(catalogEntry.ModelPath).Equals(".mdx", StringComparison.OrdinalIgnoreCase) ? "MDX" : "M2");
        float scale = float.IsFinite(catalogEntry.EffectiveScale) && catalogEntry.EffectiveScale > 0f
            ? catalogEntry.EffectiveScale
            : 1f;
        var placement = new ManualObjectPlacement(
            Guid.NewGuid(),
            catalogEntry.ModelPath,
            assetKind,
            catalogEntry.Name,
            hull.TileX,
            hull.TileY,
            PlacementEntryIndex: -1,
            UniqueId: syntheticId,
            Position: position,
            RotationDegrees: Vector3.Zero,
            Scale: scale,
            CreatedFromAssetCatalog: true,
            AuthoredUtc: DateTimeOffset.UtcNow);

        if (!ManualCurationProjectPolicy.TryAddPlacement(project, placement))
            return false;
        placementId = placement.PlacementId;
        return true;
    }

    public ManualAssociationResult RemoveAssociation(string? mapName, Guid placementId, Pm4HullIdentity hull)
        => ManualCurationProjectPolicy.TryRemoveAssociation(GetProject(mapName), placementId, hull);

    public ManualAssociationResult ReplaceAssociation(
        string? mapName,
        Guid placementId,
        Pm4HullIdentity oldHull,
        Pm4HullIdentity newHull)
        => ManualCurationProjectPolicy.TryReplaceAssociation(GetProject(mapName), placementId, oldHull, newHull);

    public void ApplyProjectToScene(string? mapName, WorldScene worldScene)
    {
        ArgumentNullException.ThrowIfNull(worldScene);
        ManualCurationProject project = GetProject(mapName);
        IPm4OverlayHost pm4Host = worldScene;
        bool changed = false;

        for (int placementIndex = 0; placementIndex < project.Placements.Count; placementIndex++)
        {
            ManualObjectPlacement placement = project.Placements[placementIndex];
            if (!TryApplyPlacement(pm4Host, project, placementIndex, placement, out bool placementChanged))
                continue;

            changed |= placementChanged;
        }

        if (changed)
            pm4Host.RebuildInstanceLists();
    }

    public void DrawPlacementEditor(string? mapName, IViewerAppHost host, WorldScene worldScene, ObjectInstance instance)
    {
        if (!SharedUiWidgets.SectionHeader("Manual object placement", "Edit the selected object in the live scene and record the curation snapshot. These actions do not write source map files.", defaultOpen: false, id: "manual-object-placement"))
            return;

        float x = instance.PlacementPosition.X;
        float y = instance.PlacementPosition.Y;
        float z = instance.PlacementPosition.Z;
        SharedUiWidgets.SetNextItemWidth(125f);
        ImGui.InputFloat("Position X##manualPlacementX", ref x, 1f, 10f, "%.3f");
        ImGui.SameLine();
        SharedUiWidgets.SetNextItemWidth(125f);
        ImGui.InputFloat("Position Y##manualPlacementY", ref y, 1f, 10f, "%.3f");
        SharedUiWidgets.SetNextItemWidth(125f);
        ImGui.InputFloat("Position Z##manualPlacementZ", ref z, 1f, 10f, "%.3f");
        if (SharedUiWidgets.ActionButton("Apply manual position##manualPlacementApply"))
        {
            Vector3 position = new(x, y, z);
            string error = string.Empty;
            bool updatedInScene = float.IsFinite(x) && float.IsFinite(y) && float.IsFinite(z) &&
                worldScene.Selection.TryUpdateSelectedPlacementPosition(position, out error);
            if (updatedInScene)
            {
                ObjectInstance updated = instance;
                updated.PlacementPosition = position;
                updated.Transform = WorldPlacementTransform.Build(position, updated.PlacementRotation, updated.PlacementScale);
                TryRecordPlacement(mapName, updated, createdFromAssetCatalog: false, out _);
            }
            else
            {
                host.StatusMessage = string.IsNullOrEmpty(error) ? "Position values must be finite." : error;
            }
        }

        float yaw = instance.PlacementRotation.Z;
        SharedUiWidgets.SetNextItemWidth(150f);
        ImGui.InputFloat("Yaw angle (degrees)##manualPlacementYaw", ref yaw, 1f, 15f, "%.3f");
        if (SharedUiWidgets.ActionButton("Apply manual yaw##manualPlacementYawApply"))
        {
            float normalizedYaw = NormalizeYaw(yaw);
            if (float.IsFinite(yaw) && host.Pm4DoodadDuplication.RotatePlacement(instance, normalizedYaw - instance.PlacementRotation.Z))
            {
                ObjectInstance updated = instance;
                updated.PlacementRotation.Z = normalizedYaw;
                updated.Transform = WorldPlacementTransform.Build(updated.PlacementPosition, updated.PlacementRotation, updated.PlacementScale);
                TryRecordPlacement(mapName, updated, createdFromAssetCatalog: false, out _);
            }
            else
            {
                host.StatusMessage = "Yaw must be finite and the selected placement must still be resident.";
            }
        }

        if (SharedUiWidgets.ActionButton("Record selected placement##manualPlacementRecord"))
        {
            host.StatusMessage = TryRecordPlacement(mapName, instance, createdFromAssetCatalog: false, out Guid id)
                ? $"Recorded manual curation placement {id:N}. Source map files were not written."
                : "Selected object is not a resident tile placement with a model path.";
        }
    }

    public void DrawAssociationEditor(
        string? mapName,
        ObjectInstance? selectedObject,
        Pm4HullIdentity? selectedHull,
        AssetCatalogEntry? selectedCatalogEntry = null,
        Vector3? selectedHullCenter = null,
        string? selectedHullLabel = null)
    {
        if (!SharedUiWidgets.SectionHeader("Manual object to PM4 hull curation", "Create explicit, reviewable object-to-hull associations. Duplicate pairs are reported, and remove or replace actions are explicit.", defaultOpen: true, id: "manual-object-hull-curation"))
            return;

        ManualCurationProject project = GetProject(mapName);
        if (SharedUiWidgets.ActionButton("Save manual curation project##manualCurationSave"))
        {
            try
            {
                string savedPath = SaveProject(mapName);
                SharedUiWidgets.SuccessText($"Saved curation-only project: {savedPath}");
            }
            catch (Exception exception)
            {
                SharedUiWidgets.BodyTextWrapped($"Curation project save failed: {exception.Message}");
            }
        }
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Reload manual curation project##manualCurationReload"))
        {
            try
            {
                project = ReloadProject(mapName);
                SharedUiWidgets.SuccessText($"Reloaded {project.Placements.Count} placement(s) and {project.Associations.Count} association(s).");
            }
            catch (Exception exception)
            {
                SharedUiWidgets.BodyTextWrapped($"Curation project load failed: {exception.Message}");
            }
        }

        Guid selectedPlacementId = Guid.Empty;
        if (selectedObject.HasValue && selectedObject.Value.HasTileCoordinate)
            selectedPlacementId = CreatePlacementId(NormalizeMapName(mapName), selectedObject.Value);

        if (selectedObject.HasValue && selectedHull.HasValue)
        {
            ObjectInstance instance = selectedObject.Value;
            SharedUiWidgets.BodyText($"Object: {instance.ModelName} ({instance.ModelPath})", muted: true);
            SharedUiWidgets.BodyText(selectedHullLabel ?? FormatHull(selectedHull.Value), muted: true);
            if (SharedUiWidgets.ActionButton("Pair selected object with hull##manualPairCreate"))
            {
                ManualAssociationResult result = TryPair(mapName, instance, selectedHull.Value);
                SharedUiWidgets.BodyText(DescribeResult(result), muted: result != ManualAssociationResult.Added);
            }
        }
        else
        {
            SharedUiWidgets.BodyText("Select one scene object and one PM4 hull to create an explicit pair.", muted: true);
        }

        if (selectedHull.HasValue && selectedHullCenter.HasValue && selectedCatalogEntry?.ModelPath is not null)
        {
            SharedUiWidgets.BodyText(
                $"Client catalog selection: {selectedCatalogEntry.Name} — {selectedCatalogEntry.ModelPath}",
                muted: true);
            if (SharedUiWidgets.ActionButton("Place selected catalog model on hull tile##manualCatalogPlacement"))
            {
                bool added = TryPlaceCatalogAsset(
                    mapName, selectedCatalogEntry, selectedHull.Value, selectedHullCenter.Value, out Guid placementId)
                    ;
                SharedUiWidgets.BodyText(added
                    ? $"Created curation-only placement {placementId:N}. Select it in the scene and explicitly pair it with the hull."
                    : "Could not create the catalog placement. Check the selected entry and hull tile.",
                    muted: !added);
            }
        }

        if (project.Associations.Count == 0)
        {
            SharedUiWidgets.BodyText("No manual associations recorded for this map.", muted: true);
            return;
        }

        ImGui.SeparatorText($"Curated associations ({project.Associations.Count})");
        for (int index = 0; index < project.Associations.Count; index++)
        {
            ManualPm4Association association = project.Associations[index];
            ManualObjectPlacement? placement = project.Placements.FirstOrDefault(item => item.PlacementId == association.PlacementId);
            ImGui.PushID(index);
            SharedUiWidgets.BodyText(placement is null
                ? $"Missing object record {association.PlacementId:N} → {FormatHull(association.Hull)}"
                : $"{placement.ModelName} ({placement.ModelPath}) → {FormatHull(association.Hull)}");

            if (selectedPlacementId == association.PlacementId && selectedHull.HasValue && selectedHull.Value != association.Hull)
            {
                if (SharedUiWidgets.ActionButton("Replace with selected hull##manualPairReplace"))
                {
                    ManualAssociationResult result = ReplaceAssociation(mapName, association.PlacementId, association.Hull, selectedHull.Value);
                    SharedUiWidgets.BodyText(DescribeResult(result), muted: result != ManualAssociationResult.Replaced);
                }
            }

            if (SharedUiWidgets.ActionButton("Remove association##manualPairRemove"))
            {
                ManualAssociationResult result = RemoveAssociation(mapName, association.PlacementId, association.Hull);
                SharedUiWidgets.BodyText(DescribeResult(result), muted: result != ManualAssociationResult.Removed);
                ImGui.PopID();
                break;
            }
            ImGui.PopID();
        }

        if (selectedHull.HasValue)
        {
            foreach (ManualObjectPlacement placement in project.Placements.Where(placement =>
                         !project.Associations.Any(association => association.PlacementId == placement.PlacementId && association.Hull == selectedHull.Value)))
            {
                ImGui.PushID(placement.PlacementId.ToString("N"));
                SharedUiWidgets.BodyText($"Unpaired placement: {placement.ModelName} ({placement.ModelPath})", muted: true);
                if (SharedUiWidgets.ActionButton("Pair saved placement with selected hull##manualPairPlacement"))
                {
                    ManualAssociationResult result = ManualCurationProjectPolicy.TryAddAssociation(
                        project, placement.PlacementId, selectedHull.Value);
                    SharedUiWidgets.BodyText(DescribeResult(result), muted: result != ManualAssociationResult.Added);
                }
                ImGui.PopID();
            }
        }
    }

    private static ManualObjectPlacement ToSnapshot(Guid placementId, ObjectInstance instance, bool createdFromAssetCatalog)
        => new(
            placementId,
            instance.ModelPath,
            instance.AssetKind,
            instance.ModelName,
            instance.TileX,
            instance.TileY,
            instance.PlacementEntryIndex,
            instance.UniqueId,
            instance.PlacementPosition,
            instance.PlacementRotation,
            instance.PlacementScale,
            createdFromAssetCatalog,
            DateTimeOffset.UtcNow);

    private static bool TryApplyPlacement(
        IPm4OverlayHost host,
        ManualCurationProject project,
        int projectPlacementIndex,
        ManualObjectPlacement placement,
        out bool changed)
    {
        changed = false;
        bool isWmo = Pm4DoodadDuplicationService.IsWmoAsset(placement.ModelPath, placement.AssetKind);
        Dictionary<(int, int), List<ObjectInstance>> tiles = isWmo ? host.TileWmoInstances : host.TileMdxInstances;
        if (!tiles.TryGetValue((placement.TileX, placement.TileY), out List<ObjectInstance>? instances))
        {
            if (!placement.CreatedFromAssetCatalog)
                return false;
            instances = [];
            tiles.Add((placement.TileX, placement.TileY), instances);
        }

        int index = instances.FindIndex(instance => instance.PlacementEntryIndex == placement.PlacementEntryIndex &&
            string.Equals(instance.ModelPath, placement.ModelPath, StringComparison.OrdinalIgnoreCase));
        if (index < 0 && placement.CreatedFromAssetCatalog)
        {
            index = instances.FindIndex(instance => instance.UniqueId == placement.UniqueId &&
                string.Equals(instance.ModelPath, placement.ModelPath, StringComparison.OrdinalIgnoreCase));
        }

        if (index < 0 && placement.CreatedFromAssetCatalog)
        {
            int uniqueId = placement.UniqueId;
            while (ContainsUniqueId(host, uniqueId))
                uniqueId++;

            Matrix4x4 createdTransform = WorldPlacementTransform.Build(
                placement.Position, placement.RotationDegrees, placement.Scale);
            var created = new ObjectInstance
            {
                ModelKey = WorldAssetManager.NormalizeKey(placement.ModelPath),
                ModelPath = placement.ModelPath,
                ModelName = placement.ModelName,
                AssetKind = placement.AssetKind,
                PlacementPosition = placement.Position,
                PlacementRotation = placement.RotationDegrees,
                PlacementScale = placement.Scale,
                Transform = createdTransform,
                BoundsMin = placement.Position,
                BoundsMax = placement.Position,
                LocalBoundsMin = Vector3.Zero,
                LocalBoundsMax = Vector3.Zero,
                BoundsResolved = false,
                UniqueId = uniqueId,
                PlacementEntryIndex = instances.Count,
                TileX = placement.TileX,
                TileY = placement.TileY,
                HasTileCoordinate = true,
                HasOpaqueRenderContent = true,
            };
            instances.Add(created);
            project.Placements[projectPlacementIndex] = placement with
            {
                UniqueId = created.UniqueId,
                PlacementEntryIndex = created.PlacementEntryIndex,
            };
            if (isWmo)
            {
                host.Assets.QueueWmoLoad(created.ModelKey);
                host.Assets.PrioritizeWmoLoad(created.ModelKey);
            }
            else
            {
                host.Assets.QueueMdxLoad(created.ModelKey);
                host.Assets.PrioritizeMdxLoad(created.ModelKey);
            }
            changed = true;
            return true;
        }

        if (index < 0)
            return false;

        ObjectInstance current = instances[index];
        changed = current.PlacementPosition != placement.Position ||
            current.PlacementRotation != placement.RotationDegrees ||
            MathF.Abs(current.PlacementScale - placement.Scale) > 1e-6f;
        if (!changed)
            return true;

        current.PlacementPosition = placement.Position;
        current.PlacementRotation = placement.RotationDegrees;
        current.PlacementScale = placement.Scale;
        current.Transform = WorldPlacementTransform.Build(current.PlacementPosition, current.PlacementRotation, current.PlacementScale);
        WorldScene.TransformBounds(current.LocalBoundsMin, current.LocalBoundsMax, current.Transform,
            out current.BoundsMin, out current.BoundsMax);
        instances[index] = current;
        return true;
    }

    private static bool ContainsUniqueId(IPm4OverlayHost host, int uniqueId)
        => host.TileMdxInstances.Values.Any(list => list.Any(instance => instance.UniqueId == uniqueId)) ||
           host.TileWmoInstances.Values.Any(list => list.Any(instance => instance.UniqueId == uniqueId));

    private static float NormalizeYaw(float yaw)
    {
        float normalized = yaw % 360f;
        if (normalized > 180f) normalized -= 360f;
        if (normalized < -180f) normalized += 360f;
        return normalized == 0f ? 0f : normalized;
    }

    private static Guid CreatePlacementId(string mapName, ObjectInstance instance)
    {
        string identity = string.Join('|', mapName, instance.TileX, instance.TileY, instance.PlacementEntryIndex,
            instance.UniqueId, instance.AssetKind, instance.ModelPath.ToUpperInvariant());
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(identity));
        return new Guid(digest.AsSpan(0, 16));
    }

    private static string FormatHull(Pm4HullIdentity hull)
    {
        float z = hull.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(hull.Ck24 << 8) : 0f;
        return $"[{hull.TileX:D2}_{hull.TileY:D2}] Obj #{hull.ObjectPartId} (Z={z:F1}yd, Lnk=0x{hull.LinkGroupObjectId:X})";
    }

    private static string ProjectPath(string mapName)
        => Path.Combine(AppContext.BaseDirectory, "output", "projects", "manual_curation", $"{mapName}.manual-curation.json");

    private static string DescribeResult(ManualAssociationResult result) => result switch
    {
        ManualAssociationResult.Added => "Manual association recorded.",
        ManualAssociationResult.Removed => "Manual association removed.",
        ManualAssociationResult.Replaced => "Manual association explicitly replaced.",
        ManualAssociationResult.AlreadyExists => "This exact object↔hull relation already exists.",
        ManualAssociationResult.PlacementNotFound => "The object does not have a valid tile-scoped placement snapshot.",
        ManualAssociationResult.InvalidHull => "The selected PM4 hull identity is invalid.",
        _ => "The requested manual association was not found.",
    };
}
