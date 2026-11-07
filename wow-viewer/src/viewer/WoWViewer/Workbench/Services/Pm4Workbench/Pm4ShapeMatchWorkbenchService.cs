using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.Runtime.World;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using WoWViewer.Workbench.Services.WorkbenchPanels;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer.Workbench.Services.Pm4Workbench;

public sealed record Pm4ProposalPreviewState(
    Pm4HullIdentity Hull,
    string ModelPath,
    string AssetKind,
    Vector3 UncalibratedCentroid,
    Vector3 CalibratedCentroid,
    Pm4CentroidCalibration AppliedCalibration,
    int? SpawnedPreviewUniqueId);

/// <summary>
/// Spec 295 Phase 4 (T014, T015): Coordinates the visual shape library, calibrated -X offset
/// state (apply-once, invert, reset), automatic shape-match proposals, 3D viewport preview,
/// human-in-the-loop review queue, and batch analysis.
/// </summary>
internal sealed class Pm4ShapeMatchWorkbenchService
{
    public static readonly Pm4CentroidCalibration DefaultMeasuredCalibration = new(
        new Vector3(0.773f, 0f, 0f),
        "development_00_00 measured +0.773 yd X correction for -X walkable-floor hull bias");

    private bool _libraryInitialized;

    public Pm4ShapeLibraryVisualService Library { get; } = new();
    public Pm4ShapeMatchReviewService Review { get; } = new();
    public Pm4ShapeMatchBatchService Batch { get; } = new();

    public bool CalibrationEnabled { get; set; } = true;
    public Vector3 CustomCalibrationOffset { get; set; } = DefaultMeasuredCalibration.Offset;
    public string CalibrationSource { get; private set; } = DefaultMeasuredCalibration.Source;

    public Pm4CentroidCalibration ActiveCalibration =>
        CalibrationEnabled
            ? new Pm4CentroidCalibration(CustomCalibrationOffset, CalibrationSource)
            : Pm4CentroidCalibration.None;

    public Pm4ShapeMatchProposal? ActiveProposal { get; private set; }
    public Pm4HullShapeInput? ActiveHullInput { get; private set; }
    public Pm4ProposalPreviewState? ActivePreview { get; private set; }
    public string? StatusMessage { get; set; }

    public void EnsureLibraryInitialized(WorldScene? worldScene = null, bool forceRebuild = false)
    {
        if (_libraryInitialized && !forceRebuild)
            return;

        IReadOnlyList<Pm4ModelShapeCandidate> resident = GatherResidentModelCandidates(worldScene);
        Library.RebuildFromCorpus(Pm4PriorKnowledgeDatabase.Instance, resident);
        _libraryInitialized = true;
    }

    public void ResetCalibrationToMeasured(WorldScene? worldScene = null)
    {
        CalibrationEnabled = true;
        CustomCalibrationOffset = DefaultMeasuredCalibration.Offset;
        CalibrationSource = DefaultMeasuredCalibration.Source;
        RefreshActiveCalibration(worldScene);
    }

    public void ZeroCalibration(WorldScene? worldScene = null)
    {
        CalibrationEnabled = false;
        RefreshActiveCalibration(worldScene);
    }

    public void InvertCalibration(WorldScene? worldScene = null)
    {
        CalibrationEnabled = true;
        CustomCalibrationOffset = -CustomCalibrationOffset;
        CalibrationSource = CalibrationSource + " (inverted)";
        RefreshActiveCalibration(worldScene);
    }

    public void SetCustomCalibrationOffset(Vector3 offset, WorldScene? worldScene = null)
    {
        CustomCalibrationOffset = offset;
        CalibrationSource = "Operator custom offset";
        RefreshActiveCalibration(worldScene);
    }

    public Pm4ShapeMatchProposal AnalyzeHull(
        WorldScene? worldScene,
        Pm4OverlayObject pm4Obj,
        int tileX,
        int tileY,
        ManualCurationProject project)
    {
        EnsureLibraryInitialized(worldScene);

        Pm4HullShapeInput hullInput = ExtractHullShapeInput(worldScene, pm4Obj, tileX, tileY);
        ActiveHullInput = hullInput;

        Pm4ShapeMatchProposal proposal = Pm4ShapeMatchAnalysisService.Analyze(
            hullInput,
            Library.GetHealthyModelCandidates(),
            ActiveCalibration);

        ActiveProposal = proposal;
        Review.EnqueueProposals([(hullInput, proposal)], project);

        int idx = -1;
        Pm4HullIdentity hullId = Pm4ShapeMatchReviewService.ToHullIdentity(hullInput);
        for (int i = 0; i < Review.Queue.Count; i++)
        {
            if (Review.Queue[i].Hull == hullId)
            {
                idx = i;
                break;
            }
        }
        if (idx >= 0)
            Review.SelectIndex(idx);

        StatusMessage = proposal.TopCandidate != null
            ? $"Proposed {System.IO.Path.GetFileName(proposal.TopCandidate.ModelPath)} (Conf: {proposal.Confidence:P0}, Dist: {proposal.TopCandidate.ShapeDistance:F3})"
            : "No credible shape candidate matched this PM4 hull.";

        return proposal;
    }

    public int AnalyzeTileHulls(
        WorldScene worldScene,
        int tileX,
        int tileY,
        ManualCurationProject project)
    {
        EnsureLibraryInitialized(worldScene);
        if (!worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileX, tileY), out var tileObjs) || tileObjs.Count == 0)
        {
            StatusMessage = $"No PM4 hulls loaded on tile ({tileX}, {tileY}).";
            return 0;
        }

        var hullInputs = ExtractTileHullInputs(worldScene, tileX, tileY, tileObjs);
        if (hullInputs.Count == 0)
            return 0;

        Pm4ShapeBatchResult batchResult = Pm4ShapeMatchAnalysisService.AnalyzeBatch(
            hullInputs,
            Library.GetHealthyModelCandidates(),
            ActiveCalibration);

        var pairs = new List<(Pm4HullShapeInput, Pm4ShapeMatchProposal)>(hullInputs.Count);
        for (int i = 0; i < hullInputs.Count; i++)
            pairs.Add((hullInputs[i], batchResult.Proposals[i]));

        Review.EnqueueProposals(pairs, project);
        if (batchResult.Proposals.Count > 0)
        {
            ActiveHullInput = hullInputs[0];
            ActiveProposal = batchResult.Proposals[0];
        }

        StatusMessage = $"Analyzed {hullInputs.Count} hulls on tile ({tileX}, {tileY}) — {Review.PendingCount} pending review.";
        return hullInputs.Count;
    }

    /// <summary>
    /// Spec 295 T015 (FR-005, FR-006): Previews a candidate model aligned to the hull in the 3D scene
    /// with the calibrated -X correction applied once and visible/resettable before commit.
    /// </summary>
    public Pm4ProposalPreviewState PreviewCandidateOnHull(
        WorldScene? worldScene,
        IViewerAppHost? host,
        Pm4HullShapeInput hullInput,
        string modelPath,
        string assetKind)
    {
        ClearViewportPreviewInstance(worldScene);

        Vector3 uncalibrated = hullInput.Descriptor.PrismCentroid;
        Pm4CentroidCalibration calibration = ActiveCalibration;
        Vector3 calibrated = calibration.Apply(uncalibrated, calibrationApplied: false);

        int? previewUniqueId = null;
        if (worldScene != null && host != null)
        {
            previewUniqueId = TrySpawnPreviewInstance(worldScene, hullInput.TileX, hullInput.TileY, modelPath, assetKind, calibrated, hullInput.Descriptor.PrimaryAxisAngleRadians);
        }

        ActivePreview = new Pm4ProposalPreviewState(
            Pm4ShapeMatchReviewService.ToHullIdentity(hullInput),
            modelPath,
            assetKind,
            uncalibrated,
            calibrated,
            calibration,
            previewUniqueId);

        StatusMessage = $"Previewing {System.IO.Path.GetFileName(modelPath)} at ({calibrated.X:F2}, {calibrated.Y:F2}, {calibrated.Z:F2}) [ΔX={calibrated.X - uncalibrated.X:+0.000;-0.000;0.000} yd].";
        return ActivePreview;
    }

    public void ClearViewportPreviewInstance(WorldScene? worldScene)
    {
        if (ActivePreview?.SpawnedPreviewUniqueId is { } uniqueId && worldScene != null)
        {
            IPm4OverlayHost pm4Host = worldScene;
            int tx = ActivePreview.Hull.TileX;
            int ty = ActivePreview.Hull.TileY;
            bool removed = false;

            if (pm4Host.TileWmoInstances.TryGetValue((tx, ty), out var wmos))
                removed |= wmos.RemoveAll(m => m.UniqueId == uniqueId) > 0;
            if (pm4Host.TileMdxInstances.TryGetValue((tx, ty), out var mdxs))
                removed |= mdxs.RemoveAll(m => m.UniqueId == uniqueId) > 0;

            if (removed)
                pm4Host.RebuildInstanceLists();
        }
        ActivePreview = null;
    }

    public IReadOnlyList<Pm4BatchTileInput> CollectLoadedBatchTiles(WorldScene? worldScene)
    {
        if (worldScene == null)
            return [];

        var list = new List<Pm4BatchTileInput>();
        foreach (var ((tx, ty), objs) in worldScene.Pm4Overlay._pm4TileObjects.OrderBy(static k => k.Key.tileY).ThenBy(static k => k.Key.tileX))
        {
            if (objs == null || objs.Count == 0)
                continue;

            var hulls = ExtractTileHullInputs(worldScene, tx, ty, objs);
            if (hulls.Count > 0)
                list.Add(new Pm4BatchTileInput(tx, ty, hulls));
        }
        return list;
    }

    public static Pm4HullShapeInput ExtractHullShapeInput(
        WorldScene? worldScene,
        Pm4OverlayObject pm4Obj,
        int tileX,
        int tileY)
    {
        List<Vector3> surfaceVerts = Pm4DoodadDuplicationService.ExtractCompositeSurfaceVertices(
            worldScene, pm4Obj, tileX, tileY, out float minWorldZ);

        List<Vector3>? mprlList = null;
        worldScene?.Pm4Overlay._pm4TilePositionRefs.TryGetValue((tileX, tileY), out mprlList);

        float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;
        Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, minWorldZ);

        Pm4GeometricShapeDescriptor descriptor = Pm4GeometricShapeDescriptor.FromPrism(prism) with
        {
            PrismCentroid = new Vector3(prism.Centroid.X, prism.Centroid.Y, placementZ ?? prism.BaseZ)
        };

        return new Pm4HullShapeInput(
            TileX: tileX,
            TileY: tileY,
            Ck24: pm4Obj.Ck24,
            ObjectPartId: pm4Obj.ObjectPartId,
            LinkGroupObjectId: pm4Obj.LinkGroupObjectId,
            Descriptor: descriptor);
    }

    private static IReadOnlyList<Pm4HullShapeInput> ExtractTileHullInputs(
        WorldScene worldScene,
        int tileX,
        int tileY,
        IReadOnlyList<Pm4OverlayObject> tileObjects)
    {
        var list = new List<Pm4HullShapeInput>();

        foreach (var obj in tileObjects)
        {
            if (obj.Triangles.Count == 0)
                continue;

            list.Add(ExtractHullShapeInput(worldScene, obj, tileX, tileY));
        }

        return list;
    }

    private void RefreshActiveCalibration(WorldScene? worldScene)
    {
        if (ActiveHullInput != null)
        {
            ActiveProposal = Pm4ShapeMatchAnalysisService.Analyze(
                ActiveHullInput,
                Library.GetHealthyModelCandidates(),
                ActiveCalibration);
        }

        if (ActivePreview != null)
        {
            Pm4CentroidCalibration cal = ActiveCalibration;
            Vector3 newCalibrated = cal.Apply(ActivePreview.UncalibratedCentroid, calibrationApplied: false);
            UpdateSpawnedPreviewPosition(worldScene, ActivePreview, newCalibrated);
            ActivePreview = ActivePreview with
            {
                CalibratedCentroid = newCalibrated,
                AppliedCalibration = cal
            };
        }
    }

    private static void UpdateSpawnedPreviewPosition(
        WorldScene? worldScene,
        Pm4ProposalPreviewState preview,
        Vector3 newPosition)
    {
        if (worldScene == null || preview.SpawnedPreviewUniqueId is not { } uniqueId)
            return;

        IPm4OverlayHost pm4Host = worldScene;
        int tx = preview.Hull.TileX;
        int ty = preview.Hull.TileY;

        void UpdateInList(List<ObjectInstance>? list)
        {
            if (list == null) return;
            int idx = list.FindIndex(m => m.UniqueId == uniqueId);
            if (idx < 0) return;

            var inst = list[idx];
            inst.PlacementPosition = newPosition;
            inst.Transform = WorldPlacementTransform.Build(inst.PlacementPosition, inst.PlacementRotation, inst.PlacementScale);
            WorldScene.TransformBounds(inst.LocalBoundsMin, inst.LocalBoundsMax, inst.Transform, out inst.BoundsMin, out inst.BoundsMax);
            list[idx] = inst;
            pm4Host.RebuildInstanceLists();
        }

        if (pm4Host.TileWmoInstances.TryGetValue((tx, ty), out var wmos))
            UpdateInList(wmos);
        if (pm4Host.TileMdxInstances.TryGetValue((tx, ty), out var mdxs))
            UpdateInList(mdxs);
    }

    private int? TrySpawnPreviewInstance(
        WorldScene worldScene,
        int tileX,
        int tileY,
        string modelPath,
        string assetKind,
        Vector3 position,
        float yawRadians)
    {
        IPm4OverlayHost pm4Host = worldScene;
        bool isWmo = Pm4DoodadDuplicationService.IsWmoAsset(modelPath, assetKind);
        string resolvedKind = isWmo ? "WMO" : Pm4DoodadDuplicationService.ResolveAssetKind(modelPath, assetKind);

        Vector3 localMin = new(-5f, -5f, 0f);
        Vector3 localMax = new(5f, 5f, 10f);
        if (Library.SelectEntry(modelPath) && Library.SelectedEntry is { } entry)
        {
            localMin = entry.BoundsMin;
            localMax = entry.BoundsMax;
        }

        float yawDeg = yawRadians * (180f / MathF.PI);
        Vector3 rotation = new(0f, 0f, yawDeg);
        Matrix4x4 transform = WorldPlacementTransform.Build(position, rotation, 1.0f);
        WorldScene.TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);

        int previewId = 0x7F000000 + Math.Abs(HashCode.Combine(tileX, tileY, modelPath));
        string modelKey = modelPath.Replace('/', '\\').ToLowerInvariant();

        var inst = new ObjectInstance
        {
            Transform = transform,
            ModelKey = modelKey,
            BoundsMin = worldMin,
            BoundsMax = worldMax,
            LocalBoundsMin = localMin,
            LocalBoundsMax = localMax,
            BoundsResolved = true,
            SelectionLocalBoundsMin = localMin,
            SelectionLocalBoundsMax = localMax,
            SelectionBoundsResolved = true,
            ModelName = System.IO.Path.GetFileName(modelPath),
            ModelPath = modelPath,
            AssetKind = resolvedKind,
            PlacementPosition = position,
            PlacementRotation = rotation,
            PlacementScale = 1.0f,
            UniqueId = previewId,
            PlacementEntryIndex = -1,
            TileX = tileX,
            TileY = tileY,
            HasTileCoordinate = true
        };

        if (isWmo)
        {
            if (!pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var list))
            {
                list = [];
                pm4Host.TileWmoInstances[(tileX, tileY)] = list;
            }
            list.RemoveAll(m => m.UniqueId == previewId);
            list.Add(inst);
            pm4Host.Assets.QueueWmoLoad(inst.ModelKey);
            pm4Host.Assets.PrioritizeWmoLoad(inst.ModelKey);
        }
        else
        {
            if (!pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var list))
            {
                list = [];
                pm4Host.TileMdxInstances[(tileX, tileY)] = list;
            }
            list.RemoveAll(m => m.UniqueId == previewId);
            list.Add(inst);
            pm4Host.Assets.QueueMdxLoad(inst.ModelKey);
            pm4Host.Assets.PrioritizeMdxLoad(inst.ModelKey);
        }

        pm4Host.RebuildInstanceLists();
        return previewId;
    }

    private static IReadOnlyList<Pm4ModelShapeCandidate> GatherResidentModelCandidates(WorldScene? worldScene)
    {
        if (worldScene == null)
            return [];

        IPm4OverlayHost pm4Host = worldScene;
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var list = new List<Pm4ModelShapeCandidate>();

        void AddFromInstances(IEnumerable<List<ObjectInstance>> groups, string defaultKind)
        {
            foreach (var group in groups)
            {
                foreach (var inst in group)
                {
                    if (string.IsNullOrWhiteSpace(inst.ModelPath) || !seen.Add(inst.ModelPath))
                        continue;

                    Vector3 min = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                        ? inst.SelectionLocalBoundsMin
                        : inst.LocalBoundsMin;
                    Vector3 max = inst.SelectionBoundsResolved && inst.SelectionLocalBoundsMax != Vector3.Zero
                        ? inst.SelectionLocalBoundsMax
                        : inst.LocalBoundsMax;

                    Vector3 span = max - min;
                    Vector3 sorted = new(
                        MathF.Max(MathF.Max(MathF.Abs(span.X), MathF.Abs(span.Y)), 0.1f),
                        MathF.Max(MathF.Min(MathF.Abs(span.X), MathF.Abs(span.Y)), 0.1f),
                        MathF.Max(MathF.Abs(span.Z), 0.1f));

                    list.Add(new Pm4ModelShapeCandidate(
                        inst.ModelPath,
                        string.IsNullOrWhiteSpace(inst.AssetKind) ? defaultKind : inst.AssetKind,
                        Pm4ShapeLibraryVisualService.BuildSyntheticDescriptor(sorted)));
                }
            }
        }

        AddFromInstances(pm4Host.TileWmoInstances.Values, "WMO");
        AddFromInstances(pm4Host.TileMdxInstances.Values, "M2");
        return list;
    }
}

