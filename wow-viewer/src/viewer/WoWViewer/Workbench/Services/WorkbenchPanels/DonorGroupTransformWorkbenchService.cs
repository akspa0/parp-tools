using System.Runtime.CompilerServices;
using System.Numerics;
using ImGuiNET;
using WowViewer.Core.Maps;
using WoWViewer.UI;

namespace WoWViewer.Workbench.Services.WorkbenchPanels;

/// <summary>Workbench UI and per-layer undo/redo state for Spec 294 donor-group transforms.</summary>
internal sealed class DonorGroupTransformWorkbenchService
{
    private readonly ConditionalWeakTable<PhaseLayerSettings, LayerState> _states = new();

    public bool DrawControls(PhaseLayerSettings layer)
    {
        ArgumentNullException.ThrowIfNull(layer);
        if (layer.TilePlacements.Count == 0)
            return false;

        ImGui.PushID(RuntimeHelpers.GetHashCode(layer));
        try
        {
            if (!SharedUiWidgets.SectionHeader("Donor group transform", "Preview a rigid transform of placed donor tiles. Tile coordinates snap to the 64×64 grid, and the authored angle and pivot remain project state.", defaultOpen: false, id: "donor-group-transform"))
            {
                return false;
            }

            LayerState state = _states.GetValue(layer, static _ => new LayerState());
            bool dirty = false;

            SharedUiWidgets.BodyText($"Group members: {layer.TilePlacements.Count}", muted: true);
            if (layer.GroupTransform is DonorGroupTransform current)
            {
                SharedUiWidgets.BodyText(
                    $"Last accepted: {current.RotationDegrees:0.###}° about ({current.PivotTileX:0.##}, {current.PivotTileY:0.##}), " +
                    $"translation ({current.TranslationX:0.##}, {current.TranslationY:0.##})",
                    muted: true);
            }

            float rotationDegrees = state.RotationDegrees;
            SharedUiWidgets.SetNextItemWidth(135f);
            if (ImGui.InputFloat("Rotation angle (degrees)##donorGroupAngle", ref rotationDegrees, 1f, 15f, "%.3f"))
            {
                state.RotationDegrees = rotationDegrees;
                state.Preview = null;
                state.PreviewSourcePlacements = [];
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("Clockwise group rotation. Any finite degree value is accepted; the default pivot is the selected group's centroid.");

            float translationX = state.TranslationX;
            SharedUiWidgets.SetNextItemWidth(110f);
            if (ImGui.InputFloat("Translation X (tiles)##donorGroupTranslateX", ref translationX, 1f, 5f, "%.2f"))
            {
                state.TranslationX = translationX;
                state.Preview = null;
                state.PreviewSourcePlacements = [];
            }
            ImGui.SameLine();
            float translationY = state.TranslationY;
            SharedUiWidgets.SetNextItemWidth(110f);
            if (ImGui.InputFloat("Translation Y (tiles)##donorGroupTranslateY", ref translationY, 1f, 5f, "%.2f"))
            {
                state.TranslationY = translationY;
                state.Preview = null;
                state.PreviewSourcePlacements = [];
            }

            if (SharedUiWidgets.ActionButton("Preview group transform##donorGroupPreview", help: "Compute a non-destructive preview. Nothing is applied until Accept."))
            {
                state.PreviewSourcePlacements = layer.TilePlacements.ToArray();
                state.PreviewSourceGroupTransform = layer.GroupTransform;
                state.Preview = DonorGroupTransformService.CreatePreview(
                    layer,
                    new DonorGroupTransform(state.RotationDegrees, state.TranslationX, state.TranslationY));
            }

            ImGui.SameLine();
            bool previewIsCurrent = state.Preview is not null &&
                state.PreviewSourceGroupTransform == layer.GroupTransform &&
                state.PreviewSourcePlacements.SequenceEqual(layer.TilePlacements);
            if (state.Preview is not null && !previewIsCurrent)
                SharedUiWidgets.BodyText("Layer placements changed after this preview. Preview again before accepting.", muted: true);

            if (SharedUiWidgets.ActionButton("Accept group transform##donorGroupAccept", enabled: state.Preview?.CanAccept == true && previewIsCurrent, help: "Commit the preview to this layer. The operation is added to undo history."))
            {
                DonorGroupTransformPreview preview = state.Preview!;
                state.Undo.Push(Capture(layer));
                TrimHistory(state.Undo);
                state.Redo.Clear();
                CopyPreviewToLayer(preview, layer);
                state.Preview = null;
                state.PreviewSourcePlacements = [];
                state.PreviewSourceGroupTransform = null;
                state.RotationDegrees = 0f;
                state.TranslationX = 0f;
                state.TranslationY = 0f;
                dirty = true;
            }

            if (state.Preview is not null)
            {
                if (SharedUiWidgets.ActionButton("Cancel transform preview##donorGroupCancel", help: "Discard this preview without changing the layer or its source donor data."))
                {
                    state.Preview = null;
                    state.PreviewSourcePlacements = [];
                    state.PreviewSourceGroupTransform = null;
                }

                DrawPreview(state.Preview, state.PreviewSourcePlacements);
            }

            if (SharedUiWidgets.ActionButton("Undo group transform##donorGroupUndo", enabled: state.Undo.Count > 0, help: "Restore the previous accepted group placement."))
            {
                LayerSnapshot currentSnapshot = Capture(layer);
                if (Restore(layer, state.Undo.Peek()))
                {
                    state.Undo.Pop();
                    state.Redo.Push(currentSnapshot);
                    state.Preview = null;
                    state.PreviewSourcePlacements = [];
                    state.PreviewSourceGroupTransform = null;
                    dirty = true;
                }
            }

            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Redo group transform##donorGroupRedo", enabled: state.Redo.Count > 0, help: "Reapply the most recently undone group placement."))
            {
                LayerSnapshot currentSnapshot = Capture(layer);
                if (Restore(layer, state.Redo.Peek()))
                {
                    state.Redo.Pop();
                    state.Undo.Push(currentSnapshot);
                    state.Preview = null;
                    state.PreviewSourcePlacements = [];
                    state.PreviewSourceGroupTransform = null;
                    dirty = true;
                }
            }

            return dirty;
        }
        finally
        {
            ImGui.PopID();
        }
    }

    private static void DrawPreview(
        DonorGroupTransformPreview preview,
        IReadOnlyList<PhaseTilePlacement> sourcePlacements)
    {
        DonorGroupTransform transform = preview.AppliedTransform;
        SharedUiWidgets.BodyText(
            $"Preview: {transform.RotationDegrees:0.###}° clockwise around ({transform.PivotTileX:0.##}, {transform.PivotTileY:0.##}); " +
            $"translation ({transform.TranslationX:0.##}, {transform.TranslationY:0.##}).",
            muted: true);

        foreach (DonorGroupTransformIssue issue in preview.Issues)
        {
            bool muted = issue.Severity != DonorGroupTransformIssueSeverity.Error;
            SharedUiWidgets.BodyTextWrapped(issue.Message, muted);
        }

        if (ImGui.BeginTable("donorGroupPreviewTiles", 2,
                ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.RowBg | ImGuiTableFlags.ScrollY | ImGuiTableFlags.SizingStretchProp,
                new Vector2(0f, 220f)))
        {
            ImGui.TableSetupColumn("Donor tile");
            ImGui.TableSetupColumn("Target before → preview");
            ImGui.TableHeadersRow();

            int count = preview.PreviewLayer.TilePlacements.Count;
            for (int index = 0; index < count; index++)
            {
                PhaseTilePlacement after = preview.PreviewLayer.TilePlacements[index];
                ImGui.TableNextRow();
                ImGui.TableSetColumnIndex(0);
                ImGui.TextUnformatted($"{after.DonorTileY:D2}_{after.DonorTileX:D2}");
                ImGui.TableSetColumnIndex(1);
                string sourceTarget = index < sourcePlacements.Count
                    ? $"{sourcePlacements[index].TargetTileY:D2}_{sourcePlacements[index].TargetTileX:D2}"
                    : "—";
                ImGui.TextUnformatted($"{sourceTarget} → {after.TargetTileY:D2}_{after.TargetTileX:D2}");
            }

            ImGui.EndTable();
        }

        if (!preview.CanAccept)
            SharedUiWidgets.BodyText("Resolve blocking issues before accepting this transform.", muted: true);
    }

    private static LayerSnapshot Capture(PhaseLayerSettings layer)
        => new(
            layer.TilePlacements.Select(static placement => placement.DonorTileX).ToArray(),
            layer.TilePlacements.Select(static placement => placement.DonorTileY).ToArray(),
            layer.TilePlacements.Select(static placement => placement.TargetTileX).ToArray(),
            layer.TilePlacements.Select(static placement => placement.TargetTileY).ToArray(),
            layer.GroupTransform);

    private static bool Restore(PhaseLayerSettings layer, LayerSnapshot snapshot)
    {
        if (layer.TilePlacements.Count != snapshot.DonorTileXs.Length)
            return false;

        for (int index = 0; index < layer.TilePlacements.Count; index++)
        {
            PhaseTilePlacement current = layer.TilePlacements[index];
            if (current.DonorTileX != snapshot.DonorTileXs[index] || current.DonorTileY != snapshot.DonorTileYs[index])
                return false;
        }

        for (int index = 0; index < layer.TilePlacements.Count; index++)
        {
            PhaseTilePlacement current = layer.TilePlacements[index];
            layer.TilePlacements[index] = current with
            {
                TargetTileX = snapshot.TargetTileXs[index],
                TargetTileY = snapshot.TargetTileYs[index],
            };
        }
        layer.GroupTransform = snapshot.GroupTransform;
        return true;
    }

    private static void CopyPreviewToLayer(DonorGroupTransformPreview preview, PhaseLayerSettings layer)
    {
        layer.TilePlacements.Clear();
        foreach (PhaseTilePlacement placement in preview.PreviewLayer.TilePlacements)
            layer.TilePlacements.Add(placement);
        layer.GroupTransform = preview.AppliedTransform;
    }

    private static void TrimHistory(Stack<LayerSnapshot> history)
    {
        if (history.Count <= 32)
            return;

        LayerSnapshot[] retained = history.Take(32).Reverse().ToArray();
        history.Clear();
        foreach (LayerSnapshot snapshot in retained)
            history.Push(snapshot);
    }

    private sealed class LayerState
    {
        public float RotationDegrees { get; set; }
        public float TranslationX { get; set; }
        public float TranslationY { get; set; }
        public DonorGroupTransformPreview? Preview { get; set; }
        public PhaseTilePlacement[] PreviewSourcePlacements { get; set; } = [];
        public DonorGroupTransform? PreviewSourceGroupTransform { get; set; }
        public Stack<LayerSnapshot> Undo { get; } = new();
        public Stack<LayerSnapshot> Redo { get; } = new();
    }

    private sealed record LayerSnapshot(
        int[] DonorTileXs,
        int[] DonorTileYs,
        int[] TargetTileXs,
        int[] TargetTileYs,
        DonorGroupTransform? GroupTransform);
}
