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

// ModelInspectorPanelService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ModelInspectorPanelService
{

    private static Vector3 QuaternionToEulerDegrees(Quaternion q)
    {
        float sinR = 2f * (q.W * q.X + q.Y * q.Z);
        float cosR = 1f - 2f * (q.X * q.X + q.Y * q.Y);
        float roll = MathF.Atan2(sinR, cosR);

        float sinP = 2f * (q.W * q.Y - q.Z * q.X);
        sinP = Math.Clamp(sinP, -1f, 1f);
        float pitch = MathF.Asin(sinP);

        float sinY = 2f * (q.W * q.Z + q.X * q.Y);
        float cosY = 1f - 2f * (q.Y * q.Y + q.Z * q.Z);
        float yaw = MathF.Atan2(sinY, cosY);

        return new Vector3(
            roll * (180f / MathF.PI),
            pitch * (180f / MathF.PI),
            yaw * (180f / MathF.PI));
    }

    private void DrawWmoDoodadInspector(WmoRenderer wmoRenderer, ref int selectedDoodadIndex, string idSuffix, Func<WmoDoodadInfo, bool>? frameDoodad, ref int groupFilterIndex)
    {
        ImGui.Separator();
        ImGui.Text("WMO Doodad Inspector");

        int doodadCount = wmoRenderer.DoodadInstanceCount;
        if (doodadCount <= 0)
        {
            selectedDoodadIndex = -1;
            ImGui.TextDisabled("The active doodad set has no resolved doodads.");
            return;
        }

        if (selectedDoodadIndex >= doodadCount)
            selectedDoodadIndex = -1;

        ImGui.TextDisabled($"Active set: {wmoRenderer.GetDoodadSetName(wmoRenderer.ActiveDoodadSet)}");
        ImGui.TextDisabled($"Doodads: {doodadCount}  Defs: {wmoRenderer.DoodadDefCount}");

        if (wmoRenderer.GroupRenderCount > 0)
        {
            if (ImGui.BeginCombo($"Filter by Group##{idSuffix}", groupFilterIndex < 0 ? "All Groups" : wmoRenderer.GetRenderGroupName(groupFilterIndex)))
            {
                if (ImGui.Selectable("All Groups", groupFilterIndex < 0))
                    groupFilterIndex = -1;
                for (int gi = 0; gi < wmoRenderer.GroupRenderCount; gi++)
                {
                    int dc = wmoRenderer.GetDoodadCountForRenderGroup(gi);
                    string gn = $"{wmoRenderer.GetRenderGroupName(gi)} ({dc} refs)";
                    bool gs = gi == groupFilterIndex;
                    if (ImGui.Selectable(gn, gs))
                        groupFilterIndex = gi;
                    if (gs)
                        ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
        }

        float listHeight = MathF.Min(220f, MathF.Max(110f, GetUniformListRowHeight() * Math.Min(doodadCount, 7)));
        if (ImGui.BeginChild($"##WmoDoodadInspector_{idSuffix}", new Vector2(0, listHeight), true))
        {
            for (int doodadIndex = 0; doodadIndex < doodadCount; doodadIndex++)
            {
                if (!wmoRenderer.TryGetDoodadInfo(doodadIndex, out WmoDoodadInfo doodad))
                    continue;

                if (groupFilterIndex >= 0)
                {
                    var rGroups = wmoRenderer.GetRenderGroupsForDoodadDef(doodad.DoodadDefIndex);
                    if (!rGroups.Contains(groupFilterIndex))
                        continue;
                }

                string label = $"[{doodadIndex}] {Path.GetFileNameWithoutExtension(doodad.ModelPath)}";
                if (!doodad.IsLoaded)
                    label += " [deferred]";
                if (!doodad.Visible)
                    label += " [hidden]";

                bool isSelected = doodadIndex == selectedDoodadIndex;
                if (ImGui.Selectable($"{label}##{idSuffix}_{doodadIndex}", isSelected))
                {
                    selectedDoodadIndex = doodadIndex;
                    frameDoodad?.Invoke(doodad);
                }

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(doodad.ModelPath);
            }
        }
        ImGui.EndChild();

        if (selectedDoodadIndex < 0 || !wmoRenderer.TryGetDoodadInfo(selectedDoodadIndex, out WmoDoodadInfo selectedDoodad))
            return;

        if (frameDoodad != null && ImGui.SmallButton($"Frame Doodad##{idSuffix}_FrameDoodad"))
            frameDoodad(selectedDoodad);

        DrawAssetPathActions("Doodad Asset", selectedDoodad.ModelPath, $"{idSuffix}_DoodadAsset");

        ImGui.Separator();
        ImGui.TextDisabled("Doodad Details");
        ImGui.TextDisabled($"Def Index: {selectedDoodad.DoodadDefIndex}");

        string doodadDefName = wmoRenderer.GetDoodadDefName(selectedDoodad.DoodadDefIndex);
        if (!string.IsNullOrEmpty(doodadDefName))
            ImGui.TextWrapped($"MODN Name: {doodadDefName}");

        ImGui.TextDisabled($"Path: {selectedDoodad.ModelPath}");

        ImGui.TextDisabled($"Visible: {(selectedDoodad.Visible ? "yes" : "no")}  Loaded: {(selectedDoodad.IsLoaded ? "yes" : "no")}");

        if (wmoRenderer.TryGetDoodadDef(selectedDoodad.DoodadDefIndex, out var doodadDef))
        {
            ImGui.TextDisabled($"Position: ({doodadDef.Position.X:F3}, {doodadDef.Position.Y:F3}, {doodadDef.Position.Z:F3})");
            ImGui.TextDisabled($"Scale: {doodadDef.Scale:F3}");

            var euler = QuaternionToEulerDegrees(doodadDef.Orientation);
            ImGui.TextDisabled($"Rotation (deg): ({euler.X:F1}, {euler.Y:F1}, {euler.Z:F1})");

            uint color = doodadDef.Color;
            byte a = (byte)((color >> 24) & 0xFF);
            byte r = (byte)((color >> 16) & 0xFF);
            byte g = (byte)((color >> 8) & 0xFF);
            byte b = (byte)(color & 0xFF);
            ImGui.TextDisabled($"Color: #{r:X2}{g:X2}{b:X2}{a:X2} (BGRA)");

            var groups = wmoRenderer.GetRenderGroupsForDoodadDef(selectedDoodad.DoodadDefIndex);
            if (groups.Count > 0)
            {
                ImGui.TextDisabled($"Referenced by {groups.Count} group(s):");
                string groupList = string.Join(", ", groups.Select(g => $"[{g}] {wmoRenderer.GetRenderGroupName(g)}"));
                ImGui.TextWrapped(groupList);
            }
            else
            {
                ImGui.TextDisabled("Not referenced by any loaded group.");
            }
        }
    }

    internal void DrawModelInfoPanelContent()
    {
        if (string.IsNullOrWhiteSpace(_modelInfo) && _renderer is not IModelRenderer && _renderer is not WmoRenderer)
        {
            ImGui.TextDisabled("No model info is available for the current selection or loaded asset.");
            return;
        }

        DrawModelInfoContent();
    }

    private void DrawModelInfoCoreContent()
    {
        if (string.IsNullOrEmpty(_modelInfo))
        {
            ImGui.TextWrapped("No model loaded.");
            return;
        }

        ImGui.TextWrapped(_modelInfo);

        if (_renderer != null && _renderer.SubObjectCount > 0)
        {
            ImGui.Separator();
            ImGui.Text("Visibility:");

            DrawRendererVisibilityControls(_renderer, "standalone");
        }
    }

    internal void DrawModelInfoContent()
    {
        DrawModelInfoCoreContent();

        if (_renderer is IModelRenderer || _renderer is WmoRenderer)
        {
            ImGui.Separator();
            DrawModelAnimationControls();
        }

        if (_renderer is IModelRenderer || _renderer is WmoRenderer)
        {
            ImGui.Separator();
            ImGui.Checkbox("Auto-frame on load", ref _autoFrameModelOnLoad);
            DrawToolbarPopupButton("Model Actions", string.Empty, "##ModelActionsPopup", () =>
            {
                if (ImGui.Button("Frame Model"))
                {
                    FrameCurrentModel();
                    ImGui.CloseCurrentPopup();
                }
            });
        }

        // Spec 231 D4: the model-info doodad-set combo was removed; the full combo
        // lives in Selected WMO Controls and the toolbar keeps the hovered-WMO quick combo.
        if (_renderer is WmoRenderer)
        {
            ImGui.Separator();
            DrawWmoLiquidRotationControls("standalone");
        }

        if (_renderer is WmoRenderer standaloneWmoRenderer)
        {
            if (TryGetStandaloneWmoAssetPath(out string standaloneWmoAssetPath))
            {
                ImGui.Separator();
                DrawAssetPathActions("WMO Asset", standaloneWmoAssetPath, "StandaloneWmoAsset");
            }

            ImGui.Separator();
            DrawStandaloneWmoGroupControls(standaloneWmoRenderer);
            DrawWmoDoodadInspector(
                standaloneWmoRenderer,
                ref _selectedStandaloneWmoDoodadIndex,
                "StandaloneWmo",
                doodad => TryFrameStandaloneWmoDoodad(standaloneWmoRenderer, doodad),
                ref _standaloneWmoDoodadGroupFilter);
        }

        if (_renderer is IModelRenderer standaloneModelRenderer)
        {
            _modelLoader.DrawStandaloneCharacterVariationControls(standaloneModelRenderer);
        }
    }

    private void DrawModelAnimationControls()
    {
        if (_renderer is not IModelRenderer modelRenderer || modelRenderer.Animator == null)
        {
            string rendererType = _renderer?.GetType().Name ?? "null";
            if (_renderer != null && _renderer is IModelRenderer mr && mr.Animator == null)
            {
                ImGui.TextDisabled($"Model loaded, but Animator is null. Renderer: {rendererType}");
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("The renderer exists but has no animation controller. Try loading a model with skeletal animation data.");
            }
            else
            {
                ImGui.TextDisabled("No animatable model is loaded.");
            }
            return;
        }

        var animator = modelRenderer.Animator;
        if (!animator.HasAnimation || animator.Sequences.Count == 0)
        {
            ImGui.TextDisabled("The loaded model has no animation sequences.");
            return;
        }

        int currentSeq = animator.CurrentSequence;
        string currentSeqName = currentSeq >= 0 && currentSeq < animator.Sequences.Count
            ? animator.Sequences[currentSeq].Name
            : "None";

        ImGui.Text("Sequence");
        ImGui.SetNextItemWidth(-1);
        if (ImGui.BeginCombo("##AnimSequence", currentSeqName))
        {
            for (int s = 0; s < animator.Sequences.Count; s++)
            {
                bool selected = s == currentSeq;
                string seqName = animator.Sequences[s].Name;
                if (string.IsNullOrEmpty(seqName))
                    seqName = $"Sequence {s}";

                if (ImGui.Selectable(seqName, selected))
                    animator.SetSequence(s);
                if (selected) ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        if (currentSeq < 0 || currentSeq >= animator.Sequences.Count)
            return;

        var seq = animator.Sequences[currentSeq];
        float seqStart = seq.Time.Start;
        float seqEnd = seq.Time.End;
        float duration = seqEnd - seqStart;
        float currentAbs = Math.Clamp(animator.CurrentFrame, seqStart, seqEnd);
        float currentRel = currentAbs - seqStart;

        bool isPlaying = animator.IsPlaying;

        ImGui.Separator();
        ImGui.Text("Playback");

        // Large prominent Play / Pause / Stop buttons
        if (ImGui.Button(isPlaying ? "Pause" : "Play", new Vector2(80, 0)))
            animator.IsPlaying = !isPlaying;
        ImGui.SameLine();
        if (ImGui.Button("Stop", new Vector2(80, 0)))
        {
            animator.IsPlaying = false;
            animator.CurrentFrame = seqStart;
        }
        ImGui.SameLine();
        if (ImGui.Button("Previous Key"))
        {
            animator.IsPlaying = false;
            animator.StepToPrevKeyframe();
        }
        ImGui.SameLine();
        if (ImGui.Button("Next Key"))
        {
            animator.IsPlaying = false;
            animator.StepToNextKeyframe();
        }

        // Loop checkbox
        bool loop = animator.Loop;
        if (ImGui.Checkbox("Loop", ref loop))
            animator.Loop = loop;

        // Speed control
        ImGui.SameLine();
        float speed = animator.PlaybackSpeed;
        string[] speedLabels = { "0.25x", "0.5x", "1x", "2x" };
        float[] speedValues = { 0.25f, 0.5f, 1.0f, 2.0f };
        ImGui.Text("Speed");
        for (int i = 0; i < speedValues.Length; i++)
        {
            ImGui.SameLine();
            bool selected = Math.Abs(speed - speedValues[i]) < 0.001f;
            if (selected)
                ImGui.PushStyleColor(ImGuiCol.Button, ImGui.GetColorU32(ImGuiCol.ButtonActive));
            if (ImGui.Button(speedLabels[i]))
                animator.PlaybackSpeed = speedValues[i];
            if (selected)
                ImGui.PopStyleColor();
        }

        ImGui.SameLine();
        if (ImGui.Button("Export JSON"))
            ExportAnimationStateJson(animator, currentSeq, currentSeqName, seqStart, seqEnd);

        // Timeline slider
        ImGui.Separator();
        ImGui.SetNextItemWidth(-1);
        if (ImGui.SliderFloat("##Timeline", ref currentRel, 0, duration, $"Frame: {currentAbs:F0} / {seqEnd:F0}"))
        {
            animator.IsPlaying = false;
            animator.CurrentFrame = seqStart + currentRel;
        }

        ImGui.Text($"Duration: {duration:F0}ms ({duration / 1000.0f:F2}s)");

        if (ImGui.TreeNode("Animation Debug"))
        {
            ImGui.Text($"Current Seq: {currentSeq}");
            ImGui.Text($"Current Abs Frame: {currentAbs:F2}");
            ImGui.Text($"Seq Range: [{seqStart}, {seqEnd}]");

            var stats = animator.GetTrackDebugStatsForCurrentSequence();
            ImGui.Text($"T keys total/in-range: {stats.TranslationKeysTotal}/{stats.TranslationKeysInSequence}");
            ImGui.Text($"R keys total/in-range: {stats.RotationKeysTotal}/{stats.RotationKeysInSequence}");
            ImGui.Text($"S keys total/in-range: {stats.ScalingKeysTotal}/{stats.ScalingKeysInSequence}");

            string minKey = stats.MinKeyTime?.ToString() ?? "n/a";
            string maxKey = stats.MaxKeyTime?.ToString() ?? "n/a";
            ImGui.Text($"All key range: [{minKey}, {maxKey}]");

            ImGui.Separator();
            ImGui.Text("Sequences (first 12):");
            int previewCount = Math.Min(12, animator.Sequences.Count);
            for (int i = 0; i < previewCount; i++)
            {
                var s = animator.Sequences[i];
                string name = string.IsNullOrWhiteSpace(s.Name) ? "<empty>" : s.Name;
                ImGui.Text($"{i}: {name} [{s.Time.Start}-{s.Time.End}]");
            }

            ImGui.TreePop();
        }
    }

    internal void ExportAnimationStateJson(
        IAnimationController animator,
        int currentSeq,
        string currentSeqName,
        float seqStart,
        float seqEnd)
    {
        string sourceName = Path.GetFileNameWithoutExtension(_loadedFilePath ?? _renderer?.GetType().Name ?? "animation");
        if (string.IsNullOrWhiteSpace(sourceName))
            sourceName = "animation";

        string defaultFileName = $"{sourceName}_animation_state.json";
        string initialDir = !string.IsNullOrWhiteSpace(_loadedFilePath) ? Path.GetDirectoryName(_loadedFilePath) ?? Environment.CurrentDirectory : Environment.CurrentDirectory;

        ImGuiPathPicker.Instance.Open(
            "Export Animation State JSON",
            ImGuiPathPickerMode.SaveFile,
            initialDir,
            ".json",
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                var payload = new
                {
                    exportedAtUtc = DateTimeOffset.UtcNow.ToString("O"),
                    source = new
                    {
                        loadedFilePath = _loadedFilePath,
                        rendererType = _renderer?.GetType().Name,
                    },
                    playback = new
                    {
                        currentSequence = currentSeq,
                        currentSequenceName = currentSeqName,
                        currentFrame = animator.CurrentFrame,
                        sequenceStart = seqStart,
                        sequenceEnd = seqEnd,
                        isPlaying = animator.IsPlaying,
                        playbackSpeed = animator.PlaybackSpeed,
                        loop = animator.Loop,
                    },
                    sequences = animator.Sequences.Select(seq => new
                    {
                        index = seq.Index,
                        name = seq.Name,
                        start = seq.Time.Start,
                        end = seq.Time.End,
                        duration = seq.Time.End - seq.Time.Start,
                    }).ToArray(),
                    debug = animator.GetTrackDebugStatsForCurrentSequence(),
                };

                try
                {
                    File.WriteAllText(picked, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
                    _statusMessage = $"Exported animation state JSON: {picked}";
                }
                catch (Exception ex)
                {
                    _statusMessage = $"Animation state export failed: {ex.Message}";
                }
            },
            defaultFileName);
    }

    internal void DrawSelectedWmoControls()
    {
        if (_worldScene == null || _worldScene.SelectedObjectType != Terrain.ObjectType.Wmo || !_worldScene.SelectedInstance.HasValue)
            return;

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        string normalizedKey = WorldAssetManager.NormalizeKey(selected.ModelPath);
        WmoRenderer? wmoRenderer = _worldScene.Assets.GetWmo(normalizedKey);
        if (wmoRenderer == null)
        {
            ImGui.Separator();
            ImGui.TextDisabled("Selected WMO controls unavailable: renderer not loaded.");
            return;
        }

        ImGui.Separator();
        ImGui.Text("Selected WMO Controls");
        ImGui.TextDisabled("Changes apply to all loaded instances of this WMO model.");

        if (wmoRenderer.DoodadSetCount > 0)
        {
            ImGui.Text("Doodad Set:");
            int activeSet = wmoRenderer.ActiveDoodadSet;
            string currentSetName = wmoRenderer.GetDoodadSetName(activeSet);
            if (ImGui.BeginCombo("##SelectedWmoDoodadSet", currentSetName))
            {
                for (int setIndex = 0; setIndex < wmoRenderer.DoodadSetCount; setIndex++)
                {
                    bool selectedSet = setIndex == activeSet;
                    if (ImGui.Selectable(wmoRenderer.GetDoodadSetName(setIndex), selectedSet))
                        wmoRenderer.SetActiveDoodadSet(setIndex);
                    if (selectedSet)
                        ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
        }

        ImGui.Text("Groups / Doodads:");
        DrawRendererVisibilityControls(wmoRenderer, "selected_wmo");
        DrawWmoDoodadInspector(
            wmoRenderer,
            ref _selectedWorldWmoDoodadIndex,
            "SelectedWmo",
            doodad => TryFrameSelectedWorldWmoDoodad(wmoRenderer, doodad),
            ref _worldWmoDoodadGroupFilter);
    }

    private void DrawRendererVisibilityControls(ISceneRenderer renderer, string idSuffix)
    {
        if (ImGui.SmallButton($"All On##{idSuffix}"))
        {
            for (int i = 0; i < renderer.SubObjectCount; i++)
                renderer.SetSubObjectVisible(i, true);
        }

        ImGui.SameLine();
        if (ImGui.SmallButton($"All Off##{idSuffix}"))
        {
            for (int i = 0; i < renderer.SubObjectCount; i++)
                renderer.SetSubObjectVisible(i, false);
        }

        ImGui.TextDisabled($"Entries: {renderer.SubObjectCount}");
        float listHeight = MathF.Min(220f, MathF.Max(110f, GetUniformListRowHeight() * Math.Min(renderer.SubObjectCount, 8)));
        if (!ImGui.BeginChild($"##SubObjectVisibility_{idSuffix}", new Vector2(0, listHeight), true))
        {
            ImGui.EndChild();
            return;
        }

        float rowHeight = GetUniformListRowHeight();
        GetVisibleListRange(renderer.SubObjectCount, rowHeight, out int startIndex, out int endIndex);
        if (startIndex > 0)
            ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

        for (int i = startIndex; i < endIndex; i++)
        {
            bool visible = renderer.GetSubObjectVisible(i);
            string label = $"{renderer.GetSubObjectName(i)}##subobj_{idSuffix}_{i}";
            if (ImGui.Checkbox(label, ref visible))
                renderer.SetSubObjectVisible(i, visible);
        }

        if (endIndex < renderer.SubObjectCount)
            ImGui.Dummy(new Vector2(0, (renderer.SubObjectCount - endIndex) * rowHeight));

        ImGui.EndChild();
    }

    internal void FrameCurrentModel()
    {
        if (_renderer is IModelRenderer modelRenderer)
        {
            var bmin = modelRenderer.BoundsMin;
            var bmax = modelRenderer.BoundsMax;
            FrameBounds(bmin, bmax, mdxMirrorX: true);
        }
        else if (_renderer is WmoRenderer wmoR)
        {
            FrameBounds(wmoR.BoundsMin, wmoR.BoundsMax, mdxMirrorX: false);
        }
    }

    internal void FrameBounds(Vector3 boundsMin, Vector3 boundsMax, bool mdxMirrorX)
    {
        var center = (boundsMin + boundsMax) * 0.5f;
        var extent = boundsMax - boundsMin;
        float radius = MathF.Max(extent.Length() * 0.5f, 1f);

        if (mdxMirrorX)
            center.X = -center.X;

        float dist = MathF.Max(radius * 3.0f, 10f);
        _camera.Position = center + new Vector3(-dist, 0, radius * 0.6f);
        _camera.Yaw = 0f;
        _camera.Pitch = -15f;
    }

    private void DrawWmoLiquidRotationControls(string idSuffix)
    {
        int quarterTurns = WmoRenderer.MliqRotationQuarterTurns;
        string currentLabel = WmoLiquidRotationLabels[Math.Clamp(quarterTurns, 0, WmoLiquidRotationLabels.Length - 1)];

        if (ImGui.BeginCombo($"WMO MLIQ Additional Rotation##{idSuffix}", currentLabel))
        {
            for (int i = 0; i < WmoLiquidRotationLabels.Length; i++)
            {
                bool selected = i == quarterTurns;
                if (ImGui.Selectable(WmoLiquidRotationLabels[i], selected))
                {
                    _hasExplicitWmoMliqRotationOverride = i != 0;
                    WmoRenderer.MliqRotationQuarterTurns = i;
                }
                if (selected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        ImGui.TextDisabled("Adds on top of the version-aware WMO MLIQ baseline. Changes are live.");
    }

    internal void DrawHoveredWmoDoodadSetCombo(WmoRenderer hoveredWmo, string sourcePath)
    {
        ImGui.SetNextItemWidth(150f);
        int activeDoodadSet = hoveredWmo.ActiveDoodadSet;
        if (ImGui.BeginCombo("##HoveredWmoDoodadSet", hoveredWmo.GetDoodadSetName(activeDoodadSet)))
        {
            for (int setIndex = 0; setIndex < hoveredWmo.DoodadSetCount; setIndex++)
            {
                bool isSetSelected = setIndex == activeDoodadSet;
                if (ImGui.Selectable(hoveredWmo.GetDoodadSetName(setIndex), isSetSelected))
                    hoveredWmo.SetActiveDoodadSet(setIndex);
                if (isSetSelected)
                    ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip($"Doodad set for hovered WMO '{Path.GetFileName(sourcePath)}'.");
    }



    internal void DrawModelAnimationsSubTab()
    {
        ImGui.TextDisabled("Animations");
        ImGui.Separator();

        DrawModelAnimationControls();

        if (_worldScene?.SelectedInstance.HasValue == true && _worldScene.SelectedObjectType == Terrain.ObjectType.Mdx)
        {
            ImGui.Separator();
            _sqlSpawnStreaming.DrawSelectedSqlGameObjectAnimationControls();

            // Also show animation controls for non-SQL world MDX instances
            var inst = _worldScene.SelectedInstance.Value;
            if (!_sqlSpawnStreaming.HasSqlGameObjectForSelectedInstance())
            {
                var mdxRenderer = _worldScene.Assets.GetMdx(inst.ModelKey);
                if (mdxRenderer?.Animator != null && mdxRenderer.Animator.HasAnimation && mdxRenderer.Animator.Sequences.Count > 0)
                {
                    DrawWorldMdxAnimationControls(mdxRenderer.Animator);
                }
            }
        }
    }

    private void DrawWorldMdxAnimationControls(IAnimationController animator)
    {
        if (!animator.HasAnimation || animator.Sequences.Count == 0)
            return;

        ImGui.Separator();
        ImGui.TextColored(new Vector4(0.85f, 1f, 0.85f, 1f), "World MDX Animation");

        int currentSeq = animator.CurrentSequence;
        string currentSeqName = currentSeq >= 0 && currentSeq < animator.Sequences.Count
            ? animator.Sequences[currentSeq].Name
            : "None";
        if (string.IsNullOrWhiteSpace(currentSeqName))
            currentSeqName = $"Sequence {currentSeq}";

        float seqStart = currentSeq >= 0 && currentSeq < animator.Sequences.Count
            ? animator.Sequences[currentSeq].Time.Start
            : 0f;
        float seqEnd = currentSeq >= 0 && currentSeq < animator.Sequences.Count
            ? animator.Sequences[currentSeq].Time.End
            : 0f;

        if (ImGui.BeginCombo("##world_mdx_anim_seq", currentSeqName))
        {
            for (int s = 0; s < animator.Sequences.Count; s++)
            {
                bool selected = s == currentSeq;
                string seqName = animator.Sequences[s].Name;
                if (string.IsNullOrEmpty(seqName))
                    seqName = $"Sequence {s}";

                if (ImGui.Selectable(seqName, selected))
                    animator.SetSequence(s);
                if (selected) ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        bool isPlaying = animator.IsPlaying;
        if (ImGui.Button(isPlaying ? "Pause" : "Play", new Vector2(80, 0)))
            animator.IsPlaying = !isPlaying;
        ImGui.SameLine();
        if (ImGui.Button("Stop", new Vector2(80, 0)))
        {
            animator.IsPlaying = false;
            if (currentSeq >= 0 && currentSeq < animator.Sequences.Count)
                animator.CurrentFrame = animator.Sequences[currentSeq].Time.Start;
        }

        bool loop = animator.Loop;
        ImGui.SameLine();
        if (ImGui.Checkbox("Loop", ref loop))
            animator.Loop = loop;

        float speed = animator.PlaybackSpeed;
        ImGui.SameLine();
        ImGui.Text("Speed");
        float[] speedValues = { 0.25f, 0.5f, 1.0f, 2.0f };
        string[] speedLabels = { "0.25x", "0.5x", "1x", "2x" };
        for (int i = 0; i < speedValues.Length; i++)
        {
            ImGui.SameLine();
            bool selected = Math.Abs(speed - speedValues[i]) < 0.001f;
            if (selected)
                ImGui.PushStyleColor(ImGuiCol.Button, ImGui.GetColorU32(ImGuiCol.ButtonActive));
            if (ImGui.Button(speedLabels[i]))
                animator.PlaybackSpeed = speedValues[i];
            if (selected)
                ImGui.PopStyleColor();
        }

        ImGui.SameLine();
        if (ImGui.Button("Export JSON##World"))
            ExportAnimationStateJson(animator, currentSeq, currentSeqName, seqStart, seqEnd);
    }

    internal void DrawModelActionsSubTab()
    {
        ImGui.TextDisabled("Actions");
        ImGui.Separator();

        if (_renderer == null || (!(_renderer is IModelRenderer) && !(_renderer is WmoRenderer)))
        {
            ImGui.TextDisabled("No model actions are available. Load a model (M2/MDX/WMO) first.");
            return;
        }

        ImGui.Checkbox("Auto-frame on load", ref _autoFrameModelOnLoad);

        if (ImGui.Button("Frame Model", new Vector2(120, 0)))
            FrameCurrentModel();

        // Spec 231 D4: the Actions doodad-set combo was removed; the full combo
        // lives in Selected WMO Controls and the toolbar keeps the hovered-WMO quick combo.
    }
}
