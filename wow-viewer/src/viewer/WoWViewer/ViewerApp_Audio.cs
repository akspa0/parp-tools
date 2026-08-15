using ImGuiNET;
using WoWViewer.Terrain;
using WowViewer.Core.Audio;

namespace WoWViewer;

/// <summary>
/// Viewer controls for proving the active client's resident audio path.
/// This is intentionally a small diagnostic surface; camera transport and
/// MIDI/DLS synthesis remain separate Spec 148 slices.
/// </summary>
public partial class ViewerApp
{
    private int _audioPreviewSoundEntryId;
    private bool _audioPreviewLoop = true;
    private float _audioMasterGain = 1f;
    private float _audioEmitterGain = 1f;
    private string _audioPanelMessage = "Select a resident SoundEntries ID or enter one manually.";

    private void DrawAudioContent()
    {
        WorldScene? scene = _worldScene;
        if (scene is null)
        {
            ImGui.TextDisabled("Load a world to configure client audio.");
            return;
        }

        ImGui.TextWrapped(scene.AudioStatus);
        ImGui.TextWrapped($"Area music: {scene.AreaMusicStatus}");
        ImGui.Text(scene.AudioMuted ? "Output: MUTED" : "Output: ON");
        ImGui.Text($"Resident emitters: {scene.ResidentAudioEmitterCount}  |  Active: {scene.ActiveAudioEmitterCount}");
        ImGui.Text($"SoundEntries rows: {scene.ResolvedAudioSoundEntryCount}  |  SoundWaterType rows: {scene.ResolvedAudioSoundWaterTypeCount}");

        bool worldTriggersEnabled = scene.AudioWorldTriggersEnabled;
        if (ImGui.Checkbox("Enable MCSE/MCNK world triggers", ref worldTriggersEnabled))
            scene.SetAudioWorldTriggersEnabled(worldTriggersEnabled);
        ImGui.TextDisabled("Off by default: resident rows remain inspectable, but looping world samples do not start automatically.");

        if (scene.AudioBackendReady)
            ImGui.TextColored(new System.Numerics.Vector4(0.45f, 0.9f, 0.55f, 1f), "OpenAL backend ready");
        else
            ImGui.TextColored(new System.Numerics.Vector4(1f, 0.65f, 0.35f, 1f), "OpenAL backend unavailable");

        ImGui.Separator();
        ImGui.Text("Emitter diagnostics");
        ImGui.TextDisabled("This inspects resident MCSE and MCNK liquid/environment records without starting audio sources.");
        if (ImGui.Button("Refresh decisions"))
            scene.RefreshAudioEmitterDiagnostics(probeFiles: false);
        ImGui.SameLine();
        if (ImGui.Button("Probe current emitters"))
            scene.RefreshAudioEmitterDiagnostics(probeFiles: true);

        IReadOnlyList<AudioTriggerDiagnostic> diagnostics = scene.AudioEmitterDiagnostics;
        ImGui.Text($"Resident trigger rows: {diagnostics.Count}");
        if (ImGui.BeginChild("##audio_emitter_diagnostics", new System.Numerics.Vector2(0f, 250f), true))
        {
            foreach (AudioTriggerDiagnostic diagnostic in diagnostics)
            {
                System.Numerics.Vector4 color = diagnostic.TerminalState switch
                {
                    AudioTriggerTerminalState.Active or AudioTriggerTerminalState.Ready
                        => new System.Numerics.Vector4(0.45f, 0.9f, 0.55f, 1f),
                    AudioTriggerTerminalState.OutOfRange or AudioTriggerTerminalState.Muted
                        or AudioTriggerTerminalState.Disabled
                        => new System.Numerics.Vector4(0.85f, 0.8f, 0.35f, 1f),
                    _ => new System.Numerics.Vector4(1f, 0.45f, 0.35f, 1f)
                };

                ImGui.TextColored(
                    color,
                    $"{diagnostic.TerminalState}  {diagnostic.TriggerKind}  SoundPoint={diagnostic.SoundPointId} SoundName={diagnostic.SoundNameId}");
                ImGui.Text(
                    $"Tile=({diagnostic.TileX},{diagnostic.TileY}) Chunk=({diagnostic.ChunkX},{diagnostic.ChunkY}) " +
                    $"Distance={diagnostic.DistanceToListener:F1}/{diagnostic.MaxDistance:F1}");
                ImGui.TextWrapped(
                    $"Raw XYZ=({diagnostic.RawPosition.X:F1}, {diagnostic.RawPosition.Y:F1}, {diagnostic.RawPosition.Z:F1})  " +
                    $"World XYZ=({diagnostic.WorldPosition.X:F1}, {diagnostic.WorldPosition.Y:F1}, {diagnostic.WorldPosition.Z:F1})");
                ImGui.TextWrapped(
                    $"Profile={diagnostic.CoordinateProfile}  Source={diagnostic.ResourceSource}  " +
                    $"Read={(diagnostic.BytesRead ? "yes" : "no")}  Decode={diagnostic.DecodeStatus}");
                if (diagnostic.TriggerKind == AudioTriggerKind.McnkLiquid)
                    ImGui.TextWrapped($"MCNK flags=0x{diagnostic.McnkFlags:X8}  LiquidFamily={diagnostic.LiquidFamily}  SoundSubtype={diagnostic.SoundWaterSubtype}");
                if (!string.IsNullOrWhiteSpace(diagnostic.SelectedVirtualPath))
                    ImGui.TextWrapped($"Path: {diagnostic.SelectedVirtualPath}");
                ImGui.TextWrapped($"Backend={diagnostic.BackendStatus}  {diagnostic.Detail}");
                ImGui.Separator();
            }

            ImGui.EndChild();
        }

        ImGui.Separator();
        ImGui.Text("SoundEntries preview");
        ImGui.TextDisabled("Preview plays the resolved client file at the current camera listener.");

        int[] residentIds = scene.ResidentAudioSoundEntryIds.ToArray();
        if (residentIds.Length > 0)
        {
            ImGui.SetNextItemWidth(-1f);
            if (ImGui.BeginCombo("Resident emitter", $"SoundEntries {_audioPreviewSoundEntryId}"))
            {
                foreach (int id in residentIds)
                {
                    bool selected = id == _audioPreviewSoundEntryId;
                    if (ImGui.Selectable($"SoundEntries {id}", selected))
                        _audioPreviewSoundEntryId = id;
                    if (selected)
                        ImGui.SetItemDefaultFocus();
                }

                ImGui.EndCombo();
            }
        }

        ImGui.SetNextItemWidth(-1f);
        ImGui.InputInt("SoundEntries ID", ref _audioPreviewSoundEntryId);
        _audioPreviewSoundEntryId = Math.Max(0, _audioPreviewSoundEntryId);
        ImGui.Checkbox("Loop preview", ref _audioPreviewLoop);
        if (ImGui.Button("Preview at Camera"))
        {
            if (_audioPreviewSoundEntryId <= 0)
            {
                _audioPanelMessage = "Enter a positive SoundEntries ID first.";
            }
            else if (scene.TryPreviewAudioSoundEntry((uint)_audioPreviewSoundEntryId, _audioPreviewLoop, out string reason))
            {
                _audioPanelMessage = reason;
            }
            else
            {
                _audioPanelMessage = reason;
            }
        }

        ImGui.SameLine();
        if (ImGui.Button("Stop Preview"))
        {
            scene.StopAudioPreview();
            _audioPanelMessage = "Audio preview stopped.";
        }

        if (!string.IsNullOrWhiteSpace(scene.AudioPreviewPath))
            ImGui.TextWrapped($"Preview: {scene.AudioPreviewPath}");
        ImGui.TextWrapped(_audioPanelMessage);
        ImGui.TextDisabled($"Last diagnostic: {scene.AudioLastDiagnostic}");

        ImGui.Separator();
        ImGui.Text("Volume");
        if (ImGui.SliderFloat("Master", ref _audioMasterGain, 0f, 2f, "%.2fx"))
            scene.SetAudioMasterGain(_audioMasterGain);
        if (ImGui.SliderFloat("Emitters", ref _audioEmitterGain, 0f, 2f, "%.2fx"))
            scene.SetAudioEmitterGain(_audioEmitterGain);

        ImGui.Separator();
        ImGui.TextDisabled("World triggers are bounded to loaded tiles and opt-in. DBC ZoneMusic is resolved from the active area; MIDI/DLS synthesis and Play + Video audio remain separate work.");
    }
}
