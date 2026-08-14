using ImGuiNET;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Viewer controls for proving the active client's resident audio path.
/// This is intentionally a small diagnostic surface; camera transport and
/// MIDI/DLS synthesis remain separate Spec 146 slices.
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
        ImGui.Text($"Resident emitters: {scene.ResidentAudioEmitterCount}  |  Active: {scene.ActiveAudioEmitterCount}");
        ImGui.Text($"SoundEntries rows: {scene.ResolvedAudioSoundEntryCount}");

        if (scene.AudioBackendReady)
            ImGui.TextColored(new System.Numerics.Vector4(0.45f, 0.9f, 0.55f, 1f), "OpenAL backend ready");
        else
            ImGui.TextColored(new System.Numerics.Vector4(1f, 0.65f, 0.35f, 1f), "OpenAL backend unavailable");

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
        ImGui.TextDisabled("Automatic MCSE playback is bounded to loaded tiles. DBC ZoneMusic is resolved from the active area; MIDI/DLS synthesis and Play + Video audio remain separate work.");
    }
}
