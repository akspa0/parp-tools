using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.Runtime.M2;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using Silk.NET.Input;

namespace WoWViewer;

public partial class ViewerApp
{
    private enum CapturePanelTab
    {
        Automation,
        CameraPath,
    }

    private CapturePanelTab? _pendingCapturePanelTab;
    private readonly M2CameraPathDocument _cameraPath = new();
    private int _selectedCameraPathKey = -1;
    private string _cameraPathName = "camera_path";
    private string _cameraPathFilePath = string.Empty;
    private bool _cameraPathPlaying;
    private bool _cameraPathVideoCaptureActive;
    private bool _showCameraPathOverlay = true;
    private bool _cameraPathLoop;
    private double _cameraPathTimeSeconds;
    private int _cameraPathDefaultKeySpacingMs = 1000;
    private string _cameraPathImportPath = string.Empty;
    private int _cameraPathImportCameraIndex;
    private int _cameraPathImportSequenceIndex;
    private int _cameraPathImportSampleIntervalMs = 125;
    private bool _cameraPathPreloadEnabled = true;
    private int _cameraPathPreloadTileRadius = 1;
    private int _cameraPathPreloadSampleSpacingMs = 500;
    private CameraPathPreloadState? _cameraPathPreload;
    private bool _cameraPathPlaybackPending;
    private bool _cameraPathVideoCapturePending;
    private bool _cameraPathKeyboardAuthoring;
    private int _cameraPathKeyboardTimeStepMs = 100;
    private float _cameraPathKeyboardRollStepDegrees = 1f;

    private bool _cameraPathKeyWasPressed;
    private bool _cameraPathUpdateWasPressed;
    private bool _cameraPathDeleteWasPressed;
    private bool _cameraPathPlayWasPressed;
    private bool _cameraPathSaveWasPressed;
    private bool _cameraPathExportWasPressed;
    private bool _cameraPathLeftWasPressed;
    private bool _cameraPathRightWasPressed;
    private bool _cameraPathHomeWasPressed;
    private bool _cameraPathEndWasPressed;
    private bool _cameraPathRetimingLeftWasPressed;
    private bool _cameraPathRetimingRightWasPressed;
    private bool _cameraPathPreviousTimeWasPressed;
    private bool _cameraPathNextTimeWasPressed;

    private const int MaxCameraPathPreloadTiles = 512;

    private sealed class CameraPathPreloadState
    {
        public required HashSet<(int tileX, int tileY)> Tiles { get; init; }
        public int StableFrames { get; set; }
        public bool Ready { get; set; }
    }

    private void OpenCapturePanelTab(CapturePanelTab tab)
    {
        _pendingCapturePanelTab = tab;
        if (_useTabUi)
        {
            _activeUtilitiesTabIndex = (int)Workbench.UtilitiesBottomTab.Capture;
            OpenWorkbenchTab(Workbench.ToolsBottomTab.Utilities);
        }
        else if (tab == CapturePanelTab.Automation)
        {
            _showCaptureAutomationWindow = true;
        }
        else
        {
            _showCameraPathWindow = true;
        }
    }

    private void DrawCapturePanelContent()
    {
        if (ImGui.BeginTabBar("##CapturePanelTabs", ImGuiTabBarFlags.FittingPolicyScroll))
        {
            bool automationTabOpen = true;
            ImGuiTabItemFlags automationFlags = _pendingCapturePanelTab == CapturePanelTab.Automation
                ? ImGuiTabItemFlags.SetSelected
                : ImGuiTabItemFlags.None;
            if (ImGui.BeginTabItem("Capture Automation", ref automationTabOpen, automationFlags))
            {
                DrawCaptureAutomationContent();
                ImGui.EndTabItem();
            }

            bool cameraPathTabOpen = true;
            ImGuiTabItemFlags cameraPathFlags = _pendingCapturePanelTab == CapturePanelTab.CameraPath
                ? ImGuiTabItemFlags.SetSelected
                : ImGuiTabItemFlags.None;
            if (ImGui.BeginTabItem("Camera Path", ref cameraPathTabOpen, cameraPathFlags))
            {
                DrawCameraPathContent();
                ImGui.EndTabItem();
            }

            _pendingCapturePanelTab = null;
            ImGui.EndTabBar();
        }
    }

    private void DrawCameraPathWindow()
    {
        if (!ImGui.Begin("Camera Path", ref _showCameraPathWindow))
        {
            ImGui.End();
            return;
        }

        DrawCameraPathContent();
        ImGui.End();
    }

    private void DrawCameraPathContent()
    {
        string currentMap = GetCurrentCaptureMapName();
        string currentBuild = GetCurrentCaptureBuildVersion();
        ImGui.TextDisabled($"Map: {currentMap}  Build: {currentBuild}");
        if (_cameraPath.HasCinematicCameraOrigin)
        {
            ImGui.TextDisabled(
                $"DBC origin: {_cameraPath.CinematicCameraModel}  tile " +
                $"({_cameraPath.CinematicCameraOriginTileX}, {_cameraPath.CinematicCameraOriginTileY})");
        }

        string name = _cameraPathName;
        if (ImGui.InputText("Path Name", ref name, 128))
            _cameraPathName = name;

        ImGui.TextDisabled($"Keys: {_cameraPath.Keyframes.Count}  Duration: {_cameraPath.DurationMs / 1000f:F2}s");
        ImGui.Checkbox("Keyboard authoring mode", ref _cameraPathKeyboardAuthoring);
        ImGui.SameLine();
        int keyboardStep = _cameraPathKeyboardTimeStepMs;
        if (ImGui.DragInt("Time Step (ms)", ref keyboardStep, 1f, 10, 5000))
            _cameraPathKeyboardTimeStepMs = Math.Clamp(keyboardStep, 10, 5000);
        float rollStep = _cameraPathKeyboardRollStepDegrees;
        if (ImGui.DragFloat("Roll Step (deg)", ref rollStep, 0.25f, 0.1f, 45f))
            _cameraPathKeyboardRollStepDegrees = Math.Clamp(rollStep, 0.1f, 45f);
        ImGui.TextDisabled("Keyboard: WASD move | Z/X roll | K add at playhead | U update | Delete remove | Ctrl+Up/Down time | arrows select | Ctrl+Left/Right retime | Space play | Ctrl+S save | Ctrl+E export");
        float playhead = (float)Math.Clamp(_cameraPathTimeSeconds, 0d, _cameraPath.DurationMs / 1000d);
        if (ImGui.SliderFloat("Timeline", ref playhead, 0f, Math.Max(0.01f, _cameraPath.DurationMs / 1000f), $"{playhead:F2}s"))
        {
            _cameraPathTimeSeconds = playhead;
            ApplyCameraPathAtCurrentTime();
        }
        if (ImGui.Button("Add Key At Playhead"))
            AddCameraPathKeyAtTime((int)MathF.Round(playhead * 1000f));
        ImGui.SameLine();
        if (ImGui.Button("Set Selected To Playhead") && _selectedCameraPathKey >= 0 && _selectedCameraPathKey < _cameraPath.Keyframes.Count)
        {
            _cameraPath.Keyframes[_selectedCameraPathKey].TimeMs = (int)MathF.Round(playhead * 1000f);
            M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
            _selectedCameraPathKey = Math.Clamp(_selectedCameraPathKey, 0, _cameraPath.Keyframes.Count - 1);
        }
        int spacing = _cameraPathDefaultKeySpacingMs;
        if (ImGui.DragInt("New Key Spacing (ms)", ref spacing, 10f, 100, 60000))
            _cameraPathDefaultKeySpacingMs = Math.Clamp(spacing, 100, 60000);

        ImGui.Checkbox("Preload path before capture", ref _cameraPathPreloadEnabled);
        ImGui.SameLine();
        int radius = _cameraPathPreloadTileRadius;
        if (ImGui.DragInt("Tile Radius", ref radius, 0.05f, 0, 2))
            _cameraPathPreloadTileRadius = Math.Clamp(radius, 0, 2);
        int sampleSpacing = _cameraPathPreloadSampleSpacingMs;
        if (ImGui.DragInt("Preload Sample Spacing (ms)", ref sampleSpacing, 10f, 100, 5000))
            _cameraPathPreloadSampleSpacingMs = Math.Clamp(sampleSpacing, 100, 5000);
        if (ImGui.Button("Warm Path") && _cameraPathPreloadEnabled)
            BeginCameraPathPreload();
        ImGui.SameLine();
        if (ImGui.Button("Release Warmup"))
            EndCameraPathPreload();
        if (_cameraPathPreload is { } preload)
        {
            string state = preload.Ready ? "ready" : "warming";
            int pending = _worldScene?.PendingCapturePreloadLoadCount ?? 0;
            ImGui.TextDisabled($"Path preload: {state}  tiles {preload.Tiles.Count}  pending {pending}  stable {preload.StableFrames}/2");
        }

        if (_cameraPathVideoCapturePending)
            ImGui.TextColored(new Vector4(1f, 0.8f, 0.25f, 1f), "Play + Video is waiting for the path warmup to finish.");
        else if (_cameraPathVideoCaptureActive)
            ImGui.TextColored(new Vector4(0.3f, 1f, 0.5f, 1f), "Play + Video is recording.");
        else if (!string.IsNullOrWhiteSpace(_statusMessage))
            ImGui.TextWrapped($"Status: {_statusMessage}");

        ImGui.Separator();
        ImGui.TextDisabled("Path collision");
        bool terrainCollision = _cameraPath.TerrainCollisionEnabled;
        if (ImGui.Checkbox("Terrain height collision", ref terrainCollision))
            _cameraPath.TerrainCollisionEnabled = terrainCollision;
        ImGui.SameLine();
        bool wmoCollision = _cameraPath.WmoCollisionEnabled;
        if (ImGui.Checkbox("WMO bounds collision", ref wmoCollision))
            _cameraPath.WmoCollisionEnabled = wmoCollision;
        float clearance = _cameraPath.CollisionClearance;
        if (ImGui.DragFloat("Camera clearance", ref clearance, 0.05f, 0f, 10f, "%.2f m"))
            _cameraPath.CollisionClearance = Math.Clamp(clearance, 0f, 10f);
        ImGui.TextDisabled("WMO collision uses loaded placement bounds; it is conservative and opt-in.");

        if (ImGui.Button("Add Current Camera Key"))
            AddCameraPathKeyFromCurrentCamera();
        ImGui.SameLine();
        if (ImGui.Button("Update Selected"))
            UpdateSelectedCameraPathKey();

        if (ImGui.Button("Play"))
            StartCameraPathPlayback();
        ImGui.SameLine();
        if (ImGui.Button("Play + Video"))
            StartCameraPathVideoCapture();
        ImGui.SameLine();
        if (ImGui.Button("Stop"))
            StopCameraPathPlayback();
        ImGui.SameLine();
        ImGui.Checkbox("Loop", ref _cameraPathLoop);
        ImGui.SameLine();
        ImGui.Checkbox("3D Overlay", ref _showCameraPathOverlay);

        if (ImGui.Button("Queue Key Stills"))
            QueueCameraPathKeyCaptures(includeUi: false);
        ImGui.SameLine();
        if (ImGui.Button("Queue Key Stills + UI"))
            QueueCameraPathKeyCaptures(includeUi: true);

        if (ImGui.Button("Delete Selected") && _selectedCameraPathKey >= 0 && _selectedCameraPathKey < _cameraPath.Keyframes.Count)
            DeleteSelectedCameraPathKey();

        ImGui.SameLine();
        if (ImGui.Button("Clear Path"))
        {
            StopCameraPathPlayback();
            _cameraPath.Keyframes.Clear();
            _selectedCameraPathKey = -1;
        }

        if (_cameraPathPlaying)
            ImGui.TextColored(new Vector4(0.3f, 1f, 0.5f, 1f), $"Playing {_cameraPathTimeSeconds:F2}s / {_cameraPath.DurationMs / 1000f:F2}s");

        if (ImGui.BeginChild("##camera_path_keys", new Vector2(0f, 210f), true))
        {
            for (int index = 0; index < _cameraPath.Keyframes.Count; index++)
            {
                M2CameraPathKeyframe key = _cameraPath.Keyframes[index];
                if (ImGui.Selectable($"{index + 1:D2}  {key.TimeMs / 1000f:F2}s##camera_path_key_{index}", index == _selectedCameraPathKey))
                    _selectedCameraPathKey = index;
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip($"pos={key.Position}\ntarget={key.Target}\nfov={key.FovDegrees:F1}\nroll={key.RollDegrees:F1}");
            }
        }
        ImGui.EndChild();

        if (_selectedCameraPathKey >= 0 && _selectedCameraPathKey < _cameraPath.Keyframes.Count)
            DrawSelectedCameraPathKey(_cameraPath.Keyframes[_selectedCameraPathKey]);

        ImGui.Separator();
        if (ImGui.Button("Save Path JSON"))
            SaveCameraPathJson();
        ImGui.SameLine();
        if (ImGui.Button("Save Native M2"))
            SaveCameraPathM2();
        ImGui.SameLine();
        if (ImGui.Button("Load Path JSON"))
            LoadCameraPathJson();

        string importPath = _cameraPathImportPath;
        if (ImGui.InputText("Import Client Asset", ref importPath, 1024))
            _cameraPathImportPath = importPath;
        int cameraIndex = _cameraPathImportCameraIndex;
        if (ImGui.DragInt("Camera Index", ref cameraIndex, 0.1f, 0, 64))
            _cameraPathImportCameraIndex = Math.Clamp(cameraIndex, 0, 64);
        int sequenceIndex = _cameraPathImportSequenceIndex;
        if (ImGui.DragInt("Sequence Index", ref sequenceIndex, 0.1f, 0, 64))
            _cameraPathImportSequenceIndex = Math.Clamp(sequenceIndex, 0, 64);
        int importInterval = _cameraPathImportSampleIntervalMs;
        if (ImGui.DragInt("Import Sample (ms)", ref importInterval, 1f, 16, 1000))
            _cameraPathImportSampleIntervalMs = Math.Clamp(importInterval, 16, 1000);
        if (ImGui.Button("Import Selected Client Camera"))
            ImportSelectedClientCameraPath();
        ImGui.SameLine();
        if (ImGui.Button("Import Loose Camera Asset"))
            ImportCameraPathM2();

        ImGui.TextDisabled("JSON is the lossless authored path; native M2 is an interoperability export with map/build metadata in the sidecar JSON.");
    }

    private bool HandleCameraPathKeyboardInput(IKeyboard keyboard, bool ctrlDown, bool shiftDown)
    {
        if (!_cameraPathKeyboardAuthoring || !IsCaptureKeyboardContextActive())
            return false;

        ImGuiIOPtr io = ImGui.GetIO();
        if (io.WantTextInput)
            return false;

        bool action = false;
        if (PressedOnce(keyboard.IsKeyPressed(Key.K), ref _cameraPathKeyWasPressed))
        {
            AddCameraPathKeyAtTime((int)Math.Round(_cameraPathTimeSeconds * 1000d));
            action = true;
        }
        if (PressedOnce(keyboard.IsKeyPressed(Key.U), ref _cameraPathUpdateWasPressed))
        {
            UpdateSelectedCameraPathKey();
            action = true;
        }
        if (PressedOnce(keyboard.IsKeyPressed(Key.Delete), ref _cameraPathDeleteWasPressed))
        {
            DeleteSelectedCameraPathKey();
            action = true;
        }
        if (PressedOnce(keyboard.IsKeyPressed(Key.Space), ref _cameraPathPlayWasPressed))
        {
            if (_cameraPathPlaying)
                StopCameraPathPlayback();
            else
                StartCameraPathPlayback();
            action = true;
        }
        if (PressedOnce(ctrlDown && keyboard.IsKeyPressed(Key.S), ref _cameraPathSaveWasPressed))
        {
            SaveCameraPathJson();
            action = true;
        }
        if (PressedOnce(ctrlDown && keyboard.IsKeyPressed(Key.E), ref _cameraPathExportWasPressed))
        {
            SaveCameraPathM2();
            action = true;
        }
        bool retimeLeft = ctrlDown && keyboard.IsKeyPressed(Key.Left);
        bool retimeRight = ctrlDown && keyboard.IsKeyPressed(Key.Right);
        if (PressedOnce(retimeLeft, ref _cameraPathRetimingLeftWasPressed))
        {
            NudgeSelectedCameraPathKeyTime(-_cameraPathKeyboardTimeStepMs);
            action = true;
        }
        if (PressedOnce(retimeRight, ref _cameraPathRetimingRightWasPressed))
        {
            NudgeSelectedCameraPathKeyTime(_cameraPathKeyboardTimeStepMs);
            action = true;
        }
        if (!ctrlDown && PressedOnce(keyboard.IsKeyPressed(Key.Left), ref _cameraPathLeftWasPressed))
        {
            SelectCameraPathKey(-1);
            action = true;
        }
        if (!ctrlDown && PressedOnce(keyboard.IsKeyPressed(Key.Right), ref _cameraPathRightWasPressed))
        {
            SelectCameraPathKey(1);
            action = true;
        }
        if (PressedOnce(keyboard.IsKeyPressed(Key.Home), ref _cameraPathHomeWasPressed))
        {
            _cameraPathTimeSeconds = 0d;
            ApplyCameraPathAtCurrentTime();
            action = true;
        }
        if (PressedOnce(keyboard.IsKeyPressed(Key.End), ref _cameraPathEndWasPressed))
        {
            _cameraPathTimeSeconds = _cameraPath.DurationMs / 1000d;
            ApplyCameraPathAtCurrentTime();
            action = true;
        }
        if (PressedOnce(ctrlDown && keyboard.IsKeyPressed(Key.Down), ref _cameraPathPreviousTimeWasPressed))
        {
            NudgeCameraPathPlayhead(-_cameraPathKeyboardTimeStepMs);
            action = true;
        }
        if (PressedOnce(ctrlDown && keyboard.IsKeyPressed(Key.Up), ref _cameraPathNextTimeWasPressed))
        {
            NudgeCameraPathPlayhead(_cameraPathKeyboardTimeStepMs);
            action = true;
        }

        bool rollLeft = keyboard.IsKeyPressed(Key.Z);
        bool rollRight = keyboard.IsKeyPressed(Key.X);
        if (rollLeft || rollRight)
        {
            float direction = (rollRight ? 1f : 0f) - (rollLeft ? 1f : 0f);
            _camera.Roll = NormalizeRollDegrees(_camera.Roll + direction * _cameraPathKeyboardRollStepDegrees * (shiftDown ? 5f : 1f));
            action = true;
        }
        return action;
    }

    private static bool PressedOnce(bool pressed, ref bool wasPressed)
    {
        bool triggered = pressed && !wasPressed;
        wasPressed = pressed;
        return triggered;
    }

    private void SelectCameraPathKey(int delta)
    {
        if (_cameraPath.Keyframes.Count == 0)
        {
            _selectedCameraPathKey = -1;
            return;
        }

        int current = _selectedCameraPathKey < 0 ? (delta < 0 ? 0 : -1) : _selectedCameraPathKey;
        _selectedCameraPathKey = Math.Clamp(current + delta, 0, _cameraPath.Keyframes.Count - 1);
    }

    private void NudgeCameraPathPlayhead(int deltaMs)
    {
        _cameraPathTimeSeconds = Math.Clamp(
            _cameraPathTimeSeconds + deltaMs / 1000d,
            0d,
            Math.Max(0d, _cameraPath.DurationMs / 1000d));
        ApplyCameraPathAtCurrentTime();
    }

    private void NudgeSelectedCameraPathKeyTime(int deltaMs)
    {
        if (_selectedCameraPathKey < 0 || _selectedCameraPathKey >= _cameraPath.Keyframes.Count)
            return;

        M2CameraPathKeyframe key = _cameraPath.Keyframes[_selectedCameraPathKey];
        key.TimeMs = Math.Max(0, key.TimeMs + deltaMs);
        M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        _selectedCameraPathKey = _cameraPath.Keyframes.IndexOf(key);
        _statusMessage = $"Retimed camera path key {_selectedCameraPathKey + 1} to {key.TimeMs} ms.";
    }

    private void DeleteSelectedCameraPathKey()
    {
        if (_selectedCameraPathKey < 0 || _selectedCameraPathKey >= _cameraPath.Keyframes.Count)
            return;

        _cameraPath.Keyframes.RemoveAt(_selectedCameraPathKey);
        _selectedCameraPathKey = Math.Clamp(_selectedCameraPathKey, -1, _cameraPath.Keyframes.Count - 1);
        M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        _statusMessage = "Deleted selected camera path key.";
    }

    private static float NormalizeRollDegrees(float roll)
    {
        while (roll > 180f)
            roll -= 360f;
        while (roll < -180f)
            roll += 360f;
        return roll;
    }

    private void DrawSelectedCameraPathKey(M2CameraPathKeyframe key)
    {
        M2CameraPathKeyframe selectedKey = key;
        int time = key.TimeMs;
        if (ImGui.DragInt("Time (ms)", ref time, 10f, 0, 3_600_000))
            key.TimeMs = time;
        float fov = key.FovDegrees;
        if (ImGui.DragFloat("FOV", ref fov, 0.25f, 1f, 179f))
            key.FovDegrees = fov;
        float roll = key.RollDegrees;
        if (ImGui.DragFloat("Roll (deg)", ref roll, 0.25f, -180f, 180f))
            key.RollDegrees = NormalizeRollDegrees(roll);
        Vector3 position = key.Position;
        if (ImGui.DragFloat3("Position", ref position, 0.5f))
            key.Position = position;
        Vector3 target = key.Target;
        if (ImGui.DragFloat3("Target", ref target, 0.5f))
            key.Target = target;
        M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        _selectedCameraPathKey = _cameraPath.Keyframes.IndexOf(selectedKey);
    }

    private void AddCameraPathKeyFromCurrentCamera()
    {
        if (_cameraPath.Keyframes.Count == 0)
            BindCameraPathToCurrentMap();

        int time = _cameraPath.Keyframes.Count == 0
            ? 0
            : _cameraPath.DurationMs + _cameraPathDefaultKeySpacingMs;
        AddCameraPathKeyAtTime(time);
    }

    private void AddCameraPathKeyAtTime(int time)
    {
        if (_cameraPath.Keyframes.Count == 0)
            BindCameraPathToCurrentMap();

        M2CameraPathKeyframe key = new()
        {
            TimeMs = Math.Max(0, time),
            Position = _camera.Position,
            Target = _camera.Position + _camera.Forward * 100f,
            FovDegrees = _fovDegrees,
            RollDegrees = _camera.Roll,
        };
        _cameraPath.Keyframes.Add(key);
        _selectedCameraPathKey = _cameraPath.Keyframes.Count - 1;
        _statusMessage = $"Added camera path key {_selectedCameraPathKey + 1} on {GetCurrentCaptureMapName()}.";
    }

    private void UpdateSelectedCameraPathKey()
    {
        if (_selectedCameraPathKey < 0 || _selectedCameraPathKey >= _cameraPath.Keyframes.Count)
            return;
        M2CameraPathKeyframe key = _cameraPath.Keyframes[_selectedCameraPathKey];
        key.Position = _camera.Position;
        key.Target = _camera.Position + _camera.Forward * 100f;
        key.FovDegrees = _fovDegrees;
        key.RollDegrees = _camera.Roll;
        _statusMessage = $"Updated camera path key {_selectedCameraPathKey + 1}.";
    }

    private bool IsCameraPathBoundToCurrentMap()
        => string.Equals(_cameraPath.MapName, GetCurrentCaptureMapName(), StringComparison.OrdinalIgnoreCase)
            && string.Equals(_cameraPath.BuildVersion, GetCurrentCaptureBuildVersion(), StringComparison.OrdinalIgnoreCase);

    private bool StartCameraPathPlayback()
    {
        if (_cameraPath.Keyframes.Count < 2)
        {
            _statusMessage = "Camera path playback requires at least two keys.";
            return false;
        }

        EnsureCameraPathBindingForCurrentMap();
        if (!IsCameraPathBoundToCurrentMap())
        {
            _statusMessage = $"Camera path is bound to {_cameraPath.MapName}/{_cameraPath.BuildVersion}; load that map/build before playback.";
            return false;
        }

        if (_cameraPathPreloadEnabled)
        {
            if (_cameraPathPreload == null && !BeginCameraPathPreload())
                return false;
            if (_cameraPathPreload is { Ready: false })
            {
                _cameraPathPlaybackPending = true;
                _statusMessage = $"Warming {_cameraPathPreload.Tiles.Count} path tiles before playback.";
                return false;
            }
        }

        _cameraPathPlaybackPending = false;
        _cameraPathTimeSeconds = 0;
        _cameraPathPlaying = true;
        _taxiRideCameraEnabled = false;
        _statusMessage = $"Playing camera path '{_cameraPath.Name}'.";
        return true;
    }

    private void StartCameraPathVideoCapture()
    {
        if (!ValidateCameraPathForPlayback())
            return;

        if (_cameraPathPreloadEnabled)
        {
            if (!BeginCameraPathPreload())
                return;

            _cameraPathVideoCapturePending = true;
            _statusMessage = $"Warming {_cameraPathPreload?.Tiles.Count ?? 0} path tiles and their objects before video capture.";
            return;
        }

        StartCameraPathVideoCaptureNow();
    }

    private void StartCameraPathVideoCaptureNow()
    {
        if (!StartCameraPathPlayback())
            return;

        if (TryStartCurrentViewVideoRecording(_videoCaptureIncludeUi, _cameraPath.Name))
        {
            _cameraPathVideoCaptureActive = true;
            return;
        }

        _cameraPathPlaying = false;
        EndCameraPathPreload();
    }

    private bool ValidateCameraPathForPlayback()
    {
        if (_cameraPath.Keyframes.Count < 2)
        {
            _statusMessage = "Camera path playback requires at least two keys.";
            return false;
        }

        EnsureCameraPathBindingForCurrentMap();
        if (!IsCameraPathBoundToCurrentMap())
        {
            _statusMessage = $"Camera path is bound to {_cameraPath.MapName}/{_cameraPath.BuildVersion}; load that map/build before playback.";
            return false;
        }

        return true;
    }

    private void StopCameraPathPlayback()
    {
        _cameraPathPlaying = false;
        _cameraPathTimeSeconds = 0;
        _cameraPathPlaybackPending = false;
        _cameraPathVideoCapturePending = false;
        if (_cameraPathVideoCaptureActive)
        {
            StopVideoRecording("Camera path video capture stopped.");
            _cameraPathVideoCaptureActive = false;
        }

        if (_captureQueue.Count == 0 && _activeCaptureRequest == null)
            EndCameraPathPreload();
    }

    private void UpdateCameraPathPlayback(double dt)
    {
        if (!_cameraPathPlaying)
            return;
        if (!IsCameraPathBoundToCurrentMap())
        {
            StopCameraPathPlayback();
            _statusMessage = "Camera path playback stopped because the active map/build changed.";
            return;
        }

        _cameraPathTimeSeconds += Math.Max(0, dt);
        double durationSeconds = Math.Max(0.001, _cameraPath.DurationMs / 1000.0);
        if (_cameraPathTimeSeconds >= durationSeconds && !_cameraPathLoop)
        {
            _cameraPathTimeSeconds = durationSeconds;
            _cameraPathPlaying = false;
            if (_cameraPathVideoCaptureActive)
            {
                StopVideoRecording("Camera path video capture completed.");
                _cameraPathVideoCaptureActive = false;
                EndCameraPathPreload();
            }
            else if (_captureQueue.Count == 0 && _activeCaptureRequest == null)
            {
                EndCameraPathPreload();
            }
        }

        ApplyCameraPathAtCurrentTime();
    }

    private void ApplyCameraPathAtCurrentTime()
    {
        if (_cameraPath.Keyframes.Count == 0)
            return;

        M2CameraPathSample sample = M2CameraPathEvaluator.Sample(_cameraPath, (int)Math.Round(_cameraPathTimeSeconds * 1000.0), _cameraPathLoop);
        Vector3 previousPosition = _camera.Position;
        Vector3 resolvedPosition = sample.Position;
        if (_worldScene != null && (_cameraPath.TerrainCollisionEnabled || _cameraPath.WmoCollisionEnabled))
        {
            _worldScene.TryResolveCameraPathCollision(
                previousPosition,
                sample.Position,
                _cameraPath.CollisionClearance,
                _cameraPath.TerrainCollisionEnabled,
                _cameraPath.WmoCollisionEnabled,
                out resolvedPosition);
        }

        _camera.Position = resolvedPosition;
        _camera.Roll = sample.RollDegrees;
        Vector3 direction = sample.Target - resolvedPosition;
        if (direction.LengthSquared() > 0.0001f)
        {
            direction = Vector3.Normalize(direction);
            _camera.Yaw = MathF.Atan2(direction.Y, direction.X) * (180f / MathF.PI);
            _camera.Pitch = MathF.Asin(Math.Clamp(direction.Z, -1f, 1f)) * (180f / MathF.PI);
        }
        _fovDegrees = Math.Clamp(sample.FovDegrees, 1f, 179f);
    }

    private void QueueCameraPathKeyCaptures(bool includeUi)
    {
        if (_cameraPath.Keyframes.Count == 0)
        {
            _statusMessage = "Add at least one camera key before queuing captures.";
            return;
        }

        if (!IsCameraPathBoundToCurrentMap())
        {
            _statusMessage = $"Camera path is bound to {_cameraPath.MapName}/{_cameraPath.BuildVersion}; load that map/build before queuing captures.";
            return;
        }

        M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        if (_cameraPathPreloadEnabled && !BeginCameraPathPreload())
            return;

        int queued = 0;
        for (int index = 0; index < _cameraPath.Keyframes.Count; index++)
        {
            M2CameraPathKeyframe key = _cameraPath.Keyframes[index];
            Vector3 direction = key.Target - key.Position;
            if (direction.LengthSquared() < 0.0001f)
                direction = Vector3.UnitX;
            direction = Vector3.Normalize(direction);

            CameraShotPoint shot = new()
            {
                Name = $"{_cameraPath.Name}_key_{index + 1:D2}",
                MapName = _cameraPath.MapName,
                BuildVersion = _cameraPath.BuildVersion,
                PositionX = key.Position.X,
                PositionY = key.Position.Y,
                PositionZ = key.Position.Z,
                Yaw = MathF.Atan2(direction.Y, direction.X) * (180f / MathF.PI),
                Pitch = MathF.Asin(Math.Clamp(direction.Z, -1f, 1f)) * (180f / MathF.PI),
                Roll = key.RollDegrees,
                FovDegrees = key.FovDegrees,
            };
            EnqueueShotCapture(
                shot,
                includeUi,
                exitAfterCapture: false,
                options: new CaptureQueueOptions
                {
                    WaitForSceneReady = true,
                    RequiredSettledFrames = 2,
                    MaxFramesBeforeCapture = 120,
                    CaptureLabel = shot.Name,
                    RequiresCameraPathPreload = _cameraPathPreloadEnabled,
                });
            queued++;
        }

        _statusMessage = $"Queued {queued} camera path key capture{(queued == 1 ? string.Empty : "s")} through the existing capture queue.";
    }

    private bool BeginCameraPathPreload()
    {
        if (_terrainManager == null || _worldScene == null)
        {
            _statusMessage = "Path preload requires an active streamed world scene.";
            return false;
        }

        if (_cameraPath.Keyframes.Count == 0)
        {
            _statusMessage = "Add at least one camera key before warming a path.";
            return false;
        }

        if (!IsCameraPathBoundToCurrentMap())
        {
            _statusMessage = $"Camera path is bound to {_cameraPath.MapName}/{_cameraPath.BuildVersion}; load that map/build before warming it.";
            return false;
        }

        M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        HashSet<(int tileX, int tileY)> tiles = BuildCameraPathPreloadTiles();
        if (tiles.Count == 0)
        {
            _statusMessage = "The camera path does not intersect any available terrain tiles.";
            return false;
        }

        if (tiles.Count > MaxCameraPathPreloadTiles)
        {
            _statusMessage = $"Path warmup needs {tiles.Count} tiles; reduce the path length, tile radius, or sample spacing (limit {MaxCameraPathPreloadTiles}).";
            return false;
        }

        _terrainManager.SetCapturePreloadTiles(tiles);
        _worldScene.CapturePreloadActive = true;
        _worldScene.QueueCapturePreloadAssets(tiles);
        _cameraPathPreload = new CameraPathPreloadState { Tiles = tiles };
        return true;
    }

    private void UpdateCameraPathPreload()
    {
        if (_cameraPathPreload is not { } preload || _terrainManager == null || _worldScene == null)
            return;

        bool tilesReady = preload.Tiles.All(tile => _terrainManager.IsTileLoaded(tile.tileX, tile.tileY));
        // Normal AOI streaming may still be loading unrelated tiles. The capture
        // gate only needs the bounded path tile set resident; waiting on the
        // global stream made Play + Video look dead on large maps.
        bool terrainReady = tilesReady;
        bool objectsReady = _worldScene.PendingCapturePreloadLoadCount == 0;
        if (!terrainReady || !objectsReady)
        {
            preload.StableFrames = 0;
            return;
        }

        preload.StableFrames++;
        if (!preload.Ready && preload.StableFrames >= 2)
        {
            preload.Ready = true;
            _statusMessage = $"Path preload ready: {preload.Tiles.Count} tiles and all queued world objects are resident.";
        }

        if (_cameraPathVideoCapturePending && preload.Ready)
        {
            _cameraPathVideoCapturePending = false;
            StartCameraPathVideoCaptureNow();
        }
        else if (_cameraPathPlaybackPending && preload.Ready)
        {
            _cameraPathPlaybackPending = false;
            StartCameraPathPlayback();
        }
    }

    private void EndCameraPathPreload()
    {
        _cameraPathVideoCapturePending = false;
        _cameraPathPlaybackPending = false;
        if (_worldScene != null)
            _worldScene.CapturePreloadActive = false;
        _terrainManager?.ClearCapturePreloadTiles();
        _cameraPathPreload = null;
    }

    private HashSet<(int tileX, int tileY)> BuildCameraPathPreloadTiles()
    {
        return CameraPathTileFootprintSelector.GetTiles(
            _cameraPath,
            WoWConstants.MapOrigin,
            WoWConstants.TileSize,
            WoWConstants.TilesPerMapEdge,
            Math.Max(100, _cameraPathPreloadSampleSpacingMs),
            _cameraPathPreloadTileRadius,
            (tileX, tileY) => _terrainManager?.Adapter.TileExists(tileX, tileY) == true);
    }

    private void DrawCameraPathOverlay(Terrain.BoundingBoxRenderer overlay)
    {
        if (!_showCameraPathOverlay || _cameraPath.Keyframes.Count == 0)
            return;

        M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        const int sampleCount = 96;
        const float pathHeight = 0.35f;
        Vector3 pathColor = new(1f, 0.65f, 0.1f);
        Vector3 targetColor = new(0.35f, 0.85f, 1f);
        Vector3 keyColor = new(1f, 0.25f, 0.1f);

        if (_cameraPath.Keyframes.Count > 1)
        {
            Vector3 previous = M2CameraPathEvaluator.Sample(_cameraPath, 0).Position;
            for (int index = 1; index <= sampleCount; index++)
            {
                int time = (int)MathF.Round(_cameraPath.DurationMs * (index / (float)sampleCount));
                Vector3 current = M2CameraPathEvaluator.Sample(_cameraPath, time).Position;
                overlay.BatchLine(previous, current, pathColor);
                previous = current;
            }
        }

        foreach (M2CameraPathKeyframe key in _cameraPath.Keyframes)
        {
            overlay.BatchOctahedron(key.Position, pathHeight * 3f, keyColor);
            overlay.BatchLine(key.Position, key.Target, targetColor);
        }
    }

    private void SaveCameraPathJson()
    {
        if (_cameraPath.Keyframes.Count == 0)
        {
            _statusMessage = "Add at least one camera key before saving a path.";
            return;
        }

        BindCameraPathToCurrentMap();
        string? path = ShowSaveFileDialogSTA("Save WoWViewer camera path", "Camera Path (*.m2cam.json)|*.m2cam.json|JSON Files (*.json)|*.json", Path.GetDirectoryName(_cameraPathFilePath), $"{_cameraPath.Name}.m2cam.json");
        if (string.IsNullOrWhiteSpace(path))
            return;
        try
        {
            M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
            File.WriteAllText(path, JsonSerializer.Serialize(_cameraPath, M2CameraPathJson.CreateOptions(writeIndented: true)));
            _cameraPathFilePath = path;
            _statusMessage = $"Saved camera path: {path}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to save camera path: {ex.Message}";
        }
    }

    private void SaveCameraPathM2()
    {
        if (_cameraPath.Keyframes.Count == 0)
        {
            _statusMessage = "Add at least one camera key before exporting M2.";
            return;
        }

        BindCameraPathToCurrentMap();
        string? path = ShowSaveFileDialogSTA("Save native M2 camera path", "M2 Files (*.m2)|*.m2", Path.GetDirectoryName(_cameraPathFilePath), $"{_cameraPath.Name}.m2");
        if (string.IsNullOrWhiteSpace(path))
            return;
        try
        {
            M2CameraPathWriter.Write(path, _cameraPath);
            string sidecar = path + ".json";
            File.WriteAllText(sidecar, JsonSerializer.Serialize(_cameraPath, M2CameraPathJson.CreateOptions(writeIndented: true)));
            _cameraPathFilePath = path;
            _statusMessage = $"Saved native M2 camera path and metadata sidecar: {path}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to export camera path M2: {ex.Message}";
        }
    }

    private void LoadCameraPathJson()
    {
        string? path = ShowFileDialogSTA("Load WoWViewer camera path", "Camera Path (*.m2cam.json;*.json)|*.m2cam.json;*.json|JSON Files (*.json)|*.json", Path.GetDirectoryName(_cameraPathFilePath));
        if (string.IsNullOrWhiteSpace(path))
            return;
        try
        {
            M2CameraPathDocument? loaded = JsonSerializer.Deserialize<M2CameraPathDocument>(File.ReadAllText(path), M2CameraPathJson.CreateOptions());
            if (loaded == null)
                throw new InvalidDataException("The camera path document was empty.");
            M2CameraPathEvaluator.NormalizeAndValidate(loaded);
            _cameraPath.Format = loaded.Format;
            _cameraPath.Name = loaded.Name;
            _cameraPath.MapName = loaded.MapName;
            _cameraPath.BuildVersion = loaded.BuildVersion;
            _cameraPath.Interpolation = loaded.Interpolation;
            _cameraPath.TerrainCollisionEnabled = loaded.TerrainCollisionEnabled;
            _cameraPath.WmoCollisionEnabled = loaded.WmoCollisionEnabled;
            _cameraPath.CollisionClearance = loaded.CollisionClearance;
            _cameraPath.CoordinatesAreWorldSpace = loaded.CoordinatesAreWorldSpace;
            _cameraPath.HasCinematicCameraOrigin = loaded.HasCinematicCameraOrigin;
            _cameraPath.CinematicCameraId = loaded.CinematicCameraId;
            _cameraPath.CinematicCameraModel = loaded.CinematicCameraModel;
            _cameraPath.CinematicCameraOrigin = loaded.CinematicCameraOrigin;
            _cameraPath.CinematicCameraOriginFacingRadians = loaded.CinematicCameraOriginFacingRadians;
            _cameraPath.CinematicCameraOriginTileX = loaded.CinematicCameraOriginTileX;
            _cameraPath.CinematicCameraOriginTileY = loaded.CinematicCameraOriginTileY;
            _cameraPath.CinematicCameraOriginSource = loaded.CinematicCameraOriginSource;
            _cameraPath.Keyframes = loaded.Keyframes;
            _cameraPathName = loaded.Name;
            _cameraPathFilePath = path;
            _selectedCameraPathKey = -1;
            _statusMessage = $"Loaded camera path '{loaded.Name}' ({loaded.Keyframes.Count} keys).";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to load camera path: {ex.Message}";
        }
    }

    private void ImportCameraPathM2()
    {
        string path = _cameraPathImportPath.Trim();
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            path = ShowFileDialogSTA("Import M2 camera path", "M2 Files (*.m2)|*.m2", Path.GetDirectoryName(_cameraPathImportPath)) ?? string.Empty;
        }
        if (string.IsNullOrWhiteSpace(path))
            return;
        try
        {
            using FileStream stream = File.OpenRead(path);
            M2ModelDocument model = M2ModelReaderDispatcher.Read(stream, path);
            M2CameraPathDocument imported = M2CameraPathImporter.Import(model);
            string metadataPath = path + ".json";
            if (File.Exists(metadataPath))
            {
                M2CameraPathDocument? metadata = JsonSerializer.Deserialize<M2CameraPathDocument>(File.ReadAllText(metadataPath), M2CameraPathJson.CreateOptions());
                if (metadata != null)
                {
                    imported.MapName = metadata.MapName;
                    imported.BuildVersion = metadata.BuildVersion;
                    imported.Name = metadata.Name;
                }
            }
            else
            {
                imported.MapName = GetCurrentCaptureMapName();
                imported.BuildVersion = GetCurrentCaptureBuildVersion();
            }
            imported.CoordinatesAreWorldSpace = true;
            ApplyImportedCameraPath(imported, path);
            _cameraPathName = imported.Name;
            _statusMessage = $"Imported M2 camera '{Path.GetFileName(path)}' as {imported.Keyframes.Count} keys for {imported.MapName}.";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to import M2 camera: {ex.Message}";
        }
    }

    private void ImportSelectedClientCameraPath()
    {
        string assetPath;
        if (!TryGetSelectedBrowserAssetPath(out assetPath))
        {
            assetPath = _cameraPathImportPath.Trim();
            if (string.IsNullOrWhiteSpace(assetPath))
            {
                _statusMessage = "Select an .m2 or .mdx camera asset in the loaded client file browser first.";
                return;
            }
        }

        string extension = Path.GetExtension(assetPath).ToLowerInvariant();
        if (extension is not ".m2" and not ".mdx")
        {
            _statusMessage = "The selected client asset is not an M2 or MDX camera asset.";
            return;
        }

        byte[]? bytes = ReadStandaloneFileData(assetPath);
        if (bytes == null || bytes.Length == 0)
        {
            _statusMessage = $"Could not read client asset '{assetPath}' from the loaded data source.";
            return;
        }

        try
        {
            M2CameraPathDocument imported;
            if (extension == ".m2")
            {
                using MemoryStream stream = new(bytes, writable: false);
                M2ModelDocument model = M2ModelReaderDispatcher.Read(stream, assetPath);
                imported = M2CameraPathImporter.Import(model, _cameraPathImportCameraIndex, _cameraPathImportSequenceIndex, _cameraPathImportSampleIntervalMs);
            }
            else
            {
                using MemoryStream stream = new(bytes, writable: false);
                imported = MdxCameraPathImporter.Import(stream, assetPath, _cameraPathImportCameraIndex, _cameraPathImportSequenceIndex, _cameraPathImportSampleIntervalMs);
            }

            imported.MapName = GetCurrentCaptureMapName();
            imported.BuildVersion = GetCurrentCaptureBuildVersion();
            CinematicCameraOrigin? origin = TryResolveCinematicCameraOrigin(assetPath);
            if (origin != null)
            {
                M2CameraPathPlacement.ApplyCinematicCameraOrigin(
                    imported,
                    origin.Id,
                    origin.Model,
                    origin.Origin,
                    origin.OriginFacingRadians,
                    origin.TileX,
                    origin.TileY);
                _statusMessage = $"Resolved {Path.GetFileName(assetPath)} to CinematicCamera.dbc origin tile ({origin.TileX}, {origin.TileY}).";
            }
            else
            {
                imported.CoordinatesAreWorldSpace = true;
                _statusMessage = $"Imported {extension.TrimStart('.').ToUpperInvariant()} camera without CinematicCamera.dbc origin; left track coordinates unchanged.";
            }
            ApplyImportedCameraPath(imported, assetPath);
            _statusMessage += $" Imported client {extension.TrimStart('.').ToUpperInvariant()} camera '{Path.GetFileName(assetPath)}' as {imported.Keyframes.Count} keys.";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to import client camera '{Path.GetFileName(assetPath)}': {ex.Message}";
        }
    }

    private void ApplyImportedCameraPath(M2CameraPathDocument imported, string sourcePath)
    {
        _cameraPath.Format = imported.Format;
        _cameraPath.Name = imported.Name;
        _cameraPath.MapName = imported.MapName;
        _cameraPath.BuildVersion = imported.BuildVersion;
        _cameraPath.Interpolation = imported.Interpolation;
        _cameraPath.TerrainCollisionEnabled = imported.TerrainCollisionEnabled;
        _cameraPath.WmoCollisionEnabled = imported.WmoCollisionEnabled;
        _cameraPath.CollisionClearance = imported.CollisionClearance;
        _cameraPath.CoordinatesAreWorldSpace = imported.CoordinatesAreWorldSpace;
        _cameraPath.HasCinematicCameraOrigin = imported.HasCinematicCameraOrigin;
        _cameraPath.CinematicCameraId = imported.CinematicCameraId;
        _cameraPath.CinematicCameraModel = imported.CinematicCameraModel;
        _cameraPath.CinematicCameraOrigin = imported.CinematicCameraOrigin;
        _cameraPath.CinematicCameraOriginFacingRadians = imported.CinematicCameraOriginFacingRadians;
        _cameraPath.CinematicCameraOriginTileX = imported.CinematicCameraOriginTileX;
        _cameraPath.CinematicCameraOriginTileY = imported.CinematicCameraOriginTileY;
        _cameraPath.CinematicCameraOriginSource = imported.CinematicCameraOriginSource;
        _cameraPath.Keyframes = imported.Keyframes;
        _cameraPathName = imported.Name;
        _cameraPathImportPath = sourcePath;
        _cameraPathFilePath = string.Empty;
        _selectedCameraPathKey = -1;
        _cameraPathTimeSeconds = 0;
    }

    private CinematicCameraOrigin? TryResolveCinematicCameraOrigin(string assetPath)
    {
        if (_dbcProvider == null
            || string.IsNullOrWhiteSpace(_dbdDir)
            || string.IsNullOrWhiteSpace(_dbcBuild))
            return null;

        var resolver = new CinematicCameraOriginResolver();
        return resolver.TryResolve(_dbcProvider, _dbdDir, _dbcBuild, assetPath, out CinematicCameraOrigin? origin)
            ? origin
            : null;
    }

    private void BindCameraPathToCurrentMap()
    {
        _cameraPath.Name = string.IsNullOrWhiteSpace(_cameraPathName) ? "camera_path" : _cameraPathName.Trim();
        _cameraPath.MapName = GetCurrentCaptureMapName();
        _cameraPath.BuildVersion = GetCurrentCaptureBuildVersion();
    }

    private void EnsureCameraPathBindingForCurrentMap()
    {
        bool missingMap = string.IsNullOrWhiteSpace(_cameraPath.MapName)
            || string.Equals(_cameraPath.MapName, "unknown", StringComparison.OrdinalIgnoreCase)
            || string.Equals(_cameraPath.MapName, "standalone", StringComparison.OrdinalIgnoreCase);
        bool missingBuild = string.IsNullOrWhiteSpace(_cameraPath.BuildVersion)
            || string.Equals(_cameraPath.BuildVersion, "unknown_build", StringComparison.OrdinalIgnoreCase);
        if (missingMap || missingBuild)
            BindCameraPathToCurrentMap();
    }
}
