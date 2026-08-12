using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.M2;
using WowViewer.Core.IO.M2;
using WowViewer.Core.Runtime.M2;

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

        string name = _cameraPathName;
        if (ImGui.InputText("Path Name", ref name, 128))
            _cameraPathName = name;

        ImGui.TextDisabled($"Keys: {_cameraPath.Keyframes.Count}  Duration: {_cameraPath.DurationMs / 1000f:F2}s");
        int spacing = _cameraPathDefaultKeySpacingMs;
        if (ImGui.DragInt("New Key Spacing (ms)", ref spacing, 10f, 100, 60000))
            _cameraPathDefaultKeySpacingMs = Math.Clamp(spacing, 100, 60000);

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
        {
            _cameraPath.Keyframes.RemoveAt(_selectedCameraPathKey);
            _selectedCameraPathKey = Math.Clamp(_selectedCameraPathKey, -1, _cameraPath.Keyframes.Count - 1);
            M2CameraPathEvaluator.NormalizeAndValidate(_cameraPath);
        }

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
                    ImGui.SetTooltip($"pos={key.Position}\ntarget={key.Target}\nfov={key.FovDegrees:F1}");
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
        if (ImGui.InputText("Import M2 Path", ref importPath, 1024))
            _cameraPathImportPath = importPath;
        if (ImGui.Button("Import M2 Camera"))
            ImportCameraPathM2();

        ImGui.TextDisabled("JSON is the lossless authored path; native M2 is an interoperability export with map/build metadata in the sidecar JSON.");
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
        M2CameraPathKeyframe key = new()
        {
            TimeMs = time,
            Position = _camera.Position,
            Target = _camera.Position + _camera.Forward * 100f,
            FovDegrees = _fovDegrees,
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

        if (!IsCameraPathBoundToCurrentMap())
        {
            _statusMessage = $"Camera path is bound to {_cameraPath.MapName}/{_cameraPath.BuildVersion}; load that map/build before playback.";
            return false;
        }

        _cameraPathTimeSeconds = 0;
        _cameraPathPlaying = true;
        _taxiRideCameraEnabled = false;
        _statusMessage = $"Playing camera path '{_cameraPath.Name}'.";
        return true;
    }

    private void StartCameraPathVideoCapture()
    {
        if (!StartCameraPathPlayback())
            return;

        if (TryStartCurrentViewVideoRecording(_videoCaptureIncludeUi, _cameraPath.Name))
        {
            _cameraPathVideoCaptureActive = true;
            return;
        }

        _cameraPathPlaying = false;
    }

    private void StopCameraPathPlayback()
    {
        _cameraPathPlaying = false;
        _cameraPathTimeSeconds = 0;
        if (_cameraPathVideoCaptureActive)
        {
            StopVideoRecording("Camera path video capture stopped.");
            _cameraPathVideoCaptureActive = false;
        }
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
            }
        }

        M2CameraPathSample sample = M2CameraPathEvaluator.Sample(_cameraPath, (int)Math.Round(_cameraPathTimeSeconds * 1000.0), _cameraPathLoop);
        _camera.Position = sample.Position;
        Vector3 direction = sample.Target - sample.Position;
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
                });
            queued++;
        }

        _statusMessage = $"Queued {queued} camera path key capture{(queued == 1 ? string.Empty : "s")} through the existing capture queue.";
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
            File.WriteAllText(path, JsonSerializer.Serialize(_cameraPath, new JsonSerializerOptions { WriteIndented = true }));
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
            File.WriteAllText(sidecar, JsonSerializer.Serialize(_cameraPath, new JsonSerializerOptions { WriteIndented = true }));
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
            M2CameraPathDocument? loaded = JsonSerializer.Deserialize<M2CameraPathDocument>(File.ReadAllText(path));
            if (loaded == null)
                throw new InvalidDataException("The camera path document was empty.");
            M2CameraPathEvaluator.NormalizeAndValidate(loaded);
            _cameraPath.Format = loaded.Format;
            _cameraPath.Name = loaded.Name;
            _cameraPath.MapName = loaded.MapName;
            _cameraPath.BuildVersion = loaded.BuildVersion;
            _cameraPath.Interpolation = loaded.Interpolation;
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
            M2ModelDocument model = M2ModelReader.Read(stream, path);
            M2CameraPathDocument imported = M2CameraPathImporter.Import(model);
            string metadataPath = path + ".json";
            if (File.Exists(metadataPath))
            {
                M2CameraPathDocument? metadata = JsonSerializer.Deserialize<M2CameraPathDocument>(File.ReadAllText(metadataPath));
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
            _cameraPath.Format = imported.Format;
            _cameraPath.Name = imported.Name;
            _cameraPath.MapName = imported.MapName;
            _cameraPath.BuildVersion = imported.BuildVersion;
            _cameraPath.Interpolation = imported.Interpolation;
            _cameraPath.Keyframes = imported.Keyframes;
            _cameraPathName = imported.Name;
            _cameraPathFilePath = string.Empty;
            _selectedCameraPathKey = -1;
            _statusMessage = $"Imported M2 camera '{Path.GetFileName(path)}' as {imported.Keyframes.Count} keys for {imported.MapName}.";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Failed to import M2 camera: {ex.Message}";
        }
    }

    private void BindCameraPathToCurrentMap()
    {
        _cameraPath.Name = string.IsNullOrWhiteSpace(_cameraPathName) ? "camera_path" : _cameraPathName.Trim();
        _cameraPath.MapName = GetCurrentCaptureMapName();
        _cameraPath.BuildVersion = GetCurrentCaptureBuildVersion();
    }
}
