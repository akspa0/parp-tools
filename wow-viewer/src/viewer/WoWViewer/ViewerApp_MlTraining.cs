using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using ImGuiNET;

namespace WoWViewer;

public partial class ViewerApp
{
    private bool _showMlTrainingDialog;
    private string _mlTrainingPythonExecutable = "python";
    private string _mlTrainingScriptPath = ResolveDefaultMlTrainingScriptPath();
    private string _mlTrainingProfile = "development-map";
    private string _mlTrainingDatasetRootsText = string.Empty;
    private string _mlTrainingIncludeMapsText = "Northrend\nLostIsles";
    private string _mlTrainingExcludeMapsText = string.Empty;
    private string _mlTrainingOutputDir = Path.Combine(OutputDir, "ml-training", "v7-development-map");
    private string _mlTrainingResumeCheckpointPath = string.Empty;
    private int _mlTrainingBatchSize = 8;
    private int _mlTrainingEpochs = 250;
    private int _mlTrainingPatience = 30;
    private int _mlTrainingSpatialGroupSize = 4;
    private int _mlTrainingSeed = 1337;
    private int _mlTrainingLimit = 0;
    private float _mlTrainingLearningRate = 1e-4f;
    private float _mlTrainingValFraction = 0.12f;
    private bool _mlTrainingNoAugment;
    private readonly List<string> _mlTrainingLog = new();
    private bool _mlTrainingScrollToBottom;
    private Process? _mlTrainingProcess;
    private DateTime _mlTrainingNextPollUtc = DateTime.MinValue;
    private DateTime _mlTrainingLogLastWriteUtc = DateTime.MinValue;
    private MlTrainingHistorySnapshot? _mlTrainingHistory;
    private string? _mlTrainingStatus;
    private string? _mlTrainingLastNotebookPath;
    private string? _mlTrainingHistoryError;

    private sealed class MlTrainingHistorySnapshot
    {
        public List<int> Epochs { get; } = new();
        public List<float> TrainLoss { get; } = new();
        public List<float> ValLoss { get; } = new();
        public Dictionary<string, List<float>> Components { get; } = new(StringComparer.OrdinalIgnoreCase);
        public int LatestEpoch => Epochs.Count > 0 ? Epochs[^1] : 0;
        public float LatestTrainLoss => TrainLoss.Count > 0 ? TrainLoss[^1] : 0f;
        public float LatestValLoss => ValLoss.Count > 0 ? ValLoss[^1] : 0f;

        public float BestValLoss
        {
            get
            {
                if (ValLoss.Count == 0)
                    return 0f;

                float best = ValLoss[0];
                for (int i = 1; i < ValLoss.Count; i++)
                {
                    if (ValLoss[i] < best)
                        best = ValLoss[i];
                }

                return best;
            }
        }
    }

    private sealed class MlTrainingSeries
    {
        public required IReadOnlyList<float> Values { get; init; }
        public required uint Color { get; init; }
    }

    private static string ResolveDefaultMlTrainingScriptPath()
    {
        string[] candidates =
        [
            Path.Combine(Environment.CurrentDirectory, "src", "WoWMapConverter", "scripts", "train_v7.py"),
            Path.Combine(Environment.CurrentDirectory, "gillijimproject_refactor", "src", "WoWMapConverter", "scripts", "train_v7.py"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "WoWMapConverter", "scripts", "train_v7.py")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "gillijimproject_refactor", "src", "WoWMapConverter", "scripts", "train_v7.py")),
        ];

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
                return candidate;
        }

        return candidates[0];
    }

    private void PrepareMlTrainingDialogInputs()
    {
        if (string.IsNullOrWhiteSpace(_mlTrainingScriptPath))
            _mlTrainingScriptPath = ResolveDefaultMlTrainingScriptPath();

        if (string.IsNullOrWhiteSpace(_mlTrainingOutputDir))
            _mlTrainingOutputDir = Path.Combine(OutputDir, "ml-training", "v7-development-map");

        if (string.IsNullOrWhiteSpace(_mlTrainingIncludeMapsText) && string.Equals(_mlTrainingProfile, "development-map", StringComparison.OrdinalIgnoreCase))
            _mlTrainingIncludeMapsText = "Northrend\nLostIsles";
    }

    private bool IsMlTrainingProcessActive()
    {
        try
        {
            return _mlTrainingProcess != null && !_mlTrainingProcess.HasExited;
        }
        catch
        {
            return false;
        }
    }

    private void UpdateMlTrainingMonitor()
    {
        if (_mlTrainingProcess != null)
        {
            try
            {
                if (_mlTrainingProcess.HasExited)
                    FinalizeMlTrainingProcessExit();
            }
            catch
            {
                FinalizeMlTrainingProcessExit();
            }
        }

        if (DateTime.UtcNow < _mlTrainingNextPollUtc)
            return;

        _mlTrainingNextPollUtc = DateTime.UtcNow.AddSeconds(1);
        TryLoadMlTrainingHistory();
    }

    private void FinalizeMlTrainingProcessExit()
    {
        if (_mlTrainingProcess == null)
            return;

        int exitCode = -1;
        try
        {
            exitCode = _mlTrainingProcess.ExitCode;
        }
        catch
        {
            // Ignore exit-code retrieval failures during shutdown.
        }

        AppendMlTrainingLog($"=== Training process exited with code {exitCode} ===");
        _mlTrainingStatus = exitCode == 0
            ? "Training completed."
            : $"Training stopped with exit code {exitCode}.";

        _mlTrainingProcess.Dispose();
        _mlTrainingProcess = null;
    }

    private void ShutdownMlTrainingMonitor()
    {
        if (_mlTrainingProcess == null)
            return;

        try
        {
            if (!_mlTrainingProcess.HasExited)
                _mlTrainingProcess.Kill(entireProcessTree: true);
        }
        catch
        {
            // Best-effort shutdown.
        }

        try
        {
            _mlTrainingProcess.Dispose();
        }
        catch
        {
            // Ignore disposal failures during shutdown.
        }

        _mlTrainingProcess = null;
    }

    private void TryLoadMlTrainingHistory()
    {
        if (string.IsNullOrWhiteSpace(_mlTrainingOutputDir))
            return;

        string historyPath = Path.Combine(_mlTrainingOutputDir, "training_log.json");
        if (!File.Exists(historyPath))
            return;

        DateTime lastWrite = File.GetLastWriteTimeUtc(historyPath);
        if (lastWrite == _mlTrainingLogLastWriteUtc)
            return;

        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(historyPath));
            JsonElement root = document.RootElement;
            var snapshot = new MlTrainingHistorySnapshot();

            if (root.TryGetProperty("epochs", out JsonElement epochsElement))
            {
                foreach (JsonElement item in epochsElement.EnumerateArray())
                {
                    if (item.TryGetInt32(out int epochValue))
                        snapshot.Epochs.Add(epochValue);
                }
            }

            if (root.TryGetProperty("train_loss", out JsonElement trainElement))
            {
                foreach (JsonElement item in trainElement.EnumerateArray())
                    snapshot.TrainLoss.Add(item.GetSingle());
            }

            if (root.TryGetProperty("val_loss", out JsonElement valElement))
            {
                foreach (JsonElement item in valElement.EnumerateArray())
                    snapshot.ValLoss.Add(item.GetSingle());
            }

            if (root.TryGetProperty("components", out JsonElement componentsElement))
            {
                foreach (JsonElement epochComponent in componentsElement.EnumerateArray())
                {
                    if (epochComponent.ValueKind != JsonValueKind.Object)
                        continue;

                    foreach (JsonProperty property in epochComponent.EnumerateObject())
                    {
                        if (!snapshot.Components.TryGetValue(property.Name, out List<float>? series))
                        {
                            series = new List<float>();
                            snapshot.Components[property.Name] = series;
                        }

                        series.Add(property.Value.GetSingle());
                    }
                }
            }

            _mlTrainingHistory = snapshot;
            _mlTrainingLogLastWriteUtc = lastWrite;
            _mlTrainingHistoryError = null;
        }
        catch (Exception ex)
        {
            _mlTrainingHistoryError = ex.Message;
        }
    }

    private void StartMlTraining()
    {
        if (IsMlTrainingProcessActive())
        {
            _mlTrainingStatus = "Training is already running.";
            return;
        }

        PrepareMlTrainingDialogInputs();
        string scriptPath = _mlTrainingScriptPath.Trim();
        if (string.IsNullOrWhiteSpace(scriptPath) || !File.Exists(scriptPath))
        {
            _mlTrainingStatus = "Training script not found. Set a valid train_v7.py path.";
            return;
        }

        string outputDir = _mlTrainingOutputDir.Trim();
        if (string.IsNullOrWhiteSpace(outputDir))
        {
            _mlTrainingStatus = "Choose an output directory first.";
            return;
        }

        Directory.CreateDirectory(outputDir);
        _mlTrainingLog.Clear();
        _mlTrainingScrollToBottom = true;
        _mlTrainingHistory = null;
        _mlTrainingHistoryError = null;
        _mlTrainingLogLastWriteUtc = DateTime.MinValue;

        var startInfo = new ProcessStartInfo
        {
            FileName = _mlTrainingPythonExecutable.Trim(),
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(scriptPath) ?? Environment.CurrentDirectory,
        };

        startInfo.ArgumentList.Add(scriptPath);
        startInfo.ArgumentList.Add("--profile");
        startInfo.ArgumentList.Add(_mlTrainingProfile);
        startInfo.ArgumentList.Add("--output-dir");
        startInfo.ArgumentList.Add(outputDir);
        startInfo.ArgumentList.Add("--batch-size");
        startInfo.ArgumentList.Add(_mlTrainingBatchSize.ToString(CultureInfo.InvariantCulture));
        startInfo.ArgumentList.Add("--epochs");
        startInfo.ArgumentList.Add(_mlTrainingEpochs.ToString(CultureInfo.InvariantCulture));
        startInfo.ArgumentList.Add("--patience");
        startInfo.ArgumentList.Add(_mlTrainingPatience.ToString(CultureInfo.InvariantCulture));
        startInfo.ArgumentList.Add("--learning-rate");
        startInfo.ArgumentList.Add(_mlTrainingLearningRate.ToString("G", CultureInfo.InvariantCulture));
        startInfo.ArgumentList.Add("--val-fraction");
        startInfo.ArgumentList.Add(_mlTrainingValFraction.ToString("G", CultureInfo.InvariantCulture));
        startInfo.ArgumentList.Add("--spatial-group-size");
        startInfo.ArgumentList.Add(_mlTrainingSpatialGroupSize.ToString(CultureInfo.InvariantCulture));
        startInfo.ArgumentList.Add("--seed");
        startInfo.ArgumentList.Add(_mlTrainingSeed.ToString(CultureInfo.InvariantCulture));

        if (_mlTrainingLimit > 0)
        {
            startInfo.ArgumentList.Add("--limit");
            startInfo.ArgumentList.Add(_mlTrainingLimit.ToString(CultureInfo.InvariantCulture));
        }

        if (_mlTrainingNoAugment)
            startInfo.ArgumentList.Add("--no-augment");

        if (!string.IsNullOrWhiteSpace(_mlTrainingResumeCheckpointPath))
        {
            startInfo.ArgumentList.Add("--resume");
            startInfo.ArgumentList.Add(_mlTrainingResumeCheckpointPath.Trim());
        }

        foreach (string root in ParseMultiValueText(_mlTrainingDatasetRootsText))
        {
            startInfo.ArgumentList.Add("--dataset-root");
            startInfo.ArgumentList.Add(root);
        }

        foreach (string mapName in ParseMultiValueText(_mlTrainingIncludeMapsText))
        {
            startInfo.ArgumentList.Add("--include-map");
            startInfo.ArgumentList.Add(mapName);
        }

        foreach (string mapName in ParseMultiValueText(_mlTrainingExcludeMapsText))
        {
            startInfo.ArgumentList.Add("--exclude-map");
            startInfo.ArgumentList.Add(mapName);
        }

        try
        {
            var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
            process.OutputDataReceived += (_, args) =>
            {
                if (!string.IsNullOrWhiteSpace(args.Data))
                    AppendMlTrainingLog(args.Data!);
            };
            process.ErrorDataReceived += (_, args) =>
            {
                if (!string.IsNullOrWhiteSpace(args.Data))
                    AppendMlTrainingLog($"ERR: {args.Data}");
            };

            if (!process.Start())
            {
                _mlTrainingStatus = "Failed to start the training process.";
                return;
            }

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            _mlTrainingProcess = process;
            _mlTrainingStatus = $"Training started (PID {process.Id}).";
            _statusMessage = _mlTrainingStatus;
            AppendMlTrainingLog($"=== Started V7 training in {outputDir} ===");
            _mlTrainingNextPollUtc = DateTime.MinValue;
        }
        catch (Exception ex)
        {
            _mlTrainingStatus = $"Failed to start training: {ex.Message}";
            AppendMlTrainingLog(_mlTrainingStatus);
        }
    }

    private void StopMlTraining()
    {
        if (_mlTrainingProcess == null)
            return;

        try
        {
            if (!_mlTrainingProcess.HasExited)
                _mlTrainingProcess.Kill(entireProcessTree: true);
        }
        catch (Exception ex)
        {
            AppendMlTrainingLog($"ERR: failed to stop training process: {ex.Message}");
        }

        FinalizeMlTrainingProcessExit();
    }

    private void AppendMlTrainingLog(string line)
    {
        lock (_mlTrainingLog)
        {
            _mlTrainingLog.Add(line);
            if (_mlTrainingLog.Count > 2000)
                _mlTrainingLog.RemoveRange(0, _mlTrainingLog.Count - 1500);
        }

        _mlTrainingScrollToBottom = true;
    }

    private static List<string> ParseMultiValueText(string text)
    {
        var values = new List<string>();
        if (string.IsNullOrWhiteSpace(text))
            return values;

        string[] parts = text.Split(['\r', '\n', ';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (string part in parts)
        {
            if (!string.IsNullOrWhiteSpace(part))
                values.Add(part);
        }

        return values;
    }

    private void DrawMlTrainingDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(860, 780), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Train V7 Terrain Model", ref _showMlTrainingDialog))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("Launch the multichannel V7.1 terrain trainer from the viewer, watch stdout/stderr, and monitor live loss curves from training_log.json. V7.1 uses minimap, normals, WDL height priors, per-tile bounds hints, liquid masks, and object-footprint masks because minimap pixels alone do not carry enough height information to reconstruct valid terrain. WDL is the critical fallback seam: for many maps we have minimaps and WDL even when full ADT-derived heightmaps are missing, so the model must preserve that low-resolution terrain correlation instead of pretending minimap-only reconstruction is sufficient. Alpha-mask and tileset decomposition should be treated as a separate minimap-to-texture-layer model, not as the core terrain-height job.");
        ImGui.Spacing();

        ImGui.Text("Python Executable:");
        ImGui.SetNextItemWidth(-1);
        ImGui.InputText("##mlTrainingPython", ref _mlTrainingPythonExecutable, 512);

        ImGui.Text("train_v7.py Path:");
        ImGui.SetNextItemWidth(-80);
        ImGui.InputText("##mlTrainingScript", ref _mlTrainingScriptPath, 1024);
        ImGui.SameLine();
        if (ImGui.Button("Browse##mlTrainingScript"))
        {
            string? result = ShowFileDialogSTA("Select train_v7.py", "Python Files (*.py)|*.py|All Files (*.*)|*.*", Path.GetDirectoryName(_mlTrainingScriptPath));
            if (!string.IsNullOrWhiteSpace(result))
                _mlTrainingScriptPath = result;
        }

        string[] profiles = ["development-map", "manual"];
        if (ImGui.BeginCombo("Profile", _mlTrainingProfile))
        {
            foreach (string profile in profiles)
            {
                bool isSelected = string.Equals(_mlTrainingProfile, profile, StringComparison.OrdinalIgnoreCase);
                if (ImGui.Selectable(profile, isSelected))
                {
                    _mlTrainingProfile = profile;
                    if (string.Equals(profile, "development-map", StringComparison.OrdinalIgnoreCase) && string.IsNullOrWhiteSpace(_mlTrainingIncludeMapsText))
                        _mlTrainingIncludeMapsText = "Northrend\nLostIsles";
                }

                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        ImGui.SameLine();
        if (ImGui.Button("Use Development Defaults"))
        {
            _mlTrainingProfile = "development-map";
            _mlTrainingIncludeMapsText = "Northrend\nLostIsles";
            _mlTrainingExcludeMapsText = string.Empty;
            _mlTrainingOutputDir = Path.Combine(OutputDir, "ml-training", "v7-development-map");
        }

        ImGui.TextWrapped("`development-map` auto-discovers likely 3.0.1 Northrend and 4.0.0.11927 LostIsles roots if your local folder names encode the era. Use explicit dataset roots when you want deterministic control.");

        ImGui.Text("Dataset Roots (one per line, optional when using auto-discovery):");
        ImGui.InputTextMultiline("##mlTrainingRoots", ref _mlTrainingDatasetRootsText, 4096, new Vector2(-1, 90));
        if (ImGui.Button("Use Current ML Dataset Root") && !string.IsNullOrWhiteSpace(_mkHarvestDatasetRoot))
        {
            _mlTrainingDatasetRootsText = string.IsNullOrWhiteSpace(_mlTrainingDatasetRootsText)
                ? _mkHarvestDatasetRoot
                : _mlTrainingDatasetRootsText.TrimEnd() + Environment.NewLine + _mkHarvestDatasetRoot;
        }

        ImGui.Text("Include Maps (one per line):");
        ImGui.InputTextMultiline("##mlTrainingIncludeMaps", ref _mlTrainingIncludeMapsText, 1024, new Vector2(-1, 54));
        ImGui.Text("Exclude Maps (one per line):");
        ImGui.InputTextMultiline("##mlTrainingExcludeMaps", ref _mlTrainingExcludeMapsText, 1024, new Vector2(-1, 54));

        ImGui.Text("Output Directory:");
        ImGui.SetNextItemWidth(-80);
        ImGui.InputText("##mlTrainingOutput", ref _mlTrainingOutputDir, 1024);
        ImGui.SameLine();
        if (ImGui.Button("Browse##mlTrainingOutput"))
        {
            string? result = ShowFolderDialogSTA("Select V7 training output directory", _mlTrainingOutputDir, showNewFolderButton: true);
            if (!string.IsNullOrWhiteSpace(result))
                _mlTrainingOutputDir = result;
        }

        ImGui.Text("Resume Checkpoint (optional):");
        ImGui.SetNextItemWidth(-80);
        ImGui.InputText("##mlTrainingResume", ref _mlTrainingResumeCheckpointPath, 1024);
        ImGui.SameLine();
        if (ImGui.Button("Browse##mlTrainingResume"))
        {
            string? result = ShowFileDialogSTA("Select V7 checkpoint", "Checkpoint (*.pt)|*.pt|All Files (*.*)|*.*", Path.GetDirectoryName(_mlTrainingResumeCheckpointPath));
            if (!string.IsNullOrWhiteSpace(result))
                _mlTrainingResumeCheckpointPath = result;
        }

        ImGui.Separator();
        ImGui.Text("Training Parameters:");
        ImGui.SetNextItemWidth(120);
        ImGui.InputInt("Batch Size", ref _mlTrainingBatchSize);
        _mlTrainingBatchSize = Math.Max(1, _mlTrainingBatchSize);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(120);
        ImGui.InputInt("Epochs", ref _mlTrainingEpochs);
        _mlTrainingEpochs = Math.Max(1, _mlTrainingEpochs);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(120);
        ImGui.InputInt("Patience", ref _mlTrainingPatience);
        _mlTrainingPatience = Math.Max(1, _mlTrainingPatience);

        ImGui.SetNextItemWidth(120);
        ImGui.InputFloat("Learning Rate", ref _mlTrainingLearningRate, 0.0f, 0.0f, "%.6f");
        _mlTrainingLearningRate = MathF.Max(1e-7f, _mlTrainingLearningRate);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(120);
        ImGui.InputFloat("Val Fraction", ref _mlTrainingValFraction, 0.0f, 0.0f, "%.3f");
        _mlTrainingValFraction = Math.Clamp(_mlTrainingValFraction, 0.01f, 0.5f);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(120);
        ImGui.InputInt("Spatial Group", ref _mlTrainingSpatialGroupSize);
        _mlTrainingSpatialGroupSize = Math.Max(1, _mlTrainingSpatialGroupSize);

        ImGui.SetNextItemWidth(120);
        ImGui.InputInt("Seed", ref _mlTrainingSeed);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(120);
        ImGui.InputInt("Limit", ref _mlTrainingLimit);
        _mlTrainingLimit = Math.Max(0, _mlTrainingLimit);
        ImGui.SameLine();
        ImGui.Checkbox("Disable augmentation", ref _mlTrainingNoAugment);

        ImGui.Separator();
        bool isRunning = IsMlTrainingProcessActive();
        if (!isRunning)
        {
            if (ImGui.Button("Start Training", new Vector2(150, 32)))
                StartMlTraining();
        }
        else
        {
            if (ImGui.Button("Stop Training", new Vector2(150, 32)))
                StopMlTraining();
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(1f, 1f, 0f, 1f), $"Running (PID {_mlTrainingProcess?.Id})");
        }

        ImGui.SameLine();
        if (ImGui.Button("Write Notebook", new Vector2(140, 32)))
            WriteMlTrainingNotebook();
        ImGui.SameLine();
        if (ImGui.Button("Open Output", new Vector2(120, 32)) && Directory.Exists(_mlTrainingOutputDir))
            OpenPathInShell(_mlTrainingOutputDir);
        ImGui.SameLine();
        if (ImGui.Button("Open Notebook", new Vector2(120, 32)) && !string.IsNullOrWhiteSpace(_mlTrainingLastNotebookPath) && File.Exists(_mlTrainingLastNotebookPath))
            OpenPathInShell(_mlTrainingLastNotebookPath);

        if (!string.IsNullOrWhiteSpace(_mlTrainingStatus))
            ImGui.TextWrapped(_mlTrainingStatus);
        if (!string.IsNullOrWhiteSpace(_mlTrainingHistoryError))
            ImGui.TextColored(new Vector4(1f, 0.45f, 0.45f, 1f), _mlTrainingHistoryError);

        if (_mlTrainingHistory != null)
        {
            ImGui.Text($"Latest Epoch: {_mlTrainingHistory.LatestEpoch}");
            ImGui.SameLine();
            ImGui.Text($"Train Loss: {_mlTrainingHistory.LatestTrainLoss:F4}");
            ImGui.SameLine();
            ImGui.Text($"Val Loss: {_mlTrainingHistory.LatestValLoss:F4}");
            ImGui.SameLine();
            ImGui.Text($"Best Val: {_mlTrainingHistory.BestValLoss:F4}");

            DrawMlTrainingPlot(
                "Loss Plot",
                new Vector2(-1, 180),
                new MlTrainingSeries { Values = _mlTrainingHistory.TrainLoss, Color = 0xFF66B2FF },
                new MlTrainingSeries { Values = _mlTrainingHistory.ValLoss, Color = 0xFFFFAA55 });
            ImGui.TextColored(new Vector4(0.40f, 0.70f, 1.0f, 1f), "Train loss");
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(1.0f, 0.67f, 0.33f, 1f), "Validation loss");

            List<MlTrainingSeries> componentSeries = [];
            if (_mlTrainingHistory.Components.TryGetValue("heightmap_global", out List<float>? globalSeries) && globalSeries.Count > 0)
                componentSeries.Add(new MlTrainingSeries { Values = globalSeries, Color = 0xFF6ED39A });
            if (_mlTrainingHistory.Components.TryGetValue("heightmap_local", out List<float>? localSeries) && localSeries.Count > 0)
                componentSeries.Add(new MlTrainingSeries { Values = localSeries, Color = 0xFF4FB3FF });
            if (_mlTrainingHistory.Components.TryGetValue("bounds", out List<float>? boundsSeries) && boundsSeries.Count > 0)
                componentSeries.Add(new MlTrainingSeries { Values = boundsSeries, Color = 0xFFB86EFF });
            if (_mlTrainingHistory.Components.TryGetValue("gradient", out List<float>? gradientSeries) && gradientSeries.Count > 0)
                componentSeries.Add(new MlTrainingSeries { Values = gradientSeries, Color = 0xFFFF8C42 });
            if (_mlTrainingHistory.Components.TryGetValue("ssim", out List<float>? ssimSeries) && ssimSeries.Count > 0)
                componentSeries.Add(new MlTrainingSeries { Values = ssimSeries, Color = 0xFF8FD694 });
            if (_mlTrainingHistory.Components.TryGetValue("edge", out List<float>? edgeSeries) && edgeSeries.Count > 0)
                componentSeries.Add(new MlTrainingSeries { Values = edgeSeries, Color = 0xFFFF6666 });

            if (componentSeries.Count > 0)
            {
                DrawMlTrainingPlot("Loss Components", new Vector2(-1, 140), componentSeries.ToArray());
                ImGui.TextColored(new Vector4(0.43f, 0.83f, 0.60f, 1f), "HM_G");
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.31f, 0.70f, 1.0f, 1f), "HM_L");
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.72f, 0.43f, 1.0f, 1f), "Bounds");
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(1.0f, 0.55f, 0.26f, 1f), "Gradient");
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(0.56f, 0.84f, 0.58f, 1f), "SSIM");
                ImGui.SameLine();
                ImGui.TextColored(new Vector4(1.0f, 0.40f, 0.40f, 1f), "Edge");
            }
        }

        ImGui.Separator();
        ImGui.TextWrapped("Notebook export writes a small `.ipynb` next to the training output that loads `training_log.json` and plots the train/validation curves with matplotlib. Use it in VS Code or Jupyter when you want a deeper look than the live viewer plot.");
        ImGui.Text("Training Log:");
        float logHeight = MathF.Max(140f, ImGui.GetContentRegionAvail().Y - 4f);
        if (ImGui.BeginChild("MlTrainingLog", new Vector2(-1, logHeight), true))
        {
            lock (_mlTrainingLog)
            {
                foreach (string line in _mlTrainingLog)
                    ImGui.TextWrapped(line);
            }

            if (_mlTrainingScrollToBottom)
            {
                ImGui.SetScrollHereY(1f);
                _mlTrainingScrollToBottom = false;
            }
        }
        ImGui.EndChild();
        ImGui.End();
    }

    private void DrawMlTrainingPlot(string label, Vector2 requestedSize, params MlTrainingSeries[] series)
    {
        float width = requestedSize.X <= 0f ? ImGui.GetContentRegionAvail().X : requestedSize.X;
        float height = requestedSize.Y <= 0f ? 160f : requestedSize.Y;
        Vector2 size = new(MathF.Max(width, 120f), MathF.Max(height, 80f));

        Vector2 topLeft = ImGui.GetCursorScreenPos();
        ImGui.InvisibleButton(label, size);
        Vector2 bottomRight = topLeft + size;
        var drawList = ImGui.GetWindowDrawList();
        drawList.AddRectFilled(topLeft, bottomRight, 0x22111111, 4f);
        drawList.AddRect(topLeft, bottomRight, 0xFF555555, 4f);

        float minValue = float.MaxValue;
        float maxValue = float.MinValue;
        int longestSeries = 0;
        foreach (MlTrainingSeries line in series)
        {
            if (line.Values.Count > longestSeries)
                longestSeries = line.Values.Count;

            foreach (float value in line.Values)
            {
                if (value < minValue)
                    minValue = value;
                if (value > maxValue)
                    maxValue = value;
            }
        }

        if (longestSeries < 2 || minValue == float.MaxValue || maxValue == float.MinValue)
        {
            drawList.AddText(topLeft + new Vector2(8, 8), 0xFFBBBBBB, "Waiting for at least two logged epochs...");
            return;
        }

        float range = maxValue - minValue;
        if (range < 1e-6f)
            range = 1f;
        float padding = range * 0.05f;
        minValue -= padding;
        maxValue += padding;
        range = maxValue - minValue;

        for (int gridIndex = 1; gridIndex < 4; gridIndex++)
        {
            float y = topLeft.Y + size.Y * (gridIndex / 4f);
            drawList.AddLine(new Vector2(topLeft.X, y), new Vector2(bottomRight.X, y), 0x22444444);
        }

        drawList.AddText(topLeft + new Vector2(8f, 6f), 0xFFBBBBBB, maxValue.ToString("F4", CultureInfo.InvariantCulture));
        drawList.AddText(new Vector2(topLeft.X + 8f, bottomRight.Y - 20f), 0xFFBBBBBB, minValue.ToString("F4", CultureInfo.InvariantCulture));

        foreach (MlTrainingSeries line in series)
        {
            if (line.Values.Count < 2)
                continue;

            for (int index = 1; index < line.Values.Count; index++)
            {
                Vector2 pointA = PlotPoint(index - 1, line.Values[index - 1], line.Values.Count, minValue, range, topLeft, size);
                Vector2 pointB = PlotPoint(index, line.Values[index], line.Values.Count, minValue, range, topLeft, size);
                drawList.AddLine(pointA, pointB, line.Color, 2f);
            }
        }
    }

    private static Vector2 PlotPoint(int index, float value, int count, float minValue, float range, Vector2 topLeft, Vector2 size)
    {
        float x = topLeft.X + (count <= 1 ? 0f : (index / (float)(count - 1)) * size.X);
        float normalizedY = (value - minValue) / range;
        float y = topLeft.Y + size.Y - normalizedY * size.Y;
        return new Vector2(x, y);
    }

    private void WriteMlTrainingNotebook()
    {
        if (string.IsNullOrWhiteSpace(_mlTrainingOutputDir))
        {
            _mlTrainingStatus = "Choose an output directory first.";
            return;
        }

        Directory.CreateDirectory(_mlTrainingOutputDir);
        string trainingLogPath = Path.Combine(_mlTrainingOutputDir, "training_log.json");
        string notebookPath = Path.Combine(_mlTrainingOutputDir, "v7_training_monitor.ipynb");

        var notebook = new
        {
            cells = new object[]
            {
                new
                {
                    cell_type = "markdown",
                    metadata = new { language = "markdown" },
                    source = new[]
                    {
                        "# V7 Training Monitor\n",
                        "Load `training_log.json` from a V7 training run and plot train/validation losses plus the tracked component curves.\n"
                    }
                },
                new
                {
                    cell_type = "code",
                    metadata = new { language = "python" },
                    source = new[]
                    {
                        "from pathlib import Path\n",
                        "import json\n",
                        "import matplotlib.pyplot as plt\n",
                        $"training_log_path = Path(r\"{trainingLogPath}\")\n",
                        "if not training_log_path.exists():\n",
                        "    raise FileNotFoundError(training_log_path)\n",
                        "data = json.loads(training_log_path.read_text())\n",
                        "epochs = data.get('epochs', [])\n",
                        "train_loss = data.get('train_loss', [])\n",
                        "val_loss = data.get('val_loss', [])\n",
                        "components = data.get('components', [])\n",
                        "metadata = data.get('metadata', {})\n",
                        "print(metadata)\n"
                    }
                },
                new
                {
                    cell_type = "code",
                    metadata = new { language = "python" },
                    source = new[]
                    {
                        "plt.figure(figsize=(10, 5))\n",
                        "plt.plot(epochs, train_loss, label='train')\n",
                        "plt.plot(epochs, val_loss, label='val')\n",
                        "plt.xlabel('Epoch')\n",
                        "plt.ylabel('Loss')\n",
                        "plt.title('V7 Train / Validation Loss')\n",
                        "plt.grid(True, alpha=0.25)\n",
                        "plt.legend()\n",
                        "plt.show()\n"
                    }
                },
                new
                {
                    cell_type = "code",
                    metadata = new { language = "python" },
                    source = new[]
                    {
                        "if components:\n",
                        "    keys = sorted({key for entry in components for key in entry.keys()})\n",
                        "    plt.figure(figsize=(10, 5))\n",
                        "    for key in keys:\n",
                        "        plt.plot(epochs[:len(components)], [entry.get(key) for entry in components], label=key)\n",
                        "    plt.xlabel('Epoch')\n",
                        "    plt.ylabel('Component Loss')\n",
                        "    plt.title('V7 Loss Components')\n",
                        "    plt.grid(True, alpha=0.25)\n",
                        "    plt.legend()\n",
                        "    plt.show()\n",
                        "else:\n",
                        "    print('No component series recorded.')\n"
                    }
                }
            },
            metadata = new
            {
                kernelspec = new { display_name = "Python 3", language = "python", name = "python3" },
                language_info = new { name = "python" }
            },
            nbformat = 4,
            nbformat_minor = 5
        };

        File.WriteAllText(notebookPath, JsonSerializer.Serialize(notebook, new JsonSerializerOptions { WriteIndented = true }));
        _mlTrainingLastNotebookPath = notebookPath;
        _mlTrainingStatus = $"Wrote notebook: {notebookPath}";
        AppendMlTrainingLog(_mlTrainingStatus);
    }

    private static void OpenPathInShell(string path)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = path,
                UseShellExecute = true,
            });
        }
        catch
        {
            // Best-effort shell open.
        }
    }
}
