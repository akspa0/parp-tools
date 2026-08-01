using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using ImGuiNET;
using WowViewer.Core.Maps;

namespace WoWViewer;

public partial class ViewerApp
{
    private void PrepareSynthesizedMinimapExportDialogInputs()
    {
        string? activeClientRoot = GetActiveGamePath();
        if (!string.IsNullOrWhiteSpace(activeClientRoot))
            _synthesizedMinimapClientRoot = activeClientRoot;

        string? activeMapName = GetCurrentSessionMapName();
        if (!string.IsNullOrWhiteSpace(activeMapName))
            _synthesizedMinimapMapName = activeMapName;

        if (string.IsNullOrWhiteSpace(_synthesizedMinimapOutputDirectory)
            && !string.IsNullOrWhiteSpace(_synthesizedMinimapMapName))
        {
            string mapSegment = SanitizeProjectPathSegment(_synthesizedMinimapMapName);
            _synthesizedMinimapOutputDirectory = Path.Combine(
                GetProjectOutputRootDirectory(),
                "synthesized-minimaps",
                mapSegment,
                $"tod-{TimeOfDayClock.FromHours(_synthesizedMinimapTimeHours).CompactText}");
        }
    }

    private void DrawSynthesizedMinimapExportDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(680, 560), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Synthesized Terrain Minimap Export", ref _showSynthesizedMinimapExportDialog))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped(
            "Build paired terrain-only and _liquid PNG minimaps directly from client BLP textures plus MCLY/MCAL, MCNR, MCSH, and decoded liquid coverage. " +
            "This does not read a shipped minimap image or use the retired VLM/MK dataset workflow.");
        ImGui.TextWrapped(
            "Global clear-weather LIT colors are used when evaluable at the selected time. Otherwise the output records a visible authored fallback; " +
            "neither mode claims unproven local-zone lighting as client-exact.");
        ImGui.Separator();

        ImGui.Text("Client root:");
        ImGui.SetNextItemWidth(-88);
        ImGui.InputText("##synthmin_client", ref _synthesizedMinimapClientRoot, 1024);
        ImGui.SameLine();
        if (ImGui.Button("Browse##synthmin_client"))
        {
            string? selected = ShowFolderDialogSTA("Select WoW client root", _synthesizedMinimapClientRoot);
            if (!string.IsNullOrWhiteSpace(selected))
                _synthesizedMinimapClientRoot = selected;
        }

        ImGui.Text("Map name:");
        ImGui.SetNextItemWidth(-1);
        ImGui.InputText("##synthmin_map", ref _synthesizedMinimapMapName, 256);

        TimeOfDayClock selectedTime = TimeOfDayClock.FromHours(_synthesizedMinimapTimeHours);
        int selectedHour = selectedTime.Hour;
        int selectedMinute = selectedTime.Minute;
        ImGui.Text("Time of day:");
        ImGui.SetNextItemWidth(92);
        bool timeChanged = ImGui.InputInt("Hour##synthmin_time", ref selectedHour, 1, 1);
        ImGui.SameLine();
        ImGui.SetNextItemWidth(92);
        timeChanged |= ImGui.InputInt("Minute##synthmin_time", ref selectedMinute, 1, 5);
        if (timeChanged)
        {
            selectedHour = Math.Clamp(selectedHour, 0, 23);
            selectedMinute = Math.Clamp(selectedMinute, 0, 59);
            _synthesizedMinimapTimeHours = new TimeOfDayClock(selectedHour, selectedMinute).Hours;
        }
        ImGui.SameLine();
        ImGui.TextDisabled("12:15 is precise; midnight 00:00 · noon 12:00");

        ImGui.Text("Tile resolution:");
        ImGui.SetNextItemWidth(160);
        ImGui.InputInt("##synthmin_resolution", ref _synthesizedMinimapResolution);
        _synthesizedMinimapResolution = Math.Clamp(_synthesizedMinimapResolution, 1, 4096);

        ImGui.Checkbox("Write per-tile PNGs", ref _synthesizedMinimapEmitTiles);
        ImGui.SameLine();
        ImGui.Checkbox("Write one stitched map PNG", ref _synthesizedMinimapEmitWholeMap);

        ImGui.Checkbox("Include WMO geometry", ref _synthesizedMinimapIncludeWmos);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Render placed WMO buildings on top of the terrain minimap with matching solar lighting.");
        ImGui.SameLine();
        ImGui.Checkbox("Bake MCSH shadows", ref _synthesizedMinimapBakeMcsh);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Include the terrain-side static shadow map (MCSH) in the output. Without this, only Lambert hillshading is used (no cast shadows).");

        ImGui.Text("Output directory:");
        ImGui.SetNextItemWidth(-88);
        ImGui.InputText("##synthmin_output", ref _synthesizedMinimapOutputDirectory, 1024);
        ImGui.SameLine();
        if (ImGui.Button("Browse##synthmin_output"))
        {
            string? selected = ShowFolderDialogSTA(
                "Select synthesized minimap output directory",
                _synthesizedMinimapOutputDirectory,
                showNewFolderButton: true);
            if (!string.IsNullOrWhiteSpace(selected))
                _synthesizedMinimapOutputDirectory = selected;
        }

        bool canStart = !_synthesizedMinimapRunning
            && !string.IsNullOrWhiteSpace(_synthesizedMinimapClientRoot)
            && !string.IsNullOrWhiteSpace(_synthesizedMinimapMapName)
            && !string.IsNullOrWhiteSpace(_synthesizedMinimapOutputDirectory)
            && (_synthesizedMinimapEmitTiles || _synthesizedMinimapEmitWholeMap);
        if (!canStart)
            ImGui.BeginDisabled();
        if (ImGui.Button(_synthesizedMinimapRunning ? "Exporting..." : "Start Export", new Vector2(140, 0)))
            StartSynthesizedMinimapExport();
        if (!canStart)
            ImGui.EndDisabled();
        ImGui.SameLine();
        if (ImGui.Button("Close", new Vector2(80, 0)) && !_synthesizedMinimapRunning)
            _showSynthesizedMinimapExportDialog = false;

        if (_synthesizedMinimapError is not null)
        {
            ImGui.Spacing();
            ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(1f, 0.32f, 0.32f, 1f));
            ImGui.TextWrapped($"Error: {_synthesizedMinimapError}");
            ImGui.PopStyleColor();
        }
        else if (_synthesizedMinimapDone)
        {
            ImGui.SameLine();
            ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(0.35f, 0.9f, 0.45f, 1f));
            ImGui.TextUnformatted("Completed — see synthesis-manifest.json in the selected output directory.");
            ImGui.PopStyleColor();
        }

        ImGui.Spacing();
        ImGui.TextUnformatted("Harvest command log");
        if (ImGui.BeginChild("##synthmin_log", new Vector2(-1, 170), true))
        {
            lock (_synthesizedMinimapLog)
            {
                foreach (string line in _synthesizedMinimapLog)
                    ImGui.TextUnformatted(line);
            }

            if (_synthesizedMinimapScrollToBottom)
            {
                ImGui.SetScrollHereY(1f);
                _synthesizedMinimapScrollToBottom = false;
            }
        }
        ImGui.EndChild();
        ImGui.End();
    }

    private void StartSynthesizedMinimapExport()
    {
        _synthesizedMinimapLog.Clear();
        _synthesizedMinimapError = null;
        _synthesizedMinimapDone = false;
        _synthesizedMinimapRunning = true;

        string clientRoot = _synthesizedMinimapClientRoot.Trim();
        string mapName = _synthesizedMinimapMapName.Trim();
        string outputDirectory = _synthesizedMinimapOutputDirectory.Trim();
        float timeHours = _synthesizedMinimapTimeHours;
        int resolution = _synthesizedMinimapResolution;
        bool emitTiles = _synthesizedMinimapEmitTiles;
        bool emitWholeMap = _synthesizedMinimapEmitWholeMap;
        bool includeWmos = _synthesizedMinimapIncludeWmos;
        bool bakeMcsh = _synthesizedMinimapBakeMcsh;

        _ = Task.Run(async () =>
        {
            try
            {
                await RunSynthesizedMinimapHarvestAsync(
                    clientRoot,
                    mapName,
                    outputDirectory,
                    timeHours,
                    resolution,
                    emitTiles,
                    emitWholeMap,
                    includeWmos,
                    bakeMcsh);
            }
            catch (Exception ex)
            {
                _synthesizedMinimapError = ex.Message;
                AppendSynthesizedMinimapLog($"[ERR] {ex.Message}");
            }
            finally
            {
                _synthesizedMinimapRunning = false;
                _synthesizedMinimapDone = _synthesizedMinimapError is null;
                _synthesizedMinimapScrollToBottom = true;
            }
        });
    }

    private async Task RunSynthesizedMinimapHarvestAsync(
        string clientRoot,
        string mapName,
        string outputDirectory,
        float timeHours,
        int resolution,
        bool emitTiles,
        bool emitWholeMap,
        bool includeWmos,
        bool bakeMcsh)
    {
        HarvestLaunchSpec? launch = ResolveHarvestLaunchSpec();
        if (launch is null)
        {
            _synthesizedMinimapError =
                "Could not find a bundled Harvest tool or a wow-viewer source checkout. " +
                "Build WowViewer.Tool.Harvest or run the documented command from the repository.";
            return;
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = launch.FileName,
            WorkingDirectory = launch.WorkingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        foreach (string argument in launch.PrefixArguments)
            startInfo.ArgumentList.Add(argument);
        startInfo.ArgumentList.Add("synthetic-minimap");
        startInfo.ArgumentList.Add("--client-root");
        startInfo.ArgumentList.Add(clientRoot);
        startInfo.ArgumentList.Add("--map");
        startInfo.ArgumentList.Add(mapName);
        startInfo.ArgumentList.Add("--output-dir");
        startInfo.ArgumentList.Add(outputDirectory);
        startInfo.ArgumentList.Add("--time-hours");
        startInfo.ArgumentList.Add(TimeOfDayClock.FromHours(timeHours).CompactText);
        startInfo.ArgumentList.Add("--resolution");
        startInfo.ArgumentList.Add(resolution.ToString(CultureInfo.InvariantCulture));
        if (emitTiles)
            startInfo.ArgumentList.Add("--per-tile");
        if (emitWholeMap)
            startInfo.ArgumentList.Add("--whole-map");
        if (includeWmos)
            startInfo.ArgumentList.Add("--include-wmos");
        if (bakeMcsh)
            startInfo.ArgumentList.Add("--bake-mcsh");

        AppendSynthesizedMinimapLog(
            $"> {launch.DisplayName} synthetic-minimap --map {mapName} --time-hours {TimeOfDayClock.FromHours(timeHours)}");
        using Process? process = Process.Start(startInfo);
        if (process is null)
            throw new InvalidOperationException("Unable to start the in-repository Harvest command.");

        Task stdout = PumpHarvestStreamAsync(process.StandardOutput, string.Empty);
        Task stderr = PumpHarvestStreamAsync(process.StandardError, "[ERR] ");
        await Task.WhenAll(process.WaitForExitAsync(), stdout, stderr);
        if (process.ExitCode != 0)
        {
            _synthesizedMinimapError = $"Harvest exited with code {process.ExitCode}. See the command log for details.";
            return;
        }

        AppendSynthesizedMinimapLog("Synthetic minimap export completed successfully.");
    }

    private async Task PumpHarvestStreamAsync(StreamReader reader, string prefix)
    {
        string? line;
        while ((line = await reader.ReadLineAsync()) is not null)
            AppendSynthesizedMinimapLog(prefix + line);
    }

    private void AppendSynthesizedMinimapLog(string line)
    {
        lock (_synthesizedMinimapLog)
        {
            _synthesizedMinimapLog.Add(line);
            if (_synthesizedMinimapLog.Count > 2_000)
                _synthesizedMinimapLog.RemoveRange(0, _synthesizedMinimapLog.Count - 2_000);
        }
        _synthesizedMinimapScrollToBottom = true;
    }

    private static HarvestLaunchSpec? ResolveHarvestLaunchSpec()
    {
        string baseDirectory = AppContext.BaseDirectory;
        foreach (string executable in new[]
                 {
                     Path.Combine(baseDirectory, "WowViewer.Tool.Harvest.exe"),
                     Path.Combine(baseDirectory, "WowViewer.Tool.Harvest")
                 })
        {
            if (File.Exists(executable))
                return new HarvestLaunchSpec(executable, Path.GetDirectoryName(executable)!, [], executable);
        }

        string? repositoryRoot = FindWowViewerRepositoryRoot();
        if (repositoryRoot is null)
            return null;

        string projectDirectory = Path.Combine(repositoryRoot, "tools", "harvest", "WowViewer.Tool.Harvest");
        foreach (string configuration in new[] { "Debug", "Release" })
        {
            string outputDirectory = Path.Combine(projectDirectory, "bin", configuration, "net10.0");
            string executable = Path.Combine(outputDirectory, "WowViewer.Tool.Harvest.exe");
            if (File.Exists(executable))
                return new HarvestLaunchSpec(executable, outputDirectory, [], executable);

            string assembly = Path.Combine(outputDirectory, "WowViewer.Tool.Harvest.dll");
            if (File.Exists(assembly))
                return new HarvestLaunchSpec("dotnet", outputDirectory, [assembly], $"dotnet {Path.GetFileName(assembly)}");
        }

        string projectPath = Path.Combine(projectDirectory, "WowViewer.Tool.Harvest.csproj");
        return File.Exists(projectPath)
            ? new HarvestLaunchSpec("dotnet", repositoryRoot, ["run", "--project", projectPath, "--"], "dotnet run --project WowViewer.Tool.Harvest")
            : null;
    }

    private static string? FindWowViewerRepositoryRoot()
    {
        foreach (string start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            for (DirectoryInfo? directory = new DirectoryInfo(start); directory is not null; directory = directory.Parent)
            {
                if (File.Exists(Path.Combine(directory.FullName, "WowViewer.slnx")))
                    return directory.FullName;
            }
        }

        return null;
    }

    private sealed record HarvestLaunchSpec(
        string FileName,
        string WorkingDirectory,
        IReadOnlyList<string> PrefixArguments,
        string DisplayName);
}
