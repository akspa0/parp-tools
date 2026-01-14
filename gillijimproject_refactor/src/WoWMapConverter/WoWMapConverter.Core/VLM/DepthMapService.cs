using System.Diagnostics;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Service for generating depth maps using DepthAnything3 Python model.
/// Wraps the Python script execution with proper error handling.
/// </summary>
public class DepthMapService
{
    private readonly string _pythonPath;
    private readonly string _scriptPath;
    private readonly string _condaEnv;

    /// <summary>
    /// Create DepthMapService with custom paths.
    /// </summary>
    /// <param name="pythonPath">Path to Python executable (or 'python' for system default)</param>
    /// <param name="scriptPath">Path to generate_depth.py script</param>
    /// <param name="condaEnv">Conda environment name (optional, uses 'da3' by default)</param>
    public DepthMapService(string? pythonPath = null, string? scriptPath = null, string? condaEnv = null)
    {
        _pythonPath = pythonPath ?? "python";
        _condaEnv = condaEnv ?? "da3";
        
        // Default script path relative to this assembly
        if (scriptPath == null)
        {
            var assemblyDir = Path.GetDirectoryName(typeof(DepthMapService).Assembly.Location);
            _scriptPath = Path.Combine(assemblyDir ?? ".", "VLM", "DepthAnything3", "generate_depth.py");
        }
        else
        {
            _scriptPath = scriptPath;
        }
    }

    /// <summary>
    /// Generate depth maps for all images in a directory.
    /// </summary>
    /// <param name="inputDir">Directory containing minimap PNG images</param>
    /// <param name="outputDir">Directory to save depth maps</param>
    /// <param name="progress">Progress reporter</param>
    /// <returns>Number of depth maps generated</returns>
    public async Task<int> GenerateDepthMapsAsync(string inputDir, string outputDir, IProgress<string>? progress = null)
    {
        if (!File.Exists(_scriptPath))
        {
            progress?.Report($"Warning: Depth script not found: {_scriptPath}");
            progress?.Report("Skipping depth map generation. Run setup_da3.ps1 first.");
            return 0;
        }

        Directory.CreateDirectory(outputDir);

        progress?.Report("Starting DepthAnything3 depth map generation...");

        // Locate python executable in venv
        var scriptDir = Path.GetDirectoryName(_scriptPath);
        var venvPython = Path.Combine(scriptDir ?? ".", ".venv", "Scripts", "python.exe"); // Windows default
        
        if (!OperatingSystem.IsWindows())
        {
            venvPython = Path.Combine(scriptDir ?? ".", ".venv", "bin", "python"); // Linux/Mac
        }

        string pythonExe = _pythonPath;
        if (File.Exists(venvPython))
        {
            pythonExe = venvPython;
            progress?.Report($"Using venv python: {pythonExe}");
        }
        else
        {
            progress?.Report($"Using system python: {pythonExe} (venv not found at {venvPython})");
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = $"\"{_scriptPath}\" --input \"{inputDir}\" --output \"{outputDir}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        try
        {
            using var process = Process.Start(startInfo);
            if (process == null)
            {
                progress?.Report("Failed to start Python process");
                return 0;
            }

            // Read output asynchronously
            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();

            await process.WaitForExitAsync();

            var output = await outputTask;
            var error = await errorTask;

            if (!string.IsNullOrWhiteSpace(output))
            {
                foreach (var line in output.Split('\n').Where(l => !string.IsNullOrWhiteSpace(l)))
                    progress?.Report(line.Trim());
            }

            if (process.ExitCode != 0)
            {
                progress?.Report($"Depth generation failed: {error}");
                return 0;
            }

            // Count generated files
            var depthFiles = Directory.GetFiles(outputDir, "*_depth.png");
            progress?.Report($"Generated {depthFiles.Length} depth maps");
            return depthFiles.Length;
        }
        catch (Exception ex)
        {
            progress?.Report($"Error running depth generation: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Check if DepthAnything3 is available.
    /// </summary>
    public bool IsAvailable()
    {
        if (!File.Exists(_scriptPath))
            return false;

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = "-c \"import depth_anything_3\"",
                UseShellExecute = false,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            process?.WaitForExit(5000);
            return process?.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }
}
