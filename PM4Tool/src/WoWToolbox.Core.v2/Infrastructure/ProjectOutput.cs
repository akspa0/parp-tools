using System;
using System.IO;

namespace WoWToolbox.Core.v2.Infrastructure
{
    /// <summary>
    /// Central helper to provide a single, timestamped output directory for any ad-hoc exports produced while
    /// running the toolchain (tests, CLI utilities, batch processors).  The folder lives at
    /// <c>project_output/{yyyy-MM-dd_HH-mm-ss}</c> relative to the solution root so that outputs from different runs
    /// never overwrite each other but are still easy to locate.
    /// </summary>
    public static class ProjectOutput
    {
        static ProjectOutput()
        {
            // Ensure directory exists immediately
            var dir = RunDirectory;
            File.WriteAllText(Path.Combine(dir, "run_info.txt"), $"Run started {DateTime.Now:u}\nMachine: {Environment.MachineName}\n");
        }

        private static readonly Lazy<string> _runDir = new(() => CreateRunDir());

        /// <summary>
        /// Gets the root folder for the current run (created on first access).
        /// </summary>
        public static string RunDirectory => _runDir.Value;

        private static string CreateRunDir()
        {
            // Walk up from the executing assembly until we reach the repository root.  Heuristic:
            //  • folder name equals "PM4Tool" (top-level repo) OR
            //  • contains a .git folder OR
            //  • contains the main solution file src/WoWToolbox.sln
            var dir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            for (int i = 0; i < 15 && !string.IsNullOrEmpty(dir); i++)
            {
                if (Path.GetFileName(dir).Equals("PM4Tool", StringComparison.OrdinalIgnoreCase) ||
                    Directory.Exists(Path.Combine(dir, ".git")) ||
                    File.Exists(Path.Combine(dir, "src", "WoWToolbox.sln")))
                {
                    break; // found repo root
                }
                var parent = Path.GetDirectoryName(dir);
                if (string.IsNullOrEmpty(parent))
                    break;
                dir = parent;
            }
            string outputRoot = Path.Combine(dir, "project_output");
            Directory.CreateDirectory(outputRoot);

            var localNow = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, TimeZoneInfo.Local);
    string ts = localNow.ToString("yyyyMMdd_HHmmss");
            string run = Path.Combine(outputRoot, ts);
            Directory.CreateDirectory(run);
            return run;
        }

        /// <summary>
        /// Gets a path located under the current run directory, creating intermediate folders as necessary.
        /// </summary>
        public static string GetPath(params string[] segments)
        {
            var path = Path.Combine(RunDirectory, Path.Combine(segments));
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            return path;
        }
    }
}
