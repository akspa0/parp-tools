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
        private static readonly Lazy<string> _runDir = new(() => CreateRunDir());

        /// <summary>
        /// Gets the root folder for the current run (created on first access).
        /// </summary>
        public static string RunDirectory => _runDir.Value;

        private static string CreateRunDir()
        {
            // Walk up from the executing assembly until we find the solution root (folder containing 'PM4Tool.sln').
            var dir = AppContext.BaseDirectory;
            for (int i = 0; i < 10; i++)
            {
                if (Directory.GetFiles(dir, "*.sln").Length > 0)
                    break;
                dir = Path.GetDirectoryName(dir)!;
            }
            string outputRoot = Path.Combine(dir, "project_output");
            Directory.CreateDirectory(outputRoot);

            string ts = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
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
