using System;
using System.IO;

namespace WoWToolbox.Core.v2.Infrastructure
{
    /// <summary>
    /// Central helper for resolving output paths that must live under the current ProjectOutput run directory.
    /// Use this instead of writing next to the source data to avoid polluting inputs.
    /// </summary>
    public static class OutputPaths
    {
        /// <summary>
        /// Returns a sanitized output path rooted at the timestamped <c>project_output</c> directory for the current run.
        /// Example: <c>GetPath("pm4", "development_00_00", "tile.obj")</c> → project_output/{timestamp}/pm4/development_00_00/tile.obj
        /// </summary>
        public static string GetPath(params string[] segments)
        {
            if (segments == null || segments.Length == 0)
                throw new ArgumentException("At least one path segment must be provided", nameof(segments));

            string root = ProjectOutput.GetPath();
            string path = root;
            foreach (var s in segments)
                path = Path.Combine(path, s);
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            return path;
        }
    }
}
