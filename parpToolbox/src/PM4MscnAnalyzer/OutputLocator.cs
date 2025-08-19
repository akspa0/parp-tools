using System;
using System.IO;

namespace PM4MscnAnalyzer
{
    internal static class OutputLocator
    {
        public static string ResolveRoot(string? outDirArg, string? session)
        {
            // If caller supplied an absolute/relative path, create and return it
            if (!string.IsNullOrWhiteSpace(outDirArg))
            {
                var full = Path.GetFullPath(outDirArg);
                Directory.CreateDirectory(full);
                return full;
            }

            // Default root under project_output using ProjectOutput helper
            var baseRoot = ParpToolbox.ProjectOutput.CreateOutputDirectory("mscn_analysis");

            // If session provided, nest a timestamped session folder under the base root
            if (!string.IsNullOrWhiteSpace(session))
            {
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var sessionDir = Path.Combine(baseRoot, $"{session}_{timestamp}");
                Directory.CreateDirectory(sessionDir);
                return sessionDir;
            }

            return baseRoot;
        }

        public static string EnsureSubfolder(string root, string name)
        {
            var path = Path.Combine(root, name);
            Directory.CreateDirectory(path);
            return path;
        }
    }
}
