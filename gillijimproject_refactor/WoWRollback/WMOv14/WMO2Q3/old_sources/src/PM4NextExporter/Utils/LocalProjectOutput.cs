using System;
using System.IO;
using System.Text.RegularExpressions;

namespace PM4NextExporter.Utils
{
    internal static class LocalProjectOutput
    {
        public static string CreateOutputDirectory(string baseName)
        {
            // Sanitize base name for directory safety
            var sanitizedBaseName = Regex.Replace(baseName, "[^a-zA-Z0-9_.-]", "_");
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

            var projectOutputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output");
            var outputDir = Path.Combine(projectOutputDir, $"{sanitizedBaseName}_{timestamp}");

            Directory.CreateDirectory(outputDir);
            return outputDir;
        }
    }
}
