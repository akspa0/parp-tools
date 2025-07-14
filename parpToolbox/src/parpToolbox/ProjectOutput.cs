using System;
using System.IO;
using System.Text.RegularExpressions;

namespace parpToolbox
{
    public static class ProjectOutput
    {
        public static string CreateOutputDirectory(string baseName)
        {
            // Sanitize the base name to remove invalid characters for a directory name
            var sanitizedBaseName = Regex.Replace(baseName, "[^a-zA-Z0-9_.-]", "_");
            
            // Get the current timestamp
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

            // Define the root output directory
            var projectOutputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output");

            // Create the final timestamped directory path
            var outputDir = Path.Combine(projectOutputDir, $"{sanitizedBaseName}_{timestamp}");

            // Create the directory if it doesn't exist
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            return outputDir;
        }
    }
}