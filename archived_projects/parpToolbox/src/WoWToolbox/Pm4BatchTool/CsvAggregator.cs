using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
    /// Small helper that concatenates many CSV files (sharing identical header) into a single CSV.
    /// Used by Program.cs after the "analyze-links" command to create master summaries.
    /// </summary>
internal static class CsvAggregator
    {
        /// <summary>
        /// Finds every file inside <paramref name="rootDir"/> matching <paramref name="searchPattern"/>,
        /// concatenates them (skipping duplicate headers) and writes the combined output to <paramref name="destPath"/>.
        /// </summary>
        internal static void Aggregate(string rootDir, string searchPattern, string destPath)
        {
            if (!Directory.Exists(rootDir))
                throw new DirectoryNotFoundException(rootDir);

            var files = Directory.GetFiles(rootDir, searchPattern, SearchOption.AllDirectories)
                                  .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                                  .ToList();
            if (files.Count == 0)
                return; // nothing to do

            // Ensure destination directory exists
            Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);

            using var writer = new StreamWriter(destPath, false);
            bool wroteHeader = false;

            foreach (var file in files)
            {
                using var reader = new StreamReader(file);
                string? header = reader.ReadLine();
                if (header == null) continue; // empty file

                if (!wroteHeader)
                {
                    writer.WriteLine(header);
                    wroteHeader = true;
                }

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    writer.WriteLine(line);
                }
            }
        }
}
