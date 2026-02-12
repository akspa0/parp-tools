using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.PM4Module
{
    public class WmoExtractorService
    {
        private readonly IArchiveSource _archiveSource;

        public WmoExtractorService(IArchiveSource archiveSource)
        {
            _archiveSource = archiveSource;
        }

        public void ExtractWmos(string listfilePath, string outputDirectory)
        {
            if (!File.Exists(listfilePath))
                throw new FileNotFoundException("Listfile not found", listfilePath);

            Directory.CreateDirectory(outputDirectory);

            Console.WriteLine("[INFO] Reading listfile...");
            var wmoFiles = File.ReadLines(listfilePath)
                .Where(l => l.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                .Select(l => l.Replace('/', '\\')) // Normalize to backslashes for MPQ
                .ToList();

            Console.WriteLine($"[INFO] Found {wmoFiles.Count} WMO entries in listfile.");
            
            int extracted = 0;
            int skipped = 0;
            int failed = 0;

            foreach (var wmoPath in wmoFiles)
            {
                string destinationPath = Path.Combine(outputDirectory, wmoPath);
                
                // Skip if already exists
                if (File.Exists(destinationPath))
                {
                    skipped++;
                    continue;
                }

                try
                {
                    if (_archiveSource.FileExists(wmoPath))
                    {
                        using var stream = _archiveSource.OpenFile(wmoPath);
                        if (stream != null)
                        {
                            string dir = Path.GetDirectoryName(destinationPath);
                            if (!string.IsNullOrEmpty(dir))
                                Directory.CreateDirectory(dir);

                            using var fs = File.Create(destinationPath);
                            stream.CopyTo(fs);
                            extracted++;
                            
                            if (extracted % 100 == 0)
                                Console.Write($"\r[INFO] Extracted {extracted} WMOs...");
                        }
                    }
                }
                catch (Exception)
                {
                    failed++;
                }
            }

            Console.WriteLine($"\n[INFO] WMO Extraction Complete. Extracted: {extracted}, Skipped: {skipped}, Failed: {failed}");
        }
    }
}
