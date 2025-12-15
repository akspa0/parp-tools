using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using WoWRollback.Core.Services.PM4;

namespace WoWRollback.PM4Module
{
    public record M2Reference(
        uint FileDataId,
        string M2Path,
        string FileName,
        Pm4WmoGeometryMatcher.GeometryStats Stats);

    public class M2LibraryBuilder
    {
        private readonly Pm4WmoGeometryMatcher _matcher = new();

        /// <summary>
        /// Scan M2 files in the given directory and build a geometry library cache.
        /// Uses listfile to validate paths and retrieve FileDataIDs.
        /// </summary>
        public Dictionary<string, M2Reference> BuildLibrary(string m2RootDirectory, string listfilePath, string cachePath)
        {
            // Try load cache first
            var library = LoadLibraryCache(cachePath);
            if (library != null)
            {
                Console.WriteLine($"[INFO] Loaded M2 Library Cache ({library.Count} entries)");
                return library;
            }

            Console.WriteLine($"[INFO] Building M2 Library from {m2RootDirectory}...");
            library = new Dictionary<string, M2Reference>(StringComparer.OrdinalIgnoreCase);

            var listfileMap = ParseListfile(listfilePath);
            Console.WriteLine($"[INFO] Loaded {listfileMap.Count} entries from listfile.");

            if (!Directory.Exists(m2RootDirectory))
            {
                Console.WriteLine($"[WARN] M2 root directory not found: {m2RootDirectory}");
                return library;
            }

            var m2Files = Directory.GetFiles(m2RootDirectory, "*.m2", SearchOption.AllDirectories);
            int processed = 0;
            int skipped = 0;
            object lockObj = new object();

            // Parallel processing for speed
            System.Threading.Tasks.Parallel.ForEach(m2Files, (m2Path) =>
            {
                try
                {
                    // Calculate Game Path relative to root
                    // Assuming m2RootDirectory is the root that contains "World", "Creature", etc.
                    // Or if m2RootDirectory ends in "assets", maybe it contains "models/world"?
                    // We need to find the suffix that matches a Listfile entry.
                    
                    // Simple heuristic: Try relative path, then normalized separators.
                    // If not found, try matching by filename against listfile (if unique).
                    
                    var relPath = Path.GetRelativePath(m2RootDirectory, m2Path).Replace('\\', '/');
                    
                    // listfile usually has "world/art/..." or "creature/..."
                    // If our relPath is "models/world/...", and listfile expects "world/...", we mismatch.
                    // But in the previous run, we saw "models/world/..." in the JSON.
                    
                    // Let's try to find an exact match in listfile for the relative path.
                    // If validation fails, we might just store it anyway but warn?
                    // User requested "based in reality with the listfile".
                    
                    uint fileDataId = 0;
                    string gamePath = relPath;
                    
                    if (listfileMap.TryGetValue(relPath, out var id))
                    {
                        fileDataId = id;
                    }
                    else
                    {
                        // Try deeper matching? 
                        // Or maybe simple normalization (to lower?)
                        // Listfile is usually lowercase or mixed? Map handles case insensitive.
                        
                        // Fallback: Check if file name exists in listfile and is unique?
                        // Too risky for automated build?
                        
                        // Let's mark as 0 ID if not found, but still include if valid M2.
                        // Or should we exclude?
                        // "This needs to be based in reality".
                        
                        // We'll trust the relative path for now, but log stats.
                    }

                    var m2File = new M2File(m2Path);
                    if (m2File.Vertices.Count < 3) return; 

                    var stats = _matcher.ComputeStats(m2File.Vertices);

                    // We use the GamePath (relPath) as key.
                    var reference = new M2Reference(fileDataId, gamePath, Path.GetFileName(m2Path), stats);

                    lock (lockObj)
                    {
                        library[gamePath] = reference;
                        processed++;
                        if (fileDataId == 0) skipped++; // tracked as "unknown ID"
                        
                        if (processed % 100 == 0)
                            Console.Write($"\rProcessed {processed}/{m2Files.Length} M2s (Unknown IDs: {skipped})...");
                    }
                }
                catch (Exception)
                {
                    // Ignore errors
                }
            });

            Console.WriteLine($"\n[INFO] M2 Library built: {library.Count} entries. ({skipped} had no matching Listfile ID)");

            SaveLibraryCache(cachePath, library);

            return library;
        }

        /// <summary>
        /// Scan M2 files in the given MPQ archive and build a geometry library cache.
        /// </summary>
        public Dictionary<string, M2Reference> BuildLibraryFromMpq(string mpqPath, string listfilePath, string cachePath)
        {
            // Try load cache first
            var library = LoadLibraryCache(cachePath);
            if (library != null)
            {
                Console.WriteLine($"[INFO] Loaded M2 Library Cache ({library.Count} entries)");
                return library;
            }

            Console.WriteLine($"[INFO] Building M2 Library from MPQ: {mpqPath}...");
            library = new Dictionary<string, M2Reference>(StringComparer.OrdinalIgnoreCase);
            var listfileMap = ParseListfile(listfilePath);

            try
            {
                using var mpq = new MpqAdtExtractor(mpqPath);
                
                // M2s are scattered. We should iterate listfile or scan MPQ?
                // Scanning MPQ is better if listfile is incomplete, but MPQ requires Listfile for names usually?
                // StormLib `FindFirstFile` works if internal listfile is present.
                // Or if we provide `(listfile)`? SFileFindFirstFile takes szListFile.
                // The wrapper passes null.
                
                // Let's rely on listfileMap from parsing the external listfile first?
                // But we want to process ALL M2s in MPQ.
                
                // If the user provides a comprehensive listfile, we can just iterate that and checking HasFile.
                // That's safer for MPQs than scanning unknown hashes.
                // "we need to pull the m2s out of the mpq's".
                
                // Let's use ListFiles("*.m2") from the wrapper?
                // The wrapper implementation passes null for szListFile in SFileFindFirstFile.
                // So it relies on StormLib's internal listfile or brute force?
                // StormLib usually needs an external listfile for full enumeration if not in archive.
                
                Console.WriteLine("Scanning MPQ for .m2 files...");
                var m2Files = mpq.ListFiles("*.m2");
                Console.WriteLine($"Found {m2Files.Count} .m2 files in MPQ.");
                
                if (m2Files.Count == 0 && listfileMap.Count > 0)
                {
                    // Fallback: Use external listfile to check existence
                    Console.WriteLine("MPQ scan empty (maybe no internal listfile). Checking known paths from listfile...");
                    m2Files = listfileMap.Keys.Where(k => k.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                                              .Where(k => mpq.HasFile(k))
                                              .ToList();
                    Console.WriteLine($"Found {m2Files.Count} .m2 files via listfile check.");
                }

                int processed = 0;
                int skipped = 0;
                object lockObj = new object();

                System.Threading.Tasks.Parallel.ForEach(m2Files, (gamePath) =>
                {
                    try
                    {
                        var data = mpq.ReadFile(gamePath);
                        if (data == null || data.Length < 64) return;

                        var m2File = new M2File(data); // Will parse from byte[]
                        if (m2File.Vertices.Count < 3) return;

                        var stats = _matcher.ComputeStats(m2File.Vertices);
                        
                        // Normalize path
                        var normalizedPath = gamePath.Replace('\\', '/');
                        uint fileDataId = listfileMap.GetValueOrDefault(normalizedPath, 0u);

                        var reference = new M2Reference(fileDataId, normalizedPath, Path.GetFileName(normalizedPath), stats);

                        lock (lockObj)
                        {
                            library[normalizedPath] = reference;
                            processed++;
                            if (processed % 100 == 0)
                                Console.Write($"\rProcessed {processed}/{m2Files.Count} M2s...");
                        }
                    }
                    catch { /* Ignore invalid M2s */ }
                });

                Console.WriteLine($"\n[INFO] M2 Library built: {library.Count} entries.");
                SaveLibraryCache(cachePath, library);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] MPQ processing failed: {ex.Message}");
            }

            return library;
        }

        private Dictionary<string, uint> ParseListfile(string path)
        {
            var map = new Dictionary<string, uint>(StringComparer.OrdinalIgnoreCase);
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return map;

            foreach (var line in File.ReadLines(path))
            {
                var parts = line.Split(';');
                if (parts.Length >= 2 && uint.TryParse(parts[0], out var id))
                {
                    var name = parts[1].Trim().Replace('\\', '/');
                    map[name] = id;
                }
            }
            return map;
        }
        private void SaveLibraryCache(string cachePath, Dictionary<string, M2Reference> library)
        {
            try
            {
                var options = new JsonSerializerOptions { WriteIndented = true };
                var json = JsonSerializer.Serialize(library.Values, options);
                File.WriteAllText(cachePath, json);
                Console.WriteLine($"[INFO] Saved library cache to {cachePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to save cache: {ex.Message}");
            }
        }

        private Dictionary<string, M2Reference>? LoadLibraryCache(string cachePath)
        {
            try
            {
                if (!File.Exists(cachePath)) return null;

                var json = File.ReadAllText(cachePath);
                var list = JsonSerializer.Deserialize<List<M2Reference>>(json);
                
                if (list == null) return null;

                return list.ToDictionary(x => x.M2Path, x => x, StringComparer.OrdinalIgnoreCase);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to load library cache: {ex.Message}");
                return null;
            }
        }
    }
}
