using System;
using System.IO;

namespace WoWRollback.Core.Services
{
    /// <summary>
    /// Preflight checks to avoid redoing expensive work when outputs already exist.
    /// </summary>
    public static class PreflightChecks
    {
        /// <summary>
        /// Returns true if LK export outputs appear complete for the given map.
        /// Criteria:
        /// - <map>.wdt exists and is non-empty
        /// - At least expectedCount ADT files exist matching pattern <map>_x_y.adt with non-zero size
        /// </summary>
        public static bool HasCompleteLkAdts(string mapName, string lkOutDir, int expectedCount)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(mapName)) return false;
                if (string.IsNullOrWhiteSpace(lkOutDir) || !Directory.Exists(lkOutDir)) return false;

                var wdt = Path.Combine(lkOutDir, mapName + ".wdt");
                if (!File.Exists(wdt) || new FileInfo(wdt).Length <= 0) return false;

                var searchPattern = mapName + "_*_*.adt";
                int count = 0;
                foreach (var f in Directory.EnumerateFiles(lkOutDir, searchPattern, SearchOption.TopDirectoryOnly))
                {
                    try
                    {
                        var fi = new FileInfo(f);
                        if (fi.Length > 0) count++;
                    }
                    catch { /* ignore broken files */ }
                }
                return count >= expectedCount && expectedCount > 0;
            }
            catch
            {
                return false;
            }
        }
    }
}
