using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace WoWRollback.Core.Services.Archive
{
    public static class ArchiveLocator
    {
        // Recognize both patch-2.MPQ and patch-enUS-2.MPQ (case-insensitive)
        private static readonly Regex PatchRegex = new Regex(@"patch(?:[-_][a-z]{2}[A-Z]{2})?[-_]?([0-9]+)\.mpq", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        public static IReadOnlyList<string> LocateMpqs(string clientRoot)
        {
            if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
                return Array.Empty<string>();

            var allMpqs = Directory.EnumerateFiles(clientRoot, "*.MPQ", SearchOption.AllDirectories)
                .Concat(Directory.EnumerateFiles(clientRoot, "*.mpq", SearchOption.AllDirectories))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            // Heuristic ordering: base first, patches ascending last (so highest patch gets highest priority)
            var baseMpqs = new List<string>();
            var patchMpqs = new List<(int Order, string Path)>();

            foreach (var path in allMpqs)
            {
                var file = Path.GetFileName(path);
                var m = PatchRegex.Match(file);
                if (m.Success && int.TryParse(m.Groups[1].Value, out var n))
                {
                    patchMpqs.Add((n, path));
                }
                else
                {
                    baseMpqs.Add(path);
                }
            }

            patchMpqs.Sort((a, b) => a.Order.CompareTo(b.Order));

            var ordered = new List<string>();
            ordered.AddRange(baseMpqs);
            ordered.AddRange(patchMpqs.Select(p => p.Path));
            return ordered;
        }
    }
}
