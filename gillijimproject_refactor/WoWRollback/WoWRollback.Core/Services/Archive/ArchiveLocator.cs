using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace WoWRollback.Core.Services.Archive
{
    public static class ArchiveLocator
    {
        // Recognize numeric patches: patch-2.MPQ, patch_enUS-2.MPQ (case-insensitive)
        private static readonly Regex PatchNumericRegex = new Regex(@"patch(?:[-_][a-z]{2}[A-Z]{2})?[-_]?([0-9]+)\.mpq", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        // Recognize letter patches: patch-A.MPQ, patch-enUS-A.MPQ (case-insensitive)
        private static readonly Regex PatchLetterRegex = new Regex(@"patch(?:[-_][a-z]{2}[A-Z]{2})?[-_]?([A-Za-z])\.mpq", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        public static IReadOnlyList<string> LocateMpqs(string clientRoot)
        {
            if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
                return Array.Empty<string>();

            var allMpqs = Directory.EnumerateFiles(clientRoot, "*.MPQ", SearchOption.AllDirectories)
                .Concat(Directory.EnumerateFiles(clientRoot, "*.mpq", SearchOption.AllDirectories))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            // Heuristic ordering: base first, numeric patches ascending, then letter patches ascending (A..Z) last for highest priority
            var baseMpqs = new List<string>();
            var patchNumerics = new List<(int Order, string Path)>();
            var patchLetters = new List<(char Letter, string Path)>();

            foreach (var path in allMpqs)
            {
                var file = Path.GetFileName(path);
                var mNum = PatchNumericRegex.Match(file);
                if (mNum.Success && int.TryParse(mNum.Groups[1].Value, out var n))
                {
                    patchNumerics.Add((n, path));
                    continue;
                }

                var mLet = PatchLetterRegex.Match(file);
                if (mLet.Success && mLet.Groups[1].Success)
                {
                    char c = char.ToUpperInvariant(mLet.Groups[1].Value[0]);
                    patchLetters.Add((c, path));
                    continue;
                }

                baseMpqs.Add(path);
            }

            patchNumerics.Sort((a, b) => a.Order.CompareTo(b.Order));
            patchLetters.Sort((a, b) => a.Letter.CompareTo(b.Letter));

            var ordered = new List<string>();
            ordered.AddRange(baseMpqs);
            ordered.AddRange(patchNumerics.Select(p => p.Path));
            ordered.AddRange(patchLetters.Select(p => p.Path));
            return ordered;
        }
    }
}
