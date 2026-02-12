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
        private static readonly Regex PatchPlainRegex = new Regex(@"^patch(?:[-_][a-z]{2}[A-Z]{2})?\.mpq$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static bool IsLocalePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            var p = path.Replace('\\', '/');
            var idx = p.IndexOf("/Data/", StringComparison.OrdinalIgnoreCase);
            if (idx < 0) return false;
            var rest = p.Substring(idx + 6);
            var slash = rest.IndexOf('/') >= 0 ? rest.IndexOf('/') : rest.Length;
            if (slash <= 0) return false;
            var seg = rest.Substring(0, slash);
            if (seg.Length != 4) return false;
            bool letters = char.IsLetter(seg[0]) && char.IsLetter(seg[1]) && char.IsLetter(seg[2]) && char.IsLetter(seg[3]);
            return letters;
        }

        public static IReadOnlyList<string> LocateMpqs(string clientRoot)
        {
            if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
                return Array.Empty<string>();

            var allMpqs = Directory.EnumerateFiles(clientRoot, "*.MPQ", SearchOption.AllDirectories)
                .Concat(Directory.EnumerateFiles(clientRoot, "*.mpq", SearchOption.AllDirectories))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            var baseMpqs = new List<string>();
            var localeNumeric = new List<(int Order, string Path)>();
            var localeLetter = new List<(char Letter, string Path)>();
            var rootNumeric = new List<(int Order, string Path)>();
            var rootLetter = new List<(char Letter, string Path)>();

            foreach (var path in allMpqs)
            {
                var file = Path.GetFileName(path);
                var mNum = PatchNumericRegex.Match(file);
                if (mNum.Success && int.TryParse(mNum.Groups[1].Value, out var n))
                {
                    if (IsLocalePath(path)) localeNumeric.Add((n, path)); else rootNumeric.Add((n, path));
                    continue;
                }

                var mLet = PatchLetterRegex.Match(file);
                if (mLet.Success && mLet.Groups[1].Success)
                {
                    char c = char.ToUpperInvariant(mLet.Groups[1].Value[0]);
                    if (IsLocalePath(path)) localeLetter.Add((c, path)); else rootLetter.Add((c, path));
                    continue;
                }

                var mPlain = PatchPlainRegex.Match(file);
                if (mPlain.Success)
                {
                    if (IsLocalePath(path)) localeNumeric.Add((1, path)); else rootNumeric.Add((1, path));
                    continue;
                }

                baseMpqs.Add(path);
            }

            localeNumeric.Sort((a, b) => a.Order.CompareTo(b.Order));
            localeLetter.Sort((a, b) => a.Letter.CompareTo(b.Letter));
            rootNumeric.Sort((a, b) => a.Order.CompareTo(b.Order));
            rootLetter.Sort((a, b) => a.Letter.CompareTo(b.Letter));

            var ordered = new List<string>();
            ordered.AddRange(baseMpqs);
            ordered.AddRange(localeNumeric.Select(p => p.Path));
            ordered.AddRange(rootNumeric.Select(p => p.Path));
            ordered.AddRange(localeLetter.Select(p => p.Path));
            ordered.AddRange(rootLetter.Select(p => p.Path));
            return ordered;
        }
    }
}
