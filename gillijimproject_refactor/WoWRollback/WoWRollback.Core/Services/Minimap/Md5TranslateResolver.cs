using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.Core.Services.Minimap
{
    public sealed class Md5TranslateIndex
    {
        // Normalized lower-case forward-slash paths
        public Dictionary<string, string> HashToPlain { get; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, string> PlainToHash { get; } = new(StringComparer.OrdinalIgnoreCase);

        public string Normalize(string s) => s.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();

        public void Add(string a, string b)
        {
            var left = Normalize(a);
            var right = Normalize(b);
            // store both directions; resolver will pick the one that exists
            if (!HashToPlain.ContainsKey(left)) HashToPlain[left] = right;
            if (!PlainToHash.ContainsKey(right)) PlainToHash[right] = left;
            // also insert reversed guess in case file lists are reversed in this build
            if (!HashToPlain.ContainsKey(right)) HashToPlain[right] = left;
            if (!PlainToHash.ContainsKey(left)) PlainToHash[left] = right;
        }
    }

    public static class Md5TranslateResolver
    {
        private static readonly string[] Candidates = new[]
        {
            // Standard locations
            "textures/Minimap/md5translate.txt",
            "textures/Minimap/md5translate.trs",
            // Alternate casing (some clients use lowercase)
            "textures/minimap/md5translate.txt",
            "textures/minimap/md5translate.trs",
            // World prefix (some early clients)
            "world/textures/Minimap/md5translate.txt",
            "world/textures/Minimap/md5translate.trs",
            "world/textures/minimap/md5translate.txt",
            "world/textures/minimap/md5translate.trs",
            // Data subfolder (loose files)
            "Data/textures/Minimap/md5translate.txt",
            "Data/textures/Minimap/md5translate.trs",
            "Data/textures/minimap/md5translate.txt",
            "Data/textures/minimap/md5translate.trs"
        };

        public static bool TryLoad(IArchiveSource source, out Md5TranslateIndex? index, out string? usedVirtualPath)
        {
            index = null;
            usedVirtualPath = null;

            // Strategy: Load MPQ version first (baseline), then overlay loose file version (updates)
            // This ensures we get all minimap tiles (MPQ has base set, loose files have updates)
            
            var idx = new Md5TranslateIndex();
            var foundAny = false;
            var sources = new List<string>();

            foreach (var candidate in Candidates)
            {
                if (!source.FileExists(candidate)) continue;

                try
                {
                    using var stream = source.OpenFile(candidate);
                    using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: false);
                    string? currentDir = null; // from lines like: "dir: Azeroth"
                    string? line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        if (string.IsNullOrWhiteSpace(line)) continue;
                        var trimmed = line.Trim();
                        if (trimmed.StartsWith("#")) continue;

                        // Handle directory context lines
                        if (trimmed.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
                        {
                            var dirName = trimmed.Substring(4).Trim();
                            if (!string.IsNullOrWhiteSpace(dirName))
                            {
                                currentDir = idx.Normalize(dirName).Trim('/');
                            }
                            continue;
                        }

                        // split on whitespace into two parts
                        var parts = trimmed.Split((char[])null!, 2, StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length == 2)
                        {
                            AddWithVariants(idx, parts[0], parts[1], currentDir);
                        }
                        else if (parts.Length > 2)
                        {
                            // best-effort: last token is destination, first is source
                            AddWithVariants(idx, parts[0], parts[^1], currentDir);
                        }
                    }

                    foundAny = true;
                    sources.Add(candidate);
                    // DON'T break - continue to merge all found files
                }
                catch
                {
                    // try next candidate
                }
            }

            if (foundAny)
            {
                index = idx;
                usedVirtualPath = string.Join("; ", sources); // Show all merged sources
                return true;
            }

            return false;
        }

        private static void AddWithVariants(Md5TranslateIndex idx, string plainRaw, string hashedRaw, string? currentDir)
        {
            // Normalize inputs
            string plain = idx.Normalize(plainRaw);
            string hashed = idx.Normalize(hashedRaw);

            // If plain lacks the minimap root, prefix using currentDir if present
            // Examples:
            //  - plain: "azeroth/map00_00.blp" => textures/minimap/azeroth/map00_00.blp
            //  - plain: "map00_00.blp" with dir=Azeroth => textures/minimap/azeroth/map00_00.blp
            var plainDir = plain.Contains('/') ? string.Empty : (currentDir ?? string.Empty);
            var plainPath = !string.IsNullOrEmpty(plainDir)
                ? $"textures/minimap/{plainDir}/{plain}"
                : (plain.StartsWith("textures/minimap/") || plain.StartsWith("world/textures/minimap/"))
                    ? plain
                    : $"textures/minimap/{plain}";

            // Build hashed path variants; in md5 files it's typically just the filename
            var hashFile = hashed.Contains('/') ? Path.GetFileName(hashed) : hashed;
            var hashedTex = $"textures/minimap/{hashFile}";
            var hashedWorld = $"world/textures/minimap/{hashFile}";
            var hashedPlural = $"textures/minimaps/{hashFile}"; // cover plural variant observed in some clients

            // Insert mappings for all variants (both directions via Add)
            idx.Add(plainPath, hashedTex);
            idx.Add(plainPath, hashedWorld);
            idx.Add(plainPath, hashedPlural);

            // Also add a bare plain (without textures/minimap) so resolver can query by short keys
            var shortPlain = plain;
            if (!shortPlain.StartsWith("textures/minimap/") && !shortPlain.StartsWith("world/textures/minimap/"))
            {
                if (!string.IsNullOrEmpty(plainDir) && !shortPlain.StartsWith(plainDir + "/"))
                {
                    shortPlain = $"{plainDir}/{shortPlain}";
                }
                idx.Add(shortPlain, hashedTex);
                idx.Add(shortPlain, hashedWorld);
                idx.Add(shortPlain, hashedPlural);
            }
        }
    }
}
