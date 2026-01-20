using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWMapConverter.Core.Services;

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
        // Data subfolder (loose files - implicitly covered by search paths but added for completeness if relative)
        "Data/textures/Minimap/md5translate.txt",
        "Data/textures/Minimap/md5translate.trs"
    };

    public static bool TryLoad(IEnumerable<string> searchPaths, NativeMpqService mpqService, out Md5TranslateIndex? index, IEnumerable<string>? extraCandidates = null)
    {
        index = null;

        // Strategy: Load MPQ version first (baseline), then overlay loose file version (updates)
        var idx = new Md5TranslateIndex();
        var foundAny = false;
        
        // 0. Check Extra Candidates (Map-specific)
        if (extraCandidates != null)
        {
            foreach (var candidate in extraCandidates)
            {
                var mpqKey = candidate.Replace("/", "\\");
                if (mpqService.FileExists(mpqKey))
                {
                    Console.WriteLine($"[Md5Translate] Found map-specific index: {mpqKey}");
                    
                    // Infer context from path: World\Maps\{MapName}\md5translate.trs
                    string? initialDir = null;
                    if (mpqKey.Contains("World\\Maps\\", StringComparison.OrdinalIgnoreCase))
                    {
                        var parts = mpqKey.Split('\\');
                        // Find "Maps" and take next
                        for (int i = 0; i < parts.Length - 1; i++)
                        {
                            if (string.Equals(parts[i], "Maps", StringComparison.OrdinalIgnoreCase))
                            {
                                initialDir = parts[i + 1];
                                Console.WriteLine($"[Md5Translate] Inferred context: {initialDir}");
                                break;
                            }
                        }
                    }

                    var data = mpqService.ReadFile(mpqKey);
                    if (data != null)
                    {
                        using var ms = new MemoryStream(data);
                        ParseStream(ms, idx, initialDir);
                        foundAny = true;
                    }
                }
            }
        }
        
        // 1. Check MPQ (Global)
        // We check candidates against the MPQ service
        foreach (var candidate in Candidates)
        {
            var mpqKey = candidate.Replace("/", "\\");
            if (mpqService.FileExists(mpqKey))
            {
                var data = mpqService.ReadFile(mpqKey);
                if (data != null)
                {
                    using var ms = new MemoryStream(data);
                    ParseStream(ms, idx); // No inferred context for global files
                    foundAny = true;
                }
            }
        }
        
        // 2. Check Disk (Loose)
        foreach (var basePath in searchPaths)
        {
            foreach (var candidate in Candidates)
            {
                var fullPath = Path.Combine(basePath, candidate);
                if (File.Exists(fullPath))
                {
                    using var fs = File.OpenRead(fullPath);
                    ParseStream(fs, idx);
                    foundAny = true;
                }
            }
        }

        if (foundAny)
        {
            index = idx;
            return true;
        }

        return false;
    }

    private static void ParseStream(Stream stream, Md5TranslateIndex idx, string? initialDir = null)
    {
        try
        {
            using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: true);
            string? currentDir = initialDir != null ? idx.Normalize(initialDir).Trim('/') : null; 
            string? line;
            int count = 0;
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
                        // Console.WriteLine($"[DEBUG] TRS Context Switch: {currentDir}");
                    }
                    continue;
                }

                // split on whitespace into two parts (tab or space)
                var parts = trimmed.Split((char[])null!, 2, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 2)
                {
                    // TRS format per wowdev.wiki/TRS.md:
                    //   block_entry := map_basename "\map" x "_" y ".blp\t" actual_filename "\n"
                    // So parts[0] = plain name (e.g., "Azeroth\map26_29.blp")
                    //    parts[1] = md5 hash filename (e.g., "fa32ced4...blp")
                    AddWithVariants(idx, parts[0], parts[1], currentDir);
                    count++;
                }
                else if (parts.Length > 2)
                {
                    // best-effort: first is plain, last is hash
                    AddWithVariants(idx, parts[0], parts[^1], currentDir);
                    count++;
                }
            }
            Console.WriteLine($"[DEBUG] Parsed {count} entries from TRS stream.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error parsing md5translate stream: {ex.Message}");
        }
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
