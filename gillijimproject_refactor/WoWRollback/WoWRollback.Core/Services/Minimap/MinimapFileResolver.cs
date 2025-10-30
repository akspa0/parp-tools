using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.Core.Services.Minimap
{
    public interface IMinimapFileResolver
    {
        bool TryResolveTile(string mapName, int tileX, int tileY, out string? blpVirtualPath);
    }

    public sealed class MinimapFileResolver : IMinimapFileResolver
    {
        private readonly IArchiveSource _source;
        private readonly Md5TranslateIndex? _index;
        private readonly Dictionary<string, List<string>> _globCache = new(StringComparer.OrdinalIgnoreCase);

        public MinimapFileResolver(IArchiveSource source, Md5TranslateIndex? index)
        {
            _source = source;
            _index = index;
        }

        private IEnumerable<string> EnumerateCached(string pattern)
        {
            if (_globCache.TryGetValue(pattern, out var cached)) return cached;
            var list = _source.EnumerateFiles(pattern).ToList();
            _globCache[pattern] = list;
            return list;
        }

        public bool TryResolveTile(string mapName, int tileX, int tileY, out string? blpVirtualPath)
        {
            blpVirtualPath = null;
            var candidates = EnumeratePlainCandidates(mapName, tileX, tileY);

            // 1) If md5translate exists, prioritize hashed lookups (for Alpha/early clients)
            if (_index is not null)
            {
                foreach (var plain in candidates)
                {
                    if (_index.PlainToHash.TryGetValue(Normalize(plain), out var hashed))
                    {
                        if (_source.FileExists(hashed))
                        {
                            blpVirtualPath = hashed;
                            return true;
                        }
                    }
                }
            }

            // 2) Direct plain candidates (for later clients without md5 hashing)
            foreach (var plain in candidates)
            {
                if (_source.FileExists(plain))
                {
                    blpVirtualPath = plain;
                    return true;
                }
            }

            // 3) Fallback scan under Textures/Minimap/<map>
            var dirPattern = $"textures/Minimap/{mapName}/*.blp";
            var entries = EnumerateCached(dirPattern)
                .Where(p => p.EndsWith($"_{tileX}_{tileY}.blp", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (entries.Count > 0)
            {
                blpVirtualPath = entries[0];
                return true;
            }

            // 4) Fallback scan under Textures/Minimap root (some clients flatten tiles)
            var rootEntries = EnumerateCached("textures/Minimap/*.blp")
                .Where(p => p.EndsWith($"{mapName}_{tileX}_{tileY}.blp", StringComparison.OrdinalIgnoreCase)
                         || p.EndsWith($"map{tileX}_{tileY}.blp", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (rootEntries.Count > 0)
            {
                blpVirtualPath = rootEntries[0];
                return true;
            }

            // 5) As last resort, try md5 hash candidates present in index for this map folder
            if (_index is not null)
            {
                var suffix = $"_{tileX}_{tileY}.blp";
                foreach (var kv in _index.HashToPlain)
                {
                    var plain = kv.Value;
                    if (!plain.StartsWith($"textures/minimap/{mapName}/", StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (!plain.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
                        continue;
                    var hashed = kv.Key;
                    if (_source.FileExists(hashed))
                    {
                        blpVirtualPath = hashed;
                        return true;
                    }
                    if (_source.FileExists(plain))
                    {
                        blpVirtualPath = plain;
                        return true;
                    }
                }
            }

            return false;
        }

        private static IEnumerable<string> EnumeratePlainCandidates(string mapName, int tileX, int tileY)
        {
            var x2 = tileX.ToString("00");
            var y2 = tileY.ToString("00");
            
            // Generate space-separated variant for 0.6.0 bug (e.g., "EmeraldDream" → "Emerald Dream")
            var mapNameWithSpace = InsertSpaceBeforeCapitals(mapName);
            
            // Alpha/early clients use "map##_##.blp" format in md5translate
            yield return $"textures/minimap/{mapName}/map{x2}_{y2}.blp";
            yield return $"{mapName}/map{x2}_{y2}.blp"; // short form for md5translate lookup
            
            // 0.6.0 bug: Try space-separated variant (e.g., "Emerald Dream" instead of "EmeraldDream")
            if (mapNameWithSpace != mapName)
            {
                yield return $"textures/minimap/{mapNameWithSpace}/map{x2}_{y2}.blp";
                yield return $"{mapNameWithSpace}/map{x2}_{y2}.blp";
            }
            
            // common forms under subfolder
            yield return $"textures/Minimap/{mapName}/{mapName}_{tileX}_{tileY}.blp";
            yield return $"textures/Minimap/{mapName}/{mapName}_{x2}_{y2}.blp";
            yield return $"textures/Minimap/{mapName}/map{tileX}_{tileY}.blp";
            
            // Space-separated variants
            if (mapNameWithSpace != mapName)
            {
                yield return $"textures/Minimap/{mapNameWithSpace}/{mapNameWithSpace}_{tileX}_{tileY}.blp";
                yield return $"textures/Minimap/{mapNameWithSpace}/{mapNameWithSpace}_{x2}_{y2}.blp";
                yield return $"textures/Minimap/{mapNameWithSpace}/map{tileX}_{tileY}.blp";
            }
            
            // sometimes placed directly under Minimap root
            yield return $"textures/Minimap/{mapName}_{tileX}_{tileY}.blp";
            yield return $"textures/Minimap/{mapName}_{x2}_{y2}.blp";
            yield return $"textures/Minimap/map{x2}_{y2}.blp";
        }
        
        /// <summary>
        /// Inserts spaces before capital letters in camelCase/PascalCase strings.
        /// E.g., "EmeraldDream" → "Emerald Dream"
        /// Handles 0.6.0 bug where some map names had spaces in md5translate.
        /// </summary>
        private static string InsertSpaceBeforeCapitals(string input)
        {
            if (string.IsNullOrEmpty(input) || input.Length == 1)
                return input;
            
            var result = new System.Text.StringBuilder();
            result.Append(input[0]);
            
            for (int i = 1; i < input.Length; i++)
            {
                if (char.IsUpper(input[i]) && !char.IsUpper(input[i - 1]))
                {
                    result.Append(' ');
                }
                result.Append(input[i]);
            }
            
            return result.ToString();
        }

        private static string Normalize(string s) => s.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }
}
