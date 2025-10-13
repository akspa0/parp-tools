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

        public MinimapFileResolver(IArchiveSource source, Md5TranslateIndex? index)
        {
            _source = source;
            _index = index;
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
            var entries = _source.EnumerateFiles(dirPattern)
                .Where(p => p.EndsWith($"_{tileX}_{tileY}.blp", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (entries.Count > 0)
            {
                blpVirtualPath = entries[0];
                return true;
            }

            // 4) Fallback scan under Textures/Minimap root (some clients flatten tiles)
            var rootEntries = _source.EnumerateFiles("textures/Minimap/*.blp")
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
            
            // Alpha/early clients use "map##_##.blp" format in md5translate
            yield return $"textures/minimap/{mapName}/map{x2}_{y2}.blp";
            yield return $"{mapName}/map{x2}_{y2}.blp"; // short form for md5translate lookup
            
            // common forms under subfolder
            yield return $"textures/Minimap/{mapName}/{mapName}_{tileX}_{tileY}.blp";
            yield return $"textures/Minimap/{mapName}/{mapName}_{x2}_{y2}.blp";
            yield return $"textures/Minimap/{mapName}/map{tileX}_{tileY}.blp";
            
            // sometimes placed directly under Minimap root
            yield return $"textures/Minimap/{mapName}_{tileX}_{tileY}.blp";
            yield return $"textures/Minimap/{mapName}_{x2}_{y2}.blp";
            yield return $"textures/Minimap/map{x2}_{y2}.blp";
        }

        private static string Normalize(string s) => s.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }
}
