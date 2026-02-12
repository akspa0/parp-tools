using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace WoWRollback.Core.Services.Config;

/// <summary>
/// Loads area override definitions from JSON/JSONC files and resolves per-map, per-version overrides.
/// </summary>
public static class AreaOverrideLoader
{
    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    /// <summary>
    /// Attempts to load area overrides from the specified directory. Returns <c>null</c> when the directory
    /// is null, missing, or contains no valid override entries.
    /// </summary>
    public static AreaOverrideResolver? LoadFromDirectory(string? directory)
    {
        if (string.IsNullOrWhiteSpace(directory)) return null;
        if (!Directory.Exists(directory)) return null;

        var mapOverrides = new Dictionary<string, MapOverrideSet>(StringComparer.OrdinalIgnoreCase);
        var files = EnumerateConfigFiles(directory).ToArray();
        if (files.Length == 0) return null;

        foreach (var file in files)
        {
            try
            {
                using var stream = File.OpenRead(file);
                var config = JsonSerializer.Deserialize<AreaOverrideConfig>(stream, s_jsonOptions);
                if (config is null) continue;
                MergeConfig(mapOverrides, config);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[WARN] Failed to load area overrides from '{file}': {ex.Message}");
            }
        }

        return mapOverrides.Count == 0 ? null : new AreaOverrideResolver(mapOverrides);
    }

    private static IEnumerable<string> EnumerateConfigFiles(string root)
    {
        foreach (var pattern in new[] { "*.json", "*.jsonc" })
        {
            foreach (var file in Directory.EnumerateFiles(root, pattern, SearchOption.AllDirectories))
            {
                yield return file;
            }
        }
    }

    private static void MergeConfig(Dictionary<string, MapOverrideSet> lookup, AreaOverrideConfig config)
    {
        if (config.Maps is null || config.Maps.Count == 0) return;

        foreach (var (mapName, mapOverride) in config.Maps)
        {
            if (string.IsNullOrWhiteSpace(mapName) || mapOverride is null) continue;
            if (!lookup.TryGetValue(mapName, out var set))
            {
                set = new MapOverrideSet();
                lookup[mapName] = set;
            }

            set.MergeDefaults(mapOverride.Overrides);

            if (mapOverride.Versions is null || mapOverride.Versions.Count == 0) continue;
            foreach (var (versionKey, versionOverride) in mapOverride.Versions)
            {
                if (string.IsNullOrWhiteSpace(versionKey) || versionOverride is null) continue;
                set.MergeVersion(versionKey, versionOverride.Overrides);
            }
        }
    }

    internal sealed class MapOverrideSet
    {
        private readonly Dictionary<int, int> _defaults = new();
        private readonly Dictionary<string, Dictionary<int, int>> _versions = new(StringComparer.OrdinalIgnoreCase);

        public void MergeDefaults(IEnumerable<AreaOverrideEntry>? entries)
        {
            if (entries is null) return;
            foreach (var entry in entries)
            {
                if (entry is null) continue;
                _defaults[entry.AlphaArea] = entry.TargetAreaId;
            }
        }

        public void MergeVersion(string versionKey, IEnumerable<AreaOverrideEntry>? entries)
        {
            if (entries is null) return;
            if (!_versions.TryGetValue(versionKey, out var dict))
            {
                dict = new Dictionary<int, int>();
                _versions[versionKey] = dict;
            }

            foreach (var entry in entries)
            {
                if (entry is null) continue;
                dict[entry.AlphaArea] = entry.TargetAreaId;
            }
        }

        public IReadOnlyDictionary<int, int>? BuildOverrides(IEnumerable<string?> versionKeys)
        {
            var result = new Dictionary<int, int>(_defaults);
            if (versionKeys is not null)
            {
                foreach (var key in versionKeys)
                {
                    if (string.IsNullOrWhiteSpace(key)) continue;
                    if (_versions.TryGetValue(key!, out var overrides))
                    {
                        foreach (var pair in overrides)
                        {
                            result[pair.Key] = pair.Value;
                        }
                    }
                }
            }

            return result.Count == 0 ? null : new ReadOnlyDictionary<int, int>(result);
        }
    }

    public sealed class AreaOverrideResolver
    {
        private readonly Dictionary<string, MapOverrideSet> _maps;

        internal AreaOverrideResolver(Dictionary<string, MapOverrideSet> maps)
        {
            _maps = maps;
        }

        public IReadOnlyDictionary<int, int>? GetOverrides(string mapName, params string?[] versionKeys)
        {
            if (string.IsNullOrWhiteSpace(mapName)) return null;
            if (!_maps.TryGetValue(mapName, out var set)) return null;
            return set.BuildOverrides(versionKeys);
        }
    }
}
