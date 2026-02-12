using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using WoWRollback.Core.Services.Archive;
using WoWFormatLib.FileProviders;

namespace WoWRollback.AnalysisModule
{
    /// <summary>
    /// IArchiveSource implementation backed by CASCLib via WoWFormatLib's CASCFileProvider.
    /// Supports simple wildcard enumeration using an optional listfile (fdid;path).
    /// Paths are treated as virtual, case-insensitive, with forward slashes.
    /// </summary>
    public sealed class CascArchiveSource : IArchiveSource
    {
        private readonly CASCFileProvider _provider = new();
        private readonly HashSet<string> _knownPaths = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, uint> _fdidByPath = new(StringComparer.OrdinalIgnoreCase);
        private readonly string _baseDir;
        private readonly string _product;

        public CascArchiveSource(string baseDir, string product = "wow", string? listfilePath = null)
        {
            _baseDir = baseDir ?? throw new ArgumentNullException(nameof(baseDir));
            _product = string.IsNullOrWhiteSpace(product) ? "wow" : product;

            // Initialize CASC from local storage (requires .build.info under baseDir or parent)
            _provider.InitCasc(basedir: _baseDir, program: _product);

            // Load optional listfile (format: id;path or id,path or id\tpath)
            if (!string.IsNullOrWhiteSpace(listfilePath) && File.Exists(listfilePath))
            {
                foreach (var line in File.ReadLines(listfilePath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var trimmed = line.Trim();
                    if (trimmed.StartsWith("#") || trimmed.StartsWith("//")) continue;

                    string[] parts = trimmed.Split(';');
                    if (parts.Length < 2) parts = trimmed.Split(',');
                    if (parts.Length < 2) parts = trimmed.Split('\t');
                    if (parts.Length < 2) continue;

                    if (!uint.TryParse(parts[0].Trim(), out var fdid)) continue;
                    var path = Normalize(parts[1].Trim());
                    if (string.IsNullOrWhiteSpace(path)) continue;

                    _knownPaths.Add(path);
                    if (!_fdidByPath.ContainsKey(path)) _fdidByPath[path] = fdid;
                }
            }
        }

        public bool FileExists(string virtualPath)
        {
            var vp = Normalize(virtualPath);
            if (string.IsNullOrWhiteSpace(vp)) return false;
            // Try direct name lookup via CASCLib
            if (_provider.FileExists(vp)) return true;
            var lower = vp.ToLowerInvariant();
            if (_provider.FileExists(lower)) return true;
            // Try listfile-backed FDID lookup
            if (_fdidByPath.TryGetValue(vp, out var fdid)) return _provider.FileExists(fdid);
            if (_fdidByPath.TryGetValue(lower, out fdid)) return _provider.FileExists(fdid);
            return false;
        }

        public Stream OpenFile(string virtualPath)
        {
            var vp = Normalize(virtualPath);
            if (_provider.FileExists(vp)) return _provider.OpenFile(vp);
            var lower = vp.ToLowerInvariant();
            if (_provider.FileExists(lower)) return _provider.OpenFile(lower);
            if (_fdidByPath.TryGetValue(vp, out var fdid)) return _provider.OpenFile(fdid);
            if (_fdidByPath.TryGetValue(lower, out fdid)) return _provider.OpenFile(fdid);
            throw new FileNotFoundException($"CASC: File not found: {virtualPath}");
        }

        public IEnumerable<string> EnumerateFiles(string pattern = "*")
        {
            if (_knownPaths.Count == 0) yield break;
            var rx = GlobToRegex(pattern);
            foreach (var p in _knownPaths)
            {
                if (rx.IsMatch(p)) yield return p;
            }
        }

        public void Dispose()
        {
            // CASCFileProvider holds static handler; nothing to dispose per instance
        }

        private static string Normalize(string path)
        {
            return path.Replace('\\', '/');
        }

        private static Regex GlobToRegex(string pattern)
        {
            // Very small glob: * only, case-insensitive, match entire string
            var escaped = Regex.Escape(Normalize(pattern)).Replace("\\*", ".*");
            return new Regex("^" + escaped + "$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        }
    }
}
