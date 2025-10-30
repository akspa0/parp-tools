using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using CASCLib;

namespace WoWRollback.Core.Services.Archive
{
    public sealed class CascArchiveSource : IArchiveSource
    {
        private readonly CASCHandler _casc;
        private readonly List<string> _paths;
        private readonly HashSet<string> _pathSet;
        private readonly Dictionary<string, int> _fdidByPath;

        public CascArchiveSource(string clientRoot, string product, string? listfilePath = null, LocaleFlags? locale = null)
        {
            var basePath = (clientRoot ?? string.Empty).Replace("_retail_", string.Empty).Replace("_ptr_", string.Empty);
            CASCConfig.LoadFlags &= ~(LoadFlags.Download | LoadFlags.Install);
            CASCConfig.ValidateData = false;
            CASCConfig.ThrowOnFileNotFound = false;
            _casc = CASCHandler.OpenLocalStorage(basePath, product);
            if (locale.HasValue)
                _casc.Root.SetFlags(locale.Value);
            else
                _casc.Root.SetFlags(LocaleFlags.enUS);

            _paths = new List<string>();
            _pathSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            _fdidByPath = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            if (!string.IsNullOrWhiteSpace(listfilePath) && File.Exists(listfilePath))
            {
                foreach (var raw in File.ReadLines(listfilePath))
                {
                    if (string.IsNullOrWhiteSpace(raw)) continue;
                    var line = raw.Trim();
                    if (line.StartsWith("#") || line.StartsWith("//")) continue;

                    string? path = null;
                    int? fdid = null;
                    var parts = line.Split(';');
                    if (parts.Length < 2) parts = line.Split(',');
                    if (parts.Length < 2) parts = line.Split('\t');

                    if (parts.Length >= 2)
                    {
                        // FDID ; path
                        if (int.TryParse(parts[0].Trim(), out var parsed)) fdid = parsed;
                        path = parts[1].Trim();
                    }
                    else
                    {
                        // Path-only listfile
                        path = line;
                    }

                    if (string.IsNullOrWhiteSpace(path)) continue;
                    var norm = PathUtils.NormalizeVirtual(path);
                    if (_pathSet.Add(norm)) _paths.Add(norm);
                    if (fdid.HasValue && !_fdidByPath.ContainsKey(norm)) _fdidByPath[norm] = fdid.Value;
                }
            }
        }

        public bool FileExists(string virtualPath)
        {
            var norm = PathUtils.NormalizeVirtual(virtualPath);
            try
            {
                if (_casc.FileExists(norm)) return true;
                if (_fdidByPath.TryGetValue(norm, out var id))
                {
                    return _casc.FileExists(id);
                }
                return false;
            }
            catch { return false; }
        }

        public Stream OpenFile(string virtualPath)
        {
            var norm = PathUtils.NormalizeVirtual(virtualPath);
            var s = _casc.OpenFile(norm);
            if (s != null) return s;
            if (_fdidByPath.TryGetValue(norm, out var id))
            {
                var byId = _casc.OpenFile(id);
                if (byId != null) return byId;
            }
            throw new FileNotFoundException($"File not found in CASC: {virtualPath}");
        }

        public IEnumerable<string> EnumerateFiles(string pattern = "*")
        {
            if (_paths.Count == 0) yield break;
            var normPattern = PathUtils.NormalizeVirtual(pattern);
            var regex = GlobToRegex(normPattern);
            foreach (var p in _paths)
            {
                if (regex.IsMatch(p)) yield return p;
            }
        }

        public void Dispose()
        {
            _casc?.Clear();
        }

        private static Regex GlobToRegex(string pattern)
        {
            var esc = Regex.Escape(pattern).Replace(@"\*", ".*").Replace(@"\?", ".");
            return new Regex("^" + esc + "$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);
        }
    }
}
