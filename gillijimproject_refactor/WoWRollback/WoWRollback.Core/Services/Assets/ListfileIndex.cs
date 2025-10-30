using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWRollback.Core.Services.Assets
{
    public sealed class ListfileIndex
    {
        private readonly Dictionary<string, uint> _fdidByPath;
        private readonly Dictionary<uint, string> _pathByFdid;
        private readonly HashSet<string> _paths;

        public string SourcePath { get; }

        public ListfileIndex(string sourcePath)
        {
            SourcePath = sourcePath;
            _fdidByPath = new Dictionary<string, uint>(StringComparer.OrdinalIgnoreCase);
            _pathByFdid = new Dictionary<uint, string>();
            _paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        }

        public static ListfileIndex Load(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("path");
            if (!File.Exists(path)) throw new FileNotFoundException("Listfile not found", path);

            var idx = new ListfileIndex(path);
            foreach (var raw in File.ReadLines(path, Encoding.UTF8))
            {
                if (string.IsNullOrWhiteSpace(raw)) continue;
                var line = raw.Trim();
                if (line.Length == 0) continue;
                if (line.StartsWith("#") || line.StartsWith("//")) continue;

                string[] parts = line.Split(';');
                if (parts.Length < 2) parts = line.Split(',');
                if (parts.Length < 2) parts = line.Split('\t');

                if (parts.Length >= 2)
                {
                    if (uint.TryParse(parts[0].Trim(), out var fdid))
                    {
                        var pathPart = Normalize(parts[1]);
                        if (pathPart.Length == 0) continue;
                        if (!idx._paths.Contains(pathPart)) idx._paths.Add(pathPart);
                        if (!idx._fdidByPath.ContainsKey(pathPart)) idx._fdidByPath[pathPart] = fdid;
                        if (!idx._pathByFdid.ContainsKey(fdid)) idx._pathByFdid[fdid] = pathPart;
                        continue;
                    }
                }

                var onlyPath = Normalize(line);
                if (onlyPath.Length == 0) continue;
                idx._paths.Add(onlyPath);
            }
            return idx;
        }

        public bool ContainsPath(string path)
        {
            var p = Normalize(path);
            return _paths.Contains(p);
        }

        public bool TryGetFdidByPath(string path, out uint fdid)
        {
            var p = Normalize(path);
            return _fdidByPath.TryGetValue(p, out fdid);
        }

        public bool TryGetPathByFdid(uint fdid, out string path)
        {
            return _pathByFdid.TryGetValue(fdid, out path!);
        }

        public IEnumerable<string> EnumeratePaths(string? startsWith = null)
        {
            if (string.IsNullOrEmpty(startsWith)) return _paths;
            var prefix = Normalize(startsWith);
            return _paths.Where(p => p.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
        }

        private static string Normalize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return string.Empty;
            var t = s.Trim().Replace('\\', '/');
            return t;
        }
    }
}
