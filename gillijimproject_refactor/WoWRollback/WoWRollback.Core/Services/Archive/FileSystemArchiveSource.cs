using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WoWRollback.Core.Services.Archive
{
    public sealed class FileSystemArchiveSource : IArchiveSource
    {
        private readonly string _root;

        public FileSystemArchiveSource(string root)
        {
            _root = Path.GetFullPath(root);
        }

        public bool FileExists(string virtualPath)
        {
            var osPath = PathUtils.ToOsPath(_root, virtualPath);
            return File.Exists(osPath);
        }

        public Stream OpenFile(string virtualPath)
        {
            var osPath = PathUtils.ToOsPath(_root, virtualPath);
            return File.OpenRead(osPath);
        }

        public IEnumerable<string> EnumerateFiles(string pattern = "*")
        {
            var normalized = pattern.Replace('\\', '/');
            var baseDir = _root;
            string searchPattern = "*";

            // If pattern contains folders, split and enumerate appropriately
            if (normalized.Contains('/'))
            {
                var lastSlash = normalized.LastIndexOf('/');
                var subdir = normalized.Substring(0, lastSlash);
                searchPattern = normalized.Substring(lastSlash + 1);
                baseDir = PathUtils.ToOsPath(_root, subdir);
            }
            else
            {
                searchPattern = normalized;
            }

            if (!Directory.Exists(baseDir)) yield break;

            foreach (var file in Directory.EnumerateFiles(baseDir, searchPattern, SearchOption.AllDirectories))
            {
                var rel = file.Substring(_root.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                yield return PathUtils.NormalizeVirtual(rel);
            }
        }

        public void Dispose() { }
    }
}
