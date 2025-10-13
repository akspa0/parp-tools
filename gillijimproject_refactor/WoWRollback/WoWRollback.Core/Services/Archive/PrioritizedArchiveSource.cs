using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WoWRollback.Core.Services.Archive
{
    public sealed class PrioritizedArchiveSource : IArchiveSource
    {
        private readonly FileSystemArchiveSource _fileSystem;
        private readonly MpqArchiveSource _mpq;
        private static readonly string[] LoosePrefixes = new[] { "", "Data", "data" };

        public PrioritizedArchiveSource(string clientRoot, IEnumerable<string> mpqPaths)
        {
            _fileSystem = new FileSystemArchiveSource(clientRoot);
            _mpq = new MpqArchiveSource(mpqPaths);
        }

        private static IEnumerable<string> ExpandLooseCandidates(string virtualPath)
        {
            var norm = PathUtils.NormalizeVirtual(virtualPath);
            foreach (var prefix in LoosePrefixes)
            {
                yield return string.IsNullOrEmpty(prefix) ? norm : PathUtils.CombineVirtual(prefix, norm);
            }
        }

        public bool FileExists(string virtualPath)
        {
            foreach (var candidate in ExpandLooseCandidates(virtualPath))
            {
                if (_fileSystem.FileExists(candidate))
                    return true;
            }
            return _mpq.FileExists(virtualPath);
        }

        public Stream OpenFile(string virtualPath)
        {
            foreach (var candidate in ExpandLooseCandidates(virtualPath))
            {
                if (_fileSystem.FileExists(candidate))
                {
                    return _fileSystem.OpenFile(candidate);
                }
            }
            return _mpq.OpenFile(virtualPath);
        }

        public IEnumerable<string> EnumerateFiles(string pattern = "*")
        {
            // Union of both sources; loose files should conceptually override MPQ, but here we just list distinct
            var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var p in _fileSystem.EnumerateFiles(pattern))
            {
                set.Add(p);
            }
            foreach (var p in _mpq.EnumerateFiles(pattern))
            {
                set.Add(p);
            }
            return set;
        }

        public void Dispose()
        {
            _fileSystem.Dispose();
            _mpq.Dispose();
        }
    }
}
