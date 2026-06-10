using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWFormatLib;
using WoWFormatLib.FileProviders;
using WoWFormatLib.Utils;

namespace WoWFormatLib.FileProviders
{
    /// <summary>
    /// Local file provider that serves files from a base directory on disk.
    /// Supports filename and FileDataID access. Filename->FDID uses Listfile with
    /// lazy loading and synthetic IDs for unmapped local files.
    /// </summary>
    public sealed class LocalFileProvider : IFileProvider
    {
        private readonly string _baseDir;
        private bool _listfileChecked;
        private readonly object _sync = new();

        // Synthetic ID mapping so that IDs returned from GetFileDataIdByName can
        // be resolved later by OpenFile(uint)/FileExists(uint)
        private readonly Dictionary<uint, string> _syntheticIdToPath = new();
        private readonly Dictionary<string, uint> _pathToSyntheticId = new(StringComparer.OrdinalIgnoreCase);

        public LocalFileProvider(string baseDir)
        {
            if (string.IsNullOrWhiteSpace(baseDir))
                throw new ArgumentException("Base directory cannot be null or empty", nameof(baseDir));

            _baseDir = Path.GetFullPath(baseDir);
        }

        public void SetBuild(string build)
        {
            // Accept any build string; typically "local". No special handling required.
            // Kept to satisfy the interface and match FileProvider expectations.
            _ = build;
        }

        public uint GetFileDataIdByName(string filename)
        {
            if (string.IsNullOrWhiteSpace(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            EnsureListfileChecked();

            // Try direct listfile match first
            if (Listfile.TryGetFileDataID(filename, out var fdid))
                return fdid;

            // If absolute and under base dir, try relative-normalized
            var relativeNorm = TryGetRelativeNormalized(filename);
            if (relativeNorm != null && Listfile.TryGetFileDataID(relativeNorm, out fdid))
                return fdid;

            // Fall back to synthetic mapping so caller can later open by FDID
            var key = relativeNorm ?? NormalizeToRepoStyle(filename);
            return GetOrCreateSyntheticId(key);
        }

        public bool FileExists(string filename)
        {
            var full = ResolveFullPath(filename);
            return File.Exists(full);
        }

        public Stream OpenFile(string filename)
        {
            var full = ResolveFullPath(filename);
            if (!File.Exists(full))
                throw new FileNotFoundException($"Local file not found: {filename}", full);

            return File.OpenRead(full);
        }

        public bool FileExists(uint filedataid)
        {
            EnsureListfileChecked();

            if (TryResolveFilename(filedataid, out var name))
            {
                var full = ResolveFullPath(name);
                return File.Exists(full);
            }

            return false;
        }

        public Stream OpenFile(uint filedataid)
        {
            EnsureListfileChecked();

            if (TryResolveFilename(filedataid, out var name))
            {
                var full = ResolveFullPath(name);
                if (!File.Exists(full))
                    throw new FileNotFoundException($"Local file not found for FDID {filedataid}: {name}", full);

                return File.OpenRead(full);
            }

            throw new FileNotFoundException($"Unable to resolve filename for FDID {filedataid}");
        }

        public Stream OpenFile(byte[] cKey)
        {
            throw new NotImplementedException();
        }

        public bool FileExists(byte[] cKey)
        {
            throw new NotImplementedException();
        }

        // Helpers
        private void EnsureListfileChecked()
        {
            if (_listfileChecked)
                return;

            lock (_sync)
            {
                if (_listfileChecked)
                    return;

                try
                {
                    // Only load if not already populated to avoid duplicate key exceptions
                    if (Listfile.FDIDToFilename.Count == 0 || Listfile.FilenameToFDID.Count == 0)
                    {
                        Listfile.Load();
                    }
                }
                catch
                {
                    // Swallow errors; provider will still work for direct filename access
                    // and synthetic-ID flows.
                }
                finally
                {
                    _listfileChecked = true;
                }
            }
        }

        private string ResolveFullPath(string filename)
        {
            if (string.IsNullOrWhiteSpace(filename))
                throw new ArgumentException("Filename cannot be null or empty", nameof(filename));

            // If already absolute, return as-is
            if (Path.IsPathRooted(filename))
                return Path.GetFullPath(filename);

            // Convert repo-style separators to OS-specific
            var fixedSeparators = filename.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
            return Path.GetFullPath(Path.Combine(_baseDir, fixedSeparators));
        }

        private string NormalizeToRepoStyle(string path)
        {
            // Produce lower-case, forward-slash path, no leading slash.
            var p = path;
            if (Path.IsPathRooted(p))
            {
                try
                {
                    var abs = Path.GetFullPath(p);
                    if (abs.StartsWith(_baseDir, StringComparison.OrdinalIgnoreCase))
                    {
                        p = abs.Substring(_baseDir.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                    }
                    else
                    {
                        // Not under base dir, keep as-is but normalize slashes
                        p = abs;
                    }
                }
                catch
                {
                    // Fallback to original
                }
            }

            p = p.Replace('\\', '/');
            if (p.StartsWith("/"))
                p = p.TrimStart('/');

            return p.ToLowerInvariant();
        }

        private string? TryGetRelativeNormalized(string path)
        {
            try
            {
                var abs = Path.GetFullPath(path);
                if (abs.StartsWith(_baseDir, StringComparison.OrdinalIgnoreCase))
                {
                    var rel = abs.Substring(_baseDir.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                    return NormalizeToRepoStyle(rel);
                }
            }
            catch
            {
                // ignore
            }
            return null;
        }

        private uint GetOrCreateSyntheticId(string normalizedPath)
        {
            lock (_sync)
            {
                if (_pathToSyntheticId.TryGetValue(normalizedPath, out var existing))
                    return existing;

                // FNV-1a 32-bit
                uint hash = 2166136261;
                foreach (var b in Encoding.UTF8.GetBytes(normalizedPath))
                {
                    hash ^= b;
                    hash *= 16777619;
                }

                uint id = hash;
                // Avoid collisions with existing synthetic mappings
                while (_syntheticIdToPath.ContainsKey(id) || Listfile.TryGetFilename(id, out _))
                {
                    unchecked { id += 1; }
                }

                _pathToSyntheticId[normalizedPath] = id;
                _syntheticIdToPath[id] = normalizedPath;
                return id;
            }
        }

        private bool TryResolveFilename(uint filedataid, out string filename)
        {
            // Prefer listfile mapping when available
            if (Listfile.TryGetFilename(filedataid, out filename))
                return true;

            // Fallback to synthetic mapping
            if (_syntheticIdToPath.TryGetValue(filedataid, out var normPath))
            {
                filename = normPath;
                return true;
            }

            filename = string.Empty;
            return false;
        }
    }
}
