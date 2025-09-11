using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using DBCD.Providers;

namespace DBCTool.Mpq
{
    internal sealed class MpqDBCProvider : IDBCProvider, IDisposable
    {
        private readonly List<(string Path, string? Prefix)> _archives;
        public IReadOnlyList<string> Archives => _archives.Select(a => a.Path).ToList();

        public MpqDBCProvider(IEnumerable<string> archives)
        {
            _archives = archives
                .Select(p => Path.GetFullPath(p))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Select(p => (Path: p, Prefix: DeriveLocalePrefix(p)))
                .ToList();
            if (_archives.Count == 0)
                throw new ArgumentException("At least one MPQ archive must be provided");
        }

        public MpqDBCProvider(IEnumerable<(string Path, string? Prefix)> archives)
        {
            _archives = archives
                .Select(a => (Path: Path.GetFullPath(a.Path), a.Prefix))
                .Distinct()
                .ToList();
            if (_archives.Count == 0)
                throw new ArgumentException("At least one MPQ archive must be provided");
        }

        public static MpqDBCProvider FromRoot(string mpqRoot)
        {
            if (string.IsNullOrWhiteSpace(mpqRoot))
                throw new ArgumentException("mpqRoot is required");

            var root = Path.GetFullPath(mpqRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

            // Determine scan root:
            // 1) If a Data subfolder exists, use that (WoW install root provided)
            // 2) Else if the folder itself contains any MPQs (e.g., Data or locale folder), use it directly
            // 3) Else error
            string scanRoot;
            var dataCandidate = Path.Combine(root, "Data");
            if (Directory.Exists(dataCandidate))
            {
                scanRoot = dataCandidate;
            }
            else if (Directory.Exists(root) && Directory.EnumerateFiles(root, "*.MPQ", SearchOption.TopDirectoryOnly).Any())
            {
                scanRoot = root;
            }
            else
            {
                throw new DirectoryNotFoundException($"MPQ root missing Data folder and contains no MPQs: {root}");
            }

            // Recursively enumerate all MPQ files and compute locale prefixes
            var archives = Directory.EnumerateFiles(scanRoot, "*.MPQ", SearchOption.AllDirectories)
                                     .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                                     .Select(p => (Path: p, Prefix: DeriveLocalePrefixRelative(scanRoot, p)))
                                     .ToList();

            return new MpqDBCProvider(archives);
        }

        public Stream StreamForTableName(string tableName, string build)
        {
            // Try both path separator variants for robustness
            var candidatePaths = new[]
            {
                $"DBFilesClient\\{tableName}.dbc",
                $"DBFilesClient/{tableName}.dbc"
            };

            // Classify archives
            static string Name(string path) => Path.GetFileName(path) ?? string.Empty;
            static bool IsSpeech(string name) => name.Contains("speech", StringComparison.OrdinalIgnoreCase);
            static bool IsBackup(string name) => name.Contains("backup", StringComparison.OrdinalIgnoreCase);
            static bool IsCoreBaseName(string name) => (name.StartsWith("common", StringComparison.OrdinalIgnoreCase)
                                                        || name.StartsWith("expansion", StringComparison.OrdinalIgnoreCase)
                                                        || name.StartsWith("lichking", StringComparison.OrdinalIgnoreCase))
                                                       && !name.StartsWith("patch", StringComparison.OrdinalIgnoreCase);
            static bool IsCorePatchName(string name) => name.Equals("patch.MPQ", StringComparison.OrdinalIgnoreCase)
                                                        || name.StartsWith("patch-", StringComparison.OrdinalIgnoreCase);
            static bool IsLocaleBaseName(string name) => name.StartsWith("base-", StringComparison.OrdinalIgnoreCase)
                                                         || name.Contains("locale-", StringComparison.OrdinalIgnoreCase);
            static bool IsLocalePatchName(string name) => name.StartsWith("patch-", StringComparison.OrdinalIgnoreCase)
                                                           && name.Contains("-", StringComparison.OrdinalIgnoreCase);

            var filtered = _archives.Where(a =>
            {
                var n = Name(a.Path);
                return !IsSpeech(n) && !IsBackup(n);
            })
            .ToList();

            int Category((string Path, string? Prefix) a)
            {
                var n = Name(a.Path);
                var isLocale = !string.IsNullOrEmpty(a.Prefix);
                if (!isLocale && IsCoreBaseName(n)) return 0;    // core bases
                if (!isLocale && IsCorePatchName(n)) return 1;   // core patches
                if (isLocale && IsLocaleBaseName(n)) return 2;   // locale bases
                if (isLocale && IsLocalePatchName(n)) return 3;  // locale patches
                return 4;                                        // others
            }

            var ordered = filtered.OrderBy(Category).ThenBy(a => a.Path, StringComparer.OrdinalIgnoreCase).ToList();
            var baseCandidates = ordered.Where(a => Category(a) == 0).ToList();
            if (baseCandidates.Count == 0)
                baseCandidates = ordered; // fallback if no obvious base

            foreach (var baseArch in baseCandidates)
            {
                if (!StormLib.SFileOpenArchive(baseArch.Path, 0, 0, out var hBase))
                    continue;

                try
                {
                    // Attach patches in deterministic order; pass locale prefix when available
                    foreach (var patch in ordered)
                    {
                        if (string.Equals(patch.Path, baseArch.Path, StringComparison.OrdinalIgnoreCase))
                            continue;
                        StormLib.SFileOpenPatchArchive(hBase, patch.Path, patch.Prefix, 0);
                    }

                    // Try opening with both path variants
                    foreach (var mpqPath in candidatePaths)
                    {
                        if (StormLib.SFileOpenFileEx(hBase, mpqPath, StormLib.SFILE_OPEN_PATCHED_FILE, out var hFile))
                        {
                            try
                            {
                                uint hi;
                                uint sizeLo = StormLib.SFileGetFileSize(hFile, out hi);
                                long size = ((long)hi << 32) | sizeLo;
                                if (size <= 0)
                                    throw new IOException($"MPQ file opened but size invalid: {mpqPath}");

                                byte[] buffer = new byte[size];
                                unsafe
                                {
                                    fixed (byte* p = buffer)
                                    {
                                        if (!StormLib.SFileReadFile(hFile, (IntPtr)p, (uint)size, out var read, IntPtr.Zero) || read != (uint)size)
                                        {
                                            int err = Marshal.GetLastWin32Error();
                                            throw new IOException($"SFileReadFile failed (err={err}) for {mpqPath}");
                                        }
                                    }
                                }

                                return new MemoryStream(buffer, 0, buffer.Length, writable: false, publiclyVisible: true);
                            }
                            finally
                            {
                                StormLib.SFileCloseFile(hFile);
                            }
                        }
                    }
                }
                finally
                {
                    StormLib.SFileCloseArchive(hBase);
                }
            }

            // Fallback: try to open the file directly from individual archives (without patching).
            // This can succeed when the full file resides in a base MPQ (common/expansion/lichking) or locale base.
            var perArchiveOrder = ordered.AsEnumerable().Reverse().ToList(); // try later (patch) archives first
            foreach (var ap in perArchiveOrder)
            {
                if (!StormLib.SFileOpenArchive(ap.Path, 0, 0, out var h))
                    continue;
                try
                {
                    foreach (var mpqPath in candidatePaths)
                    {
                        if (StormLib.SFileOpenFileEx(h, mpqPath, StormLib.SFILE_OPEN_FROM_MPQ, out var fh))
                        {
                            try
                            {
                                uint hi;
                                uint sizeLo = StormLib.SFileGetFileSize(fh, out hi);
                                long size = ((long)hi << 32) | sizeLo;
                                if (size <= 0)
                                    continue;

                                byte[] buffer = new byte[size];
                                unsafe
                                {
                                    fixed (byte* p = buffer)
                                    {
                                        if (!StormLib.SFileReadFile(fh, (IntPtr)p, (uint)size, out var read, IntPtr.Zero) || read != (uint)size)
                                        {
                                            continue;
                                        }
                                    }
                                }

                                return new MemoryStream(buffer, 0, buffer.Length, writable: false, publiclyVisible: true);
                            }
                            finally
                            {
                                StormLib.SFileCloseFile(fh);
                            }
                        }
                    }
                }
                finally
                {
                    StormLib.SFileCloseArchive(h);
                }
            }

            throw new FileNotFoundException($"Could not open DBFilesClient\\{tableName}.dbc from any provided MPQ archives. Searched {Archives.Count} archives.");
        }

        private static string? DeriveLocalePrefix(string path)
        {
            try
            {
                var parent = Path.GetFileName(Path.GetDirectoryName(path)?.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) ?? string.Empty);
                if (!string.IsNullOrEmpty(parent) && parent.Length == 4 && parent.All(char.IsLetter))
                    return parent; // e.g., enUS
            }
            catch { }
            return null;
        }

        private static string? DeriveLocalePrefixRelative(string scanRoot, string path)
        {
            try
            {
                var rel = Path.GetRelativePath(scanRoot, path);
                var firstSep = rel.IndexOfAny(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar });
                if (firstSep > 0)
                {
                    var first = rel.Substring(0, firstSep);
                    if (first.Length == 4 && first.All(char.IsLetter))
                        return first;
                }
            }
            catch { }
            return null;
        }

        public void Dispose() { }
    }
}
