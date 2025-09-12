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
        public static bool Verbose { get; set; } = false;
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

        // Create a provider from explicit archive paths, filtering to a specific locale directory (e.g., enUS) if provided
        public static MpqDBCProvider FromArchives(IEnumerable<string> archives, string? localeOnly)
        {
            IEnumerable<string> filtered = archives;
            if (!string.IsNullOrWhiteSpace(localeOnly))
            {
                filtered = archives.Where(p => (IsUnderLocaleDir(p, localeOnly!) || IsCoreArchive(p)))
                                   .Where(p => !IsSpeechOrBackup(p));
            }
            var tuples = filtered.Select(p => (Path: Path.GetFullPath(p), Prefix: DeriveLocalePrefix(p)));
            return new MpqDBCProvider(tuples);
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

        // Overload that restricts enumeration to a specific locale folder (e.g., Data\enUS)
        public static MpqDBCProvider FromRoot(string mpqRoot, string? localeOnly)
        {
            if (string.IsNullOrWhiteSpace(mpqRoot))
                throw new ArgumentException("mpqRoot is required");

            var root = Path.GetFullPath(mpqRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

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

            var all = Directory.EnumerateFiles(scanRoot, "*.MPQ", SearchOption.AllDirectories)
                               .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                               .ToList();
            if (!string.IsNullOrWhiteSpace(localeOnly))
            {
                all = all.Where(p => (IsUnderLocaleDir(p, localeOnly!) || IsCoreArchive(p)))
                         .Where(p => !IsSpeechOrBackup(p))
                         .ToList();
            }

            var tuples = all.Select(p => (Path: p, Prefix: DeriveLocalePrefixRelative(scanRoot, p))).ToList();
            return new MpqDBCProvider(tuples);
        }

        public Stream StreamForTableName(string tableName, string build)
        {
            // Try both path separator variants for robustness
            var candidatePaths = new[]
            {
                $"DBFilesClient\\{tableName}.dbc",
                $"DBFilesClient/{tableName}.dbc"
            };

            // PASS 0: Fast direct-open over all provided archives (exact order), since probe shows many archives contain the file
            foreach (var arch in _archives)
            {
                // Skip locale archives in PASS 0; patched fragments exist there and require composite view
                if (!IsCoreArchive(arch.Path))
                    continue;
                if (!StormLib.SFileOpenArchive(arch.Path, 0, 0, out var h))
                {
                    if (Verbose) Console.WriteLine($"[MPQ] open-archive FAILED: {arch.Path}");
                    continue;
                }
                try
                {
                    if (Verbose) Console.WriteLine($"[MPQ] open-archive OK: {arch.Path}");
                    foreach (var mpqPath in candidatePaths)
                    {
                        bool visible = StormLib.SFileHasFile(h, mpqPath);
                        if (Verbose) Console.WriteLine($"[MPQ]   has-file {(visible ? "YES" : "no ")}: {Path.GetFileName(arch.Path)} → {mpqPath}");
                        if (!visible)
                            continue;

                        uint[] scopes =
                        {
                            StormLib.SFILE_OPEN_FROM_MPQ,
                            StormLib.SFILE_OPEN_BASE_FILE,
                            StormLib.SFILE_OPEN_PATCHED_FILE
                        };

                        foreach (var scope in scopes)
                        {
                            if (Verbose) Console.WriteLine($"[MPQ]     try-open scope={scope}: {mpqPath}");
                            if (StormLib.SFileOpenFileEx(h, mpqPath, scope, out var fh))
                            {
                                try
                                {
                                    uint hi;
                                    uint sizeLo = StormLib.SFileGetFileSize(fh, out hi);
                                    byte[] buffer;
                                    if (sizeLo == 0xFFFFFFFFu)
                                    {
                                        int err = Marshal.GetLastWin32Error();
                                        if (err == 6) // Invalid handle
                                        {
                                            if (Verbose) Console.WriteLine($"[MPQ]       size FAILED err={err}, trying unknown-size read");
                                            buffer = ReadAllFromFileUnknown(fh, maxLimit: 64 * 1024 * 1024); // 64MB safety
                                        }
                                        else
                                        {
                                            throw new IOException($"SFileGetFileSize failed (err={err})");
                                        }
                                    }
                                    else
                                    {
                                        long size = ((long)hi << 32) | sizeLo;
                                        if (size <= 0 || size > int.MaxValue)
                                        {
                                            if (Verbose) Console.WriteLine($"[MPQ]       invalid size {size}, trying unknown-size read");
                                            buffer = ReadAllFromFileUnknown(fh, maxLimit: 64 * 1024 * 1024);
                                        }
                                        else
                                        {
                                            buffer = ReadAllFromFile(fh, (int)size);
                                        }
                                    }

                                    if (!IsWdbc(buffer))
                                    {
                                        if (Verbose) Console.WriteLine($"[MPQ]       not WDBC, skipping {mpqPath} ({buffer.Length} bytes)");
                                        continue;
                                    }
                                    if (Verbose) Console.WriteLine($"[MPQ]       OPEN OK {mpqPath} bytes={buffer.Length}");
                                    return new MemoryStream(buffer, 0, buffer.Length, writable: false, publiclyVisible: true);
                                }
                                finally
                                {
                                    StormLib.SFileCloseFile(fh);
                                }
                            }
                            else if (Verbose)
                            {
                                int err = Marshal.GetLastWin32Error();
                                Console.WriteLine($"[MPQ]     open FAILED scope={scope} err={err}: {mpqPath}");
                            }
                        }
                    }
                }
                finally
                {
                    StormLib.SFileCloseArchive(h);
                }
            }

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

            var ordered = _archives
                .Where(a => !IsSpeech(Name(a.Path)) && !IsBackup(Name(a.Path)))
                .ToList();

            var coreBases = ordered.Where(a => IsCoreBaseName(Name(a.Path))).ToList();
            var corePatches = ordered.Where(a => IsCorePatchName(Name(a.Path)))
                                     .OrderBy(a => CorePatchIndex(Name(a.Path)))
                                     .ToList();
            var localeBases = ordered.Where(a => IsLocaleBaseName(Name(a.Path)))
                                     .OrderBy(a => Path.GetFileName(a.Path), StringComparer.OrdinalIgnoreCase)
                                     .ToList();
            var localePatches = ordered.Where(a => IsLocalePatchName(Name(a.Path)))
                                       .OrderBy(a => LocalePatchIndex(Name(a.Path), a.Prefix))
                                       .ToList();

            // Composite open: Core base → core patches → locale bases → locale patches
            foreach (var baseArch in coreBases.Concat(localeBases))
            {
                if (!StormLib.SFileOpenArchive(baseArch.Path, 0, 0, out var hBase))
                    continue;
                try
                {
                    if (Verbose) Console.WriteLine($"[MPQ] composite base: {Path.GetFileName(baseArch.Path)} prefix={(baseArch.Prefix ?? "<none>")}");
                    // Attach in correct order
                    var patchChain = corePatches
                        .Concat(localeBases)
                        .Concat(localePatches)
                        .Where(p => !string.Equals(p.Path, baseArch.Path, StringComparison.OrdinalIgnoreCase))
                        .ToList();

                    foreach (var patch in patchChain)
                    {
                        // Locale patches require a locale prefix like "enUS\\"; core patches pass null
                        string? prefix = patch.Prefix != null ? (patch.Prefix.EndsWith("\\") || patch.Prefix.EndsWith("/") ? patch.Prefix : patch.Prefix + "\\") : null;
                        if (Verbose) Console.WriteLine($"[MPQ]   attach patch: {Path.GetFileName(patch.Path)} prefix={(prefix ?? "<null>")}");
                        StormLib.SFileOpenPatchArchive(hBase, patch.Path, prefix, 0);
                    }

                    // Try opening with both path variants
                    foreach (var mpqPath in candidatePaths)
                    {
                        if (!StormLib.SFileHasFile(hBase, mpqPath))
                            continue;

                        uint[] scopes =
                        {
                            StormLib.SFILE_OPEN_PATCHED_FILE,
                            StormLib.SFILE_OPEN_FROM_MPQ,
                            StormLib.SFILE_OPEN_BASE_FILE
                        };

                        foreach (var scope in scopes)
                        {
                            if (StormLib.SFileOpenFileEx(hBase, mpqPath, scope, out var hFile))
                            {
                                try
                                {
                                    uint hi;
                                    uint sizeLo = StormLib.SFileGetFileSize(hFile, out hi);
                                    byte[] buffer;
                                    if (sizeLo == 0xFFFFFFFFu)
                                    {
                                        buffer = ReadAllFromFileUnknown(hFile, maxLimit: 64 * 1024 * 1024);
                                    }
                                    else
                                    {
                                        long size = ((long)hi << 32) | sizeLo;
                                        if (size <= 0 || size > int.MaxValue)
                                        {
                                            buffer = ReadAllFromFileUnknown(hFile, maxLimit: 64 * 1024 * 1024);
                                        }
                                        else
                                        {
                                            buffer = ReadAllFromFile(hFile, (int)size);
                                        }
                                    }

                                    if (!IsWdbc(buffer))
                                        continue;
                                    return new MemoryStream(buffer, 0, buffer.Length, writable: false, publiclyVisible: true);
                                }
                                finally
                                {
                                    StormLib.SFileCloseFile(hFile);
                                }
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
                        if (!StormLib.SFileHasFile(h, mpqPath))
                            continue;

                        uint[] scopes =
                        {
                            StormLib.SFILE_OPEN_FROM_MPQ,
                            StormLib.SFILE_OPEN_BASE_FILE,
                            StormLib.SFILE_OPEN_PATCHED_FILE
                        };

                        foreach (var scope in scopes)
                        {
                            if (StormLib.SFileOpenFileEx(h, mpqPath, scope, out var fh))
                            {
                                try
                                {
                                    uint hi;
                                    uint sizeLo = StormLib.SFileGetFileSize(fh, out hi);
                                    byte[] buffer;
                                    if (sizeLo == 0xFFFFFFFFu)
                                    {
                                        buffer = ReadAllFromFileUnknown(fh, maxLimit: 64 * 1024 * 1024);
                                    }
                                    else
                                    {
                                        long size = ((long)hi << 32) | sizeLo;
                                        if (size <= 0 || size > int.MaxValue)
                                        {
                                            buffer = ReadAllFromFileUnknown(fh, maxLimit: 64 * 1024 * 1024);
                                        }
                                        else
                                        {
                                            buffer = ReadAllFromFile(fh, (int)size);
                                        }
                                    }

                                    if (!IsWdbc(buffer))
                                        continue;
                                    return new MemoryStream(buffer, 0, buffer.Length, writable: false, publiclyVisible: true);
                                }
                                finally
                                {
                                    StormLib.SFileCloseFile(fh);
                                }
                            }
                        }
                    }
                }
                finally
                {
                    StormLib.SFileCloseArchive(h);
                }
            }

            throw new FileNotFoundException($"Could not open DBFilesClient\\{tableName}.dbc from any provided MPQ archives. Searched {_archives.Count} archives.");
        }

        private static bool IsSpeechOrBackup(string path)
        {
            var name = Path.GetFileName(path);
            if (string.IsNullOrEmpty(name)) return false;
            return name.IndexOf("speech", StringComparison.OrdinalIgnoreCase) >= 0
                || name.IndexOf("backup", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        private static bool IsCoreArchive(string path)
        {
            var name = Path.GetFileName(path) ?? string.Empty;
            if (string.IsNullOrEmpty(name)) return false;
            // Core bases
            if (name.StartsWith("common", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.StartsWith("expansion", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.StartsWith("lichking", StringComparison.OrdinalIgnoreCase)) return true;
            // Core patches
            if (name.Equals("patch.MPQ", StringComparison.OrdinalIgnoreCase)) return true;
            if (name.StartsWith("patch-", StringComparison.OrdinalIgnoreCase)) return true;
            return false;
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

        // Returns true if the file path is under Data\{locale} (case-insensitive, segment-aware)
        private static bool IsUnderLocaleDir(string path, string locale)
        {
            try
            {
                var full = Path.GetFullPath(path);
                var parts = full.Split(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < parts.Length - 1; i++)
                {
                    if (string.Equals(parts[i], "Data", StringComparison.OrdinalIgnoreCase))
                    {
                        if (i + 1 < parts.Length && string.Equals(parts[i + 1], locale, StringComparison.OrdinalIgnoreCase))
                            return true;
                    }
                }
            }
            catch { }
            return false;
        }

        public void Dispose() { }

        // Read all bytes from an open StormLib file handle using chunked reads.
        private static byte[] ReadAllFromFile(IntPtr hFile, int size)
        {
            byte[] buffer = new byte[size];
            int offset = 0;
            const int CHUNK = 128 * 1024;
            unsafe
            {
                fixed (byte* p = buffer)
                {
                    while (offset < size)
                    {
                        uint toRead = (uint)Math.Min(CHUNK, size - offset);
                        if (!StormLib.SFileReadFile(hFile, (IntPtr)(p + offset), toRead, out var read, IntPtr.Zero))
                        {
                            int err = Marshal.GetLastWin32Error();
                            // 0x26 = ERROR_HANDLE_EOF
                            if (err == 0x26)
                                break;
                            throw new IOException($"SFileReadFile failed (err={err})");
                        }
                        if (read == 0)
                            break;
                        offset += (int)read;
                    }
                }
            }

            if (offset == 0)
                throw new IOException("No data read from MPQ file");
            if (offset != size)
                Array.Resize(ref buffer, offset);
            return buffer;
        }

        // Read without known size; stops on EOF. Max limit prevents runaway reads on corrupt handles.
        private static byte[] ReadAllFromFileUnknown(IntPtr hFile, int maxLimit)
        {
            const int CHUNK = 128 * 1024;
            using var ms = new MemoryStream();
            byte[] tmp = new byte[CHUNK];
            unsafe
            {
                fixed (byte* p = tmp)
                {
                    while (ms.Length < maxLimit)
                    {
                        if (!StormLib.SFileReadFile(hFile, (IntPtr)p, (uint)tmp.Length, out var read, IntPtr.Zero))
                        {
                            int err = Marshal.GetLastWin32Error();
                            if (err == 0x26) // EOF
                                break;
                            throw new IOException($"SFileReadFile failed (err={err})");
                        }
                        if (read == 0)
                            break;
                        ms.Write(tmp, 0, (int)read);
                        if (read < tmp.Length)
                            break; // likely EOF next
                    }
                }
            }
            if (ms.Length == 0)
                return Array.Empty<byte>();
            return ms.ToArray();
        }

        private static bool IsWdbc(byte[] data)
        {
            return data.Length >= 4 && (
                   (data[0] == (byte)'W' && data[1] == (byte)'D' && data[2] == (byte)'B' && data[3] == (byte)'C')
                || (data[0] == (byte)'W' && data[1] == (byte)'D' && data[2] == (byte)'B' && data[3] == (byte)'2')
            );
        }

        private static int CorePatchIndex(string name)
        {
            // Ensure patch.MPQ < patch-2.MPQ < patch-3.MPQ
            if (name.Equals("patch.MPQ", StringComparison.OrdinalIgnoreCase)) return 1;
            if (name.Equals("patch-2.MPQ", StringComparison.OrdinalIgnoreCase)) return 2;
            if (name.Equals("patch-3.MPQ", StringComparison.OrdinalIgnoreCase)) return 3;
            return 100; // unknown
        }

        private static int LocalePatchIndex(string name, string? locale)
        {
            // Order like patch-enUS.MPQ < patch-enUS-2.MPQ < patch-enUS-3.MPQ
            // Fallback to filename lexical if not matching pattern
            if (string.IsNullOrEmpty(locale)) return 100;
            string loc = locale;
            if (name.Equals($"patch-{loc}.MPQ", StringComparison.OrdinalIgnoreCase)) return 1;
            if (name.Equals($"patch-{loc}-2.MPQ", StringComparison.OrdinalIgnoreCase)) return 2;
            if (name.Equals($"patch-{loc}-3.MPQ", StringComparison.OrdinalIgnoreCase)) return 3;
            return 100;
        }
    }
}
