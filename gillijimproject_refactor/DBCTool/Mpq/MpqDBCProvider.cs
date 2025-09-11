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
        private readonly List<string> _archives;

        public MpqDBCProvider(IEnumerable<string> archives)
        {
            _archives = archives.Select(Path.GetFullPath).ToList();
            if (_archives.Count == 0)
                throw new ArgumentException("At least one MPQ archive must be provided");
        }

        public static MpqDBCProvider FromRoot(string mpqRoot)
        {
            if (string.IsNullOrWhiteSpace(mpqRoot))
                throw new ArgumentException("mpqRoot is required");

            // Accept either the WoW install root (contains Data) or the Data folder itself
            string dataDir;
            if (string.Equals(Path.GetFileName(Path.GetFullPath(mpqRoot).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)), "Data", StringComparison.OrdinalIgnoreCase))
            {
                dataDir = mpqRoot; // mpqRoot already points to Data
            }
            else
            {
                dataDir = Path.Combine(mpqRoot, "Data");
            }
            if (!Directory.Exists(dataDir))
                throw new DirectoryNotFoundException($"MPQ root missing Data folder: {dataDir}");

            // Gather archives from Data and any locale subfolders (e.g., enUS, enGB, deDE)
            var archives = new List<string>();
            archives.AddRange(Directory.EnumerateFiles(dataDir, "*.MPQ", SearchOption.TopDirectoryOnly));

            foreach (var localeDir in Directory.EnumerateDirectories(dataDir))
            {
                var name = Path.GetFileName(localeDir) ?? string.Empty;
                // Locale folders like enUS, enGB, deDE ... (length==4) or specific known names
                if (name.Length == 4 || name.Equals("enUS", StringComparison.OrdinalIgnoreCase) || name.Equals("enGB", StringComparison.OrdinalIgnoreCase))
                {
                    archives.AddRange(Directory.EnumerateFiles(localeDir, "*.MPQ", SearchOption.TopDirectoryOnly));
                }
            }

            // Sort lexicographically; we'll also pick base candidates by name patterns below
            archives.Sort(StringComparer.OrdinalIgnoreCase);
            return new MpqDBCProvider(archives);
        }

        public Stream StreamForTableName(string tableName, string build)
        {
            // MPQ logical path typically uses backslashes
            string mpqPath = $"DBFilesClient\\{tableName}.dbc";

            // Choose base candidates: prioritize locale bases, then core bases
            var fileNames = _archives.Select(Path.GetFileName).ToList();
            bool IsLocaleBase(string f) => f != null && (f.Contains("locale-", StringComparison.OrdinalIgnoreCase) || f.StartsWith("base-", StringComparison.OrdinalIgnoreCase));
            bool IsCoreBase(string f) => f != null && (f.StartsWith("common", StringComparison.OrdinalIgnoreCase) || f.StartsWith("expansion", StringComparison.OrdinalIgnoreCase) || f.StartsWith("lichking", StringComparison.OrdinalIgnoreCase));
            var baseCandidates = _archives
                .OrderBy(f =>
                {
                    var name = Path.GetFileName(f) ?? string.Empty;
                    return IsLocaleBase(name) ? 0 : (IsCoreBase(name) ? 1 : 2);
                })
                .ThenBy(f => f, StringComparer.OrdinalIgnoreCase)
                .ToList();

            // Try each archive as base, attach the rest as patches in order
            foreach (var basePath in baseCandidates)
            {
                if (!StormLib.SFileOpenArchive(basePath, 0, 0, out var hBase))
                {
                    continue; // try next base
                }

                try
                {
                    // Attach all other archives as patches — StormLib will use relevant ones
                    foreach (var patchPath in _archives)
                    {
                        if (string.Equals(patchPath, basePath, StringComparison.OrdinalIgnoreCase))
                            continue;
                        // Attach; ignore failures (non-patch or incompatible)
                        StormLib.SFileOpenPatchArchive(hBase, patchPath, null!, 0);
                    }

                    // Try open the target file as patched
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
                finally
                {
                    StormLib.SFileCloseArchive(hBase);
                }
            }

            throw new FileNotFoundException($"Could not open {mpqPath} from any provided MPQ archives.");
        }

        public void Dispose() { }
    }
}
