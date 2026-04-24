using System.Reflection;
using System.Runtime.InteropServices;

namespace WowViewer.Core.IO.Files;

internal static class StormLibPatchArchiveReader
{
    private const string StormLib = "StormLib.dll";
    private const uint MpqOpenReadOnly = 0x00000100;

    static StormLibPatchArchiveReader()
    {
        NativeLibrary.SetDllImportResolver(typeof(StormLibPatchArchiveReader).Assembly, ResolveStormLib);
    }

    public static bool TryFileExists(IReadOnlyList<string> archivePaths, string virtualPath)
    {
        return TryWithArchiveHandles(
            archivePaths,
            handle => SFileHasFile(handle, virtualPath),
            out bool found)
            ? found
            : false;
    }

    public static byte[]? TryReadFile(IReadOnlyList<string> archivePaths, string virtualPath)
    {
        return TryWithArchiveHandles(
            archivePaths,
            handle => TryReadFileFromHandle(handle, virtualPath),
            out byte[]? data)
            ? data
            : null;
    }

    private static bool TryWithArchiveHandles<TResult>(
        IReadOnlyList<string> archivePaths,
        Func<IntPtr, TResult?> probe,
        out TResult? result)
    {
        result = default;

        string[] orderedArchives = archivePaths
            .Where(static path => !string.IsNullOrWhiteSpace(path) && File.Exists(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(static path => GetMpqPriority(Path.GetFileName(path)))
            .ThenBy(static path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (orderedArchives.Length == 0)
            return false;

        List<string> baseArchives = orderedArchives
            .Where(static path => !Path.GetFileName(path).StartsWith("patch", StringComparison.OrdinalIgnoreCase))
            .ToList();
        List<string> patchArchives = orderedArchives
            .Where(static path => Path.GetFileName(path).StartsWith("patch", StringComparison.OrdinalIgnoreCase))
            .ToList();

        var openHandles = new List<IntPtr>();

        try
        {
            uint priority = 1;
            if (baseArchives.Count > 0)
            {
                IntPtr baseHandle = OpenArchive(baseArchives[0], priority++);
                if (baseHandle != IntPtr.Zero)
                {
                    openHandles.Add(baseHandle);

                    foreach (string patchArchive in patchArchives)
                    {
                        if (!SFileOpenPatchArchive(baseHandle, patchArchive, null, 0))
                        {
                            IntPtr patchHandle = OpenArchive(patchArchive, priority++);
                            if (patchHandle != IntPtr.Zero)
                                openHandles.Add(patchHandle);
                        }
                    }
                }

                foreach (string baseArchive in baseArchives.Skip(1))
                {
                    IntPtr handle = OpenArchive(baseArchive, priority++);
                    if (handle != IntPtr.Zero)
                        openHandles.Add(handle);
                }
            }
            else
            {
                foreach (string patchArchive in patchArchives)
                {
                    IntPtr handle = OpenArchive(patchArchive, priority++);
                    if (handle != IntPtr.Zero)
                        openHandles.Add(handle);
                }
            }

            foreach (IntPtr handle in openHandles)
            {
                TResult? candidate = probe(handle);
                if (candidate is null)
                    continue;

                if (candidate is bool boolCandidate && !boolCandidate)
                    continue;

                result = candidate;
                return true;
            }

            return false;
        }
        catch (DllNotFoundException)
        {
            MpqDiagnostics.Increment("MpqStormLibUnavailableCount");
            return false;
        }
        catch (BadImageFormatException)
        {
            MpqDiagnostics.Increment("MpqStormLibUnavailableCount");
            return false;
        }
        finally
        {
            foreach (IntPtr handle in openHandles)
            {
                if (handle != IntPtr.Zero)
                    SFileCloseArchive(handle);
            }
        }
    }

    private static byte[]? TryReadFileFromHandle(IntPtr archiveHandle, string virtualPath)
    {
        IntPtr fileHandle = IntPtr.Zero;
        bool opened = SFileOpenFileEx(archiveHandle, virtualPath, 0, ref fileHandle);

        if (opened && fileHandle != IntPtr.Zero)
        {
            try
            {
                uint size = SFileGetFileSize(fileHandle, out uint sizeHigh);
                if (size == uint.MaxValue || sizeHigh != 0 || size == 0)
                    return null;

                byte[] buffer = new byte[size];
                GCHandle pinned = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                try
                {
                    if (!SFileReadFile(fileHandle, pinned.AddrOfPinnedObject(), size, out uint bytesRead, IntPtr.Zero))
                        return null;

                    if (bytesRead != size)
                        Array.Resize(ref buffer, checked((int)bytesRead));

                    MpqDiagnostics.Increment("MpqStormLibFallbackReadHitCount");
                    return buffer;
                }
                finally
                {
                    pinned.Free();
                }
            }
            finally
            {
                SFileCloseFile(fileHandle);
            }
        }

        if (!opened && !SFileHasFile(archiveHandle, virtualPath))
            return null;

        string tempPath = Path.Combine(Path.GetTempPath(), $"wowviewer_stormlib_{Guid.NewGuid():N}.bin");
        try
        {
            if (!SFileExtractFile(archiveHandle, virtualPath, tempPath, 0) || !File.Exists(tempPath))
                return null;

            MpqDiagnostics.Increment("MpqStormLibFallbackReadHitCount");
            return File.ReadAllBytes(tempPath);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    private static IntPtr OpenArchive(string path, uint priority)
    {
        return SFileOpenArchive(path, priority, MpqOpenReadOnly, out IntPtr handle) ? handle : IntPtr.Zero;
    }

    private static IntPtr ResolveStormLib(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, StormLib, StringComparison.OrdinalIgnoreCase))
            return IntPtr.Zero;

        foreach (string candidate in EnumerateStormLibCandidates())
        {
            if (!File.Exists(candidate))
                continue;

            if (NativeLibrary.TryLoad(candidate, out IntPtr handle))
                return handle;
        }

        return IntPtr.Zero;
    }

    private static IEnumerable<string> EnumerateStormLibCandidates()
    {
        yield return Path.Combine(AppContext.BaseDirectory, StormLib);

        DirectoryInfo? current = new(AppContext.BaseDirectory);
        for (int depth = 0; depth < 8 && current is not null; depth++, current = current.Parent)
        {
            if (!File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                continue;

            yield return Path.Combine(current.FullName, "libs", "Marlamin", "WoWTools.Minimaps", "StormLibWrapper", StormLib);
            yield return Path.Combine(current.FullName, "gillijimproject_refactor", "lib", "WoWTools.Minimaps", "StormLibWrapper", StormLib);
        }
    }

    private static int GetMpqPriority(string filename)
    {
        string lower = filename.ToLowerInvariant();
        if (lower.StartsWith("patch", StringComparison.OrdinalIgnoreCase))
        {
            string nameWithoutExtension = lower.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase)
                ? lower[..^4]
                : lower;
            string[] parts = nameWithoutExtension.Split('-', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 1 && parts[0] == "patch")
            {
                bool isLocale = false;
                int suffixIndex = 1;
                if (parts.Length >= 2 && parts[1].Length == 4 && parts[1].All(char.IsLetter))
                {
                    isLocale = true;
                    suffixIndex = 2;
                }

                string? suffix = parts.Length > suffixIndex ? parts[suffixIndex] : null;
                int suffixRank = 0;
                if (!string.IsNullOrEmpty(suffix))
                {
                    if (int.TryParse(suffix, out int number))
                    {
                        suffixRank = Math.Clamp(number, 0, 499);
                    }
                    else if (suffix.Length == 1 && suffix[0] >= 'a' && suffix[0] <= 'z')
                    {
                        suffixRank = 500 + (suffix[0] - 'a' + 1);
                    }
                    else
                    {
                        suffixRank = 900;
                    }
                }

                return (isLocale ? 2000 : 1000) + suffixRank;
            }

            return 2900;
        }

        if (lower.Contains("enus", StringComparison.Ordinal) ||
            lower.Contains("engb", StringComparison.Ordinal) ||
            lower.Contains("dede", StringComparison.Ordinal) ||
            lower.Contains("locale", StringComparison.Ordinal))
        {
            return 500;
        }

        if (lower.StartsWith("expansion", StringComparison.Ordinal) || lower.StartsWith("lichking", StringComparison.Ordinal))
            return 300;

        if (lower == "common.mpq" || lower == "common-2.mpq")
            return 100;

        return 200;
    }

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true, CharSet = CharSet.Auto)]
    private static extern bool SFileOpenArchive(
        [MarshalAs(UnmanagedType.LPTStr)] string szMpqName,
        uint dwPriority,
        uint dwFlags,
        out IntPtr phMpq);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileCloseArchive(IntPtr hMpq);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileOpenFileEx(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPStr)] string szFileName,
        uint dwSearchScope,
        ref IntPtr phFile);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern uint SFileGetFileSize(IntPtr hFile, out uint fileSizeHigh);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileReadFile(
        IntPtr hFile,
        IntPtr lpBuffer,
        uint dwToRead,
        out uint pdwRead,
        IntPtr lpOverlapped);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileCloseFile(IntPtr hFile);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileExtractFile(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPStr)] string szToExtract,
        [MarshalAs(UnmanagedType.LPStr)] string szExtracted,
        uint dwSearchScope);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true, CharSet = CharSet.Auto)]
    private static extern bool SFileOpenPatchArchive(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPTStr)] string szPatchMpqName,
        [MarshalAs(UnmanagedType.LPStr)] string? szPatchPathPrefix,
        uint dwFlags);

    [DllImport(StormLib, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileHasFile(IntPtr hMpq, [MarshalAs(UnmanagedType.LPStr)] string szFileName);
}