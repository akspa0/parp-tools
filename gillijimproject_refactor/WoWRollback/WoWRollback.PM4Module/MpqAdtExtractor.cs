using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace WoWRollback.PM4Module;

/// <summary>
/// Extracts ADT files from MPQ archives using StormLib.
/// Used to harvest texture data from older monolithic ADTs (e.g., 2.4.3 expansion.MPQ).
/// </summary>
public sealed class MpqAdtExtractor : IDisposable
{
    private IntPtr _hArchive;
    private readonly string _archivePath;
    private bool _disposed;

    #region StormLib P/Invoke
    
    private const string DllName = "StormLib.dll";

    [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SFileOpenArchive(string szArchiveName, uint dwPriority, uint dwFlags, out IntPtr phArchive);

    [DllImport(DllName, SetLastError = true)]
    private static extern bool SFileCloseArchive(IntPtr hArchive);

    [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SFileOpenFileEx(IntPtr hMpq, string szFileName, uint dwSearchScope, out IntPtr phFile);

    [DllImport(DllName, SetLastError = true)]
    private static extern uint SFileGetFileSize(IntPtr hFile, out uint pdwFileSizeHigh);

    [DllImport(DllName, SetLastError = true)]
    private static extern bool SFileReadFile(IntPtr hFile, IntPtr lpBuffer, uint dwToRead, out uint pdwRead, IntPtr lpOverlapped);

    [DllImport(DllName, SetLastError = true)]
    private static extern bool SFileCloseFile(IntPtr hFile);

    [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SFileHasFile(IntPtr hMpq, string szFileName);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct SFILE_FIND_DATA
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 1024)]
        public string cFileName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 1024)]
        public string szPlainName;
        public uint dwHashIndex;
        public uint dwBlockIndex;
        public uint dwFileSize;
        public uint dwCompressedSize;
        public uint dwFlags;
        public uint dwFileTimeLow;
        public uint dwFileTimeHigh;
        public uint lcLocale;
        public uint wPlatform;
    }

    [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr SFileFindFirstFile(IntPtr hMpq, string szMask, out SFILE_FIND_DATA lpFindFile, string? szListFile);

    [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SFileFindNextFile(IntPtr hFind, out SFILE_FIND_DATA lpFindFile);

    [DllImport(DllName, SetLastError = true)]
    private static extern bool SFileFindClose(IntPtr hFind);

    private const uint SFILE_OPEN_FROM_MPQ = 0x00000000;
    
    #endregion

    public MpqAdtExtractor(string archivePath)
    {
        _archivePath = Path.GetFullPath(archivePath);
        
        if (!File.Exists(_archivePath))
            throw new FileNotFoundException($"MPQ archive not found: {_archivePath}");

        if (!SFileOpenArchive(_archivePath, 0, 0, out _hArchive))
        {
            int err = Marshal.GetLastWin32Error();
            throw new IOException($"Failed to open MPQ archive: {_archivePath} (error={err})");
        }
    }

    /// <summary>
    /// List all files in the archive matching a pattern.
    /// </summary>
    public List<string> ListFiles(string pattern = "*")
    {
        var results = new List<string>();
        
        var hFind = SFileFindFirstFile(_hArchive, pattern, out var findData, null);
        if (hFind == IntPtr.Zero)
            return results;

        try
        {
            do
            {
                if (!string.IsNullOrEmpty(findData.cFileName))
                    results.Add(findData.cFileName);
            }
            while (SFileFindNextFile(hFind, out findData));
        }
        finally
        {
            SFileFindClose(hFind);
        }

        return results;
    }

    /// <summary>
    /// List ADT files for a specific map.
    /// </summary>
    public List<string> ListMapAdts(string mapName)
    {
        var pattern = $"World\\Maps\\{mapName}\\*.adt";
        var all = ListFiles(pattern);
        
        // Filter to root ADTs only (not _obj0, _tex0 which shouldn't exist in 2.x anyway)
        var results = new List<string>();
        foreach (var f in all)
        {
            var name = Path.GetFileNameWithoutExtension(f);
            if (!name.Contains("_obj") && !name.Contains("_tex"))
                results.Add(f);
        }
        return results;
    }

    /// <summary>
    /// Check if a file exists in the archive.
    /// </summary>
    public bool HasFile(string mpqPath)
    {
        return SFileHasFile(_hArchive, mpqPath);
    }

    /// <summary>
    /// Read a file from the archive.
    /// </summary>
    public byte[]? ReadFile(string mpqPath)
    {
        if (!SFileOpenFileEx(_hArchive, mpqPath, SFILE_OPEN_FROM_MPQ, out var hFile))
            return null;

        try
        {
            uint sizeLo = SFileGetFileSize(hFile, out uint sizeHi);
            if (sizeLo == 0xFFFFFFFF)
                return null;

            long size = ((long)sizeHi << 32) | sizeLo;
            if (size <= 0 || size > 100_000_000) // 100MB sanity limit
                return null;

            var buffer = new byte[size];
            unsafe
            {
                fixed (byte* p = buffer)
                {
                    int offset = 0;
                    const int CHUNK = 128 * 1024;
                    while (offset < size)
                    {
                        uint toRead = (uint)Math.Min(CHUNK, size - offset);
                        if (!SFileReadFile(hFile, (IntPtr)(p + offset), toRead, out var read, IntPtr.Zero))
                            break;
                        if (read == 0)
                            break;
                        offset += (int)read;
                    }
                    
                    if (offset != size)
                        Array.Resize(ref buffer, offset);
                }
            }

            return buffer;
        }
        finally
        {
            SFileCloseFile(hFile);
        }
    }

    /// <summary>
    /// Extract all ADTs for a map to a directory.
    /// </summary>
    public int ExtractMapAdts(string mapName, string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        
        var adts = ListMapAdts(mapName);
        Console.WriteLine($"[MPQ] Found {adts.Count} ADTs for map '{mapName}'");
        
        int extracted = 0;
        foreach (var mpqPath in adts)
        {
            var data = ReadFile(mpqPath);
            if (data == null || data.Length == 0)
            {
                Console.WriteLine($"  [SKIP] {mpqPath} - failed to read");
                continue;
            }

            var fileName = Path.GetFileName(mpqPath);
            var outputPath = Path.Combine(outputDir, fileName);
            File.WriteAllBytes(outputPath, data);
            Console.WriteLine($"  [OK] {fileName} ({data.Length:N0} bytes)");
            extracted++;
        }

        return extracted;
    }

    /// <summary>
    /// Extract a single ADT file.
    /// </summary>
    public bool ExtractAdt(string mapName, int tileX, int tileY, string outputPath)
    {
        var mpqPath = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
        
        var data = ReadFile(mpqPath);
        if (data == null || data.Length == 0)
            return false;

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, data);
        return true;
    }

    public void Dispose()
    {
        if (!_disposed && _hArchive != IntPtr.Zero)
        {
            SFileCloseArchive(_hArchive);
            _hArchive = IntPtr.Zero;
            _disposed = true;
        }
    }
}
