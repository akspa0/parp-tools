using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// Service to manage global MPQ archives (Textures.mpq, Terrain.mpq, etc.)
/// and provide unified file access.
/// </summary>
public class MpqArchiveService : IDisposable
{
    private readonly List<IntPtr> _archives = new();
    private bool _disposed;

    #region StormLib P/Invoke (Copied from AlphaMpqAssetReader)
    
    private const string STORMLIB = "StormLib.dll";
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true, CharSet = CharSet.Auto)]
    private static extern bool SFileOpenArchive(
        [MarshalAs(UnmanagedType.LPTStr)] string szMpqName,
        uint dwPriority,
        uint dwFlags,
        out IntPtr phMpq);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileCloseArchive(IntPtr hMpq);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileOpenFileEx(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPStr)] string szFileName,
        uint dwSearchScope,
        ref IntPtr phFile);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern uint SFileGetFileSize(IntPtr hFile, out uint fileSizeHigh);
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileReadFile(
        IntPtr hFile,
        IntPtr lpBuffer,
        uint dwToRead,
        out uint pdwRead,
        IntPtr lpOverlapped);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileCloseFile(IntPtr hFile);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileExtractFile(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPStr)] string szToExtract,
        [MarshalAs(UnmanagedType.LPStr)] string szExtracted,
        uint dwSearchScope);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileAddListFile(IntPtr hMpq, [MarshalAs(UnmanagedType.LPStr)] string szListFile);
    
    // Archive open flags
    private const uint SFILE_OPEN_HARD_DISK_FILE = 2;
    private const uint MPQ_OPEN_READ_ONLY = 0x00000100;
    
    #endregion

    public void LoadArchives(IEnumerable<string> searchPaths)
    {
        var pathsToSearch = new HashSet<string>();
        
        foreach (var path in searchPaths)
        {
            if (Directory.Exists(path))
                pathsToSearch.Add(path);
            
            // Also check Data subfolder (WoW client structure)
            var dataSubdir = Path.Combine(path, "Data");
            if (Directory.Exists(dataSubdir))
            {
                pathsToSearch.Add(dataSubdir);
                
                // Check for locale folders (3.3.5+: Data/enUS, Data/deDE, etc.)
                foreach (var localeDir in Directory.GetDirectories(dataSubdir))
                {
                    var localeName = Path.GetFileName(localeDir);
                    // Common WoW locale codes
                    if (localeName.Length == 4 && char.IsLetter(localeName[0]) && char.IsLetter(localeName[1]) &&
                        char.IsUpper(localeName[2]) && char.IsUpper(localeName[3]))
                    {
                        pathsToSearch.Add(localeDir);
                    }
                }
            }
        }
        
        Console.WriteLine($"[MpqService] Searching for MPQ archives in {pathsToSearch.Count} paths:");
        foreach (var p in pathsToSearch)
            Console.WriteLine($"  - {p}");
        
        // Collect all MPQ files
        var allMpqFiles = new List<string>();
        foreach (var path in pathsToSearch)
        {
            var mpqFiles = Directory.GetFiles(path, "*.mpq", SearchOption.TopDirectoryOnly);
            allMpqFiles.AddRange(mpqFiles);
            
            var mpqFilesUpper = Directory.GetFiles(path, "*.MPQ", SearchOption.TopDirectoryOnly);
            allMpqFiles.AddRange(mpqFilesUpper);
        }
        
        // Remove duplicates
        allMpqFiles = allMpqFiles.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        
        // Sort by priority: base mpqs first (low priority), patch mpqs last (high priority)
        // StormLib searches from highest priority -> lowest, so we load patches LAST
        // Order: common, expansion, lichking, locale, then patches (patch, patch-2, patch-3, etc.)
        allMpqFiles = allMpqFiles
            .OrderBy(f => GetMpqPriority(Path.GetFileName(f)))
            .ToList();
        
        Console.WriteLine($"[MpqService] Loading {allMpqFiles.Count} archives in priority order:");
        uint priority = 1;
        foreach (var mpq in allMpqFiles)
        {
            OpenArchive(mpq, priority++);
        }
        Console.WriteLine($"Initialized MpqArchiveService with {_archives.Count} archives.");
    }
    
    private static int GetMpqPriority(string filename)
    {
        var lower = filename.ToLowerInvariant();
        
        // Patches get highest priority (loaded last = searched first)
        if (lower.StartsWith("patch"))
        {
            // patch.mpq = 1000, patch-2.mpq = 1002, patch-3.mpq = 1003, etc.
            if (lower == "patch.mpq") return 1000;
            if (lower.StartsWith("patch-"))
            {
                var numPart = lower.Replace("patch-", "").Replace(".mpq", "");
                if (int.TryParse(numPart, out int num))
                    return 1000 + num;
            }
            return 1099; // Other patch files
        }
        
        // Locale-specific (enUS, etc.) - medium-high priority
        if (lower.Contains("enus") || lower.Contains("engb") || lower.Contains("dede") || lower.Contains("locale"))
            return 500;
        
        // Expansion packs - medium priority
        if (lower.StartsWith("expansion") || lower.StartsWith("lichking"))
            return 300;
        
        // Base game files - lowest priority
        if (lower == "common.mpq" || lower == "common-2.mpq")
            return 100;
        
        return 200; // Everything else
    }
    
    public void AddListFile(string listFilePath)
    {
        if (!File.Exists(listFilePath)) return;
        int count = 0;
        foreach (var hMpq in _archives)
        {
            if (SFileAddListFile(hMpq, listFilePath)) count++;
        }
        Console.WriteLine($"[MpqService] Added listfile to {count}/{_archives.Count} archives.");
    }

    private void OpenArchive(string path, uint priority = 0)
    {
        if (SFileOpenArchive(path, priority, MPQ_OPEN_READ_ONLY, out var hMpq))
        {
            _archives.Add(hMpq);
            Console.WriteLine($"  [{priority:D3}] {Path.GetFileName(path)}");
        }
    }
    public bool FileExists(string virtualPath)
    {
        foreach (var hMpq in _archives)
        {
            IntPtr hFile = IntPtr.Zero;
            if (SFileOpenFileEx(hMpq, virtualPath, 0, ref hFile))
            {
                SFileCloseFile(hFile);
                return true;
            }
        }
        return false;
    }

    public byte[]? ReadFile(string virtualPath)
    {
        foreach (var hMpq in _archives)
        {
            // Fallback: SFileOpenFileEx keeps returning 0 handle on some systems/versions of StormLib
            // So we try SFileExtractFile as a robust fallback.
            
            IntPtr hFile = IntPtr.Zero;
            bool opened = SFileOpenFileEx(hMpq, virtualPath, 0, ref hFile);
            
            if (opened)
            {
               Console.WriteLine($"[MpqService] DEBUG: Opened '{virtualPath}' in {hMpq}. Handle: {hFile}");
               if (hFile != IntPtr.Zero)
               {
               try
                {
                    uint sizeHigh = 0;
                    var size = SFileGetFileSize(hFile, out sizeHigh);
                    if (size > 0 && size != uint.MaxValue) 
                    {
                        Console.WriteLine($"[MpqService] Reading '{virtualPath}' from dictionary/archive {hMpq} (Size: {size})");
                        var buffer = new byte[size];
                        var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                        try
                        {
                            if (SFileReadFile(hFile, handle.AddrOfPinnedObject(), size, out var bytesRead, IntPtr.Zero))
                            {
                                Console.WriteLine($"[MpqService] SFileReadFile success. Read: {bytesRead} / {size}");
                                if (bytesRead != size) Array.Resize(ref buffer, (int)bytesRead);
                                return buffer;
                            }
                        }
                        finally
                        {
                            handle.Free();
                        }
                    }
                    else
                    {
                         Console.WriteLine($"[MpqService] File '{virtualPath}' invalid size: {size}. hFile: {hFile}");
                    }
                }
                finally
                {
                    SFileCloseFile(hFile);
                }
            }


            }
        }
        return null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        foreach (var hMpq in _archives)
        {
            SFileCloseArchive(hMpq);
        }
        _archives.Clear();
        _disposed = true;
    }
}
