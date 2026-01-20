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
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true, CharSet = CharSet.Auto)]
    private static extern bool SFileAddListFile(IntPtr hMpq, [MarshalAs(UnmanagedType.LPStr)] string szListFile);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true, CharSet = CharSet.Auto)]
    private static extern bool SFileOpenPatchArchive(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPTStr)] string szPatchMpqName,
        [MarshalAs(UnmanagedType.LPStr)] string? szPatchPathPrefix,
        uint dwFlags);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileHasFile(IntPtr hMpq, [MarshalAs(UnmanagedType.LPStr)] string szFileName);
    
    // Archive open flags
    private const uint SFILE_OPEN_HARD_DISK_FILE = 2;
    private const uint MPQ_OPEN_READ_ONLY = 0x00000100;
    
    // Track base archive for patch linking
    private IntPtr _baseArchive = IntPtr.Zero;
    
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
        
        // Separate base archives from patches
        var baseArchives = allMpqFiles.Where(f => !Path.GetFileName(f).ToLowerInvariant().StartsWith("patch")).ToList();
        var patchArchives = allMpqFiles.Where(f => Path.GetFileName(f).ToLowerInvariant().StartsWith("patch"))
            .OrderBy(f => GetMpqPriority(Path.GetFileName(f)))
            .ToList();
        
        // 1. Open all BASE archives independently (all will be searched)
        Console.WriteLine($"[MpqService] Opening {baseArchives.Count} base archives:");
        uint priority = 1;
        foreach (var mpq in baseArchives)
        {
            if (OpenArchive(mpq, priority++))
            {
                // Track first opened archive as base for patching
                if (_baseArchive == IntPtr.Zero && _archives.Count > 0)
                    _baseArchive = _archives[0];
            }
        }
        
        // 2. Link patches to the BASE archive (StormLib patch chain)
        if (_baseArchive != IntPtr.Zero && patchArchives.Count > 0)
        {
            Console.WriteLine($"[MpqService] Linking {patchArchives.Count} patch archives to base:");
            foreach (var patchPath in patchArchives)
            {
                var patchName = Path.GetFileName(patchPath);
                if (SFileOpenPatchArchive(_baseArchive, patchPath, null, 0))
                {
                    Console.WriteLine($"  [PATCH] {patchName} -> linked to base");
                }
                else
                {
                    // Fallback: open as independent archive (so we still search it)
                    Console.WriteLine($"  [PATCH] {patchName} -> link failed, opening independently");
                    OpenArchive(patchPath, priority++);
                }
            }
        }
        else if (patchArchives.Count > 0)
        {
            // No base archive found, open patches independently
            Console.WriteLine($"[MpqService] No base archive, opening {patchArchives.Count} patches independently:");
            foreach (var mpq in patchArchives)
            {
                OpenArchive(mpq, priority++);
            }
        }
        
        Console.WriteLine($"[MpqService] Initialized with {_archives.Count} archives ({patchArchives.Count} patches linked).");
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

    private bool OpenArchive(string path, uint priority = 0)
    {
        if (SFileOpenArchive(path, priority, MPQ_OPEN_READ_ONLY, out var hMpq))
        {
            _archives.Add(hMpq);
            Console.WriteLine($"  [{priority:D3}] {Path.GetFileName(path)}");
            return true;
        }
        return false;
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
            IntPtr hFile = IntPtr.Zero;
            bool opened = SFileOpenFileEx(hMpq, virtualPath, 0, ref hFile);
            
            if (opened && hFile != IntPtr.Zero)
            {
                try
                {
                    uint sizeHigh = 0;
                    var size = SFileGetFileSize(hFile, out sizeHigh);
                    if (size > 0 && size != uint.MaxValue) 
                    {
                        var buffer = new byte[size];
                        var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                        try
                        {
                            if (SFileReadFile(hFile, handle.AddrOfPinnedObject(), size, out var bytesRead, IntPtr.Zero))
                            {
                                if (bytesRead != size) Array.Resize(ref buffer, (int)bytesRead);
                                return buffer;
                            }
                        }
                        finally
                        {
                            handle.Free();
                        }
                    }
                }
                finally
                {
                    SFileCloseFile(hFile);
                }
            }
            
            // Fallback: SFileExtractFile for when handle is 0 but file exists (patch chain issue)
            if (opened || SFileHasFile(hMpq, virtualPath))
            {
                var tempPath = Path.Combine(Path.GetTempPath(), $"mpq_extract_{Guid.NewGuid():N}.tmp");
                try
                {
                    if (SFileExtractFile(hMpq, virtualPath, tempPath, 0))
                    {
                        var data = File.ReadAllBytes(tempPath);
                        File.Delete(tempPath);
                        return data;
                    }
                }
                catch
                {
                    if (File.Exists(tempPath)) File.Delete(tempPath);
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
