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
    /// <param name="pattern">File pattern (e.g., "*.adt")</param>
    /// <param name="listFile">Optional path to external listfile for file enumeration</param>
    public List<string> ListFiles(string pattern = "*", string? listFile = null)
    {
        var results = new List<string>();
        
        var hFind = SFileFindFirstFile(_hArchive, pattern, out var findData, listFile);
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
    /// <param name="mapName">Map name (e.g., "Azeroth")</param>
    /// <param name="listFile">Optional path to external listfile for file enumeration</param>
    public List<string> ListMapAdts(string mapName, string? listFile = null)
    {
        var pattern = $"World\\Maps\\{mapName}\\*.adt";
        var all = ListFiles(pattern, listFile);
        
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

    #region ADT Extraction Methods
    
    /// <summary>
    /// Extract placements (M2/WMO with UniqueID, position, rotation) from an ADT file.
    /// </summary>
    public AdtPlacementData? ExtractPlacements(string mapName, int tileX, int tileY)
    {
        var mapLower = mapName.ToLowerInvariant();
        
        // Try lowercase paths first (MPQs often store lowercase), then mixed case
        var paths = new[]
        {
            // Lowercase split format
            ($"world\\maps\\{mapLower}\\{mapLower}_{tileX}_{tileY}_obj0.adt", true),
            // Lowercase monolithic
            ($"world\\maps\\{mapLower}\\{mapLower}_{tileX}_{tileY}.adt", false),
            // Mixed case split format
            ($"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_obj0.adt", true),
            // Mixed case monolithic  
            ($"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt", false)
        };
        
        byte[]? adtBytes = null;
        bool isSplit = false;
        
        foreach (var (path, split) in paths)
        {
            if (HasFile(path))
            {
                adtBytes = ReadFile(path);
                isSplit = split;
                break;
            }
        }
        
        if (adtBytes == null || adtBytes.Length < 12)
            return null;
            
        return ExtractPlacementsFromBytes(adtBytes, mapName, tileX, tileY, isSplit);
    }
    
    /// <summary>
    /// Extract placements from raw ADT bytes.
    /// </summary>
    public AdtPlacementData ExtractPlacementsFromBytes(byte[] adtBytes, string mapName, int tileX, int tileY, bool isSplitFormat)
    {
        var result = new AdtPlacementData
        {
            Map = mapName,
            TileX = tileX,
            TileY = tileY,
            M2Placements = new List<M2Placement>(),
            WmoPlacements = new List<WmoPlacement>()
        };
        
        // Parse name tables first (MMDX/MWMO)
        var m2Names = new List<string>();
        var wmoNames = new List<string>();
        
        // Scan for chunks
        int pos = 0;
        while (pos + 8 <= adtBytes.Length)
        {
            string sig = Encoding.ASCII.GetString(adtBytes, pos, 4);
            int size = BitConverter.ToInt32(adtBytes, pos + 4);
            
            if (size < 0 || pos + 8 + size > adtBytes.Length) break;
            
            // Normalize FourCC (reverse for on-disk format)
            string readable = new string(sig.Reverse().ToArray());
            int dataStart = pos + 8;
            
            switch (readable)
            {
                case "MMDX": // M2 model names
                    m2Names = ParseNullTerminatedStrings(adtBytes, dataStart, size);
                    break;
                    
                case "MWMO": // WMO names
                    wmoNames = ParseNullTerminatedStrings(adtBytes, dataStart, size);
                    break;
                    
                case "MDDF": // M2 placements (36 bytes each)
                    ParseMddf(adtBytes, dataStart, size, m2Names, result.M2Placements);
                    break;
                    
                case "MODF": // WMO placements (64 bytes each)
                    ParseModf(adtBytes, dataStart, size, wmoNames, result.WmoPlacements);
                    break;
            }
            
            pos += 8 + size;
            // Align to 4 bytes
            if ((pos & 3) != 0) pos = (pos + 3) & ~3;
        }
        
        return result;
    }
    
    private List<string> ParseNullTerminatedStrings(byte[] data, int offset, int length)
    {
        var result = new List<string>();
        int pos = offset;
        int end = offset + length;
        
        while (pos < end)
        {
            int nullPos = Array.IndexOf(data, (byte)0, pos, end - pos);
            if (nullPos == -1) nullPos = end;
            
            int len = nullPos - pos;
            if (len > 0)
            {
                string s = Encoding.UTF8.GetString(data, pos, len);
                if (!string.IsNullOrWhiteSpace(s))
                    result.Add(s);
            }
            pos = nullPos + 1;
        }
        return result;
    }
    
    private void ParseMddf(byte[] data, int offset, int length, List<string> names, List<M2Placement> placements)
    {
        // MDDF entry: 36 bytes each
        // 0-3: nameId (index into MMDX)
        // 4-7: uniqueId
        // 8-19: position (3 floats)
        // 20-31: rotation (3 floats)
        // 32-33: scale (ushort, divide by 1024)
        // 34-35: flags
        
        const int entrySize = 36;
        int count = length / entrySize;
        
        for (int i = 0; i < count; i++)
        {
            int start = offset + (i * entrySize);
            if (start + entrySize > data.Length) break;
            
            uint nameId = BitConverter.ToUInt32(data, start);
            uint uniqueId = BitConverter.ToUInt32(data, start + 4);
            float posX = BitConverter.ToSingle(data, start + 8);
            float posY = BitConverter.ToSingle(data, start + 12);
            float posZ = BitConverter.ToSingle(data, start + 16);
            float rotX = BitConverter.ToSingle(data, start + 20);
            float rotY = BitConverter.ToSingle(data, start + 24);
            float rotZ = BitConverter.ToSingle(data, start + 28);
            ushort scaleRaw = BitConverter.ToUInt16(data, start + 32);
            ushort flags = BitConverter.ToUInt16(data, start + 34);
            
            string path = (nameId < names.Count) ? names[(int)nameId] : $"<MMDX:{nameId}>";
            float scale = scaleRaw / 1024f;
            
            placements.Add(new M2Placement
            {
                Path = path,
                UniqueId = uniqueId,
                PositionX = posX,
                PositionY = posY,
                PositionZ = posZ,
                RotationX = rotX,
                RotationY = rotY,
                RotationZ = rotZ,
                Scale = scale,
                Flags = flags
            });
        }
    }
    
    private void ParseModf(byte[] data, int offset, int length, List<string> names, List<WmoPlacement> placements)
    {
        // MODF entry: 64 bytes each
        // 0-3: nameId (index into MWMO)
        // 4-7: uniqueId
        // 8-19: position (3 floats)
        // 20-31: rotation (3 floats)
        // 32-43: extentsLower (3 floats) 
        // 44-55: extentsUpper (3 floats)
        // 56-57: flags
        // 58-59: doodadSet
        // 60-61: nameSet
        // 62-63: scale (ushort, Cata+)
        
        const int entrySize = 64;
        int count = length / entrySize;
        
        for (int i = 0; i < count; i++)
        {
            int start = offset + (i * entrySize);
            if (start + entrySize > data.Length) break;
            
            uint nameId = BitConverter.ToUInt32(data, start);
            uint uniqueId = BitConverter.ToUInt32(data, start + 4);
            float posX = BitConverter.ToSingle(data, start + 8);
            float posY = BitConverter.ToSingle(data, start + 12);
            float posZ = BitConverter.ToSingle(data, start + 16);
            float rotX = BitConverter.ToSingle(data, start + 20);
            float rotY = BitConverter.ToSingle(data, start + 24);
            float rotZ = BitConverter.ToSingle(data, start + 28);
            ushort flags = BitConverter.ToUInt16(data, start + 56);
            ushort doodadSet = BitConverter.ToUInt16(data, start + 58);
            ushort nameSet = BitConverter.ToUInt16(data, start + 60);
            ushort scaleRaw = BitConverter.ToUInt16(data, start + 62);
            
            string path = (nameId < names.Count) ? names[(int)nameId] : $"<MWMO:{nameId}>";
            float scale = scaleRaw > 0 ? scaleRaw / 1024f : 1.0f;
            
            placements.Add(new WmoPlacement
            {
                Path = path,
                UniqueId = uniqueId,
                PositionX = posX,
                PositionY = posY,
                PositionZ = posZ,
                RotationX = rotX,
                RotationY = rotY,
                RotationZ = rotZ,
                Scale = scale,
                Flags = flags,
                DoodadSet = doodadSet,
                NameSet = nameSet
            });
        }
    }
    
    /// <summary>
    /// Extract terrain data (heights, textures, layers) from an ADT file.
    /// </summary>
    public TileTerrainData? ExtractTerrain(string mapName, int tileX, int tileY)
    {
        var mapLower = mapName.ToLowerInvariant();
        
        // Try lowercase first (MPQs often store lowercase), then mixed case
        var paths = new[]
        {
            $"world\\maps\\{mapLower}\\{mapLower}_{tileX}_{tileY}.adt",
            $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt"
        };
        
        byte[]? adtBytes = null;
        foreach (var path in paths)
        {
            if (HasFile(path))
            {
                adtBytes = ReadFile(path);
                break;
            }
        }
        
        if (adtBytes == null || adtBytes.Length < 12)
            return null;
            
        return ExtractTerrainFromBytes(adtBytes, mapName, tileX, tileY);
    }
    
    /// <summary>
    /// Extract terrain data from raw ADT bytes.
    /// </summary>
    public TileTerrainData ExtractTerrainFromBytes(byte[] adtBytes, string mapName, int tileX, int tileY)
    {
        var result = new TileTerrainData
        {
            Map = mapName,
            TileX = tileX,
            TileY = tileY,
            Textures = new List<string>(),
            Chunks = new List<ChunkTerrainData>()
        };
        
        // First pass: collect textures (MTEX)
        int pos = 0;
        while (pos + 8 <= adtBytes.Length)
        {
            string sig = Encoding.ASCII.GetString(adtBytes, pos, 4);
            int size = BitConverter.ToInt32(adtBytes, pos + 4);
            
            if (size < 0 || pos + 8 + size > adtBytes.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            int dataStart = pos + 8;
            
            if (readable == "MTEX")
            {
                result.Textures = ParseNullTerminatedStrings(adtBytes, dataStart, size);
            }
            
            pos += 8 + size;
        }
        
        // Second pass: find and parse MCNKs
        pos = 0;
        int chunkIdx = 0;
        while (pos + 8 <= adtBytes.Length)
        {
            string sig = Encoding.ASCII.GetString(adtBytes, pos, 4);
            int size = BitConverter.ToInt32(adtBytes, pos + 4);
            
            if (size < 0 || pos + 8 + size > adtBytes.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            if (readable == "MCNK" && size >= 128)
            {
                var mcnkData = new byte[size];
                Buffer.BlockCopy(adtBytes, pos + 8, mcnkData, 0, size);
                
                var chunkData = ExtractChunkTerrainData(mcnkData, chunkIdx);
                if (chunkData != null)
                {
                    result.Chunks.Add(chunkData);
                }
                chunkIdx++;
            }
            
            pos += 8 + size;
        }
        
        return result;
    }
    
    private ChunkTerrainData? ExtractChunkTerrainData(byte[] mcnkData, int chunkIdx)
    {
        if (mcnkData.Length < 128) return null;
        
        var result = new ChunkTerrainData
        {
            Idx = chunkIdx,
            Heights = new List<float>(),
            Layers = new List<TextureLayerInfo>()
        };
        
        // Parse MCNK header (first 128 bytes)
        // Read base height at offset 104 (4 bytes float)
        float baseHeight = 0;
        if (mcnkData.Length >= 108)
        {
            baseHeight = BitConverter.ToSingle(mcnkData, 104);
        }
        
        // Parse subchunks within MCNK (starting after 128-byte header)
        int pos = 128;
        while (pos + 8 <= mcnkData.Length)
        {
            string sig = Encoding.ASCII.GetString(mcnkData, pos, 4);
            int size = BitConverter.ToInt32(mcnkData, pos + 4);
            
            if (size < 0 || pos + 8 + size > mcnkData.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            int dataStart = pos + 8;
            
            switch (readable)
            {
                case "MCVT": // Heights (145 floats = 580 bytes)
                    ParseMcvt(mcnkData, dataStart, size, baseHeight, result.Heights);
                    break;
                    
                case "MCLY": // Texture layers (16 bytes each)
                    ParseMcly(mcnkData, dataStart, size, result.Layers);
                    break;
                    
                case "MCAL": // Alpha maps
                    // Store raw alpha for now - can be decoded later
                    if (size > 0)
                    {
                        result.AlphaRaw = new byte[size];
                        Buffer.BlockCopy(mcnkData, dataStart, result.AlphaRaw, 0, size);
                    }
                    break;
            }
            
            pos += 8 + size;
        }
        
        return result.Heights.Count > 0 ? result : null;
    }
    
    private void ParseMcvt(byte[] data, int offset, int length, float baseHeight, List<float> heights)
    {
        // MCVT contains 145 floats (9x9 + 8x8 interleaved)
        int count = Math.Min(145, length / 4);
        for (int i = 0; i < count; i++)
        {
            int pos = offset + (i * 4);
            if (pos + 4 > data.Length) break;
            
            float h = BitConverter.ToSingle(data, pos);
            heights.Add(baseHeight + h);
        }
    }
    
    private void ParseMcly(byte[] data, int offset, int length, List<TextureLayerInfo> layers)
    {
        // MCLY entry: 16 bytes each
        // 0-3: textureId (index into MTEX)
        // 4-7: flags
        // 8-11: offsetInMCAL
        // 12-15: effectId (ground effect)
        
        const int entrySize = 16;
        int count = length / entrySize;
        
        for (int i = 0; i < count; i++)
        {
            int start = offset + (i * entrySize);
            if (start + entrySize > data.Length) break;
            
            uint textureId = BitConverter.ToUInt32(data, start);
            uint flags = BitConverter.ToUInt32(data, start + 4);
            uint alphaOffset = BitConverter.ToUInt32(data, start + 8);
            
            layers.Add(new TextureLayerInfo
            {
                TextureId = (int)textureId,
                Flags = flags,
                AlphaOffset = (int)alphaOffset
            });
        }
    }
    
    #endregion

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

#region Data Models

/// <summary>ADT placement data extracted from MDDF/MODF chunks.</summary>
public class AdtPlacementData
{
    public string Map { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public List<M2Placement> M2Placements { get; set; } = new();
    public List<WmoPlacement> WmoPlacements { get; set; } = new();
}

/// <summary>M2 placement entry from MDDF chunk.</summary>
public record M2Placement
{
    public string Path { get; init; } = "";
    public uint UniqueId { get; init; }
    public float PositionX { get; init; }
    public float PositionY { get; init; }
    public float PositionZ { get; init; }
    public float RotationX { get; init; }
    public float RotationY { get; init; }
    public float RotationZ { get; init; }
    public float Scale { get; init; }
    public ushort Flags { get; init; }
}

/// <summary>WMO placement entry from MODF chunk.</summary>
public record WmoPlacement
{
    public string Path { get; init; } = "";
    public uint UniqueId { get; init; }
    public float PositionX { get; init; }
    public float PositionY { get; init; }
    public float PositionZ { get; init; }
    public float RotationX { get; init; }
    public float RotationY { get; init; }
    public float RotationZ { get; init; }
    public float Scale { get; init; }
    public ushort Flags { get; init; }
    public ushort DoodadSet { get; init; }
    public ushort NameSet { get; init; }
}

/// <summary>Terrain data extracted from an ADT tile.</summary>
public class TileTerrainData
{
    public string Map { get; set; } = "";
    public int TileX { get; set; }
    public int TileY { get; set; }
    public List<string> Textures { get; set; } = new();
    public List<ChunkTerrainData> Chunks { get; set; } = new();
}

/// <summary>Terrain data for a single MCNK chunk.</summary>
public class ChunkTerrainData
{
    public int Idx { get; set; }
    public List<float> Heights { get; set; } = new();
    public List<TextureLayerInfo> Layers { get; set; } = new();
    public byte[]? AlphaRaw { get; set; }
}

/// <summary>Texture layer info from MCLY chunk.</summary>
public class TextureLayerInfo
{
    public int TextureId { get; set; }
    public uint Flags { get; set; }
    public int AlphaOffset { get; set; }
}

#endregion

/// <summary>
/// Static terrain parser for extracting terrain data from raw ADT bytes
/// without requiring an MPQ-backed instance.
/// </summary>
public static class AdtTerrainParser
{
    /// <summary>
    /// Extract terrain data (heights, textures, layers, alpha) from raw ADT bytes.
    /// </summary>
    public static TileTerrainData Parse(byte[] adtBytes, string mapName, int tileX, int tileY)
    {
        var result = new TileTerrainData
        {
            Map = mapName,
            TileX = tileX,
            TileY = tileY,
            Textures = new List<string>(),
            Chunks = new List<ChunkTerrainData>()
        };
        
        // First pass: collect textures (MTEX)
        int pos = 0;
        while (pos + 8 <= adtBytes.Length)
        {
            string sig = Encoding.ASCII.GetString(adtBytes, pos, 4);
            int size = BitConverter.ToInt32(adtBytes, pos + 4);
            
            if (size < 0 || pos + 8 + size > adtBytes.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            int dataStart = pos + 8;
            
            if (readable == "MTEX")
            {
                result.Textures = ParseNullTerminatedStrings(adtBytes, dataStart, size);
            }
            
            pos += 8 + size;
        }
        
        // Second pass: find and parse MCNKs
        pos = 0;
        int chunkIdx = 0;
        while (pos + 8 <= adtBytes.Length)
        {
            string sig = Encoding.ASCII.GetString(adtBytes, pos, 4);
            int size = BitConverter.ToInt32(adtBytes, pos + 4);
            
            if (size < 0 || pos + 8 + size > adtBytes.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            if (readable == "MCNK" && size >= 128)
            {
                var mcnkData = new byte[size];
                Buffer.BlockCopy(adtBytes, pos + 8, mcnkData, 0, size);
                
                var chunkData = ExtractChunkTerrainData(mcnkData, chunkIdx);
                if (chunkData != null)
                {
                    result.Chunks.Add(chunkData);
                }
                chunkIdx++;
            }
            
            pos += 8 + size;
        }
        
        return result;
    }
    
    private static List<string> ParseNullTerminatedStrings(byte[] data, int offset, int length)
    {
        var result = new List<string>();
        int start = offset;
        
        for (int i = offset; i < offset + length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                {
                    var str = Encoding.ASCII.GetString(data, start, i - start);
                    if (!string.IsNullOrWhiteSpace(str))
                        result.Add(str);
                }
                start = i + 1;
            }
        }
        
        return result;
    }
    
    private static ChunkTerrainData? ExtractChunkTerrainData(byte[] mcnkData, int chunkIdx)
    {
        if (mcnkData.Length < 128) return null;
        
        var result = new ChunkTerrainData
        {
            Idx = chunkIdx,
            Heights = new List<float>(),
            Layers = new List<TextureLayerInfo>()
        };
        
        // Parse MCNK header (first 128 bytes)
        // Read base height at offset 104 (4 bytes float)
        float baseHeight = 0;
        if (mcnkData.Length >= 108)
        {
            baseHeight = BitConverter.ToSingle(mcnkData, 104);
        }
        
        // Parse subchunks within MCNK (starting after 128-byte header)
        int pos = 128;
        while (pos + 8 <= mcnkData.Length)
        {
            string sig = Encoding.ASCII.GetString(mcnkData, pos, 4);
            int size = BitConverter.ToInt32(mcnkData, pos + 4);
            
            if (size < 0 || pos + 8 + size > mcnkData.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            int dataStart = pos + 8;
            
            switch (readable)
            {
                case "MCVT": // Heights (145 floats = 580 bytes)
                    ParseMcvt(mcnkData, dataStart, size, baseHeight, result.Heights);
                    break;
                    
                case "MCLY": // Texture layers (16 bytes each)
                    ParseMcly(mcnkData, dataStart, size, result.Layers);
                    break;
                    
                case "MCAL": // Alpha maps
                    // Store raw alpha for now - can be decoded later
                    if (size > 0)
                    {
                        result.AlphaRaw = new byte[size];
                        Buffer.BlockCopy(mcnkData, dataStart, result.AlphaRaw, 0, size);
                    }
                    break;
            }
            
            pos += 8 + size;
        }
        
        return result.Heights.Count > 0 ? result : null;
    }
    
    private static void ParseMcvt(byte[] data, int offset, int length, float baseHeight, List<float> heights)
    {
        int count = Math.Min(145, length / 4);
        for (int i = 0; i < count; i++)
        {
            int p = offset + (i * 4);
            if (p + 4 > data.Length) break;
            
            float h = BitConverter.ToSingle(data, p);
            heights.Add(baseHeight + h);
        }
    }
    
    private static void ParseMcly(byte[] data, int offset, int length, List<TextureLayerInfo> layers)
    {
        const int entrySize = 16;
        int count = length / entrySize;
        
        for (int i = 0; i < count; i++)
        {
            int start = offset + (i * entrySize);
            if (start + entrySize > data.Length) break;
            
            uint textureId = BitConverter.ToUInt32(data, start);
            uint flags = BitConverter.ToUInt32(data, start + 4);
            uint alphaOffset = BitConverter.ToUInt32(data, start + 8);
            
            layers.Add(new TextureLayerInfo
            {
                TextureId = (int)textureId,
                Flags = flags,
                AlphaOffset = (int)alphaOffset
            });
        }
    }
}
