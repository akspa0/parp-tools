using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWRollback.PM4Module;

/// <summary>
/// Validates ADT chunk structure to catch corruption before Noggit rendering.
/// </summary>
public class AdtChunkValidator
{
    public record ValidationResult(
        string FilePath,
        bool IsValid,
        List<string> Errors,
        List<string> Warnings,
        ChunkStats Stats
    );
    
    public record ChunkStats(
        int MwmoCount,
        int MwidCount,
        int ModfCount,
        int MmdxCount,
        int MmidCount,
        int MddfCount
    );
    
    /// <summary>
    /// Validate a single ADT file for chunk structure integrity.
    /// </summary>
    public ValidationResult Validate(string adtPath)
    {
        var errors = new List<string>();
        var warnings = new List<string>();
        var stats = new ChunkStats(0, 0, 0, 0, 0, 0);
        
        if (!File.Exists(adtPath))
        {
            errors.Add($"File not found: {adtPath}");
            return new ValidationResult(adtPath, false, errors, warnings, stats);
        }
        
        try
        {
            var bytes = File.ReadAllBytes(adtPath);
            var str = Encoding.ASCII.GetString(bytes);
            
            // Find chunk positions
            int mwmoIdx = str.IndexOf("OMWM"); // MWMO reversed
            int mwidIdx = str.IndexOf("DIWM"); // MWID reversed
            int modfIdx = str.IndexOf("FDOM"); // MODF reversed
            int mmdxIdx = str.IndexOf("XDMM"); // MMDX reversed
            int mmidIdx = str.IndexOf("DIMM"); // MMID reversed
            int mddfIdx = str.IndexOf("FDDM"); // MDDF reversed
            
            // Parse MWMO (WMO names)
            var wmoNames = new List<string>();
            var wmoOffsets = new List<int>();
            if (mwmoIdx >= 0)
            {
                int mwmoSize = BitConverter.ToInt32(bytes, mwmoIdx + 4);
                int offset = 0;
                int start = mwmoIdx + 8;
                
                for (int i = 0; i < mwmoSize; i++)
                {
                    if (bytes[start + i] == 0)
                    {
                        if (i > offset)
                        {
                            wmoOffsets.Add(offset);
                            wmoNames.Add(Encoding.ASCII.GetString(bytes, start + offset, i - offset));
                        }
                        offset = i + 1;
                    }
                }
            }
            
            // Parse MWID (WMO name offsets)
            var mwidOffsets = new List<int>();
            if (mwidIdx >= 0)
            {
                int mwidSize = BitConverter.ToInt32(bytes, mwidIdx + 4);
                int count = mwidSize / 4;
                for (int i = 0; i < count; i++)
                {
                    mwidOffsets.Add(BitConverter.ToInt32(bytes, mwidIdx + 8 + i * 4));
                }
            }
            
            // Validate MWMO/MWID alignment
            if (wmoNames.Count > 0 && mwidOffsets.Count > 0)
            {
                if (wmoNames.Count != mwidOffsets.Count)
                {
                    errors.Add($"MWMO/MWID count mismatch: {wmoNames.Count} names vs {mwidOffsets.Count} offsets");
                }
                else
                {
                    for (int i = 0; i < wmoNames.Count; i++)
                    {
                        if (i < wmoOffsets.Count && wmoOffsets[i] != mwidOffsets[i])
                        {
                            errors.Add($"MWID[{i}] offset {mwidOffsets[i]} != expected {wmoOffsets[i]} for '{wmoNames[i]}'");
                        }
                    }
                }
            }
            
            // Parse MODF entries and validate NameId references
            var modfUniqueIds = new List<uint>();
            int modfCount = 0;
            if (modfIdx >= 0)
            {
                int modfSize = BitConverter.ToInt32(bytes, modfIdx + 4);
                modfCount = modfSize / 64;
                
                for (int i = 0; i < modfCount; i++)
                {
                    int off = modfIdx + 8 + i * 64;
                    uint nameId = BitConverter.ToUInt32(bytes, off);
                    uint uniqueId = BitConverter.ToUInt32(bytes, off + 4);
                    
                    modfUniqueIds.Add(uniqueId);
                    
                    // Validate NameId is within MWMO count
                    if (nameId >= wmoNames.Count && wmoNames.Count > 0)
                    {
                        errors.Add($"MODF[{i}] NameId={nameId} exceeds MWMO count ({wmoNames.Count})");
                    }
                }
            }
            
            // Parse MMDX (M2 names)
            var m2Names = new List<string>();
            var m2Offsets = new List<int>();
            if (mmdxIdx >= 0)
            {
                int mmdxSize = BitConverter.ToInt32(bytes, mmdxIdx + 4);
                int offset = 0;
                int start = mmdxIdx + 8;
                
                for (int i = 0; i < mmdxSize; i++)
                {
                    if (bytes[start + i] == 0)
                    {
                        if (i > offset)
                        {
                            m2Offsets.Add(offset);
                            m2Names.Add(Encoding.ASCII.GetString(bytes, start + offset, i - offset));
                        }
                        offset = i + 1;
                    }
                }
            }
            
            // Parse MMID (M2 name offsets)
            var mmidOffsets = new List<int>();
            if (mmidIdx >= 0)
            {
                int mmidSize = BitConverter.ToInt32(bytes, mmidIdx + 4);
                int count = mmidSize / 4;
                for (int i = 0; i < count; i++)
                {
                    mmidOffsets.Add(BitConverter.ToInt32(bytes, mmidIdx + 8 + i * 4));
                }
            }
            
            // Validate MMDX/MMID alignment
            if (m2Names.Count > 0 && mmidOffsets.Count > 0)
            {
                if (m2Names.Count != mmidOffsets.Count)
                {
                    errors.Add($"MMDX/MMID count mismatch: {m2Names.Count} names vs {mmidOffsets.Count} offsets");
                }
            }
            
            // Parse MDDF entries and validate NameId references
            var mddfUniqueIds = new List<uint>();
            int mddfCount = 0;
            if (mddfIdx >= 0)
            {
                int mddfSize = BitConverter.ToInt32(bytes, mddfIdx + 4);
                mddfCount = mddfSize / 36;
                
                for (int i = 0; i < mddfCount; i++)
                {
                    int off = mddfIdx + 8 + i * 36;
                    uint nameId = BitConverter.ToUInt32(bytes, off);
                    uint uniqueId = BitConverter.ToUInt32(bytes, off + 4);
                    
                    mddfUniqueIds.Add(uniqueId);
                    
                    // Validate NameId is within MMDX count
                    if (nameId >= m2Names.Count && m2Names.Count > 0)
                    {
                        errors.Add($"MDDF[{i}] NameId={nameId} exceeds MMDX count ({m2Names.Count})");
                    }
                }
            }
            
            // Check for duplicate UniqueIds within tile
            var allIds = modfUniqueIds.Concat(mddfUniqueIds).ToList();
            var duplicates = allIds.GroupBy(x => x).Where(g => g.Count() > 1).ToList();
            foreach (var dup in duplicates)
            {
                errors.Add($"Duplicate UniqueId {dup.Key} appears {dup.Count()} times within this tile");
            }
            
            stats = new ChunkStats(
                wmoNames.Count,
                mwidOffsets.Count,
                modfCount,
                m2Names.Count,
                mmidOffsets.Count,
                mddfCount
            );
            
            return new ValidationResult(
                adtPath,
                errors.Count == 0,
                errors,
                warnings,
                stats
            );
        }
        catch (Exception ex)
        {
            errors.Add($"Parse error: {ex.Message}");
            return new ValidationResult(adtPath, false, errors, warnings, stats);
        }
    }
    
    /// <summary>
    /// Validate all ADTs in a directory and check cross-tile UniqueID collisions.
    /// </summary>
    public (List<ValidationResult> Results, Dictionary<uint, List<string>> CrossTileCollisions) ValidateDirectory(string dirPath)
    {
        var results = new List<ValidationResult>();
        var globalUniqueIds = new Dictionary<uint, List<string>>(); // UniqueId -> list of tiles
        
        var adtFiles = Directory.GetFiles(dirPath, "*.adt")
            .Where(f => !Path.GetFileName(f).Contains("_obj") && !Path.GetFileName(f).Contains("_tex"))
            .OrderBy(f => f)
            .ToList();
        
        Console.WriteLine($"[INFO] Validating {adtFiles.Count} ADT files...");
        
        foreach (var adtPath in adtFiles)
        {
            var result = Validate(adtPath);
            results.Add(result);
            
            // Track UniqueIds for cross-tile collision detection
            var tileName = Path.GetFileNameWithoutExtension(adtPath);
            
            // Re-parse to get UniqueIds (could optimize by returning from Validate)
            try
            {
                var bytes = File.ReadAllBytes(adtPath);
                var str = Encoding.ASCII.GetString(bytes);
                
                int modfIdx = str.IndexOf("FDOM");
                if (modfIdx >= 0)
                {
                    int modfSize = BitConverter.ToInt32(bytes, modfIdx + 4);
                    int count = modfSize / 64;
                    for (int i = 0; i < count; i++)
                    {
                        uint uid = BitConverter.ToUInt32(bytes, modfIdx + 8 + i * 64 + 4);
                        if (!globalUniqueIds.ContainsKey(uid))
                            globalUniqueIds[uid] = new List<string>();
                        globalUniqueIds[uid].Add(tileName);
                    }
                }
                
                int mddfIdx = str.IndexOf("FDDM");
                if (mddfIdx >= 0)
                {
                    int mddfSize = BitConverter.ToInt32(bytes, mddfIdx + 4);
                    int count = mddfSize / 36;
                    for (int i = 0; i < count; i++)
                    {
                        uint uid = BitConverter.ToUInt32(bytes, mddfIdx + 8 + i * 36 + 4);
                        if (!globalUniqueIds.ContainsKey(uid))
                            globalUniqueIds[uid] = new List<string>();
                        globalUniqueIds[uid].Add($"{tileName}:MDDF");
                    }
                }
            }
            catch { /* ignore parse errors for collision tracking */ }
        }
        
        // Find cross-tile collisions
        var collisions = globalUniqueIds
            .Where(kvp => kvp.Value.Distinct().Count() > 1)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        
        // Summary
        int validCount = results.Count(r => r.IsValid);
        int errorCount = results.Count(r => !r.IsValid);
        
        Console.WriteLine($"\n=== Validation Summary ===");
        Console.WriteLine($"Valid ADTs: {validCount}");
        Console.WriteLine($"Invalid ADTs: {errorCount}");
        Console.WriteLine($"Cross-tile UniqueID collisions: {collisions.Count}");
        
        if (collisions.Count > 0)
        {
            Console.WriteLine($"\nTop 10 cross-tile collisions:");
            foreach (var (uid, tiles) in collisions.Take(10))
            {
                Console.WriteLine($"  UniqueId {uid}: {string.Join(", ", tiles.Distinct().Take(5))}{(tiles.Distinct().Count() > 5 ? $" (+{tiles.Distinct().Count() - 5} more)" : "")}");
            }
        }
        
        return (results, collisions);
    }
}
