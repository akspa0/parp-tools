// ADT Patcher V2 - Clean ADT patching interface
// Wraps the working MuseumAdtPatcher with a simpler interface using Pipeline models
// Part of the PM4 Clean Reimplementation

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace WoWRollback.PM4Module.Pipeline;

/// <summary>
/// Clean ADT patcher that patches WMO/M2 placements into ADT files.
/// Uses the working MuseumAdtPatcher internally but with a simplified interface.
/// </summary>
public class AdtPatcherV2
{
    private readonly MuseumAdtPatcher _patcher = new();
    private readonly AdtPatcher _adtPatcher = new();
    
    private uint _nextUniqueId = 1;
    
    /// <summary>
    /// Set the starting unique ID for MODF/MDDF entries.
    /// </summary>
    public void SetStartingUniqueId(uint id) => _nextUniqueId = id;
    
    /// <summary>
    /// Get the current next unique ID.
    /// </summary>
    public uint GetNextUniqueId() => _nextUniqueId;
    
    #region WMO Patching
    
    /// <summary>
    /// Patch WMO placements into an ADT file using Pipeline models.
    /// </summary>
    public PatchResult PatchWmoPlacements(
        string sourceAdtPath,
        string outputPath,
        IEnumerable<ModfEntry> modfEntries,
        IReadOnlyList<string> wmoNames)
    {
        try
        {
            if (!File.Exists(sourceAdtPath))
            {
                return new PatchResult(
                    OutputPath: outputPath,
                    Success: false,
                    ModfCount: 0,
                    MddfCount: 0,
                    Error: $"Source ADT not found: {sourceAdtPath}"
                );
            }
            
            var entries = modfEntries.ToList();
            
            if (entries.Count == 0)
            {
                // No entries - just copy the source file
                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                File.Copy(sourceAdtPath, outputPath, overwrite: true);
                
                return new PatchResult(
                    OutputPath: outputPath,
                    Success: true,
                    ModfCount: 0,
                    MddfCount: 0
                );
            }
            
            // Convert Pipeline ModfEntry to AdtPatcher.ModfEntry
            var adtEntries = entries.Select(e => ConvertToAdtModfEntry(e)).ToList();
            
            // Use the working MuseumAdtPatcher
            _patcher.PatchWmoPlacements(
                sourceAdtPath,
                outputPath,
                wmoNames,
                adtEntries,
                ref _nextUniqueId
            );
            
            return new PatchResult(
                OutputPath: outputPath,
                Success: true,
                ModfCount: entries.Count,
                MddfCount: 0
            );
        }
        catch (Exception ex)
        {
            return new PatchResult(
                OutputPath: outputPath,
                Success: false,
                ModfCount: 0,
                MddfCount: 0,
                Error: ex.Message
            );
        }
    }
    
    /// <summary>
    /// Patch M2/doodad placements into an ADT file using Pipeline models.
    /// </summary>
    public PatchResult PatchM2Placements(
        string sourceAdtPath,
        string outputPath,
        IEnumerable<MddfEntry> mddfEntries,
        IReadOnlyList<string> m2Names)
    {
        try
        {
            if (!File.Exists(sourceAdtPath))
            {
                return new PatchResult(
                    OutputPath: outputPath,
                    Success: false,
                    ModfCount: 0,
                    MddfCount: 0,
                    Error: $"Source ADT not found: {sourceAdtPath}"
                );
            }
            
            var entries = mddfEntries.ToList();
            
            if (entries.Count == 0)
            {
                return new PatchResult(
                    OutputPath: outputPath,
                    Success: true,
                    ModfCount: 0,
                    MddfCount: 0
                );
            }
            
            // Convert Pipeline MddfEntry to AdtPatcher.MddfEntry
            var adtEntries = entries.Select(e => ConvertToAdtMddfEntry(e)).ToList();
            
            // Use the working MuseumAdtPatcher
            _patcher.PatchDoodadPlacements(
                sourceAdtPath,
                outputPath,
                m2Names,
                adtEntries,
                ref _nextUniqueId
            );
            
            return new PatchResult(
                OutputPath: outputPath,
                Success: true,
                ModfCount: 0,
                MddfCount: entries.Count
            );
        }
        catch (Exception ex)
        {
            return new PatchResult(
                OutputPath: outputPath,
                Success: false,
                ModfCount: 0,
                MddfCount: 0,
                Error: ex.Message
            );
        }
    }
    
    /// <summary>
    /// Patch both WMO and M2 placements into an ADT file.
    /// </summary>
    public PatchResult PatchAll(
        string sourceAdtPath,
        string outputPath,
        IEnumerable<ModfEntry> modfEntries,
        IReadOnlyList<string> wmoNames,
        IEnumerable<MddfEntry> mddfEntries,
        IReadOnlyList<string> m2Names)
    {
        // First patch WMOs
        var wmoResult = PatchWmoPlacements(sourceAdtPath, outputPath, modfEntries, wmoNames);
        if (!wmoResult.Success)
            return wmoResult;
        
        // Then patch M2s (if any) - use output as source since it now has WMO changes
        var m2Entries = mddfEntries.ToList();
        if (m2Entries.Count > 0)
        {
            var m2Result = PatchM2Placements(outputPath, outputPath, m2Entries, m2Names);
            if (!m2Result.Success)
                return m2Result;
            
            return new PatchResult(
                OutputPath: outputPath,
                Success: true,
                ModfCount: wmoResult.ModfCount,
                MddfCount: m2Result.MddfCount
            );
        }
        
        return wmoResult;
    }
    
    #endregion
    
    #region Batch Operations
    
    /// <summary>
    /// Patch a batch of tiles with their respective MODF entries.
    /// </summary>
    public IEnumerable<PatchResult> PatchTiles(
        string sourceDirectory,
        string outputDirectory,
        IEnumerable<ModfEntry> allModfEntries,
        IReadOnlyList<string> wmoNames,
        string mapName = "development")
    {
        // Group entries by tile
        var entriesByTile = allModfEntries
            .GroupBy(e => GetTileFromPosition(e.Position))
            .ToDictionary(g => g.Key, g => g.ToList());
        
        Console.WriteLine($"[INFO] Patching {entriesByTile.Count} tiles with MODF entries");
        
        foreach (var (tile, entries) in entriesByTile)
        {
            var sourceAdtPath = Path.Combine(sourceDirectory, $"{mapName}_{tile.x}_{tile.y}.adt");
            var outputAdtPath = Path.Combine(outputDirectory, $"{mapName}_{tile.x}_{tile.y}.adt");
            
            var result = PatchWmoPlacements(sourceAdtPath, outputAdtPath, entries, wmoNames);
            yield return result;
        }
    }
    
    #endregion
    
    #region Conversion Helpers
    
    /// <summary>
    /// Convert Pipeline ModfEntry to AdtPatcher.ModfEntry.
    /// </summary>
    private static AdtPatcher.ModfEntry ConvertToAdtModfEntry(ModfEntry entry)
    {
        return new AdtPatcher.ModfEntry
        {
            NameId = (uint)entry.NameIndex,
            UniqueId = entry.UniqueId,
            Position = entry.Position,
            Rotation = entry.Rotation,
            BoundsMin = entry.BoundsMin,
            BoundsMax = entry.BoundsMax,
            Flags = entry.Flags,
            DoodadSet = entry.DoodadSet,
            NameSet = entry.NameSet,
            Scale = entry.Scale
        };
    }
    
    /// <summary>
    /// Convert Pipeline MddfEntry to AdtPatcher.MddfEntry.
    /// </summary>
    private static AdtPatcher.MddfEntry ConvertToAdtMddfEntry(MddfEntry entry)
    {
        return new AdtPatcher.MddfEntry
        {
            NameId = (uint)entry.NameIndex,
            UniqueId = entry.UniqueId,
            Position = entry.Position,
            Rotation = entry.Rotation,
            Scale = entry.Scale,
            Flags = entry.Flags
        };
    }
    
    /// <summary>
    /// Calculate tile coordinates from world position.
    /// </summary>
    private static (int x, int y) GetTileFromPosition(Vector3 position)
    {
        const float TileSize = 533.33333f;
        
        // ADT positions are stored as XZY, so position.Y is actually Z (vertical)
        // and position.Z is actually Y (horizontal)
        // Convert back to calculate tile: X = 32 - (pos.X / TileSize), Y = 32 - (pos.Z / TileSize)
        int tileX = Math.Clamp((int)(32 - (position.X / TileSize)), 0, 63);
        int tileY = Math.Clamp((int)(32 - (position.Z / TileSize)), 0, 63);
        
        return (tileX, tileY);
    }
    
    #endregion
}
