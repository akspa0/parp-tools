using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Warcraft.NET.Files.WDT.Root.WotLK;
using Warcraft.NET.Files.WDT.Flags;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Analyzes WDT (World Data Table) files to extract:
/// - Map flags (terrain vs WMO-only)
/// - WMO placements for WMO-only maps
/// - Tile presence information
/// </summary>
public sealed class WdtAnalyzer
{
    /// <summary>
    /// Analyzes a WDT file from an MPQ archive.
    /// </summary>
    public WdtAnalysisResult Analyze(IArchiveSource src, string mapName)
    {
        try
        {
            var wdtPath = $"world/maps/{mapName}/{mapName}.wdt";
            
            if (!src.FileExists(wdtPath))
            {
                return new WdtAnalysisResult(
                    Success: false,
                    ErrorMessage: $"WDT not found: {wdtPath}",
                    MapName: mapName,
                    HasTerrain: false,
                    IsWmoOnly: false,
                    WmoPlacement: null,
                    TileCount: 0);
            }

            using var stream = src.OpenFile(wdtPath);
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            var data = ms.ToArray();

            var wdt = new WorldDataTable(data);

            // Parse MPHD flags
            var flags = wdt.Header.Flags;
            bool hasWmo = flags.HasFlag(MPHDFlags.UseGlobalMapObject);
            bool hasTerrain = !hasWmo; // If UseGlobalMapObject is set, typically no terrain

            // Count tiles with ADT data (64x64 grid)
            int tileCount = 0;
            if (wdt.Tiles?.Entries != null)
            {
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        if (wdt.Tiles.Entries[x, y].Flags != 0)
                            tileCount++;
                    }
                }
            }

            // Extract WMO placement if present
            WmoPlacementInfo? wmoPlacement = null;
            if (hasWmo && wdt.WorldModelObjects != null && wdt.WorldModelObjectPlacementInfo != null)
            {
                // WMO-only maps have a single WMO referenced
                if (wdt.WorldModelObjects.Filenames.Count > 0 && 
                    wdt.WorldModelObjectPlacementInfo.MODFEntries.Count > 0)
                {
                    var wmoPath = wdt.WorldModelObjects.Filenames[0];
                    var modf = wdt.WorldModelObjectPlacementInfo.MODFEntries[0];

                    wmoPlacement = new WmoPlacementInfo(
                        WmoPath: wmoPath,
                        UniqueId: modf.UniqueId,
                        PositionX: modf.Position.X,
                        PositionY: modf.Position.Y,
                        PositionZ: modf.Position.Z,
                        RotationX: modf.Rotation.Pitch,
                        RotationY: modf.Rotation.Yaw,
                        RotationZ: modf.Rotation.Roll,
                        Scale: modf.Scale,
                        DoodadSet: modf.DoodadSet,
                        NameSet: modf.NameSet
                    );
                }
            }

            return new WdtAnalysisResult(
                Success: true,
                ErrorMessage: null,
                MapName: mapName,
                HasTerrain: hasTerrain || tileCount > 0, // Hybrid: WDT says WMO-only but tiles exist
                IsWmoOnly: hasWmo,
                WmoPlacement: wmoPlacement,
                TileCount: tileCount);
        }
        catch (Exception ex)
        {
            return new WdtAnalysisResult(
                Success: false,
                ErrorMessage: $"Failed to analyze WDT: {ex.Message}",
                MapName: mapName,
                HasTerrain: false,
                IsWmoOnly: false,
                WmoPlacement: null,
                TileCount: 0);
        }
    }
}

/// <summary>
/// Information about a WMO placement from a WDT file.
/// </summary>
public record WmoPlacementInfo(
    string WmoPath,
    int UniqueId,
    float PositionX,
    float PositionY,
    float PositionZ,
    float RotationX,
    float RotationY,
    float RotationZ,
    ushort Scale,
    ushort DoodadSet,
    ushort NameSet
);

/// <summary>
/// Result of WDT analysis.
/// </summary>
public record WdtAnalysisResult(
    bool Success,
    string? ErrorMessage,
    string MapName,
    bool HasTerrain,
    bool IsWmoOnly,
    WmoPlacementInfo? WmoPlacement,
    int TileCount
);
