using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using Warcraft.NET.Files.ADT.TerrainObject.Zero;
using Warcraft.NET.Files.ADT.Entries;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Correlates PM4 pathfinding objects with ADT MODF placements.
/// Uses development_29_39 as the Rosetta Stone for understanding PM4↔WMO relationships.
/// </summary>
public sealed class Pm4AdtCorrelator
{
    /// <summary>
    /// MODF placement entry with parsed data.
    /// </summary>
    public record ModfPlacement(
        uint NameId,
        int UniqueId,
        Vector3 Position,
        Vector3 Rotation,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        ushort Flags,
        ushort DoodadSet,
        ushort NameSet,
        ushort Scale);

    /// <summary>
    /// MPRL entry from PM4 file.
    /// </summary>
    public record MprlEntry(
        int Index,
        ushort Unknown0x00,
        short Unknown0x02,
        ushort Unknown0x04,
        ushort Unknown0x06,
        Vector3 Position,
        short Unknown0x14,
        ushort Unknown0x16);

    /// <summary>
    /// MSLK entry from PM4 file - scene graph/hierarchy node.
    /// Per PM4 spec: link_id encodes tile crossing as 0xFFFFYYXX (YY=Y tile, XX=X tile)
    /// </summary>
    public record MslkEntry(
        int Index,
        byte ObjectTypeFlags,    // 0x00 - Object type classification (1-18 values)
        byte ObjectSubtype,      // 0x01 - Object subtype/variant (0-7 values)
        ushort Reserved,         // 0x02 - Always 0x0000
        uint ParentIndex,        // 0x04 - Parent grouping ID (used for object hierarchy)
        int MspiFirstIndex,      // 0x08 - Index into MSPI (-1 for non-geometry nodes)
        byte MspiIndexCount,     // 0x0B - Number of MSPI indices
        uint LinkId,             // 0x0C - Tile crossing: 0xFFFFYYXX (YY=Y tile, XX=X tile)
        ushort ReferenceIndex,   // 0x10 - Cross-reference to other structures
        ushort SystemFlag)       // 0x12 - Always 0x8000
    {
        /// <summary>
        /// Decode LinkId to tile coordinates. Returns (tileX, tileY) or null if not a tile link.
        /// Format: 0xFFFFYYXX where YY=Y tile, XX=X tile
        /// </summary>
        public (int TileX, int TileY)? DecodeTileLink()
        {
            if ((LinkId & 0xFFFF0000) == 0xFFFF0000)
            {
                int tileX = (int)(LinkId & 0xFF);
                int tileY = (int)((LinkId >> 8) & 0xFF);
                return (tileX, tileY);
            }
            return null;
        }
    }

    /// <summary>
    /// Correlation result between PM4 object and MODF placement.
    /// </summary>
    public record CorrelationResult(
        int MprlIndex,
        Vector3 MprlPosition,
        int? ModfIndex,
        int? ModfUniqueId,
        Vector3? ModfPosition,
        float? Distance,
        string? WmoNameIndex);

    /// <summary>
    /// Parse _obj0.adt file and extract MODF placements.
    /// </summary>
    public List<ModfPlacement> ParseObj0Adt(string obj0AdtPath)
    {
        var placements = new List<ModfPlacement>();
        
        if (!File.Exists(obj0AdtPath))
        {
            Console.WriteLine($"[ERROR] _obj0.adt not found: {obj0AdtPath}");
            return placements;
        }

        try
        {
            var data = File.ReadAllBytes(obj0AdtPath);
            var obj0 = new TerrainObjectZero(data);

            if (obj0.WorldModelObjectPlacementInfo?.MODFEntries != null)
            {
                foreach (var entry in obj0.WorldModelObjectPlacementInfo.MODFEntries)
                {
                    placements.Add(new ModfPlacement(
                        entry.NameId,
                        entry.UniqueId,
                        entry.Position,
                        new Vector3(entry.Rotation.Pitch, entry.Rotation.Yaw, entry.Rotation.Roll),
                        entry.BoundingBox.Minimum,
                        entry.BoundingBox.Maximum,
                        (ushort)entry.Flags,
                        entry.DoodadSet,
                        entry.NameSet,
                        entry.Scale));
                }
            }

            Console.WriteLine($"[INFO] Parsed {placements.Count} MODF placements from {Path.GetFileName(obj0AdtPath)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to parse _obj0.adt: {ex.Message}");
        }

        return placements;
    }

    /// <summary>
    /// Parse PM4 file and extract MPRL entries (object positions).
    /// Simple binary parser - PM4 format is chunked like ADT.
    /// </summary>
    public List<MprlEntry> ParsePm4Mprl(string pm4Path)
    {
        var entries = new List<MprlEntry>();

        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"[ERROR] PM4 not found: {pm4Path}");
            return entries;
        }

        try
        {
            using var fs = File.OpenRead(pm4Path);
            using var br = new BinaryReader(fs);

            // PM4 is chunked format - scan for MPRL chunk
            // Note: WoW stores FourCCs reversed, so MPRL is stored as "LRPM"
            while (fs.Position < fs.Length - 8)
            {
                var chunkIdBytes = br.ReadBytes(4);
                var chunkId = new string(new[] { (char)chunkIdBytes[3], (char)chunkIdBytes[2], (char)chunkIdBytes[1], (char)chunkIdBytes[0] });
                var chunkSize = br.ReadUInt32();

                if (chunkId == "MPRL")
                {
                    // MPRL entry is 24 bytes
                    const int entrySize = 24;
                    int entryCount = (int)(chunkSize / entrySize);

                    for (int i = 0; i < entryCount; i++)
                    {
                        var unk0x00 = br.ReadUInt16();
                        var unk0x02 = br.ReadInt16();
                        var unk0x04 = br.ReadUInt16();
                        var unk0x06 = br.ReadUInt16();
                        var posX = br.ReadSingle();
                        var posY = br.ReadSingle();
                        var posZ = br.ReadSingle();
                        var unk0x14 = br.ReadInt16();
                        var unk0x16 = br.ReadUInt16();

                        entries.Add(new MprlEntry(
                            i,
                            unk0x00,
                            unk0x02,
                            unk0x04,
                            unk0x06,
                            new Vector3(posX, posY, posZ),
                            unk0x14,
                            unk0x16));
                    }

                    Console.WriteLine($"[INFO] Parsed {entries.Count} MPRL entries from {Path.GetFileName(pm4Path)}");
                    break;
                }
                else
                {
                    // Skip this chunk
                    fs.Seek(chunkSize, SeekOrigin.Current);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to parse PM4: {ex.Message}");
        }

        return entries;
    }

    /// <summary>
    /// Correlate PM4 MPRL entries with MODF placements by position proximity.
    /// </summary>
    public List<CorrelationResult> Correlate(
        List<MprlEntry> mprlEntries,
        List<ModfPlacement> modfPlacements,
        float maxDistance = 100.0f)
    {
        var results = new List<CorrelationResult>();

        foreach (var mprl in mprlEntries)
        {
            ModfPlacement? bestMatch = null;
            int bestMatchIndex = -1;
            float bestDistance = float.MaxValue;

            for (int i = 0; i < modfPlacements.Count; i++)
            {
                var modf = modfPlacements[i];
                var distance = Vector3.Distance(mprl.Position, modf.Position);

                if (distance < bestDistance && distance <= maxDistance)
                {
                    bestDistance = distance;
                    bestMatch = modf;
                    bestMatchIndex = i;
                }
            }

            results.Add(new CorrelationResult(
                mprl.Index,
                mprl.Position,
                bestMatch != null ? bestMatchIndex : null,
                bestMatch?.UniqueId,
                bestMatch?.Position,
                bestMatch != null ? bestDistance : null,
                bestMatch != null ? $"NameId:{bestMatch.NameId}" : null));
        }

        return results;
    }

    /// <summary>
    /// Parse MSLK entries from PM4 file - scene graph/hierarchy nodes.
    /// </summary>
    public List<MslkEntry> ParsePm4Mslk(string pm4Path)
    {
        var entries = new List<MslkEntry>();

        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"[ERROR] PM4 not found: {pm4Path}");
            return entries;
        }

        try
        {
            using var fs = File.OpenRead(pm4Path);
            using var br = new BinaryReader(fs);

            while (fs.Position < fs.Length - 8)
            {
                var chunkIdBytes = br.ReadBytes(4);
                var chunkId = new string(new[] { (char)chunkIdBytes[3], (char)chunkIdBytes[2], (char)chunkIdBytes[1], (char)chunkIdBytes[0] });
                var chunkSize = br.ReadUInt32();

                if (chunkId == "MSLK")
                {
                    // MSLK entry is 20 bytes
                    const int entrySize = 20;
                    int entryCount = (int)(chunkSize / entrySize);

                    for (int i = 0; i < entryCount; i++)
                    {
                        var objTypeFlags = br.ReadByte();
                        var objSubtype = br.ReadByte();
                        var reserved = br.ReadUInt16();
                        var groupObjId = br.ReadUInt32();
                        
                        // Read 24-bit signed int for MspiFirstIndex
                        byte b1 = br.ReadByte();
                        byte b2 = br.ReadByte();
                        byte b3 = br.ReadByte();
                        int mspiFirst = b1 | (b2 << 8) | (b3 << 16);
                        if ((b3 & 0x80) != 0) mspiFirst |= unchecked((int)0xFF000000);
                        
                        var mspiCount = br.ReadByte();
                        var matColorId = br.ReadUInt32();
                        var refIndex = br.ReadUInt16();
                        var sysFlag = br.ReadUInt16();

                        entries.Add(new MslkEntry(
                            i,
                            objTypeFlags,
                            objSubtype,
                            reserved,
                            groupObjId,  // ParentIndex
                            mspiFirst,
                            mspiCount,
                            matColorId,  // LinkId (tile crossing)
                            refIndex,
                            sysFlag));
                    }

                    Console.WriteLine($"[INFO] Parsed {entries.Count} MSLK entries from {Path.GetFileName(pm4Path)}");
                    break;
                }
                else
                {
                    fs.Seek(chunkSize, SeekOrigin.Current);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to parse PM4 MSLK: {ex.Message}");
        }

        return entries;
    }

    /// <summary>
    /// Parse WMO names from _obj0.adt MWMO chunk.
    /// </summary>
    public List<string> ParseWmoNames(string obj0AdtPath)
    {
        var names = new List<string>();

        if (!File.Exists(obj0AdtPath))
            return names;

        try
        {
            using var fs = File.OpenRead(obj0AdtPath);
            using var br = new BinaryReader(fs);

            while (fs.Position < fs.Length - 8)
            {
                var chunkIdBytes = br.ReadBytes(4);
                var chunkId = new string(new[] { (char)chunkIdBytes[3], (char)chunkIdBytes[2], (char)chunkIdBytes[1], (char)chunkIdBytes[0] });
                var chunkSize = br.ReadUInt32();

                if (chunkId == "MWMO")
                {
                    // MWMO is null-terminated strings
                    var data = br.ReadBytes((int)chunkSize);
                    var current = new System.Text.StringBuilder();
                    foreach (var b in data)
                    {
                        if (b == 0)
                        {
                            if (current.Length > 0)
                            {
                                names.Add(current.ToString());
                                current.Clear();
                            }
                        }
                        else
                        {
                            current.Append((char)b);
                        }
                    }
                    break;
                }
                else
                {
                    fs.Seek(chunkSize, SeekOrigin.Current);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to parse MWMO: {ex.Message}");
        }

        return names;
    }

    /// <summary>
    /// Run full correlation analysis on a tile.
    /// </summary>
    public void AnalyzeTile(string pm4Path, string obj0AdtPath, string outputCsvPath)
    {
        Console.WriteLine($"[INFO] Analyzing tile correlation:");
        Console.WriteLine($"  PM4: {pm4Path}");
        Console.WriteLine($"  _obj0.adt: {obj0AdtPath}");

        var mprlEntries = ParsePm4Mprl(pm4Path);
        var mslkEntries = ParsePm4Mslk(pm4Path);
        var modfPlacements = ParseObj0Adt(obj0AdtPath);
        var wmoNames = ParseWmoNames(obj0AdtPath);

        if (wmoNames.Count > 0)
        {
            Console.WriteLine($"[INFO] WMO names in tile:");
            for (int i = 0; i < wmoNames.Count; i++)
            {
                Console.WriteLine($"  [{i}] {wmoNames[i]}");
            }
        }

        // Analyze MSLK hierarchy with MODF cross-reference
        if (mslkEntries.Count > 0)
        {
            AnalyzeMslkHierarchy(mslkEntries, Path.Combine(Path.GetDirectoryName(outputCsvPath) ?? ".", 
                Path.GetFileNameWithoutExtension(pm4Path) + "_mslk_hierarchy.csv"), modfPlacements);
        }

        if (mprlEntries.Count == 0)
        {
            Console.WriteLine("[WARN] No MPRL entries found in PM4");
            return;
        }

        if (modfPlacements.Count == 0)
        {
            Console.WriteLine("[WARN] No MODF placements found in _obj0.adt");
        }

        var correlations = Correlate(mprlEntries, modfPlacements);

        // Write CSV
        using var sw = new StreamWriter(outputCsvPath);
        sw.WriteLine("mprl_index,mprl_x,mprl_y,mprl_z,modf_index,modf_uniqueid,modf_x,modf_y,modf_z,distance,wmo_name_id");

        int matched = 0;
        int unmatched = 0;

        foreach (var c in correlations)
        {
            sw.WriteLine(string.Join(",",
                c.MprlIndex,
                c.MprlPosition.X.ToString("F3"),
                c.MprlPosition.Y.ToString("F3"),
                c.MprlPosition.Z.ToString("F3"),
                c.ModfIndex?.ToString() ?? "",
                c.ModfUniqueId?.ToString() ?? "",
                c.ModfPosition?.X.ToString("F3") ?? "",
                c.ModfPosition?.Y.ToString("F3") ?? "",
                c.ModfPosition?.Z.ToString("F3") ?? "",
                c.Distance?.ToString("F3") ?? "",
                c.WmoNameIndex ?? ""));

            if (c.ModfIndex.HasValue)
                matched++;
            else
                unmatched++;
        }

        Console.WriteLine($"[RESULT] Correlation complete:");
        Console.WriteLine($"  MPRL entries: {mprlEntries.Count}");
        Console.WriteLine($"  MODF placements: {modfPlacements.Count}");
        Console.WriteLine($"  Matched: {matched}");
        Console.WriteLine($"  Unmatched: {unmatched}");
        Console.WriteLine($"  Output: {outputCsvPath}");

        // Also dump MODF summary
        Console.WriteLine($"\n[MODF Summary]");
        for (int i = 0; i < modfPlacements.Count; i++)
        {
            var m = modfPlacements[i];
            Console.WriteLine($"  [{i}] NameId:{m.NameId} UniqueId:{m.UniqueId} Pos:({m.Position.X:F1},{m.Position.Y:F1},{m.Position.Z:F1})");
        }
    }

    /// <summary>
    /// Analyze MSLK hierarchy and output statistics.
    /// Cross-reference with MODF UniqueIds to find mapping patterns.
    /// </summary>
    private void AnalyzeMslkHierarchy(List<MslkEntry> entries, string outputCsvPath, 
        List<ModfPlacement>? modfPlacements = null)
    {
        Console.WriteLine($"\n[MSLK Hierarchy Analysis]");
        Console.WriteLine($"  Total entries: {entries.Count}");

        // Group by ParentIndex (object hierarchy grouping)
        var byParent = entries.GroupBy(e => e.ParentIndex).OrderBy(g => g.Key).ToList();
        Console.WriteLine($"  Unique ParentIndex values: {byParent.Count}");

        // Count entries with geometry (MspiFirstIndex >= 0)
        var withGeometry = entries.Where(e => e.MspiFirstIndex >= 0).ToList();
        var withoutGeometry = entries.Where(e => e.MspiFirstIndex < 0).ToList();
        Console.WriteLine($"  Entries with geometry (MspiFirstIndex >= 0): {withGeometry.Count}");
        Console.WriteLine($"  Entries without geometry (MspiFirstIndex < 0): {withoutGeometry.Count}");

        // Analyze ObjectTypeFlags distribution
        var byType = entries.GroupBy(e => e.ObjectTypeFlags).OrderBy(g => g.Key).ToList();
        Console.WriteLine($"  ObjectTypeFlags distribution:");
        foreach (var g in byType)
        {
            Console.WriteLine($"    Type {g.Key}: {g.Count()} entries");
        }

        // Analyze ObjectSubtype distribution
        var bySubtype = entries.GroupBy(e => e.ObjectSubtype).OrderBy(g => g.Key).ToList();
        Console.WriteLine($"  ObjectSubtype distribution:");
        foreach (var g in bySubtype)
        {
            Console.WriteLine($"    Subtype {g.Key}: {g.Count()} entries");
        }

        // Analyze LinkId (tile crossing) patterns
        Console.WriteLine($"\n  LinkId (Tile Crossing) Analysis:");
        var linkIds = entries.Select(e => e.LinkId).Distinct().OrderBy(x => x).ToList();
        Console.WriteLine($"    Unique LinkId values: {linkIds.Count}");
        Console.WriteLine($"    Sample values: {string.Join(", ", linkIds.Take(10).Select(x => $"0x{x:X8}"))}");
        
        // Decode tile links
        var tileLinks = entries
            .Select(e => e.DecodeTileLink())
            .Where(t => t.HasValue)
            .Select(t => t!.Value)
            .Distinct()
            .OrderBy(t => t.TileY).ThenBy(t => t.TileX)
            .ToList();
        Console.WriteLine($"    Decoded tile references: {tileLinks.Count}");
        foreach (var (tileX, tileY) in tileLinks.Take(10))
        {
            var count = entries.Count(e => e.DecodeTileLink() == (tileX, tileY));
            Console.WriteLine($"      Tile ({tileX}, {tileY}): {count} entries");
        }

        // Write detailed CSV
        using var sw = new StreamWriter(outputCsvPath);
        sw.WriteLine("index,obj_type_flags,obj_subtype,parent_index,mspi_first_index,mspi_index_count,link_id,link_tile_x,link_tile_y,reference_index,system_flag");

        foreach (var e in entries)
        {
            var tileLink = e.DecodeTileLink();
            sw.WriteLine(string.Join(",",
                e.Index,
                e.ObjectTypeFlags,
                e.ObjectSubtype,
                e.ParentIndex,
                e.MspiFirstIndex,
                e.MspiIndexCount,
                $"0x{e.LinkId:X8}",
                tileLink?.TileX.ToString() ?? "",
                tileLink?.TileY.ToString() ?? "",
                e.ReferenceIndex,
                $"0x{e.SystemFlag:X4}"));
        }

        Console.WriteLine($"  MSLK CSV: {outputCsvPath}");

        // Show parent index summary
        Console.WriteLine($"\n[MSLK ParentIndex Summary]");
        foreach (var g in byParent.Take(20)) // Show first 20 groups
        {
            var geoCount = g.Count(e => e.MspiFirstIndex >= 0);
            var totalMspi = g.Where(e => e.MspiFirstIndex >= 0).Sum(e => e.MspiIndexCount);
            Console.WriteLine($"  ParentIndex {g.Key}: {g.Count()} entries, {geoCount} with geometry, {totalMspi} total MSPI indices");
        }
        if (byParent.Count > 20)
        {
            Console.WriteLine($"  ... and {byParent.Count - 20} more groups");
        }

        // Cross-reference with MODF UniqueIds
        if (modfPlacements != null && modfPlacements.Count > 0)
        {
            Console.WriteLine($"\n[MSLK ↔ MODF Cross-Reference Analysis]");
            
            var modfUniqueIds = modfPlacements.Select(m => (uint)m.UniqueId).ToHashSet();
            Console.WriteLine($"  MODF UniqueIds: {string.Join(", ", modfUniqueIds.OrderBy(x => x).Take(10))}{(modfUniqueIds.Count > 10 ? "..." : "")}");
            Console.WriteLine($"  MODF UniqueId range: {modfUniqueIds.Min()} - {modfUniqueIds.Max()}");

            // Check if any ParentIndex values match MODF UniqueIds directly
            var parentIds = entries.Select(e => e.ParentIndex).Distinct().ToHashSet();
            var directMatches = parentIds.Intersect(modfUniqueIds).ToList();
            Console.WriteLine($"  Direct ParentIndex ↔ UniqueId matches: {directMatches.Count}");
            if (directMatches.Count > 0)
            {
                Console.WriteLine($"    Matching IDs: {string.Join(", ", directMatches.OrderBy(x => x).Take(20))}");
            }

            // Check ReferenceIndex matches
            var refIndices = entries.Select(e => (uint)e.ReferenceIndex).Distinct().ToHashSet();
            var refMatches = refIndices.Intersect(modfUniqueIds).ToList();
            Console.WriteLine($"  ReferenceIndex ↔ UniqueId matches: {refMatches.Count}");

            // Check LinkId for UniqueId encoding (unlikely but check)
            var linkIdMatches = entries.Where(e => modfUniqueIds.Contains(e.LinkId)).ToList();
            Console.WriteLine($"  LinkId ↔ UniqueId matches: {linkIdMatches.Count}");

            // Composite key analysis - try different field combinations
            Console.WriteLine($"\n[Composite Key Analysis]");
            
            // Key: (ObjectTypeFlags, ObjectSubtype, ParentIndex)
            var compositeKeys = entries
                .GroupBy(e => (e.ObjectTypeFlags, e.ObjectSubtype, e.ParentIndex))
                .OrderByDescending(g => g.Count())
                .Take(20)
                .ToList();
            Console.WriteLine($"  Unique (Type, Subtype, ParentIndex) combinations: {entries.GroupBy(e => (e.ObjectTypeFlags, e.ObjectSubtype, e.ParentIndex)).Count()}");
            Console.WriteLine($"  Top composite keys by entry count:");
            foreach (var ck in compositeKeys.Take(10))
            {
                var geoCount = ck.Count(e => e.MspiFirstIndex >= 0);
                var totalMspi = ck.Where(e => e.MspiFirstIndex >= 0).Sum(e => e.MspiIndexCount);
                Console.WriteLine($"    ({ck.Key.ObjectTypeFlags}, {ck.Key.ObjectSubtype}, {ck.Key.ParentIndex}): {ck.Count()} entries, {geoCount} geo, {totalMspi} MSPI");
            }

            // Look for Type 1 entries (container nodes) that might group objects
            var type1Entries = entries.Where(e => e.ObjectTypeFlags == 1).ToList();
            Console.WriteLine($"\n  Type 1 (container) entries: {type1Entries.Count}");
            Console.WriteLine($"  Type 1 unique ParentIndex values: {type1Entries.Select(e => e.ParentIndex).Distinct().Count()}");
            
            // Check if Type 1 ParentIndex values correlate with MODF count
            Console.WriteLine($"  MODF placement count: {modfPlacements.Count}");
            Console.WriteLine($"  Ratio MSLK ParentIndex groups / MODF placements: {(float)byParent.Count / modfPlacements.Count:F2}");

            // CK24 analysis from PM4 spec: (CompositeKey & 0xFFFFFF00) >> 8
            Console.WriteLine($"\n[CK24 Composite Key Analysis (from PM4 spec)]");
            var ck24Values = entries
                .Select(e => (e.ParentIndex & 0xFFFFFF00) >> 8)
                .Distinct()
                .OrderBy(x => x)
                .ToList();
            Console.WriteLine($"  Unique CK24 values: {ck24Values.Count}");
            Console.WriteLine($"  Sample CK24: {string.Join(", ", ck24Values.Take(10))}");
            
            // Check if CK24 matches MODF UniqueIds
            var ck24Matches = ck24Values.Where(ck => modfUniqueIds.Contains(ck)).ToList();
            Console.WriteLine($"  CK24 ↔ UniqueId matches: {ck24Matches.Count}");
        }
    }

    /// <summary>
    /// Analyze PM4 file only (without _obj0.adt).
    /// </summary>
    public void AnalyzePm4Only(string pm4Path, string outputDir)
    {
        Console.WriteLine($"[INFO] Analyzing PM4 file:");
        Console.WriteLine($"  PM4: {pm4Path}");

        Directory.CreateDirectory(outputDir);
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);

        var mprlEntries = ParsePm4Mprl(pm4Path);
        var mslkEntries = ParsePm4Mslk(pm4Path);

        Console.WriteLine($"\n[PM4 Summary]");
        Console.WriteLine($"  MPRL entries: {mprlEntries.Count}");
        Console.WriteLine($"  MSLK entries: {mslkEntries.Count}");

        // Analyze MSLK hierarchy (without MODF cross-reference)
        if (mslkEntries.Count > 0)
        {
            AnalyzeMslkHierarchy(mslkEntries, Path.Combine(outputDir, $"{baseName}_mslk.csv"), null);
        }

        // Analyze tile references in MSLK
        Console.WriteLine($"\n[Tile Reference Analysis]");
        var tileRefs = mslkEntries
            .Select(e => e.DecodeTileLink())
            .Where(t => t.HasValue)
            .Select(t => t!.Value)
            .GroupBy(t => t)
            .OrderBy(g => g.Key.TileY).ThenBy(g => g.Key.TileX)
            .ToList();

        Console.WriteLine($"  Referenced tiles: {tileRefs.Count}");
        foreach (var g in tileRefs)
        {
            Console.WriteLine($"    Tile ({g.Key.TileX}, {g.Key.TileY}): {g.Count()} entries");
        }

        // Output MPRL positions
        if (mprlEntries.Count > 0)
        {
            var mprlCsv = Path.Combine(outputDir, $"{baseName}_mprl.csv");
            using var sw = new StreamWriter(mprlCsv);
            sw.WriteLine("index,x,y,z,unk0,unk2,unk4,unk6,unk14,unk16");
            foreach (var e in mprlEntries)
            {
                sw.WriteLine(string.Join(",",
                    e.Index,
                    e.Position.X.ToString("F3"),
                    e.Position.Y.ToString("F3"),
                    e.Position.Z.ToString("F3"),
                    e.Unknown0x00,
                    e.Unknown0x02,
                    e.Unknown0x04,
                    e.Unknown0x06,
                    e.Unknown0x14,
                    e.Unknown0x16));
            }
            Console.WriteLine($"  MPRL CSV: {mprlCsv}");
        }
    }
}
