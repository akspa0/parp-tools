using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using Warcraft.NET.Files.ADT.Entries;
using Warcraft.NET.Files.ADT.Chunks;
using Warcraft.NET.Types;

namespace WoWRollback.PM4Module;

/// <summary>
/// Patches PM4 placement data into existing WoWMuseum ADTs.
/// Uses Warcraft.NET to properly handle chunk ordering and offset calculations.
/// </summary>
public sealed class Pm4AdtPatcher
{
    /// <summary>
    /// Patch WMO placements from PM4 reconstruction into an existing ADT file.
    /// Adds WMO names to MWMO and placements to MODF.
    /// </summary>
    /// <param name="inputAdtPath">Path to input ADT (WoWMuseum format)</param>
    /// <param name="outputAdtPath">Path to write patched ADT</param>
    /// <param name="wmoNames">WMO paths to add to MWMO chunk</param>
    /// <param name="modfEntries">MODF placement entries (NameId references index in wmoNames)</param>
    public void PatchWithWmoData(
        string inputAdtPath,
        string outputAdtPath,
        List<string> wmoNames,
        List<MODFEntry> modfEntries)
    {
        Console.WriteLine($"[INFO] Loading ADT: {inputAdtPath}");
        
        var bytes = File.ReadAllBytes(inputAdtPath);
        var adt = new Terrain(bytes);
        
        int origWmoCount = adt.WorldModelObjects?.Filenames?.Count ?? 0;
        int origModfCount = adt.WorldModelObjectPlacementInfo?.MODFEntries?.Count ?? 0;
        
        // Initialize MWMO if needed
        if (adt.WorldModelObjects == null)
            adt.WorldModelObjects = new MWMO();
        adt.WorldModelObjects.Filenames ??= new List<string>();
        
        // Initialize MWID if needed (Warcraft.NET may handle this automatically)
        if (adt.WorldModelObjectIndices == null)
            adt.WorldModelObjectIndices = new MWID();
        adt.WorldModelObjectIndices.ModelFilenameOffsets ??= new List<uint>();
        
        // Add WMO names - track offset for each
        int baseNameId = adt.WorldModelObjects.Filenames.Count;
        uint currentOffset = 0;
        if (adt.WorldModelObjects.Filenames.Count > 0)
        {
            // Calculate current end offset
            foreach (var name in adt.WorldModelObjects.Filenames)
                currentOffset += (uint)(name.Length + 1); // +1 for null terminator
        }
        
        if (wmoNames != null && wmoNames.Count > 0)
        {
            foreach (var wmoName in wmoNames)
            {
                adt.WorldModelObjects.Filenames.Add(wmoName);
                adt.WorldModelObjectIndices.ModelFilenameOffsets.Add(currentOffset);
                currentOffset += (uint)(wmoName.Length + 1);
            }
        }
        
        // Initialize MODF if needed
        if (adt.WorldModelObjectPlacementInfo == null)
            adt.WorldModelObjectPlacementInfo = new MODF();
        adt.WorldModelObjectPlacementInfo.MODFEntries ??= new List<MODFEntry>();
        
        // Add MODF entries - adjust NameId to account for existing WMOs
        foreach (var entry in modfEntries)
        {
            // Create new entry with adjusted NameId
            var adjustedEntry = new MODFEntry
            {
                NameId = (uint)(baseNameId + entry.NameId),
                UniqueId = entry.UniqueId,
                Position = entry.Position,
                Rotation = entry.Rotation,
                BoundingBox = entry.BoundingBox,
                Flags = entry.Flags,
                DoodadSet = entry.DoodadSet,
                NameSet = entry.NameSet,
                Scale = entry.Scale
            };
            adt.WorldModelObjectPlacementInfo.MODFEntries.Add(adjustedEntry);
        }
        
        Console.WriteLine($"[INFO] Added {wmoNames.Count} WMO names (was {origWmoCount})");
        Console.WriteLine($"[INFO] Added {modfEntries.Count} MODF entries (was {origModfCount})");
        
        // Serialize and save - Warcraft.NET handles all offset calculations
        var outputBytes = adt.Serialize();
        
        Directory.CreateDirectory(Path.GetDirectoryName(outputAdtPath)!);
        File.WriteAllBytes(outputAdtPath, outputBytes);
        
        Console.WriteLine($"[INFO] Wrote patched ADT: {outputAdtPath} ({outputBytes.Length:N0} bytes)");
    }
    
    /// <summary>
    /// Patch PM4 placements into an existing ADT file.
    /// </summary>
    /// <param name="inputAdtPath">Path to input ADT (WoWMuseum format)</param>
    /// <param name="outputAdtPath">Path to write patched ADT</param>
    /// <param name="mddfEntries">MDDF entries to add (M2 doodads)</param>
    /// <param name="modfEntries">MODF entries to add (WMOs)</param>
    public void PatchAdt(
        string inputAdtPath,
        string outputAdtPath,
        List<MDDFEntry>? mddfEntries = null,
        List<MODFEntry>? modfEntries = null)
    {
        Console.WriteLine($"[INFO] Loading ADT: {inputAdtPath}");
        
        var bytes = File.ReadAllBytes(inputAdtPath);
        var adt = new Terrain(bytes);
        
        Console.WriteLine($"[INFO] Original ADT:");
        Console.WriteLine($"  MDDF entries: {adt.ModelPlacementInfo?.MDDFEntries?.Count ?? 0}");
        Console.WriteLine($"  MODF entries: {adt.WorldModelObjectPlacementInfo?.MODFEntries?.Count ?? 0}");
        Console.WriteLine($"  Models (MMDX): {adt.Models?.Filenames?.Count ?? 0}");
        Console.WriteLine($"  WMOs (MWMO): {adt.WorldModelObjects?.Filenames?.Count ?? 0}");
        
        // Add MDDF entries
        if (mddfEntries != null && mddfEntries.Count > 0)
        {
            if (adt.ModelPlacementInfo == null)
                adt.ModelPlacementInfo = new MDDF();
            
            adt.ModelPlacementInfo.MDDFEntries ??= new List<MDDFEntry>();
            adt.ModelPlacementInfo.MDDFEntries.AddRange(mddfEntries);
            Console.WriteLine($"[INFO] Added {mddfEntries.Count} MDDF entries");
        }
        
        // Add MODF entries
        if (modfEntries != null && modfEntries.Count > 0)
        {
            if (adt.WorldModelObjectPlacementInfo == null)
                adt.WorldModelObjectPlacementInfo = new MODF();
            
            adt.WorldModelObjectPlacementInfo.MODFEntries ??= new List<MODFEntry>();
            adt.WorldModelObjectPlacementInfo.MODFEntries.AddRange(modfEntries);
            Console.WriteLine($"[INFO] Added {modfEntries.Count} MODF entries");
        }
        
        // Serialize and save
        var outputBytes = adt.Serialize();
        
        Directory.CreateDirectory(Path.GetDirectoryName(outputAdtPath)!);
        File.WriteAllBytes(outputAdtPath, outputBytes);
        
        Console.WriteLine($"[INFO] Wrote patched ADT: {outputAdtPath} ({outputBytes.Length:N0} bytes)");
    }
    
    /// <summary>
    /// Simple test: load ADT, serialize it back, verify it's still valid.
    /// </summary>
    public bool TestRoundtrip(string inputAdtPath, string outputAdtPath)
    {
        Console.WriteLine($"[TEST] Roundtrip test: {Path.GetFileName(inputAdtPath)}");
        
        try
        {
            var bytes = File.ReadAllBytes(inputAdtPath);
            Console.WriteLine($"  Input size: {bytes.Length:N0} bytes");
            
            var adt = new Terrain(bytes);
            Console.WriteLine($"  Parsed OK - {adt.Chunks?.Length ?? 0} MCNKs");
            
            var outputBytes = adt.Serialize();
            Console.WriteLine($"  Serialized size: {outputBytes.Length:N0} bytes");
            
            Directory.CreateDirectory(Path.GetDirectoryName(outputAdtPath)!);
            File.WriteAllBytes(outputAdtPath, outputBytes);
            
            // Verify we can re-read it
            var verifyBytes = File.ReadAllBytes(outputAdtPath);
            var verifyAdt = new Terrain(verifyBytes);
            Console.WriteLine($"  Verify OK - {verifyAdt.Chunks?.Length ?? 0} MCNKs");
            
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  [FAIL] {ex.Message}");
            return false;
        }
    }
}
