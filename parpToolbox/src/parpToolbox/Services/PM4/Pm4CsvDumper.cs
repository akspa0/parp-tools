namespace ParpToolbox.Services.PM4;

using System;
using System.Globalization;
using System.IO;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Dedicated CSV export service for PM4 chunk data analysis.
/// Exports all chunk types to separate CSV files for field investigation.
/// </summary>
internal static class Pm4CsvDumper
{
    public static void DumpAllChunks(Pm4Scene scene, string outputRoot)
    {
        // root path for this run: csv_dump_YYYYMMDD_HHmmss
        string runRoot = Path.Combine(outputRoot, $"csv_dump_{DateTime.Now:yyyyMMdd_HHmmss}");
        Directory.CreateDirectory(runRoot);
        
        // For each chunk type, make a dedicated sub-folder
        DumpMslkChunk(scene, Path.Combine(runRoot, "MSLK"));
        DumpMsurChunk(scene, Path.Combine(runRoot, "MSUR"));
        if (scene.Properties.Count > 0)
            DumpMprrChunk(scene, Path.Combine(runRoot, "MPRR"));
        DumpChunkStats(scene, runRoot);
        
        Console.WriteLine($"CSV dump complete! Files written under: {runRoot}");
    }
    
    private static void DumpMslkChunk(Pm4Scene scene, string outputRoot)
    {
        Directory.CreateDirectory(outputRoot);
        var csvPath = Path.Combine(outputRoot, "mslk_dump.csv");
        using var writer = new StreamWriter(csvPath);
        
        writer.WriteLine("Idx,Unknown0x00,Unknown0x01,Unknown0x02,Unknown0x04,MspiFirstIndex,MspiIndexCount,LinkIdRaw,HasValidTiles,TileY,TileX,Unknown0x10,Unknown0x12");
        
        int idx = 0;
        foreach (var link in scene.Links)
        {
            // Use legacy tile decoding logic
            bool hasValidTiles = link.TryDecodeTileCoordinates(out int tileX, out int tileY);
            
            writer.WriteLine(string.Join(',',
                idx++,
                $"0x{link.Unknown_0x00:X2}",
                $"0x{link.Unknown_0x01:X2}",
                $"0x{link.Unknown_0x02:X4}",
                $"0x{link.Unknown_0x04:X8}",
                link.MspiFirstIndex,
                link.MspiIndexCount,
                $"0x{link.LinkIdRaw:X8}",
                hasValidTiles,
                tileY,
                tileX,
                $"0x{link.Unknown_0x10:X4}",
                $"0x{link.Unknown_0x12:X4}"));
        }
        
        Console.WriteLine($"MSLK: {scene.Links.Count} entries -> {csvPath}");
    }
    
    private static void DumpMsurChunk(Pm4Scene scene, string outputRoot)
    {
        Directory.CreateDirectory(outputRoot);
        var csvPath = Path.Combine(outputRoot, "msur_dump.csv");
        using var writer = new StreamWriter(csvPath);
        
        writer.WriteLine("Idx,SurfaceKey,SurfaceKeyHi16,SurfaceKeyLo16,SurfaceGroupKey,Flags0x00,AttributeMask,MsviFirstIndex,IndexCount,IsM2Bucket,IsLiquidCandidate,Nx,Ny,Nz,Height,MdosIndex");
        
        int idx = 0;
        foreach (var surf in scene.Surfaces)
        {
            writer.WriteLine(string.Join(',',
                idx++,
                $"0x{surf.SurfaceKey:X8}",
                $"0x{surf.SurfaceKeyHigh16:X4}",
                $"0x{surf.SurfaceKeyLow16:X4}",
                surf.SurfaceGroupKey,
                $"0x{surf.FlagsOrUnknown_0x00:X2}",
                $"0x{surf.SurfaceAttributeMask:X2}",
                surf.MsviFirstIndex,
                surf.IndexCount,
                surf.IsM2Bucket,
                surf.IsLiquidCandidate,
                surf.Nx.ToString("F6", CultureInfo.InvariantCulture),
                surf.Ny.ToString("F6", CultureInfo.InvariantCulture),
                surf.Nz.ToString("F6", CultureInfo.InvariantCulture),
                surf.Height.ToString("F6", CultureInfo.InvariantCulture),
                surf.MdosIndex));
        }
        
        Console.WriteLine($"MSUR: {scene.Surfaces.Count} entries -> {csvPath}");
    }
    

    
    private static void DumpMprrChunk(Pm4Scene scene, string outputRoot)
    {
        Directory.CreateDirectory(outputRoot);
        var csvPath = Path.Combine(outputRoot, "mprr_dump.csv");
        using var writer = new StreamWriter(csvPath);
        
        writer.WriteLine("Idx,Value1,Value2");
        
        int idx = 0;
        foreach (var propRef in scene.Properties)
        {
            writer.WriteLine(string.Join(',',
                idx++,
                propRef.Value1,
                propRef.Value2));
        }
        
        Console.WriteLine($"MPRR: {scene.Properties.Count} entries -> {csvPath}");
    }
    
    private static void DumpChunkStats(Pm4Scene scene, string outputRoot)
    {
        Directory.CreateDirectory(outputRoot);
        var csvPath = Path.Combine(outputRoot, "chunk_stats.csv");
        using var writer = new StreamWriter(csvPath);
        
        writer.WriteLine("ChunkType,Count,Description");
        writer.WriteLine($"MSVT,{scene.Vertices.Count},Render vertices");
        writer.WriteLine($"MSVI,{scene.Indices.Count},Render indices");
        writer.WriteLine($"MSLK,{scene.Links.Count},Link entries");
        writer.WriteLine($"MSUR,{scene.Surfaces.Count},Surface definitions");
        writer.WriteLine($"MPRR,{scene.Properties.Count},Property references");
        
        Console.WriteLine($"Stats: 5 chunk types -> {csvPath}");
    }
}
