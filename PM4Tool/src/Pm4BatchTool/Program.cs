using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using WoWToolbox.Core.v2.Services.PM4;
using WoWToolbox.Core.v2.Infrastructure;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.PM4;

// Simple CLI front-end for Pm4BatchProcessor. Usage:
//   dotnet Pm4BatchTool.dll <pm4-file-or-directory> [--wmo <wmoDataDir>]
// Outputs OBJ + summary to ProjectOutput/pm4.

class Program
{
    static async Task<int> Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: Pm4BatchTool <pm4-file-or-directory> [--wmo <wmoDataDir>] [--diag] [--force] [--debug-chunks]\n" +
                              "       Pm4BatchTool terrain-mesh  <pm4-root-dir> <output-obj-path> [--no-stitch] [--tile <x> <y>]\n" +
                              "       Pm4BatchTool terrain-stamp <pm4-root-dir> <output-obj-path>");
            return 1;
        }

        // Custom command: mscn-ranges (debug axis mapping)
        if (string.Equals(args[0], "mscn-ranges", StringComparison.OrdinalIgnoreCase))
        {
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Usage: Pm4BatchTool mscn-ranges <pm4-path>");
                return 1;
            }
            string pm4Path = args[1];
            try
            {
                DumpMscnRanges(pm4Path);
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }
        }

        // Custom command: terrain-mesh
        if (string.Equals(args[0], "terrain-mesh", StringComparison.OrdinalIgnoreCase))
        {
            if (args.Length < 3)
            {
                Console.Error.WriteLine("Usage: Pm4BatchTool terrain-mesh <pm4-root-dir> <output-obj-path> [--no-stitch] [--tile <x> <y>]");
                return 1;
            }
            string rootDir = args[1];
            string outObj = args[2];
            if (!Path.IsPathRooted(outObj))
            {
                outObj = ProjectOutput.GetPath("analysis", outObj);
            }
            bool stitch = true;
            int? tileX = null;
            int? tileY = null;
            for (int i = 3; i < args.Length; i++)
            {
                if (string.Equals(args[i], "--no-stitch", StringComparison.OrdinalIgnoreCase))
                {
                    stitch = false;
                }
                else if (string.Equals(args[i], "--tile", StringComparison.OrdinalIgnoreCase) && i + 2 < args.Length && int.TryParse(args[i + 1], out int tx) && int.TryParse(args[i + 2], out int ty))
                {
                    tileX = tx;
                    tileY = ty;
                    i += 2;
                }
            }
            try
            {
                WoWToolbox.Core.v2.Services.PM4.TerrainMeshExporter.Export(rootDir, outObj, stitch, tileX, tileY);
                Console.WriteLine($"Terrain mesh OBJ written to {outObj}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }
        }

        // Custom command: terrain-stamp
        if (string.Equals(args[0], "terrain-stamp", StringComparison.OrdinalIgnoreCase))
        {
            if (args.Length < 3)
            {
                Console.Error.WriteLine("Usage: Pm4BatchTool terrain-stamp <pm4-root-dir> <output-obj-path> [--tile <x> <y>]");
                return 1;
            }
            string rootDir = args[1];
            string outObj = args[2];
            int? tileX = null;
            int? tileY = null;
            bool includePlate = true;
            bool includeStamp = true;
            for (int i=3;i<args.Length;i++)
            {
                if (string.Equals(args[i],"--tile",StringComparison.OrdinalIgnoreCase) && i+2<args.Length && int.TryParse(args[i+1],out int tx) && int.TryParse(args[i+2],out int ty))
                {
                    tileX = tx; tileY = ty; i+=2;
                }
                else if (string.Equals(args[i],"--no-plate",StringComparison.OrdinalIgnoreCase))
                {
                    includePlate = false;
                }
                else if (string.Equals(args[i],"--no-stamp",StringComparison.OrdinalIgnoreCase))
                {
                    includeStamp = false;
                }
            }
            try
            {
                WoWToolbox.Core.v2.Services.PM4.TerrainStampExporter.Export(rootDir, outObj, tileX, tileY, includePlate, includeStamp);
                Console.WriteLine($"Terrain stamp OBJ written to {outObj}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }
        }

        string inputPath = args[0];
        string? wmoDir = null;
        bool dumpDiag = args.Contains("--diag");
        bool force = args.Contains("--force");
        bool debugChunks = args.Contains("--debug-chunks");
        bool chunkQuilts = args.Contains("--chunk-quilts");
        bool masterQuilt = args.Contains("--master-quilt");
        bool analyzeMslk = args.Contains("--analyze-mslk");
        bool mergePlate = args.Contains("--with-plate");
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--wmo" when i + 1 < args.Length:
                    wmoDir = args[i + 1];
                    i++;
                    break;
                case "--chunk-quilts":
                    chunkQuilts = true;
                    break;
                case "--master-quilt":
                    masterQuilt = true;
                    break;
                case "--with-plate":
                    mergePlate = true;
                    break;
                case "--analyze-mslk":
                        analyzeMslk = true;
                        break;
                    case "--diag":
                    dumpDiag = true;
                    break;
            }
        }

        if (!File.Exists(inputPath) && !Directory.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input path '{inputPath}' does not exist.");
            return 1;
        }

        // Direct instantiation to avoid DI complexity
        var coordinateService = new WoWToolbox.Core.v2.Services.PM4.CoordinateService();
        IBuildingExtractionService extraction = new WoWToolbox.Core.v2.Services.PM4.PM4BuildingExtractionService(coordinateService);
        WoWToolbox.Core.v2.Services.PM4.IWmoMatcher matcher = new WoWToolbox.Core.v2.Services.PM4.WmoMatcher(wmoDir ?? Path.Combine(Environment.CurrentDirectory, "wmo_data"));
        WoWToolbox.Core.v2.Services.PM4.IPm4BatchProcessor processor = new WoWToolbox.Core.v2.Services.PM4.Pm4BatchProcessor(extraction, matcher);

        if (File.Exists(inputPath))
        {
            if (force)
            {
                string stem = Path.GetFileNameWithoutExtension(inputPath);
                string dir = Path.Combine(processor.RunDirectory, stem);
                if (Directory.Exists(dir))
                {
                    try { Directory.Delete(dir, true); } catch { }
                }
            }

            if (dumpDiag)
            {
                DumpDiagnostics(inputPath);
            }
            var res = processor.Process(inputPath);
            Console.WriteLine(res.Success ? "Success" : $"Failed: {res.ErrorMessage}");

            // Merge with white plate if requested
            if (res.Success && mergePlate)
            {
                string stem = Path.GetFileNameWithoutExtension(inputPath);
                if (TryParseTileCoords(stem, out int tx, out int ty))
                {
                    string objPath = Path.Combine(processor.RunDirectory, stem, stem + ".obj");
                    string merged = Path.Combine(processor.RunDirectory, stem, stem + "_merged.obj");
                    try
                    {
                        TileMeshMerger.MergeWithPlate(objPath, tx, ty, merged);
                        Console.WriteLine($"  ↳ merged plate OBJ → {merged}");
                    }
                    catch (Exception ex)
                    {
                        Console.Error.WriteLine($"  ↳ merge failed: {ex.Message}");
                    }
                }
            }

            // Export debug chunks if requested
            if (debugChunks)
            {
                try
                {
                    // Reload PM4 (processor already exported OBJ inside Process)
                    var pm4ForDebug = PM4File.FromFile(inputPath);
                    string outputDir = Path.Combine(processor.RunDirectory, Path.GetFileNameWithoutExtension(inputPath));
                    string basePath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputPath));
                    await ChunkDebugExporter.ExportChunksAsync(pm4ForDebug, basePath);
                    Console.WriteLine($"  Exported debug chunks to: {basePath}_*.obj");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Error exporting debug chunks: {ex.Message}");
                }
            }
        }
        else
        {
            var pm4Files = Directory.EnumerateFiles(inputPath, "*.pm4", SearchOption.AllDirectories);
            int ok = 0;
            int fail = 0;
            var processed = new List<(PM4File file,string name)>();
            foreach (var file in pm4Files)
            {
                if (force)
                {
                    // remove previous output folder if it exists
                    string stem = Path.GetFileNameWithoutExtension(file);
                    string dir = Path.Combine(processor.RunDirectory, stem);
                    if (Directory.Exists(dir))
                    {
                        try { Directory.Delete(dir, true); } catch { /* ignore */ }
                    }
                }

                if (dumpDiag)
                {
                    DumpDiagnostics(file);
                }

                var r = processor.Process(file);
                if (r.Success)
                {
                    ok++;
                    if (mergePlate)
                    {
                        string stem = Path.GetFileNameWithoutExtension(file);
                        if (TryParseTileCoords(stem, out int tx2, out int ty2))
                        {
                            string objPath2 = Path.Combine(processor.RunDirectory, stem, stem + ".obj");
                            string merged2 = Path.Combine(processor.RunDirectory, stem, stem + "_merged.obj");
                            try
                            {
                                TileMeshMerger.MergeWithPlate(objPath2, tx2, ty2, merged2);
                                Console.WriteLine($"  ↳ merged plate OBJ → {merged2}");
                            }
                            catch (Exception ex)
                            {
                                Console.Error.WriteLine($"  ↳ merge failed: {ex.Message}");
                            }
                        }
                    }
                    try { processed.Add((PM4File.FromFile(file), Path.GetFileNameWithoutExtension(file))); } catch {}                }
                else
                {
                    fail++;
                    Console.WriteLine($"Failed {file}: {r.ErrorMessage}");
                }
            }
            Console.WriteLine($"Processed {ok + fail} files: {ok} succeeded, {fail} failed.");

            string analysisDir = Path.Combine(Environment.CurrentDirectory, "project_output", "analysis");
            Directory.CreateDirectory(analysisDir);

            // Optional: export per-chunk quilts
            if (chunkQuilts)
            {
                try
                {
                    WoWToolbox.Core.v2.Services.PM4.TileQuiltChunkExporter.ExportChunkQuilts(inputPath, analysisDir);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Error exporting chunk quilts: {ex.Message}");
                }
            }

            // Optional: export master quilt OBJ
            if (masterQuilt)
            {
                try
                {
                    string masterPath = Path.Combine(analysisDir, "master_quilt.obj");
                    WoWToolbox.Core.v2.Services.PM4.UnifiedQuiltExporter.Export(inputPath, masterPath);
                    Console.WriteLine($"  ↳ Master quilt OBJ → {masterPath}");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Error exporting master quilt: {ex.Message}");
                }
            }

            if (processed.Count > 0)
            {
                // Cross-tile link graph (existing logic)
                var graph = WoWToolbox.Core.v2.Services.PM4.MslkLinkGraphBuilder.Build(processed);
                string linkCsv = Path.Combine(analysisDir, "mslk_links.csv");
                using (var sw = new StreamWriter(linkCsv))
                {
                    sw.WriteLine("LinkIdHex,TileCount,EntryCount,TileList");
                    foreach (var kv in graph.OrderBy(k => k.Key))
                    {
                        var tilesStr = string.Join(";", kv.Value.Select(e => $"{e.TileX},{e.TileY}"));
                        sw.WriteLine($"0x{kv.Key:X8},{kv.Value.Select(e => (e.TileX,e.TileY)).Distinct().Count()},{kv.Value.Count},{tilesStr}");
                    }
                }
                Console.WriteLine($"  ↳ MSLK link graph → {linkCsv}");

                // Per-tile structure audit (existing)
                var auditRows = new List<WoWToolbox.Core.v2.Services.PM4.MslkStructureAuditor.AuditRow>();
                foreach (var (pm4, name) in processed)
                    auditRows.AddRange(WoWToolbox.Core.v2.Services.PM4.MslkStructureAuditor.Audit(pm4, name));
                var auditCsv = Path.Combine(analysisDir, "mslk_structure_audit.csv");
                WoWToolbox.Core.v2.Services.PM4.MslkStructureAuditor.WriteCsv(auditRows, auditCsv);
                Console.WriteLine($"  ↳ MSLK structure audit → {auditCsv}");

                // New: inter-tile CSV listing every entry with src/dest tile mapping
                if (analyzeMslk)
                {
                    var rows = WoWToolbox.Core.v2.Services.PM4.MslkInterTileAnalyzer.Build(processed);
                    var interCsv = Path.Combine(analysisDir, "mslk_inter_tile.csv");
                    WoWToolbox.Core.v2.Services.PM4.MslkInterTileAnalyzer.WriteCsv(rows, interCsv);
                    Console.WriteLine($"  ↳ MSLK inter-tile analysis → {interCsv}");

                    // Step 1 summary: group-level pattern statistics
                    var summaryRows = WoWToolbox.Core.v2.Services.PM4.MslkPatternAnalyzer.Build(processed);
                    var summaryCsv = Path.Combine(analysisDir, "mslk_patterns.csv");
                    WoWToolbox.Core.v2.Services.PM4.MslkPatternAnalyzer.WriteCsv(summaryRows, summaryCsv);
                    Console.WriteLine($"  ↳ MSLK pattern summary → {summaryCsv}");
                }
            }


        }

        return 0;

        // Local function to dump diagnostic CSV for a single PM4
        void DumpMscnRanges(string pm4Path)
        {
            var pm4 = PM4File.FromFile(pm4Path);
            var verts = pm4.MSCN?.ExteriorVertices;
            if (verts == null || verts.Count == 0)
            {
                Console.WriteLine("No MSCN vertices found.");
                return;
            }
            var maps = new (string label, Func<Vector3, Vector3> map)[]
            {
                ("raw (X,Y,Z)",           (Vector3 v)=> new Vector3(v.X, v.Y, v.Z)),
                ("swapXY",                (Vector3 v)=> new Vector3(v.Y, v.X, v.Z)),
                ("(Y,-X,Z)",              (Vector3 v)=> new Vector3(v.Y, -v.X, v.Z)),
                ("(-Y,X,Z)",              (Vector3 v)=> new Vector3(-v.Y, v.X, v.Z)),
                ("(X,-Y,Z)",              (Vector3 v)=> new Vector3(v.X, -v.Y, v.Z)),
                ("(Z,X,Y)",               (Vector3 v)=> new Vector3(v.Z, v.X, v.Y)),
                ("(Z,Y,X)",               (Vector3 v)=> new Vector3(v.Z, v.Y, v.X))
            };
            foreach (var (label, mapFn) in maps)
            {
                float minX=float.MaxValue,minY=float.MaxValue,minZ=float.MaxValue;
                float maxX=float.MinValue,maxY=float.MinValue,maxZ=float.MinValue;
                foreach (var v in verts)
                {
                    var m = mapFn(v);
                    minX = MathF.Min(minX, m.X); maxX = MathF.Max(maxX, m.X);
                    minY = MathF.Min(minY, m.Y); maxY = MathF.Max(maxY, m.Y);
                    minZ = MathF.Min(minZ, m.Z); maxZ = MathF.Max(maxZ, m.Z);
                }
                Console.WriteLine($"{label}: X=[{minX:F1},{maxX:F1}] Δ={maxX-minX:F1}, Y=[{minY:F1},{maxY:F1}] Δ={maxY-minY:F1}, Z=[{minZ:F1},{maxZ:F1}] Δ={maxZ-minZ:F1}");
            }
        }

        void DumpDiagnostics(string pm4Path)
        {
            try
            {
                var pm4 = PM4File.FromFile(pm4Path);
                string tsDir = Path.Combine(Environment.CurrentDirectory, "project_output", "diagnostics");
                Directory.CreateDirectory(tsDir);
                string csvName = Path.GetFileNameWithoutExtension(pm4Path) + ".csv";
                MsLkDiagnostics.DumpEntryCsv(pm4, Path.Combine(tsDir, csvName));
                Console.WriteLine($"  ↳ diagnostics → {csvName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ↳ diagnostics failed: {ex.Message}");
            }
        }
    }

    private static bool TryParseTileCoords(string stem, out int tileX, out int tileY)
    {
        tileX = tileY = 0;
        var parts = stem.Split('_');
        if (parts.Length < 2) return false;
        return int.TryParse(parts[^2], out tileX) && int.TryParse(parts[^1], out tileY);
    }
}
