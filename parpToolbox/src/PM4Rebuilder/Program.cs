using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;
using System.Collections.Generic;
using PM4Rebuilder.Pipeline;

namespace PM4Rebuilder
{
    internal static class Program
    {
        private const string Usage = "Usage: PM4Rebuilder direct-export <pm4Input> [outDir]     - Direct PM4 to OBJ export (RECOMMENDED)\n       OR: PM4Rebuilder export-db <pm4Input> [outDir]       - Export PM4 data to SQLite database\n       OR: PM4Rebuilder export-buildings <scene.db> [outDir] - Export buildings from database to OBJ files\n       OR: PM4Rebuilder export-subcomponents <scene.db> [outDir]\n       OR: PM4Rebuilder analyze-building-groups <scene.db> [outDir]";

        public static async Task<int> Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("PM4Rebuilder – proof-of-concept object assembler\n" + Usage);
                return 1;
            }

            // Quick command shortcut: explore <pm4Dir>
            if (args[0].Equals("explore", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("ERROR: explore command requires <pm4File|directory> argument.");
                    return 1;
                }

                string exploreInput = args[1];
                string exploreOut = Path.Combine(Directory.GetCurrentDirectory(), "pm4_explore_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                Directory.CreateDirectory(exploreOut);
                PipelineLogger.Initialize(exploreOut);

                var sceneExpl = await SceneLoaderHelper.LoadSceneAsync(exploreInput, includeAdjacent: false, applyTransform: true, altTransform: false);
                PipelineLogger.Log($"[EXPLORE] Scene loaded: Vertices={sceneExpl.Vertices.Count}, Triangles={sceneExpl.Triangles.Count}");
                ExploreHarness.Run(sceneExpl, exploreOut, exportObj: true);
                PipelineLogger.Log("[EXPLORE] Completed automated exploration.");
                return 0;
            }

            // Direct PM4 to OBJ export (recommended approach)
            if (args[0].Equals("direct-export", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("ERROR: direct-export command requires <pm4Input> argument.");
                    return 1;
                }
                string pm4Input = args[1];
                string outDirDirect = args.Length >= 3 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "direct_export", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                Directory.CreateDirectory(outDirDirect);
                
                int exitCode = DirectPm4Exporter.ExportBuildings(pm4Input, outDirDirect);
                return exitCode;
            }

                        // Quick command shortcut: analyze-db <dbPath> [outDir]
            if (args[0].Equals("export-db", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("ERROR: export-db command requires <pm4File|directory> argument.");
                    return 1;
                }
                string pm4Input = args[1];
                string outDirExport = args.Length >= 3 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "pm4_db_export", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                Directory.CreateDirectory(outDirExport);

                // Load scene (single tile for now, no transforms)
                var sceneExport = await SceneLoaderHelper.LoadSceneAsync(pm4Input, includeAdjacent:false, applyTransform:false, altTransform:false);
                string dbPath = Path.Combine(outDirExport, "PM4_Scene_analysis.db");

                var exporter = new ParpToolbox.Services.PM4.Database.Pm4DatabaseExporter(dbPath);
                await exporter.ExportSceneAsync(sceneExport, Path.GetFileName(pm4Input), pm4Input);

                Console.WriteLine($"[EXPORT-DB] Database written to {dbPath}");
                return 0;
            }

            if (args[0].Equals("analyze-db", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("ERROR: analyze-db command requires <dbPath> argument.");
                    return 1;
                }
                string dbPath = args[1];
                string outDirAnalyze = args.Length >= 3 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "pm4_db_analysis", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                Directory.CreateDirectory(outDirAnalyze);
                int exitCode = DatabaseRelationshipAnalyzer.Analyze(dbPath, outDirAnalyze);
                return exitCode;
            }

            if (args[0].Equals("export-subcomponents", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("ERROR: export-subcomponents command requires <scene.db> argument.");
                    return 1;
                }
                string dbPath = args[1];
                string outDirExport = args.Length >= 3 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                Directory.CreateDirectory(outDirExport);
                int exitCode = BulkSubComponentExporter.ExportAll(dbPath, outDirExport);
                return exitCode;
            }

            if (args[0].Equals("analyze-building-groups", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("ERROR: analyze-building-groups command requires <scene.db> argument.");
                    return 1;
                }
                string dbPath = args[1];
                string outDirAnalyze = args.Length >= 3 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                Directory.CreateDirectory(outDirAnalyze);
                int exitCode = BuildingAggregationAnalyzer.Analyze(dbPath, outDirAnalyze);
                return exitCode;
            }

            if (args[0].Equals("export-buildings", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("ERROR: export-buildings command requires <scene.db> argument.");
                    return 1;
                }
                string dbPath = args[1];
                string outDirBuildings = args.Length >= 3 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                Directory.CreateDirectory(outDirBuildings);
                int exitCode = BuildingLevelExporter.ExportAllBuildings(dbPath, outDirBuildings);
                return exitCode;
            }

            if (args[0].Equals("export-subcomponent", StringComparison.OrdinalIgnoreCase))
            {
                if (args.Length < 3)
                {
                    Console.WriteLine("PM4Rebuilder - Command Line Interface");
                    Console.WriteLine("Usage:");
                    Console.WriteLine("  direct-export <pm4Input> [outDir]       - Direct PM4 to OBJ export (RECOMMENDED)");
                    Console.WriteLine("  export-db <pm4Input> [outDir]           - Export PM4 data to SQLite database");
                    Console.WriteLine("  export-buildings <scene.db> [outDir]    - Export buildings from database to OBJ files");
                    Console.WriteLine("  export-subcomponent <dbPath> <ObjectId> [outObj] - Export single object to OBJ");
                    return 1;
                }
                string dbPath = args[1];
                if (!uint.TryParse(args[2], out uint objectId))
                {
                    Console.WriteLine("ERROR: <ObjectId> must be an unsigned integer.");
                    return 1;
                }
                string outObj;
                if (args.Length >= 4)
                {
                    outObj = args[3];
                }
                else
                {
                    string stampDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                    Directory.CreateDirectory(stampDir);
                    outObj = Path.Combine(stampDir, $"subcomponent_{objectId}.obj");
                }

                SubComponentObjExporter.Export(dbPath, objectId, outObj);
                return 0;
            }

            string? inputPath = null;
            bool includeAdjacent = false;
            bool rawCoords = false;
            bool altTransform = false;
            bool combinedOutput = false;
            bool perChunkOutput = false;
            bool testTransforms = false;
            bool analyzeLinkage = false;
            bool validateData = false;
            
            bool singleTile = false;
            bool batchAll = false;
            bool batchExportObj = false;
            bool auditOnly = false;
            bool perObject = false;
            bool globalObj = false;
            
            string? outDir = null;
            string? dumpDir = null;

            // naïve arg parse - handle both input path and flags
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--include-adjacent":
                        includeAdjacent = true;
                        break;
                    case "--out":
                        if (i + 1 < args.Length)
                        {
                            outDir = args[++i];
                        }
                        break;
                    case "--dump-mscn":
                        if (i + 1 < args.Length)
                        {
                            dumpDir = args[++i];
                        }
                        break;
                    case "--raw":
                        rawCoords = true;
                        break;
                    case "--alt":
                        altTransform = true;
                        break;
                    case "--combined":
                        combinedOutput = true;
                        break;
                    case "--per-chunk":
                        perChunkOutput = true;
                        break;
                    case "--test-transforms":
                        testTransforms = true;
                        break;
                    case "--analyze-linkage":
                        analyzeLinkage = true;
                        break;
                    case "--validate-data":
                        validateData = true;
                        break;
                    case "--audit-only":
                        auditOnly = true;
                        break;
                    case "--per-object":
                        perObject = true; 
                        break;
                    case "--global-obj":
                        globalObj = true; 
                        break;
                    case "--single-tile":
                        singleTile = true;
                        break;
                    case "--batch-all":
                        batchAll = true;
                        break;
                    case "--batch-export-obj":
                        batchExportObj = true;
                        break;
                    default:
                        // If it's not a flag, treat it as input path
                        if (!args[i].StartsWith("--") && inputPath == null)
                        {
                            inputPath = args[i];
                        }
                        break;
                }
            }

            outDir ??= Path.Combine(Directory.GetCurrentDirectory(), "pm4rebuilder_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            Directory.CreateDirectory(outDir);

            // Initialize tee logger (console + file)
            PipelineLogger.Initialize(outDir);

            // Map new flags to existing behavior
            combinedOutput = globalObj;

            // mutual exclusivity
            if (rawCoords && altTransform)
            {
                Console.WriteLine("--raw and --alt are mutually exclusive. Choose only one transformation mode.");
                return 1;
            }
            if (combinedOutput && perChunkOutput)
            {
                Console.WriteLine("--combined and --per-chunk cannot be used together.");
                return 1;
            }

            Console.WriteLine($"Input: {inputPath}\nOutput dir: {outDir}\nInclude adjacent: {includeAdjacent}\nDump MSCN: {(dumpDir ?? "(none)")}\nRaw coords: {rawCoords}\nAlt transform: {altTransform}\nCombined OBJ: {combinedOutput}\nPer-chunk OBJ: {perChunkOutput}\n");

            try
            {
                // Batch process ALL PM4 files if requested (data recovery mode)
                if (batchAll)
                {
                    if (inputPath == null || !Directory.Exists(inputPath))
                    {
                        Console.WriteLine("ERROR: --batch-all requires a directory path containing PM4 files.");
                        return 1;
                    }
                    
                    Console.WriteLine("=== PM4 BATCH DATA RECOVERY ANALYSIS ===");
                    Console.WriteLine("Processing ALL PM4 files for comprehensive data extraction...");
                    return await Pm4BatchAnalyzer.ProcessAllFiles(inputPath, outDir);
                }
                
                // Batch export OBJs using decoded field algorithm
                if (batchExportObj)
                {
                    if (inputPath == null || !Directory.Exists(inputPath))
                    {
                        Console.WriteLine("ERROR: --batch-export-obj requires a directory path containing PM4 files.");
                        return 1;
                    }
                    
                    Console.WriteLine("=== PM4 BATCH OBJ EXPORT WITH DECODED FIELDS ===");
                    Console.WriteLine("Exporting OBJs using HasGeometry flag and decoded field algorithm...");
                    return await Pm4BatchObjExporter.ExportAllWithDecodedFields(inputPath, outDir);
                }
                
                if (inputPath == null)
                {
                    Console.WriteLine("ERROR: Input path is required for non-batch operations.");
                    return 1;
                }
                
                var scene = await SceneLoaderHelper.LoadSceneAsync(inputPath, includeAdjacent, !rawCoords, altTransform);
                Console.WriteLine($"Loaded scene: Vertices={scene.Vertices.Count}, Triangles={scene.Triangles.Count}");

                // Step A: Chunk audit
                ChunkAuditor.Audit(scene, outDir);
                if (auditOnly)
                {
                    Console.WriteLine("Audit-only mode complete. No OBJ export performed.");
                    return 0;
                }

                // Step B: Build object graph
                var objects = ObjectGraphBuilder.Build(scene, outDir);
                Console.WriteLine($"Resolved {objects.Count} objects after graph analysis");

                // Step C: Geometry extraction & deduplication
                GeometryExtractor.CleanGeometry(objects);

                // Optionally dump MSCN diagnostics
                if (dumpDir != null)
                {
                    Directory.CreateDirectory(dumpDir);
                    MscnAnalyzer.Dump(scene, dumpDir);
                }

                // Test coordinate transformations if requested
                if (testTransforms)
                {
                    Console.WriteLine("Running coordinate transformation tests...");
                    CoordinateTransformTester.TestTransforms(scene, outDir);
                    Console.WriteLine("Transform testing complete. Check output directory for test OBJ files.");
                    return 0; // Exit after testing
                }

                // Analyze MSCN linkage if requested
                if (analyzeLinkage)
                {
                    Console.WriteLine("Running MSCN linkage analysis...");
                    MscnLinkageAnalyzer.AnalyzeLinkage(scene, outDir);
                    Console.WriteLine("MSCN linkage analysis complete. Check output directory for analysis files.");
                    return 0; // Exit after analysis
                }

                // Validate data comprehensiveness if requested
                if (validateData)
                {
                    Console.WriteLine("Running comprehensive PM4 data validation...");
                    Pm4DataValidator.ValidateAndExport(scene, outDir);
                    Console.WriteLine("Data validation complete. Check output directory for detailed CSV analysis.");
                    return 0; // Exit after validation
                }

                // Export single tile with all geometry if requested
                if (singleTile)
                {
                    Console.WriteLine("Exporting single PM4 tile with all geometry...");
                    SingleTileExporter.ExportSingleTile(scene, outDir);
                    Console.WriteLine("Single tile export complete.");
                    return 0; // Exit after single tile export
                }



                // Step D: OBJ writing
                if (perChunkOutput)
                {
                    // Use the authoritative exporter that understands MSUR↔MSVI↔MSLK linkage.
                    ParpToolbox.Services.PM4.Pm4SurfaceGroupExporter.ExportSurfaceGroupsFromScene(scene, outDir);
                }
                else if (globalObj)
                {
                    Directory.CreateDirectory(outDir);
                    string combinedPath = Path.Combine(outDir, "scene_combined.obj");
                    CombinedObjectExporter.Export(objects, scene, combinedPath);
                    Console.WriteLine($"Combined OBJ written to {combinedPath}");
                }
                else // per-object or merged single
                {
                    // Export options: individual OBJs or single merged OBJ
                    if (!perObject)
                    {
                        // Export merged single OBJ when per-object grouping disabled
                        ObjectExporter.ExportMergedSingleOBJ(objects, scene, outDir, "complete_pm4_export");
                        Console.WriteLine($"[MERGED OBJ] {objects.Count} objects exported as complete_pm4_export.obj");
                    }
                    else
                    {
                        // Export each object to its own OBJ file for now (future: groups within tile file)
                        for (int i = 0; i < objects.Count; i++)
                        {
                            var obj = objects[i];
                            ObjectExporter.Export(obj, scene, outDir);
                            Console.WriteLine($"[{i + 1}/{objects.Count}] Exported {obj.Name}: {obj.Triangles.Count} faces");
                        }
                    }
                }
                Console.WriteLine("Export complete. Objects written to output directory.");
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}\n{ex.StackTrace}");
                return 1;
            }
        }
    }

    #region Helpers (temporary placeholders)
    internal static class SceneLoaderHelper
    {
        /// <summary>
        /// Loads a PM4 scene from a single tile file or a directory containing tiles.
        /// If <paramref name="includeAdjacent"/> is true, a 3×3 neighborhood is merged via <see cref="Pm4Adapter.LoadRegion"/>.
        /// After loading, coordinates are unified to OBJ-friendly right-handed space.
        /// </summary>
        public static async Task<Pm4Scene> LoadSceneAsync(string path, bool includeAdjacent, bool applyTransform, bool altTransform)
        {
            return await Task.Run(() =>
            {
                var adapter = new Pm4Adapter();
                string? seedTile = null;

                if (File.Exists(path) && Path.GetExtension(path).Equals(".pm4", StringComparison.OrdinalIgnoreCase))
                {
                    seedTile = path;
                }
                else if (Directory.Exists(path))
                {
                    seedTile = Directory.GetFiles(path, "*.pm4").FirstOrDefault();
                }
                else if (path.Contains("*"))
                {
                    // simple glob handling
                    var dir = Path.GetDirectoryName(path);
                    var mask = Path.GetFileName(path);
                    seedTile = Directory.GetFiles(dir ?? ".", mask).FirstOrDefault();
                }

                if (seedTile == null)
                    throw new FileNotFoundException("No PM4 tile(s) found for the supplied path or pattern.");

                Pm4Scene scene = includeAdjacent ? adapter.LoadRegion(seedTile) : adapter.Load(seedTile);

                // Extract MSCN vertices into dedicated list BEFORE coordinate transformation
                var mscnChunk = scene.ExtraChunks.FirstOrDefault(ch => ch.GetType().Name.Contains("Mscn", StringComparison.OrdinalIgnoreCase));
                if (mscnChunk != null)
                {
                    var vertsProp = mscnChunk.GetType().GetProperty("Vertices");
                    if (vertsProp?.GetValue(mscnChunk) is System.Collections.IEnumerable vertsEnum)
                    {
                        foreach (dynamic v in vertsEnum)
                        {
                            // Raw MSCN vertices (no scale). They will be X-flipped and Y/Z-swapped by the unification step below.
                            scene.MscnVertices.Add(new System.Numerics.Vector3((float)v.X, (float)v.Y, (float)v.Z));
                        }
                    }
                }

                // Apply coordinate system fixes if requested
                if (applyTransform)
                {
                    TransformUtils.ApplyCoordinateUnification(scene);
                }

                return scene;
            });
        }
    }

    internal static class ObjExporter
    {
        public static Task ExportAsync(Pm4Scene scene, string path)
        {
            using var sw = new StreamWriter(path);
            sw.WriteLine("# PM4Rebuilder prototype OBJ");

            // Write vertices
            foreach (var v in scene.Vertices)
            {
                sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
            }

            // Faces are 1-indexed in OBJ
            foreach (var (A, B, C) in scene.Triangles)
            {
                sw.WriteLine($"f {A + 1} {B + 1} {C + 1}");
            }
            return Task.CompletedTask;
        }
    }
    #endregion
}
