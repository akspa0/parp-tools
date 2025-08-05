using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;
using System.Collections.Generic;

namespace PM4Rebuilder
{
    internal static class Program
    {
        private const string Usage = "Usage: PM4Rebuilder <pm4File|directory> [--include-adjacent] [--out <dir>] [--dump-mscn <dir>] [--raw] [--alt] [--combined] [--per-chunk] [--test-transforms] [--analyze-linkage] [--validate-data] [--single-obj] [--single-tile]";

        public static async Task<int> Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("PM4Rebuilder – proof-of-concept object assembler\n" + Usage);
                return 1;
            }

            var inputPath = args[0];
            bool includeAdjacent = false;
            bool rawCoords = false;
            bool altTransform = false;
            bool combinedOutput = false;
            bool perChunkOutput = false;
            bool testTransforms = false;
            bool analyzeLinkage = false;
            bool validateData = false;
            bool singleObj = false;
            bool singleTile = false;
            string? outDir = null;
            string? dumpDir = null;

            // naïve arg parse
            for (int i = 1; i < args.Length; i++)
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
                    case "--single-obj":
                        singleObj = true;
                        break;
                    case "--single-tile":
                        singleTile = true;
                        break;
                }
            }

            outDir ??= Path.Combine(Directory.GetCurrentDirectory(), "pm4rebuilder_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            Directory.CreateDirectory(outDir);

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
                var scene = await SceneLoaderHelper.LoadSceneAsync(inputPath, includeAdjacent, !rawCoords, altTransform);
                Console.WriteLine($"Loaded scene: Vertices={scene.Vertices.Count}, Triangles={scene.Triangles.Count}");

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

                // Assemble objects
                var objects = Pm4ObjectAssembler.AssembleObjects(scene);
                Console.WriteLine($"Assembled {objects.Count} candidate objects");

                if (perChunkOutput)
                {
                    // Use the authoritative exporter that understands MSUR↔MSVI↔MSLK linkage.
                    ParpToolbox.Services.PM4.Pm4SurfaceGroupExporter.ExportSurfaceGroupsFromScene(scene, outDir);
                }
                else if (combinedOutput)
                {
                    Directory.CreateDirectory(outDir);
                    string combinedPath = Path.Combine(outDir, "scene_combined.obj");
                    CombinedObjectExporter.Export(objects, scene, combinedPath);
                    Console.WriteLine($"Combined OBJ written to {combinedPath}");
                }
                else
                {
                    // Export options: individual OBJs or single merged OBJ
                    if (singleObj)
                    {
                        // Export all objects merged into a single OBJ file
                        ObjectExporter.ExportMergedSingleOBJ(objects, scene, outDir, "complete_pm4_export");
                        Console.WriteLine($"[SINGLE OBJ] All {objects.Count} objects exported to single file: complete_pm4_export.obj");
                    }
                    else
                    {
                        // Export each object to its own OBJ (default behavior)
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
