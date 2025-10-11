using System;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpDataHarvester.Export;
using ParpDataHarvester.Export.Builders;
using ParpDataHarvester.Export.Geometry;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;

namespace ParpDataHarvester.Commands
{
    internal static class ExportGlbRawCommand
    {
        internal static int Run(ReadOnlySpan<string> args)
        {
            string? inPath = null;
            string? outDir = null;
            bool perRegion = false;
            string mode = "objects"; // or "surfaces"
            bool flipX = false; // optional X-axis inversion for visualization parity

            for (int i = 0; i < args.Length; i++)
            {
                var a = args[i];
                switch (a)
                {
                    case "--in":
                        if (i + 1 < args.Length) inPath = args[++i];
                        else { Console.Error.WriteLine("--in requires a value"); return 2; }
                        break;
                    case "--out":
                        if (i + 1 < args.Length) outDir = args[++i];
                        else { Console.Error.WriteLine("--out requires a value"); return 2; }
                        break;
                    case "--per-region":
                        perRegion = true;
                        break;
                    case "--mode":
                        if (i + 1 < args.Length) mode = args[++i];
                        else { Console.Error.WriteLine("--mode requires a value"); return 2; }
                        break;
                    case "--flip-x":
                        flipX = true;
                        break;
                    default:
                        Console.Error.WriteLine($"Unknown option: {a}");
                        return 2;
                }
            }

            if (string.IsNullOrWhiteSpace(inPath) || string.IsNullOrWhiteSpace(outDir))
            {
                Console.Error.WriteLine("--in and --out are required");
                return 2;
            }

            Directory.CreateDirectory(outDir);

            Console.WriteLine("[harvester] export-glb-raw starting");
            Console.WriteLine($"  in:  {inPath}");
            Console.WriteLine($"  out: {Path.GetFullPath(outDir)}");
            Console.WriteLine($"  per-region: {perRegion}");
            Console.WriteLine($"  mode: {mode}");
            Console.WriteLine($"  flip-x: {flipX}");

            try
            {
                var adapter = new Pm4Adapter();
                var assembler = new RawGeometryAssembler();
                var graphBuilder = new MslkNodeGraphBuilder();

                bool isFile = File.Exists(inPath);
                bool isDir = Directory.Exists(inPath);
                if (!isFile && !isDir)
                {
                    Console.Error.WriteLine($"Input path not found: {inPath}");
                    return 2;
                }

                if (isFile)
                {
                    var scene = adapter.Load(inPath);
                    var result = assembler.Assemble(scene, mode);
                    var graph = graphBuilder.Build(scene, result.Primitives);
                    Console.WriteLine($"[assemble] vertices={result.Vertices.Count}, primitives={result.Primitives.Count}");
                    var tilesDir = Path.Combine(outDir, "tiles");
                    Directory.CreateDirectory(tilesDir);
                    var baseName = Path.GetFileNameWithoutExtension(inPath);
                    var outPath = Path.Combine(tilesDir, baseName + ".glb");
                    GltfRawWriter.WriteGlb(outPath, ApplyFlipXIfNeeded(result.Vertices, flipX), result.Primitives, baseName);
                    Console.WriteLine($"[harvester] wrote {outPath}");
                }
                else if (isDir && perRegion)
                {
                    var global = Pm4GlobalTileLoader.LoadRegion(inPath);
                    var scene = Pm4GlobalTileLoader.ToStandardScene(global);
                    var result = assembler.Assemble(scene, mode);
                    var graph = graphBuilder.Build(scene, result.Primitives);
                    Console.WriteLine($"[assemble] vertices={result.Vertices.Count}, primitives={result.Primitives.Count}");
                    var regionDir = Path.Combine(outDir, "region");
                    Directory.CreateDirectory(regionDir);
                    var regionName = new DirectoryInfo(inPath).Name;
                    var outPath = Path.Combine(regionDir, regionName + ".glb");
                    GltfRawWriter.WriteGlb(outPath, ApplyFlipXIfNeeded(result.Vertices, flipX), result.Primitives, regionName);
                    Console.WriteLine($"[harvester] wrote {outPath}");
                }
                else if (isDir)
                {
                    var tilesDir = Path.Combine(outDir, "tiles");
                    Directory.CreateDirectory(tilesDir);
                    var files = Directory.GetFiles(inPath, "*.pm4", SearchOption.TopDirectoryOnly)
                        .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                        .ToArray();
                    Console.WriteLine($"[harvester] exporting {files.Length} tiles...");
                    foreach (var file in files)
                    {
                        try
                        {
                            var scene = adapter.Load(file);
                            var result = assembler.Assemble(scene, mode);
                            var graph = graphBuilder.Build(scene, result.Primitives);
                            Console.WriteLine($"[assemble] {Path.GetFileName(file)}: vertices={result.Vertices.Count}, primitives={result.Primitives.Count}");
                            var baseName = Path.GetFileNameWithoutExtension(file);
                            var outPath = Path.Combine(tilesDir, baseName + ".glb");
                            GltfRawWriter.WriteGlb(outPath, ApplyFlipXIfNeeded(result.Vertices, flipX), result.Primitives, baseName);
                            Console.WriteLine($"  wrote {outPath}");
                        }
                        catch (Exception exTile)
                        {
                            Console.Error.WriteLine($"  failed {Path.GetFileName(file)}: {exTile.Message}");
                        }
                    }
                }

                Console.WriteLine("[harvester] completed without errors");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[harvester] error: {ex.Message}");
                return 1;
            }
        }

        private static IReadOnlyList<Vector3> ApplyFlipXIfNeeded(IReadOnlyList<Vector3> vertices, bool flipX)
        {
            if (!flipX) return vertices;
            var flipped = new Vector3[vertices.Count];
            for (int i = 0; i < vertices.Count; i++)
            {
                var v = vertices[i];
                flipped[i] = new Vector3(-v.X, v.Y, v.Z);
            }
            return flipped;
        }
    }
}

