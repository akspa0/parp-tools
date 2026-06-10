using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox;
using PM4NextExporter.Services;
using PM4NextExporter.Model;
using PM4NextExporter.Assembly;
using PM4NextExporter.Exporters;

namespace PM4NextExporter.Cli
{
    internal static class Program
    {
        private const string ToolName = "pm4next-export";

        private static int Main(string[] args)
        {
            if (args.Length == 0 || args.Contains("-h") || args.Contains("--help"))
            {
                PrintHelp();
                return 0;
            }

            var options = ParseArgs(args);
            if (string.IsNullOrWhiteSpace(options.InputPath))
            {
                Console.Error.WriteLine("[error] Missing input path.\n");
                PrintHelp();
                return 1;
            }

            if (!Directory.Exists(options.InputPath) && !File.Exists(options.InputPath))
            {
                Console.Error.WriteLine($"[error] Input path not found: {options.InputPath}");
                return 2;
            }

            // Establish output directory
            var baseName = string.IsNullOrWhiteSpace(options.OutDirBaseName)
                ? "pm4next"
                : options.OutDirBaseName!;
            var outDir = ProjectOutput.CreateOutputDirectory(baseName);

            // Simple log tee
            var logPath = Path.Combine(outDir, "run.log");
            void Log(string msg)
            {
                Console.WriteLine(msg);
                try { File.AppendAllText(logPath, msg + Environment.NewLine); } catch { /* ignore */ }
            }

            Log($"[{ToolName}] starting");
            Log($" input: {options.InputPath}");
            Log($" outDir: {outDir}");
            Log($" batch: {options.Batch}");
            Log($" format: {options.Format}");
            Log($" assembly: {options.AssemblyStrategy}");
            Log($" include-adjacent: {options.IncludeAdjacent}");
            Log($" csv-diagnostics: {options.CsvDiagnostics} -> {options.CsvOut ?? "(default)"}");
            Log($" groups: {(options.GroupKeys.Count == 0 ? "(none)" : string.Join(",", options.GroupKeys))}");
            Log($" parent16-swap: {options.Parent16Swap}");
            Log($" audit-only: {options.AuditOnly}");
            Log($" no-remap: {options.NoRemap}");
            Log($" align-with-mscn: {options.AlignWithMscn}");
            if (options.AlignWithMscn)
            {
                Log("[info] --align-with-mscn: Mesh exporters mirror X centered per object/tile to preserve orientation without quadrant flips.");
                Log("[info] --align-with-mscn: This takes precedence over --legacy-obj-parity for meshes. MSCN outputs remain unmirrored.");
            }
            if (options.CkSplitByType)
            {
                Log($" ck-split-by-type: {options.CkSplitByType}");
            }
            if (options.ExportMscnObj)
            {
                Log($" export-mscn-obj: {options.ExportMscnObj}");
            }
            if (options.ExportObjectMscn)
            {
                Log($" export-object-mscn: {options.ExportObjectMscn}");
            }
            if (options.NameObjectsWithTile)
            {
                Log($" name-with-tile: {options.NameObjectsWithTile}");
            }
            if (options.MscnOnly)
            {
                Log($" mscn-only: {options.MscnOnly}");
            }
            if (options.AssemblyStrategy == AssemblyStrategy.MslkParent)
            {
                Log($" mslk-parent-min-tris: {options.MslkParentMinTriangles}");
                Log($" mslk-parent-allow-fallback: {options.MslkParentAllowFallbackScan}");
            }
            if (options.Correlates.Count > 0)
                Log($" correlate: {string.Join(",", options.Correlates.Select(c => c.Item1 + ":" + c.Item2))}");

            // Audit-only cross-tile path
            if (options.AuditOnly)
            {
                Log("[audit] Running cross-tile audit only...");
                CrossTileVertexResolver.Audit(options.InputPath!, outDir);
                Log("[audit] Cross-tile audit completed.");
                return 0;
            }

            // MSCN-only short-circuit: iterate per tile, skip mesh assembly/export
            if (options.MscnOnly)
            {
                Log("[info] MSCN-only mode: skipping mesh assembly/export, processing per tile.");
                var loader2 = new SceneLoader();
                var inputs = new List<string>();
                if (Directory.Exists(options.InputPath!))
                {
                    inputs.AddRange(Directory.GetFiles(options.InputPath!, "*.pm4"));
                }
                else if (File.Exists(options.InputPath!))
                {
                    inputs.Add(options.InputPath!);
                }
                else
                {
                    Console.Error.WriteLine($"[error] Input path not found: {options.InputPath}");
                    return 2;
                }

                inputs.Sort(StringComparer.OrdinalIgnoreCase);
                int processed = 0;
                foreach (var file in inputs)
                {
                    // Derive a per-tile output subfolder to avoid file overwrite
                    string subName;
                    try
                    {
                        var coord = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.TileCoordinate.FromFileName(Path.GetFileName(file));
                        subName = $"tile_X{coord.X:00}_Y{coord.Y:00}";
                    }
                    catch
                    {
                        subName = Path.GetFileNameWithoutExtension(file);
                    }
                    var perTileOutDir = Path.Combine(outDir, subName);
                    Directory.CreateDirectory(perTileOutDir);

                    var perTileScene = loader2.LoadSingleTile(file, includeAdjacent: false, applyMscnRemap: !options.NoRemap);
                    Log($" [mscn-only] tile='{Path.GetFileName(file)}' mscn={perTileScene.MscnVertices?.Count ?? 0}");

                    if (options.ExportMscnObj)
                    {
                        PM4NextExporter.Exporters.MscnObjExporter.Export(perTileScene, perTileOutDir, options.LegacyObjParity, options.NameObjectsWithTile, options.AlignWithMscn);
                        Log("  exported MSCN OBJ");
                    }

                    if (options.CsvDiagnostics)
                    {
                        var perTileCsvDir = options.CsvOut != null ? Path.Combine(options.CsvOut, subName) : perTileOutDir;
                        Directory.CreateDirectory(perTileCsvDir);
                        var mscnCount = perTileScene.MscnVertices?.Count ?? 0;
                        if (mscnCount == 0)
                        {
                            Log("  [warn] MSCN vertices empty; skipping mscn_vertices.csv");
                        }
                        else
                        {
                            DiagnosticsService.WriteMscnCsv(perTileCsvDir, perTileScene);
                            Log("  wrote mscn_vertices.csv");
                        }
                    }

                    processed++;
                }

                Log($"[info] MSCN-only completed. tiles={processed}");
                return 0;
            }

            // Minimal pipeline wiring
            var loader = new SceneLoader();
            var scene = loader.LoadSingleTile(options.InputPath!, options.IncludeAdjacent, applyMscnRemap: !options.NoRemap);
            Log($" scene: vertices={scene.VertexCount} surfaces={scene.SurfaceCount} mscn={scene.MscnVertices?.Count ?? 0}");
            
            // Choose assembler
            IAssembler assembler = options.AssemblyStrategy switch
            {
                AssemblyStrategy.ParentIndex => new ParentHierarchyAssembler(),
                AssemblyStrategy.MsurIndexCount => new MsurIndexCountAssembler(),
                AssemblyStrategy.MsurCompositeKey => new MsurCompositeKeyAssembler(),
                AssemblyStrategy.SurfaceKey => new SurfaceKeyAssembler(),
                AssemblyStrategy.SurfaceKeyAA => new SurfaceKeyAAAssembler(),
                AssemblyStrategy.CompositeHierarchy => new CompositeHierarchyAssembler(),
                AssemblyStrategy.ContainerHierarchy8Bit => new ContainerHierarchy8BitAssembler(),
                AssemblyStrategy.CompositeBytePair => new CompositeBytePairAssembler(),
                AssemblyStrategy.Parent16 => new Parent16Assembler(),
                AssemblyStrategy.MslkParent => new MslkParentAssembler(),
                AssemblyStrategy.MslkInstance => new MslkInstanceAssembler(false),
                AssemblyStrategy.MslkInstanceCk24 => new MslkInstanceAssembler(true),
                _ => new CompositeHierarchyAssembler()
            };

            // Guidance: MSLK parent assembler is experimental; resolves surfaces via MSPI first-index + tile offsets
            if (options.AssemblyStrategy == AssemblyStrategy.MslkParent)
            {
                Log("[info] Using MSLK parent assembly (experimental): groups by ParentId and resolves surfaces via tile index offsets.");
                Log("[hint] Use --mslk-parent-min-tris <N> (e.g., 300-1000) to suppress tiny fragments.");
                Log("[hint] Disable cross-tile fallback scan by default; enable with --mslk-parent-allow-fallback if necessary.");
            }

            // Guidance: SurfaceKey can merge unrelated objects; composite-hierarchy is usually more accurate
            if (options.AssemblyStrategy == AssemblyStrategy.SurfaceKey)
            {
                Log("[hint] SurfaceKey grouping may merge unrelated objects into the same OBJ.");
                Log("[hint] For more accurate per-object outputs, consider --assembly composite-hierarchy.");
            }

            var assembled = assembler.Assemble(scene, options).ToList();
            Log($" assembled objects: {assembled.Count}");

            // Export
            switch (options.Format)
            {
                case ExportFormat.Obj:
                    PM4NextExporter.Exporters.ObjExporter.Export(assembled, outDir, options.LegacyObjParity, options.ProjectLocal, options.AlignWithMscn);
                    Log(" exported OBJ");
                    if (options.ExportTiles)
                    {
                        // Per-tile export grouped by assembled object metadata to preserve object boundaries
                        if (options.ProjectLocal)
                        {
                            Log("[info] per-tile OBJ export preserves global coordinates; --project-local is ignored for tiles");
                        }
                        Log("[info] per-tile OBJ uses global coordinates (no local projection). Object-level X mirroring matches OBJ export and is controlled by --legacy-obj-parity; when --align-with-mscn is set, mirroring is centered per tile");
                        PM4NextExporter.Exporters.PerTileObjectsExporter.Export(assembled, outDir, options.LegacyObjParity, options.ProjectLocal, options.AlignWithMscn);
                        Log(" exported per-tile OBJ");
                    }
                    break;
                case ExportFormat.Gltf:
                case ExportFormat.Glb:
                    Log("[warn] glTF/GLB export not implemented yet");
                    break;
            }

            // Per-object MSCN sidecars (OBJ points + CSV)
            if (options.ExportObjectMscn)
            {
                var mscnCount = scene.MscnVertices?.Count ?? 0;
                if (mscnCount == 0)
                {
                    Log("[warn] MSCN vertices empty; skipping per-object MSCN sidecars");
                }
                else
                {
                    var remapApplied = !options.NoRemap; // mirrors how we invoked the loader
                    var attrib = PM4NextExporter.Services.MscnAttribution.Attribute(scene, assembled, remapApplied);
                    PM4NextExporter.Exporters.ObjectMscnSidecarExporter.Export(scene, assembled, attrib, outDir, options.LegacyObjParity, options.NameObjectsWithTile, options.AlignWithMscn);
                    Log(" exported per-object MSCN sidecars");
                }
            }

            // Optional: export MSCN vertices as separate OBJ layers for validation
            if (options.ExportMscnObj)
            {
                PM4NextExporter.Exporters.MscnObjExporter.Export(scene, outDir, options.LegacyObjParity, options.NameObjectsWithTile, options.AlignWithMscn);
                Log(" exported MSCN OBJ layers");
            }

            // Diagnostics
            if (options.CsvDiagnostics)
            {
                var diagDir = options.CsvOut ?? outDir;
                DiagnosticsService.WriteSurfaceCsv(diagDir, scene);
                DiagnosticsService.WriteCompositeSummaryCsv(diagDir, scene);
                DiagnosticsService.WriteMslkLinksCsv(diagDir, scene);
                var mscnCount = scene.MscnVertices?.Count ?? 0;
                if (mscnCount == 0)
                {
                    Log("[warn] MSCN vertices empty; skipping mscn_vertices.csv");
                }
                else
                {
                    DiagnosticsService.WriteMscnCsv(diagDir, scene);
                }
                DiagnosticsService.WriteAssemblyCoverageCsv(diagDir, assembled);
                DiagnosticsService.WriteSurfaceParentHitsCsv(diagDir, scene);
                Log(" diagnostics CSV written");
            }

            Log("[info] CLI skeleton with SceneLoader, assembler, exporter initialized.");

            // Exit success for now
            return 0;
        }

        private static Model.Options ParseArgs(string[] args)
        {
            var opts = new Model.Options();
            var i = 0;
            while (i < args.Length)
            {
                var a = args[i];
                switch (a)
                {
                    case "--out":
                        opts.OutDirBaseName = NextOrThrow(args, ref i, a);
                        break;
                    case "--format":
                        opts.Format = ParseFormat(NextOrThrow(args, ref i, a));
                        break;
                    case "--assembly":
                        opts.AssemblyStrategy = ParseAssembly(NextOrThrow(args, ref i, a));
                        break;
                    case "--group":
                        opts.GroupKeys.Add(ParseGroup(NextOrThrow(args, ref i, a)));
                        break;
                    case "--parent16-swap":
                        opts.Parent16Swap = true;
                        break;
                    case "--batch":
                        opts.Batch = true;
                        break;
                    case "--include-adjacent":
                        opts.IncludeAdjacent = true;
                        break;
                    case "--csv-diagnostics":
                        opts.CsvDiagnostics = true;
                        break;
                    case "--csv-out":
                        opts.CsvOut = NextOrThrow(args, ref i, a);
                        break;
                    case "--legacy-obj-parity":
                        opts.LegacyObjParity = true;
                        break;
                    case "--audit-only":
                        opts.AuditOnly = true;
                        break;
                    case "--no-remap":
                        opts.NoRemap = true;
                        break;
                    case "--mscn-only":
                        opts.MscnOnly = true;
                        break;
                    case "--ck-split-by-type":
                        opts.CkSplitByType = true;
                        break;
                    case "--export-mscn-obj":
                        opts.ExportMscnObj = true;
                        break;
                    case "--export-object-mscn":
                        opts.ExportObjectMscn = true;
                        break;
                    case "--project-local":
                        opts.ProjectLocal = true;
                        break;
                    case "--export-tiles":
                        opts.ExportTiles = true;
                        break;
                    case "--name-with-tile":
                        opts.NameObjectsWithTile = true;
                        break;
                    case "--align-with-mscn":
                        opts.AlignWithMscn = true;
                        break;
                    case "--mslk-parent-min-tris":
                        if (!int.TryParse(NextOrThrow(args, ref i, a), out var minTris) || minTris < 0)
                            throw new ArgumentException("--mslk-parent-min-tris requires a non-negative integer");
                        opts.MslkParentMinTriangles = minTris;
                        break;
                    case "--mslk-parent-allow-fallback":
                        opts.MslkParentAllowFallbackScan = true;
                        break;
                    case "--correlate":
                        var pair = NextOrThrow(args, ref i, a);
                        var parts = pair.Split(':');
                        if (parts.Length != 2) throw new ArgumentException("--correlate expects keyA:keyB");
                        opts.Correlates.Add(Tuple.Create(parts[0], parts[1]));
                        break;
                    default:
                        if (!a.StartsWith("-"))
                        {
                            // first non-flag arg is input
                            if (string.IsNullOrWhiteSpace(opts.InputPath)) opts.InputPath = a;
                            else opts.AdditionalInputs.Add(a);
                        }
                        else
                        {
                            throw new ArgumentException($"Unknown argument: {a}");
                        }
                        break;
                }
                i++;
            }
            return opts;
        }

        private static string NextOrThrow(string[] args, ref int i, string flag)
        {
            if (i + 1 >= args.Length) throw new ArgumentException($"{flag} requires a value");
            return args[++i];
        }

        private static Model.ExportFormat ParseFormat(string s)
            => s.ToLowerInvariant() switch
            {
                "obj" => Model.ExportFormat.Obj,
                "gltf" => Model.ExportFormat.Gltf,
                "glb" => Model.ExportFormat.Glb,
                _ => throw new ArgumentException("--format must be obj|gltf|glb")
            };

        private static Model.AssemblyStrategy ParseAssembly(string s)
            => s.ToLowerInvariant() switch
            {
                "parent-index" => Model.AssemblyStrategy.ParentIndex,
                "msur-indexcount" => Model.AssemblyStrategy.MsurIndexCount,
                "msur-compositekey" => Model.AssemblyStrategy.MsurCompositeKey,
                "surface-key" => Model.AssemblyStrategy.SurfaceKey,
                "surface-key-aa" => Model.AssemblyStrategy.SurfaceKeyAA,
                "composite-hierarchy" => Model.AssemblyStrategy.CompositeHierarchy,
                "container-hierarchy-8bit" => Model.AssemblyStrategy.ContainerHierarchy8Bit,
                "composite-hierarchy-8bit" => Model.AssemblyStrategy.ContainerHierarchy8Bit,
                "composite-bytepair" => Model.AssemblyStrategy.CompositeBytePair,
                "parent16" => Model.AssemblyStrategy.Parent16,
                "mslk-parent" => Model.AssemblyStrategy.MslkParent,
                "mslk-instance" => Model.AssemblyStrategy.MslkInstance,
                "mslk-instance+ck24" => Model.AssemblyStrategy.MslkInstanceCk24,
                _ => throw new ArgumentException("--assembly must be parent-index|msur-indexcount|msur-compositekey|surface-key|surface-key-aa|composite-hierarchy|container-hierarchy-8bit|composite-bytepair|parent16|mslk-parent|mslk-instance|mslk-instance+ck24")
            };

        private static Model.GroupKey ParseGroup(string s)
            => s.ToLowerInvariant() switch
            {
                "parent16" => Model.GroupKey.Parent16,
                "parent16-container" => Model.GroupKey.Parent16Container,
                "parent16-object" => Model.GroupKey.Parent16Object,
                "surface" => Model.GroupKey.Surface,
                "flags" => Model.GroupKey.Flags,
                "type" => Model.GroupKey.Type,
                "sortkey" => Model.GroupKey.SortKey,
                "tile" => Model.GroupKey.Tile,
                _ => throw new ArgumentException("--group value not recognized")
            };

        private static void PrintHelp()
        {
            Console.WriteLine(@$"{ToolName}
Usage:
  {ToolName} <pm4Input|directory> [--out <dir>] [--include-adjacent] [--format obj|gltf|glb]
  [--assembly parent-index|msur-indexcount|msur-compositekey|surface-key|surface-key-aa|composite-hierarchy|container-hierarchy-8bit|composite-bytepair|parent16|mslk-parent|mslk-instance|mslk-instance+ck24]
  [--group parent16|parent16-container|parent16-object|surface|flags|type|sortkey|tile]
          [--parent16-swap] [--csv-diagnostics] [--csv-out <dir>] [--correlate <keyA:keyB>]
  [--batch] [--legacy-obj-parity] [--audit-only] [--no-remap] [--ck-split-by-type]
  [--mslk-parent-min-tris <N>] [--mslk-parent-allow-fallback] [--export-mscn-obj] [--export-object-mscn] [--mscn-only] [--export-tiles] [--project-local] [--name-with-tile] [--align-with-mscn]

Flags:
  --align-with-mscn        Align mesh exporters with MSCN's +X/+Y/+Z by applying centered X-mirroring per object/tile.
                           Overrides --legacy-obj-parity for mesh mirroring; MSCN outputs remain unmirrored.
");
        }
    }
}
