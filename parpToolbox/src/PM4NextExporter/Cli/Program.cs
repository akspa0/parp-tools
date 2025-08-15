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
            if (options.CkSplitByType)
            {
                Log($" ck-split-by-type: {options.CkSplitByType}");
            }
            if (options.ExportMscnObj)
            {
                Log($" export-mscn-obj: {options.ExportMscnObj}");
            }
            if (options.NameObjectsWithTile)
            {
                Log($" name-with-tile: {options.NameObjectsWithTile}");
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

            // Minimal pipeline wiring
            var loader = new SceneLoader();
            var scene = loader.LoadSingleTile(options.InputPath!, options.IncludeAdjacent, applyMscnRemap: !options.NoRemap);
            Log($" scene: vertices={scene.VertexCount} surfaces={scene.SurfaceCount} mscn={scene.MscnVertices?.Count ?? 0}");
            
            // Choose assembler
            IAssembler assembler = options.AssemblyStrategy switch
            {
                AssemblyStrategy.ParentIndex => new ParentHierarchyAssembler(),
                AssemblyStrategy.MsurIndexCount => new MsurIndexCountAssembler(),
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
                    PM4NextExporter.Exporters.ObjExporter.Export(assembled, outDir, options.LegacyObjParity, options.ProjectLocal);
                    Log(" exported OBJ");
                    if (options.ExportTiles)
                    {
                        // Per-tile export grouped by assembled object metadata to preserve object boundaries
                        if (options.ProjectLocal)
                        {
                            Log("[info] per-tile OBJ export preserves global coordinates; --project-local is ignored for tiles");
                        }
                        Log("[info] per-tile OBJ uses global coordinates (no local projection). Object-level X mirroring matches OBJ export and is controlled by --legacy-obj-parity");
                        PM4NextExporter.Exporters.PerTileObjectsExporter.Export(assembled, outDir, options.LegacyObjParity, options.ProjectLocal);
                        Log(" exported per-tile OBJ");
                    }
                    break;
                case ExportFormat.Gltf:
                case ExportFormat.Glb:
                    Log("[warn] glTF/GLB export not implemented yet");
                    break;
            }

            // Optional: export MSCN vertices as separate OBJ layers for validation
            if (options.ExportMscnObj)
            {
                PM4NextExporter.Exporters.MscnObjExporter.Export(scene, outDir, options.LegacyObjParity, options.NameObjectsWithTile);
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
                    case "--ck-split-by-type":
                        opts.CkSplitByType = true;
                        break;
                    case "--export-mscn-obj":
                        opts.ExportMscnObj = true;
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
                _ => throw new ArgumentException("--assembly must be parent-index|msur-indexcount|surface-key|surface-key-aa|composite-hierarchy|container-hierarchy-8bit|composite-bytepair|parent16|mslk-parent|mslk-instance|mslk-instance+ck24")
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
  [--assembly parent-index|msur-indexcount|surface-key|surface-key-aa|composite-hierarchy|container-hierarchy-8bit|composite-bytepair|parent16|mslk-parent|mslk-instance|mslk-instance+ck24]
  [--group parent16|parent16-container|parent16-object|surface|flags|type|sortkey|tile]
          [--parent16-swap] [--csv-diagnostics] [--csv-out <dir>] [--correlate <keyA:keyB>]
  [--batch] [--legacy-obj-parity] [--audit-only] [--no-remap] [--ck-split-by-type]
  [--mslk-parent-min-tris <N>] [--mslk-parent-allow-fallback] [--export-mscn-obj] [--export-tiles] [--project-local] [--name-with-tile]
");
        }
    }
}
