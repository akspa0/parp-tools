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
            var scene = loader.LoadSingleTile(options.InputPath!, options.IncludeAdjacent);
            Log($" scene: vertices={scene.VertexCount} surfaces={scene.SurfaceCount}");
            
            // Choose assembler
            IAssembler assembler = options.AssemblyStrategy switch
            {
                AssemblyStrategy.ParentIndex => new ParentHierarchyAssembler(),
                AssemblyStrategy.MsurIndexCount => new MsurIndexCountAssembler(),
                AssemblyStrategy.SurfaceKey => new SurfaceKeyAssembler(),
                AssemblyStrategy.CompositeHierarchy => new CompositeHierarchyAssembler(),
                AssemblyStrategy.ContainerHierarchy8Bit => new ContainerHierarchy8BitAssembler(),
                AssemblyStrategy.CompositeBytePair => new CompositeBytePairAssembler(),
                AssemblyStrategy.Parent16 => new Parent16Assembler(),
                _ => new ParentHierarchyAssembler()
            };

            var assembled = assembler.Assemble(scene, options).ToList();
            Log($" assembled objects: {assembled.Count}");

            // Export
            switch (options.Format)
            {
                case ExportFormat.Obj:
                    PM4NextExporter.Exporters.ObjExporter.Export(assembled, outDir, options.LegacyObjParity);
                    Log(" exported OBJ");
                    break;
                case ExportFormat.Gltf:
                case ExportFormat.Glb:
                    Log("[warn] glTF/GLB export not implemented yet");
                    break;
            }

            // Diagnostics
            if (options.CsvDiagnostics)
            {
                var diagDir = options.CsvOut ?? outDir;
                DiagnosticsService.WriteSurfaceCsv(diagDir, scene);
                DiagnosticsService.WriteCompositeSummaryCsv(diagDir, scene);
                DiagnosticsService.WriteMscnCsv(diagDir, scene);
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
                "composite-hierarchy" => Model.AssemblyStrategy.CompositeHierarchy,
                "container-hierarchy-8bit" => Model.AssemblyStrategy.ContainerHierarchy8Bit,
                "composite-hierarchy-8bit" => Model.AssemblyStrategy.ContainerHierarchy8Bit,
                "composite-bytepair" => Model.AssemblyStrategy.CompositeBytePair,
                "parent16" => Model.AssemblyStrategy.Parent16,
                _ => throw new ArgumentException("--assembly must be parent-index|msur-indexcount|surface-key|composite-hierarchy|container-hierarchy-8bit|composite-bytepair|parent16")
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
            Console.WriteLine($@"{ToolName}
Usage:
  {ToolName} <pm4Input|directory> [--out <dir>] [--include-adjacent] [--format obj|gltf|glb]
  [--assembly parent-index|msur-indexcount|surface-key|composite-hierarchy|container-hierarchy-8bit|composite-hierarchy-8bit|composite-bytepair|parent16]
  [--group parent16|parent16-container|parent16-object|surface|flags|type|sortkey|tile]
  [--parent16-swap] [--csv-diagnostics] [--csv-out <dir>] [--correlate <keyA:keyB>]
  [--batch] [--legacy-obj-parity] [--audit-only] [--no-remap]
");
        }
    }
}
