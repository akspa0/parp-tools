using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.CommandLine;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using WoWToolbox.MSCNExplorer.Analysis;
using System.Collections.Generic;
using WoWToolbox.Core.WMO;
using System.CommandLine.Invocation;
using WoWToolbox.Core.Models;
using WoWToolbox.Common.Analysis;
using System.Globalization;

namespace WoWToolbox.MSCNExplorer
{
    class Program
    {
        static void Main(string[] args)
        {
            // Early exit for no args or help request
            if (args.Length == 0 || args.Contains("-h") || args.Contains("--help"))
            {
                // Use System.CommandLine parsing/help later
                Console.WriteLine("MSCNExplorer - Basic Usage (More commands available, see code/use --help):");
                Console.WriteLine("  MSCNExplorer <PM4|PD4 file path> # Basic MSCN dump");
                Console.WriteLine("  MSCNExplorer compare-root <PM4|PD4 file> <root_wmo_file> # Load WMO, compare with PM4 MSCN");
                Console.WriteLine("  MSCNExplorer list-chunks <wmo_group_file>");
                Console.WriteLine("  MSCNExplorer list-root-chunks <wmo_root_file>");
                return;
            }

            // Simple parsing for compare-root before System.CommandLine setup
            if (args.Length == 3 && args[0].Equals("compare-root", StringComparison.OrdinalIgnoreCase))
            {
                string pm4Path = args[1];
                string rootWmoPath = args[2];
                if (!File.Exists(pm4Path) || !File.Exists(rootWmoPath))
                {
                    Console.WriteLine("Error: PM4 or WMO file not found for compare-root.");
                    return;
                }

                WmoGroupMesh? mergedMesh = WmoMeshExporter.LoadMergedWmoMesh(rootWmoPath);
                if (mergedMesh == null || mergedMesh.Vertices.Count == 0 || mergedMesh.Triangles.Count == 0)
                {
                    Console.WriteLine("No valid group meshes loaded or merged mesh is invalid. Aborting analysis.");
                    return;
                }

                PM4File? loadedPm4File = null;
                try
                {
                    loadedPm4File = PM4File.FromFile(pm4Path);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load file as PM4: {ex.Message}");
                    return;
                }
                var mscnChunk = loadedPm4File?.MSCN;
                if (mscnChunk == null)
                {
                    Console.WriteLine("Missing MSCN chunk in PM4.");
                    return;
                }

                var analyzer = new MscnMeshComparisonAnalyzer();
                analyzer.MscnPoints.AddRange(mscnChunk.Vectors.Select(v => new System.Numerics.Vector3(v.X, v.Y, v.Z)));
                analyzer.Mesh = mergedMesh;
                if (analyzer.Mesh == null || analyzer.Mesh.Vertices.Count == 0)
                {
                    Console.WriteLine("Merged mesh is null or empty. Aborting analysis.");
                    return;
                }
                analyzer.AnalyzeProximity(0.1f);

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string outputRoot = Path.Combine("output", $"compare_root_{Path.GetFileNameWithoutExtension(pm4Path)}_{Path.GetFileNameWithoutExtension(rootWmoPath)}_{timestamp}");
                Directory.CreateDirectory(outputRoot);
                string findingsPath = Path.Combine(outputRoot, "findings.txt");
                string mscnObjPath = Path.Combine(outputRoot, "mscn_points.obj");
                string meshObjPath = Path.Combine(outputRoot, "wmo_mesh.obj");

                using (var objWriter = new StreamWriter(mscnObjPath))
                {
                    objWriter.WriteLine("o MSCN_Points");
                    foreach (var v in mscnChunk.Vectors)
                    {
                        objWriter.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");
                    }
                }
                Console.WriteLine($"Exported MSCN points to {mscnObjPath}");

                WmoMeshExporter.SaveMergedWmoToObj(mergedMesh, meshObjPath);
                Console.WriteLine($"Exported merged WMO mesh to {meshObjPath}");

                Console.WriteLine($"Analysis complete. Results logged and OBJ files saved in: {outputRoot}");
                return;
            }

            // If no specific command matched and it wasn't a single file dump
            Console.WriteLine("Invalid arguments or command not recognized. Use -h or --help for options.");
        }
    }
}
