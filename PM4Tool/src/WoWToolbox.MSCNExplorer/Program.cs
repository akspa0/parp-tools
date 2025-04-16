using System;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using WoWToolbox.MSCNExplorer.Analysis;
using System.Collections.Generic;
using WoWToolbox.Core.WMO;
using System.CommandLine;
using System.CommandLine.Invocation;

namespace WoWToolbox.MSCNExplorer
{
    class Program
    {
        static void Main(string[] args)
        {
            // Ensure these are declared at the very top so all code paths can use them
            List<WmoGroupMesh> groupMeshes = new();
            WmoGroupMesh mergedMesh = null;

            if (args.Length == 3 && args[0].Equals("compare", StringComparison.OrdinalIgnoreCase))
            {
                string mscnPath = args[1];
                string wmoPath = args[2];
                if (!File.Exists(mscnPath))
                {
                    Console.WriteLine($"MSCN points file not found: {mscnPath}");
                    return;
                }
                if (!File.Exists(wmoPath))
                {
                    Console.WriteLine($"WMO group file not found: {wmoPath}");
                    return;
                }
                var analyzer = new MscnMeshComparisonAnalyzer();
                analyzer.LoadMscnPoints(mscnPath);
                analyzer.LoadWmoMesh(wmoPath);
                analyzer.AnalyzeProximity(0.1f); // Default threshold
                // Optionally: analyzer.ExportResults(...)
                return;
            }

            // New: compare-pm4 mode
            if (args.Length == 3 && args[0].Equals("compare-pm4", StringComparison.OrdinalIgnoreCase))
            {
                string pm4Path = args[1];
                string wmoPath = args[2];
                if (!File.Exists(pm4Path))
                {
                    Console.WriteLine($"PM4/PD4 file not found: {pm4Path}");
                    return;
                }
                if (!File.Exists(wmoPath))
                {
                    Console.WriteLine($"WMO group file not found: {wmoPath}");
                    return;
                }
                // Load PM4/PD4 and extract MSCN vectors
                PM4File? pm4File = null;
                try
                {
                    pm4File = new PM4File(File.ReadAllBytes(pm4Path));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load file as PM4: {ex.Message}");
                    return;
                }
                var mscnChunk = pm4File.MSCN;
                if (mscnChunk == null)
                {
                    Console.WriteLine("Missing MSCN chunk.");
                    return;
                }
                // Load WMO mesh
                var analyzer = new MscnMeshComparisonAnalyzer();
                analyzer.MscnPoints.AddRange(mscnChunk.Vectors.Select(v => new System.Numerics.Vector3(v.X, v.Y, v.Z)));
                analyzer.LoadWmoMesh(wmoPath);
                analyzer.AnalyzeProximity(0.1f); // Default threshold
                // Optionally: analyzer.ExportResults(...)
                return;
            }

            // Updated: compare-root mode (no groups_dir argument)
            if (args.Length == 3 && args[0].Equals("compare-root", StringComparison.OrdinalIgnoreCase))
            {
                string pm4Path = args[1];
                string rootWmoPath = args[2];
                string groupsDir = Path.GetDirectoryName(rootWmoPath);
                if (!File.Exists(pm4Path))
                {
                    Console.WriteLine($"PM4/PD4 file not found: {pm4Path}");
                    return;
                }
                if (!File.Exists(rootWmoPath))
                {
                    Console.WriteLine($"Root WMO file not found: {rootWmoPath}");
                    return;
                }
                if (string.IsNullOrEmpty(groupsDir) || !Directory.Exists(groupsDir))
                {
                    Console.WriteLine($"Groups directory not found: {groupsDir}");
                    return;
                }
                // Load group file info (count and internal names)
                var (groupCount, internalGroupNames) = WmoRootLoader.LoadGroupInfo(rootWmoPath);
                if (groupCount <= 0)
                {
                     Console.WriteLine($"Root WMO reported {groupCount} groups. Cannot load group files.");
                     return;
                }
                Console.WriteLine($"Root WMO references {groupCount} groups. Internal names found: {internalGroupNames.Count}");

                string rootBaseName = Path.GetFileNameWithoutExtension(rootWmoPath);

                // WMO files are not monolithic; do NOT concatenate root and group files.
                // Instead, parse each group file individually and merge the meshes.
                groupMeshes.Clear();
                for (int i = 0; i < groupCount; i++)
                {
                    string? groupPathToLoad = null;
                    string internalName = (i < internalGroupNames.Count) ? internalGroupNames[i] : null;
                    string numberedName = $"{rootBaseName}_{i:D3}.wmo";

                    // 1. Try finding the file using the internal name from MOGN
                    if (!string.IsNullOrEmpty(internalName))
                    {
                        string potentialInternalPath = Path.Combine(groupsDir, internalName);
                        if (File.Exists(potentialInternalPath))
                        {
                            groupPathToLoad = potentialInternalPath;
                            Console.WriteLine($"[DEBUG] Found group file using internal name: {groupPathToLoad}");
                        }
                        else
                        {
                             Console.WriteLine($"[DEBUG] Group file not found using internal name: {potentialInternalPath}");
                        }
                    }
                    else
                    {
                         Console.WriteLine($"[DEBUG] No internal name available for group index {i}.");
                    }

                    // 2. If not found by internal name, try the numbered convention
                    if (groupPathToLoad == null)
                    {
                        string potentialNumberedPath = Path.Combine(groupsDir, numberedName);
                        if (File.Exists(potentialNumberedPath))
                        {
                            groupPathToLoad = potentialNumberedPath;
                            Console.WriteLine($"[DEBUG] Found group file using numbered convention: {groupPathToLoad}");
                        }
                         else
                        {
                             Console.WriteLine($"[DEBUG] Group file not found using numbered convention: {potentialNumberedPath}");
                        }
                    }

                    // 3. If still not found, skip this group
                    if (groupPathToLoad == null)
                    {
                        Console.WriteLine($"[WARN] Could not find group file for index {i} (tried internal: '{internalName ?? "N/A"}', numbered: '{numberedName}'). Skipping.");
                        continue;
                    }

                    // 4. Load the found group file
                    Console.WriteLine($"[DEBUG] Attempting to load group file: {groupPathToLoad}");
                    try
                    {
                        using var groupStream = File.OpenRead(groupPathToLoad);
                        var mesh = WmoGroupMesh.LoadFromStream(groupStream, groupPathToLoad);
                        groupMeshes.Add(mesh);
                        Console.WriteLine($"[OK] Loaded group: {groupPathToLoad} (Vertices: {mesh.Vertices.Count}, Tris: {mesh.Triangles.Count})");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[ERR] Failed to load group {groupPathToLoad}: {ex.Message}");
                    }
                }
                Console.WriteLine($"Loaded {groupMeshes.Count} of {groupCount} group files.");
                if (groupMeshes.Count == 0)
                {
                    Console.WriteLine("No group meshes loaded. Aborting analysis.");
                    return;
                }
                mergedMesh = WmoGroupMesh.MergeMeshes(groupMeshes);
                // Load MSCN vectors from PM4/PD4
                PM4File? loadedPm4File = null;
                try
                {
                    loadedPm4File = new PM4File(File.ReadAllBytes(pm4Path));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load file as PM4: {ex.Message}");
                    return;
                }
                var mscnChunk = loadedPm4File.MSCN;
                if (mscnChunk == null)
                {
                    Console.WriteLine("Missing MSCN chunk.");
                    return;
                }
                var analyzer = new MscnMeshComparisonAnalyzer();
                analyzer.MscnPoints.AddRange(mscnChunk.Vectors.Select(v => new System.Numerics.Vector3(v.X, v.Y, v.Z)));
                analyzer.Mesh = mergedMesh;
                if (analyzer.Mesh == null || analyzer.Mesh.Vertices == null || analyzer.Mesh.Vertices.Count == 0)
                {
                    Console.WriteLine("Merged mesh is null or empty. Aborting analysis.");
                    return;
                }
                analyzer.AnalyzeProximity(0.1f); // Default threshold
                // Optionally: analyzer.ExportResults(...)
                return;
            }

            // New: list-chunks mode
            if (args.Length == 2 && args[0].Equals("list-chunks", StringComparison.OrdinalIgnoreCase))
            {
                string wmoFile = args[1];
                if (!File.Exists(wmoFile))
                {
                    Console.WriteLine($"File not found: {wmoFile}");
                    return;
                }
                WoWToolbox.Core.WMO.WmoGroupMesh.ListChunks(wmoFile);
                return;
            }

            if (args.Length < 1)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("  MSCNExplorer <PM4|PD4 file path>");
                Console.WriteLine("  MSCNExplorer compare <mscn_points.txt> <wmo_group_file>");
                Console.WriteLine("  MSCNExplorer compare-pm4 <PM4|PD4 file> <wmo_group_file>");
                Console.WriteLine("  MSCNExplorer compare-root <PM4|PD4 file> <root_wmo_file>");
                Console.WriteLine("  MSCNExplorer list-chunks <wmo_file>");
                return;
            }

            string filePath = args[0];
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"File not found: {filePath}");
                return;
            }

            // Prepare output directory
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string outputRoot = Path.Combine("output", $"mscn_{timestamp}");
            Directory.CreateDirectory(outputRoot);
            string findingsPath = Path.Combine(outputRoot, "findings.txt");
            string objPath = Path.Combine(outputRoot, "mscn_points.obj");

            using var findingsWriter = new StreamWriter(findingsPath);
            void Log(string msg)
            {
                Console.WriteLine(msg);
                findingsWriter.WriteLine(msg);
            }

            // If groupMeshes populated, merge them
            if (groupMeshes.Count > 0)
                mergedMesh = WmoGroupMesh.MergeMeshes(groupMeshes);

            // Try to load as PM4 or PD4
            PM4File? pm4FileDefault = null;
            try
            {
                pm4FileDefault = new PM4File(File.ReadAllBytes(filePath));
            }
            catch (Exception ex)
            {
                Log($"Failed to load file as PM4: {ex.Message}");
                return;
            }

            var mscnChunkDefault = pm4FileDefault.MSCN;
            if (mscnChunkDefault == null)
            {
                Log("Missing MSCN chunk.");
                return;
            }

            Log($"MSCN vector count: {mscnChunkDefault.Vectors.Count}");
            Log("First 10 MSCN vectors (Vector3):");
            for (int i = 0; i < Math.Min(10, mscnChunkDefault.Vectors.Count); i++)
            {
                var v = mscnChunkDefault.Vectors[i];
                Log($"  [{i}] (X={v.X:F6}, Y={v.Y:F6}, Z={v.Z:F6})");
            }

            // Export to OBJ (MSCN points)
            using (var objWriter = new StreamWriter(objPath))
            {
                objWriter.WriteLine("o MSCN_Points");
                foreach (var v in mscnChunkDefault.Vectors)
                {
                    objWriter.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");
                }
            }
            Log($"Exported MSCN vectors as points to {objPath}");

            // --- Example: Export merged WMO mesh to OBJ (if loaded) ---
            if (mergedMesh != null && mergedMesh.Vertices.Count > 0)
            {
                string meshObjPath = Path.Combine(outputRoot, "wmo_mesh.obj");
                using (var meshWriter = new StreamWriter(meshObjPath))
                {
                    meshWriter.WriteLine("o WMO_Mesh");
                    foreach (var v in mergedMesh.Vertices)
                        meshWriter.WriteLine($"v {v.Position.X:F6} {v.Position.Y:F6} {v.Position.Z:F6}");
                    foreach (var tri in mergedMesh.Triangles)
                        meshWriter.WriteLine($"f {tri.Index0 + 1} {tri.Index1 + 1} {tri.Index2 + 1}");
                }
                Log($"Exported merged WMO mesh to {meshObjPath}");
            }
            // --- End Example ---

            // --- Hook: Future M2 integration ---
            // Use M2ModelHelper to load and merge M2 doodad geometry here if desired
            // --- End Hook ---

            var rootCommand = new RootCommand("MSCNExplorer: A tool for analyzing MSCN chunk data and related WMO/PM4 structures.");

            var fileArgument = new Argument<FileInfo>("file", "Path to the input file (WMO or PM4).");
            var wmoFileArgument = new Argument<FileInfo>("wmo-file", "Path to the WMO file.");
            var pm4FileArgument = new Argument<FileInfo>("pm4-file", "Path to the PM4 file.");

            // --- list-chunks command ---
            var listChunksCommand = new Command("list-chunks", "Lists all chunks in a given WMO group file.")
            {
                fileArgument
            };
            listChunksCommand.SetHandler((file) =>
            {
                Console.WriteLine($"Listing chunks for: {file.FullName}");
                try
                {
                    WoWToolbox.Core.WMO.WmoGroupMesh.ListChunks(file.FullName);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error listing chunks: {ex.Message}");
                }
            }, fileArgument);
            rootCommand.AddCommand(listChunksCommand);


            // --- compare-root command ---
            var compareRootCommand = new Command("compare-root", "Loads a PM4 and a WMO root, then compares/analyses.")
            {
                pm4FileArgument,
                wmoFileArgument
            };
            compareRootCommand.SetHandler((pm4File, wmoFile) =>
            {
                string rootWmoPath = wmoFile.FullName;
                string groupsDir = Path.GetDirectoryName(rootWmoPath);
                if (string.IsNullOrEmpty(groupsDir) || !Directory.Exists(groupsDir))
                {
                    Console.WriteLine($"Groups directory not found: {groupsDir}");
                    return;
                }
                // Load group file info (count and internal names)
                var (groupCount, internalGroupNames) = WmoRootLoader.LoadGroupInfo(rootWmoPath);
                if (groupCount <= 0)
                {
                     Console.WriteLine($"Root WMO reported {groupCount} groups. Cannot load group files.");
                     return;
                }
                Console.WriteLine($"Root WMO references {groupCount} groups. Internal names found: {internalGroupNames.Count}");

                string rootBaseName = Path.GetFileNameWithoutExtension(rootWmoPath);

                // WMO files are not monolithic; do NOT concatenate root and group files.
                // Instead, parse each group file individually and merge the meshes.
                groupMeshes.Clear();
                for (int i = 0; i < groupCount; i++)
                {
                    string? groupPathToLoad = null;
                    string internalName = (i < internalGroupNames.Count) ? internalGroupNames[i] : null;
                    string numberedName = $"{rootBaseName}_{i:D3}.wmo";

                    // 1. Try finding the file using the internal name from MOGN
                    if (!string.IsNullOrEmpty(internalName))
                    {
                        string potentialInternalPath = Path.Combine(groupsDir, internalName);
                        if (File.Exists(potentialInternalPath))
                        {
                            groupPathToLoad = potentialInternalPath;
                            Console.WriteLine($"[DEBUG] Found group file using internal name: {groupPathToLoad}");
                        }
                        else
                        {
                             Console.WriteLine($"[DEBUG] Group file not found using internal name: {potentialInternalPath}");
                        }
                    }
                    else
                    {
                         Console.WriteLine($"[DEBUG] No internal name available for group index {i}.");
                    }

                    // 2. If not found by internal name, try the numbered convention
                    if (groupPathToLoad == null)
                    {
                        string potentialNumberedPath = Path.Combine(groupsDir, numberedName);
                        if (File.Exists(potentialNumberedPath))
                        {
                            groupPathToLoad = potentialNumberedPath;
                            Console.WriteLine($"[DEBUG] Found group file using numbered convention: {groupPathToLoad}");
                        }
                         else
                        {
                             Console.WriteLine($"[DEBUG] Group file not found using numbered convention: {potentialNumberedPath}");
                        }
                    }

                    // 3. If still not found, skip this group
                    if (groupPathToLoad == null)
                    {
                        Console.WriteLine($"[WARN] Could not find group file for index {i} (tried internal: '{internalName ?? "N/A"}', numbered: '{numberedName}'). Skipping.");
                        continue;
                    }

                    // 4. Load the found group file
                    Console.WriteLine($"[DEBUG] Attempting to load group file: {groupPathToLoad}");
                    try
                    {
                        using var groupStream = File.OpenRead(groupPathToLoad);
                        var mesh = WmoGroupMesh.LoadFromStream(groupStream, groupPathToLoad);
                        groupMeshes.Add(mesh);
                        Console.WriteLine($"[OK] Loaded group: {groupPathToLoad} (Vertices: {mesh.Vertices.Count}, Tris: {mesh.Triangles.Count})");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[ERR] Failed to load group {groupPathToLoad}: {ex.Message}");
                    }
                }
                Console.WriteLine($"Loaded {groupMeshes.Count} of {groupCount} group files.");
                if (groupMeshes.Count == 0)
                {
                    Console.WriteLine("No group meshes loaded. Aborting analysis.");
                    return;
                }
                mergedMesh = WmoGroupMesh.MergeMeshes(groupMeshes);
                // Load MSCN vectors from PM4/PD4
                PM4File? loadedPm4File = null;
                try
                {
                    loadedPm4File = new PM4File(File.ReadAllBytes(pm4File.FullName));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load file as PM4: {ex.Message}");
                    return;
                }
                var mscnChunk = loadedPm4File.MSCN;
                if (mscnChunk == null)
                {
                    Console.WriteLine("Missing MSCN chunk.");
                    return;
                }
                var analyzer = new MscnMeshComparisonAnalyzer();
                analyzer.MscnPoints.AddRange(mscnChunk.Vectors.Select(v => new System.Numerics.Vector3(v.X, v.Y, v.Z)));
                analyzer.Mesh = mergedMesh;
                if (analyzer.Mesh == null || analyzer.Mesh.Vertices == null || analyzer.Mesh.Vertices.Count == 0)
                {
                    Console.WriteLine("Merged mesh is null or empty. Aborting analysis.");
                    return;
                }
                analyzer.AnalyzeProximity(0.1f); // Default threshold
                // Optionally: analyzer.ExportResults(...)
                return;
            }, pm4FileArgument, wmoFileArgument);
            rootCommand.AddCommand(compareRootCommand);

            rootCommand.Invoke(args);
        }
    }
}
