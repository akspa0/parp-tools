#nullable enable
using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Threading.Tasks;
using System.Collections.Generic;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.WMO;
using System.Numerics;
using System.IO;
using System.Linq;
using System.Text;
using WoWToolbox.Core.Helpers;
using WoWToolbox.Core.Models;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace WoWToolbox.PM4WmoMatcher
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            var pm4Option = new Option<string>("--pm4", "Path to a PM4 file or directory of PM4 files (required)") { IsRequired = true };
            var wmoOption = new Option<string?>("--wmo", "Path to a directory containing WMO files (required unless --skip-wmo-comparison is used)");
            var outputOption = new Option<string>("--output", () => "./output", "Output directory for results/logs/visualizations (must be a directory, not a file)");
            var maxCandidatesOption = new Option<int>("--max-candidates", () => 5, "Number of top WMO matches to report per PM4 mesh island (default: 5)");
            var minVerticesOption = new Option<int>("--min-vertices", () => 10, "Minimum vertex count for mesh islands to consider (default: 10)");
            var verboseOption = new Option<bool>("--verbose", "Enable verbose logging/diagnostics (optional)");
            var visualizeOption = new Option<bool>("--visualize", "Output aligned mesh files for manual inspection (optional)");
            var exportBaselineOption = new Option<bool>("--export-baseline", "Export baseline OBJ files for the first PM4 and WMO file (full MSVT/MSVI and merged mesh)");
            var exportMprlOption = new Option<bool>("--export-mprl", "Export all MPRL positions from the first PM4 file as a point cloud OBJ (mprl_points.obj) in the output directory.");
            var exportChunksOption = new Option<string?>("--export-pm4-chunks", "Export geometry from individual PM4 chunks (MSVT mesh, MSCN/MSPV/MPRL points) into a specified directory.");
            var skipWmoComparisonOption = new Option<bool>("--skip-wmo-comparison", "Skip WMO comparison and only output PM4 MSCN/MSPV point clouds (optional)");
            var useMslkObjectsOption = new Option<bool>("--use-mslk-objects", "Use individual MSLK scene graph objects for matching instead of combined MSCN+MSPV point clouds (recommended for precision)");
            var preprocessWmoOption = new Option<string?>("--preprocess-wmo", "Preprocess WMO files: extract walkable surfaces and save to cache directory");
            var preprocessPm4Option = new Option<string?>("--preprocess-pm4", "Preprocess PM4 files: extract MSLK objects and save to cache directory");
            var analyzeCacheOption = new Option<string?>("--analyze-cache", "Analyze preprocessed cache directory for PM4/WMO correlations");

            var rootCommand = new RootCommand("PM4–WMO mesh matching tool")
            {
                pm4Option, wmoOption, outputOption, maxCandidatesOption, minVerticesOption, verboseOption, visualizeOption, exportBaselineOption, exportMprlOption, exportChunksOption, skipWmoComparisonOption, useMslkObjectsOption, preprocessWmoOption, preprocessPm4Option, analyzeCacheOption
            };

            rootCommand.SetHandler((InvocationContext ctx) =>
            {
                var pm4 = ctx.ParseResult.GetValueForOption(pm4Option)!;
                var wmo = ctx.ParseResult.GetValueForOption(wmoOption);
                var output = ctx.ParseResult.GetValueForOption(outputOption)!;
                var maxCandidates = ctx.ParseResult.GetValueForOption(maxCandidatesOption);
                var minVertices = ctx.ParseResult.GetValueForOption(minVerticesOption);
                var verbose = ctx.ParseResult.GetValueForOption(verboseOption);
                var visualize = ctx.ParseResult.GetValueForOption(visualizeOption);
                var exportBaseline = ctx.ParseResult.GetValueForOption(exportBaselineOption);
                var exportMprl = ctx.ParseResult.GetValueForOption(exportMprlOption);
                var exportChunks = ctx.ParseResult.GetValueForOption(exportChunksOption);
                var skipWmoComparison = ctx.ParseResult.GetValueForOption(skipWmoComparisonOption);
                var useMslkObjects = ctx.ParseResult.GetValueForOption(useMslkObjectsOption);
                var preprocessWmo = ctx.ParseResult.GetValueForOption(preprocessWmoOption);
                var preprocessPm4 = ctx.ParseResult.GetValueForOption(preprocessPm4Option);
                var analyzeCache = ctx.ParseResult.GetValueForOption(analyzeCacheOption);

                // Enforce output as a directory
                if (File.Exists(output))
                {
                    throw new IOException($"--output must be a directory, but a file exists at: {output}");
                }
                if (!Directory.Exists(output))
                {
                    Directory.CreateDirectory(output);
                }
                string logPath = Path.Combine(output, "pm4_wmo_matcher.log");
                Logger.Init(logPath);
                try
                {
                    Logger.Log("[INFO] PM4–WMO Matcher CLI");
                    Logger.Log($"  --pm4: {pm4}");
                    Logger.Log($"  --wmo: {wmo}");
                    Logger.Log($"  --output: {output}");
                    Logger.Log($"  --max-candidates: {maxCandidates}");
                    Logger.Log($"  --min-vertices: {minVertices}");
                    Logger.Log($"  --verbose: {verbose}");
                    Logger.Log($"  --visualize: {visualize}");
                    Logger.Log($"  --export-baseline: {exportBaseline}");
                    Logger.Log($"  --export-mprl: {exportMprl}");
                    Logger.Log($"  --export-pm4-chunks: {exportChunks}");
                    Logger.Log($"  --skip-wmo-comparison: {skipWmoComparison}");
                    Logger.Log($"  --use-mslk-objects: {useMslkObjects}");
                    Logger.Log($"  --preprocess-wmo: {preprocessWmo}");
                    Logger.Log($"  --preprocess-pm4: {preprocessPm4}");
                    Logger.Log($"  --analyze-cache: {analyzeCache}");

                    // Handle preprocessing modes
                    if (!string.IsNullOrEmpty(preprocessWmo))
                    {
                        Logger.Log("[PREPROCESS] WMO preprocessing mode activated");
                        if (string.IsNullOrEmpty(wmo))
                        {
                            throw new ArgumentException("--wmo is required for WMO preprocessing");
                        }
                        PreprocessWmoFiles(wmo, preprocessWmo, output);
                        return;
                    }

                    if (!string.IsNullOrEmpty(preprocessPm4))
                    {
                        Logger.Log("[PREPROCESS] PM4 preprocessing mode activated");
                        PreprocessPm4Files(pm4, preprocessPm4, output);
                        return;
                    }

                    if (!string.IsNullOrEmpty(analyzeCache))
                    {
                        Logger.Log("[ANALYZE] Cache analysis mode activated");
                        AnalyzePreprocessedCache(analyzeCache, output, maxCandidates, visualize);
                        return;
                    }

                    // Validate that WMO is provided if we're not skipping WMO comparison
                    if (!skipWmoComparison && string.IsNullOrEmpty(wmo))
                    {
                        throw new ArgumentException("--wmo is required unless --skip-wmo-comparison is enabled");
                    }

                    // Batch extract PM4 meshes - NEW: Support MSLK objects
                    Logger.Log("[INFO] Extracting PM4 meshes...");
                    var pm4Meshes = Directory.Exists(pm4) ? 
                        MeshExtractor.ExtractAllFromDirectory(pm4, true, useMslkObjects) : 
                        MeshExtractor.ExtractFromPm4(pm4, useMslkObjects);
                    Logger.Log($"[INFO] Extracted {pm4Meshes.Count} PM4 mesh candidates.");

                    // NEW: Export MSLK objects as individual OBJ files if using MSLK mode
                    if (useMslkObjects && pm4Meshes.Count > 0)
                    {
                        var mslkOutputDir = Path.Combine(output, "mslk_objects");
                        Directory.CreateDirectory(mslkOutputDir);
                        Logger.Log($"[INFO] Exporting {pm4Meshes.Count} MSLK objects to {mslkOutputDir}...");
                        
                        foreach (var mesh in pm4Meshes)
                        {
                            try
                            {
                                var baseName = Path.GetFileNameWithoutExtension(mesh.SourceFile);
                                var objFileName = $"{baseName}_{mesh.SubObjectName}.obj";
                                var objPath = Path.Combine(mslkOutputDir, objFileName);
                                
                                // Export the mesh candidate as an OBJ file
                                mesh.WriteObj(objPath);
                                Logger.Log($"[MSLK_EXPORT] {mesh.SubObjectName}: {mesh.Vertices.Count} vertices → {objFileName}");
                            }
                            catch (Exception ex)
                            {
                                Logger.Log($"[MSLK_EXPORT][ERROR] Failed to export {mesh.SubObjectName}: {ex.Message}");
                            }
                        }
                        Logger.Log($"[INFO] MSLK object export complete: {pm4Meshes.Count} files in {mslkOutputDir}");
                    }

                    // Only extract WMO meshes if we're not skipping WMO comparison
                    List<MeshCandidate> wmoMeshes = new List<MeshCandidate>();
                    if (!skipWmoComparison)
                    {
                        Logger.Log("[INFO] Extracting WMO meshes...");
                        wmoMeshes = Directory.Exists(wmo) ? MeshExtractor.ExtractAllFromDirectory(wmo, false, false) : MeshExtractor.ExtractFromWmo(wmo);
                        Logger.Log($"[INFO] Extracted {wmoMeshes.Count} WMO mesh candidates.");
                    }
                    else
                    {
                        Logger.Log("[INFO] Skipping WMO comparison as requested.");
                    }

                    // Normalize and compute features
                    // If skipping WMO comparison, only process PM4 meshes
                    var meshesToProcess = skipWmoComparison ? pm4Meshes : pm4Meshes.Concat(wmoMeshes);
                    foreach (var mesh in meshesToProcess)
                    {
                        mesh.NormalizeInPlace();
                        var (vCount, tCount, bbox, centroid) = mesh.ComputeFeatures();
                        Logger.Log($"[MESH] {mesh.SourceType} | {mesh.SourceFile} | Verts: {vCount}, Tris: {tCount}, BBox: ({bbox.minX:F2},{bbox.minY:F2},{bbox.minZ:F2})-({bbox.maxX:F2},{bbox.maxY:F2},{bbox.maxZ:F2}), Centroid: ({centroid.cx:F2},{centroid.cy:F2},{centroid.cz:F2})");
                        if (mesh.Indices.Count > 0)
                        {
                            int minIdx = mesh.Indices.Min();
                            int maxIdx = mesh.Indices.Max();
                            Logger.Log($"[MESH] Index range: {minIdx} to {maxIdx} (vertex count: {mesh.Vertices.Count})");
                            if (minIdx < 0 || maxIdx >= mesh.Vertices.Count)
                            {
                                Logger.Log($"[WARN] Mesh indices out of bounds for {mesh.SourceFile} (minIdx: {minIdx}, maxIdx: {maxIdx}, vertex count: {mesh.Vertices.Count})");
                            }
                        }
                    }

                    // Skip WMO comparison if requested
                    if (!skipWmoComparison)
                    {
                        // --- Point cloud matching logic with relaxed filters ---
                        Logger.Log("[INFO] Matching PM4 point clouds to WMO point clouds...");
                        int totalWithMatches = 0, totalSkipped = 0;
                        
                        // Re-enabled Matching Loop
                        foreach (var pm4Mesh in pm4Meshes)
                        {
                            var pm4Points = pm4Mesh.Vertices; // These are combined MSCN/MSPV points
                            if (pm4Points.Count < minVertices) continue;
                            
                            // Scoring list based on Modified Hausdorff Distance
                            var scored = new List<(MeshCandidate wmoMesh, double mhdScore, int pm4Verts, int wmoVerts)>();

                            foreach (var wmoMesh in wmoMeshes)
                            {
                                var wmoPoints = wmoMesh.Vertices;
                                if (wmoPoints.Count == 0) continue;

                                // --- Modified Hausdorff Distance Comparison ---
                                int pm4VertexCount = pm4Points.Count;
                                int wmoVertexCount = wmoPoints.Count;

                                // Calculate Modified Hausdorff Distance 
                                // pm4Points are transformed MSCN/MSPV. WMO points are likely world space.
                                double mhdScore = ModifiedHausdorffDistance(pm4Points, wmoPoints);
                                
                                // Store results
                                scored.Add((wmoMesh, mhdScore, pm4VertexCount, wmoVertexCount));
                                
                                if(verbose)
                                {
                                    // Update logging
                                    Logger.Log($"  [CANDIDATE] WMO: {wmoMesh.SourceFile} (V:{wmoVertexCount}) | MHDScore: {mhdScore:F6}");
                                }
                            }
                            // Sort by Modified Hausdorff Distance (lowest is best)
                            var top = scored.OrderBy(x => x.mhdScore).Take(maxCandidates).ToList();
                            if (top.Count == 0)
                            {
                                totalSkipped++;
                                continue;
                            }
                            totalWithMatches++;
                            // Updated log for PM4 source
                            Logger.Log($"[MATCH] PM4 (MSCN+MSPV): {pm4Mesh.SourceFile} | Verts: {pm4Points.Count}"); 
                            for (int i = 0; i < top.Count; i++)
                            {
                                // Log new scores
                                var (wmoMesh, mhdScore, pm4Verts, wmoVerts) = top[i]; 
                                Logger.Log($"  #{i + 1}: WMO: {wmoMesh.SourceFile} (V:{wmoVerts}) | MHDScore: {mhdScore:F6}"); 
                                if (visualize)
                                { 
                                    try
                                    {
                                        // Create a unique directory for each match
                                        string basePm4Name = Path.GetFileNameWithoutExtension(pm4Mesh.SourceFile);
                                        string baseWmoName = Path.GetFileNameWithoutExtension(wmoMesh.SourceFile);
                                        string matchDir = Path.Combine(output, $"match_{basePm4Name}_to_{baseWmoName}_{i + 1}");
                                        
                                        // Ensure directory exists
                                        if (!Directory.Exists(matchDir))
                                            Directory.CreateDirectory(matchDir);
                                        
                                        // Export PM4 point cloud (MSCN+MSPV with transform applied)
                                        string pm4ObjPath = Path.Combine(matchDir, "pm4.obj");
                                        using (var sw = new StreamWriter(pm4ObjPath, false, Encoding.UTF8))
                                        {
                                            sw.WriteLine("# PM4 Object Point Cloud (MSCN+MSPV, Z Negated)");
                                            sw.WriteLine($"# Source: {pm4Mesh.SourceFile}");
                                            sw.WriteLine($"# Vertex Count: {pm4Points.Count}");
                                            sw.WriteLine($"# MHD Score: {mhdScore:F6}");
                                            sw.WriteLine("# Transformation applied: (X, -Y, -Z) for alignment with MSVT");
                                            foreach (var v in pm4Points)
                                            {
                                                sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
                                            }
                                        }
                                        
                                        // Export WMO point cloud
                                        string wmoObjPath = Path.Combine(matchDir, "wmo.obj");
                                        using (var sw = new StreamWriter(wmoObjPath, false, Encoding.UTF8))
                                        {
                                            sw.WriteLine("# WMO Object Point Cloud");
                                            sw.WriteLine($"# Source: {wmoMesh.SourceFile}");
                                            sw.WriteLine($"# Vertex Count: {wmoMesh.Vertices.Count}");
                                            sw.WriteLine($"# MHD Score: {mhdScore:F6}");
                                            foreach (var v in wmoMesh.Vertices)
                                            {
                                                sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
                                            }
                                        }
                                        
                                        Logger.Log($"    [VIS] Exported match visualization to {matchDir}");
                                    }
                                    catch (Exception ex)
                                    {
                                        Logger.Log($"    [VIS][ERROR] Failed to export visualization: {ex.Message}");
                                    }
                                }
                            }
                        } // End foreach pm4Mesh
                        // End Re-enabled Matching Loop
                        
                        Logger.Log($"[SUMMARY] Objects with matches: {totalWithMatches}, objects skipped: {totalSkipped}");
                        Logger.Log("[DONE] Matching complete.");
                    }
                    else
                    {
                        // If skipping WMO comparison, just output the PM4 MSCN/MSPV point clouds
                        Logger.Log("[INFO] Outputting PM4 MSCN/MSPV point clouds without WMO comparison...");
                        
                        foreach (var pm4Mesh in pm4Meshes)
                        {
                            try
                            {
                                // Create a directory for each PM4 file
                                string basePm4Name = Path.GetFileNameWithoutExtension(pm4Mesh.SourceFile);
                                string outputDir = Path.Combine(output, $"pm4_{basePm4Name}");
                                
                                // Ensure directory exists
                                if (!Directory.Exists(outputDir))
                                    Directory.CreateDirectory(outputDir);
                                
                                // Export PM4 point cloud (MSCN+MSPV with original coordinates)
                                string pm4ObjPath = Path.Combine(outputDir, "pm4_mscn_mspv.obj");
                                using (var sw = new StreamWriter(pm4ObjPath, false, Encoding.UTF8))
                                {
                                    sw.WriteLine("# PM4 Object Point Cloud (MSCN+MSPV combined)");
                                    sw.WriteLine($"# Source: {pm4Mesh.SourceFile}");
                                    sw.WriteLine($"# Vertex Count: {pm4Mesh.Vertices.Count}");
                                    sw.WriteLine("# Original coordinates (not normalized)");
                                    
                                    // Get back original PM4 file for direct extraction to avoid normalization
                                    try
                                    {
                                        var pm4File = WoWToolbox.Core.Navigation.PM4.PM4File.FromFile(pm4Mesh.SourceFile);
                                        
                                        // Extract and write original MSCN points
                                        if (pm4File.MSCN != null)
                                        {
                                            sw.WriteLine($"# MSCN Points: {pm4File.MSCN.ExteriorVertices.Count}");
                                            foreach (var v in pm4File.MSCN.ExteriorVertices)
                                            {
                                                sw.WriteLine($"v {v.X} {v.Y} {v.Z} # MSCN");
                                            }
                                        }
                                        
                                        // Extract and write original MSPV points
                                        if (pm4File.MSPV != null)
                                        {
                                            sw.WriteLine($"# MSPV Points: {pm4File.MSPV.Vertices.Count}");
                                            foreach (var v in pm4File.MSPV.Vertices)
                                            {
                                                sw.WriteLine($"v {v.X} {v.Y} {v.Z} # MSPV");
                                            }
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        // If we can't read the original file, fall back to the normalized points
                                        Logger.Log($"[PM4][WARN] Could not read original PM4 file, falling back to normalized points: {ex.Message}");
                                        sw.WriteLine("# WARNING: Using normalized points (original file could not be read)");
                                        foreach (var v in pm4Mesh.Vertices)
                                        {
                                            sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
                                        }
                                    }
                                }
                                
                                Logger.Log($"[PM4] Exported MSCN/MSPV point cloud for {basePm4Name} to {pm4ObjPath}");
                            }
                            catch (Exception ex)
                            {
                                Logger.Log($"[PM4][ERROR] Failed to export PM4 point cloud: {ex.Message}");
                            }
                        }
                        
                        Logger.Log("[DONE] PM4 point cloud export complete.");
                    }

                    if (exportBaseline)
                    {
                        string baselineDir = Path.Combine(output, "baseline_objs");
                        Directory.CreateDirectory(baselineDir);
                        // Export first PM4
                        var pm4Files = Directory.Exists(pm4) ? Directory.GetFiles(pm4, "*.pm4", SearchOption.AllDirectories) : new[] { pm4 };
                        if (pm4Files.Length > 0)
                        {
                            try
                            {
                                var pm4File = pm4Files[0];
                                var pm4ObjPath = Path.Combine(baselineDir, "pm4.obj");
                                // Use direct MSVT/MSVI extraction for baseline
                                var pm4Data = WoWToolbox.Core.Navigation.PM4.PM4File.FromFile(pm4File);
                                var verts = pm4Data.MSVT?.Vertices;
                                var indices = pm4Data.MSVI?.Indices; // Keep reading indices for logging/potential future use
                                
                                if (verts != null && verts.Count > 0)
                                {
                                    // Log original mesh info if available
                                    if (indices != null && indices.Count > 0)
                                        Logger.Log($"[BASELINE] PM4 raw mesh: {verts.Count} verts, {indices.Count / 3} tris");
                                    else
                                        Logger.Log($"[BASELINE] PM4 raw vertices: {verts.Count} verts (no index data)");

                                    // Export MSVT as Point Cloud OBJ
                                    using var sw = new StreamWriter(pm4ObjPath, false, Encoding.UTF8);
                                    sw.WriteLine("# PM4 Baseline Point Cloud (MSVT)");
                                    sw.WriteLine($"# Source: {pm4File}");
                                    foreach (var v in verts)
                                    {
                                        // Use original coordinates, negating only Y
                                        sw.WriteLine($"v {v.X} {-v.Y} {v.Z}"); 
                                    }
                                    
                                    Logger.Log($"[BASELINE] Exported PM4 baseline POINT CLOUD: {pm4ObjPath} (Verts: {verts.Count})"); // Updated log message
                                }
                                else
                                {
                                    Logger.Log($"[BASELINE] No valid PM4 mesh found in {pm4File}");
                                }
                            }
                            catch (Exception ex)
                            {
                                Logger.Log($"[BASELINE] Failed to export PM4 OBJ: {ex.Message}");
                            }
                        }
                        
                        // Export first WMO only if we're not skipping WMO comparison
                        if (!skipWmoComparison && !string.IsNullOrEmpty(wmo))
                        {
                            var wmoFiles = Directory.Exists(wmo) ? Directory.GetFiles(wmo, "*.wmo", SearchOption.AllDirectories) : new[] { wmo };
                            if (wmoFiles.Length > 0)
                            {
                                try
                                {
                                    var wmoFile = wmoFiles[0];
                                    var wmoObjPath = Path.Combine(baselineDir, "wmo.obj");
                                    // Use the same extraction logic as matching
                                    var baselineWmoMeshes = MeshExtractor.ExtractFromWmo(wmoFile);
                                    if (baselineWmoMeshes != null && baselineWmoMeshes.Count > 0)
                                    {
                                        var mesh = baselineWmoMeshes[0];
                                        Logger.Log($"[BASELINE] WMO extracted mesh: {mesh.Vertices.Count} verts, {mesh.Indices.Count / 3} tris");
                                        for (int vi = 0; vi < Math.Min(3, mesh.Vertices.Count); vi++)
                                            Logger.Log($"[BASELINE] WMO v[{vi}]: {mesh.Vertices[vi].X}, {mesh.Vertices[vi].Y}, {mesh.Vertices[vi].Z}");
                                        for (int ti = 0; ti < Math.Min(3, mesh.Indices.Count / 3); ti++)
                                        {
                                            int a = mesh.Indices[ti * 3], b = mesh.Indices[ti * 3 + 1], c = mesh.Indices[ti * 3 + 2];
                                            Logger.Log($"[BASELINE] WMO f[{ti}]: {a}, {b}, {c}");
                                        }
                                        using var sw = new StreamWriter(wmoObjPath, false, Encoding.UTF8);
                                        foreach (var v in mesh.Vertices)
                                            sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
                                        for (int i = 0; i + 2 < mesh.Indices.Count; i += 3)
                                        {
                                            int a = mesh.Indices[i], b = mesh.Indices[i + 1], c = mesh.Indices[i + 2];
                                            if (a < 0 || b < 0 || c < 0 || a >= mesh.Vertices.Count || b >= mesh.Vertices.Count || c >= mesh.Vertices.Count)
                                            {
                                                Logger.Log($"[BASELINE][WARN] Skipping face with out-of-bounds index: f {a} {b} {c} (vertex count: {mesh.Vertices.Count})");
                                                continue;
                                            }
                                            sw.WriteLine($"f {a + 1} {b + 1} {c + 1}");
                                        }
                                        Logger.Log($"[BASELINE] Exported WMO OBJ (extracted mesh): {wmoObjPath} (Verts: {mesh.Vertices.Count}, Tris: {mesh.Indices.Count / 3})");
                                    }
                                    else
                                    {
                                        Logger.Log($"[BASELINE] No valid WMO mesh found in {wmoFile}");
                                    }
                                }
                                catch (Exception ex)
                                {
                                    Logger.Log($"[BASELINE] Failed to export WMO OBJ: {ex.Message}");
                                }
                            }
                        }
                    }

                    if (exportMprl)
                    {
                        var pm4Files = Directory.Exists(pm4) ? Directory.GetFiles(pm4, "*.pm4", SearchOption.AllDirectories) : new[] { pm4 };
                        if (pm4Files.Length > 0)
                        {
                            var pm4File = pm4Files[0];
                            var pm4Data = WoWToolbox.Core.Navigation.PM4.PM4File.FromFile(pm4File);
                            var mprl = pm4Data.MPRL;
                            if (mprl != null && mprl.Entries.Count > 0)
                            {
                                string mprlObjPath = Path.Combine(output, "mprl_points.obj");
                                using var sw = new StreamWriter(mprlObjPath, false, Encoding.UTF8);
                                sw.WriteLine("# MPRL positions exported as point cloud");
                                sw.WriteLine($"# PM4 file: {pm4File}");
                                sw.WriteLine($"# Entries: {mprl.Entries.Count}");
                                foreach (var entry in mprl.Entries)
                                {
                                    var p = entry.Position;
                                    sw.WriteLine($"v {p.X} {p.Y} {p.Z}");
                                }
                                Logger.Log($"[MPRL] Exported {mprl.Entries.Count} MPRL positions to {mprlObjPath}");
                            }
                            else
                            {
                                Logger.Log($"[MPRL] No MPRL entries found in {pm4File}");
                            }
                        }
                        else
                        {
                            Logger.Log("[MPRL] No PM4 files found for MPRL export.");
                        }
                    }

                    // Add a helper method to extract and merge all PM4 point cloud data
                    if (!skipWmoComparison) // Only do unified export if not skipping WMO comparison
                    {
                        Logger.Log("[INFO] Generating unified PM4 point clouds...");
                        if (Directory.Exists(pm4))
                        {
                            var pm4Files = Directory.GetFiles(pm4, "*.pm4", SearchOption.AllDirectories);
                            int successCount = 0, errorCount = 0;
                            foreach (var pm4File in pm4Files)
                            {
                                try
                                {
                                    ExportUnifiedPm4PointCloud(pm4File, output);
                                    successCount++;
                                }
                                catch (Exception ex)
                                {
                                    Logger.Log($"[ERROR] Failed to process {Path.GetFileName(pm4File)}: {ex.Message}");
                                    errorCount++;
                                }
                            }
                            Logger.Log($"[INFO] Unified point cloud generation complete. Success: {successCount}, Errors: {errorCount}");
                        }
                        else if (File.Exists(pm4) && pm4.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase))
                        {
                            try
                            {
                                ExportUnifiedPm4PointCloud(pm4, output);
                                Logger.Log("[INFO] Unified point cloud generation complete.");
                            }
                            catch (Exception ex)
                            {
                                Logger.Log($"[ERROR] Failed to process unified point cloud: {ex.Message}");
                            }
                        }
                    }

                    // Export individual PM4 chunks if requested
                    if (!string.IsNullOrEmpty(exportChunks))
                    {
                        Logger.Log($"[INFO] Exporting individual PM4 chunks to: {exportChunks}");
                        if (!Directory.Exists(exportChunks))
                        {
                            Directory.CreateDirectory(exportChunks);
                        }
                        
                        var pm4FilesToProcess = Directory.Exists(pm4)
                            ? Directory.GetFiles(pm4, "*.pm4", SearchOption.AllDirectories)
                            : new[] { pm4 };
                            
                        foreach (var pm4File in pm4FilesToProcess)
                        {
                            try
                            {
                                Logger.Log($"  Processing PM4 for chunk export: {Path.GetFileName(pm4File)}");
                                var pm4Data = WoWToolbox.Core.Navigation.PM4.PM4File.FromFile(pm4File);
                                string pm4BaseName = Path.GetFileNameWithoutExtension(pm4File);
                                string chunkOutputDir = Path.Combine(exportChunks, pm4BaseName);
                                Directory.CreateDirectory(chunkOutputDir); // Ensure subdirectory exists

                                // Export MSVT Mesh
                                if (pm4Data.MSVT != null && pm4Data.MSVI != null)
                                {
                                    ExportMsvtMesh(pm4Data.MSVT, pm4Data.MSVI, chunkOutputDir, pm4BaseName);
                                }

                                // Export COMBINED MSCN/MSPV Points
                                string combinedMscnMspvPath = Path.Combine(chunkOutputDir, $"{pm4BaseName}_mscn_mspv_points.obj");
                                // Clear file if it exists before appending
                                if(File.Exists(combinedMscnMspvPath)) File.Delete(combinedMscnMspvPath);

                                if (pm4Data.MSCN != null)
                                {
                                    ExportCombinedPointCloud(pm4Data.MSCN.ExteriorVertices.Select(v => (v.X, v.Y, v.Z)), combinedMscnMspvPath, pm4BaseName, "MSCN");
                                }
                                if (pm4Data.MSPV != null)
                                {
                                    ExportCombinedPointCloud(pm4Data.MSPV.Vertices.Select(v => (v.X, v.Y, v.Z)), combinedMscnMspvPath, pm4BaseName, "MSPV");
                                }

                                // Export MPRL Points (Remains separate)
                                if (pm4Data.MPRL != null)
                                {
                                    // Use the original ExportPointCloud helper for MPRL
                                    // Exports with original coordinates (X, Y, Z)
                                    ExportPointCloud(pm4Data.MPRL.Entries.Select(e => (e.Position.X, e.Position.Y, e.Position.Z)), chunkOutputDir, pm4BaseName, "mprl");
                                }

                                // Add call to export complete mesh with both MSVT and MSCN data
                                ExportCompletePm4Mesh(pm4File, chunkOutputDir);
                            }
                            catch(Exception ex)
                            {
                                Logger.Log($"[ERROR] Failed processing {Path.GetFileName(pm4File)} for chunk export: {ex.Message}");
                            }
                        }
                        Logger.Log($"[INFO] Finished exporting PM4 chunks.");
                    }

                    // New function to export a complete PM4 mesh combining MSVT mesh data with MSCN exterior vertices
                    ExportCompletePm4Mesh(pm4, output);
                }
                finally
                {
                    Logger.Close();
                }
            });

            return await rootCommand.InvokeAsync(args);
        }

        // Helper: check if any parent in the path is a file
        private static bool AnyParentIsFile(string path)
        {
            var dir = Path.GetFullPath(path);
            while (!string.IsNullOrEmpty(dir) && dir != Path.GetPathRoot(dir))
            {
                if (File.Exists(dir))
                    return true;
                dir = Path.GetDirectoryName(dir);
            }
            return false;
        }

        // Helper to load the first group mesh for a WMO root file
        private static WoWToolbox.Core.WMO.WmoGroupMesh? LoadFirstWmoGroupMesh(string wmoRootPath)
        {
            string groupsDir = Path.GetDirectoryName(wmoRootPath) ?? ".";
            string rootBaseName = Path.GetFileNameWithoutExtension(wmoRootPath);
            var (groupCount, internalGroupNames) = WoWToolbox.Core.WMO.WmoRootLoader.LoadGroupInfo(wmoRootPath);
            if (groupCount <= 0)
                return null;
            string? groupPathToLoad = WoWToolbox.Core.WMO.WmoMeshExporter.FindGroupFilePath(0, rootBaseName, groupsDir, internalGroupNames);
            if (groupPathToLoad == null)
                return null;
            using var groupStream = File.OpenRead(groupPathToLoad);
            return WoWToolbox.Core.WMO.WmoGroupMesh.LoadFromStream(groupStream, groupPathToLoad);
        }

        // *** REMOVED AverageNearestNeighbor function definition ***
        /*
        static double AverageNearestNeighbor(List<(float X, float Y, float Z)> src, List<(float X, float Y, float Z)> dst)
        { 
            // ... old implementation ...
        }
        */
        
        // +++ ADDED Modified Hausdorff Distance Helpers +++
        // Calculates the average distance to the nearest point for all points in a cloud.
        static double AverageDistanceToNearest(List<(float X, float Y, float Z)> cloudA, List<(float X, float Y, float Z)> cloudB)
        {
            if (cloudA.Count == 0 || cloudB.Count == 0) return double.MaxValue;
            double totalMinDistance = 0;
            foreach (var pA in cloudA)
            {
                double minSqDist = double.MaxValue;
                foreach (var pB in cloudB)
                {
                    double dx = pA.X - pB.X;
                    double dy = pA.Y - pB.Y;
                    double dz = pA.Z - pB.Z;
                    double sqDist = dx * dx + dy * dy + dz * dz;
                    if (sqDist < minSqDist)
                    {
                        minSqDist = sqDist;
                    }
                }
                // If minSqDist remains MaxValue, cloudB was empty, return MaxValue or handle error
                if (minSqDist == double.MaxValue) return double.MaxValue; 
                totalMinDistance += Math.Sqrt(minSqDist); 
            }
            return totalMinDistance / cloudA.Count;
        }

        // Calculates the Modified Hausdorff Distance (max of the two average nearest neighbor distances)
        static double ModifiedHausdorffDistance(List<(float X, float Y, float Z)> cloudA, List<(float X, float Y, float Z)> cloudB)
        {
            double avgDistAtoB = AverageDistanceToNearest(cloudA, cloudB);
            double avgDistBtoA = AverageDistanceToNearest(cloudB, cloudA);
            // Handle cases where one cloud might be empty
            if (avgDistAtoB == double.MaxValue || avgDistBtoA == double.MaxValue) 
                return double.MaxValue;
            return Math.Max(avgDistAtoB, avgDistBtoA);
        }

        // Add a helper method to extract and merge all PM4 point cloud data
        private static void ExportUnifiedPm4PointCloud(string pm4Path, string outputDir)
        {
            try
            {
                var pm4 = WoWToolbox.Core.Navigation.PM4.PM4File.FromFile(pm4Path);
                var points = new Dictionary<(float X, float Y, float Z), HashSet<string>>();
                float tol = 1e-4f;

                // Helper to find an existing key within tolerance
                (float X, float Y, float Z)? FindExistingKey((float X, float Y, float Z) p)
                {
                    foreach (var key in points.Keys)
                    {
                        if (Math.Abs(key.X - p.X) < tol && Math.Abs(key.Y - p.Y) < tol && Math.Abs(key.Z - p.Z) < tol)
                        {
                            return key;
                        }
                    }
                    return null;
                }

                // Helper to add or update point data
                void AddPoint((float X, float Y, float Z) p, string tag)
                {
                    var existingKey = FindExistingKey(p);
                    if (existingKey.HasValue)
                    {
                        points[existingKey.Value].Add(tag);
                    }
                    else
                    {
                        points[p] = new HashSet<string> { tag };
                    }
                }

                // MSVT (mesh vertices)
                if (pm4.MSVT != null)
                {
                    foreach (var v in pm4.MSVT.Vertices)
                    {
                        var worldPos = ToUnifiedWorld(v.ToWorldCoordinates());
                        AddPoint((worldPos.X, worldPos.Y, worldPos.Z), "mesh");
                    }
                }
                // MSCN (exterior)
                if (pm4.MSCN != null)
                {
                    foreach (var v in pm4.MSCN.ExteriorVertices)
                    {
                        var vector3 = new Vector3(v.X, v.Y, v.Z);
                        var worldPos = ToUnifiedWorld(vector3);
                        AddPoint((worldPos.X, worldPos.Y, worldPos.Z), "exterior");
                    }
                }
                // MSPV (path vertices)
                if (pm4.MSPV != null)
                {
                    foreach (var v in pm4.MSPV.Vertices)
                    {
                        var vector3 = new Vector3(v.X, v.Y, v.Z);
                        var worldPos = ToUnifiedWorld(vector3);
                        AddPoint((worldPos.X, worldPos.Y, worldPos.Z), "path");
                    }
                }
                // MPRL (reference points)
                if (pm4.MPRL != null)
                {
                    foreach (var entry in pm4.MPRL.Entries)
                    {
                        var v = entry.Position;
                        var vector3 = new Vector3(v.X, v.Y, v.Z);
                        var worldPos = ToUnifiedWorld(vector3);
                        AddPoint((worldPos.X, worldPos.Y, worldPos.Z), "reference");
                    }
                }
                // Output OBJ
                string outPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(pm4Path) + "_pointcloud.obj");
                using var sw = new StreamWriter(outPath, false, Encoding.UTF8);
                sw.WriteLine("# Unified PM4 point cloud: mesh, exterior, path, reference");
                foreach (var kvp in points)
                {
                    var t = kvp.Key;
                    var tags = string.Join(",", kvp.Value);
                    sw.WriteLine($"v {t.X} {t.Y} {t.Z} # {tags}");
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[ERROR] Failed to export unified point cloud for {Path.GetFileName(pm4Path)}: {ex.Message}");
            }
        }

        // +++ ADDED Helper Functions for Chunk Export +++
        private static void ExportMsvtMesh(WoWToolbox.Core.Navigation.PM4.Chunks.MSVTChunk msvt, WoWToolbox.Core.Navigation.PM4.Chunks.MSVIChunk msvi, string outputDir, string baseName)
        {
            string objPath = Path.Combine(outputDir, $"{baseName}_msvt_mesh.obj");
            var verts = msvt.Vertices;
            var indices = msvi.Indices;
            try
            {
                using var sw = new StreamWriter(objPath, false, Encoding.UTF8);
                sw.WriteLine($"# MSVT Mesh for {baseName}");
                sw.WriteLine($"# Vertices: {verts.Count}, Indices: {indices.Count}");

                foreach (var v in verts)
                {
                    var worldPos = ToUnifiedWorld(v.ToWorldCoordinates());
                    sw.WriteLine($"v {worldPos.X} {worldPos.Y} {worldPos.Z}");
                }

                sw.WriteLine("# Faces");
                for (int i = 0; i + 2 < indices.Count; i += 3)
                {
                    int a = (int)indices[i]; int b = (int)indices[i + 1]; int c = (int)indices[i + 2];
                    if (a < 0 || b < 0 || c < 0 || a >= verts.Count || b >= verts.Count || c >= verts.Count)
                    {
                        // Log warning? Skip for now in helper.
                        continue;
                    }
                    sw.WriteLine($"f {a + 1} {b + 1} {c + 1}");
                }
                Logger.Log($"    Exported MSVT mesh to {objPath}");
            }
            catch (Exception ex)
            {
                Logger.Log($"[ERROR] Failed to export MSVT mesh to {objPath}: {ex.Message}");
            }
        }

        // Helper for exporting MSCN/MSPV points combined (Appends)
        private static void ExportCombinedPointCloud(IEnumerable<(float X, float Y, float Z)> points, string outputPath, string baseName, string chunkSource)
        {
             try
            {
                // Use StreamWriter with append = true
                using var sw = new StreamWriter(outputPath, true, Encoding.UTF8);
                if (new FileInfo(outputPath).Length == 0) // Write header only if file is new/empty
                {
                    sw.WriteLine($"# Combined MSCN/MSPV Points for {baseName}");
                }
                sw.WriteLine($"# Appending {chunkSource} points"); // Add comment indicating source
                int count = 0;
                foreach (var p in points)
                {
                    var vector3 = new Vector3(p.X, p.Y, p.Z);
                    var worldPos = ToUnifiedWorld(vector3);
                    sw.WriteLine($"v {worldPos.X} {worldPos.Y} {worldPos.Z}"); 
                    count++;
                }
                // Log count for this specific appended chunk
                Logger.Log($"    Appended {chunkSource} points ({count}) to {outputPath}");
            }
            catch (Exception ex)
            {
                Logger.Log($"[ERROR] Failed to append {chunkSource} points to {outputPath}: {ex.Message}");
            }
        }

        // Original helper, now only used for MPRL
        private static void ExportPointCloud(IEnumerable<(float X, float Y, float Z)> points, string outputDir, string baseName, string chunkName)
        {
            string objPath = Path.Combine(outputDir, $"{baseName}_{chunkName}_points.obj");
            try
            {
                using var sw = new StreamWriter(objPath, false, Encoding.UTF8);
                sw.WriteLine($"# {chunkName.ToUpperInvariant()} Points for {baseName}");
                int count = 0;
                foreach (var p in points)
                {
                    var vector3 = new Vector3(p.X, p.Y, p.Z);
                    var worldPos = ToUnifiedWorld(vector3);
                    sw.WriteLine($"v {worldPos.X} {worldPos.Y} {worldPos.Z}"); 
                    count++;
                }
                sw.WriteLine($"# Total points: {count}");
                Logger.Log($"    Exported {chunkName.ToUpperInvariant()} points ({count}) to {objPath}");
            }
            catch (Exception ex)
            {
                Logger.Log($"[ERROR] Failed to export {chunkName.ToUpperInvariant()} points to {objPath}: {ex.Message}");
            }
        }

        // New function to export a complete PM4 mesh combining MSVT mesh data with MSCN exterior vertices
        private static void ExportCompletePm4Mesh(string pm4Path, string outputDir)
        {
            try
            {
                var pm4Data = WoWToolbox.Core.Navigation.PM4.PM4File.FromFile(pm4Path);
                string basename = Path.GetFileNameWithoutExtension(pm4Path);
                string objPath = Path.Combine(outputDir, $"{basename}_complete_mesh.obj");
                
                // Only proceed if we have MSVT
                if (pm4Data.MSVT == null)
                {
                    Logger.Log($"[ERROR] Cannot export complete mesh: {basename} is missing MSVT chunk");
                    return;
                }
                
                using var sw = new StreamWriter(objPath, false, Encoding.UTF8);
                sw.WriteLine($"# Complete PM4 Mesh for {basename} - All Geometry with Proper Faces");
                sw.WriteLine($"# Exported: {DateTime.Now}");
                sw.WriteLine($"# Features: MSVT render mesh + MSCN collision + MSPV structure with MSLK linking");
                
                // Track all vertices with their source chunk and vertex offsets
                var allVertices = new List<(float X, float Y, float Z, string Source)>();
                int mscnVertexOffset = 0;
                int mspvVertexOffset = 0;
                
                // 1. Add all MSVT vertices (render mesh)
                sw.WriteLine("# MSVT Render Vertices");
                foreach (var v in pm4Data.MSVT.Vertices)
                {
                    var worldPos = ToUnifiedWorld(v.ToWorldCoordinates());
                    allVertices.Add((worldPos.X, worldPos.Y, worldPos.Z, "MSVT"));
                }
                mscnVertexOffset = allVertices.Count; // MSCN vertices start after MSVT
                
                // 2. Add MSCN collision vertices if available
                if (pm4Data.MSCN != null && pm4Data.MSCN.ExteriorVertices.Count > 0)
                {
                    sw.WriteLine($"# MSCN Collision Vertices");
                    foreach (var v in pm4Data.MSCN.ExteriorVertices)
                    {
                        // Convert C3Vector to Vector3 for coordinate transformation
                        var vector3 = new Vector3(v.X, v.Y, v.Z);
                        var worldPos = ToUnifiedWorld(vector3);
                        allVertices.Add((worldPos.X, worldPos.Y, worldPos.Z, "MSCN"));
                    }
                }
                mspvVertexOffset = allVertices.Count; // MSPV vertices start after MSCN
                
                // 3. Add MSPV structure vertices if available
                if (pm4Data.MSPV != null && pm4Data.MSPV.Vertices.Count > 0)
                {
                    sw.WriteLine($"# MSPV Structure Vertices");
                    foreach (var v in pm4Data.MSPV.Vertices)
                    {
                        // Convert C3Vector to Vector3 for coordinate transformation
                        var vector3 = new Vector3(v.X, v.Y, v.Z);
                        var worldPos = ToUnifiedWorld(vector3);
                        allVertices.Add((worldPos.X, worldPos.Y, worldPos.Z, "MSPV"));
                    }
                }
                
                // Write all vertices
                foreach (var v in allVertices)
                {
                    sw.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6} # {v.Source}");
                }
                sw.WriteLine();
                
                int totalFaces = 0;
                
                // RENDER MESH FACES: Use MSVI indices for MSVT vertices
                if (pm4Data.MSVI != null && pm4Data.MSVI.Indices.Count >= 3)
                {
                    sw.WriteLine("# MSVT Render Mesh Faces (via MSVI indices)");
                    sw.WriteLine("o MSVT_RenderMesh");
                    
                    for (int i = 0; i + 2 < pm4Data.MSVI.Indices.Count; i += 3)
                    {
                        uint idx1 = pm4Data.MSVI.Indices[i];
                        uint idx2 = pm4Data.MSVI.Indices[i + 1];
                        uint idx3 = pm4Data.MSVI.Indices[i + 2];
                        
                        // Validate indices are within MSVT range
                        if (idx1 < pm4Data.MSVT.Vertices.Count && 
                            idx2 < pm4Data.MSVT.Vertices.Count && 
                            idx3 < pm4Data.MSVT.Vertices.Count &&
                            idx1 != idx2 && idx1 != idx3 && idx2 != idx3)
                        {
                            // OBJ uses 1-based indexing
                            sw.WriteLine($"f {idx1 + 1} {idx2 + 1} {idx3 + 1}");
                            totalFaces++;
                        }
                    }
                    sw.WriteLine();
                }
                
                // STRUCTURE FACES: Use MSLK entries with MSPI indices for MSCN/MSPV geometry
                if (pm4Data.MSLK != null && pm4Data.MSPI != null && 
                    pm4Data.MSLK.Entries.Count > 0 && pm4Data.MSPI.Indices.Count > 0)
                {
                    sw.WriteLine("# Structure Faces (via MSLK->MSPI linking to MSCN/MSPV)");
                    sw.WriteLine("o Structure_Geometry");
                    
                    foreach (var mslkEntry in pm4Data.MSLK.Entries)
                    {
                        // MSLK entries reference MSPI indices which point to MSCN/MSPV vertices
                        if (mslkEntry.MspiFirstIndex >= 0 && 
                            mslkEntry.MspiIndexCount >= 3 && 
                            mslkEntry.MspiFirstIndex + mslkEntry.MspiIndexCount <= pm4Data.MSPI.Indices.Count)
                        {
                            // Get the vertex indices from MSPI
                            var structureIndices = new List<uint>();
                            for (int i = 0; i < mslkEntry.MspiIndexCount; i++)
                            {
                                uint mspiIdx = pm4Data.MSPI.Indices[mslkEntry.MspiFirstIndex + i];
                                structureIndices.Add(mspiIdx);
                            }
                            
                            // Determine which geometry chunk these indices reference
                            // Based on our analysis: lower indices usually reference MSCN, higher ones MSPV
                            var validIndices = new List<uint>();
                            foreach (uint idx in structureIndices)
                            {
                                uint adjustedIdx = 0;
                                bool isValid = false;
                                
                                // Try MSCN first (collision geometry)
                                if (pm4Data.MSCN != null && idx < pm4Data.MSCN.ExteriorVertices.Count)
                                {
                                    adjustedIdx = idx + (uint)mscnVertexOffset + 1; // +1 for OBJ 1-based indexing
                                    isValid = true;
                                }
                                // Try MSPV (structure geometry)
                                else if (pm4Data.MSPV != null && idx < pm4Data.MSPV.Vertices.Count)
                                {
                                    adjustedIdx = idx + (uint)mspvVertexOffset + 1; // +1 for OBJ 1-based indexing
                                    isValid = true;
                                }
                                
                                if (isValid)
                                {
                                    validIndices.Add(adjustedIdx);
                                }
                            }
                            
                            // Create triangular faces using triangle fan pattern
                            if (validIndices.Count >= 3)
                            {
                                for (int i = 1; i < validIndices.Count - 1; i++)
                                {
                                    uint v1 = validIndices[0];     // Fan center
                                    uint v2 = validIndices[i];     // Current edge
                                    uint v3 = validIndices[i + 1]; // Next edge
                                    
                                    // Validate adjusted indices are within our total vertex count
                                    if (v1 <= allVertices.Count && v2 <= allVertices.Count && v3 <= allVertices.Count)
                                    {
                                        sw.WriteLine($"f {v1} {v2} {v3}");
                                        totalFaces++;
                                    }
                                }
                            }
                        }
                    }
                    sw.WriteLine();
                }
                
                // Statistics
                int msvtCount = pm4Data.MSVT.Vertices.Count;
                int mscnCount = pm4Data.MSCN?.ExteriorVertices.Count ?? 0;
                int mspvCount = pm4Data.MSPV?.Vertices.Count ?? 0;
                
                Logger.Log($"[PM4_COMPLETE] Exported complete mesh for {basename}:");
                Logger.Log($"[PM4_COMPLETE]   Total vertices: {allVertices.Count} (MSVT: {msvtCount}, MSCN: {mscnCount}, MSPV: {mspvCount})");
                Logger.Log($"[PM4_COMPLETE]   Total faces: {totalFaces}");
                Logger.Log($"[PM4_COMPLETE]   Output: {objPath}");
            }
            catch (Exception ex)
            {
                Logger.Log($"[ERROR] Failed to export complete PM4 mesh for {Path.GetFileName(pm4Path)}: {ex.Message}");
            }
        }

        // Add helper function near other helpers:
        private static Vector3 ToUnifiedWorld(Vector3 v)
        {
            // Apply: X' = -Y, Y' = -Z, Z' = X
            // This corrects horizontal mirroring and Z inversion for combined mesh exports.
            return new Vector3(-v.Y, -v.Z, v.X);
        }

        /// <summary>
        /// Preprocess WMO files: extract walkable surfaces and save as .obj files
        /// </summary>
        private static void PreprocessWmoFiles(string wmoPath, string cacheDir, string outputDir)
        {
            try
            {
                Directory.CreateDirectory(outputDir);
                var wmoOutputDir = Path.Combine(outputDir, "wmo_walkable");
                Directory.CreateDirectory(wmoOutputDir);
                
                Logger.Log($"[PREPROCESS_WMO] Processing WMO files from: {wmoPath}");
                Logger.Log($"[PREPROCESS_WMO] Output directory: {wmoOutputDir}");
                
                var wmoFiles = Directory.Exists(wmoPath) ? 
                    Directory.GetFiles(wmoPath, "*.wmo", SearchOption.AllDirectories) : 
                    new[] { wmoPath };
                
                int processedCount = 0;
                int totalMeshes = 0;
                
                foreach (var wmoFile in wmoFiles)
                {
                    try
                    {
                        Logger.Log($"[PREPROCESS_WMO] Processing: {Path.GetFileName(wmoFile)}");
                        
                        // Extract walkable surfaces
                        var candidates = MeshExtractor.ExtractFromWmo(wmoFile);
                        
                        foreach (var candidate in candidates)
                        {
                            var baseName = Path.GetFileNameWithoutExtension(wmoFile);
                            var objFileName = $"wmo_{baseName}_{candidate.SubObjectName}.obj";
                            var objPath = Path.Combine(wmoOutputDir, objFileName);
                            
                            // Write as .obj file
                            candidate.WriteObj(objPath);
                            totalMeshes++;
                            
                            Logger.Log($"[PREPROCESS_WMO]   → {candidate.Vertices.Count} walkable points → {objFileName}");
                        }
                        
                        processedCount++;
                    }
                    catch (Exception ex)
                    {
                        Logger.Log($"[PREPROCESS_WMO][ERROR] Failed to process {wmoFile}: {ex.Message}");
                    }
                }
                
                Logger.Log($"[PREPROCESS_WMO] Completed: {processedCount} WMO files, {totalMeshes} walkable surface .obj files");
                Logger.Log($"[PREPROCESS_WMO] Output: {wmoOutputDir}");
            }
            catch (Exception ex)
            {
                Logger.Log($"[PREPROCESS_WMO][ERROR] {ex.Message}");
            }
        }

        /// <summary>
        /// Preprocess PM4 files: extract MSLK objects and save as .obj files
        /// </summary>
        private static void PreprocessPm4Files(string pm4Path, string cacheDir, string outputDir)
        {
            try
            {
                Directory.CreateDirectory(outputDir);
                var pm4OutputDir = Path.Combine(outputDir, "pm4_mslk");
                Directory.CreateDirectory(pm4OutputDir);
                
                Logger.Log($"[PREPROCESS_PM4] Processing PM4 files from: {pm4Path}");
                Logger.Log($"[PREPROCESS_PM4] Output directory: {pm4OutputDir}");
                
                var pm4Files = Directory.Exists(pm4Path) ? 
                    Directory.GetFiles(pm4Path, "*.pm4", SearchOption.AllDirectories) : 
                    new[] { pm4Path };
                
                int processedCount = 0;
                int totalMeshes = 0;
                
                foreach (var pm4File in pm4Files)
                {
                    try
                    {
                        Logger.Log($"[PREPROCESS_PM4] Processing: {Path.GetFileName(pm4File)}");
                        
                        // Extract MSLK objects (use single object per file for clean testing)
                        var candidates = MeshExtractor.ExtractFromPm4(pm4File, useMslkObjects: true);
                        
                        foreach (var candidate in candidates.Take(1)) // Take only first object for clean testing
                        {
                            // Only process objects with meaningful geometry
                            if (candidate.Vertices.Count < 50) continue;
                            
                            var baseName = Path.GetFileNameWithoutExtension(pm4File);
                            var objFileName = $"pm4_{baseName}_{candidate.SubObjectName}.obj";
                            var objPath = Path.Combine(pm4OutputDir, objFileName);
                            
                            // Write as .obj file
                            candidate.WriteObj(objPath);
                            totalMeshes++;
                            
                            Logger.Log($"[PREPROCESS_PM4]   → {candidate.Vertices.Count} vertices, {candidate.Indices.Count / 3} triangles → {objFileName}");
                        }
                        
                        processedCount++;
                    }
                    catch (Exception ex)
                    {
                        Logger.Log($"[PREPROCESS_PM4][ERROR] Failed to process {pm4File}: {ex.Message}");
                    }
                }
                
                Logger.Log($"[PREPROCESS_PM4] Completed: {processedCount} PM4 files, {totalMeshes} MSLK object .obj files");
                Logger.Log($"[PREPROCESS_PM4] Output: {pm4OutputDir}");
            }
            catch (Exception ex)
            {
                Logger.Log($"[PREPROCESS_PM4][ERROR] {ex.Message}");
            }
        }

        /// <summary>
        /// Analyze preprocessed mesh files for PM4/WMO correlations
        /// </summary>
        private static void AnalyzePreprocessedCache(string cacheDir, string outputDir, int maxCandidates, bool visualize)
        {
            try
            {
                Logger.Log($"[ANALYZE] Analyzing preprocessed mesh files: {cacheDir}");
                
                var pm4OutputDir = Path.Combine(cacheDir, "pm4_mslk");
                var wmoOutputDir = Path.Combine(cacheDir, "wmo_walkable");
                
                if (!Directory.Exists(pm4OutputDir) || !Directory.Exists(wmoOutputDir))
                {
                    Logger.Log("[ANALYZE][ERROR] Preprocessed directories not found. Run --preprocess-pm4 and --preprocess-wmo first.");
                    Logger.Log($"[ANALYZE][ERROR] Looking for: {pm4OutputDir} and {wmoOutputDir}");
                    return;
                }
                
                // Load PM4 mesh files
                Logger.Log("[ANALYZE] Loading PM4 MSLK object meshes...");
                var pm4Files = Directory.GetFiles(pm4OutputDir, "pm4_*.obj");
                var pm4Candidates = new List<MeshCandidate>();
                
                foreach (var pm4File in pm4Files.Take(maxCandidates))
                {
                    try
                    {
                        var candidate = LoadObjFile(pm4File, "PM4_MSLK");
                        if (candidate != null && candidate.Vertices.Count > 0)
                        {
                            pm4Candidates.Add(candidate);
                            Logger.Log($"[ANALYZE]   PM4: {Path.GetFileName(pm4File)} ({candidate.Vertices.Count} vertices)");
                        }
                    }
                    catch (Exception ex)
                    {
                        Logger.Log($"[ANALYZE][WARN] Failed to load PM4 mesh {pm4File}: {ex.Message}");
                    }
                }
                
                // Load WMO mesh files
                Logger.Log("[ANALYZE] Loading WMO walkable surface meshes...");
                var wmoFiles = Directory.GetFiles(wmoOutputDir, "wmo_*.obj");
                var wmoCandidates = new List<MeshCandidate>();
                
                foreach (var wmoFile in wmoFiles.Take(maxCandidates))
                {
                    try
                    {
                        var candidate = LoadObjFile(wmoFile, "WMO_WALKABLE");
                        if (candidate != null && candidate.Vertices.Count > 0)
                        {
                            wmoCandidates.Add(candidate);
                            Logger.Log($"[ANALYZE]   WMO: {Path.GetFileName(wmoFile)} ({candidate.Vertices.Count} vertices)");
                        }
                    }
                    catch (Exception ex)
                    {
                        Logger.Log($"[ANALYZE][WARN] Failed to load WMO mesh {wmoFile}: {ex.Message}");
                    }
                }
                
                Logger.Log($"[ANALYZE] Performing correlation analysis between {pm4Candidates.Count} PM4 objects and {wmoCandidates.Count} WMO surfaces...");
                
                // Perform geometric matching
                var matchResults = new List<(MeshCandidate pm4, MeshCandidate wmo, double distance)>();
                
                foreach (var pm4 in pm4Candidates)
                {
                    foreach (var wmo in wmoCandidates)
                    {
                        try
                        {
                            double distance = ModifiedHausdorffDistance(pm4.Vertices, wmo.Vertices);
                            matchResults.Add((pm4, wmo, distance));
                        }
                        catch (Exception ex)
                        {
                            Logger.Log($"[ANALYZE][WARN] Failed to compute distance between {pm4.SubObjectName} and {wmo.SubObjectName}: {ex.Message}");
                        }
                    }
                }
                
                // Sort by best matches
                var bestMatches = matchResults.OrderBy(r => r.distance).Take(10).ToList();
                
                Logger.Log($"[ANALYZE] Top matches found:");
                foreach (var match in bestMatches)
                {
                    Logger.Log($"[ANALYZE]   {match.pm4.SubObjectName} ↔ {match.wmo.SubObjectName}: distance = {match.distance:F2}");
                }
                
                // Export visualization if requested
                if (visualize && bestMatches.Any())
                {
                    var vizDir = Path.Combine(outputDir, "matches_visualization");
                    Directory.CreateDirectory(vizDir);
                    
                    for (int i = 0; i < Math.Min(3, bestMatches.Count); i++)
                    {
                        var match = bestMatches[i];
                        var pm4VizPath = Path.Combine(vizDir, $"match_{i:D2}_pm4_{match.pm4.SubObjectName}.obj");
                        var wmoVizPath = Path.Combine(vizDir, $"match_{i:D2}_wmo_{match.wmo.SubObjectName}.obj");
                        
                        match.pm4.WriteObj(pm4VizPath);
                        match.wmo.WriteObj(wmoVizPath);
                        
                        Logger.Log($"[ANALYZE]   Visualization exported: match_{i:D2}_*.obj");
                    }
                }
                
                Logger.Log("[ANALYZE] Analysis complete - results logged above");
            }
            catch (Exception ex)
            {
                Logger.Log($"[ANALYZE][ERROR] {ex.Message}");
            }
        }
        
        /// <summary>
        /// Load a mesh candidate from an .obj file
        /// </summary>
        private static MeshCandidate? LoadObjFile(string objPath, string sourceType)
        {
            try
            {
                var lines = File.ReadAllLines(objPath);
                var vertices = new List<(float X, float Y, float Z)>();
                var indices = new List<int>();
                
                foreach (var line in lines)
                {
                    if (line.StartsWith("v "))
                    {
                        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 4 && 
                            float.TryParse(parts[1], out float x) &&
                            float.TryParse(parts[2], out float y) &&
                            float.TryParse(parts[3], out float z))
                        {
                            vertices.Add((x, y, z));
                        }
                    }
                    else if (line.StartsWith("f "))
                    {
                        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        for (int i = 1; i < parts.Length; i++)
                        {
                            var indexPart = parts[i].Split('/')[0]; // Handle f v1/vt1/vn1 format
                            if (int.TryParse(indexPart, out int vertexIndex))
                            {
                                indices.Add(vertexIndex - 1); // Convert to 0-based indexing
                            }
                        }
                    }
                }
                
                return new MeshCandidate
                {
                    SourceFile = objPath,
                    SourceType = sourceType,
                    SubObjectName = Path.GetFileNameWithoutExtension(objPath),
                    Vertices = vertices,
                    Indices = indices
                };
            }
            catch (Exception)
            {
                return null;
            }
        }
    }

    // Shared mesh data structure for matching
    public class MeshCandidate
    {
        public List<(float X, float Y, float Z)> Vertices { get; set; } = new();
        public List<int> Indices { get; set; } = new();
        public string SourceFile { get; set; } = string.Empty;
        public string SourceType { get; set; } = string.Empty; // "PM4" or "WMO"
        public string? SubObjectName { get; set; } // For WMO group or PM4 mesh island
        public List<(float X, float Y, float Z)>? ExteriorPoints { get; set; } = null;

        // Normalize mesh: center at origin, scale to unit bounding box
        public void NormalizeInPlace()
        {
            if (Vertices.Count == 0) return;
            float minX = Vertices.Min(v => v.X), maxX = Vertices.Max(v => v.X);
            float minY = Vertices.Min(v => v.Y), maxY = Vertices.Max(v => v.Y);
            float minZ = Vertices.Min(v => v.Z), maxZ = Vertices.Max(v => v.Z);
            float cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
            float scale = Math.Max(maxX - minX, Math.Max(maxY - minY, maxZ - minZ));
            if (scale == 0) scale = 1;
            for (int i = 0; i < Vertices.Count; i++)
            {
                var v = Vertices[i];
                Vertices[i] = ((v.X - cx) / scale, (v.Y - cy) / scale, (v.Z - cz) / scale);
            }
        }

        // Compute basic features: vertex/triangle count, bounding box, centroid
        public (int vertexCount, int triangleCount, (float minX, float minY, float minZ, float maxX, float maxY, float maxZ) bbox, (float cx, float cy, float cz) centroid) ComputeFeatures()
        {
            int vCount = Vertices.Count;
            int tCount = Indices.Count / 3;
            if (vCount == 0)
                return (0, 0, (0, 0, 0, 0, 0, 0), (0, 0, 0));
            float minX = Vertices.Min(v => v.X), maxX = Vertices.Max(v => v.X);
            float minY = Vertices.Min(v => v.Y), maxY = Vertices.Max(v => v.Y);
            float minZ = Vertices.Min(v => v.Z), maxZ = Vertices.Max(v => v.Z);
            float cx = Vertices.Average(v => v.X);
            float cy = Vertices.Average(v => v.Y);
            float cz = Vertices.Average(v => v.Z);
            return (vCount, tCount, (minX, minY, minZ, maxX, maxY, maxZ), (cx, cy, cz));
        }

        // Compute RMSD between two normalized meshes (vertex count must match)
        public static double ComputeRmsd(MeshCandidate a, MeshCandidate b)
        {
            if (a.Vertices.Count != b.Vertices.Count || a.Vertices.Count == 0)
                return double.MaxValue;
            double sum = 0;
            for (int i = 0; i < a.Vertices.Count; i++)
            {
                var va = a.Vertices[i];
                var vb = b.Vertices[i];
                double dx = va.X - vb.X, dy = va.Y - vb.Y, dz = va.Z - vb.Z;
                sum += dx * dx + dy * dy + dz * dz;
            }
            return Math.Sqrt(sum / a.Vertices.Count);
        }

        // Export mesh as OBJ file
        public void WriteObj(string path)
        {
            using var sw = new StreamWriter(path, false, Encoding.UTF8);
            // Output mesh vertices, flagging if also exterior
            for (int i = 0; i < Vertices.Count; i++)
            {
                var v = Vertices[i];
                bool isExterior = ExteriorPoints != null && ExteriorPoints.Any(ep => Math.Abs(ep.X - v.X) < 1e-4 && Math.Abs(ep.Y - v.Y) < 1e-4 && Math.Abs(ep.Z - v.Z) < 1e-4);
                sw.Write($"v {v.X} {v.Y} {v.Z}");
                if (isExterior) sw.Write(" # exterior");
                sw.WriteLine();
            }
            // Output faces as before
            for (int i = 0; i < Indices.Count; i += 3)
            {
                if (i + 2 >= Indices.Count) break;
                int a = Indices[i], b = Indices[i + 1], c = Indices[i + 2];
                if (a < 0 || b < 0 || c < 0 || a >= Vertices.Count || b >= Vertices.Count || c >= Vertices.Count)
                {
                    Logger.Log($"[WARN] Skipping face with out-of-bounds index: f {a} {b} {c} (vertex count: {Vertices.Count})");
                    continue;
                }
                sw.WriteLine($"f {a + 1} {b + 1} {c + 1}");
            }
            // Output exterior points as a separate object/group for visualization
            if (ExteriorPoints != null && ExteriorPoints.Count > 0)
            {
                sw.WriteLine("o exterior_points");
                foreach (var ep in ExteriorPoints)
                    sw.WriteLine($"v {ep.X} {ep.Y} {ep.Z} # exterior");
            }
        }
    }

    // Utility for mesh extraction
    public static class MeshExtractor
    {
        // *** ENHANCED: Extract MSLK objects OR combined MSCN/MSPV points ***
        public static List<MeshCandidate> ExtractFromPm4(string pm4Path, bool useMslkObjects = false)
        {
            var candidates = new List<MeshCandidate>();
            Logger.Log($"[EXTRACT_PM4] Loading {Path.GetFileName(pm4Path)}...");
            
            if (useMslkObjects)
            {
                return ExtractMslkObjectsFromPm4(pm4Path);
            }
            
            try
            {
                var pm4 = PM4File.FromFile(pm4Path);
                var combinedPoints = new List<(float X, float Y, float Z)>();

                // Extract MSCN points, preserve original coordinates
                if (pm4.MSCN != null)
                {
                    foreach(var p in pm4.MSCN.ExteriorVertices)
                    {
                         combinedPoints.Add((p.X, p.Y, p.Z)); // Use original coordinates
                    }
                    Logger.Log($"[EXTRACT_PM4_DETAIL] Added {pm4.MSCN.ExteriorVertices.Count} MSCN points for {Path.GetFileName(pm4Path)}");
                }

                // Extract MSPV points, preserve original coordinates
                 if (pm4.MSPV != null)
                {
                    foreach(var p in pm4.MSPV.Vertices)
                    {
                         combinedPoints.Add((p.X, p.Y, p.Z)); // Use original coordinates
                    }
                    Logger.Log($"[EXTRACT_PM4_DETAIL] Added {pm4.MSPV.Vertices.Count} MSPV points for {Path.GetFileName(pm4Path)}");
                }
                
                // Create single candidate if points exist
                if (combinedPoints.Count > 0)
                {
                    var mesh = new MeshCandidate
                    {
                        Vertices = combinedPoints,
                        Indices = new List<int>(), // Point cloud
                        SourceFile = pm4Path,
                        SourceType = "PM4",
                        SubObjectName = "MSCN_MSPV_Combined", // Indicate source
                        ExteriorPoints = null 
                    };
                    candidates.Add(mesh);
                    Logger.Log($"[EXTRACT_PM4_DETAIL] Created combined MSCN/MSPV candidate. Total Verts: {mesh.Vertices.Count}");
                }
                else
                {
                    Logger.Log($"[EXTRACT_PM4_DETAIL] No MSCN or MSPV points found in {Path.GetFileName(pm4Path)}");
                }

                // *** Removed MSVT extraction logic & MDSF/MDOS logic ***
            }
            catch (Exception ex)
            {
                Logger.Log($"[WARN] Failed to extract PM4 mesh from {pm4Path}: {ex.Message}");
                if (ex.Message.Contains("MSLK") || ex.InnerException?.Message?.Contains("MSLK") == true)
                {
                    Logger.Log($"[WARN] MSLK chunk not found - this is handled by WoWToolbox.Core, not Warcraft.NET directly. PM4 may be corrupt or from a different version.");
                }
            }
            Logger.Log($"[EXTRACT_PM4] Finished {Path.GetFileName(pm4Path)}, created {candidates.Count} candidate(s).");
            return candidates;
        }

        /// <summary>
        /// ✨ NEW: Extract individual MSLK scene graph objects as separate mesh candidates
        /// </summary>
        private static List<MeshCandidate> ExtractMslkObjectsFromPm4(string pm4Path)
        {
            var candidates = new List<MeshCandidate>();
            Logger.Log($"[EXTRACT_MSLK] Loading MSLK objects from {Path.GetFileName(pm4Path)}...");
            
            try
            {
                var pm4File = PM4File.FromFile(pm4Path);
                
                if (pm4File.MSLK?.Entries == null || pm4File.MSLK.Entries.Count == 0)
                {
                    Logger.Log($"[WARN] No MSLK data found in {pm4Path}");
                    return candidates;
                }

                // Skip files with insufficient MPRR data (indicates parsing issues)
                if (pm4File.MPRR?.Sequences == null || pm4File.MPRR.Sequences.Count < 100)
                {
                    Logger.Log($"[WARN] Insufficient MPRR data in {pm4Path} ({pm4File.MPRR?.Sequences?.Count ?? 0} sequences). Skipping due to potential parsing issues.");
                    return candidates;
                }

                // Perform MSLK hierarchy analysis
                var hierarchyAnalyzer = new WoWToolbox.Core.Navigation.PM4.MslkHierarchyAnalyzer();
                var hierarchyResult = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);
                // Use root hierarchy but limit to first object only for testing
                var allObjectSegments = hierarchyAnalyzer.SegmentObjectsByHierarchy(hierarchyResult);
                var objectSegments = allObjectSegments.Take(1).ToList(); // Just test with 1 object for proof of concept

                // Export each object as a separate mesh candidate
                var objectMeshExporter = new WoWToolbox.Core.Navigation.PM4.MslkObjectMeshExporter();
                
                foreach (var objectSegment in objectSegments)
                {
                    try
                    {
                        // Extract mesh data using render-mesh-only mode for clean geometry
                        var meshData = ExtractMslkObjectMeshData(objectSegment, pm4File, objectMeshExporter);
                        
                        // Only include objects with meaningful geometry (at least 50 vertices)
                        if (meshData.Vertices.Count >= 50)
                        {
                            var mesh = new MeshCandidate
                            {
                                SourceFile = pm4Path,
                                SourceType = "PM4_MSLK",
                                SubObjectName = $"Object_{objectSegment.RootIndex}",
                                Vertices = meshData.Vertices,
                                Indices = meshData.Indices
                            };
                            
                            candidates.Add(mesh);
                            Logger.Log($"[EXTRACT_MSLK] Object {objectSegment.RootIndex}: {meshData.Vertices.Count} vertices, {meshData.Indices.Count / 3} triangles");
                        }
                        else
                        {
                            Logger.Log($"[SKIP] Object {objectSegment.RootIndex}: Only {meshData.Vertices.Count} vertices (too small)");
                        }
                    }
                    catch (Exception ex)
                    {
                        Logger.Log($"[WARN] Failed to extract MSLK object {objectSegment.RootIndex} from {pm4Path}: {ex.Message}");
                    }
                }
                
                Logger.Log($"[EXTRACT_MSLK] Extracted {candidates.Count} MSLK objects from {Path.GetFileName(pm4Path)}");
            }
            catch (Exception ex)
            {
                Logger.Log($"[WARN] Failed to extract MSLK objects from {pm4Path}: {ex.Message}");
            }
            
            return candidates;
        }

        /// <summary>
        /// Helper to extract mesh data from a single MSLK object segment
        /// </summary>
        private static (List<(float X, float Y, float Z)> Vertices, List<int> Indices) ExtractMslkObjectMeshData(
            WoWToolbox.Core.Navigation.PM4.MslkHierarchyAnalyzer.ObjectSegmentationResult objectSegment,
            WoWToolbox.Core.Navigation.PM4.PM4File pm4File,
            WoWToolbox.Core.Navigation.PM4.MslkObjectMeshExporter objectMeshExporter)
        {
            var vertices = new List<(float X, float Y, float Z)>();
            var indices = new List<int>();
            
            try
            {
                // Create a temporary file to capture the object mesh data
                var tempPath = Path.GetTempFileName();
                try
                {
                    // Export the object with render-mesh-only mode for clean geometry
                    objectMeshExporter.ExportObjectMesh(objectSegment, pm4File, tempPath, renderMeshOnly: true);
                    
                    // Parse the exported OBJ to extract vertices
                    var objContent = File.ReadAllLines(tempPath);
                    foreach (var line in objContent)
                    {
                        if (line.StartsWith("v "))
                        {
                            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                            if (parts.Length >= 4 && 
                                float.TryParse(parts[1], out float x) &&
                                float.TryParse(parts[2], out float y) &&
                                float.TryParse(parts[3], out float z))
                            {
                                vertices.Add((x, y, z));
                            }
                        }
                        else if (line.StartsWith("f "))
                        {
                            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                            if (parts.Length >= 4)
                            {
                                for (int i = 1; i < parts.Length; i++)
                                {
                                    var indexPart = parts[i].Split('/')[0]; // Handle f v1/vt1/vn1 format
                                    if (int.TryParse(indexPart, out int vertexIndex))
                                    {
                                        indices.Add(vertexIndex - 1); // Convert to 0-based indexing
                                    }
                                }
                            }
                        }
                    }
                }
                finally
                {
                    if (File.Exists(tempPath))
                        File.Delete(tempPath);
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[WARN] Failed to extract mesh data for object {objectSegment.RootIndex}: {ex.Message}");
            }
            
            return (vertices, indices);
        }

        // Extract walkable/top-facing surfaces from WMO file (focuses on navigation-relevant geometry)
        public static List<MeshCandidate> ExtractFromWmo(string wmoPath)
        {
            var candidates = new List<MeshCandidate>();
            try
            {
                var merged = WmoMeshExporter.LoadMergedWmoMesh(wmoPath);
                if (merged == null || merged.Vertices.Count == 0 || merged.Triangles.Count == 0)
                    return candidates;

                // Extract walkable surfaces by filtering for top-facing triangles
                var walkableSurfaces = ExtractWalkableSurfaces(merged);
                
                if (walkableSurfaces.Count > 0)
                {
                    var mesh = new MeshCandidate
                    {
                        Vertices = walkableSurfaces,
                        Indices = new List<int>(), // Point cloud for now
                        SourceFile = wmoPath,
                        SourceType = "WMO_WALKABLE",
                        SubObjectName = "TopFacingSurfaces"
                    };
                    candidates.Add(mesh);
                    Logger.Log($"[EXTRACT_WMO] Extracted {walkableSurfaces.Count} walkable surface points from {Path.GetFileName(wmoPath)}");
                }
                else
                {
                    Logger.Log($"[EXTRACT_WMO] No walkable surfaces found in {Path.GetFileName(wmoPath)}");
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"[WARN] Failed to extract WMO walkable surfaces from {wmoPath}: {ex.Message}");
            }
            return candidates;
        }

        /// <summary>
        /// Extract only walkable/top-facing surfaces from WMO mesh
        /// Focuses on horizontal surfaces that correspond to navigation data
        /// </summary>
        private static List<(float X, float Y, float Z)> ExtractWalkableSurfaces(WoWToolbox.Core.WMO.WmoGroupMesh merged)
        {
            var walkablePoints = new List<(float X, float Y, float Z)>();
            
            try
            {
                // Define what constitutes a "walkable" surface
                const float walkableNormalThreshold = 0.7f; // Normal Y component must be > 0.7 (roughly 45° slope)
                
                // Process triangles to find top-facing/walkable surfaces
                for (int i = 0; i < merged.Triangles.Count; i++)
                {
                    var triangle = merged.Triangles[i];
                    
                    // Skip non-renderable triangles
                    if ((triangle.Flags & WoWToolbox.Core.WMO.WmoGroupMesh.FLAG_RENDER) == 0)
                        continue;
                    
                    // Get triangle vertices using correct property names
                    if (triangle.Index0 >= merged.Vertices.Count || 
                        triangle.Index1 >= merged.Vertices.Count || 
                        triangle.Index2 >= merged.Vertices.Count)
                        continue;
                        
                    var v1 = merged.Vertices[triangle.Index0];
                    var v2 = merged.Vertices[triangle.Index1];
                    var v3 = merged.Vertices[triangle.Index2];
                    
                    // Calculate triangle normal using Position property
                    var edge1 = new Vector3(v2.Position.X - v1.Position.X, v2.Position.Y - v1.Position.Y, v2.Position.Z - v1.Position.Z);
                    var edge2 = new Vector3(v3.Position.X - v1.Position.X, v3.Position.Y - v1.Position.Y, v3.Position.Z - v1.Position.Z);
                    var normal = Vector3.Cross(edge1, edge2);
                    
                    if (normal.Length() > 0)
                    {
                        normal = Vector3.Normalize(normal);
                        
                        // Check if surface is walkable (facing upward, not walls/ceilings)
                        // In WoW coordinate system, Y is typically up
                        // Only include surfaces that face upward (positive Y normal) and are roughly horizontal
                        if (normal.Y > walkableNormalThreshold && normal.Y > Math.Max(Math.Abs(normal.X), Math.Abs(normal.Z)))
                        {
                            // Apply coordinate transformation to match PM4 data coordinate system
                            // Using the same transform pattern as ToUnifiedWorld: (X, Y, Z) → (-Y, -Z, X)
                            var transformedV1 = new Vector3(-v1.Position.Y, -v1.Position.Z, v1.Position.X);
                            var transformedV2 = new Vector3(-v2.Position.Y, -v2.Position.Z, v2.Position.X);
                            var transformedV3 = new Vector3(-v3.Position.Y, -v3.Position.Z, v3.Position.X);
                            
                            // Add triangle vertices as walkable surface points
                            walkablePoints.Add((transformedV1.X, transformedV1.Y, transformedV1.Z));
                            walkablePoints.Add((transformedV2.X, transformedV2.Y, transformedV2.Z));
                            walkablePoints.Add((transformedV3.X, transformedV3.Y, transformedV3.Z));
                        }
                    }
                }
                
                // Remove duplicate points (optional optimization)
                var uniquePoints = new HashSet<(float X, float Y, float Z)>();
                foreach (var point in walkablePoints)
                {
                    // Round to avoid floating point precision issues
                    var roundedPoint = (
                        (float)Math.Round(point.X, 3),
                        (float)Math.Round(point.Y, 3), 
                        (float)Math.Round(point.Z, 3)
                    );
                    uniquePoints.Add(roundedPoint);
                }
                
                return uniquePoints.ToList();
            }
            catch (Exception ex)
            {
                Logger.Log($"[WARN] Error extracting walkable surfaces: {ex.Message}");
                return walkablePoints;
            }
        }

        // Recursively extract all meshes from a directory of PM4 or WMO files
        public static List<MeshCandidate> ExtractAllFromDirectory(string dir, bool isPm4, bool useMslkObjects)
        {
            var candidates = new List<MeshCandidate>();
            if (!Directory.Exists(dir))
            {
                Console.WriteLine($"[WARN] Directory not found: {dir}");
                return candidates;
            }
            var ext = isPm4 ? ".pm4" : ".wmo";
            var files = Directory.GetFiles(dir, "*" + ext, SearchOption.AllDirectories);
            foreach (var file in files)
            {
                var meshes = isPm4 ? ExtractFromPm4(file, useMslkObjects) : ExtractFromWmo(file);
                candidates.AddRange(meshes);
            }
            return candidates;
        }
    }

    // Simple logger that writes to both console and a log file
    public static class Logger
    {
        private static StreamWriter? _writer;
        public static void Init(string logPath)
        {
            if (_writer != null) _writer.Dispose();
            // If the log file path itself is a directory, error
            if (Directory.Exists(logPath))
                throw new IOException($"Cannot create log file '{logPath}' because a directory with the same name already exists.");
            // If the log file path itself is a file, that's fine (we will overwrite)
            var dir = Path.GetDirectoryName(logPath);
            if (!string.IsNullOrEmpty(dir))
            {
                // If the parent directory exists and is a file, error
                if (File.Exists(dir))
                    throw new IOException($"Cannot create log directory '{dir}' because a file with the same name already exists.");
                // Only create the directory if it does not exist
                if (!Directory.Exists(dir))
                    Directory.CreateDirectory(dir);
            }
            _writer = new StreamWriter(logPath, false, Encoding.UTF8) { AutoFlush = true };
        }
        public static void Log(string msg)
        {
            Console.WriteLine(msg);
            _writer?.WriteLine(msg);
        }
        public static void Close()
        {
            _writer?.Dispose();
            _writer = null;
        }
    }
} 