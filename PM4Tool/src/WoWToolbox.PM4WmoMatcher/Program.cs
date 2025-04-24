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

namespace WoWToolbox.PM4WmoMatcher
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            var pm4Option = new Option<string>("--pm4", "Path to a PM4 file or directory of PM4 files (required)") { IsRequired = true };
            var wmoOption = new Option<string>("--wmo", "Path to a directory containing WMO files (required)") { IsRequired = true };
            var outputOption = new Option<string>("--output", () => "./output", "Output directory for results/logs/visualizations (must be a directory, not a file)");
            var maxCandidatesOption = new Option<int>("--max-candidates", () => 5, "Number of top WMO matches to report per PM4 mesh island (default: 5)");
            var minVerticesOption = new Option<int>("--min-vertices", () => 10, "Minimum vertex count for mesh islands to consider (default: 10)");
            var verboseOption = new Option<bool>("--verbose", "Enable verbose logging/diagnostics (optional)");
            var visualizeOption = new Option<bool>("--visualize", "Output aligned mesh files for manual inspection (optional)");
            var exportBaselineOption = new Option<bool>("--export-baseline", "Export baseline OBJ files for the first PM4 and WMO file (full MSVT/MSVI and merged mesh)");
            var exportMprlOption = new Option<bool>("--export-mprl", "Export all MPRL positions from the first PM4 file as a point cloud OBJ (mprl_points.obj) in the output directory.");

            var rootCommand = new RootCommand("PM4–WMO mesh matching tool")
            {
                pm4Option, wmoOption, outputOption, maxCandidatesOption, minVerticesOption, verboseOption, visualizeOption, exportBaselineOption, exportMprlOption
            };

            rootCommand.SetHandler((InvocationContext ctx) =>
            {
                var pm4 = ctx.ParseResult.GetValueForOption(pm4Option)!;
                var wmo = ctx.ParseResult.GetValueForOption(wmoOption)!;
                var output = ctx.ParseResult.GetValueForOption(outputOption)!;
                var maxCandidates = ctx.ParseResult.GetValueForOption(maxCandidatesOption);
                var minVertices = ctx.ParseResult.GetValueForOption(minVerticesOption);
                var verbose = ctx.ParseResult.GetValueForOption(verboseOption);
                var visualize = ctx.ParseResult.GetValueForOption(visualizeOption);
                var exportBaseline = ctx.ParseResult.GetValueForOption(exportBaselineOption);
                var exportMprl = ctx.ParseResult.GetValueForOption(exportMprlOption);

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

                    // Batch extract PM4 meshes
                    Logger.Log("[INFO] Extracting PM4 meshes...");
                    var pm4Meshes = Directory.Exists(pm4) ? MeshExtractor.ExtractAllFromDirectory(pm4, true) : MeshExtractor.ExtractFromPm4(pm4);
                    Logger.Log($"[INFO] Extracted {pm4Meshes.Count} PM4 mesh candidates.");

                    // Batch extract WMO meshes
                    Logger.Log("[INFO] Extracting WMO meshes...");
                    var wmoMeshes = Directory.Exists(wmo) ? MeshExtractor.ExtractAllFromDirectory(wmo, false) : MeshExtractor.ExtractFromWmo(wmo);
                    Logger.Log($"[INFO] Extracted {wmoMeshes.Count} WMO mesh candidates.");

                    // Normalize and compute features
                    foreach (var mesh in pm4Meshes.Concat(wmoMeshes))
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

                    // --- Point cloud matching logic with relaxed filters ---
                    Logger.Log("[INFO] Matching PM4 point clouds to WMO point clouds using relaxed filters...");
                    int totalWithMatches = 0, totalSkipped = 0;
                    foreach (var pm4Mesh in pm4Meshes)
                    {
                        var pm4Points = pm4Mesh.Vertices;
                        if (pm4Points.Count < minVertices) continue;
                        // Compute PM4 features
                        float pm4MinX = pm4Points.Min(p => p.X), pm4MaxX = pm4Points.Max(p => p.X);
                        float pm4MinY = pm4Points.Min(p => p.Y), pm4MaxY = pm4Points.Max(p => p.Y);
                        float pm4MinZ = pm4Points.Min(p => p.Z), pm4MaxZ = pm4Points.Max(p => p.Z);
                        float pm4W = pm4MaxX - pm4MinX, pm4H = pm4MaxY - pm4MinY, pm4D = pm4MaxZ - pm4MinZ;
                        var pm4Centroid = (X: pm4Points.Average(p => p.X), Y: pm4Points.Average(p => p.Y), Z: pm4Points.Average(p => p.Z));
                        //var pm4PCA = ComputePrincipalAxis(pm4Points); // Old PCA
                        var scored = new List<(MeshCandidate wmoMesh, double alignedAvgNNDist, double centroidDist, double pcaAngle, double bboxDiff)>();
                        int wmoCounter = 0;

                        // Compute PM4 orientation for alignment
                        Matrix4x4 pm4Orientation = ComputeOrientationMatrix(pm4Points);
                        Matrix4x4.Invert(pm4Orientation, out Matrix4x4 pm4AlignRotation); // Rotation to align PM4 to world axes

                        foreach (var wmoMesh in wmoMeshes)
                        {
                            wmoCounter++;
                            if(wmoCounter % 100 == 0) // Log progress every 100 WMOs
                            {
                                Logger.Log($"[PROGRESS] Comparing PM4: {Path.GetFileName(pm4Mesh.SourceFile)} ({pm4Mesh.SubObjectName ?? "full"}) with WMO #{wmoCounter}: {Path.GetFileName(wmoMesh.SourceFile)}");
                            }

                            var wmoPoints = wmoMesh.Vertices;
                            if (wmoPoints.Count == 0) continue;

                            // --- Original Feature Comparison (kept for logging/context) ---
                            float wmoMinX = wmoPoints.Min(q => q.X), wmoMaxX = wmoPoints.Max(q => q.X);
                            float wmoMinY = wmoPoints.Min(q => q.Y), wmoMaxY = wmoPoints.Max(q => q.Y);
                            float wmoMinZ = wmoPoints.Min(q => q.Z), wmoMaxZ = wmoPoints.Max(q => q.Z);
                            float wmoW = wmoMaxX - wmoMinX, wmoH = wmoMaxY - wmoMinY, wmoD = wmoMaxZ - wmoMinZ;
                            double bboxDiff = Math.Abs(pm4W - wmoW) / Math.Max(1e-6f, Math.Max(pm4W, wmoW)) + Math.Abs(pm4H - wmoH) / Math.Max(1e-6f, Math.Max(pm4H, wmoH)) + Math.Abs(pm4D - wmoD) / Math.Max(1e-6f, Math.Max(pm4D, wmoD));
                            var wmoCentroid = (X: wmoPoints.Average(q => q.X), Y: wmoPoints.Average(q => q.Y), Z: wmoPoints.Average(q => q.Z));
                            double centroidDist = Math.Sqrt(Math.Pow(pm4Centroid.X - wmoCentroid.X, 2) + Math.Pow(pm4Centroid.Y - wmoCentroid.Y, 2) + Math.Pow(pm4Centroid.Z - wmoCentroid.Z, 2));
                            var pm4PCA_vec = MatrixToMajorAxis(pm4Orientation); // Get major axis for angle calculation
                            var wmoOrientationForAngle = ComputeOrientationMatrix(wmoPoints);
                            var wmoPCA_vec = MatrixToMajorAxis(wmoOrientationForAngle);
                            double pcaAngle = Math.Acos(Math.Clamp(Vector3.Dot(pm4PCA_vec, wmoPCA_vec), -1.0f, 1.0f)) * (180.0 / Math.PI);

                            // --- PCA Alignment and Comparison ---
                            // 1. Compute WMO orientation and alignment rotation
                            Matrix4x4 wmoOrientation = ComputeOrientationMatrix(wmoPoints);
                            Matrix4x4.Invert(wmoOrientation, out Matrix4x4 wmoAlignRotation);

                            // 2. Apply alignment rotations
                            var alignedPm4Points = ApplyRotation(pm4Points, pm4AlignRotation);
                            var alignedWmoPoints = ApplyRotation(wmoPoints, wmoAlignRotation);

                            // 3. Normalize aligned clouds
                            var normalizedAlignedPm4 = NormalizePoints(alignedPm4Points);
                            var normalizedAlignedWmo = NormalizePoints(alignedWmoPoints);

                            // 4. Compute NN distance on aligned, normalized clouds
                            double avgNN1_aligned = AverageNearestNeighbor(normalizedAlignedPm4, normalizedAlignedWmo);
                            double avgNN2_aligned = AverageNearestNeighbor(normalizedAlignedWmo, normalizedAlignedPm4);
                            double maxAvgNNDist_aligned = Math.Max(avgNN1_aligned, avgNN2_aligned);

                            // Add score using aligned distance, but keep original features for logging
                            scored.Add((wmoMesh, maxAvgNNDist_aligned, centroidDist, pcaAngle, bboxDiff));

                            if(verbose)
                            {
                                Logger.Log($"  [CANDIDATE] WMO: {wmoMesh.SourceFile} | AlignedNN: {maxAvgNNDist_aligned:F6} | CentroidDist: {centroidDist:F4} | PCAangle: {pcaAngle:F2} | BBoxDiff: {bboxDiff:F4}");
                            }
                        }
                        // Sort by ALIGNED maxAvgNNDist (lowest is best)
                        var top = scored.OrderBy(x => x.alignedAvgNNDist).Take(maxCandidates).ToList();
                        if (top.Count == 0)
                        {
                            totalSkipped++;
                            continue;
                        }
                        totalWithMatches++;
                        Logger.Log($"[MATCH] PM4: {pm4Mesh.SourceFile} ({pm4Mesh.SubObjectName ?? "full"}) | Verts: {pm4Points.Count}");
                        for (int i = 0; i < top.Count; i++)
                        {
                            // Log aligned distance score
                            var (wmoMesh, alignedAvgNNDist, centroidDist, pcaAngle, bboxDiff) = top[i];
                            Logger.Log($"  #{i + 1}: WMO: {wmoMesh.SourceFile} | AlignedNN: {alignedAvgNNDist:F6} | CentroidDist: {centroidDist:F4} | PCAangle: {pcaAngle:F2} | BBoxDiff: {bboxDiff:F4}");
                            if (visualize)
                            {
                                string safe(string s) => string.Join("_", s.Split(Path.GetInvalidFileNameChars()));
                                string pm4Name = safe(Path.GetFileNameWithoutExtension(pm4Mesh.SourceFile) + (pm4Mesh.SubObjectName != null ? "_" + pm4Mesh.SubObjectName : ""));
                                string wmoName = safe(Path.GetFileNameWithoutExtension(wmoMesh.SourceFile) + (wmoMesh.SubObjectName != null ? "_" + wmoMesh.SubObjectName : ""));
                                string outDir = Path.Combine(output, "obj_matches", $"{pm4Name}__{wmoName}");
                                if (AnyParentIsFile(outDir))
                                {
                                    Logger.Log($"[ERROR] Cannot create OBJ export directory '{outDir}' or one of its parents because a file with the same name already exists. Skipping OBJ export for this pair.");
                                }
                                else if (File.Exists(outDir))
                                {
                                    Logger.Log($"[ERROR] Cannot create OBJ export directory '{outDir}' because a file with the same name already exists. Skipping OBJ export for this pair.");
                                }
                                else
                                {
                                    Directory.CreateDirectory(outDir);
                                    string pm4ObjPath = Path.Combine(outDir, "pm4.obj");
                                    string wmoObjPath = Path.Combine(outDir, "wmo.obj");
                                    var pm4MeshData = new MeshData(
                                        pm4Mesh.Vertices.Select(v => new System.Numerics.Vector3(v.X, v.Y, v.Z)).ToList(),
                                        new List<int>() // No faces
                                    );
                                    PM4MeshExporter.SaveMeshDataToObj(pm4MeshData, pm4ObjPath);
                                    var mergedWmo = WmoMeshExporter.LoadMergedWmoMesh(wmoMesh.SourceFile);
                                    if (mergedWmo != null)
                                        WmoMeshExporter.SaveMergedWmoToObj(mergedWmo, wmoObjPath);
                                    Logger.Log($"    [OBJ] Exported to {outDir}");
                                }
                            }
                        }
                    }
                    Logger.Log($"[SUMMARY] Objects with matches: {totalWithMatches}, objects skipped: {totalSkipped}");
                    Logger.Log("[DONE] Matching complete.");

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
                                var indices = pm4Data.MSVI?.Indices;
                                if (verts != null && indices != null && verts.Count > 0 && indices.Count > 0)
                                {
                                    Logger.Log($"[BASELINE] PM4 raw mesh: {verts.Count} verts, {indices.Count / 3} tris");
                                    for (int vi = 0; vi < Math.Min(3, verts.Count); vi++)
                                        Logger.Log($"[BASELINE] PM4 v[{vi}]: {verts[vi].X}, {verts[vi].Y}, {verts[vi].Z}");
                                    for (int ti = 0; ti < Math.Min(3, indices.Count / 3); ti++)
                                    {
                                        int a = (int)indices[ti * 3], b = (int)indices[ti * 3 + 1], c = (int)indices[ti * 3 + 2];
                                        Logger.Log($"[BASELINE] PM4 f[{ti}]: {a}, {b}, {c}");
                                    }
                                    using var sw = new StreamWriter(pm4ObjPath, false, Encoding.UTF8);
                                    foreach (var v in verts)
                                        sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
                                    for (int i = 0; i + 2 < indices.Count; i += 3)
                                    {
                                        int a = (int)indices[i], b = (int)indices[i + 1], c = (int)indices[i + 2];
                                        if (a < 0 || b < 0 || c < 0 || a >= verts.Count || b >= verts.Count || c >= verts.Count)
                                        {
                                            Logger.Log($"[BASELINE][WARN] Skipping face with out-of-bounds index: f {a} {b} {c} (vertex count: {verts.Count})");
                                            continue;
                                        }
                                        sw.WriteLine($"f {a + 1} {b + 1} {c + 1}");
                                    }
                                    Logger.Log($"[BASELINE] Exported PM4 OBJ (raw mesh): {pm4ObjPath} (Verts: {verts.Count}, Tris: {indices.Count / 3})");
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
                        // Export first WMO
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
                    if (Directory.Exists(pm4))
                    {
                        var pm4Files = Directory.GetFiles(pm4, "*.pm4", SearchOption.AllDirectories);
                        foreach (var pm4File in pm4Files)
                            ExportUnifiedPm4PointCloud(pm4File, output);
                    }
                    else if (File.Exists(pm4) && pm4.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase))
                    {
                        ExportUnifiedPm4PointCloud(pm4, output);
                    }
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

        // --- Helper functions for PCA and NN ---

        // Computes Covariance Matrix (simple implementation)
        static double[,] ComputeCovarianceMatrix(List<(float X, float Y, float Z)> points)
        {
            if (points.Count == 0) return new double[3, 3];

            double cx = points.Average(p => p.X);
            double cy = points.Average(p => p.Y);
            double cz = points.Average(p => p.Z);

            double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
            foreach (var p in points)
            {
                double dx = p.X - cx, dy = p.Y - cy, dz = p.Z - cz;
                xx += dx * dx; xy += dx * dy; xz += dx * dz;
                yy += dy * dy; yz += dy * dz; zz += dz * dz;
            }

            double n = points.Count;
            return new double[3, 3] {
                { xx / n, xy / n, xz / n },
                { xy / n, yy / n, yz / n },
                { xz / n, yz / n, zz / n }
            };
        }

        // Placeholder for computing orientation matrix (PCA axes) - requires SVD or Eigen library
        // This simplified version returns Identity, meaning no alignment will actually happen
        // TODO: Replace with actual PCA calculation (e.g., using Accord.Math or MathNet.Numerics if available, or implement Jacobi eigenvalue)
        static Matrix4x4 ComputeOrientationMatrix(List<(float X, float Y, float Z)> points)
        {
             if (points.Count < 3) return Matrix4x4.Identity;
            // Actual implementation would compute covariance matrix and find its eigenvectors.
            // The eigenvectors form the columns (or rows) of the orientation matrix.
            // Example placeholder:
            // double[,] cov = ComputeCovarianceMatrix(points);
            // (Vector3 axis1, Vector3 axis2, Vector3 axis3) = ComputeEigenvectors(cov); // Needs implementation
            // return new Matrix4x4(
            //     axis1.X, axis1.Y, axis1.Z, 0,
            //     axis2.X, axis2.Y, axis2.Z, 0,
            //     axis3.X, axis3.Y, axis3.Z, 0,
            //     0, 0, 0, 1
            // );
            return Matrix4x4.Identity; // <<<<< Placeholder: No actual orientation computed
        }

        // Helper to get the major axis (e.g., first column) from an orientation matrix
        static Vector3 MatrixToMajorAxis(Matrix4x4 matrix)
        {
             // Assuming columns are eigenvectors/axes
             return Vector3.Normalize(new Vector3(matrix.M11, matrix.M21, matrix.M31));
        }


        // Apply rotation matrix to a list of points
        static List<(float X, float Y, float Z)> ApplyRotation(List<(float X, float Y, float Z)> points, Matrix4x4 rotation)
        {
            var rotated = new List<(float X, float Y, float Z)>(points.Count);
            foreach (var p in points)
            {
                var vec = new Vector3(p.X, p.Y, p.Z);
                var rotatedVec = Vector3.Transform(vec, rotation);
                rotated.Add((rotatedVec.X, rotatedVec.Y, rotatedVec.Z));
            }
            return rotated;
        }

        // Normalize a list of points (center at origin, scale to unit box)
        static List<(float X, float Y, float Z)> NormalizePoints(List<(float X, float Y, float Z)> points)
        { 
            if (points.Count == 0) return new List<(float X, float Y, float Z)>();

            float minX = points.Min(v => v.X), maxX = points.Max(v => v.X);
            float minY = points.Min(v => v.Y), maxY = points.Max(v => v.Y);
            float minZ = points.Min(v => v.Z), maxZ = points.Max(v => v.Z);
            float cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
            float scale = Math.Max(maxX - minX, Math.Max(maxY - minY, maxZ - minZ));
            if (scale < 1e-6f) scale = 1; // Avoid division by zero or near-zero

            var normalized = new List<(float X, float Y, float Z)>(points.Count);
            for (int i = 0; i < points.Count; i++)
            {
                var v = points[i];
                normalized.Add(((v.X - cx) / scale, (v.Y - cy) / scale, (v.Z - cz) / scale));
            }
            return normalized;
        }

        // Original ComputePrincipalAxis (only returns major axis as double[]) - kept for reference/potential use elsewhere
        static double[] ComputePrincipalAxis(List<(float X, float Y, float Z)> points)
        {
            // Compute covariance matrix
            double cx = points.Average(p => p.X);
            double cy = points.Average(p => p.Y);
            double cz = points.Average(p => p.Z);
            double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
            foreach (var p in points)
            {
                double dx = p.X - cx, dy = p.Y - cy, dz = p.Z - cz;
                xx += dx * dx; xy += dx * dy; xz += dx * dz;
                yy += dy * dy; yz += dy * dz; zz += dz * dz;
            }
            var cov = new double[3, 3] {
                { xx, xy, xz },
                { xy, yy, yz },
                { xz, yz, zz }
            };
            // Power iteration for dominant eigenvector
            double[] v = { 1, 0, 0 };
            for (int iter = 0; iter < 10; iter++)
            {
                double[] v2 = new double[3];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        v2[i] += cov[i, j] * v[j];
                double norm = Math.Sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
                for (int i = 0; i < 3; i++) v[i] = v2[i] / norm;
            }
            return v;
        }
        static double Dot(double[] a, double[] b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        static double AverageNearestNeighbor(List<(float X, float Y, float Z)> src, List<(float X, float Y, float Z)> dst)
        {
            double sum = 0;
            foreach (var p in src)
            {
                double minDist = double.MaxValue;
                foreach (var q in dst)
                {
                    double dx = p.X - q.X, dy = p.Y - q.Y, dz = p.Z - q.Z;
                    double dist = dx * dx + dy * dy + dz * dz;
                    if (dist < minDist) minDist = dist;
                }
                sum += Math.Sqrt(minDist);
            }
            return sum / src.Count;
        }

        // Add a helper method to extract and merge all PM4 point cloud data
        private static void ExportUnifiedPm4PointCloud(string pm4Path, string outputDir)
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

            (float X, float Y, float Z) ToTuple(System.Numerics.Vector3 v) => (v.X, v.Y, v.Z);

            // MSVT (mesh vertices)
            if (pm4.MSVT != null)
            {
                foreach (var v in pm4.MSVT.Vertices)
                {
                    var w = v.ToWorldCoordinates();
                    AddPoint((w.X, w.Y, -w.Z), "mesh"); // Negate Z
                }
            }
            // MSCN (exterior)
            if (pm4.MSCN != null)
            {
                foreach (var v in pm4.MSCN.ExteriorVertices)
                {
                    AddPoint((v.X, v.Y, -v.Z), "exterior"); // Negate Z, Assume already in world coords
                }
            }
            // MSPV (path vertices)
            if (pm4.MSPV != null)
            {
                foreach (var v in pm4.MSPV.Vertices)
                {
                     AddPoint((v.X, v.Y, -v.Z), "path"); // Negate Z, Assume already in world coords
                }
            }
            // MPRL (reference points)
            if (pm4.MPRL != null)
            {
                foreach (var entry in pm4.MPRL.Entries)
                {
                    var v = entry.Position;
                    AddPoint((v.X, v.Y, -v.Z), "reference"); // Negate Z, Assume already in world coords
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
        // Extract per-object meshes from a PM4 file using MDSF and MDOS
        public static List<MeshCandidate> ExtractFromPm4(string pm4Path)
        {
            var candidates = new List<MeshCandidate>();
            try
            {
                var pm4 = PM4File.FromFile(pm4Path);
                if (pm4.MSVT == null || pm4.MSVI == null || pm4.MSUR == null)
                    return candidates;
                var verts = pm4.MSVT.Vertices;
                var indices = pm4.MSVI.Indices;
                var surfaces = pm4.MSUR.Entries;
                var mdsf = pm4.MDSF;
                var mdos = pm4.MDOS;

                // If MDSF and MDOS are present, group surfaces by object
                if (mdsf != null && mdos != null && mdsf.Entries.Count > 0 && mdos.Entries.Count > 0)
                {
                    // Map: mdos_index -> list of msur_index
                    var objectToSurfaces = new Dictionary<uint, List<uint>>();
                    foreach (var entry in mdsf.Entries)
                    {
                        if (!objectToSurfaces.ContainsKey(entry.mdos_index))
                            objectToSurfaces[entry.mdos_index] = new List<uint>();
                        objectToSurfaces[entry.mdos_index].Add(entry.msur_index);
                    }
                    foreach (var kvp in objectToSurfaces)
                    {
                        uint mdosIndex = kvp.Key;
                        var msurIndices = kvp.Value;
                        var objVerts = new List<(float X, float Y, float Z)>();
                        var objIndices = new List<int>();
                        var vertMap = new Dictionary<int, int>(); // global->local
                        int nextLocalIdx = 0;
                        foreach (var msurIdx in msurIndices)
                        {
                            if (msurIdx >= surfaces.Count) continue;
                            var surf = surfaces[(int)msurIdx];
                            int start = (int)surf.MsviFirstIndex;
                            int count = surf.IndexCount;
                            // Gather indices for this surface
                            for (int i = 0; i < count; i += 3)
                            {
                                if (start + i + 2 >= indices.Count) break;
                                int[] tri = { (int)indices[start + i], (int)indices[start + i + 1], (int)indices[start + i + 2] };
                                for (int j = 0; j < 3; j++)
                                {
                                    if (!vertMap.ContainsKey(tri[j]))
                                    {
                                        vertMap[tri[j]] = nextLocalIdx++;
                                        var v = verts[tri[j]];
                                        objVerts.Add((v.X, v.Y, -v.Z)); // Negate Z
                                    }
                                }
                                objIndices.Add(vertMap[tri[0]]);
                                objIndices.Add(vertMap[tri[1]]);
                                objIndices.Add(vertMap[tri[2]]);
                            }
                        }
                        if (objVerts.Count > 0 && objIndices.Count > 0)
                        {
                            var exteriorPoints = pm4.MSCN?.ExteriorVertices?.Select(v => (v.X, v.Y, v.Z)).ToList();
                            candidates.Add(new MeshCandidate
                            {
                                Vertices = objVerts,
                                Indices = objIndices,
                                SourceFile = pm4Path,
                                SourceType = "PM4",
                                SubObjectName = $"Object_{mdosIndex}",
                                ExteriorPoints = exteriorPoints
                            });
                        }
                    }
                }
                else
                {
                    // Fallback: extract all geometry as one mesh
                    var exteriorPoints = pm4.MSCN?.ExteriorVertices?.Select(v => (v.X, v.Y, v.Z)).ToList();
                    var mesh = new MeshCandidate
                    {
                        Vertices = verts.Select(v => (v.X, v.Y, -v.Z)).ToList(), // Negate Z
                        Indices = indices.Select(i => (int)i).ToList(),
                        SourceFile = pm4Path,
                        SourceType = "PM4",
                        SubObjectName = null,
                        ExteriorPoints = exteriorPoints
                    };
                    candidates.Add(mesh);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to extract PM4 mesh from {pm4Path}: {ex.Message}");
            }
            return candidates;
        }

        // Extract all mesh data from a WMO file (merged groups as one mesh)
        public static List<MeshCandidate> ExtractFromWmo(string wmoPath)
        {
            var candidates = new List<MeshCandidate>();
            try
            {
                var merged = WmoMeshExporter.LoadMergedWmoMesh(wmoPath);
                if (merged == null || merged.Vertices.Count == 0 || merged.Triangles.Count == 0)
                    return candidates;
                // Use only triangles with FLAG_RENDER for the point cloud
                var pointCloud = merged.ExtractPointCloud(WoWToolbox.Core.WMO.WmoGroupMesh.FLAG_RENDER, null, false);
                var mesh = new MeshCandidate
                {
                    Vertices = pointCloud.Select(v => (v.X, v.Y, v.Z)).ToList(),
                    Indices = new List<int>(), // No faces for point cloud
                    SourceFile = wmoPath,
                    SourceType = "WMO",
                    SubObjectName = null // TODO: group name if needed
                };
                candidates.Add(mesh);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to extract WMO mesh from {wmoPath}: {ex.Message}");
            }
            return candidates;
        }

        // Recursively extract all meshes from a directory of PM4 or WMO files
        public static List<MeshCandidate> ExtractAllFromDirectory(string dir, bool isPm4)
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
                var meshes = isPm4 ? ExtractFromPm4(file) : ExtractFromWmo(file);
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