using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWToolbox.Common.Analysis;
using WoWToolbox.Core.Models;
using MathNet.Numerics.LinearAlgebra;
using System.Text;

namespace WoWToolbox.ObjWmoMatcher
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            var clustersOption = new Option<string>("--clusters", "Directory containing PM4 OBJ cluster files (input)");
            clustersOption.IsRequired = true;
            var wmosOption = new Option<string>("--wmos", "Directory containing WMO OBJ files (searched recursively)");
            wmosOption.IsRequired = true;
            var outputOption = new Option<string>("--output", "Output file path (YAML/JSON/CSV)");
            outputOption.IsRequired = true;
            var topOption = new Option<int>("--top", () => 1, "Number of top matches to record per cluster (default: 1)");
            var formatOption = new Option<string>("--format", () => "yaml", "Output format: yaml, json, or csv (default: yaml)");

            var rootCommand = new RootCommand("Batch OBJ-to-WMO mesh matcher and placement extractor")
            {
                clustersOption, wmosOption, outputOption, topOption, formatOption
            };

            rootCommand.SetHandler(async (string clusters, string wmos, string output, int top, string format) =>
            {
                try
                {
                    Console.WriteLine($"[INFO] Starting cluster OBJ load from: {clusters}");
                    var clusterMeshes = LoadAllObjMeshes(clusters);
                    Console.WriteLine($"[INFO] Loaded {clusterMeshes.Count} cluster OBJ meshes.");

                    Console.WriteLine($"[INFO] Starting WMO OBJ load from: {wmos}");
                    var wmoMeshes = LoadAllObjMeshes(wmos, recursive: true);
                    Console.WriteLine($"[INFO] Loaded {wmoMeshes.Count} WMO OBJ meshes (recursive).");

                    int clusterIdx = 0;
                    var yaml = new StringBuilder();
                    yaml.AppendLine("placements:");
                    foreach (var cluster in clusterMeshes)
                    {
                        clusterIdx++;
                        string clusterPath = cluster.Key;
                        MeshData clusterMesh = cluster.Value;
                        Console.WriteLine($"[INFO] Processing cluster {clusterIdx}/{clusterMeshes.Count}: {Path.GetFileName(clusterPath)}");
                        double bestScore = double.MaxValue;
                        string bestWmo = "";
                        Matrix<double>? bestRotation = null;
                        Vector<double>? bestTranslation = null;
                        int wmoIdx = 0;
                        foreach (var wmo in wmoMeshes)
                        {
                            wmoIdx++;
                            string wmoPath = wmo.Key;
                            MeshData wmoMesh = wmo.Value;
                            if (wmoIdx <= 5 || wmoIdx == wmoMeshes.Count) // Only print first 5 and last
                                Console.WriteLine($"[INFO]   Comparing to WMO {wmoIdx}/{wmoMeshes.Count}: {Path.GetFileName(wmoPath)}");
                            double rmsd;
                            Matrix<double>? rotation;
                            Vector<double>? translation;
                            if (clusterMesh.Vertices.Count == wmoMesh.Vertices.Count && clusterMesh.Vertices.Count > 0)
                            {
                                rmsd = RigidRegistrationRmsd(clusterMesh, wmoMesh, out rotation, out translation);
                            }
                            else
                            {
                                rmsd = double.MaxValue;
                                rotation = null;
                                translation = null;
                            }
                            if (rmsd < bestScore)
                            {
                                bestScore = rmsd;
                                bestWmo = wmoPath;
                                bestRotation = rotation;
                                bestTranslation = translation;
                            }
                        }
                        Console.WriteLine($"[MATCH] Cluster: {Path.GetFileName(clusterPath)} -> Best WMO: {Path.GetFileName(bestWmo)} (RMSD: {bestScore:F4})");
                        if (bestRotation != null && bestTranslation != null)
                        {
                            Console.WriteLine($"  [Transform] Rotation:\n{bestRotation}");
                            Console.WriteLine($"  [Transform] Translation: {bestTranslation}");
                        }
                        // --- YAML output ---
                        yaml.AppendLine($"  - cluster: {Path.GetFileName(clusterPath)}");
                        yaml.AppendLine($"    matched_wmo: {Path.GetFileName(bestWmo)}");
                        yaml.AppendLine($"    rmsd: {bestScore:F6}");
                        if (bestRotation != null)
                        {
                            yaml.AppendLine("    rotation:");
                            for (int i = 0; i < 3; i++)
                                yaml.AppendLine($"      - [{bestRotation[i,0]:F6}, {bestRotation[i,1]:F6}, {bestRotation[i,2]:F6}]");
                        }
                        if (bestTranslation != null)
                        {
                            yaml.AppendLine($"    translation: [{bestTranslation[0]:F6}, {bestTranslation[1]:F6}, {bestTranslation[2]:F6}]");
                        }
                    }
                    File.WriteAllText(output, yaml.ToString());
                    Console.WriteLine($"[INFO] Placement YAML written to: {output}");
                    Console.WriteLine("[INFO] All clusters processed. Exiting.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ERR] Exception occurred: {ex.Message}\n{ex.StackTrace}");
                }
                await Task.CompletedTask;
            },
            clustersOption, wmosOption, outputOption, topOption, formatOption);

            return await rootCommand.InvokeAsync(args);
        }

        static Dictionary<string, MeshData> LoadAllObjMeshes(string directory, bool recursive = false)
        {
            var meshes = new Dictionary<string, MeshData>();
            if (!Directory.Exists(directory))
            {
                Console.WriteLine($"[ERR] Directory not found: {directory}");
                return meshes;
            }
            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            var objFiles = Directory.GetFiles(directory, "*.obj", searchOption);
            int fileIdx = 0;
            foreach (var file in objFiles)
            {
                fileIdx++;
                try
                {
                    Console.WriteLine($"[INFO]   Loading OBJ {fileIdx}/{objFiles.Length}: {file}");
                    var mesh = MeshAnalysisUtils.LoadObjToMeshData(file);
                    meshes[file] = mesh;
                    Console.WriteLine($"[INFO]   Loaded OBJ: {file} (vertices: {mesh.Vertices.Count}, faces: {mesh.Indices.Count / 3})");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] Failed to load OBJ: {file} ({ex.Message})");
                }
            }
            return meshes;
        }

        // SVD-based rigid registration (Kabsch algorithm) and RMSD calculation
        static double RigidRegistrationRmsd(MeshData a, MeshData b, out Matrix<double>? rotation, out Vector<double>? translation)
        {
            int n = a.Vertices.Count;
            var matA = Matrix<double>.Build.Dense(n, 3);
            var matB = Matrix<double>.Build.Dense(n, 3);
            for (int i = 0; i < n; i++)
            {
                matA[i, 0] = a.Vertices[i].X;
                matA[i, 1] = a.Vertices[i].Y;
                matA[i, 2] = a.Vertices[i].Z;
                matB[i, 0] = b.Vertices[i].X;
                matB[i, 1] = b.Vertices[i].Y;
                matB[i, 2] = b.Vertices[i].Z;
            }
            // Center both sets
            var centroidA = matA.ColumnSums() / n;
            var centroidB = matB.ColumnSums() / n;
            for (int i = 0; i < n; i++)
            {
                matA.SetRow(i, matA.Row(i) - centroidA);
                matB.SetRow(i, matB.Row(i) - centroidB);
            }
            // Covariance matrix
            var h = matA.TransposeThisAndMultiply(matB);
            var svd = h.Svd();
            rotation = svd.VT.TransposeThisAndMultiply(svd.U.Transpose());
            // Ensure right-handed coordinate system
            if (rotation.Determinant() < 0)
            {
                var fix = Matrix<double>.Build.DenseIdentity(3);
                fix[2, 2] = -1;
                rotation = rotation * fix;
            }
            // Compute translation
            translation = centroidB - rotation * centroidA;
            // Apply transform to A and compute RMSD
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                var vA = Vector<double>.Build.DenseOfArray(new double[] { a.Vertices[i].X, a.Vertices[i].Y, a.Vertices[i].Z });
                var vAtrans = rotation * vA + translation;
                var vB = Vector<double>.Build.DenseOfArray(new double[] { b.Vertices[i].X, b.Vertices[i].Y, b.Vertices[i].Z });
                sum += (vAtrans - vB).L2Norm() * (vAtrans - vB).L2Norm();
            }
            return Math.Sqrt(sum / n);
        }
    }
}
