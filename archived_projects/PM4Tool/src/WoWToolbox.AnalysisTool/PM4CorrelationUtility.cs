using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.AnalysisTool
{
    public static class PM4CorrelationUtility
    {
        private const float Epsilon = 0.001f;
        private const float CoordinateOffset = 17066.666f;

        public static void AnalyzeDirectory(string directory)
        {
            var pm4Files = Directory.EnumerateFiles(directory, "*.pm4", SearchOption.AllDirectories).ToList();
            int totalFiles = pm4Files.Count;
            int withMSVT = 0, withMSCN = 0, withMSLK = 0, withMPRL = 0, withMPRR = 0;
            Console.WriteLine($"Found {totalFiles} PM4 files in {directory}\n");
            foreach (var pm4Path in pm4Files)
            {
                var pm4 = PM4File.FromFile(pm4Path);
                Console.WriteLine($"Analyzing: {pm4Path}");
                bool hasMSVT = pm4.MSVT != null && pm4.MSVT.Vertices.Count > 0;
                bool hasMSCN = pm4.MSCN != null && pm4.MSCN.ExteriorVertices.Count > 0;
                bool hasMSLK = pm4.MSLK != null && pm4.MSLK.Entries.Count > 0;
                bool hasMPRL = pm4.MPRL != null && pm4.MPRL.Entries.Count > 0;
                bool hasMPRR = pm4.MPRR != null && pm4.MPRR.Sequences.Count > 0;
                if (hasMSVT) withMSVT++;
                if (hasMSCN) withMSCN++;
                if (hasMSLK) withMSLK++;
                if (hasMPRL) withMPRL++;
                if (hasMPRR) withMPRR++;
                if (!hasMSVT) Console.WriteLine("  [!] No MSVT chunk");
                if (!hasMSCN) Console.WriteLine("  [!] No MSCN chunk");
                if (!hasMSLK) Console.WriteLine("  [!] No MSLK chunk");
                if (!hasMPRL) Console.WriteLine("  [!] No MPRL chunk");
                if (!hasMPRR) Console.WriteLine("  [!] No MPRR chunk");
                // 1. MSCN <-> MSVT
                if (hasMSCN && hasMSVT)
                {
                    var mscnPoints = pm4.MSCN.ExteriorVertices.ToList();
                    int matchCount = 0;
                    foreach (var mscn in mscnPoints)
                    {
                        if (pm4.MSVT.Vertices.Any(v => Vector3.Distance(mscn, new Vector3(v.X, v.Y, v.Z)) < Epsilon))
                            matchCount++;
                    }
                    Console.WriteLine($"  MSCN points matching MSVT vertices: {matchCount} / {mscnPoints.Count}");
                }
                // 2. MSLK -> MSVI -> MSVT <-> MSCN
                if (hasMSLK && pm4.MSVI != null && hasMSVT && hasMSCN)
                {
                    var mscnPoints = pm4.MSCN.ExteriorVertices.ToList();
                    var mslkNodeToMscn = 0;
                    var referencedMscnIndices = new HashSet<int>();
                    for (int i = 0; i < pm4.MSLK.Entries.Count; i++)
                    {
                        var entry = pm4.MSLK.Entries[i];
                        if (entry.Unknown_0x10 < pm4.MSVI.Indices.Count)
                        {
                            var msvtIdx = pm4.MSVI.Indices[entry.Unknown_0x10];
                            if (msvtIdx < pm4.MSVT.Vertices.Count)
                            {
                                var v = new Vector3(pm4.MSVT.Vertices[(int)msvtIdx].X, pm4.MSVT.Vertices[(int)msvtIdx].Y, pm4.MSVT.Vertices[(int)msvtIdx].Z);
                                // Find the closest MSCN point
                                int closestIdx = -1;
                                float minDist = float.MaxValue;
                                for (int j = 0; j < mscnPoints.Count; j++)
                                {
                                    float dist = Vector3.Distance(v, mscnPoints[j]);
                                    if (dist < minDist)
                                    {
                                        minDist = dist;
                                        closestIdx = j;
                                    }
                                }
                                if (minDist < Epsilon)
                                {
                                    mslkNodeToMscn++;
                                    referencedMscnIndices.Add(closestIdx);
                                }
                            }
                        }
                    }
                    int neverReferenced = mscnPoints.Count - referencedMscnIndices.Count;
                    Console.WriteLine($"  MSLK node entries matching MSCN points: {mslkNodeToMscn} / {pm4.MSLK.Entries.Count}");
                    Console.WriteLine($"  Unique MSCN points referenced by MSLK nodes: {referencedMscnIndices.Count} / {mscnPoints.Count}");
                    Console.WriteLine($"  MSCN points never referenced by any MSLK node: {neverReferenced}");
                }
                // 3. MPRL positions <-> MSVT and MSCN
                if (hasMPRL && hasMSVT && hasMSCN)
                {
                    var mscnPoints = pm4.MSCN.ExteriorVertices.ToList();
                    int mprlToMsvt = 0, mprlToMscn = 0;
                    foreach (var entry in pm4.MPRL.Entries)
                    {
                        var fixedMprl = new Vector3(CoordinateOffset - entry.Position.X, CoordinateOffset - entry.Position.Y, entry.Position.Z);
                        if (pm4.MSVT.Vertices.Any(v => Vector3.Distance(fixedMprl, new Vector3(v.X, v.Y, v.Z)) < Epsilon))
                            mprlToMsvt++;
                        if (mscnPoints.Any(mscn => Vector3.Distance(fixedMprl, mscn) < Epsilon))
                            mprlToMscn++;
                    }
                    Console.WriteLine($"  MPRL positions matching MSVT: {mprlToMsvt} / {pm4.MPRL.Entries.Count}");
                    Console.WriteLine($"  MPRL positions matching MSCN: {mprlToMscn} / {pm4.MPRL.Entries.Count}");
                }
                // --- Debug: Print centroid, bounding box, and sample points for MSVT and MSCN ---
                if (hasMSVT && hasMSCN)
                {
                    var msvtPoints = pm4.MSVT.Vertices.Select(v => new Vector3(v.X, v.Y, v.Z)).ToList();
                    var mscnPoints = pm4.MSCN.ExteriorVertices.ToList();
                    // Centroids
                    Vector3 msvtCentroid = msvtPoints.Aggregate(Vector3.Zero, (a, b) => a + b) / msvtPoints.Count;
                    Vector3 mscnCentroid = mscnPoints.Aggregate(Vector3.Zero, (a, b) => a + b) / mscnPoints.Count;
                    // Bounding boxes
                    Vector3 msvtMin = new Vector3(msvtPoints.Min(p => p.X), msvtPoints.Min(p => p.Y), msvtPoints.Min(p => p.Z));
                    Vector3 msvtMax = new Vector3(msvtPoints.Max(p => p.X), msvtPoints.Max(p => p.Y), msvtPoints.Max(p => p.Z));
                    Vector3 mscnMin = new Vector3(mscnPoints.Min(p => p.X), mscnPoints.Min(p => p.Y), mscnPoints.Min(p => p.Z));
                    Vector3 mscnMax = new Vector3(mscnPoints.Max(p => p.X), mscnPoints.Max(p => p.Y), mscnPoints.Max(p => p.Z));
                    Console.WriteLine($"  MSVT centroid: {msvtCentroid}");
                    Console.WriteLine($"  MSCN centroid: {mscnCentroid}");
                    Console.WriteLine($"  MSVT bounding box: min {msvtMin}, max {msvtMax}");
                    Console.WriteLine($"  MSCN bounding box: min {mscnMin}, max {mscnMax}");
                    // First 5 points
                    Console.WriteLine("  First 5 MSVT points:");
                    foreach (var p in msvtPoints.Take(5)) Console.WriteLine($"    {p}");
                    Console.WriteLine("  First 5 MSCN points:");
                    foreach (var p in mscnPoints.Take(5)) Console.WriteLine($"    {p}");
                    // Average offset from each MSCN to nearest MSVT
                    double avgOffset = mscnPoints.Average(mscn => msvtPoints.Min(msvt => Vector3.Distance(mscn, msvt)));
                    Console.WriteLine($"  Average offset from MSCN to nearest MSVT: {avgOffset:F6}");
                }
                Console.WriteLine();
            }
            Console.WriteLine($"\nSummary:");
            Console.WriteLine($"  Files with MSVT: {withMSVT} / {totalFiles}");
            Console.WriteLine($"  Files with MSCN: {withMSCN} / {totalFiles}");
            Console.WriteLine($"  Files with MSLK: {withMSLK} / {totalFiles}");
            Console.WriteLine($"  Files with MPRL: {withMPRL} / {totalFiles}");
            Console.WriteLine($"  Files with MPRR: {withMPRR} / {totalFiles}");
        }
    }
} 