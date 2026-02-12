using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Linq;

namespace WoWRollback.PM4Module.Analysis;

/// <summary>
/// Dumps PM4 geometry (MSVT mesh and MSCN point cloud) to Wavefront OBJ format.
/// </summary>
public class Pm4GeometryDumper
{
    public void Dump(string pm4Path, string outDir)
    {
        if (!File.Exists(pm4Path)) return;

        var pm4 = PM4File.FromFile(pm4Path);
        string baseName = Path.GetFileNameWithoutExtension(pm4Path);
        Directory.CreateDirectory(outDir);

        // 1. Dump MSVT (Render Geometry)
        if (pm4.MeshVertices.Count > 0)
        {
            using var swWmo = new StreamWriter(Path.Combine(outDir, $"{baseName}_msvt_wmo.obj"));
            using var swM2 = new StreamWriter(Path.Combine(outDir, $"{baseName}_msvt_m2.obj"));

            // Write Headers
            swWmo.WriteLine($"# {baseName} MSVT WMO Geometry (CK24!=0, GroupKey!=0)");
            swM2.WriteLine($"# {baseName} MSVT M2/Residual Geometry (GroupKey=0)");

            // Write all vertices to BOTH files (simplest way to keep indices valid)
            // Optimization: Could re-index, but file size impact is negligible for this tool.
            foreach (var v in pm4.MeshVertices)
            {
                // Convert Y-Up (PM4) to Z-Up (WoW/Tools)
                string vLine = $"v {v.X:F4} {v.Z:F4} {v.Y:F4}";
                swWmo.WriteLine(vLine);
                swM2.WriteLine(vLine);
            }

            // Group surfaces by CK24
            var groups = pm4.Surfaces.GroupBy(s => s.CK24).ToList();
            
            foreach (var group in groups)
            {
                // Start a new group in OBJs
                swWmo.WriteLine($"g ck24_{group.Key:X6}");
                swM2.WriteLine($"g ck24_{group.Key:X6}");
                
                foreach (var surf in group)
                {
                    uint start = surf.MsviFirstIndex;
                    uint count = surf.IndexCount;

                    // Determine if this is WMO or M2 data
                    // Filter: WMO = GroupKey != 0 (and usually CK24 != 0, though we keep groups for structure)
                    bool isWmo = (surf.GroupKey != 0);

                    var targetSw = isWmo ? swWmo : swM2;
                    targetSw.WriteLine($"# Surface: Key={surf.GroupKey} Mask={surf.AttributeMask:X} Count={count}");

                    // PM4 topology is Triangle List (3 indices per face)
                    for (int i = 0; i < count; i += 3)
                    {
                        if (start + i + 2 >= pm4.MeshIndices.Count) break;

                        uint idx1 = pm4.MeshIndices[(int)(start + i)];
                        uint idx2 = pm4.MeshIndices[(int)(start + i + 1)];
                        uint idx3 = pm4.MeshIndices[(int)(start + i + 2)];

                        // Validate vertex indices
                        if (idx1 < pm4.MeshVertices.Count && 
                            idx2 < pm4.MeshVertices.Count && 
                            idx3 < pm4.MeshVertices.Count)
                        {
                            targetSw.WriteLine($"f {idx1 + 1} {idx2 + 1} {idx3 + 1}");
                        }
                    }
                }
            }
            Console.WriteLine($"[INFO] Wrote {baseName}_msvt_wmo.obj");
            Console.WriteLine($"[INFO] Wrote {baseName}_msvt_m2.obj");
        }

        // 2. Dump MSCN (Point Cloud)
        if (pm4.ExteriorVertices.Count > 0)
        {
            string objPath = Path.Combine(outDir, $"{baseName}_mscn.obj");
            using var sw = new StreamWriter(objPath);
            sw.WriteLine($"# MSCN Point Cloud from {baseName}");
            sw.WriteLine($"# Points: {pm4.ExteriorVertices.Count}");

            foreach (var v in pm4.ExteriorVertices)
            {
                // MSCN requires coordinate transform: Y, X, Z (Swap X/Y)
                // This aligns it with MSVT in PM4 Space.
                var alignedV = new Vector3(v.Y, v.X, v.Z);
                
                // Then convert to Z-Up (X, Z, Y) for output
                sw.WriteLine($"v {alignedV.X:F4} {alignedV.Z:F4} {alignedV.Y:F4}");
            }

            // Write point cloud as single vertices (some viewers support 'p')
            sw.WriteLine("g MSCN_Points");
            /*
            for (int i = 0; i < pm4.ExteriorVertices.Count; i++)
            {
                sw.WriteLine($"p {i+1}");
            }
            */
            // Or just leave as vertices for "Load as Point Cloud"
            // Alternatively, create tiny triangles? No, too heavy.
            
            Console.WriteLine($"[INFO] Wrote {objPath} ({pm4.ExteriorVertices.Count} points)");
        }
    }
}
