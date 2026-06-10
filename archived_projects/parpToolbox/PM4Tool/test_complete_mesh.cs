using System;
using System.IO;
using WoWToolbox.Core.Navigation.PM4;

namespace TestCompleteMesh
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                string pm4Path = "test_data/development/development_22_18.pm4";
                string outputDir = "output/test_fixed_faces";
                
                Console.WriteLine("Testing complete mesh export...");
                Console.WriteLine($"Input: {pm4Path}");
                Console.WriteLine($"Output: {outputDir}");
                
                // Load PM4 file
                var pm4Data = PM4File.FromFile(pm4Path);
                string basename = Path.GetFileNameWithoutExtension(pm4Path);
                string objPath = Path.Combine(outputDir, $"{basename}_complete_mesh.obj");
                
                // Only proceed if we have MSVT
                if (pm4Data.MSVT == null)
                {
                    Console.WriteLine($"ERROR: Cannot export complete mesh: {basename} is missing MSVT chunk");
                    return;
                }
                
                using var sw = new StreamWriter(objPath, false, System.Text.Encoding.UTF8);
                sw.WriteLine($"# Complete PM4 Mesh for {basename} - All Geometry with Proper Faces");
                sw.WriteLine($"# Exported: {DateTime.Now}");
                sw.WriteLine($"# Features: MSVT render mesh + MSCN collision + MSPV structure with MSLK linking");
                
                // Track all vertices with their source chunk and vertex offsets
                var allVertices = new List<(float X, float Y, float Z, string Source)>();
                int msvtVertexOffset = 0;
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
                        var vector3 = new System.Numerics.Vector3(v.X, v.Y, v.Z);
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
                        var vector3 = new System.Numerics.Vector3(v.X, v.Y, v.Z);
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
                
                Console.WriteLine($"Complete mesh exported for {basename}:");
                Console.WriteLine($"  Total vertices: {allVertices.Count} (MSVT: {msvtCount}, MSCN: {mscnCount}, MSPV: {mspvCount})");
                Console.WriteLine($"  Total faces: {totalFaces}");
                Console.WriteLine($"  Output: {objPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
        
        private static System.Numerics.Vector3 ToUnifiedWorld(System.Numerics.Vector3 v)
        {
            // Apply: X' = -Y, Y' = -Z, Z' = X
            // This corrects horizontal mirroring and Z inversion for combined mesh exports.
            return new System.Numerics.Vector3(-v.Y, -v.Z, v.X);
        }
    }
} 