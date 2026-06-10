using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using WoWToolbox.Core.Navigation.PM4;

namespace TestCompleteExport;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            string pm4Dir = "../test_data/original_development/development";
            string outputDir = "../output/complete_combined_mesh";
            
            Console.WriteLine("Building separate render and collision meshes from all PM4 files...");
            Console.WriteLine($"Input directory: {pm4Dir}");
            Console.WriteLine($"Output directory: {outputDir}");
            
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);
            
            // Find all PM4 files
            var pm4Files = Directory.GetFiles(pm4Dir, "*.pm4");
            Console.WriteLine($"Found {pm4Files.Length} PM4 files to process");
            
            if (pm4Files.Length == 0)
            {
                Console.WriteLine("No PM4 files found!");
                return;
            }
            
            // Create separate files for render mesh and collision mesh
            string renderMeshPath = Path.Combine(outputDir, "development_complete_render_mesh.obj");
            string collisionMeshPath = Path.Combine(outputDir, "development_complete_collision_mesh.obj");
            
            using var renderSw = new StreamWriter(renderMeshPath, false, System.Text.Encoding.UTF8);
            using var collisionSw = new StreamWriter(collisionMeshPath, false, System.Text.Encoding.UTF8);
            
            // Write headers
            renderSw.WriteLine($"# Complete Development Render Mesh - All PM4 Files");
            renderSw.WriteLine($"# Generated: {DateTime.Now}");
            renderSw.WriteLine($"# Files processed: {pm4Files.Length}");
            renderSw.WriteLine($"# Contains: MSVT render vertices + MSPV structure points (suitable for game engine import)");
            renderSw.WriteLine($"# Excludes: MSCN collision data (see separate collision file)");
            renderSw.WriteLine();
            
            collisionSw.WriteLine($"# Complete Development Collision Mesh - All PM4 Files");
            collisionSw.WriteLine($"# Generated: {DateTime.Now}");
            collisionSw.WriteLine($"# Files processed: {pm4Files.Length}");
            collisionSw.WriteLine($"# Contains: MSCN collision boundary vertices (for physics/collision detection)");
            collisionSw.WriteLine($"# Separate from render mesh for game engine compatibility");
            collisionSw.WriteLine();
            
            // Track statistics
            var renderVertices = new List<(float X, float Y, float Z, string Source, string File)>();
            var collisionVertices = new List<(float X, float Y, float Z, string File)>();
            int totalRenderFaces = 0;
            int totalFiles = 0;
            int errorFiles = 0;
            
            // First pass: Collect all vertices
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var fileName = Path.GetFileNameWithoutExtension(pm4Path);
                    Console.WriteLine($"Processing {fileName}...");
                    
                    var pm4Data = PM4File.FromFile(pm4Path);
                    
                    // Skip files without MSVT (can't generate render mesh)
                    if (pm4Data.MSVT == null)
                    {
                        Console.WriteLine($"  Skipping {fileName}: No MSVT chunk");
                        continue;
                    }
                    
                    int fileRenderStartIdx = renderVertices.Count;
                    int fileCollisionStartIdx = collisionVertices.Count;
                    
                    // Add MSVT render vertices (primary geometry)
                    renderSw.WriteLine($"# {fileName} - MSVT Render Vertices ({pm4Data.MSVT.Vertices.Count})");
                    foreach (var v in pm4Data.MSVT.Vertices)
                    {
                        // Apply MSVT coordinate transform: (Y, X, Z) for proper alignment
                        var coords = new Vector3(v.Y, v.X, v.Z);
                        renderVertices.Add((coords.X, coords.Y, coords.Z, "MSVT", fileName));
                    }
                    
                    // Add MSPV structure vertices if available (secondary geometry)
                    if (pm4Data.MSPV != null && pm4Data.MSPV.Vertices.Count > 0)
                    {
                        renderSw.WriteLine($"# {fileName} - MSPV Structure Vertices ({pm4Data.MSPV.Vertices.Count})");
                        foreach (var v in pm4Data.MSPV.Vertices)
                        {
                            // Apply MSPV coordinate transform: direct (X, Y, Z)
                            var coords = new Vector3(v.X, v.Y, v.Z);
                            renderVertices.Add((coords.X, coords.Y, coords.Z, "MSPV", fileName));
                        }
                    }
                    
                    // Add MSCN collision vertices to separate file
                    if (pm4Data.MSCN != null && pm4Data.MSCN.ExteriorVertices.Count > 0)
                    {
                        collisionSw.WriteLine($"# {fileName} - MSCN Collision Vertices ({pm4Data.MSCN.ExteriorVertices.Count})");
                        foreach (var v in pm4Data.MSCN.ExteriorVertices)
                        {
                            // Apply MSCN coordinate transform: Y-axis correction + 180° rotation
                            var corrected = new Vector3(v.X, -v.Y, v.Z);
                            var coords = new Vector3(corrected.X, -corrected.Y, corrected.Z);
                            collisionVertices.Add((coords.X, coords.Y, coords.Z, fileName));
                        }
                    }
                    
                    totalFiles++;
                    Console.WriteLine($"  Added {renderVertices.Count - fileRenderStartIdx} render vertices, {collisionVertices.Count - fileCollisionStartIdx} collision vertices from {fileName}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ERROR processing {Path.GetFileName(pm4Path)}: {ex.Message}");
                    errorFiles++;
                }
            }
            
            // Write all render vertices
            renderSw.WriteLine($"\n# All Render Vertices Combined ({renderVertices.Count} total)");
            foreach (var v in renderVertices)
            {
                renderSw.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6} # {v.Source} from {v.File}");
            }
            renderSw.WriteLine();
            
            // Write all collision vertices  
            collisionSw.WriteLine($"\n# All Collision Vertices Combined ({collisionVertices.Count} total)");
            foreach (var v in collisionVertices)
            {
                collisionSw.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6} # MSCN from {v.File}");
            }
            collisionSw.WriteLine();
            
            // Generate faces for render mesh only (collision is just vertices for now)
            int globalRenderOffset = 1; // OBJ uses 1-based indexing
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var fileName = Path.GetFileNameWithoutExtension(pm4Path);
                    var pm4Data = PM4File.FromFile(pm4Path);
                    
                    if (pm4Data.MSVT == null) continue;
                    
                    // RENDER MESH FACES: Use MSUR entries to define triangle fans for MSVT vertices
                    if (pm4Data.MSUR != null && pm4Data.MSVI != null && 
                        pm4Data.MSUR.Entries.Count > 0 && pm4Data.MSVI.Indices.Count > 0)
                    {
                        renderSw.WriteLine($"# {fileName} - MSVT Render Mesh Faces");
                        renderSw.WriteLine($"o {fileName}_RenderMesh");
                        
                        foreach (var msur in pm4Data.MSUR.Entries)
                        {
                            // Validate MSVI range for this surface
                            if (msur.MsviFirstIndex >= 0 && 
                                msur.MsviFirstIndex + msur.IndexCount <= pm4Data.MSVI.Indices.Count &&
                                msur.IndexCount >= 3)
                            {
                                // Get surface indices
                                var surfaceIndices = new List<uint>();
                                for (int j = 0; j < msur.IndexCount; j++)
                                {
                                    uint msvtIdx = pm4Data.MSVI.Indices[(int)msur.MsviFirstIndex + j];
                                    surfaceIndices.Add(msvtIdx);
                                }
                                
                                // Generate triangle fan (first vertex is center)
                                if (surfaceIndices.Count >= 3)
                                {
                                    for (int k = 1; k < surfaceIndices.Count - 1; k++)
                                    {
                                        uint idx1 = surfaceIndices[0];     // Fan center
                                        uint idx2 = surfaceIndices[k];     // Current edge
                                        uint idx3 = surfaceIndices[k + 1]; // Next edge
                                        
                                        // Validate indices are within MSVT range
                                        if (idx1 < pm4Data.MSVT.Vertices.Count && 
                                            idx2 < pm4Data.MSVT.Vertices.Count && 
                                            idx3 < pm4Data.MSVT.Vertices.Count &&
                                            idx1 != idx2 && idx1 != idx3 && idx2 != idx3)
                                        {
                                            // Apply global render vertex offset (MSVT vertices only)
                                            renderSw.WriteLine($"f {idx1 + globalRenderOffset} {idx2 + globalRenderOffset} {idx3 + globalRenderOffset}");
                                            totalRenderFaces++;
                                        }
                                    }
                                }
                            }
                        }
                        renderSw.WriteLine();
                    }
                    
                    // STRUCTURE FACES: Use MSLK entries for MSPV geometry if available
                    if (pm4Data.MSLK != null && pm4Data.MSPI != null && pm4Data.MSPV != null &&
                        pm4Data.MSLK.Entries.Count > 0 && pm4Data.MSPI.Indices.Count > 0)
                    {
                        renderSw.WriteLine($"# {fileName} - Structure Faces (MSLK->MSPV)");
                        renderSw.WriteLine($"o {fileName}_Structure");
                        
                        foreach (var mslkEntry in pm4Data.MSLK.Entries)
                        {
                            if (mslkEntry.MspiFirstIndex >= 0 && 
                                mslkEntry.MspiIndexCount >= 3 && 
                                mslkEntry.MspiFirstIndex + mslkEntry.MspiIndexCount <= pm4Data.MSPI.Indices.Count)
                            {
                                // Get valid MSPV indices from MSPI
                                var validIndices = new List<uint>();
                                for (int i = 0; i < mslkEntry.MspiIndexCount; i++)
                                {
                                    uint mspiIdx = pm4Data.MSPI.Indices[mslkEntry.MspiFirstIndex + i];
                                    
                                    if (mspiIdx < pm4Data.MSPV.Vertices.Count)
                                    {
                                        // Calculate offset for MSPV vertices in render mesh (after MSVT vertices)
                                        uint adjustedIdx = (uint)(globalRenderOffset + pm4Data.MSVT.Vertices.Count + mspiIdx);
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
                                        
                                        if (v1 <= renderVertices.Count && v2 <= renderVertices.Count && v3 <= renderVertices.Count)
                                        {
                                            renderSw.WriteLine($"f {v1} {v2} {v3}");
                                            totalRenderFaces++;
                                        }
                                    }
                                }
                            }
                        }
                        renderSw.WriteLine();
                    }
                    
                    // Update global render vertex offset for next file (MSVT + MSPV only)
                    int msvtCount = pm4Data.MSVT.Vertices.Count;
                    int mspvCount = pm4Data.MSPV?.Vertices.Count ?? 0;
                    globalRenderOffset += msvtCount + mspvCount;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ERROR generating faces for {Path.GetFileName(pm4Path)}: {ex.Message}");
                }
            }
            
            // Final statistics
            Console.WriteLine($"\nSeparate mesh export completed:");
            Console.WriteLine($"  RENDER MESH:");
            Console.WriteLine($"    Total vertices: {renderVertices.Count} (MSVT + MSPV)");
            Console.WriteLine($"    Total faces: {totalRenderFaces}");
            Console.WriteLine($"    Output: {renderMeshPath}");
            Console.WriteLine($"  COLLISION MESH:");
            Console.WriteLine($"    Total vertices: {collisionVertices.Count} (MSCN)");
            Console.WriteLine($"    Total faces: 0 (vertices only for collision detection)");
            Console.WriteLine($"    Output: {collisionMeshPath}");
            Console.WriteLine($"  FILES:");
            Console.WriteLine($"    Processed successfully: {totalFiles}");
            Console.WriteLine($"    Files with errors: {errorFiles}");
            Console.WriteLine();
            Console.WriteLine("Files are now separated for optimal game engine compatibility:");
            Console.WriteLine("- Use render mesh for 3D visualization and texturing");
            Console.WriteLine("- Use collision mesh for physics and boundary detection");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"FATAL ERROR: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
    }
}
