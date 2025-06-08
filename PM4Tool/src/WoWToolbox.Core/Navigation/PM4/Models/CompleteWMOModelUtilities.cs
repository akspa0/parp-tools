using System.Numerics;

namespace WoWToolbox.Core.Navigation.PM4.Models
{
    /// <summary>
    /// Utilities for processing CompleteWMOModel instances.
    /// Provides normal generation, OBJ export, and other model processing operations.
    /// </summary>
    public static class CompleteWMOModelUtilities
    {
        /// <summary>
        /// Generates smooth vertex normals for a CompleteWMOModel by averaging face normals.
        /// Updates the model's Normals collection in place.
        /// </summary>
        /// <param name="model">The model to generate normals for</param>
        public static void GenerateNormals(CompleteWMOModel model)
        {
            model.Normals.Clear();
            model.Normals.AddRange(Enumerable.Repeat(Vector3.Zero, model.Vertices.Count));
            
            // Calculate face normals and accumulate
            for (int i = 0; i < model.TriangleIndices.Count; i += 3)
            {
                int i0 = model.TriangleIndices[i];
                int i1 = model.TriangleIndices[i + 1];
                int i2 = model.TriangleIndices[i + 2];
                
                if (i0 < model.Vertices.Count && i1 < model.Vertices.Count && i2 < model.Vertices.Count)
                {
                    var v0 = model.Vertices[i0];
                    var v1 = model.Vertices[i1];
                    var v2 = model.Vertices[i2];
                    
                    var normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
                    
                    model.Normals[i0] += normal;
                    model.Normals[i1] += normal;
                    model.Normals[i2] += normal;
                }
            }
            
            // Normalize accumulated normals
            for (int i = 0; i < model.Normals.Count; i++)
            {
                if (model.Normals[i] != Vector3.Zero)
                {
                    model.Normals[i] = Vector3.Normalize(model.Normals[i]);
                }
                else
                {
                    model.Normals[i] = Vector3.UnitY; // Default up normal
                }
            }
        }
        
        /// <summary>
        /// Exports a CompleteWMOModel to OBJ format with accompanying MTL material file.
        /// Creates professional 3D-compatible output with normals, texture coordinates, and materials.
        /// </summary>
        /// <param name="model">The model to export</param>
        /// <param name="outputPath">Path for the .obj file (corresponding .mtl will be created automatically)</param>
        public static void ExportToOBJ(CompleteWMOModel model, string outputPath)
        {
            using (var writer = new StreamWriter(outputPath))
            {
                // Write OBJ header
                writer.WriteLine($"# Complete WMO Model Export");
                writer.WriteLine($"# Generated: {DateTime.Now}");
                writer.WriteLine($"# Source: {model.Metadata.GetValueOrDefault("SourceFile", "Unknown")}");
                writer.WriteLine($"# Category: {model.Category}");
                writer.WriteLine($"# Vertices: {model.VertexCount:N0}");
                writer.WriteLine($"# Faces: {model.FaceCount:N0}");
                writer.WriteLine();
                
                writer.WriteLine($"mtllib {Path.GetFileNameWithoutExtension(outputPath)}.mtl");
                writer.WriteLine($"usemtl {model.MaterialName}");
                writer.WriteLine();
                
                // Write vertices
                foreach (var vertex in model.Vertices)
                {
                    writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }
                
                // Write texture coordinates
                foreach (var texCoord in model.TexCoords)
                {
                    writer.WriteLine($"vt {texCoord.X:F6} {texCoord.Y:F6}");
                }
                
                // Write normals
                foreach (var normal in model.Normals)
                {
                    writer.WriteLine($"vn {normal.X:F6} {normal.Y:F6} {normal.Z:F6}");
                }
                
                writer.WriteLine();
                
                // Write faces
                for (int i = 0; i < model.TriangleIndices.Count; i += 3)
                {
                    int v1 = model.TriangleIndices[i] + 1; // OBJ is 1-indexed
                    int v2 = model.TriangleIndices[i + 1] + 1;
                    int v3 = model.TriangleIndices[i + 2] + 1;
                    
                    if (model.TexCoords.Count > 0 && model.Normals.Count > 0)
                    {
                        writer.WriteLine($"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}");
                    }
                    else if (model.Normals.Count > 0)
                    {
                        writer.WriteLine($"f {v1}//{v1} {v2}//{v2} {v3}//{v3}");
                    }
                    else
                    {
                        writer.WriteLine($"f {v1} {v2} {v3}");
                    }
                }
            }
            
            // Create material file
            var mtlPath = Path.ChangeExtension(outputPath, ".mtl");
            using (var writer = new StreamWriter(mtlPath))
            {
                writer.WriteLine($"newmtl {model.MaterialName}");
                writer.WriteLine("Ka 0.2 0.2 0.2");
                writer.WriteLine("Kd 0.8 0.8 0.8");
                writer.WriteLine("Ks 0.5 0.5 0.5");
                writer.WriteLine("Ns 96.0");
                writer.WriteLine("d 1.0");
            }
        }
        
        /// <summary>
        /// Calculates the bounding box of all vertices in the model.
        /// </summary>
        /// <param name="model">The model to calculate bounds for</param>
        /// <returns>Bounding box containing all vertices, or null if no vertices</returns>
        public static BoundingBox3D? CalculateBoundingBox(CompleteWMOModel model)
        {
            if (model.Vertices.Count == 0)
                return null;
                
            var min = model.Vertices[0];
            var max = model.Vertices[0];
            
            foreach (var vertex in model.Vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }
            
            var center = (min + max) * 0.5f;
            var size = max - min;
            
            return new BoundingBox3D
            {
                Min = min,
                Max = max,
                Center = center,
                Size = size
            };
        }
    }
} 