// GltfExporter removed – stub to satisfy old references
namespace ParpToolbox.Services.PM4
{
    internal static class GltfExporter { }
}
#if false
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;

namespace ParpToolbox.Services.PM4
{
    internal static class GltfExporter
    {
        public static void Export(List<Pm4MsurObjectAssembler.MsurObject> objects, Pm4Scene scene, string outputPath)
        {
            ConsoleLogger.WriteLine($"[gltf] Building scene with {objects.Count} objects …");

            var sceneBuilder = new SceneBuilder();
            var material = new MaterialBuilder("default");

            foreach (var obj in objects)
            {
                var mesh = new MeshBuilder<VertexPositionNormal, VertexEmpty, VertexEmpty>($"Obj_{obj.SurfaceGroupKey}");
                var prim = mesh.UsePrimitive(material);

                foreach (var (a, b, c) in obj.Triangles)
                {
                    prim.AddTriangle(ToVertex(a), ToVertex(b), ToVertex(c));
                }

                sceneBuilder.AddRigidMesh(mesh, Matrix4x4.Identity)
                            .WithName(obj.ObjectType);
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

            if (outputPath.EndsWith(".glb", StringComparison.OrdinalIgnoreCase))
                sceneBuilder.ToGltf2().SaveGLB(outputPath);
            else
                sceneBuilder.ToGltf2().SaveGLTF(outputPath);

            ConsoleLogger.WriteLine($"[gltf] Export successful: {Path.GetFileName(outputPath)}");

            // Local helper
            VertexPositionNormal ToVertex(int index)
            {
                Vector3 v = (index >= 0 && index < scene.Vertices.Count) ? scene.Vertices[index] : Vector3.Zero;
                if (v == Vector3.Zero && (index < 0 || index >= scene.Vertices.Count))
                    ConsoleLogger.WriteLine($"[gltf] Warning: invalid vertex index {index}");

                // Mirror X to convert WoW LH to glTF RH
                return new VertexPositionNormal(new Vector3(-v.X, v.Y, v.Z), Vector3.UnitY);
            }
        }
    }
}
using System.Numerics;
using ParpToolbox.Utils;
using ParpToolbox.Formats.PM4;
using SharpGLTF.Geometry;
using SharpGLTF.Geometry.VertexTypes;
using SharpGLTF.Materials;
using SharpGLTF.Scenes;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Simple glTF 2.0 exporter for assembled PM4 objects.  
/// This first-pass implementation focuses on geometry; materials and UVs are ignored.
/// </summary>
internal static class GltfExporter
{
    /// <summary>
    /// Export assembled PM4 objects to glTF/GLB.
    /// </summary>
    public static void Export(List<Pm4MsurObjectAssembler.MsurObject> objects, Pm4Scene scene, string outputPath)
    {
        ConsoleLogger.WriteLine($"[gltf] Building scene with {objects.Count} objects …");

        var sceneBuilder = new SceneBuilder();
        var defaultMat   = new MaterialBuilder("default");

        foreach (var obj in objects)
        {
            // Build a mesh for this object
            var meshBuilder = new MeshBuilder<VertexPositionNormal, VertexEmpty, VertexEmpty>($"Obj_{obj.SurfaceGroupKey}");
            var prim        = meshBuilder.UsePrimitive(defaultMat);

            // Triangles
            foreach (var (a, b, c) in obj.Triangles)
            {
                prim.AddTriangle(ToVertex(a), ToVertex(b), ToVertex(c));
            }

            // Attach the mesh to the scene graph
            sceneBuilder.AddRigidMesh(meshBuilder, Matrix4x4.Identity)
                        .WithName(obj.ObjectType);
        }

        // Ensure output directory exists
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        // Save
        if (outputPath.EndsWith(".glb", StringComparison.OrdinalIgnoreCase))
            sceneBuilder.ToGltf2().SaveGLB(outputPath);
        else
            sceneBuilder.ToGltf2().SaveGLTF(outputPath);

        ConsoleLogger.WriteLine($"[gltf] Export successful: {Path.GetFileName(outputPath)}");

        // Local helper ------------------------------------------------------
        VertexPositionNormal ToVertex(int globalIndex)
        {
            Vector3 v = (globalIndex >= 0 && globalIndex < scene.Vertices.Count)
                ? scene.Vertices[globalIndex]
                : Vector3.Zero;

            if (v == Vector3.Zero && (globalIndex < 0 || globalIndex >= scene.Vertices.Count))
            {
                ConsoleLogger.WriteLine($"[gltf] Warning: invalid vertex index {globalIndex}");
            }

            // Mirror X to convert left-handed WoW coords to glTF right-handed
            return new VertexPositionNormal(new Vector3(-v.X, v.Y, v.Z), Vector3.UnitY);
        }
    }
}
{
    /// <summary>
    /// Exports assembled objects (from <see cref="Pm4MsurObjectAssembler"/>) to a .gltf or .glb file.
    /// </summary>
    /// <param name="objects">List of assembled objects.</param>
    /// <param name="scene">Original PM4 scene (for vertex lookup).</param>
    /// <param name="outputPath">Full output file path, extension dictates format (.gltf or .glb).</param>
    public static void Export(List<Pm4MsurObjectAssembler.MsurObject> objects, Pm4Scene scene, string outputPath)
    {
        ConsoleLogger.WriteLine($"[gltf] Building scene with {objects.Count} objects …");

        // Build glTF scene
        var sceneBuilder = new SceneBuilder();
        var defaultMat   = new MaterialBuilder("default");

        foreach (var obj in objects)
        {
            // Create a mesh for this assembled object
            var meshBuilder = new MeshBuilder<VertexPositionNormal, VertexEmpty, VertexEmpty>($"Obj_{obj.SurfaceGroupKey}");
            var prim        = meshBuilder.UsePrimitive(defaultMat);

            // Add triangles
            foreach (var (a, b, c) in obj.Triangles)
            {
                prim.AddTriangle(ToVertex(a), ToVertex(b), ToVertex(c));
            }

            // Attach mesh to the scene graph
            sceneBuilder.AddRigidMesh(meshBuilder, Matrix4x4.Identity)
                        .WithName(obj.ObjectType);
        }

        // Ensure directory exists
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        // Save out
        if (outputPath.EndsWith(".glb", StringComparison.OrdinalIgnoreCase))
            sceneBuilder.ToGltf2().SaveGLB(outputPath);
        else
            sceneBuilder.ToGltf2().SaveGLTF(outputPath);

        ConsoleLogger.WriteLine($"[gltf] Export successful: {Path.GetFileName(outputPath)}");

        // Local helper converts PM4 vertex to SharpGLTF vertex
        VertexPositionNormal ToVertex(int globalIndex)
        {
            Vector3 v;
            if (globalIndex >= 0 && globalIndex < scene.Vertices.Count)
            {
                v = scene.Vertices[globalIndex];
            }
            else
            {
                v = Vector3.Zero;
                ConsoleLogger.WriteLine($"[gltf] Warning: invalid vertex index {globalIndex}");
            }

            // Mirror X to match OBJ path (WoW left-handed → glTF right-handed)
            return new VertexPositionNormal(new Vector3(-v.X, v.Y, v.Z), Vector3.UnitY);
        }
    }
    {
        ConsoleLogger.WriteLine($"[gltf] Building scene with {objects.Count} objects …");

        var sceneBuilder = new SceneBuilder();
        var defaultMat = new MaterialBuilder("default");

        foreach (var obj in objects)
        {
            var meshBuilder = new MeshBuilder<VertexPositionNormal, VertexEmpty, VertexEmpty>($"Obj_{obj.SurfaceGroupKey}");
            var prim = meshBuilder.UsePrimitive(defaultMat);

            foreach (var (A, B, C) in obj.Triangles)
            {
                prim.AddTriangle(ToVertex(A), ToVertex(B), ToVertex(C));
            }

            // helper converts global index to vertex struct
            VertexPositionNormal ToVertex(int globalIndex)
            {
                Vector3 v;
                if (globalIndex >= 0 && globalIndex < scene.Vertices.Count)
                    v = scene.Vertices[globalIndex];
                else
                {
                    v = Vector3.Zero;
                    ConsoleLogger.WriteLine($"[gltf] Warning: invalid vertex index {globalIndex}");
                }
                return new VertexPositionNormal(new Vector3(-v.X, v.Y, v.Z), Vector3.UnitY);
            }

            int MapVertex(int globalIndex)
            {
                if (!vMap.TryGetValue(globalIndex, out int localIdx))
                {
                    Vector3 v;
                    if (globalIndex >= 0 && globalIndex < scene.Vertices.Count)
                        v = scene.Vertices[globalIndex];
                    }
                    else
                    {
                        v = Vector3.Zero;
                        ConsoleLogger.WriteLine($"[gltf] Warning: invalid vertex index {globalIndex}");
                    }

                    // SharpGLTF expects right-handed; our OBJ exporter flips X so we do the same here.
                    var vp = new VertexPositionNormal(new Vector3(-v.X, v.Y, v.Z), Vector3.UnitY);
                    localIdx = prim.AddVertex(vp);
                    vMap[globalIndex] = localIdx;
                }
                return localIdx;
            }
        }

        // Save
        var dir = Path.GetDirectoryName(outputPath)!;
        Directory.CreateDirectory(dir);

        if (outputPath.EndsWith(".glb", StringComparison.OrdinalIgnoreCase))
        {
            sceneBuilder.ToGltf2().SaveGLB(outputPath);
        }
        else
        {
            sceneBuilder.ToGltf2().SaveGLTF(outputPath);
        }

        ConsoleLogger.WriteLine($"[gltf] Export successful: {Path.GetFileName(outputPath)}");
    }
}
