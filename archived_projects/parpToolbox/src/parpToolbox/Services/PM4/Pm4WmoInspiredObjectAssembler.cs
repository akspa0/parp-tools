using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// PM4 object assembler inspired by WMO organizational logic from wow.tools.local.
    /// 
    /// WMO Structure:
    /// - MOGI: Group info (bounding boxes, flags)
    /// - MOGP: Group headers (batch references)  
    /// - MOBA: Render batches (firstFace, numFaces, firstVertex, lastVertex, materialID)
    /// 
    /// PM4 Equivalent:
    /// - MPRL: Placements (positions, flags, object IDs)
    /// - MSLK: Links (geometry references)
    /// - MSUR: Surfaces (IndexCount, MspiFirstIndex, GroupKey) = MOBA equivalent
    /// </summary>
    public class Pm4WmoInspiredObjectAssembler
    {
        public Pm4WmoInspiredObjectAssembler()
        {
        }

        public class WmoInspiredObject
        {
            public uint ObjectId { get; set; }
            public string ObjectName { get; set; } = "";
            public List<Triangle> Triangles { get; set; } = new();
            public Vector3 BoundingBoxMin { get; set; }
            public Vector3 BoundingBoxMax { get; set; }
            public int TotalVertices => Triangles.SelectMany(t => new[] { t.V1, t.V2, t.V3 }).Distinct().Count();
            public int TotalTriangles => Triangles.Count;
        }

        public class Triangle
        {
            public Vector3 V1 { get; set; }
            public Vector3 V2 { get; set; }
            public Vector3 V3 { get; set; }
            public uint MaterialId { get; set; }
        }

        /// <summary>
        /// Assemble PM4 objects using WMO organizational logic:
        /// 1. Group by MPRL.Unknown4 (object IDs) - like WMO groups
        /// 2. Find MSLK links for each object - like MOGP batch references
        /// 3. Extract geometry from MSUR surfaces - like MOBA render batches
        /// </summary>
        public List<WmoInspiredObject> AssembleObjects(Pm4Scene scene)
        {
            ConsoleLogger.WriteLine("=== PM4 WMO-Inspired Object Assembly ===");
            ConsoleLogger.WriteLine($"Scene: {scene.Placements.Count} placements, {scene.Links.Count} links, {scene.Surfaces.Count} surfaces");

            var objects = new List<WmoInspiredObject>();

            // Step 1: Group by MPRL.Unknown4 (like WMO group IDs)
            var objectIds = scene.Placements.Select(p => p.Unknown4).Distinct().ToList();
            ConsoleLogger.WriteLine($"Found {objectIds.Count} unique object IDs");

            foreach (var objectId in objectIds)
            {
                var wmoObject = new WmoInspiredObject
                {
                    ObjectId = objectId,
                    ObjectName = $"Building_Object_{objectId}"
                };

                // Step 2: Find MSLK links for this object (like MOGP batch references)
                var objectLinks = scene.Links.Where(link => link.ParentId == objectId).ToList();
                ConsoleLogger.WriteLine($"Object {objectId}: Found {objectLinks.Count} geometry links");

                // Step 3: Extract geometry from MSUR surfaces (like MOBA render batches)
                foreach (var link in objectLinks)
                {
                    if (link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
                    {
                        ExtractGeometryFromSurface(link, scene, wmoObject);
                    }
                }

                if (wmoObject.Triangles.Count > 0)
                {
                    CalculateBoundingBox(wmoObject);
                    objects.Add(wmoObject);
                    ConsoleLogger.WriteLine($"Object {objectId}: {wmoObject.TotalTriangles} triangles, {wmoObject.TotalVertices} vertices");
                }
                else
                {
                    ConsoleLogger.WriteLine($"Object {objectId}: No geometry found (container/grouping node)");
                }
            }

            ConsoleLogger.WriteLine($"=== Assembly Complete: {objects.Count} objects with geometry ===");
            return objects;
        }

        /// <summary>
        /// Extract geometry from MSUR surface using WMO MOBA batch logic:
        /// - MspiFirstIndex = firstFace (like MOBA.firstFace)
        /// - MspiIndexCount = numFaces (like MOBA.numFaces)  
        /// - SurfaceRefIndex = materialID (like MOBA.materialID)
        /// </summary>
        private void ExtractGeometryFromSurface(MslkEntry link, Pm4Scene scene, WmoInspiredObject wmoObject)
        {
            try
            {
                var startIndex = link.MspiFirstIndex;
                var triangleCount = link.MspiIndexCount / 3;

                ConsoleLogger.WriteLine($"  Surface: startIndex={startIndex}, triangleCount={triangleCount}, SurfaceRef={link.SurfaceRefIndex}");

                for (int i = 0; i < triangleCount; i++)
                {
                    var indexOffset = startIndex + (i * 3);
                    
                    if (indexOffset + 2 >= scene.Indices.Count)
                    {
                        ConsoleLogger.WriteLine($"    WARNING: Index {indexOffset + 2} out of bounds (max: {scene.Indices.Count - 1})");
                        break;
                    }

                    var i1 = scene.Indices[indexOffset];
                    var i2 = scene.Indices[indexOffset + 1];
                    var i3 = scene.Indices[indexOffset + 2];

                    // Apply WMO coordinate system (X-axis inversion for proper orientation)
                    var v1 = GetVertexWithTransform((uint)i1, scene);
                    var v2 = GetVertexWithTransform((uint)i2, scene);
                    var v3 = GetVertexWithTransform((uint)i3, scene);

                    if (v1.HasValue && v2.HasValue && v3.HasValue)
                    {
                        var triangle = new Triangle
                        {
                            V1 = v1.Value,
                            V2 = v2.Value,
                            V3 = v3.Value,
                            MaterialId = (uint)link.SurfaceRefIndex // Like WMO MOBA.materialID
                        };

                        wmoObject.Triangles.Add(triangle);
                    }
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"    ERROR extracting surface geometry: {ex.Message}");
            }
        }

        /// <summary>
        /// Get vertex with WMO-style coordinate transform (X-axis inversion)
        /// </summary>
        private Vector3? GetVertexWithTransform(uint vertexIndex, Pm4Scene scene)
        {
            if (vertexIndex >= scene.Vertices.Count)
            {
                return null; // Out of bounds
            }

            var vertex = scene.Vertices[(int)vertexIndex];
            
            // Apply WMO coordinate system transformation
            return CoordinateTransformationService.ApplyPm4Transformation(vertex);
        }

        /// <summary>
        /// Calculate bounding box for the assembled object
        /// </summary>
        private void CalculateBoundingBox(WmoInspiredObject wmoObject)
        {
            if (wmoObject.Triangles.Count == 0) return;

            var allVertices = wmoObject.Triangles.SelectMany(t => new[] { t.V1, t.V2, t.V3 });
            
            wmoObject.BoundingBoxMin = new Vector3(
                allVertices.Min(v => v.X),
                allVertices.Min(v => v.Y),
                allVertices.Min(v => v.Z)
            );
            
            wmoObject.BoundingBoxMax = new Vector3(
                allVertices.Max(v => v.X),
                allVertices.Max(v => v.Y),
                allVertices.Max(v => v.Z)
            );
        }

        /// <summary>
        /// Export WMO-inspired objects to OBJ files using WMO conventions
        /// </summary>
        public void ExportObjects(List<WmoInspiredObject> objects, string outputDirectory)
        {
            ConsoleLogger.WriteLine($"=== Exporting {objects.Count} WMO-Inspired Objects ===");

            foreach (var obj in objects)
            {
                var fileName = $"{obj.ObjectName}.obj";
                var filePath = Path.Combine(outputDirectory, fileName);

                ConsoleLogger.WriteLine($"Exporting {obj.ObjectName}: {obj.TotalTriangles} triangles → {fileName}");

                using (var writer = new StreamWriter(filePath))
                {
                    writer.WriteLine($"# WMO-Inspired PM4 Object: {obj.ObjectName}");
                    writer.WriteLine($"# Object ID: {obj.ObjectId}");
                    writer.WriteLine($"# Triangles: {obj.TotalTriangles}, Vertices: {obj.TotalVertices}");
                    writer.WriteLine($"# Bounding Box: ({obj.BoundingBoxMin.X:F2}, {obj.BoundingBoxMin.Y:F2}, {obj.BoundingBoxMin.Z:F2}) to ({obj.BoundingBoxMax.X:F2}, {obj.BoundingBoxMax.Y:F2}, {obj.BoundingBoxMax.Z:F2})");
                    writer.WriteLine();

                    // Write vertices (deduplicated)
                    var uniqueVertices = obj.Triangles
                        .SelectMany(t => new[] { t.V1, t.V2, t.V3 })
                        .Distinct()
                        .ToList();

                    var vertexMap = new Dictionary<Vector3, int>();
                    for (int i = 0; i < uniqueVertices.Count; i++)
                    {
                        var v = uniqueVertices[i];
                        writer.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");
                        vertexMap[v] = i + 1; // OBJ uses 1-based indexing
                    }

                    writer.WriteLine();

                    // Write faces grouped by material (like WMO)
                    var trianglesByMaterial = obj.Triangles.GroupBy(t => t.MaterialId);
                    foreach (var materialGroup in trianglesByMaterial)
                    {
                        writer.WriteLine($"# Material Group: {materialGroup.Key}");
                        foreach (var triangle in materialGroup)
                        {
                            var i1 = vertexMap[triangle.V1];
                            var i2 = vertexMap[triangle.V2];
                            var i3 = vertexMap[triangle.V3];
                            writer.WriteLine($"f {i1} {i2} {i3}");
                        }
                        writer.WriteLine();
                    }
                }
            }

            ConsoleLogger.WriteLine("=== Export Complete ===");
        }
    }
}
