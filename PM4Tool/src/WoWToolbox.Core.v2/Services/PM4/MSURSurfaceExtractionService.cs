using System.Numerics;
using WoWToolbox.Core.v2.Foundation.Data;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.Transforms;
using WoWToolbox.Core.v2.Models.PM4.Chunks;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4;

/// <summary>
/// Service for extracting and processing MSUR surface data from PM4 files.
/// Based on proven logic from PM4FileTests.cs methods.
/// </summary>
public class MSURSurfaceExtractionService
{
    /// <summary>
    /// Represents a group of MSUR surfaces that belong to the same building/structure
    /// </summary>
    public class SurfaceGroup
    {
        public List<int> SurfaceIndices { get; set; } = new();
        public BoundingBox3D? GroupBounds { get; set; }
        public string GroupType { get; set; } = "Unknown";
        public int EstimatedBuildingIndex { get; set; }
    }

    /// <summary>
    /// Represents geometry data extracted from a single MSUR surface
    /// </summary>
    public class SurfaceGeometry
    {
        public int SurfaceIndex { get; set; }
        public List<Vector3> Vertices { get; set; } = new();
        public List<int> TriangleIndices { get; set; } = new();
        public Vector3 SurfaceNormal { get; set; }
        public float SurfaceHeight { get; set; }
        public BoundingBox3D? SurfaceBounds { get; set; }
        public SurfaceOrientation Orientation { get; set; }
        public float SurfaceArea { get; set; }
    }

    /// <summary>
    /// Surface orientation types for matching strategy
    /// </summary>
    public enum SurfaceOrientation
    {
        TopFacing,      // Roofs, upper visible surfaces (normal points up)
        BottomFacing,   // Foundations, walkable surfaces (normal points down)
        Vertical,       // Walls, sides (normal is horizontal)
        Mixed           // Complex geometry with mixed orientations
    }

    /// <summary>
    /// Group MSUR surfaces into buildings using spatial clustering.
    /// Based on GroupMSURSurfacesIntoBuildings() from PM4FileTests.cs
    /// </summary>
    public List<SurfaceGroup> GroupSurfacesIntoBuildings(PM4File pm4File)
    {
        var groups = new List<SurfaceGroup>();
        
        if (pm4File.MSUR?.Entries == null || !pm4File.MSUR.Entries.Any())
            return groups;

        var processedSurfaces = new HashSet<int>();
        
        for (int i = 0; i < pm4File.MSUR.Entries.Count; i++)
        {
            if (processedSurfaces.Contains(i))
                continue;

            var surfaceBounds = CalculateMSURSurfaceBounds(pm4File, i);
            if (!surfaceBounds.HasValue)
                continue;

            var group = new SurfaceGroup
            {
                SurfaceIndices = new List<int> { i },
                GroupBounds = surfaceBounds.Value,
                EstimatedBuildingIndex = groups.Count
            };

            // Find nearby surfaces using spatial clustering (tolerance from original implementation)
            const float spatialTolerance = 10.0f;
            
            for (int j = i + 1; j < pm4File.MSUR.Entries.Count; j++)
            {
                if (processedSurfaces.Contains(j))
                    continue;

                var otherBounds = CalculateMSURSurfaceBounds(pm4File, j);
                if (!otherBounds.HasValue)
                    continue;

                // Check if surfaces are spatially close (based on original logic)
                if (AreBoundsNearby(surfaceBounds.Value, otherBounds.Value, spatialTolerance))
                {
                    group.SurfaceIndices.Add(j);
                    processedSurfaces.Add(j);
                    
                    // Expand group bounds
                    group.GroupBounds = CombineBounds(group.GroupBounds.Value, otherBounds.Value);
                }
            }

            processedSurfaces.Add(i);
            groups.Add(group);
        }

        return groups;
    }

    /// <summary>
    /// Calculate bounding box for a specific MSUR surface.
    /// Based on CalculateMSURSurfaceBounds() from PM4FileTests.cs
    /// </summary>
    public BoundingBox3D? CalculateMSURSurfaceBounds(PM4File pm4File, int surfaceIndex)
    {
        if (pm4File.MSUR?.Entries == null || 
            surfaceIndex < 0 || 
            surfaceIndex >= pm4File.MSUR.Entries.Count)
            return null;

        var surface = pm4File.MSUR.Entries[surfaceIndex];
        
        if (pm4File.MSVI?.Indices == null || 
            surface.MsviFirstIndex >= pm4File.MSVI.Indices.Count ||
            surface.IndexCount == 0)
            return null;

        var vertices = new List<Vector3>();

        // Extract vertices using the same logic as PM4FileTests.cs
        for (uint i = 0; i < surface.IndexCount; i++)
        {
            var msviIndex = surface.MsviFirstIndex + i;
            if (msviIndex >= pm4File.MSVI.Indices.Count)
                break;

            var vertexIndex = pm4File.MSVI.Indices[(int)msviIndex];
            
            Vector3 vertex;
            if (pm4File.MSVT?.Vertices != null && vertexIndex < pm4File.MSVT.Vertices.Count)
            {
                // Use MSVT vertex data with coordinate transformation
                vertex = Pm4CoordinateTransforms.FromMsvtVertex(new MsvtVertex
                {
                    X = pm4File.MSVT.Vertices[(int)vertexIndex].X,
                    Y = pm4File.MSVT.Vertices[(int)vertexIndex].Y,
                    Z = pm4File.MSVT.Vertices[(int)vertexIndex].Z
                });
            }
            else if (pm4File.MSPV?.Vertices != null && vertexIndex < pm4File.MSPV.Vertices.Count)
            {
                // Use MSPV vertex data with coordinate transformation
                vertex = Pm4CoordinateTransforms.FromMspvVertex(pm4File.MSPV.Vertices[(int)vertexIndex]);
            }
            else
            {
                continue; // Skip invalid vertex indices
            }

            vertices.Add(vertex);
        }

        if (!vertices.Any())
            return null;

        // Calculate bounding box
        var min = new Vector3(
            vertices.Min(v => v.X),
            vertices.Min(v => v.Y), 
            vertices.Min(v => v.Z)
        );
        
        var max = new Vector3(
            vertices.Max(v => v.X),
            vertices.Max(v => v.Y),
            vertices.Max(v => v.Z)
        );

        return new BoundingBox3D(min, max);
    }

    /// <summary>
    /// Extract complete geometry from a single MSUR surface.
    /// Based on ExtractSingleMSURSurfaceIntoModel() from PM4FileTests.cs
    /// </summary>
    public SurfaceGeometry ExtractSurfaceGeometry(PM4File pm4File, int surfaceIndex)
    {
        var geometry = new SurfaceGeometry
        {
            SurfaceIndex = surfaceIndex
        };

        if (pm4File.MSUR?.Entries == null || 
            surfaceIndex < 0 || 
            surfaceIndex >= pm4File.MSUR.Entries.Count)
            return geometry;

        var surface = pm4File.MSUR.Entries[surfaceIndex];
        
        // Extract surface metadata
        geometry.SurfaceNormal = surface.SurfaceNormal;
        geometry.SurfaceHeight = surface.SurfaceHeight;
        geometry.Orientation = DetermineSurfaceOrientation(surface.SurfaceNormal);

        if (pm4File.MSVI?.Indices == null || 
            surface.MsviFirstIndex >= pm4File.MSVI.Indices.Count ||
            surface.IndexCount == 0)
            return geometry;

        var vertices = new List<Vector3>();
        var triangleIndices = new List<int>();

        // Extract vertices (same logic as CalculateMSURSurfaceBounds)
        for (uint i = 0; i < surface.IndexCount; i++)
        {
            var msviIndex = surface.MsviFirstIndex + i;
            if (msviIndex >= pm4File.MSVI.Indices.Count)
                break;

            var vertexIndex = pm4File.MSVI.Indices[(int)msviIndex];
            
            Vector3 vertex;
            if (pm4File.MSVT?.Vertices != null && vertexIndex < pm4File.MSVT.Vertices.Count)
            {
                vertex = Pm4CoordinateTransforms.FromMsvtVertex(new MsvtVertex
                {
                    X = pm4File.MSVT.Vertices[(int)vertexIndex].X,
                    Y = pm4File.MSVT.Vertices[(int)vertexIndex].Y,
                    Z = pm4File.MSVT.Vertices[(int)vertexIndex].Z
                });
            }
            else if (pm4File.MSPV?.Vertices != null && vertexIndex < pm4File.MSPV.Vertices.Count)
            {
                vertex = Pm4CoordinateTransforms.FromMspvVertex(pm4File.MSPV.Vertices[(int)vertexIndex]);
            }
            else
            {
                continue;
            }

            vertices.Add(vertex);
            triangleIndices.Add(vertices.Count - 1); // 0-based index
        }

        geometry.Vertices = vertices;
        geometry.TriangleIndices = triangleIndices;

        // Calculate bounds and surface area
        if (vertices.Any())
        {
            var min = new Vector3(
                vertices.Min(v => v.X),
                vertices.Min(v => v.Y),
                vertices.Min(v => v.Z)
            );
            
            var max = new Vector3(
                vertices.Max(v => v.X),
                vertices.Max(v => v.Y),
                vertices.Max(v => v.Z)
            );

            geometry.SurfaceBounds = new BoundingBox3D(min, max);
            geometry.SurfaceArea = EstimateSurfaceArea(triangleIndices, vertices);
        }

        return geometry;
    }

    /// <summary>
    /// Create CompleteWMOModel from a group of MSUR surfaces.
    /// Based on CreateBuildingFromMSURSurfaces_Corrected() from PM4FileTests.cs
    /// </summary>
    public CompleteWMOModel CreateBuildingFromSurfaceGroup(PM4File pm4File, SurfaceGroup surfaceGroup, string sourceFileName, int buildingIndex)
    {
        var model = new CompleteWMOModel
        {
            FileName = $"{sourceFileName}_Building_{buildingIndex:D3}",
            Category = "PM4_Building"
        };

        foreach (var surfaceIndex in surfaceGroup.SurfaceIndices)
        {
            var surfaceGeometry = ExtractSurfaceGeometry(pm4File, surfaceIndex);
            
            if (surfaceGeometry.Vertices.Any())
            {
                var vertexOffset = model.VertexCount;
                
                // Add vertices
                model.AddVertices(surfaceGeometry.Vertices.ToArray());
                
                // Add triangle indices with offset
                var offsetIndices = surfaceGeometry.TriangleIndices
                    .Select(idx => idx + vertexOffset)
                    .ToArray();
                model.AddTriangleIndices(offsetIndices);
            }
        }

        // Set metadata
        model.Metadata["PM4_SourceFile"] = sourceFileName;
        model.Metadata["PM4_BuildingIndex"] = buildingIndex;
        model.Metadata["PM4_SurfaceCount"] = surfaceGroup.SurfaceIndices.Count;
        model.Metadata["PM4_GroupBounds"] = surfaceGroup.GroupBounds?.ToString() ?? "Unknown";

        return model;
    }

    /// <summary>
    /// Determine surface orientation based on normal vector
    /// </summary>
    public SurfaceOrientation DetermineSurfaceOrientation(Vector3 normal)
    {
        var normalizedNormal = Vector3.Normalize(normal);
        
        // Check if normal points mostly up (roof/top surface)
        if (normalizedNormal.Y > 0.7f)
            return SurfaceOrientation.TopFacing;
        
        // Check if normal points mostly down (foundation/bottom surface)  
        if (normalizedNormal.Y < -0.7f)
            return SurfaceOrientation.BottomFacing;
        
        // Check if normal is mostly horizontal (wall/vertical surface)
        if (Math.Abs(normalizedNormal.Y) < 0.3f)
            return SurfaceOrientation.Vertical;
        
        return SurfaceOrientation.Mixed;
    }

    /// <summary>
    /// Check if two bounding boxes are spatially nearby
    /// </summary>
    private bool AreBoundsNearby(BoundingBox3D bounds1, BoundingBox3D bounds2, float tolerance)
    {
        var distance = Vector3.Distance(bounds1.Center, bounds2.Center);
        var combinedSize = (bounds1.Size.Length() + bounds2.Size.Length()) / 2f;
        
        return distance <= combinedSize + tolerance;
    }

    /// <summary>
    /// Combine two bounding boxes into a larger encompassing box
    /// </summary>
    private BoundingBox3D CombineBounds(BoundingBox3D bounds1, BoundingBox3D bounds2)
    {
        var min = new Vector3(
            Math.Min(bounds1.Min.X, bounds2.Min.X),
            Math.Min(bounds1.Min.Y, bounds2.Min.Y),
            Math.Min(bounds1.Min.Z, bounds2.Min.Z)
        );
        
        var max = new Vector3(
            Math.Max(bounds1.Max.X, bounds2.Max.X),
            Math.Max(bounds1.Max.Y, bounds2.Max.Y),
            Math.Max(bounds1.Max.Z, bounds2.Max.Z)
        );

        return new BoundingBox3D(min, max);
    }

    /// <summary>
    /// Estimate surface area from triangle mesh
    /// </summary>
    private float EstimateSurfaceArea(List<int> triangleIndices, List<Vector3> vertices)
    {
        float totalArea = 0f;
        
        for (int i = 0; i + 2 < triangleIndices.Count; i += 3)
        {
            var v1 = vertices[triangleIndices[i]];
            var v2 = vertices[triangleIndices[i + 1]];
            var v3 = vertices[triangleIndices[i + 2]];
            
            // Calculate triangle area using cross product
            var edge1 = v2 - v1;
            var edge2 = v3 - v1;
            var crossProduct = Vector3.Cross(edge1, edge2);
            var triangleArea = crossProduct.Length() / 2f;
            
            totalArea += triangleArea;
        }
        
        return totalArea;
    }
} 