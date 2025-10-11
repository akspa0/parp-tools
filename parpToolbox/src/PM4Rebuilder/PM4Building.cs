using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder;

/// <summary>
/// Represents a complete PM4 building assembled using native PM4 linkage data.
/// This is the target output of the unified PM4 architecture - building-scale objects with 38K-654K triangles.
/// </summary>
public class PM4Building
{
    /// <summary>
    /// Unique building identifier from MPRL.Unknown4 (parent object ID).
    /// </summary>
    public uint BuildingId { get; set; }

    /// <summary>
    /// Building position in world coordinates from MPRL placement data.
    /// </summary>
    public Vector3 Position { get; set; }

    /// <summary>
    /// All vertices for this building assembled from multiple PM4 tiles and surfaces.
    /// These are in world coordinates with cross-tile references resolved.
    /// </summary>
    public List<Vector3> Vertices { get; set; } = new();

    /// <summary>
    /// Triangle indices referencing the Vertices list.
    /// Every 3 indices form one triangle.
    /// </summary>
    public List<uint> Indices { get; set; } = new();

    /// <summary>
    /// Source MSLK link entries that were used to assemble this building.
    /// Useful for debugging and understanding the assembly process.
    /// </summary>
    public List<MslkEntry> SourceLinks { get; set; } = new();

    /// <summary>
    /// Source MSUR surface entries that contributed geometry to this building.
    /// Useful for debugging and material assignment.
    /// </summary>
    public List<MsurChunk.Entry> SourceSurfaces { get; set; } = new();

    /// <summary>
    /// 3D bounding box containing all vertices of this building.
    /// </summary>
    public BoundingBox3D Bounds { get; set; } = new();

    /// <summary>
    /// Additional metadata about this building.
    /// </summary>
    public PM4BuildingMetadata Metadata { get; set; } = new();

    /// <summary>
    /// Get the number of triangles in this building.
    /// </summary>
    public int TriangleCount => Indices.Count / 3;

    /// <summary>
    /// Get the number of vertices in this building.
    /// </summary>
    public int VertexCount => Vertices.Count;

    /// <summary>
    /// Check if this building has valid geometry.
    /// </summary>
    public bool HasValidGeometry => Vertices.Count > 0 && Indices.Count >= 3 && (Indices.Count % 3 == 0);

    /// <summary>
    /// Calculate the surface area of this building (approximate).
    /// </summary>
    public float CalculateSurfaceArea()
    {
        if (!HasValidGeometry) return 0f;

        float totalArea = 0f;
        for (int i = 0; i < Indices.Count; i += 3)
        {
            if (i + 2 < Indices.Count)
            {
                var v1 = Vertices[(int)Indices[i]];
                var v2 = Vertices[(int)Indices[i + 1]];
                var v3 = Vertices[(int)Indices[i + 2]];

                // Calculate triangle area using cross product
                var edge1 = v2 - v1;
                var edge2 = v3 - v1;
                var cross = Vector3.Cross(edge1, edge2);
                totalArea += cross.Length() * 0.5f;
            }
        }

        return totalArea;
    }

    /// <summary>
    /// Get a summary string for debugging and logging.
    /// </summary>
    public override string ToString()
    {
        return $"PM4Building {BuildingId}: {VertexCount} vertices, {TriangleCount} triangles, {SourceLinks.Count} links, {SourceSurfaces.Count} surfaces";
    }

    /// <summary>
    /// Validate the building geometry and return any issues found.
    /// </summary>
    public List<string> ValidateGeometry()
    {
        var issues = new List<string>();

        if (Vertices.Count == 0)
            issues.Add("No vertices");

        if (Indices.Count == 0)
            issues.Add("No indices");

        if (Indices.Count % 3 != 0)
            issues.Add($"Index count {Indices.Count} is not divisible by 3");

        // Check for out-of-bounds indices
        var maxVertexIndex = (uint)(Vertices.Count - 1);
        var invalidIndices = Indices.Where(i => i > maxVertexIndex).ToList();
        if (invalidIndices.Any())
            issues.Add($"{invalidIndices.Count} out-of-bounds indices (max valid: {maxVertexIndex})");

        // Check for degenerate triangles
        int degenerateTriangles = 0;
        for (int i = 0; i < Indices.Count; i += 3)
        {
            if (i + 2 < Indices.Count)
            {
                var idx1 = Indices[i];
                var idx2 = Indices[i + 1];
                var idx3 = Indices[i + 2];

                if (idx1 == idx2 || idx2 == idx3 || idx1 == idx3)
                    degenerateTriangles++;
            }
        }

        if (degenerateTriangles > 0)
            issues.Add($"{degenerateTriangles} degenerate triangles");

        return issues;
    }

    /// <summary>
    /// Export this building to OBJ format string.
    /// </summary>
    public string ToObjString(string? objectName = null)
    {
        if (!HasValidGeometry)
            return $"# Building {BuildingId} has no valid geometry\n";

        objectName ??= $"Building_{BuildingId}";

        var obj = new System.Text.StringBuilder();
        obj.AppendLine($"# PM4 Building {BuildingId}");
        obj.AppendLine($"# Vertices: {VertexCount}, Triangles: {TriangleCount}");
        obj.AppendLine($"# Position: {Position}");
        obj.AppendLine($"# Bounds: {Bounds}");
        obj.AppendLine($"o {objectName}");
        obj.AppendLine();

        // Write vertices
        foreach (var vertex in Vertices)
        {
            // Apply X-axis flip for correct orientation (from memory bank)
            obj.AppendLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
        }

        obj.AppendLine();

        // Write faces (convert from 0-based to 1-based indexing)
        for (int i = 0; i < Indices.Count; i += 3)
        {
            if (i + 2 < Indices.Count)
            {
                var idx1 = Indices[i] + 1;
                var idx2 = Indices[i + 1] + 1;
                var idx3 = Indices[i + 2] + 1;
                obj.AppendLine($"f {idx1} {idx2} {idx3}");
            }
        }

        return obj.ToString();
    }
}

/// <summary>
/// 3D bounding box for PM4 buildings.
/// </summary>
public struct BoundingBox3D
{
    public Vector3 Min { get; set; }
    public Vector3 Max { get; set; }
    public Vector3 Size { get; set; }

    public Vector3 Center => (Min + Max) * 0.5f;
    public float Volume => Size.X * Size.Y * Size.Z;
    public bool IsValid => Size.X > 0 && Size.Y > 0 && Size.Z > 0;

    public override string ToString()
    {
        return $"BBox[{Min} to {Max}, size {Size}]";
    }
}

/// <summary>
/// Additional metadata about a PM4 building.
/// </summary>
public class PM4BuildingMetadata
{
    /// <summary>
    /// Which tiles contributed geometry to this building.
    /// </summary>
    public HashSet<(int tileX, int tileY)> SourceTiles { get; set; } = new();

    /// <summary>
    /// Assembly timestamp.
    /// </summary>
    public DateTime AssemblyTime { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Assembly duration in milliseconds.
    /// </summary>
    public long AssemblyDurationMs { get; set; }

    /// <summary>
    /// Number of cross-tile references resolved during assembly.
    /// </summary>
    public int CrossTileReferencesResolved { get; set; }

    /// <summary>
    /// Any warnings or issues encountered during assembly.
    /// </summary>
    public List<string> AssemblyWarnings { get; set; } = new();

    /// <summary>
    /// Building quality metrics.
    /// </summary>
    public PM4BuildingQuality Quality { get; set; } = new();
}

/// <summary>
/// Quality metrics for a PM4 building.
/// </summary>
public class PM4BuildingQuality
{
    /// <summary>
    /// Percentage of expected geometry assembled (0-100).
    /// </summary>
    public float GeometryCompleteness { get; set; }

    /// <summary>
    /// Whether this building meets building-scale expectations (38K-654K triangles).
    /// </summary>
    public bool IsBuildingScale { get; set; }

    /// <summary>
    /// Number of geometry validation issues.
    /// </summary>
    public int ValidationIssues { get; set; }

    /// <summary>
    /// Whether all cross-tile references were resolved successfully.
    /// </summary>
    public bool CrossTileReferencesResolved { get; set; } = true;

    /// <summary>
    /// Overall quality score (0-100).
    /// </summary>
    public float OverallScore => CalculateOverallScore();

    private float CalculateOverallScore()
    {
        float score = 100f;

        // Penalize for incomplete geometry
        score *= (GeometryCompleteness / 100f);

        // Penalize for not being building scale
        if (!IsBuildingScale) score *= 0.5f;

        // Penalize for validation issues
        score -= (ValidationIssues * 5f);

        // Penalize for unresolved cross-tile references
        if (!CrossTileReferencesResolved) score *= 0.3f;

        return Math.Max(0f, Math.Min(100f, score));
    }

    public override string ToString()
    {
        return $"Quality: {OverallScore:F1}% (geometry: {GeometryCompleteness:F1}%, building-scale: {IsBuildingScale}, issues: {ValidationIssues})";
    }
}
