namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;


/// <summary>
/// Assembles PM4 objects using MSUR SurfaceGroupKey grouping with MPRL transformation support,
/// as documented in legacy notes and CSV analysis. Applies MPRL placement transformations
/// to correctly position geometry in world space.
/// </summary>
internal static class Pm4MsurObjectAssembler
{
    /// <summary>
    /// Represents a complete building object assembled using MSUR SurfaceGroupKey grouping.
    /// </summary>
    public record MsurObject(
        byte SurfaceGroupKey,          // MSUR.SurfaceGroupKey - the actual object identifier
        int SurfaceCount,              // Number of surfaces in this object
        List<(int A, int B, int C)> Triangles, // All triangles for this object
        Vector3 BoundingCenter,        // Calculated center point
        int VertexCount,
        string ObjectType              // Descriptive name based on surface properties
    );

    /// <summary>
    /// MPRL to MSLK mapping entry for transformation application.
    /// </summary>
    private record MprlMslkMapping(
        int MprlIndex,
        uint MprlUnknown4,
        uint MslkParentIndex,
        Vector3 Position,
        MprlChunk.Entry MprlEntry
    );

    /// <summary>
    /// Transforms an MPRL entry position to world coordinates.
    /// Based on legacy coordinate transformation: (-Z, Y, X) mapping.
    /// </summary>
    private static Vector3 TransformMprlPosition(MprlChunk.Entry entry)
    {
        // PM4-relative transform: Rotate 90 degrees counter-clockwise to align with other chunks
        return new Vector3(-entry.Position.Z, entry.Position.Y, entry.Position.X);
    }

    /// <summary>
    /// Builds MPRL to MSLK mapping based on cross-reference patterns discovered in CSV analysis.
    /// Maps MPRL.Unknown4 values to MSLK.ParentIndex values for transformation application.
    /// </summary>
    private static List<MprlMslkMapping> BuildMprlMslkMappings(Pm4Scene scene)
    {
        var mappings = new List<MprlMslkMapping>();
        
        ConsoleLogger.WriteLine($"Building MPRL→MSLK mappings from {scene.Placements.Count} MPRL entries and {scene.Links.Count} MSLK entries...");
        
        for (int mprlIndex = 0; mprlIndex < scene.Placements.Count; mprlIndex++)
        {
            var mprlEntry = scene.Placements[mprlIndex];
            var unknown4 = mprlEntry.Unknown4;
            
            // Find MSLK entries where ParentIndex matches MPRL.Unknown4
            var matchingMslkEntries = scene.Links
                .Select((link, index) => new { Link = link, Index = index })
                .Where(x => x.Link.ParentIndex == unknown4)
                .ToList();
            
            if (matchingMslkEntries.Any())
            {
                ConsoleLogger.WriteLine($"  MPRL[{mprlIndex}].Unknown4={unknown4} → {matchingMslkEntries.Count} MSLK matches");
                
                foreach (var match in matchingMslkEntries)
                {
                    mappings.Add(new MprlMslkMapping(
                        mprlIndex,
                        unknown4,
                        match.Link.ParentIndex,
                        TransformMprlPosition(mprlEntry),
                        mprlEntry
                    ));
                }
            }
        }
        
        ConsoleLogger.WriteLine($"Built {mappings.Count} MPRL→MSLK mappings");
        return mappings;
    }

    /// <summary>
    /// Assembles PM4 objects by grouping geometry using MSUR SurfaceGroupKey with MPRL transformation support.
    /// Builds local vertex pools per object and correctly remaps face indices to avoid out-of-bounds errors.
    /// </summary>
    public static List<MsurObject> AssembleObjectsByMsurIndex(Pm4Scene scene)
    {
        var assembledObjects = new List<MsurObject>();
        
        ConsoleLogger.WriteLine($"Assembling PM4 objects using SurfaceGroupKey grouping with MPRL transformations...");
        ConsoleLogger.WriteLine($"  MSUR surfaces: {scene.Surfaces.Count}");
        ConsoleLogger.WriteLine($"  MSLK links: {scene.Links.Count}");
        ConsoleLogger.WriteLine($"  MPRL placements: {scene.Placements.Count}");
        ConsoleLogger.WriteLine($"  Scene vertices: {scene.Vertices.Count} (indices 0-{scene.Vertices.Count - 1})");
        ConsoleLogger.WriteLine($"  Scene indices: {scene.Indices.Count}");
        
        // Build MPRL to MSLK mappings for transformation application
        var mprlMappings = BuildMprlMslkMappings(scene);
        
        // Process each MSUR surface individually, then group by SurfaceGroupKey
        var objectsBySurfaceGroupKey = new Dictionary<uint, List<(Vector3[] vertices, int[] faces)>>();
        
        // Track vertex index usage for debugging
        int maxVertexIndexSeen = -1;
        int outOfBoundsCount = 0;
        
        ConsoleLogger.WriteLine($"  Processing {scene.Surfaces.Count} individual MSUR surfaces...");
        
        foreach (var surface in scene.Surfaces)
        {
            if (surface.IndexCount <= 0) continue; // Skip surfaces without geometry
            
            // Extract geometry for this specific surface only
            var surfaceVertices = new List<Vector3>();
            var surfaceFaces = new List<int>();
            
            ExtractSurfaceTrianglesWithTransform(surface, scene, mprlMappings, surfaceVertices, surfaceFaces, ref maxVertexIndexSeen, ref outOfBoundsCount);
            
            if (surfaceVertices.Count > 0 && surfaceFaces.Count > 0)
            {
                // Group by SurfaceGroupKey for final object assembly
                if (!objectsBySurfaceGroupKey.TryGetValue(surface.SurfaceGroupKey, out var surfaceList))
                {
                    surfaceList = new List<(Vector3[], int[])>();
                    objectsBySurfaceGroupKey[surface.SurfaceGroupKey] = surfaceList;
                }
                
                surfaceList.Add((surfaceVertices.ToArray(), surfaceFaces.ToArray()));
            }
        }
        
        ConsoleLogger.WriteLine($"  Found {objectsBySurfaceGroupKey.Count} distinct SurfaceGroupKey groups (complete objects)");
        
        // Assemble each object by consolidating all surfaces with the same SurfaceGroupKey
        foreach (var (surfaceGroupKey, surfaceList) in objectsBySurfaceGroupKey)
        {
            var consolidatedVertices = new List<Vector3>();
            var consolidatedFaces = new List<int>();
            
            ConsoleLogger.WriteLine($"    Processing SurfaceGroupKey {surfaceGroupKey} with {surfaceList.Count} surfaces");
            
            // Consolidate all surfaces for this IndexCount into a single object
            foreach (var (surfaceVertices, surfaceFaces) in surfaceList)
            {
                int vertexOffset = consolidatedVertices.Count;
                
                // Add vertices
                consolidatedVertices.AddRange(surfaceVertices);
                
                // Add faces with proper vertex offset
                for (int i = 0; i < surfaceFaces.Length; i++)
                {
                    consolidatedFaces.Add(surfaceFaces[i] + vertexOffset);
                }
            }
            
            if (consolidatedVertices.Count > 0)
            {
                ConsoleLogger.WriteLine($"      Consolidated {consolidatedVertices.Count} vertices and {consolidatedFaces.Count} face indices for SurfaceGroupKey {surfaceGroupKey}");
                
                // Calculate bounding center from processed vertices
                var boundingCenter = CalculateBoundingCenter(consolidatedVertices);
                var objectType = GetObjectTypeName(surfaceGroupKey);
                
                // Convert faces to triangles for compatibility
                var triangles = new List<(int A, int B, int C)>();
                for (int i = 0; i < consolidatedFaces.Count; i += 3)
                {
                    if (i + 2 < consolidatedFaces.Count)
                    {
                        triangles.Add((consolidatedFaces[i], consolidatedFaces[i + 1], consolidatedFaces[i + 2]));
                    }
                }
                
                var msurObject = new MsurObject(
                    SurfaceGroupKey: (byte)surfaceGroupKey, // Use SurfaceGroupKey as identifier
                    SurfaceCount: surfaceList.Count,
                    Triangles: triangles,
                    BoundingCenter: boundingCenter,
                    VertexCount: consolidatedVertices.Count,
                    ObjectType: objectType
                );
                
                assembledObjects.Add(msurObject);
                
                ConsoleLogger.WriteLine($"      Object SurfaceGroupKey={surfaceGroupKey}: {triangles.Count} triangles, {consolidatedVertices.Count} vertices, {surfaceList.Count} surfaces");
                
                if (triangles.Count == 0)
                {
                    ConsoleLogger.WriteLine($"        WARNING: No triangles generated for SurfaceGroupKey {surfaceGroupKey}!");
                }
            }
        }
        
        ConsoleLogger.WriteLine($"Assembled {assembledObjects.Count} complete building objects");
        
        // Log vertex index statistics for debugging
        ConsoleLogger.WriteLine($"Vertex Index Statistics:");
        ConsoleLogger.WriteLine($"  Scene has vertices 0-{scene.Vertices.Count - 1} ({scene.Vertices.Count} total)");
        ConsoleLogger.WriteLine($"  Max vertex index accessed: {maxVertexIndexSeen}");
        ConsoleLogger.WriteLine($"  Out-of-bounds vertex accesses: {outOfBoundsCount}");
        if (outOfBoundsCount > 0)
        {
            ConsoleLogger.WriteLine($"  ERROR: {outOfBoundsCount} vertex indices were out of bounds - data loss occurred!");
        }
        
        return assembledObjects;
    }
    
    /// <summary>
    /// Extracts triangles from a surface using legacy-compatible plane projection logic.
    /// This matches the exact behavior of the working MsurObjectExporter.
    /// </summary>
    private static void ExtractSurfaceTriangles(
        ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface,
        Pm4Scene scene,
        List<Vector3> vertices,
        List<int> faces,
        ref int maxVertexIndexSeen,
        ref int outOfBoundsCount)
    {
        // Legacy coordinate transformation and plane projection logic
        var surfaceNormal = new Vector3(surface.Nx, surface.Ny, surface.Nz);
        
        // Check if surface normal should be swapped (legacy flag 18 check)
        bool shouldSwapNormal = ShouldSwapNormal(surface);
        if (shouldSwapNormal)
        {
            surfaceNormal = new Vector3(surface.Ny, surface.Nx, surface.Nz);
        }
        
        var finalNormal = Vector3.Normalize(surfaceNormal);
        bool usePlane = finalNormal.LengthSquared() > 0.01f; // Only project if normal is valid
        
        // Extract triangles for this surface with proper bounds checking
        int startIndex = (int)surface.MsviFirstIndex;
        int indexCount = (int)surface.IndexCount;
        
        // Validate index bounds before processing
        if (startIndex < 0 || startIndex >= scene.Indices.Count)
        {
            ConsoleLogger.WriteLine($"Warning: Surface has invalid start index {startIndex} (scene has {scene.Indices.Count} indices)");
            return;
        }
        
        if (startIndex + indexCount > scene.Indices.Count)
        {
            ConsoleLogger.WriteLine($"Warning: Surface index range [{startIndex}, {startIndex + indexCount}) exceeds scene indices ({scene.Indices.Count})");
            indexCount = scene.Indices.Count - startIndex;
        }
        
        // Build vertex mapping for this surface to avoid duplicate vertices
        var vertexMap = new Dictionary<int, int>();
        
        for (int i = 0; i < indexCount; i += 3)
        {
            int m = startIndex + i;
            if (m + 2 >= scene.Indices.Count) break; // Final bounds check
            
            int aIdx = scene.Indices[m];
            int bIdx = scene.Indices[m + 1];
            int cIdx = scene.Indices[m + 2];
            
            // Track max vertex index for debugging
            maxVertexIndexSeen = Math.Max(maxVertexIndexSeen, Math.Max(aIdx, Math.Max(bIdx, cIdx)));
            
            // Add vertices with plane projection and proper local indexing
            int localA = GetOrAddVertex(aIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, ref outOfBoundsCount);
            int localB = GetOrAddVertex(bIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, ref outOfBoundsCount);
            int localC = GetOrAddVertex(cIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, ref outOfBoundsCount);
            
            // Only add valid triangles
            if (localA >= 0 && localB >= 0 && localC >= 0)
            {
                faces.Add(localA);
                faces.Add(localB);
                faces.Add(localC);
            }
        }
    }
    
    /// <summary>
    /// Extracts triangles from a surface with MPRL transformation support.
    /// Applies MPRL placement transformations to correctly position geometry in world space.
    /// </summary>
    private static void ExtractSurfaceTrianglesWithTransform(
        ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface,
        Pm4Scene scene,
        List<MprlMslkMapping> mprlMappings,
        List<Vector3> vertices,
        List<int> faces,
        ref int maxVertexIndexSeen,
        ref int outOfBoundsCount)
    {
        // Legacy coordinate transformation and plane projection logic
        var surfaceNormal = new Vector3(surface.Nx, surface.Ny, surface.Nz);
        
        // Check if surface normal should be swapped (legacy flag 18 check)
        bool shouldSwapNormal = ShouldSwapNormal(surface);
        if (shouldSwapNormal)
        {
            surfaceNormal = new Vector3(surface.Ny, surface.Nx, surface.Nz);
        }
        
        var finalNormal = Vector3.Normalize(surfaceNormal);
        bool usePlane = finalNormal.LengthSquared() > 0.01f; // Only project if normal is valid
        
        // Find applicable MPRL transformations for this surface's group
        var applicableTransforms = mprlMappings
            .Where(m => m.MslkParentIndex == surface.SurfaceGroupKey)
            .ToList();
        
        if (applicableTransforms.Any())
        {
            ConsoleLogger.WriteLine($"    Surface SurfaceGroupKey={surface.SurfaceGroupKey}: Found {applicableTransforms.Count} MPRL transforms");
        }
        
        // Extract triangles for this surface with proper bounds checking
        int startIndex = (int)surface.MsviFirstIndex;
        int indexCount = (int)surface.IndexCount;
        
        // Validate index bounds before processing
        if (startIndex < 0 || startIndex >= scene.Indices.Count)
        {
            ConsoleLogger.WriteLine($"Warning: Surface has invalid start index {startIndex} (scene has {scene.Indices.Count} indices)");
            return;
        }
        
        if (startIndex + indexCount > scene.Indices.Count)
        {
            ConsoleLogger.WriteLine($"Warning: Surface index range [{startIndex}, {startIndex + indexCount}) exceeds scene indices ({scene.Indices.Count})");
            indexCount = scene.Indices.Count - startIndex;
        }
        
        // Build vertex mapping for this surface to avoid duplicate vertices
        var vertexMap = new Dictionary<int, int>();
        
        for (int i = 0; i < indexCount; i += 3)
        {
            int m = startIndex + i;
            if (m + 2 >= scene.Indices.Count) break; // Final bounds check
            
            int aIdx = scene.Indices[m];
            int bIdx = scene.Indices[m + 1];
            int cIdx = scene.Indices[m + 2];
            
            // Track max vertex index for debugging
            maxVertexIndexSeen = Math.Max(maxVertexIndexSeen, Math.Max(aIdx, Math.Max(bIdx, cIdx)));
            
            // Add vertices with plane projection, MPRL transformation, and proper local indexing
            int localA = GetOrAddVertexWithTransform(aIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, applicableTransforms, ref outOfBoundsCount);
            int localB = GetOrAddVertexWithTransform(bIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, applicableTransforms, ref outOfBoundsCount);
            int localC = GetOrAddVertexWithTransform(cIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, applicableTransforms, ref outOfBoundsCount);
            
            // Only add valid triangles - skip any triangles with invalid vertex indices
            if (localA >= 0 && localB >= 0 && localC >= 0)
            {
                faces.Add(localA);
                faces.Add(localB);
                faces.Add(localC);
            }
            else
            {
                // Log skipped triangles for debugging
                if (outOfBoundsCount <= 10) // Only log first few to avoid spam
                {
                    ConsoleLogger.WriteLine($"      Skipped triangle with invalid vertices: A={localA}, B={localB}, C={localC} (indices: {aIdx}, {bIdx}, {cIdx})");
                }
            }
        }
    }
    
    /// <summary>
    /// Gets or adds a vertex with plane projection to the local vertex pool, avoiding duplicates.
    /// </summary>
    private static int GetOrAddVertex(int vertexIndex, Pm4Scene scene, List<Vector3> vertices,
        Dictionary<int, int> vertexMap, bool usePlane, Vector3 normal, float surfaceHeight, ref int outOfBoundsCount)
    {
        // Check if vertex is already in the local pool
        if (vertexMap.TryGetValue(vertexIndex, out int existingLocalIndex))
        {
            return existingLocalIndex;
        }
        
        // Validate global vertex index
        if (vertexIndex < 0 || vertexIndex >= scene.Vertices.Count)
        {
            outOfBoundsCount++;
            if (outOfBoundsCount <= 10) // Only log first 10 to avoid spam
            {
                ConsoleLogger.WriteLine($"Warning: Invalid vertex index {vertexIndex} (max: {scene.Vertices.Count - 1})");
            }
            return -1; // Return invalid index to skip this vertex
        }
        
        // Get raw vertex and apply legacy coordinate transform (Y,X,Z)
        var rawVertex = scene.Vertices[vertexIndex];
        var transformedVertex = new Vector3(rawVertex.Y, rawVertex.X, rawVertex.Z);
        
        // Apply plane projection if needed (exact legacy logic)
        if (usePlane)
        {
            float d = Vector3.Dot(normal, transformedVertex) - surfaceHeight;
            transformedVertex -= normal * d;
        }
        
        // Add to local vertex pool and update mapping
        int localIndex = vertices.Count;
        vertices.Add(transformedVertex);
        vertexMap[vertexIndex] = localIndex;
        
        return localIndex;
    }
    
    /// <summary>
    /// Gets or adds a vertex with plane projection and MPRL transformation to the local vertex pool, avoiding duplicates.
    /// Applies MPRL placement transformations to correctly position geometry in world space.
    /// </summary>
    private static int GetOrAddVertexWithTransform(int vertexIndex, Pm4Scene scene, List<Vector3> vertices,
        Dictionary<int, int> vertexMap, bool usePlane, Vector3 normal, float surfaceHeight, 
        List<MprlMslkMapping> applicableTransforms, ref int outOfBoundsCount)
    {
        // Check if vertex is already in the local pool
        if (vertexMap.TryGetValue(vertexIndex, out int existingLocalIndex))
        {
            return existingLocalIndex;
        }
        
        // Validate global vertex index
        if (vertexIndex < 0 || vertexIndex >= scene.Vertices.Count)
        {
            outOfBoundsCount++;
            if (outOfBoundsCount <= 10) // Only log first 10 to avoid spam
            {
                ConsoleLogger.WriteLine($"Warning: Invalid vertex index {vertexIndex} (max: {scene.Vertices.Count - 1})");
            }
            return -1; // Return invalid index to skip this vertex
        }
        
        // Get raw vertex and apply legacy coordinate transform (Y,X,Z)
        var rawVertex = scene.Vertices[vertexIndex];
        var transformedVertex = new Vector3(rawVertex.Y, rawVertex.X, rawVertex.Z);
        
        // Check for zero coordinates before transformation and log them
        if (transformedVertex == Vector3.Zero)
        {
            if (outOfBoundsCount <= 5) // Only log first few to avoid spam
            {
                ConsoleLogger.WriteLine($"      WARNING: Raw vertex[{vertexIndex}] is (0,0,0) before transformation: {rawVertex.X}, {rawVertex.Y}, {rawVertex.Z}");
            }
        }
        
        // Apply plane projection if needed (exact legacy logic)
        if (usePlane)
        {
            float d = Vector3.Dot(normal, transformedVertex) - surfaceHeight;
            transformedVertex -= normal * d;
        }
        
        // Apply MPRL transformations if available
        if (applicableTransforms.Any())
        {
            // For now, apply the first available transformation
            // TODO: Implement proper hierarchical transformation logic
            var mprlTransform = applicableTransforms.First();
            
            // Apply MPRL position offset
            transformedVertex += mprlTransform.Position;
            
            // Log transformation for debugging
            if (vertices.Count < 5) // Only log first few vertices to avoid spam
            {
                ConsoleLogger.WriteLine($"      Applied MPRL transform: vertex[{vertexIndex}] offset by {mprlTransform.Position}");
            }
        }
        
        // Final check for zero coordinates after all transformations
        if (transformedVertex == Vector3.Zero)
        {
            if (outOfBoundsCount <= 5) // Only log first few to avoid spam
            {
                ConsoleLogger.WriteLine($"      WARNING: Final vertex[{vertexIndex}] is (0,0,0) after all transformations!");
            }
        }
        
        // Add to local vertex pool and update mapping
        int localIndex = vertices.Count;
        vertices.Add(transformedVertex);
        vertexMap[vertexIndex] = localIndex;
        
        return localIndex;
    }
    
    /// <summary>
    /// Determines if surface normal should be swapped (simplified legacy logic).
    /// </summary>
    private static bool ShouldSwapNormal(ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface)
    {
        // Simplified version - in legacy this checks additional surface properties
        // For now, use a basic heuristic based on surface flags
        return surface.FlagsOrUnknown_0x00 == 18;
    }
    
    /// <summary>
    /// Calculates the bounding center from a set of vertices.
    /// </summary>
    private static Vector3 CalculateBoundingCenter(List<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return Vector3.Zero;
        
        var sum = Vector3.Zero;
        foreach (var vertex in vertices)
        {
            sum += vertex;
        }
        
        return sum / vertices.Count;
    }
    
    /// <summary>
    /// Generates a descriptive name for an object based on its SurfaceGroupKey.
    /// </summary>
    private static string GetObjectTypeName(uint surfaceGroupKey)
    {
        // Use SurfaceGroupKey patterns to identify object types based on CSV analysis
        string baseName = surfaceGroupKey switch
        {
            3 => "Building_Object_Group_3",
            19 => "Building_Object_Group_19",
            _ => $"Object_Group_{surfaceGroupKey}"
        };
        
        return baseName;
    }
    
    /// <summary>
    /// Determines if a MSUR surface is linked to a specific MSLK entry based on index ranges.
    /// Uses MSLK.MspiFirstIndex and surface.FirstIndex to establish linkage.
    /// </summary>
    private static bool IsLinkedToMslk(
        ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface, 
        ParpToolbox.Formats.P4.Chunks.Common.MslkEntry mslkLink, 
        Pm4Scene scene)
    {
        // Check if surface index range overlaps with MSLK index range
        // MSLK.MspiFirstIndex points to the start of geometry indices for this object
        // MSUR.FirstIndex defines the start of this surface's indices
        
        if (mslkLink.MspiFirstIndex == -1) return false; // Container node, no geometry
        
        // Simple linkage check: surface belongs to MSLK if its index range is within MSLK bounds
        // This is a simplified approach - in a full implementation, we'd need more sophisticated
        // index range analysis based on the MSPI buffer structure
        
        int surfaceStart = (int)surface.MsviFirstIndex;
        int surfaceEnd = surfaceStart + (int)surface.IndexCount;
        int mslkStart = mslkLink.MspiFirstIndex;
        
        // For now, use a simple heuristic: surfaces with similar index ranges are likely linked
        // This will need refinement based on actual MSPI buffer analysis
        return Math.Abs(surfaceStart - mslkStart) < 1000; // Arbitrary threshold for proximity
    }
    
    /// <summary>
    /// Exports MSUR-based objects to individual OBJ files.
    /// </summary>
    public static void ExportMsurObjects(List<MsurObject> objects, Pm4Scene scene, string outputRoot)
    {
        var objDir = Path.Combine(outputRoot, "msur_objects");
        Directory.CreateDirectory(objDir);
        
        ConsoleLogger.WriteLine($"Exporting {objects.Count} MSUR-based objects to {objDir}");
        
        foreach (var obj in objects)
        {
            var objPath = Path.Combine(objDir, $"{obj.ObjectType}.obj");
            
            using var writer = new StreamWriter(objPath);
            writer.WriteLine($"# PM4 MSUR Object - SurfaceGroupKey: {obj.SurfaceGroupKey}");
            writer.WriteLine($"# Surface Count: {obj.SurfaceCount}");
            writer.WriteLine($"# Center: {obj.BoundingCenter}");
            writer.WriteLine($"# Triangles: {obj.Triangles.Count}");
            writer.WriteLine($"# Vertices: {obj.VertexCount}");
            writer.WriteLine();
            
            // Write vertices used by this object (with coordinate fix)
            var vertexMapping = new Dictionary<int, int>();
            var usedVertices = obj.Triangles
                .SelectMany(t => new[] { t.A, t.B, t.C })
                .Distinct()
                .OrderBy(i => i)
                .ToList();
            
            for (int i = 0; i < usedVertices.Count; i++)
            {
                int originalIndex = usedVertices[i];
                vertexMapping[originalIndex] = i + 1; // OBJ uses 1-based indexing
                
                // The PM4 adapter already provides unified vertex data in scene.Vertices
                Vector3 vertex;
                if (originalIndex < scene.Vertices.Count)
                {
                    vertex = scene.Vertices[originalIndex];
                }
                else
                {
                    ConsoleLogger.WriteLine($"Warning: Invalid vertex index {originalIndex}, max: {scene.Vertices.Count - 1}");
                    vertex = Vector3.Zero;
                }
                
                writer.WriteLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}"); // Fix X-axis
            }
            
            writer.WriteLine();
            writer.WriteLine($"g {obj.ObjectType}");
            
            // Write faces
            foreach (var (a, b, c) in obj.Triangles)
            {
                writer.WriteLine($"f {vertexMapping[a]} {vertexMapping[b]} {vertexMapping[c]}");
            }
            
            writer.WriteLine();
            
            ConsoleLogger.WriteLine($"    Exported object SurfaceGroupKey={obj.SurfaceGroupKey} to {Path.GetFileName(objPath)}");
        }
        
        ConsoleLogger.WriteLine($"Exported {objects.Count} MSUR-based objects");
    }
}
