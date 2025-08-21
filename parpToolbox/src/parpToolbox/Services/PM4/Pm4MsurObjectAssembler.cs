namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;


/// <summary>
/// Assembles PM4 objects using MSUR SurfaceGroupKey grouping with MPRL transformation support,
/// as documented in legacy notes and CSV analysis. Applies MPRL placement transformations
/// to correctly position geometry in world space.
/// </summary>
public static class Pm4MsurObjectAssembler
{
    /// <summary>
    /// Represents a complete building object assembled using MSUR SurfaceGroupKey grouping.
    /// </summary>
    public record MsurObject(
        byte SurfaceGroupKey,          // MSUR.SurfaceGroupKey - the actual object identifier
        int SurfaceCount,              // Number of surfaces in this object
        List<(int A, int B, int C)> Triangles, // All triangles for this object
        List<Vector3> Vertices,        // Consolidated vertex list for export
        Vector3 BoundingCenter,        // Geometry centroid (diagnostic)
        Vector3 PlacementCenter,       // Placement centroid (scene/world space)
        int PlacementCount,            // Number of unique placements aggregated
        int VertexCount,
        string ObjectType              // Descriptive name based on surface properties
    );

    /// <summary>
    /// MPRL to MSLK mapping entry for transformation application.
    /// </summary>
    private record MprlMslkMapping(
        int MprlIndex,
        uint PlacementIndexKey,
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
        return CoordinateTransformationService.ApplyPm4Transformation(new Vector3(entry.Position.X, entry.Position.Y, entry.Position.Z));
    }

    /// <summary>
    /// Builds MPRL to MSLK mapping based on cross-reference patterns discovered in CSV analysis.
    /// Maps MPRL.Unknown4 values to MSLK.ParentIndex values for transformation application.
    /// </summary>
    private static List<MprlMslkMapping> BuildMprlMslkMappings(Pm4Scene scene)
    {
        var mappings = new List<MprlMslkMapping>();

        ConsoleLogger.WriteLine($"Building MPRL.Unknown4 ↔ MSLK.ParentIndex mappings ({scene.Placements.Count} placements, {scene.Links.Count} links)...");

        // Index placements by Unknown4 (cast to uint for direct comparison with ParentIndex)
        var placementsByKey = scene.Placements
            .Select((p, i) => (Index: i, Entry: p))
            .GroupBy(t => (uint)t.Entry.Unknown4)
            .ToDictionary(g => g.Key, g => g.ToList());

        int linkWithMatches = 0;
        foreach (var link in scene.Links)
        {
            if (placementsByKey.TryGetValue(link.ParentIndex, out var plist))
            {
                linkWithMatches++;
                foreach (var t in plist)
                {
                    var pos = TransformMprlPosition(t.Entry);
                    mappings.Add(new MprlMslkMapping(
                        t.Index,
                        (uint)t.Entry.Unknown4,
                        link.ParentIndex,
                        pos,
                        t.Entry
                    ));
                }
            }
        }

        ConsoleLogger.WriteLine($"Built {mappings.Count} placement→link mappings across {linkWithMatches} links");
        return mappings;
    }

    /// <summary>
    /// Assembles PM4 objects by grouping geometry using MSUR SurfaceGroupKey with MPRL transformation support.
    /// Builds local vertex pools per object and correctly remaps face indices to avoid out-of-bounds errors.
    /// </summary>
    public static List<MsurObject> AssembleObjectsByMsurIndex(Pm4Scene scene)
    {
        var assembledObjects = new List<MsurObject>();
        
        ConsoleLogger.WriteLine($"Assembling PM4 building objects via MSLK.ParentIndex (mapped from MPRL.Unknown4) with placement transforms...");
        ConsoleLogger.WriteLine($"  MSUR surfaces: {scene.Surfaces.Count}");
        ConsoleLogger.WriteLine($"  MSLK links: {scene.Links.Count}");
        ConsoleLogger.WriteLine($"  MPRL placements: {scene.Placements.Count}");
        ConsoleLogger.WriteLine($"  Scene vertices: {scene.Vertices.Count} (indices 0-{scene.Vertices.Count - 1})");
        ConsoleLogger.WriteLine($"  Scene indices: {scene.Indices.Count}");
        
        // Build MPRL to MSLK mappings for transformation application
        var mprlMappings = BuildMprlMslkMappings(scene);
        
        // --- Build quick lookup: surface index ranges ----
        var surfaceRanges = scene.Surfaces.Select(s => new
        {
            Surface = s,
            Start = (int)s.MsviFirstIndex,
            End = (int)s.MsviFirstIndex + s.IndexCount - 1
        }).OrderBy(r => r.Start).ToArray();
        var surfaceStarts = surfaceRanges.Select(r => r.Start).ToArray();

        // --- Map MSLK geometry nodes to owning surfaces ---
        var linksBySurface = new Dictionary<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry, List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>>();
        foreach (var link in scene.Links.OfType<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>())
        {
            if (link.MspiFirstIndex < 0 || link.MspiIndexCount == 0)
                continue; // skip non-geometry nodes

            int nodeStart = link.MspiFirstIndex;
            int nodeEnd   = link.MspiFirstIndex + link.MspiIndexCount - 1;

            // Binary search to find candidate surface by start index
            int idx = Array.BinarySearch(surfaceStarts, nodeStart);
            if (idx < 0) idx = ~idx - 1; // previous range
            if (idx < 0) continue;

            var range = surfaceRanges[idx];
            if (nodeStart >= range.Start && nodeEnd <= range.End)
            {
                var owner = range.Surface;
                if (!linksBySurface.TryGetValue(owner, out var list))
                {
                    list = new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
                    linksBySurface[owner] = list;
                }
                list.Add(link);
            }
        }
        // Build reverse mapping: ParentIndex -> set of MSUR surfaces linked via MSLK nodes
        var surfacesByParent = new Dictionary<uint, HashSet<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>>();
        foreach (var kvp in linksBySurface)
        {
            var surface = kvp.Key;
            foreach (var link in kvp.Value)
            {
                if (!surfacesByParent.TryGetValue(link.ParentIndex, out var set))
                {
                    set = new HashSet<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
                    surfacesByParent[link.ParentIndex] = set;
                }
                set.Add(surface);
            }
        }

        ConsoleLogger.WriteLine($"  Found {surfacesByParent.Count} distinct MSLK.ParentIndex groups (buildings)");

        // Track vertex index usage for debugging
        int maxVertexIndexSeen = -1;
        int outOfBoundsCount = 0;

        foreach (var (parentIndex, surfaceSet) in surfacesByParent)
        {
            var consolidatedVertices = new List<Vector3>();
            var consolidatedFaces = new List<int>();

            // Placement transforms for this building (from MPRL mappings)
            var transforms = mprlMappings
                .Where(m => m.MslkParentIndex == parentIndex)
                .Select(m => m.Position)
                .ToList();
            if (transforms.Count == 0) transforms = new List<Vector3> { Vector3.Zero };

            ConsoleLogger.WriteLine($"    Building ParentIndex=0x{parentIndex:X8}: {surfaceSet.Count} surfaces, {transforms.Count} placements");

            foreach (var surface in surfaceSet)
            {
                if (surface.IndexCount <= 0) continue;
                ExtractSurfaceTrianglesWithTransforms(
                    surface,
                    scene,
                    transforms,
                    consolidatedVertices,
                    consolidatedFaces,
                    ref maxVertexIndexSeen,
                    ref outOfBoundsCount);
            }

            if (consolidatedVertices.Count == 0 || consolidatedFaces.Count == 0)
                continue;

            // Calculate centers
            var boundingCenter = CalculateBoundingCenter(consolidatedVertices);

            // Placement centroid/count (dedup transforms)
            Vector3 placementCenter = boundingCenter;
            int placementCount = 0;
            {
                var uniq = new Dictionary<string, Vector3>();
                foreach (var v in transforms)
                {
                    var key = string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0:F6},{1:F6},{2:F6}", v.X, v.Y, v.Z);
                    if (!uniq.ContainsKey(key)) uniq[key] = v;
                }
                placementCount = uniq.Count;
                if (placementCount > 0)
                {
                    var sum = Vector3.Zero;
                    foreach (var v in uniq.Values) sum += v;
                    placementCenter = sum / placementCount;
                }
            }

            // Convert face list to triangles
            var triangles = new List<(int A, int B, int C)>();
            for (int i = 0; i < consolidatedFaces.Count; i += 3)
            {
                if (i + 2 < consolidatedFaces.Count)
                {
                    triangles.Add((consolidatedFaces[i], consolidatedFaces[i + 1], consolidatedFaces[i + 2]));
                }
            }

            byte groupKey = 0; // Not meaningful across merged surfaces
            var msurObject = new MsurObject(
                SurfaceGroupKey: groupKey,
                SurfaceCount: surfaceSet.Count,
                Triangles: triangles,
                Vertices: consolidatedVertices,
                BoundingCenter: boundingCenter,
                PlacementCenter: placementCenter,
                PlacementCount: placementCount,
                VertexCount: consolidatedVertices.Count,
                ObjectType: $"Building_Parent_{parentIndex:X8}"
            );

            assembledObjects.Add(msurObject);

            ConsoleLogger.WriteLine($"      Built Building ParentIndex=0x{parentIndex:X8}: {triangles.Count} triangles, {consolidatedVertices.Count} vertices, {surfaceSet.Count} surfaces");
        }

        ConsoleLogger.WriteLine($"Assembled {assembledObjects.Count} building objects");
        
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
    /// Assembles PM4 objects by grouping geometry using MSUR.CompositeKey (32-bit key).
    /// Mirrors the flow of AssembleObjectsByMsurIndex, but swaps grouping key to CompositeKey.
    /// Applies MPRL placement transforms via MSLK link mapping for correct world positioning.
    /// </summary>
    public static List<MsurObject> AssembleObjectsByCompositeKey(Pm4Scene scene)
    {
        var assembledObjects = new List<MsurObject>();

        ConsoleLogger.WriteLine($"Assembling PM4 objects using CompositeKey grouping with MPRL transformations...");
        ConsoleLogger.WriteLine($"  MSUR surfaces: {scene.Surfaces.Count}");
        ConsoleLogger.WriteLine($"  MSLK links: {scene.Links.Count}");
        ConsoleLogger.WriteLine($"  MPRL placements: {scene.Placements.Count}");
        ConsoleLogger.WriteLine($"  Scene vertices: {scene.Vertices.Count} (indices 0-{scene.Vertices.Count - 1})");
        ConsoleLogger.WriteLine($"  Scene indices: {scene.Indices.Count}");

        // Build MPRL to MSLK mappings for transformation application
        var mprlMappings = BuildMprlMslkMappings(scene);

        // --- Build quick lookup: surface index ranges ----
        var surfaceRanges = scene.Surfaces.Select(s => new
        {
            Surface = s,
            Start = (int)s.MsviFirstIndex,
            End = (int)s.MsviFirstIndex + s.IndexCount - 1
        }).OrderBy(r => r.Start).ToArray();
        var surfaceStarts = surfaceRanges.Select(r => r.Start).ToArray();

        // --- Map MSLK geometry nodes to owning surfaces ---
        var linksBySurface = new Dictionary<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry, List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>>();
        foreach (var link in scene.Links.OfType<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>())
        {
            if (link.MspiFirstIndex < 0 || link.MspiIndexCount == 0)
                continue; // skip non-geometry nodes

            int nodeStart = link.MspiFirstIndex;
            int nodeEnd   = link.MspiFirstIndex + link.MspiIndexCount - 1;

            // Binary search to find candidate surface by start index
            int idx = Array.BinarySearch(surfaceStarts, nodeStart);
            if (idx < 0) idx = ~idx - 1; // previous range
            if (idx < 0) continue;

            var range = surfaceRanges[idx];
            if (nodeStart >= range.Start && nodeEnd <= range.End)
            {
                var owner = range.Surface;
                if (!linksBySurface.TryGetValue(owner, out var list))
                {
                    list = new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
                    linksBySurface[owner] = list;
                }
                list.Add(link);
            }
        }

        // Process each MSUR surface individually, then group by CompositeKey
        var objectsByComposite = new Dictionary<uint, List<(Vector3[] vertices, int[] faces)>>();
        var placementsByComposite = new Dictionary<uint, List<Vector3>>();

        int maxVertexIndexSeen = -1;
        int outOfBoundsCount = 0;

        ConsoleLogger.WriteLine($"  Processing {scene.Surfaces.Count} individual MSUR surfaces...");

        foreach (var surface in scene.Surfaces)
        {
            if (surface.IndexCount <= 0)
                continue;

            var surfaceVertices = new List<Vector3>();
            var surfaceFaces = new List<int>();

            // Build transform list for this surface based on linked MSLK parent indices
            List<Vector3> geomTransforms = new() { Vector3.Zero }; // default if no placements
            List<Vector3> placementTransforms = new();              // only real placements
            if (linksBySurface.TryGetValue(surface, out var linkedNodes))
            {
                var parentIds = linkedNodes.Select(l => l.ParentIndex).Distinct().ToHashSet();
                var matched = mprlMappings.Where(m => parentIds.Contains(m.MslkParentIndex)).ToList();
                if (matched.Count > 0)
                {
                    geomTransforms = matched.Select(m => m.Position).ToList();
                    placementTransforms = geomTransforms;
                }
            }

            ExtractSurfaceTrianglesWithTransforms(surface, scene, geomTransforms, surfaceVertices, surfaceFaces, ref maxVertexIndexSeen, ref outOfBoundsCount);

            if (surfaceVertices.Count > 0 && surfaceFaces.Count > 0)
            {
                uint compositeKey = surface.CompositeKey;
                if (!objectsByComposite.TryGetValue(compositeKey, out var surfaceList))
                {
                    surfaceList = new List<(Vector3[], int[])>();
                    objectsByComposite[compositeKey] = surfaceList;
                }
                surfaceList.Add((surfaceVertices.ToArray(), surfaceFaces.ToArray()));

                if (placementTransforms.Count > 0)
                {
                    if (!placementsByComposite.TryGetValue(compositeKey, out var plist))
                    {
                        plist = new List<Vector3>();
                        placementsByComposite[compositeKey] = plist;
                    }
                    plist.AddRange(placementTransforms);
                }
            }
        }

        ConsoleLogger.WriteLine($"  Found {objectsByComposite.Count} distinct CompositeKey groups (complete objects)");

        foreach (var (compositeKey, surfaceList) in objectsByComposite)
        {
            var consolidatedVertices = new List<Vector3>();
            var consolidatedFaces = new List<int>();

            ConsoleLogger.WriteLine($"    Processing CompositeKey 0x{compositeKey:X8} with {surfaceList.Count} surfaces");

            foreach (var (surfaceVertices, surfaceFaces) in surfaceList)
            {
                int vertexOffset = consolidatedVertices.Count;
                consolidatedVertices.AddRange(surfaceVertices);
                for (int i = 0; i < surfaceFaces.Length; i++)
                {
                    consolidatedFaces.Add(surfaceFaces[i] + vertexOffset);
                }
            }

            if (consolidatedVertices.Count > 0)
            {
                ConsoleLogger.WriteLine($"      Consolidated {consolidatedVertices.Count} vertices and {consolidatedFaces.Count} face indices for CompositeKey 0x{compositeKey:X8}");

                var boundingCenter = CalculateBoundingCenter(consolidatedVertices);

                Vector3 placementCenter = boundingCenter; // fallback
                int placementCount = 0;
                if (placementsByComposite.TryGetValue(compositeKey, out var pList) && pList.Count > 0)
                {
                    var uniq = new Dictionary<string, Vector3>();
                    foreach (var v in pList)
                    {
                        var key = string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0:F6},{1:F6},{2:F6}", v.X, v.Y, v.Z);
                        if (!uniq.ContainsKey(key)) uniq[key] = v;
                    }
                    placementCount = uniq.Count;
                    if (placementCount > 0)
                    {
                        var sum = Vector3.Zero;
                        foreach (var v in uniq.Values) sum += v;
                        placementCenter = sum / placementCount;
                    }
                }

                var triangles = new List<(int A, int B, int C)>();
                for (int i = 0; i < consolidatedFaces.Count; i += 3)
                {
                    if (i + 2 < consolidatedFaces.Count)
                    {
                        triangles.Add((consolidatedFaces[i], consolidatedFaces[i + 1], consolidatedFaces[i + 2]));
                    }
                }

                // groupKey not meaningful under composite grouping - set to 0
                byte groupKey = 0;
                var msurObject = new MsurObject(
                    SurfaceGroupKey: groupKey,
                    SurfaceCount: surfaceList.Count,
                    Triangles: triangles,
                    Vertices: consolidatedVertices,
                    BoundingCenter: boundingCenter,
                    PlacementCenter: placementCenter,
                    PlacementCount: placementCount,
                    VertexCount: consolidatedVertices.Count,
                    ObjectType: $"Composite_{compositeKey:X8}"
                );

                assembledObjects.Add(msurObject);

                ConsoleLogger.WriteLine($"      Object CompositeKey=0x{compositeKey:X8}: {triangles.Count} triangles, {consolidatedVertices.Count} vertices, {surfaceList.Count} surfaces");
                if (triangles.Count == 0)
                {
                    ConsoleLogger.WriteLine($"        WARNING: No triangles generated for CompositeKey 0x{compositeKey:X8}!");
                }
            }
        }

        ConsoleLogger.WriteLine($"Assembled {assembledObjects.Count} composite-key objects");

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
    /// Assembles PM4 objects by grouping geometry using the two-byte selector key:
    /// (SurfaceGroupKey &lt;&lt; 8) | SurfaceAttributeMask (aka XX/YY discovered in July-22 research).
    /// This yields much finer-grained, semantically correct pieces than IndexCount grouping.
    /// </summary>
    public static List<MsurObject> AssembleObjectsBySelectorKey(Pm4Scene scene)
    {
        var assembledObjects = new List<MsurObject>();
        var groups = new Dictionary<int, List<(List<Vector3> verts, List<int> faces)>>();

        int dummyMax = 0;
        int dummyOob = 0;

        foreach (var surface in scene.Surfaces)
        {
            if (surface.IndexCount == 0) continue;

            int selectorKey = (surface.SurfaceGroupKey << 8) | surface.SurfaceAttributeMask;

            var verts = new List<Vector3>();
            var faces = new List<int>();
            ExtractSurfaceTriangles(surface, scene, verts, faces, ref dummyMax, ref dummyOob);
            if (verts.Count == 0 || faces.Count == 0) continue;

            if (!groups.TryGetValue(selectorKey, out var list))
            {
                list = new List<(List<Vector3>, List<int>)>();
                groups[selectorKey] = list;
            }
            list.Add((verts, faces));
        }

        foreach (var (selectorKey, surfaceBatches) in groups)
        {
            var consolidatedVerts = new List<Vector3>();
            var consolidatedFaces = new List<int>();
            foreach (var (verts, faces) in surfaceBatches)
            {
                int offset = consolidatedVerts.Count;
                consolidatedVerts.AddRange(verts);
                for (int i = 0; i < faces.Count; i++)
                {
                    consolidatedFaces.Add(faces[i] + offset);
                }
            }
            if (consolidatedFaces.Count == 0) continue;
            var tris = new List<(int A,int B,int C)>();
            for (int i = 0; i < consolidatedFaces.Count; i += 3)
            {
                if (i + 2 >= consolidatedFaces.Count) break;
                tris.Add((consolidatedFaces[i], consolidatedFaces[i+1], consolidatedFaces[i+2]));
            }
            var center = CalculateBoundingCenter(consolidatedVerts);
            var obj = new MsurObject(
                SurfaceGroupKey: (byte)(selectorKey >> 8),
                SurfaceCount: surfaceBatches.Count,
                Triangles: tris,
                Vertices: consolidatedVerts,
                BoundingCenter: center,
                PlacementCenter: center,
                PlacementCount: 0,
                VertexCount: consolidatedVerts.Count,
                ObjectType: $"Selector_{selectorKey:X4}");
            assembledObjects.Add(obj);
        }

        ConsoleLogger.WriteLine($"Assembled {assembledObjects.Count} selector‐key objects (XX/YY grouping)");
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
    // Extract triangles applying one or more translation transforms (replicated per transform)
    private static void ExtractSurfaceTrianglesWithTransforms(
        ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry surface,
        Pm4Scene scene,
        List<Vector3> transforms,
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
        
        // Apply each provided transform (often one, sometimes many)
        var applicableTransforms = transforms.Count > 0 ? transforms : new List<Vector3>{Vector3.Zero};
        if (applicableTransforms.Count > 1)
        {
            ConsoleLogger.WriteLine($"    Surface SurfaceGroupKey={surface.SurfaceGroupKey}: {applicableTransforms.Count} placement transforms");
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

        foreach (var offset in applicableTransforms)
        {
            // Build vertex mapping per transform to replicate geometry for each placement
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
                int localA = GetOrAddVertexWithTransform(aIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, offset, ref outOfBoundsCount);
                int localB = GetOrAddVertexWithTransform(bIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, offset, ref outOfBoundsCount);
                int localC = GetOrAddVertexWithTransform(cIdx, scene, vertices, vertexMap, usePlane, finalNormal, surface.Height, offset, ref outOfBoundsCount);

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
        var transformedVertex = CoordinateTransformationService.ApplyPm4Transformation(rawVertex);
        
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
        Vector3 offset, ref int outOfBoundsCount)
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
        
        // Get raw vertex and apply unified coordinate transformation
        var rawVertex = scene.Vertices[vertexIndex];
        var transformedVertex = CoordinateTransformationService.ApplyPm4Transformation(rawVertex);
        
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
        
        // Apply the provided placement offset (per-instance replication)
        if (offset != Vector3.Zero && vertices.Count < 5) // Only log first few vertices to avoid spam
        {
            ConsoleLogger.WriteLine($"      Applied MPRL transform: vertex[{vertexIndex}] offset by {offset}");
        }
        transformedVertex += offset;
        
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
        return surface.GroupKey == 18;
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
            
            // Write vertices directly from the assembled object (already transformed and placed)
            for (int i = 0; i < obj.Vertices.Count; i++)
            {
                var v = obj.Vertices[i];
                writer.WriteLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");
            }
            
            writer.WriteLine();
            writer.WriteLine($"g {obj.ObjectType}");
            
            // Write faces
            foreach (var (a, b, c) in obj.Triangles)
            {
                // OBJ uses 1-based indexing
                writer.WriteLine($"f {a + 1} {b + 1} {c + 1}");
            }
            
            writer.WriteLine();
            
            ConsoleLogger.WriteLine($"    Exported object SurfaceGroupKey={obj.SurfaceGroupKey} to {Path.GetFileName(objPath)}");
        }
        
        ConsoleLogger.WriteLine($"Exported {objects.Count} MSUR-based objects");
    }
}
