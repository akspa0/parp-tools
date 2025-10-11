using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder;

/// <summary>
/// PM4-Native Assembly Engine that uses PM4's built-in linkage system to assemble complete buildings.
/// This follows the critical discovery that PM4 contains all assembly instructions natively.
/// No spatial clustering heuristics needed - uses MSLK/MPRL/MSUR relationships as designed.
/// </summary>
public class PM4NativeAssembler
{
    private readonly PM4UnifiedMap _unifiedMap;
    private readonly IEnumerable<MslkEntry> _linkSource;
    private readonly IEnumerable<MprlChunk.Entry> _placementSource;

    private readonly Dictionary<uint, List<MslkEntry>> _linksByParent = new();
    private bool _diagnosticLogged;
    private readonly Dictionary<uint, MsurChunk.Entry> _surfacesByKey = new();
    private readonly Dictionary<uint, MprlChunk.Entry> _placementsByObjectId = new();

    /// <summary>
    /// Create assembler for the whole unified map (legacy behaviour).
    /// </summary>
    public PM4NativeAssembler(PM4UnifiedMap unifiedMap)
        : this(unifiedMap, unifiedMap.AllMslkLinks, unifiedMap.AllMprlPlacements)
    {
    }

    /// <summary>
    /// Create assembler that works on a subset of links/placements (e.g., a single tile)
    /// while still using the global vertex/surface pools from the unified map.
    /// </summary>
    public PM4NativeAssembler(PM4UnifiedMap unifiedMap,
        IEnumerable<MslkEntry> linkSubset,
        IEnumerable<MprlChunk.Entry> placementSubset)
    {
        _unifiedMap = unifiedMap ?? throw new ArgumentNullException(nameof(unifiedMap));
        _linkSource = linkSubset ?? throw new ArgumentNullException(nameof(linkSubset));
        _placementSource = placementSubset ?? throw new ArgumentNullException(nameof(placementSubset));

        BuildLookupTables();
    }

    /// <summary>
    /// Assemble complete buildings using PM4's native linkage system.
    /// This is the correct approach - no spatial clustering, just follow PM4 data structures.
    /// </summary>
    public List<PM4Building> AssembleBuildings()
    {
        Console.WriteLine("[PM4 NATIVE ASSEMBLER] Starting building assembly using PM4 native linkage system...");

        var buildings = new List<PM4Building>();

        // Step 1: Group MSLK links by ParentIndex to identify building objects
        Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Found {_linksByParent.Count} unique parent objects");

        // Step 2: For each parent object, assemble its complete building geometry
        foreach (var kvp in _linksByParent.OrderBy(p => p.Key))
        {
            var parentId = kvp.Key;
            var childLinks = kvp.Value;

            try
            {
                var building = AssembleBuildingFromPM4Data(parentId, childLinks);
                if (building != null && building.Vertices.Count > 0)
                {
                    buildings.Add(building);
                    Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Building {parentId}: {building.Vertices.Count} vertices, {building.TriangleCount} triangles");
                }
                else
                {
                    Console.WriteLine($"[PM4 NATIVE ASSEMBLER WARNING] Building {parentId}: No geometry assembled");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PM4 NATIVE ASSEMBLER ERROR] Failed to assemble building {parentId}: {ex.Message}");
            }
        }

        Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Successfully assembled {buildings.Count} buildings");
        
        // Validate building scale (should be 38K-654K triangles per building from memory bank)
        ValidateBuildingScale(buildings);

        return buildings;
    }

    /// <summary>
    /// Assemble a single building using PM4's native data structures.
    /// Follows MSLK.ParentIndex → geometry collection → MSUR surface assembly.
    /// </summary>
    private PM4Building? AssembleBuildingFromPM4Data(uint parentId, List<MslkEntry> childLinks)
    {
        var building = new PM4Building
        {
            BuildingId = parentId,
            SourceLinks = childLinks,
            Vertices = new List<Vector3>(),
            Indices = new List<uint>(),
            SourceSurfaces = new List<MsurChunk.Entry>()
        };

        // Get placement info for this building (position, etc.)
        if (_placementsByObjectId.TryGetValue(parentId, out var placement))
        {
            building.Position = ExtractPositionFromPlacement(placement);
            Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Building {parentId} position: {building.Position}");
        }

        // Step 1: Collect all geometry from child MSLK links
        var vertexStartIndex = 0;
        foreach (var link in childLinks)
        {
            try
            {
                // Skip container nodes (no geometry)
                if (IsContainerNode(link))
                {
                    Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Building {parentId}: Skipping container node");
                    continue;
                }

                // Follow MSLK.SurfaceRefIndex → MSUR.SurfaceKey relationship
                if (TryGetSurfaceForLink(link, out var surface))
                {
                    building.SourceSurfaces.Add(surface);
                    
                    // Extract geometry for this surface
                    var surfaceGeometry = ExtractGeometryFromSurface(surface, link);
                    if (surfaceGeometry.vertices.Count > 0)
                    {
                        // Add vertices with proper indexing
                        building.Vertices.AddRange(surfaceGeometry.vertices);
                        
                        // Add indices with vertex offset correction
                        foreach (var index in surfaceGeometry.indices)
                        {
                            building.Indices.Add((uint)(vertexStartIndex + index));
                        }
                        
                        vertexStartIndex += surfaceGeometry.vertices.Count;
                        
                        Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Building {parentId}: Added surface with {surfaceGeometry.vertices.Count} vertices");
                    }
                }
                else
                {
                    Console.WriteLine($"[PM4 NATIVE ASSEMBLER WARNING] Building {parentId}: No surface found for link");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PM4 NATIVE ASSEMBLER ERROR] Building {parentId}: Failed to process link: {ex.Message}");
            }
        }

        // Calculate bounding box
        if (building.Vertices.Count > 0)
        {
            building.Bounds = CalculateBoundingBox(building.Vertices);
            Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Building {parentId}: Bounds {building.Bounds}");
        }

        return building.Vertices.Count > 0 ? building : null;
    }

    /// <summary>
    /// Build lookup tables for efficient PM4 linkage resolution.
    /// </summary>
    private void BuildLookupTables()
    {
        Console.WriteLine("[PM4 NATIVE ASSEMBLER] Building PM4 linkage lookup tables...");

        // One-time diagnostic dump of property names/values to identify correct fields
        if (!_diagnosticLogged)
        {
            var firstLink = _linkSource.FirstOrDefault();
            if (firstLink != null)
            {
                Console.WriteLine("[DEBUG] MSLK Entry properties: " + string.Join(", ", firstLink.GetType().GetProperties().Select(p => p.Name)));
                foreach (var p in firstLink.GetType().GetProperties())
                {
                    try { Console.WriteLine($"[DEBUG]   {p.Name} = {p.GetValue(firstLink)}"); } catch { }
                }
            }
            var firstPlacement = _placementSource.FirstOrDefault();
            if (firstPlacement != null)
            {
                Console.WriteLine("[DEBUG] MPRL Placement properties: " + string.Join(", ", firstPlacement.GetType().GetProperties().Select(p => p.Name)));
                foreach (var p in firstPlacement.GetType().GetProperties())
                {
                    try { Console.WriteLine($"[DEBUG]   {p.Name} = {p.GetValue(firstPlacement)}"); } catch { }
                }
            }
            _diagnosticLogged = true;
        }

        // Group MSLK links by ParentIndex for building identification
        foreach (var link in _linkSource)
        {
            var parentIndex = GetParentIndex(link);
            if (parentIndex.HasValue)
            {
                if (!_linksByParent.ContainsKey(parentIndex.Value))
                {
                    _linksByParent[parentIndex.Value] = new List<MslkEntry>();
                }
                _linksByParent[parentIndex.Value].Add(link);
            }
        }

        // Index MSUR surfaces by their global list index so that MSLK.SurfaceRefIndex
        // (which we adjusted to global indices during map aggregation) resolves directly.
        for (int i = 0; i < _unifiedMap.AllMsurSurfaces.Count; i++)
        {
            _surfacesByKey[(uint)i] = _unifiedMap.AllMsurSurfaces[i];
        }

        // Index MPRL placements by object ID (subset)
        foreach (var placement in _placementSource)
        {
            var objectId = GetObjectId(placement);
            if (objectId.HasValue)
            {
                _placementsByObjectId[objectId.Value] = placement;
            }
        }

        Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Built lookup tables:");
        Console.WriteLine($"  - Links by parent: {_linksByParent.Count} parent objects");
        Console.WriteLine($"  - Surfaces by key: {_surfacesByKey.Count} surfaces");
        Console.WriteLine($"  - Placements by object ID: {_placementsByObjectId.Count} placements");
    }

    /// <summary>
    /// Try to get surface for a MSLK link using the SurfaceRefIndex → SurfaceKey relationship.
    /// </summary>
    private bool TryGetSurfaceForLink(MslkEntry link, out MsurChunk.Entry surface)
    {
        surface = default;
        
        var surfaceRefIndex = GetSurfaceRefIndex(link);
        if (!surfaceRefIndex.HasValue)
            return false;

        return _surfacesByKey.TryGetValue(surfaceRefIndex.Value, out surface);
    }

    /// <summary>
    /// Extract geometry from a surface entry using the unified vertex/index pools.
    /// This resolves cross-tile references properly.
    /// </summary>
    private (List<Vector3> vertices, List<uint> indices) ExtractGeometryFromSurface(MsurChunk.Entry surface, MslkEntry link)
    {
        var vertices = new List<Vector3>();
        var indices = new List<uint>();

        try
        {
            // Get surface geometry parameters
            var indexCount = GetIndexCount(surface);
            var firstIndex = GetFirstIndex(surface);
            
            if (!indexCount.HasValue || !firstIndex.HasValue)
            {
                Console.WriteLine("[PM4 NATIVE ASSEMBLER WARNING] Surface missing geometry parameters");
                return (vertices, indices);
            }

            // Determine which vertex pool to use (MSVT vs MSPV)
            bool useMSVTPool = ShouldUseMSVTPool(surface);
            var globalVertices = useMSVTPool ? _unifiedMap.GlobalMSVTVertices : _unifiedMap.GlobalMSPVVertices;
            var globalIndices = useMSVTPool ? _unifiedMap.GlobalMSVIIndices : _unifiedMap.GlobalMSPIIndices;

            // Extract triangle indices for this surface
            var surfaceIndices = new List<uint>();
            for (int i = 0; i < indexCount.Value && (firstIndex.Value + i) < globalIndices.Count; i++)
            {
                var globalIndex = globalIndices[(int)(firstIndex.Value + i)];
                surfaceIndices.Add(globalIndex);
            }

            // Build vertex list with proper cross-tile resolution
            var usedVertexIndices = surfaceIndices.Distinct().OrderBy(i => i).ToList();
            var vertexIndexMap = new Dictionary<uint, uint>();
            
            foreach (var globalVertexIndex in usedVertexIndices)
            {
                if (globalVertexIndex < globalVertices.Count)
                {
                    vertexIndexMap[globalVertexIndex] = (uint)vertices.Count;
                    vertices.Add(globalVertices[(int)globalVertexIndex]);
                }
                else
                {
                    Console.WriteLine($"[PM4 NATIVE ASSEMBLER WARNING] Invalid vertex index {globalVertexIndex} (max: {globalVertices.Count})");
                }
            }

            // Remap triangle indices to local vertex list
            for (int i = 0; i < surfaceIndices.Count; i++)
            {
                if (vertexIndexMap.TryGetValue(surfaceIndices[i], out var localIndex))
                {
                    indices.Add(localIndex);
                }
            }

            Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Surface geometry: {vertices.Count} vertices, {indices.Count / 3} triangles");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[PM4 NATIVE ASSEMBLER ERROR] Failed to extract surface geometry: {ex.Message}");
        }

        return (vertices, indices);
    }

    /// <summary>
    /// Validate building scale against expected values from memory bank.
    /// Buildings should have 38K-654K triangles, not fragments.
    /// </summary>
    private void ValidateBuildingScale(List<PM4Building> buildings)
    {
        if (buildings.Count == 0) return;

        var triangleCounts = buildings.Select(b => b.TriangleCount).ToList();
        var minTriangles = triangleCounts.Min();
        var maxTriangles = triangleCounts.Max();
        var avgTriangles = (int)triangleCounts.Average();

        Console.WriteLine($"[PM4 NATIVE ASSEMBLER] Building scale validation:");
        Console.WriteLine($"  - Building count: {buildings.Count}");
        Console.WriteLine($"  - Triangle range: {minTriangles:N0} - {maxTriangles:N0} (avg: {avgTriangles:N0})");

        // Check against expected values from memory bank
        var expectedMinTriangles = 38000;  // 38K from memory
        var expectedMaxTriangles = 654000; // 654K from memory
        var expectedBuildingCount = 458;   // From memory bank analysis

        if (buildings.Count < expectedBuildingCount / 2)
        {
            Console.WriteLine($"[PM4 NATIVE ASSEMBLER WARNING] Building count {buildings.Count} is much lower than expected ~{expectedBuildingCount}");
        }

        if (maxTriangles < expectedMinTriangles)
        {
            Console.WriteLine($"[PM4 NATIVE ASSEMBLER WARNING] Max triangle count {maxTriangles:N0} is below expected minimum {expectedMinTriangles:N0}");
            Console.WriteLine("  This suggests buildings are still fragmented instead of properly assembled");
        }

        if (avgTriangles >= expectedMinTriangles)
        {
            Console.WriteLine($"[PM4 NATIVE ASSEMBLER SUCCESS] Average triangle count {avgTriangles:N0} indicates building-scale objects!");
        }
    }

    // Helper methods for dynamic property access (since we're working with chunk entries)
    // These use reflection or known property patterns to extract values

    private uint? GetParentIndex(MslkEntry link)
    {
        // Use reflection or known property to get ParentIndex
        // This would need to access the specific field from the link entry
        // For now, using a placeholder - this needs actual implementation
        try
        {
            var type = link.GetType();
        var prop = type.GetProperty("ParentIndex", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase) ??
                   type.GetProperty("ParentIndex_0x04", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase) ??
                   type.GetProperty("ParentIndex0x04", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
            if (prop == null) return null;
            var val = prop.GetValue(link);
            return val switch
            {
                uint u => u,
                int i when i >= 0 => (uint)i,
                ushort us => us,
                short s when s >= 0 => (uint)s,
                _ => null
            };
        }
        catch
        {
            return null;
        }
    }

    private uint? GetSurfaceRefIndex(MslkEntry link)
    {
        try
        {
            var prop = link.GetType().GetProperty("SurfaceRefIndex");
            if (prop == null) return null;
            var value = prop.GetValue(link);
            switch (value)
            {
                case uint u:
                    return u;
                case int i when i >= 0:
                    return (uint)i;
                case ushort us:
                    return us;
                case short s when s >= 0:
                    return (uint)s;
                default:
                    return null;
            }
        }
        catch
        {
            return null;
        }
    }

    private uint? GetSurfaceKey(MsurChunk.Entry surface)
    {
        try
        {
            var prop = surface.GetType().GetProperty("SurfaceKey");
            return prop?.GetValue(surface) as uint?;
        }
        catch
        {
            return null;
        }
    }

    private uint? GetObjectId(MprlChunk.Entry placement)
    {
        try
        {
            var type = placement.GetType();
            var prop = type.GetProperty("Unknown4", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase) ??
                       type.GetProperty("ObjectId", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase) ??
                       type.GetProperty("ParentIndex", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
            if (prop == null) return null;
            var val = prop.GetValue(placement);
            return val switch
            {
                uint u => u,
                int i when i >= 0 => (uint)i,
                ushort us => us,
                short s when s >= 0 => (uint)s,
                _ => null
            };
        }
        catch
        {
            return null;
        }
    }

    private bool _msurDiagnosticLogged;

    private uint? GetIndexCount(MsurChunk.Entry surface)
    {
        try
        {
            var type = surface.GetType();
            var prop = type.GetProperty("IndexCount", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase) ??
                       type.GetProperty("MspiIndexCount", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
            if (prop == null) 
            {
                if (!_msurDiagnosticLogged)
                {
                    Console.WriteLine("[DEBUG] MSUR Surface properties (IndexCount missing): " + string.Join(", ", type.GetProperties().Select(p => p.Name)));
                    foreach (var p in type.GetProperties())
                    {
                        try { Console.WriteLine($"[DEBUG]   {p.Name} = {p.GetValue(surface)}"); } catch { }
                    }
                    _msurDiagnosticLogged = true;
                }
                return null;
            }
            var val = prop.GetValue(surface);
            return val switch
            {
                uint u => u,
                int i when i >= 0 => (uint)i,
                ushort us => us,
                short s when s >= 0 => (uint)s,
                _ => null
            };
        }
        catch
        {
            return null;
        }
    }

    private uint? GetFirstIndex(MsurChunk.Entry surface)
    {
        try
        {
            var type = surface.GetType();
            var prop = type.GetProperty("FirstIndex", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase) ??
                       type.GetProperty("MsviFirstIndex", BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
            if (prop == null) 
            {
                if (!_msurDiagnosticLogged)
                {
                    Console.WriteLine("[DEBUG] MSUR Surface properties (FirstIndex missing): " + string.Join(", ", type.GetProperties().Select(p => p.Name)));
                    foreach (var p in type.GetProperties())
                    {
                        try { Console.WriteLine($"[DEBUG]   {p.Name} = {p.GetValue(surface)}"); } catch { }
                    }
                    _msurDiagnosticLogged = true;
                }
                return null;
            }
            var val = prop.GetValue(surface);
            return val switch
            {
                uint u => u,
                int i when i >= 0 => (uint)i,
                ushort us => us,
                short s when s >= 0 => (uint)s,
                _ => null
            };
        }
        catch
        {
            return null;
        }
    }

    private bool IsContainerNode(MslkEntry link)
    {
        // Container nodes typically have MspiFirstIndex = -1
        try
        {
            var prop = link.GetType().GetProperty("MspiFirstIndex");
            var value = prop?.GetValue(link);
            return value is int intValue && intValue == -1;
        }
        catch
        {
            return false;
        }
    }

    private bool ShouldUseMSVTPool(MsurChunk.Entry surface)
    {
        // Determine whether this surface uses MSVT or MSPV vertex pool
        // This would need surface flags or other indicators
        // For now, default to MSVT (most common)
        return true;
    }

    private Vector3 ExtractPositionFromPlacement(MprlChunk.Entry placement)
    {
        try
        {
            var xProp = placement.GetType().GetProperty("X") ?? placement.GetType().GetProperty("PositionX");
            var yProp = placement.GetType().GetProperty("Y") ?? placement.GetType().GetProperty("PositionY");
            var zProp = placement.GetType().GetProperty("Z") ?? placement.GetType().GetProperty("PositionZ");

            var x = (float?)(xProp?.GetValue(placement)) ?? 0f;
            var y = (float?)(yProp?.GetValue(placement)) ?? 0f;
            var z = (float?)(zProp?.GetValue(placement)) ?? 0f;

            return new Vector3(x, y, z);
        }
        catch
        {
            return Vector3.Zero;
        }
    }

    private BoundingBox3D CalculateBoundingBox(List<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return new BoundingBox3D();

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

        return new BoundingBox3D
        {
            Min = min,
            Max = max,
            Size = max - min
        };
    }
}
