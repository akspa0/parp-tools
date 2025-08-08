using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder;

/// <summary>
/// Unified PM4 Exporter that implements the complete unified map architecture.
/// This is the main entry point for the new PM4 export pipeline that:
/// 1. Loads all PM4 files as unified map
/// 2. Uses PM4 native assembly for building extraction  
/// 3. Exports complete buildings (not fragments)
/// </summary>
public static class UnifiedPM4Exporter
{
    /// <summary>
    /// Export complete buildings from a PM4 directory using the unified map architecture.
    /// This is the correct approach that resolves cross-tile references and produces building-scale objects.
    /// </summary>
    /// <param name="pm4Directory">Directory containing PM4 files</param>
    /// <param name="outputDirectory">Output directory for building OBJ files</param>
    /// <param name="exportStrategy">Export strategy (per-building, per-tile, or unified)</param>
    /// <returns>Export summary with statistics and validation results</returns>
    public static async Task<PM4ExportSummary> ExportBuildingsAsync(
        string pm4Directory, 
        string outputDirectory, 
        PM4ExportStrategy exportStrategy = PM4ExportStrategy.PerBuilding)
    {
        var startTime = DateTime.UtcNow;
        var summary = new PM4ExportSummary
        {
            StartTime = startTime,
            InputDirectory = pm4Directory,
            OutputDirectory = outputDirectory,
            ExportStrategy = exportStrategy
        };

        try
        {
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("PM4 UNIFIED ARCHITECTURE EXPORT");
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine($"Input: {pm4Directory}");
            Console.WriteLine($"Output: {outputDirectory}");
            Console.WriteLine($"Strategy: {exportStrategy}");
            Console.WriteLine();

            // Phase 1: Load all PM4 files as unified map
            Console.WriteLine("Phase 1: Loading PM4 Unified Map...");
            var mapLoader = new PM4MapLoader();
            var unifiedMap = await mapLoader.LoadRegionAsync(pm4Directory);
            
            summary.TileCount = unifiedMap.TileData.Count;
            summary.TotalVertices = unifiedMap.TotalVertexCount;
            summary.TotalIndices = unifiedMap.TotalIndexCount;
            summary.TotalLinkageEntries = unifiedMap.TotalLinkageCount;

            var mapSummary = unifiedMap.GetSummary();
            Console.WriteLine($"Unified Map Loaded: {mapSummary}");
            Console.WriteLine();

            // Phase 2: PM4 Building Assembly using working direct-export logic...
            Console.WriteLine("Phase 2: PM4 Building Assembly using proven direct-export approach...");
            var buildings = AssembleBuildingsFromUnifiedMap(unifiedMap);

            summary.BuildingCount = buildings.Count;
            summary.Buildings = buildings;

            if (buildings.Count == 0)
            {
                Console.WriteLine("❌ NO BUILDINGS ASSEMBLED - This indicates a problem with the PM4 linkage system");
                summary.Success = false;
                summary.ErrorMessage = "No buildings could be assembled from PM4 data";
                return summary;
            }

            // Validate building scale (should be 38K-654K triangles from memory bank)
            ValidateBuildingScale(buildings, summary);
            Console.WriteLine();

            // Phase 3: Export buildings using selected strategy
            Console.WriteLine($"Phase 3: Exporting Buildings ({exportStrategy})...");
            Directory.CreateDirectory(outputDirectory);

            switch (exportStrategy)
            {
                case PM4ExportStrategy.PerBuilding:
                    await ExportPerBuildingAsync(buildings, outputDirectory, summary);
                    break;
                    
                case PM4ExportStrategy.PerTile:
                    await ExportPerTileAsync(buildings, unifiedMap, outputDirectory, summary);
                    break;
                    
                case PM4ExportStrategy.Unified:
                    await ExportUnifiedAsync(buildings, outputDirectory, summary);
                    break;
            }

            // Phase 4: Generate export report
            await GenerateExportReportAsync(summary, outputDirectory);

            summary.Success = true;
            summary.EndTime = DateTime.UtcNow;
            summary.TotalDuration = summary.EndTime - summary.StartTime;

            Console.WriteLine();
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine("EXPORT COMPLETE");
            Console.WriteLine("=".PadRight(80, '='));
            Console.WriteLine($"Duration: {summary.TotalDuration.TotalSeconds:F1}s");
            Console.WriteLine($"Buildings: {summary.BuildingCount}");
            Console.WriteLine($"Files: {summary.ExportedFileCount}");
            Console.WriteLine($"Success: {summary.Success}");
            Console.WriteLine();

            return summary;
        }
        catch (Exception ex)
        {
            summary.Success = false;
            summary.ErrorMessage = ex.Message;
            summary.EndTime = DateTime.UtcNow;

            Console.WriteLine($"❌ EXPORT FAILED: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");

            return summary;
        }
    }

    /// <summary>
    /// Export each building as a separate OBJ file.
    /// This is the recommended strategy for building-level analysis.
    /// </summary>
    private static async Task ExportPerBuildingAsync(List<PM4Building> buildings, string outputDirectory, PM4ExportSummary summary)
    {
        var buildingDir = Path.Combine(outputDirectory, "buildings");
        Directory.CreateDirectory(buildingDir);

        foreach (var building in buildings)
        {
            try
            {
                var objFileName = $"Building_{building.BuildingId:D6}.obj";
                var objPath = Path.Combine(buildingDir, objFileName);

                var objContent = building.ToObjString($"Building_{building.BuildingId}");
                await File.WriteAllTextAsync(objPath, objContent);

                summary.ExportedFileCount++;
                Console.WriteLine($"✅ Exported {objFileName}: {building.VertexCount} vertices, {building.TriangleCount} triangles");

                // Validate building
                var issues = building.ValidateGeometry();
                if (issues.Any())
                {
                    summary.ValidationIssues.AddRange(issues.Select(issue => $"Building {building.BuildingId}: {issue}"));
                    Console.WriteLine($"⚠️  Building {building.BuildingId} has {issues.Count} validation issues");
                }
            }
            catch (Exception ex)
            {
                var error = $"Failed to export building {building.BuildingId}: {ex.Message}";
                summary.ExportErrors.Add(error);
                Console.WriteLine($"❌ {error}");
            }
        }

        Console.WriteLine($"Per-building export complete: {summary.ExportedFileCount} files in {buildingDir}");
    }

    /// <summary>
    /// Export buildings grouped by tile boundaries.
    /// </summary>
    private static async Task ExportPerTileAsync(List<PM4Building> buildings, PM4UnifiedMap unifiedMap, string outputDirectory, PM4ExportSummary summary)
    {
        var tileDir = Path.Combine(outputDirectory, "tiles");
        Directory.CreateDirectory(tileDir);

        // Group buildings by tile (using position to determine tile)
        var buildingsByTile = GroupBuildingsByTile(buildings, unifiedMap);

        foreach (var kvp in buildingsByTile)
        {
            var (tileX, tileY) = kvp.Key;
            var tileBuildings = kvp.Value;

            try
            {
                var objFileName = $"Tile_{tileX:D2}_{tileY:D2}.obj";
                var objPath = Path.Combine(tileDir, objFileName);

                var objContent = BuildTileObjContent(tileBuildings, tileX, tileY);
                await File.WriteAllTextAsync(objPath, objContent);

                summary.ExportedFileCount++;
                var totalVertices = tileBuildings.Sum(b => b.VertexCount);
                var totalTriangles = tileBuildings.Sum(b => b.TriangleCount);
                Console.WriteLine($"✅ Exported {objFileName}: {tileBuildings.Count} buildings, {totalVertices} vertices, {totalTriangles} triangles");
            }
            catch (Exception ex)
            {
                var error = $"Failed to export tile ({tileX},{tileY}): {ex.Message}";
                summary.ExportErrors.Add(error);
                Console.WriteLine($"❌ {error}");
            }
        }

        Console.WriteLine($"Per-tile export complete: {summary.ExportedFileCount} files in {tileDir}");
    }

    /// <summary>
    /// Export all buildings as a single unified OBJ file.
    /// </summary>
    private static async Task ExportUnifiedAsync(List<PM4Building> buildings, string outputDirectory, PM4ExportSummary summary)
    {
        try
        {
            var objPath = Path.Combine(outputDirectory, "PM4_Unified_Scene.obj");
            var objContent = BuildUnifiedObjContent(buildings);
            await File.WriteAllTextAsync(objPath, objContent);

            summary.ExportedFileCount = 1;
            var totalVertices = buildings.Sum(b => b.VertexCount);
            var totalTriangles = buildings.Sum(b => b.TriangleCount);
            Console.WriteLine($"✅ Exported unified scene: {buildings.Count} buildings, {totalVertices} vertices, {totalTriangles} triangles");
        }
        catch (Exception ex)
        {
            var error = $"Failed to export unified scene: {ex.Message}";
            summary.ExportErrors.Add(error);
            Console.WriteLine($"❌ {error}");
        }
    }

    /// <summary>
    /// Validate building scale against expected values from memory bank.
    /// </summary>
    private static void ValidateBuildingScale(List<PM4Building> buildings, PM4ExportSummary summary)
    {
        if (buildings.Count == 0) return;

        var triangleCounts = buildings.Select(b => b.TriangleCount).ToList();
        summary.MinTrianglesPerBuilding = triangleCounts.Min();
        summary.MaxTrianglesPerBuilding = triangleCounts.Max();
        summary.AvgTrianglesPerBuilding = (int)triangleCounts.Average();

        // Expected values from memory bank
        var expectedMinTriangles = 38000;  // 38K
        var expectedMaxTriangles = 654000; // 654K
        var expectedBuildingCount = 458;   // Expected buildings

        Console.WriteLine($"Building Scale Validation:");
        Console.WriteLine($"  Building Count: {buildings.Count} (expected ~{expectedBuildingCount})");
        Console.WriteLine($"  Triangle Range: {summary.MinTrianglesPerBuilding:N0} - {summary.MaxTrianglesPerBuilding:N0}");
        Console.WriteLine($"  Average Triangles: {summary.AvgTrianglesPerBuilding:N0}");

        // Validate against expectations
        if (buildings.Count < expectedBuildingCount / 4)
        {
            summary.ValidationIssues.Add($"Building count {buildings.Count} is much lower than expected ~{expectedBuildingCount}");
            Console.WriteLine($"⚠️  Building count may be too low");
        }

        if (summary.MaxTrianglesPerBuilding < expectedMinTriangles)
        {
            summary.ValidationIssues.Add($"Max triangle count {summary.MaxTrianglesPerBuilding:N0} below expected minimum {expectedMinTriangles:N0}");
            Console.WriteLine($"⚠️  Buildings appear fragmented (max triangles too low)");
        }

        if (summary.AvgTrianglesPerBuilding >= expectedMinTriangles)
        {
            Console.WriteLine($"✅ Building scale achieved! Average {summary.AvgTrianglesPerBuilding:N0} triangles per building");
        }
        else
        {
            Console.WriteLine($"⚠️  Average triangle count {summary.AvgTrianglesPerBuilding:N0} below building scale ({expectedMinTriangles:N0})");
        }
    }

    /// <summary>
    /// Group buildings by tile coordinates based on their positions.
    /// </summary>
    private static Dictionary<(int tileX, int tileY), List<PM4Building>> GroupBuildingsByTile(List<PM4Building> buildings, PM4UnifiedMap unifiedMap)
    {
        var buildingsByTile = new Dictionary<(int, int), List<PM4Building>>();

        foreach (var building in buildings)
        {
            // Determine tile from building position (simplified algorithm)
            var tileCoords = EstimateTileFromPosition(building.Position, unifiedMap);
            
            if (!buildingsByTile.ContainsKey(tileCoords))
            {
                buildingsByTile[tileCoords] = new List<PM4Building>();
            }
            
            buildingsByTile[tileCoords].Add(building);
        }

        return buildingsByTile;
    }

    /// <summary>
    /// Estimate tile coordinates from world position.
    /// </summary>
    private static (int tileX, int tileY) EstimateTileFromPosition(System.Numerics.Vector3 position, PM4UnifiedMap unifiedMap)
    {
        // This is a simplified estimation - would need actual tile size constants
        var tileSize = 533.333f; // Approximate WoW tile size
        var tileX = (int)Math.Floor(position.X / tileSize);
        var tileY = (int)Math.Floor(position.Y / tileSize);
        return (tileX, tileY);
    }

    /// <summary>
    /// Build OBJ content for a single tile containing multiple buildings.
    /// </summary>
    private static string BuildTileObjContent(List<PM4Building> buildings, int tileX, int tileY)
    {
        var obj = new System.Text.StringBuilder();
        obj.AppendLine($"# PM4 Tile ({tileX},{tileY}) - {buildings.Count} buildings");
        obj.AppendLine($"# Generated by PM4 Unified Architecture Export");
        obj.AppendLine();

        var vertexOffset = 0;
        foreach (var building in buildings)
        {
            obj.AppendLine($"# Building {building.BuildingId}");
            obj.AppendLine($"o Building_{building.BuildingId}");

            // Write vertices for this building
            foreach (var vertex in building.Vertices)
            {
                obj.AppendLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }

            // Write faces for this building (with vertex offset)
            for (int i = 0; i < building.Indices.Count; i += 3)
            {
                if (i + 2 < building.Indices.Count)
                {
                    var idx1 = building.Indices[i] + vertexOffset + 1;
                    var idx2 = building.Indices[i + 1] + vertexOffset + 1;
                    var idx3 = building.Indices[i + 2] + vertexOffset + 1;
                    obj.AppendLine($"f {idx1} {idx2} {idx3}");
                }
            }

            vertexOffset += building.Vertices.Count;
            obj.AppendLine();
        }

        return obj.ToString();
    }

    /// <summary>
    /// Build OBJ content for unified scene with all buildings.
    /// </summary>
    private static string BuildUnifiedObjContent(List<PM4Building> buildings)
    {
        var obj = new System.Text.StringBuilder();
        obj.AppendLine($"# PM4 Unified Scene - {buildings.Count} buildings");
        obj.AppendLine($"# Generated by PM4 Unified Architecture Export");
        obj.AppendLine();

        var vertexOffset = 0;
        foreach (var building in buildings)
        {
            obj.AppendLine($"g Building_{building.BuildingId}");

            // Write vertices
            foreach (var vertex in building.Vertices)
            {
                obj.AppendLine($"v {-vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }

            vertexOffset += building.Vertices.Count;
        }

        obj.AppendLine();

        // Write faces for all buildings
        vertexOffset = 0;
        foreach (var building in buildings)
        {
            obj.AppendLine($"# Faces for Building {building.BuildingId}");
            
            for (int i = 0; i < building.Indices.Count; i += 3)
            {
                if (i + 2 < building.Indices.Count)
                {
                    var idx1 = building.Indices[i] + vertexOffset + 1;
                    var idx2 = building.Indices[i + 1] + vertexOffset + 1;
                    var idx3 = building.Indices[i + 2] + vertexOffset + 1;
                    obj.AppendLine($"f {idx1} {idx2} {idx3}");
                }
            }

            vertexOffset += building.Vertices.Count;
        }

        return obj.ToString();
    }

    /// <summary>
    /// Generate comprehensive export report.
    /// </summary>
    private static async Task GenerateExportReportAsync(PM4ExportSummary summary, string outputDirectory)
    {
        var reportPath = Path.Combine(outputDirectory, "PM4_Export_Report.txt");
        var report = summary.GenerateReport();
        await File.WriteAllTextAsync(reportPath, report);
        Console.WriteLine($"📄 Export report saved: {reportPath}");
    }

    /// <summary>
    /// Assemble buildings from unified map using proven direct-export approach.
    /// Uses working MSLK linkage logic adapted for unified map data.
    /// </summary>
    private static bool _mslkDiagnosticLogged;
    private static bool _msurDiagnosticLogged;

    private static List<PM4Building> AssembleBuildingsFromUnifiedMap(PM4UnifiedMap unifiedMap)
    {
        var buildings = new List<PM4Building>();

        try
        {
            Console.WriteLine($"[UNIFIED ASSEMBLER] Processing {unifiedMap.AllMslkLinks.Count} MSLK links, {unifiedMap.AllMsurSurfaces.Count} surfaces");

            // Group MSLK entries by ParentIndex (same as DirectPm4Exporter's ParentId logic)
            var linksByParent = unifiedMap.AllMslkLinks
                .Where(link => !IsContainerNode(link)) // Skip container nodes
                .GroupBy(link => GetParentIndex(link))
                .Where(group => group.Key.HasValue && group.Any())
                .ToList();

            Console.WriteLine($"[UNIFIED ASSEMBLER] Found {linksByParent.Count} unique parent objects");

            uint buildingId = 0;
            foreach (var parentGroup in linksByParent)
            {
                var building = AssembleBuildingFromParentGroup(unifiedMap, parentGroup.Key.Value, parentGroup.ToList());
                if (building != null && building.TriangleCount > 0)
                {
                    buildings.Add(building);
                    Console.WriteLine($"[UNIFIED ASSEMBLER] Building {buildingId}: {building.TriangleCount} triangles");
                    buildingId++;
                }
            }

            Console.WriteLine($"[UNIFIED ASSEMBLER] Successfully assembled {buildings.Count} buildings");
            return buildings;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[UNIFIED ASSEMBLER ERROR] Assembly failed: {ex.Message}");
            return buildings;
        }
    }

    private static uint? GetParentIndex(MslkEntry link)
    {
        try
        {
            var prop = link.GetType().GetProperty("ParentIndex");
            return prop?.GetValue(link) as uint?;
        }
        catch
        {
            return null;
        }
    }

    private static bool IsContainerNode(MslkEntry link)
    {
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

    private static PM4Building? AssembleBuildingFromParentGroup(PM4UnifiedMap unifiedMap, uint buildingId, List<MslkEntry> links)
    {
        try
        {
            var building = new PM4Building
            {
                BuildingId = buildingId,
                SourceLinks = links,
                Vertices = new List<Vector3>(),
                Indices = new List<uint>(),
                SourceSurfaces = new List<MsurChunk.Entry>()
            };

            // Use a dictionary to map global vertex indices to local indices for deduplication
            var vertexIndexMap = new Dictionary<(float, float, float), int>();
            var nextLocalIndex = 0;
            
            Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Assembling building {buildingId} from {links.Count} links");

            // Process each link to extract geometry
            foreach (var link in links)
            {
                // Skip container nodes (no geometry)
                if (IsContainerNode(link))
                {
                    Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Skipping container node link");
                    continue;
                }

                // Extract triangles from this link
                var (linkVertices, linkTriangles) = ExtractTrianglesFromLink(unifiedMap, link);
                
                Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Link contributed {linkVertices.Count} vertices and {linkTriangles.Count} triangles");

                if (linkVertices.Count > 0 && linkTriangles.Count > 0)
                {
                    // Remap vertices with deduplication
                    var localIndexRemapping = new List<int>();
                    
                    foreach (var vertex in linkVertices)
                    {
                        var vertexKey = (vertex.X, vertex.Y, vertex.Z);
                        
                        if (vertexIndexMap.TryGetValue(vertexKey, out var existingIndex))
                        {
                            // Vertex already exists, use existing index
                            localIndexRemapping.Add(existingIndex);
                        }
                        else
                        {
                            // Add new vertex and create mapping
                            vertexIndexMap[vertexKey] = nextLocalIndex;
                            localIndexRemapping.Add(nextLocalIndex);
                            building.Vertices.Add(vertex);
                            nextLocalIndex++;
                        }
                    }
                    
                    // Add triangles with remapped indices
                    foreach (var (v1, v2, v3) in linkTriangles)
                    {
                        if (v1 < localIndexRemapping.Count && v2 < localIndexRemapping.Count && v3 < localIndexRemapping.Count)
                        {
                            building.Indices.Add((uint)localIndexRemapping[v1]);
                            building.Indices.Add((uint)localIndexRemapping[v2]);
                            building.Indices.Add((uint)localIndexRemapping[v3]);
                        }
                    }
                }
            }
            
            Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Building {buildingId} assembled with {building.Vertices.Count} vertices and {building.Indices.Count / 3} triangles");
            
            return building;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[UNIFIED ASSEMBLER ERROR] Failed to assemble building {buildingId}: {ex.Message}");
            return null;
        }
    }

    private static (List<Vector3> vertices, List<(int, int, int)> triangles) ExtractTrianglesFromLink(PM4UnifiedMap unifiedMap, MslkEntry link)
    {
        var vertices = new List<Vector3>();
        var triangles = new List<(int, int, int)>();

        try
        {
            // Get surface reference index with robust type handling
            var surfaceRefProp = link.GetType().GetProperty("SurfaceRefIndex");
            if (surfaceRefProp == null)
            {
                Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Link missing SurfaceRefIndex property");
                return (vertices, triangles);
            }

            var surfaceRefValue = surfaceRefProp.GetValue(link);
            uint surfaceIndex = surfaceRefValue switch
            {
                uint u => u,
                int i when i >= 0 => (uint)i,
                ushort us => us,
                short s when s >= 0 => (uint)s,
                _ => throw new InvalidOperationException($"Unexpected SurfaceRefIndex type: {surfaceRefValue?.GetType()}")
            };

            Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Extracted surface index: {surfaceIndex}");

            // Validate surface index bounds (this is where cross-tile reference issue will show)
            if (surfaceIndex >= unifiedMap.AllMsurSurfaces.Count)
            {
                Console.WriteLine($"[UNIFIED ASSEMBLER WARNING] Surface index {surfaceIndex} out of range (0-{unifiedMap.AllMsurSurfaces.Count - 1})");
                return (vertices, triangles);
            }
            
            Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Processing surface {surfaceIndex}");

            var surface = unifiedMap.AllMsurSurfaces[(int)surfaceIndex];
            
            // Extract vertex indices from surface using robust property access
            var firstIndexProp = surface.GetType().GetProperty("MsviFirstIndex") ?? surface.GetType().GetProperty("FirstIndex");
            var indexCountProp = surface.GetType().GetProperty("IndexCount");
            
            // Also check for MSPI properties as an alternative
            var mspiFirstIndexProp = surface.GetType().GetProperty("MspiFirstIndex");
            
            if (firstIndexProp?.GetValue(surface) is uint firstIndex && 
                indexCountProp?.GetValue(surface) is uint indexCount)
            {
                Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Surface {surfaceIndex} using MSVI pool: firstIndex={firstIndex}, indexCount={indexCount}");
                
                // Extract vertices and build triangles from MSVI pool
                for (uint i = 0; i < indexCount && (firstIndex + i + 2) < unifiedMap.GlobalMSVIIndices.Count; i += 3)
                {
                    var i1 = unifiedMap.GlobalMSVIIndices[(int)(firstIndex + i)];
                    var i2 = unifiedMap.GlobalMSVIIndices[(int)(firstIndex + i + 1)];
                    var i3 = unifiedMap.GlobalMSVIIndices[(int)(firstIndex + i + 2)];

                    Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Triangle indices: {i1}, {i2}, {i3}");
                    
                    // Check if indices are valid for MSVT vertices
                    if (i1 < unifiedMap.GlobalMSVTVertices.Count && i2 < unifiedMap.GlobalMSVTVertices.Count && i3 < unifiedMap.GlobalMSVTVertices.Count)
                    {
                        int baseIdx = vertices.Count;
                        var v1 = unifiedMap.GlobalMSVTVertices[(int)i1];
                        var v2 = unifiedMap.GlobalMSVTVertices[(int)i2];
                        var v3 = unifiedMap.GlobalMSVTVertices[(int)i3];
                        
                        // Apply coordinate system fix: flip X-axis for proper orientation (like DirectPm4Exporter)
                        vertices.Add(new Vector3(-v1.X, v1.Y, v1.Z));
                        vertices.Add(new Vector3(-v2.X, v2.Y, v2.Z));
                        vertices.Add(new Vector3(-v3.X, v3.Y, v3.Z));
                        
                        triangles.Add((baseIdx, baseIdx + 1, baseIdx + 2));
                        Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Added triangle from MSVT pool");
                    }
                    else
                    {
                        Console.WriteLine($"[UNIFIED ASSEMBLER WARNING] Invalid MSVT vertex indices: {i1}, {i2}, {i3} (max: {unifiedMap.GlobalMSVTVertices.Count})");
                    }
                }
            }
            else if (mspiFirstIndexProp?.GetValue(surface) is int mspiFirstIndex && mspiFirstIndex >= 0 &&
                     indexCountProp?.GetValue(surface) is uint mspiIndexCount)
            {
                Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Surface {surfaceIndex} using MSPI pool: firstIndex={mspiFirstIndex}, indexCount={mspiIndexCount}");
                
                // Extract vertices and build triangles from MSPI pool (references MSPV vertices)
                for (uint i = 0; i < mspiIndexCount && (mspiFirstIndex + i + 2) < unifiedMap.GlobalMSPIIndices.Count; i += 3)
                {
                    var i1 = unifiedMap.GlobalMSPIIndices[mspiFirstIndex + (int)i];
                    var i2 = unifiedMap.GlobalMSPIIndices[mspiFirstIndex + (int)i + 1];
                    var i3 = unifiedMap.GlobalMSPIIndices[mspiFirstIndex + (int)i + 2];

                    Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] MSPI Triangle indices: {i1}, {i2}, {i3}");
                    
                    // Check if indices are valid for MSPV vertices
                    if (i1 < unifiedMap.GlobalMSPVVertices.Count && i2 < unifiedMap.GlobalMSPVVertices.Count && i3 < unifiedMap.GlobalMSPVVertices.Count)
                    {
                        int baseIdx = vertices.Count;
                        var v1 = unifiedMap.GlobalMSPVVertices[(int)i1];
                        var v2 = unifiedMap.GlobalMSPVVertices[(int)i2];
                        var v3 = unifiedMap.GlobalMSPVVertices[(int)i3];
                        
                        // Apply coordinate system fix: flip X-axis for proper orientation (like DirectPm4Exporter)
                        vertices.Add(new Vector3(-v1.X, v1.Y, v1.Z));
                        vertices.Add(new Vector3(-v2.X, v2.Y, v2.Z));
                        vertices.Add(new Vector3(-v3.X, v3.Y, v3.Z));
                        
                        triangles.Add((baseIdx, baseIdx + 1, baseIdx + 2));
                        Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Added triangle from MSPV pool");
                    }
                    else
                    {
                        Console.WriteLine($"[UNIFIED ASSEMBLER WARNING] Invalid MSPV vertex indices: {i1}, {i2}, {i3} (max: {unifiedMap.GlobalMSPVVertices.Count})");
                    }
                }
            }
            else
            {
                if (!_msurDiagnosticLogged)
                {
                    Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] MSUR Surface properties: {string.Join(", ", surface.GetType().GetProperties().Select(p => p.Name))}");
                    foreach (var p in surface.GetType().GetProperties())
                    {
                        try { Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG]   {p.Name} = {p.GetValue(surface)}"); } catch { }
                    }
                    _msurDiagnosticLogged = true;
                }
                Console.WriteLine($"[UNIFIED ASSEMBLER WARNING] Surface {surfaceIndex} missing geometry parameters");
            }

            Console.WriteLine($"[UNIFIED ASSEMBLER DEBUG] Extracted {vertices.Count} vertices and {triangles.Count} triangles from surface {surfaceIndex}");
            return (vertices, triangles);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[UNIFIED ASSEMBLER ERROR] Failed to extract triangles from link: {ex.Message}");
            return (vertices, triangles);
        }
    }
    
    /// <summary>
    /// Extract ParentIndex from MSLK link entry using reflection.
    /// This is the building identifier that groups related surface fragments.
    /// </summary>
    private static uint GetParentIndex(dynamic link)
    {
        try
        {
            // Try to get ParentIndex property (most common)
            var parentIndexProp = link.GetType().GetProperty("ParentIndex");
            if (parentIndexProp != null)
            {
                var value = parentIndexProp.GetValue(link);
                return value switch
                {
                    uint u => u,
                    int i when i >= 0 => (uint)i,
                    ushort us => us,
                    short s when s >= 0 => (uint)s,
                    byte b => b,
                    _ => 0
                };
            }
            
            // Fallback to ParentId if ParentIndex not found
            var parentIdProp = link.GetType().GetProperty("ParentId");
            if (parentIdProp != null)
            {
                var value = parentIdProp.GetValue(link);
                return value switch
                {
                    uint u => u,
                    int i when i >= 0 => (uint)i,
                    ushort us => us,
                    short s when s >= 0 => (uint)s,
                    byte b => b,
                    _ => 0
                };
            }
            
            Console.WriteLine($"[UNIFIED ASSEMBLER WARNING] Link missing ParentIndex/ParentId property");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[UNIFIED ASSEMBLER ERROR] Failed to extract ParentIndex: {ex.Message}");
            return 0;
        }
    }
}

/// <summary>
/// Export strategy options for PM4 unified architecture.
/// </summary>
public enum PM4ExportStrategy
{
    /// <summary>Export each building as separate OBJ file (recommended).</summary>
    PerBuilding,
    
    /// <summary>Export buildings grouped by tile boundaries.</summary>
    PerTile,
    
    /// <summary>Export all buildings as single unified OBJ file.</summary>
    Unified
}
