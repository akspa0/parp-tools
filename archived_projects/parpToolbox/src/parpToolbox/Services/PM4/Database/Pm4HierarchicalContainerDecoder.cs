using Microsoft.EntityFrameworkCore;
using ParpToolbox.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Decodes PM4 hierarchical container system using BoundsCenterX/Y/Z fields
    /// to reconstruct complete objects through container relationship traversal.
    /// </summary>
    public class Pm4HierarchicalContainerDecoder
    {
        private readonly string _databasePath;
        private readonly string _outputDirectory;
        
        public Pm4HierarchicalContainerDecoder(string databasePath, string outputDirectory)
        {
            _databasePath = databasePath ?? throw new ArgumentNullException(nameof(databasePath));
            _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));
            
            if (!Directory.Exists(_outputDirectory))
            {
                Directory.CreateDirectory(_outputDirectory);
            }
        }
        
        /// <summary>
        /// Main analysis entry point for hierarchical container decoding.
        /// </summary>
        public async Task AnalyzeAsync()
        {
            ConsoleLogger.WriteLine("[HIERARCHICAL DECODER] Starting PM4 hierarchical container analysis...");

            using var context = new Pm4DatabaseContext(_databasePath);

            // Load all relevant data for cross-chunk container analysis
            var surfaces = await context.Surfaces.ToListAsync();
            var links = await context.Links.ToListAsync();
            var placements = await context.Placements.ToListAsync();

            if (!surfaces.Any())
            {
                ConsoleLogger.WriteLine("No surfaces found in database.");
                return;
            }

            ConsoleLogger.WriteLine($"[HIERARCHICAL DECODER] Loaded {surfaces.Count} surfaces, {links.Count} links, {placements.Count} placements");

            // Step 1: Map container hierarchies within surfaces
            await MapSurfaceContainerHierarchy(surfaces);
            
            // Step 2: Cross-reference with MSLK links
            await CrossReferenceWithLinks(surfaces, links);
            
            // Step 3: Cross-reference with MPRL placements
            await CrossReferenceWithPlacements(surfaces, placements);
            
            // Step 4: Build complete object hierarchy trees
            await BuildObjectHierarchyTrees(surfaces, links, placements);
            
            // Step 5: Validate hierarchical object assembly
            await ValidateHierarchicalAssembly(surfaces, links, placements);

            ConsoleLogger.WriteLine($"[HIERARCHICAL DECODER] Analysis complete. Reports saved to: {_outputDirectory}");
        }

        /// <summary>
        /// Step 1: Map container hierarchies within surface data
        /// </summary>
        private async Task MapSurfaceContainerHierarchy(List<Pm4Surface> surfaces)
        {
            ConsoleLogger.WriteLine("[HIERARCHICAL DECODER] STEP 1: Mapping surface container hierarchy...");
            
            // Analyze container ID distributions
            var containerXStats = surfaces.GroupBy(s => s.BoundsCenterX).Select(g => new { 
                ContainerX = g.Key, 
                Count = g.Count(),
                AvgTriangles = g.Average(s => s.IndexCount),
                TotalTriangles = g.Sum(s => s.IndexCount)
            }).OrderBy(x => x.ContainerX).ToList();
            
            var containerYStats = surfaces.GroupBy(s => s.BoundsCenterY).Select(g => new { 
                ContainerY = g.Key, 
                Count = g.Count(),
                AvgTriangles = g.Average(s => s.IndexCount),
                TotalTriangles = g.Sum(s => s.IndexCount)
            }).OrderBy(x => x.ContainerY).ToList();
            
            var containerZStats = surfaces.GroupBy(s => s.BoundsCenterZ).Select(g => new { 
                ContainerZ = g.Key, 
                Count = g.Count(),
                AvgTriangles = g.Average(s => s.IndexCount),
                TotalTriangles = g.Sum(s => s.IndexCount)
            }).OrderBy(x => x.ContainerZ).ToList();

            ConsoleLogger.WriteLine($"  Container X range: {containerXStats.First().ContainerX} to {containerXStats.Last().ContainerX} ({containerXStats.Count} levels)");
            ConsoleLogger.WriteLine($"  Container Y range: {containerYStats.First().ContainerY} to {containerYStats.Last().ContainerY} ({containerYStats.Count} levels)");
            ConsoleLogger.WriteLine($"  Container Z range: {containerZStats.First().ContainerZ} to {containerZStats.Last().ContainerZ} ({containerZStats.Count} levels)");

            // Analyze hierarchical combinations
            var hierarchicalCombos = surfaces.GroupBy(s => new { 
                ContainerX = s.BoundsCenterX, 
                ContainerY = s.BoundsCenterY, 
                ContainerZ = s.BoundsCenterZ 
            }).Select(g => new {
                Combination = g.Key,
                Count = g.Count(),
                TotalTriangles = g.Sum(s => s.IndexCount),
                SampleSurfaces = g.Take(5).Select(s => new { s.Id, s.MsviFirstIndex, s.IndexCount }).ToList()
            }).OrderByDescending(x => x.Count).ToList();

            ConsoleLogger.WriteLine($"  Found {hierarchicalCombos.Count} unique container combinations");
            ConsoleLogger.WriteLine($"  Top combinations by surface count:");
            foreach (var combo in hierarchicalCombos.Take(10))
            {
                ConsoleLogger.WriteLine($"    ({combo.Combination.ContainerX}, {combo.Combination.ContainerY}, {combo.Combination.ContainerZ}): {combo.Count} surfaces, {combo.TotalTriangles} triangles");
            }

            // Save container hierarchy mapping
            var csvPath = Path.Combine(_outputDirectory, "surface_container_hierarchy.csv");
            var csvContent = "ContainerX,ContainerY,ContainerZ,SurfaceCount,TotalTriangles,AvgTriangles,SampleSurfaceIds\n" +
                string.Join("\n", hierarchicalCombos.Select(c => 
                    $"{c.Combination.ContainerX},{c.Combination.ContainerY},{c.Combination.ContainerZ},{c.Count},{c.TotalTriangles},{c.TotalTriangles / (double)c.Count:F2},\"{string.Join(";", c.SampleSurfaces.Select(s => s.Id))}\""));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Surface container hierarchy saved to: {csvPath}");
        }

        /// <summary>
        /// Step 2: Cross-reference surface containers with MSLK links
        /// </summary>
        private async Task CrossReferenceWithLinks(List<Pm4Surface> surfaces, List<Pm4Link> links)
        {
            ConsoleLogger.WriteLine("[HIERARCHICAL DECODER] STEP 2: Cross-referencing with MSLK links...");
            
            // Map surfaces to links by various potential relationships
            var surfaceToLinkMappings = new List<dynamic>();
            
            foreach (var surface in surfaces)
            {
                // Look for potential link relationships
                var potentialLinks = links.Where(l => 
                    l.ParentIndex == surface.BoundsCenterX ||
                    l.ParentIndex == surface.BoundsCenterY ||
                    l.ParentIndex == surface.BoundsCenterZ ||
                    l.MspiFirstIndex == surface.MsviFirstIndex ||
                    l.MspiIndexCount == surface.IndexCount
                ).ToList();

                if (potentialLinks.Any())
                {
                    surfaceToLinkMappings.Add(new {
                        SurfaceId = surface.Id,
                        ContainerX = surface.BoundsCenterX,
                        ContainerY = surface.BoundsCenterY,
                        ContainerZ = surface.BoundsCenterZ,
                        MsviFirstIndex = surface.MsviFirstIndex,
                        IndexCount = surface.IndexCount,
                        LinkedCount = potentialLinks.Count,
                        LinkIds = potentialLinks.Select(l => l.Id).ToList(),
                        LinkParentIndices = potentialLinks.Select(l => l.ParentIndex).Distinct().ToList()
                    });
                }
            }

            ConsoleLogger.WriteLine($"  Found {surfaceToLinkMappings.Count} surfaces with potential link relationships");
            
            // Analyze link relationship patterns
            var linkPatterns = surfaceToLinkMappings.GroupBy(m => new { 
                LinkedCount = m.LinkedCount 
            }).Select(g => new {
                LinkedCount = g.Key.LinkedCount,
                SurfaceCount = g.Count()
            }).OrderByDescending(x => x.SurfaceCount).ToList();

            ConsoleLogger.WriteLine($"  Link relationship patterns:");
            foreach (var pattern in linkPatterns)
            {
                ConsoleLogger.WriteLine($"    {pattern.LinkedCount} links: {pattern.SurfaceCount} surfaces");
            }

            // Save surface-to-link cross-reference
            var csvPath = Path.Combine(_outputDirectory, "surface_link_cross_reference.csv");
            var csvContent = "SurfaceId,ContainerX,ContainerY,ContainerZ,MsviFirstIndex,IndexCount,LinkedCount,LinkIds,LinkParentIndices\n" +
                string.Join("\n", surfaceToLinkMappings.Select(m => 
                    $"{m.SurfaceId},{m.ContainerX},{m.ContainerY},{m.ContainerZ},{m.MsviFirstIndex},{m.IndexCount},{m.LinkedCount},\"{string.Join(";", m.LinkIds)}\",\"{string.Join(";", m.LinkParentIndices)}\""));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Surface-link cross-reference saved to: {csvPath}");
        }

        /// <summary>
        /// Step 3: Cross-reference surface containers with MPRL placements
        /// </summary>
        private async Task CrossReferenceWithPlacements(List<Pm4Surface> surfaces, List<Pm4Placement> placements)
        {
            ConsoleLogger.WriteLine("[HIERARCHICAL DECODER] STEP 3: Cross-referencing with MPRL placements...");
            
            // Map surfaces to placements by container IDs
            var surfaceToPlacementMappings = new List<dynamic>();
            
            foreach (var surface in surfaces)
            {
                // Look for placement relationships with container IDs
                var potentialPlacements = placements.Where(p => 
                    p.Unknown4 == surface.BoundsCenterX ||
                    p.Unknown4 == surface.BoundsCenterY ||
                    p.Unknown4 == surface.BoundsCenterZ ||
                    p.Unknown6 == surface.BoundsCenterX ||
                    p.Unknown6 == surface.BoundsCenterY ||
                    p.Unknown6 == surface.BoundsCenterZ
                ).ToList();

                if (potentialPlacements.Any())
                {
                    surfaceToPlacementMappings.Add(new {
                        SurfaceId = surface.Id,
                        ContainerX = surface.BoundsCenterX,
                        ContainerY = surface.BoundsCenterY,
                        ContainerZ = surface.BoundsCenterZ,
                        PlacementCount = potentialPlacements.Count,
                        PlacementIds = potentialPlacements.Select(p => p.Id).ToList(),
                        PlacementUnknown4s = potentialPlacements.Select(p => p.Unknown4).Distinct().ToList(),
                        PlacementUnknown6s = potentialPlacements.Select(p => p.Unknown6).Distinct().ToList()
                    });
                }
            }

            ConsoleLogger.WriteLine($"  Found {surfaceToPlacementMappings.Count} surfaces with potential placement relationships");

            // Analyze placement relationship patterns
            var placementPatterns = surfaceToPlacementMappings.GroupBy(m => new { 
                PlacementCount = m.PlacementCount 
            }).Select(g => new {
                PlacementCount = g.Key.PlacementCount,
                SurfaceCount = g.Count()
            }).OrderByDescending(x => x.SurfaceCount).ToList();

            ConsoleLogger.WriteLine($"  Placement relationship patterns:");
            foreach (var pattern in placementPatterns)
            {
                ConsoleLogger.WriteLine($"    {pattern.PlacementCount} placements: {pattern.SurfaceCount} surfaces");
            }

            // Save surface-to-placement cross-reference
            var csvPath = Path.Combine(_outputDirectory, "surface_placement_cross_reference.csv");
            var csvContent = "SurfaceId,ContainerX,ContainerY,ContainerZ,PlacementCount,PlacementIds,PlacementUnknown4s,PlacementUnknown6s\n" +
                string.Join("\n", surfaceToPlacementMappings.Select(m => 
                    $"{m.SurfaceId},{m.ContainerX},{m.ContainerY},{m.ContainerZ},{m.PlacementCount},\"{string.Join(";", m.PlacementIds)}\",\"{string.Join(";", m.PlacementUnknown4s)}\",\"{string.Join(";", m.PlacementUnknown6s)}\""));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Surface-placement cross-reference saved to: {csvPath}");
        }

        /// <summary>
        /// Step 4: Build complete object hierarchy trees
        /// </summary>
        private async Task BuildObjectHierarchyTrees(List<Pm4Surface> surfaces, List<Pm4Link> links, List<Pm4Placement> placements)
        {
            ConsoleLogger.WriteLine("[HIERARCHICAL DECODER] STEP 4: Building object hierarchy trees...");
            
            // Build hierarchical object trees by grouping related surfaces, links, and placements
            var objectTrees = new List<dynamic>();
            
            // Group surfaces by their container combinations and build trees
            var containerGroups = surfaces.GroupBy(s => new { 
                ContainerX = s.BoundsCenterX, 
                ContainerY = s.BoundsCenterY, 
                ContainerZ = s.BoundsCenterZ 
            }).ToList();

            foreach (var group in containerGroups)
            {
                var groupSurfaces = group.ToList();
                
                // Find related links for this container group
                var relatedLinks = links.Where(l => 
                    groupSurfaces.Any(s => 
                        l.ParentIndex == s.BoundsCenterX ||
                        l.ParentIndex == s.BoundsCenterY ||
                        l.ParentIndex == s.BoundsCenterZ ||
                        l.MspiFirstIndex == s.MsviFirstIndex
                    )
                ).ToList();

                // Find related placements for this container group
                var relatedPlacements = placements.Where(p => 
                    groupSurfaces.Any(s => 
                        p.Unknown4 == s.BoundsCenterX ||
                        p.Unknown4 == s.BoundsCenterY ||
                        p.Unknown4 == s.BoundsCenterZ ||
                        p.Unknown6 == s.BoundsCenterX ||
                        p.Unknown6 == s.BoundsCenterY ||
                        p.Unknown6 == s.BoundsCenterZ
                    )
                ).ToList();

                // Create object tree node
                var objectTree = new {
                    ContainerX = group.Key.ContainerX,
                    ContainerY = group.Key.ContainerY,
                    ContainerZ = group.Key.ContainerZ,
                    SurfaceCount = groupSurfaces.Count,
                    TotalTriangles = groupSurfaces.Sum(s => s.IndexCount),
                    RelatedLinkCount = relatedLinks.Count,
                    RelatedPlacementCount = relatedPlacements.Count,
                    SurfaceIds = groupSurfaces.Select(s => s.Id).ToList(),
                    LinkIds = relatedLinks.Select(l => l.Id).ToList(),
                    PlacementIds = relatedPlacements.Select(p => p.Id).ToList(),
                    // Calculate object "completeness" score
                    CompletenessScore = (groupSurfaces.Count > 0 ? 1 : 0) + 
                                      (relatedLinks.Count > 0 ? 1 : 0) + 
                                      (relatedPlacements.Count > 0 ? 1 : 0)
                };

                objectTrees.Add(objectTree);
            }

            // Sort by completeness and size for analysis
            objectTrees = objectTrees.OrderByDescending(t => t.CompletenessScore)
                                   .ThenByDescending(t => t.TotalTriangles)
                                   .ToList();

            ConsoleLogger.WriteLine($"  Built {objectTrees.Count} object hierarchy trees");
            ConsoleLogger.WriteLine($"  Top complete objects by triangle count:");
            foreach (var tree in objectTrees.Take(10))
            {
                ConsoleLogger.WriteLine($"    Container({tree.ContainerX},{tree.ContainerY},{tree.ContainerZ}): {tree.TotalTriangles} triangles, {tree.SurfaceCount} surfaces, {tree.RelatedLinkCount} links, {tree.RelatedPlacementCount} placements (score: {tree.CompletenessScore})");
            }

            // Save object hierarchy trees
            var csvPath = Path.Combine(_outputDirectory, "object_hierarchy_trees.csv");
            var csvContent = "ContainerX,ContainerY,ContainerZ,SurfaceCount,TotalTriangles,RelatedLinkCount,RelatedPlacementCount,CompletenessScore,SurfaceIds,LinkIds,PlacementIds\n" +
                string.Join("\n", objectTrees.Select(t => 
                    $"{t.ContainerX},{t.ContainerY},{t.ContainerZ},{t.SurfaceCount},{t.TotalTriangles},{t.RelatedLinkCount},{t.RelatedPlacementCount},{t.CompletenessScore},\"{string.Join(";", t.SurfaceIds)}\",\"{string.Join(";", t.LinkIds)}\",\"{string.Join(";", t.PlacementIds)}\""));
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Object hierarchy trees saved to: {csvPath}");
        }

        /// <summary>
        /// Step 5: Validate hierarchical object assembly
        /// </summary>
        private async Task ValidateHierarchicalAssembly(List<Pm4Surface> surfaces, List<Pm4Link> links, List<Pm4Placement> placements)
        {
            ConsoleLogger.WriteLine("[HIERARCHICAL DECODER] STEP 5: Validating hierarchical object assembly...");
            
            // Validate that hierarchical approach produces better object groupings than spatial clustering
            var hierarchicalStats = new
            {
                TotalSurfaces = surfaces.Count,
                TotalLinks = links.Count,
                TotalPlacements = placements.Count,
                UniqueContainerCombinations = surfaces.GroupBy(s => new { s.BoundsCenterX, s.BoundsCenterY, s.BoundsCenterZ }).Count(),
                AverageTrianglesPerContainer = surfaces.GroupBy(s => new { s.BoundsCenterX, s.BoundsCenterY, s.BoundsCenterZ })
                                                      .Average(g => g.Sum(s => s.IndexCount)),
                LargestContainerTriangles = surfaces.GroupBy(s => new { s.BoundsCenterX, s.BoundsCenterY, s.BoundsCenterZ })
                                                   .Max(g => g.Sum(s => s.IndexCount)),
                SmallestContainerTriangles = surfaces.GroupBy(s => new { s.BoundsCenterX, s.BoundsCenterY, s.BoundsCenterZ })
                                                    .Min(g => g.Sum(s => s.IndexCount))
            };

            ConsoleLogger.WriteLine($"  Hierarchical assembly validation:");
            ConsoleLogger.WriteLine($"    Total surfaces: {hierarchicalStats.TotalSurfaces}");
            ConsoleLogger.WriteLine($"    Total links: {hierarchicalStats.TotalLinks}");
            ConsoleLogger.WriteLine($"    Total placements: {hierarchicalStats.TotalPlacements}");
            ConsoleLogger.WriteLine($"    Unique container combinations: {hierarchicalStats.UniqueContainerCombinations}");
            ConsoleLogger.WriteLine($"    Average triangles per container: {hierarchicalStats.AverageTrianglesPerContainer:F1}");
            ConsoleLogger.WriteLine($"    Triangle range per container: {hierarchicalStats.SmallestContainerTriangles} to {hierarchicalStats.LargestContainerTriangles}");

            // Calculate object size distribution
            var containerSizes = surfaces.GroupBy(s => new { s.BoundsCenterX, s.BoundsCenterY, s.BoundsCenterZ })
                                        .Select(g => g.Sum(s => s.IndexCount))
                                        .OrderByDescending(size => size)
                                        .ToList();

            var sizeDistribution = new
            {
                TinyObjects = containerSizes.Count(size => size < 100),
                SmallObjects = containerSizes.Count(size => size >= 100 && size < 1000),
                MediumObjects = containerSizes.Count(size => size >= 1000 && size < 10000),
                LargeObjects = containerSizes.Count(size => size >= 10000 && size < 100000),
                HugeObjects = containerSizes.Count(size => size >= 100000)
            };

            ConsoleLogger.WriteLine($"  Object size distribution:");
            ConsoleLogger.WriteLine($"    Tiny (<100 triangles): {sizeDistribution.TinyObjects}");
            ConsoleLogger.WriteLine($"    Small (100-1K triangles): {sizeDistribution.SmallObjects}");
            ConsoleLogger.WriteLine($"    Medium (1K-10K triangles): {sizeDistribution.MediumObjects}");
            ConsoleLogger.WriteLine($"    Large (10K-100K triangles): {sizeDistribution.LargeObjects}");
            ConsoleLogger.WriteLine($"    Huge (>100K triangles): {sizeDistribution.HugeObjects}");

            // Save validation results
            var csvPath = Path.Combine(_outputDirectory, "hierarchical_assembly_validation.csv");
            var csvContent = "Metric,Value\n" +
                $"TotalSurfaces,{hierarchicalStats.TotalSurfaces}\n" +
                $"TotalLinks,{hierarchicalStats.TotalLinks}\n" +
                $"TotalPlacements,{hierarchicalStats.TotalPlacements}\n" +
                $"UniqueContainerCombinations,{hierarchicalStats.UniqueContainerCombinations}\n" +
                $"AverageTrianglesPerContainer,{hierarchicalStats.AverageTrianglesPerContainer:F1}\n" +
                $"LargestContainerTriangles,{hierarchicalStats.LargestContainerTriangles}\n" +
                $"SmallestContainerTriangles,{hierarchicalStats.SmallestContainerTriangles}\n" +
                $"TinyObjects,{sizeDistribution.TinyObjects}\n" +
                $"SmallObjects,{sizeDistribution.SmallObjects}\n" +
                $"MediumObjects,{sizeDistribution.MediumObjects}\n" +
                $"LargeObjects,{sizeDistribution.LargeObjects}\n" +
                $"HugeObjects,{sizeDistribution.HugeObjects}\n";
            
            await File.WriteAllTextAsync(csvPath, csvContent);
            ConsoleLogger.WriteLine($"  Hierarchical assembly validation saved to: {csvPath}");
        }
    }
}
