using System;
using System.IO;
using System.Linq;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Simple demo program to analyze MSLK hierarchy and output results to console.
    /// Can be called directly without running complex test frameworks.
    /// </summary>
    public static class MslkHierarchyDemo
    {
        public static void RunHierarchyAnalysis()
        {
            Console.WriteLine("🔗 MSLK HIERARCHY ANALYSIS DEMO");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine("Analyzing MSLK as a node-link system to discover object hierarchies");
            Console.WriteLine();

            // Find test PM4 files
            var testDataPaths = new[]
            {
                "test_data/original_development",
                "test_data/development",
                "test_data/development_335"
            };

            string? testDataDir = null;
            foreach (var path in testDataPaths)
            {
                if (Directory.Exists(path))
                {
                    testDataDir = path;
                    Console.WriteLine($"✅ Found test data: {path}");
                    break;
                }
            }

            if (testDataDir == null)
            {
                Console.WriteLine("❌ No test data directories found. Expected:");
                foreach (var path in testDataPaths)
                {
                    Console.WriteLine($"   - {path}");
                }
                return;
            }

            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4", SearchOption.AllDirectories)
                                   .Take(2) // Analyze 2 files for demo
                                   .ToArray();

            if (!pm4Files.Any())
            {
                Console.WriteLine($"❌ No PM4 files found in {testDataDir}");
                return;
            }

            Console.WriteLine($"🎯 Found {pm4Files.Length} PM4 files to analyze");
            Console.WriteLine();

            var hierarchyAnalyzer = new MslkHierarchyAnalyzer();

            foreach (var filePath in pm4Files)
            {
                var fileName = Path.GetFileName(filePath);
                Console.WriteLine($"📄 ANALYZING: {fileName}");
                Console.WriteLine(new string('─', 60));

                try
                {
                    var pm4File = PM4File.FromFile(filePath);

                    if (pm4File.MSLK?.Entries == null || !pm4File.MSLK.Entries.Any())
                    {
                        Console.WriteLine("   ⚠️  No MSLK chunk or entries found");
                        Console.WriteLine();
                        continue;
                    }

                    Console.WriteLine($"   📊 MSLK Entries: {pm4File.MSLK.Entries.Count}");

                    // Perform hierarchy analysis
                    var hierarchyResult = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);

                    // Segment objects by hierarchy
                    var objectSegments = hierarchyAnalyzer.SegmentObjectsByHierarchy(hierarchyResult);

                    // ✨ NEW: Generate ALL segmentation strategies for comprehensive analysis
                    var allStrategies = hierarchyAnalyzer.SegmentByAllStrategies(hierarchyResult);

                    // Generate and display detailed report
                    var report = hierarchyAnalyzer.GenerateHierarchyReport(hierarchyResult, fileName);
                    Console.WriteLine(report);

                    // Export TXT
                    var outputDir = Path.Combine("output");
                    Directory.CreateDirectory(outputDir);
                    var txtPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}.mslk.txt");
                    File.WriteAllText(txtPath, report);
                    Console.WriteLine($"[TXT] Analysis written to: {txtPath}");

                    // YAML exports removed to prevent circular reference issues

                    // Export segmentation as TXT
                    var objectsTxtPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}.mslk.objects.txt");
                    using (var writer = new StreamWriter(objectsTxtPath))
                    {
                        foreach (var obj in objectSegments)
                        {
                            writer.WriteLine($"Object Root: {obj.RootIndex}");
                            writer.WriteLine($"  Geometry Nodes: {string.Join(", ", obj.GeometryNodeIndices)}");
                            writer.WriteLine($"  Doodad Nodes: {string.Join(", ", obj.DoodadNodeIndices)}");
                            writer.WriteLine();
                        }
                    }
                    Console.WriteLine($"[TXT] Object segmentation written to: {objectsTxtPath}");

                    // ✨ NEW: SPATIAL CLUSTERING + WMO-STYLE EXPORT
                    Console.WriteLine("\n=== SPATIAL CLUSTERING ANALYSIS ===");
                    var objectMeshExporter = new MslkObjectMeshExporter();
                    var spatialClusteringAnalyzer = new MslkSpatialClusteringAnalyzer();
                    var baseFileName = Path.GetFileNameWithoutExtension(fileName);
                    
                    // 1. Get individual geometry components for spatial analysis
                    var individualGeometry = hierarchyAnalyzer.SegmentByIndividualGeometry(hierarchyResult);
                    Console.WriteLine($"🔍 Analyzing {individualGeometry.Count} individual geometry components for spatial clustering...");
                    
                    // 2. Extract spatial properties from each component
                    var spatialComponents = spatialClusteringAnalyzer.AnalyzeSpatialComponents(individualGeometry, pm4File);
                    
                    // 3. Perform spatial clustering with different parameters
                    var clusteringParams = new MslkSpatialClusteringAnalyzer.ClusteringParameters
                    {
                        MaxDistance = 50.0f,        // WMO-style clustering distance
                        MinComponentsPerCluster = 2,
                        MaxComponentSize = 500.0f,
                        PreserveHierarchy = true
                    };
                    
                    var clusteringResult = spatialClusteringAnalyzer.PerformSpatialClustering(spatialComponents, clusteringParams);
                    
                    // 4. Generate and save clustering report
                    var clusteringReport = spatialClusteringAnalyzer.GenerateClusteringReport(clusteringResult, fileName);
                    var clusteringReportPath = Path.Combine(outputDir, $"{baseFileName}.clustering.txt");
                    File.WriteAllText(clusteringReportPath, clusteringReport);
                    Console.WriteLine($"📋 Clustering analysis written to: {clusteringReportPath}");
                    
                    // 5. Export clustered assemblies (WMO-style grouped components)
                    Console.WriteLine("\n🏗️  EXPORTING SPATIAL CLUSTERS AS WMO-STYLE ASSEMBLIES:");
                    var spatialAssembliesDir = Path.Combine(outputDir, "spatial_assemblies");
                    Directory.CreateDirectory(spatialAssembliesDir);
                    
                    var clusterExportCount = 0;
                    var totalComponentsExported = 0;
                    
                    foreach (var cluster in clusteringResult.Clusters)
                    {
                        try
                        {
                            var clusterDir = Path.Combine(spatialAssembliesDir, cluster.ClusterName);
                            Directory.CreateDirectory(clusterDir);
                            
                            // Export cluster manifest (WMO-style)
                            var manifestPath = Path.Combine(clusterDir, "cluster_manifest.txt");
                            var manifest = GenerateClusterManifest(cluster, baseFileName);
                            File.WriteAllText(manifestPath, manifest);
                            
                            // Export individual components in the cluster
                            var componentsDir = Path.Combine(clusterDir, "components");
                            Directory.CreateDirectory(componentsDir);
                            
                            var clusterComponents = 0;
                            foreach (var component in cluster.Components)
                            {
                                var objFileName = $"component_{component.NodeIndex:D3}.obj";
                                var objPath = Path.Combine(componentsDir, objFileName);
                                
                                objectMeshExporter.ExportObjectMesh(component.SegmentResult, pm4File, objPath, renderMeshOnly: true);
                                clusterComponents++;
                            }
                            
                            Console.WriteLine($"  ✅ {cluster.Type} cluster '{cluster.ClusterName}': {clusterComponents} components");
                            clusterExportCount++;
                            totalComponentsExported += clusterComponents;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"  ❌ Failed to export cluster '{cluster.ClusterName}': {ex.Message}");
                        }
                    }
                    
                    // 6. Export standalone components that didn't cluster
                    if (clusteringResult.UnclusteredComponents.Any())
                    {
                        Console.WriteLine("\n🔧 EXPORTING STANDALONE COMPONENTS:");
                        var standaloneDir = Path.Combine(outputDir, "standalone_components");
                        Directory.CreateDirectory(standaloneDir);
                        
                        var standaloneCount = 0;
                        foreach (var component in clusteringResult.UnclusteredComponents)
                        {
                            try
                            {
                                var objFileName = $"{baseFileName}.standalone_{component.NodeIndex:D3}.obj";
                                var objPath = Path.Combine(standaloneDir, objFileName);
                                
                                objectMeshExporter.ExportObjectMesh(component.SegmentResult, pm4File, objPath, renderMeshOnly: true);
                                standaloneCount++;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"  ❌ Failed to export standalone component {component.NodeIndex}: {ex.Message}");
                            }
                        }
                        Console.WriteLine($"  ✅ Exported {standaloneCount} standalone components");
                        totalComponentsExported += standaloneCount;
                    }
                    
                    // 7. Summary
                    Console.WriteLine($"\n📊 SPATIAL CLUSTERING EXPORT COMPLETE:");
                    Console.WriteLine($"   🎯 Total Clusters: {clusterExportCount} organized assemblies");
                    Console.WriteLine($"   🔧 Total Components: {totalComponentsExported} individual geometry objects");
                    Console.WriteLine($"   📈 Clustering Efficiency: {clusteringResult.Stats.ClusteringEfficiency:P1} reduction");
                    Console.WriteLine($"   📁 Spatial assemblies: {spatialAssembliesDir}");
                    Console.WriteLine($"   💡 Each cluster contains related components grouped by spatial proximity");

                    // Generate and display Mermaid diagram
                    Console.WriteLine("\n=== MERMAID HIERARCHY DIAGRAM ===");
                    Console.WriteLine("```mermaid");
                    var mermaid = hierarchyAnalyzer.GenerateHierarchyMermaid(hierarchyResult, fileName, 15);
                    Console.WriteLine(mermaid);
                    Console.WriteLine("```");
                    Console.WriteLine();

                    // Show raw data samples for manual inspection
                    Console.WriteLine("=== RAW DATA SAMPLES FOR MANUAL INSPECTION ===");
                    var samples = pm4File.MSLK.Entries.Take(10);
                    foreach (var (entry, index) in samples.Select((e, i) => (e, i)))
                    {
                        Console.WriteLine($"Entry {index}:");
                        Console.WriteLine($"  Flags_0x00: 0x{entry.Unknown_0x00:X2} ({entry.Unknown_0x00})");
                        Console.WriteLine($"  Sequence_0x01: {entry.Unknown_0x01}");
                        Console.WriteLine($"  Unknown_0x02: 0x{entry.Unknown_0x02:X4}");
                        Console.WriteLine($"  ParentIndex_0x04: {entry.Unknown_0x04}");
                        Console.WriteLine($"  MSPI_first_index: {entry.MspiFirstIndex}");
                        Console.WriteLine($"  MSPI_index_count: {entry.MspiIndexCount}");
                        Console.WriteLine($"  Unknown_0x0C: 0x{entry.Unknown_0x0C:X8}");
                        Console.WriteLine($"  CrossRef_0x10: {entry.Unknown_0x10}");
                        Console.WriteLine($"  Constant_0x12: 0x{entry.Unknown_0x12:X4}");
                        Console.WriteLine($"  Type: {(entry.MspiFirstIndex == -1 ? "Doodad Node" : "Geometry Node")}");
                        Console.WriteLine();
                    }

                    Console.WriteLine(new string('═', 80));
                    Console.WriteLine();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Error analyzing {fileName}: {ex.Message}");
                    Console.WriteLine();
                }
            }

            Console.WriteLine("✅ MSLK spatial clustering analysis complete!");
            Console.WriteLine();
            Console.WriteLine("📋 OUTPUTS GENERATED:");
            Console.WriteLine("   📄 TXT Reports: Detailed hierarchy analysis and clustering statistics");
            Console.WriteLine("   🎯 Spatial Clusters: WMO-style grouped assemblies based on proximity");
            Console.WriteLine("   🔧 Standalone Components: Individual objects that didn't cluster");
            Console.WriteLine("   📈 Mermaid Diagrams: Visual hierarchy representations");
            Console.WriteLine();
            Console.WriteLine("🗂️ SPATIAL CLUSTERING OUTPUT STRUCTURE:");
            Console.WriteLine("   📁 spatial_assemblies/ - Clustered geometry assemblies");
            Console.WriteLine("      ├── MajorStructure_XXXcomp_YYY/ - Large architectural elements");
            Console.WriteLine("      ├── DetailCluster_XXXcomp_YYY/ - Small details grouped together");
            Console.WriteLine("      ├── LinearStructure_XXXcomp_YYY/ - Wall-like arrangements");
            Console.WriteLine("      ├── CompactCluster_XXXcomp_YYY/ - Tightly grouped components");
            Console.WriteLine("      └── Standalone_XXX/ - Individual large components");
            Console.WriteLine("          ├── cluster_manifest.txt - WMO-style component info");
            Console.WriteLine("          └── components/ - Individual OBJ files in the cluster");
            Console.WriteLine("   📁 standalone_components/ - Components that didn't cluster");
            Console.WriteLine();
            Console.WriteLine("🎯 SPATIAL CLUSTERING BENEFITS:");
            Console.WriteLine("   ✨ Intelligently groups nearby geometry components together");
            Console.WriteLine("   ✨ Reduces file count from thousands to manageable assemblies");
            Console.WriteLine("   ✨ Follows WMO-style organizational patterns");
            Console.WriteLine("   ✨ Preserves spatial relationships and component details");
            Console.WriteLine("   ✨ Each cluster contains manifests with component metadata");
            Console.WriteLine();
            Console.WriteLine("🏗️  CLUSTER TYPES EXPLAINED:");
            Console.WriteLine("   🏛️  MajorStructure: Large, important architectural elements");
            Console.WriteLine("   🔧 DetailCluster: Small details grouped by proximity");
            Console.WriteLine("   📏 LinearStructure: Wall-like or linear arrangements");
            Console.WriteLine("   📦 CompactCluster: Tightly grouped related components");
            Console.WriteLine("   🏗️  Standalone: Individual components too large/isolated to cluster");
            Console.WriteLine();
            Console.WriteLine("💡 PRO TIPS:");
            Console.WriteLine("   🎯 Import entire cluster directories to reconstruct spatial assemblies");
            Console.WriteLine("   📖 Check cluster manifests for component positions and metadata");
            Console.WriteLine("   🔧 Use clustering efficiency metric to evaluate grouping quality");
            Console.WriteLine("   🏗️  This approach mimics WMO file organization patterns!");
        }

        /// <summary>
        /// Generate WMO-style cluster manifest for spatial assemblies
        /// </summary>
        private static string GenerateClusterManifest(MslkSpatialClusteringAnalyzer.SpatialCluster cluster, string baseFileName)
        {
            var manifest = new System.Text.StringBuilder();
            
            manifest.AppendLine($"=== SPATIAL CLUSTER MANIFEST ===");
            manifest.AppendLine($"Cluster Name: {cluster.ClusterName}");
            manifest.AppendLine($"Cluster Type: {cluster.Type}");
            manifest.AppendLine($"Component Count: {cluster.Components.Count}");
            manifest.AppendLine($"Total Volume: {cluster.TotalVolume:F1}");
            manifest.AppendLine($"Density: {cluster.Density:F3}");
            manifest.AppendLine($"Centroid: ({cluster.ClusterCentroid.X:F1}, {cluster.ClusterCentroid.Y:F1}, {cluster.ClusterCentroid.Z:F1})");
            manifest.AppendLine($"Bounds: [{cluster.ClusterBoundsMin.X:F1}, {cluster.ClusterBoundsMin.Y:F1}, {cluster.ClusterBoundsMin.Z:F1}] to [{cluster.ClusterBoundsMax.X:F1}, {cluster.ClusterBoundsMax.Y:F1}, {cluster.ClusterBoundsMax.Z:F1}]");
            manifest.AppendLine($"Generated: {DateTime.Now}");
            manifest.AppendLine();
            
            manifest.AppendLine("=== COMPONENT LIST ===");
            foreach (var component in cluster.Components.OrderBy(c => c.NodeIndex))
            {
                manifest.AppendLine($"Component {component.NodeIndex:D3}:");
                manifest.AppendLine($"  File: component_{component.NodeIndex:D3}.obj");
                manifest.AppendLine($"  Position: ({component.Centroid.X:F1}, {component.Centroid.Y:F1}, {component.Centroid.Z:F1})");
                manifest.AppendLine($"  Volume: {component.Volume:F1}");
                manifest.AppendLine($"  Vertices: {component.VertexCount}");
                manifest.AppendLine($"  Triangles: {component.TriangleCount}");
                manifest.AppendLine();
            }
            
            manifest.AppendLine("=== USAGE INSTRUCTIONS ===");
            manifest.AppendLine("1. This cluster represents spatially related geometry components");
            manifest.AppendLine("2. Individual components are in the 'components/' subdirectory");
            manifest.AppendLine("3. Import all components together to reconstruct the spatial assembly");
            manifest.AppendLine("4. Use the position data to correctly place components in 3D space");
            manifest.AppendLine("5. This follows WMO-style grouping where related objects are clustered together");
            
            return manifest.ToString();
        }

        /// <summary>
        /// Quick analysis of a single PM4 file - useful for focused testing
        /// </summary>
        public static void AnalyzeSingleFile(string pm4FilePath)
        {
            if (!File.Exists(pm4FilePath))
            {
                Console.WriteLine($"❌ File not found: {pm4FilePath}");
                return;
            }

            var fileName = Path.GetFileName(pm4FilePath);
            Console.WriteLine($"🎯 SINGLE FILE ANALYSIS: {fileName}");
            Console.WriteLine(new string('═', 60));

            try
            {
                var pm4File = PM4File.FromFile(pm4FilePath);

                if (pm4File.MSLK?.Entries == null)
                {
                    Console.WriteLine("❌ No MSLK chunk found");
                    return;
                }

                var hierarchyAnalyzer = new MslkHierarchyAnalyzer();
                var result = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);

                // Quick summary
                Console.WriteLine($"📊 QUICK SUMMARY:");
                Console.WriteLine($"   Total Nodes: {result.AllNodes.Count}");
                Console.WriteLine($"   Root Nodes: {result.RootNodes.Count}");
                Console.WriteLine($"   Max Depth: {result.MaxDepth}");
                Console.WriteLine($"   Geometry Nodes: {result.GeometryNodeCount}");
                Console.WriteLine($"   Doodad Nodes: {result.DoodadNodeCount}");
                Console.WriteLine();

                // ✨ ENHANCED: Export all strategies for single file analysis
                Console.WriteLine("\n📊 MULTI-STRATEGY OBJECT EXPORT:");
                var allStrategies = hierarchyAnalyzer.SegmentByAllStrategies(result);
                var objectMeshExporter = new MslkObjectMeshExporter();
                var outputDir = Path.Combine("output", "single_analysis");
                Directory.CreateDirectory(outputDir);
                var baseFileName = Path.GetFileNameWithoutExtension(fileName);
                
                // Export organized multi-strategy
                objectMeshExporter.ExportAllStrategiesOrganized(allStrategies, pm4File, outputDir, baseFileName);
                Console.WriteLine($"Multi-strategy meshes written to: {outputDir}");

                // Mermaid diagram
                Console.WriteLine("\n```mermaid");
                var mermaid = hierarchyAnalyzer.GenerateHierarchyMermaid(result, fileName, 20);
                Console.WriteLine(mermaid);
                Console.WriteLine("```");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }
        }
    }
} 