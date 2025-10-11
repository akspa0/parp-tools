using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Exports PM4 MSLK hierarchy as hierarchical model assemblies.
    /// Treats individual geometry nodes as sub-components that assemble into larger WMO models.
    /// </summary>
    public class MslkModelAssemblyExporter
    {
        /// <summary>
        /// Model Assembly represents a hierarchical WMO structure with sub-components
        /// </summary>
        public class ModelAssembly
        {
            public int RootNodeIndex { get; set; }
            public string AssemblyName { get; set; } = "";
            public List<SubComponent> Components { get; set; } = new();
            public MslkObjectMeshExporter.ObjectBoundingBox? BoundingBox { get; set; }
            public int TotalVertices { get; set; }
            public int TotalTriangles { get; set; }
            public ModelAssemblyType AssemblyType { get; set; }
        }

        /// <summary>
        /// Sub-component within a model assembly
        /// </summary>
        public class SubComponent
        {
            public int NodeIndex { get; set; }
            public string ComponentName { get; set; } = "";
            public string ObjFileName { get; set; } = "";
            public int ParentIndex { get; set; }
            public int Depth { get; set; }
            public ComponentType Type { get; set; }
            public List<SubComponent> Children { get; set; } = new();
            public MslkObjectMeshExporter.ObjectBoundingBox? LocalBounds { get; set; }
            public int VertexCount { get; set; }
            public int TriangleCount { get; set; }
        }

        public enum ModelAssemblyType
        {
            PrimaryStructure,    // Self-referencing root objects
            ComplexAssembly,     // Multi-level hierarchy
            SimpleGroup,         // Flat grouping
            Standalone          // Individual components
        }

        public enum ComponentType
        {
            GeometryNode,       // Renderable geometry
            DoodadNode,         // Decoration/detail objects
            StructuralElement,  // Architectural components
            DetailElement       // Fine detail/trim
        }

        /// <summary>
        /// Analyze MSLK hierarchy and create model assemblies from individual components
        /// </summary>
        public List<ModelAssembly> CreateModelAssemblies(MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult, string baseFileName)
        {
            var assemblies = new List<ModelAssembly>();

            // Identify root assemblies (self-referencing nodes)
            var rootNodes = IdentifyRootAssemblies(hierarchyResult);
            
            // Build primary assemblies from root nodes
            foreach (var rootNode in rootNodes)
            {
                var assembly = CreatePrimaryAssembly(rootNode, hierarchyResult, baseFileName);
                assemblies.Add(assembly);
            }

            // Group remaining nodes into logical assemblies by parent relationships
            var ungroupedNodes = GetUngroupedNodes(hierarchyResult, rootNodes);
            var secondaryAssemblies = CreateSecondaryAssemblies(ungroupedNodes, hierarchyResult, baseFileName);
            assemblies.AddRange(secondaryAssemblies);

            return assemblies;
        }

        /// <summary>
        /// Export model assemblies with hierarchical structure
        /// </summary>
        public void ExportModelAssemblies(List<ModelAssembly> assemblies, PM4File pm4File, string outputDir, string baseFileName)
        {
            var assemblyDir = Path.Combine(outputDir, "model_assemblies");
            Directory.CreateDirectory(assemblyDir);

            var objectMeshExporter = new MslkObjectMeshExporter();

            Console.WriteLine($"\n🏗️  EXPORTING MODEL ASSEMBLIES ({assemblies.Count} assemblies)");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");

            // Export each model assembly
            foreach (var assembly in assemblies)
            {
                var assemblySubDir = Path.Combine(assemblyDir, assembly.AssemblyName);
                Directory.CreateDirectory(assemblySubDir);

                // Export assembly manifest
                ExportAssemblyManifest(assembly, assemblySubDir);

                // Export component hierarchy documentation
                ExportComponentHierarchy(assembly, assemblySubDir);

                // Export individual sub-components
                ExportSubComponents(assembly, pm4File, objectMeshExporter, assemblySubDir, baseFileName);

                Console.WriteLine($"📦 {assembly.AssemblyName} ({assembly.Components.Count} components, {assembly.AssemblyType})");
            }

            // Export assembly overview
            ExportAssemblyOverview(assemblies, assemblyDir, baseFileName);
            
            Console.WriteLine($"\n✅ Model assemblies exported to: {assemblyDir}");
        }

        private List<MslkHierarchyAnalyzer.HierarchyNode> IdentifyRootAssemblies(MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult)
        {
            // Find self-referencing nodes (primary model roots)
            return hierarchyResult.AllNodes
                .Where(node => node.ParentIndex_0x04 == node.Index)
                .ToList();
        }

        private ModelAssembly CreatePrimaryAssembly(MslkHierarchyAnalyzer.HierarchyNode rootNode, MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult, string baseFileName)
        {
            var assembly = new ModelAssembly
            {
                RootNodeIndex = rootNode.Index,
                AssemblyName = $"{baseFileName}_Assembly_{rootNode.Index:D3}",
                AssemblyType = ModelAssemblyType.PrimaryStructure
            };

            // Build component tree from this root
            var rootComponent = CreateSubComponent(rootNode, 0);
            assembly.Components.Add(rootComponent);

            // Add child components recursively
            BuildComponentTree(rootComponent, rootNode, hierarchyResult, 1);

            // Calculate assembly bounds and stats
            CalculateAssemblyStats(assembly, hierarchyResult);

            return assembly;
        }

        private void BuildComponentTree(SubComponent parentComponent, MslkHierarchyAnalyzer.HierarchyNode parentNode, MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult, int depth)
        {
            // Find all nodes that reference this parent (excluding self-references)
            var children = hierarchyResult.AllNodes
                .Where(node => node.ParentIndex_0x04 == parentNode.Index && node.Index != parentNode.Index)
                .ToList();

            foreach (var childNode in children)
            {
                var childComponent = CreateSubComponent(childNode, depth);
                parentComponent.Children.Add(childComponent);

                // Recursively build deeper levels (limit depth to prevent infinite recursion)
                if (depth < 5)
                {
                    BuildComponentTree(childComponent, childNode, hierarchyResult, depth + 1);
                }
            }
        }

        private SubComponent CreateSubComponent(MslkHierarchyAnalyzer.HierarchyNode node, int depth)
        {
            return new SubComponent
            {
                NodeIndex = node.Index,
                ComponentName = $"Component_{node.Index:D3}",
                ObjFileName = $"geom_{node.Index:D3}.obj",
                ParentIndex = (int)node.ParentIndex_0x04,
                Depth = depth,
                Type = node.HasGeometry ? ComponentType.GeometryNode : ComponentType.DoodadNode,
                LocalBounds = null // Initialize as nullable
            };
        }

        private List<MslkHierarchyAnalyzer.HierarchyNode> GetUngroupedNodes(MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult, List<MslkHierarchyAnalyzer.HierarchyNode> rootNodes)
        {
            var processedNodes = new HashSet<int>();
            
            // Mark all nodes that are part of primary assemblies
            foreach (var rootNode in rootNodes)
            {
                MarkProcessedNodes(rootNode, hierarchyResult, processedNodes);
            }
            
            // Return nodes that haven't been processed
            return hierarchyResult.AllNodes
                .Where(node => !processedNodes.Contains(node.Index))
                .ToList();
        }

        private void MarkProcessedNodes(MslkHierarchyAnalyzer.HierarchyNode rootNode, MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult, HashSet<int> processedNodes)
        {
            if (processedNodes.Contains(rootNode.Index))
                return;
                
            processedNodes.Add(rootNode.Index);
            
            // Mark all children recursively
            var children = hierarchyResult.AllNodes
                .Where(node => node.ParentIndex_0x04 == rootNode.Index && node.Index != rootNode.Index)
                .ToList();
                
            foreach (var child in children)
            {
                MarkProcessedNodes(child, hierarchyResult, processedNodes);
            }
        }

        private List<ModelAssembly> CreateSecondaryAssemblies(List<MslkHierarchyAnalyzer.HierarchyNode> ungroupedNodes, MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult, string baseFileName)
        {
            var assemblies = new List<ModelAssembly>();

            // Group nodes by common parents
            var groupedNodes = ungroupedNodes
                .GroupBy(node => node.ParentIndex_0x04)
                .Where(group => group.Count() >= 2) // Only groups with multiple children
                .ToList();

            foreach (var group in groupedNodes)
            {
                var assembly = new ModelAssembly
                {
                    RootNodeIndex = (int)group.Key,
                    AssemblyName = $"{baseFileName}_Group_{group.Key:D3}",
                    AssemblyType = ModelAssemblyType.ComplexAssembly
                };

                foreach (var node in group)
                {
                    var component = CreateSubComponent(node, 0);
                    assembly.Components.Add(component);
                }

                CalculateAssemblyStats(assembly, hierarchyResult);
                assemblies.Add(assembly);
            }

            return assemblies;
        }

        private void CalculateAssemblyStats(ModelAssembly assembly, MslkHierarchyAnalyzer.HierarchyAnalysisResult hierarchyResult)
        {
            // Calculate total vertices and triangles (placeholder - would need actual geometry data)
            assembly.TotalVertices = assembly.Components.Count * 8; // Rough estimate
            assembly.TotalTriangles = assembly.Components.Count * 4; // Rough estimate

            // Calculate combined bounding box
            if (assembly.Components.Any())
            {
                var geometryComponents = assembly.Components.Where(c => c.Type == ComponentType.GeometryNode).ToList();
                if (geometryComponents.Any())
                {
                    var bounds = geometryComponents.First().LocalBounds;
                    foreach (var component in geometryComponents.Skip(1))
                    {
                        bounds = CombineBounds(bounds, component.LocalBounds);
                    }
                    assembly.BoundingBox = bounds;
                }
            }
        }

        private MslkObjectMeshExporter.ObjectBoundingBox CombineBounds(MslkObjectMeshExporter.ObjectBoundingBox a, MslkObjectMeshExporter.ObjectBoundingBox b)
        {
            return new MslkObjectMeshExporter.ObjectBoundingBox
            {
                Min = new Vector3(
                    Math.Min(a.Min.X, b.Min.X),
                    Math.Min(a.Min.Y, b.Min.Y),
                    Math.Min(a.Min.Z, b.Min.Z)
                ),
                Max = new Vector3(
                    Math.Max(a.Max.X, b.Max.X),
                    Math.Max(a.Max.Y, b.Max.Y),
                    Math.Max(a.Max.Z, b.Max.Z)
                )
            };
        }

        private void ExportAssemblyManifest(ModelAssembly assembly, string outputDir)
        {
            var manifestPath = Path.Combine(outputDir, "assembly_manifest.txt");
            var content = new StringBuilder();

            content.AppendLine($"=== MODEL ASSEMBLY MANIFEST ===");
            content.AppendLine($"Assembly Name: {assembly.AssemblyName}");
            content.AppendLine($"Root Node: {assembly.RootNodeIndex}");
            content.AppendLine($"Assembly Type: {assembly.AssemblyType}");
            content.AppendLine($"Component Count: {assembly.Components.Count}");
            content.AppendLine($"Total Vertices: {assembly.TotalVertices}");
            content.AppendLine($"Total Triangles: {assembly.TotalTriangles}");
            content.AppendLine($"Generated: {DateTime.Now}");
            content.AppendLine();

            content.AppendLine("=== COMPONENT HIERARCHY ===");
            foreach (var component in assembly.Components)
            {
                WriteComponentHierarchy(content, component, 0);
            }

            content.AppendLine();
            content.AppendLine("=== USAGE INSTRUCTIONS ===");
            content.AppendLine("1. Individual components are in the 'components/' subdirectory");
            content.AppendLine("2. Each component is a separate OBJ file for detailed analysis");
            content.AppendLine("3. Use the hierarchy information to understand relationships");
            content.AppendLine("4. Import components into 3D software to reconstruct the full model");

            File.WriteAllText(manifestPath, content.ToString());
        }

        private void WriteComponentHierarchy(StringBuilder content, SubComponent component, int indent)
        {
            var indentStr = new string(' ', indent * 2);
            content.AppendLine($"{indentStr}├─ {component.ComponentName} ({component.Type})");
            content.AppendLine($"{indentStr}│  ├─ OBJ File: {component.ObjFileName}");
            content.AppendLine($"{indentStr}│  ├─ Parent Node: {component.ParentIndex}");
            content.AppendLine($"{indentStr}│  └─ Hierarchy Depth: {component.Depth}");

            foreach (var child in component.Children)
            {
                WriteComponentHierarchy(content, child, indent + 1);
            }
        }

        private void ExportComponentHierarchy(ModelAssembly assembly, string outputDir)
        {
            var hierarchyPath = Path.Combine(outputDir, "component_hierarchy.md");
            var content = new StringBuilder();

            content.AppendLine($"# {assembly.AssemblyName} - Component Hierarchy");
            content.AppendLine();
            content.AppendLine($"**Assembly Type**: {assembly.AssemblyType}");
            content.AppendLine($"**Root Node**: {assembly.RootNodeIndex}");
            content.AppendLine($"**Total Components**: {assembly.Components.Count}");
            content.AppendLine();

            content.AppendLine("## Hierarchical Structure");
            content.AppendLine();
            content.AppendLine("```mermaid");
            content.AppendLine("graph TD");

            foreach (var component in assembly.Components)
            {
                WriteMermaidComponent(content, component);
            }

            content.AppendLine("```");
            content.AppendLine();

            content.AppendLine("## Component Details");
            content.AppendLine();
            foreach (var component in assembly.Components)
            {
                WriteMarkdownComponentDetails(content, component, 0);
            }

            File.WriteAllText(hierarchyPath, content.ToString());
        }

        private void WriteMermaidComponent(StringBuilder content, SubComponent component)
        {
            var nodeId = $"N{component.NodeIndex}";
            var nodeLabel = $"{component.ComponentName}<br/>{component.Type}";
            content.AppendLine($"    {nodeId}[\"{nodeLabel}\"]");

            foreach (var child in component.Children)
            {
                var childId = $"N{child.NodeIndex}";
                content.AppendLine($"    {nodeId} --> {childId}");
                WriteMermaidComponent(content, child);
            }
        }

        private void WriteMarkdownComponentDetails(StringBuilder content, SubComponent component, int depth)
        {
            var indent = new string(' ', depth * 2);
            content.AppendLine($"{indent}- **{component.ComponentName}** ({component.Type})");
            content.AppendLine($"{indent}  - **OBJ File**: `{component.ObjFileName}`");
            content.AppendLine($"{indent}  - **Parent Node**: {component.ParentIndex}");
            content.AppendLine($"{indent}  - **Hierarchy Depth**: {component.Depth}");

            foreach (var child in component.Children)
            {
                WriteMarkdownComponentDetails(content, child, depth + 1);
            }
        }

        private void ExportSubComponents(ModelAssembly assembly, PM4File pm4File, MslkObjectMeshExporter objectMeshExporter, string outputDir, string baseFileName)
        {
            var componentsDir = Path.Combine(outputDir, "components");
            Directory.CreateDirectory(componentsDir);

            // Export each sub-component as individual OBJ in the assembly structure
            foreach (var component in assembly.Components)
            {
                if (component.Type == ComponentType.GeometryNode)
                {
                    try
                    {
                        var segmentResult = new MslkHierarchyAnalyzer.ObjectSegmentationResult
                        {
                            RootIndex = component.NodeIndex,
                            GeometryNodeIndices = new List<int> { component.NodeIndex },
                            DoodadNodeIndices = new List<int>(),
                            SegmentationType = "individual_geometry"
                        };

                        var objPath = Path.Combine(componentsDir, component.ObjFileName);
                        objectMeshExporter.ExportObjectMesh(segmentResult, pm4File, objPath, renderMeshOnly: true);

                        Console.WriteLine($"  └─ Component {component.NodeIndex}: {component.ObjFileName}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  ❌ Failed to export component {component.NodeIndex}: {ex.Message}");
                    }
                }
            }
        }

        private void ExportAssemblyOverview(List<ModelAssembly> assemblies, string outputDir, string baseFileName)
        {
            var overviewPath = Path.Combine(outputDir, "assembly_overview.md");
            var content = new StringBuilder();

            content.AppendLine($"# {baseFileName} - Model Assembly Overview");
            content.AppendLine();
            content.AppendLine($"**Generated**: {DateTime.Now}");
            content.AppendLine($"**Total Assemblies**: {assemblies.Count}");
            content.AppendLine($"**Total Components**: {assemblies.Sum(a => a.Components.Count)}");
            content.AppendLine();

            content.AppendLine("## Assembly Summary");
            content.AppendLine();
            content.AppendLine("| Assembly | Type | Components | Root Node |");
            content.AppendLine("|----------|------|------------|-----------|");

            foreach (var assembly in assemblies)
            {
                content.AppendLine($"| {assembly.AssemblyName} | {assembly.AssemblyType} | {assembly.Components.Count} | {assembly.RootNodeIndex} |");
            }

            content.AppendLine();
            content.AppendLine("## Hierarchical Model Structure");
            content.AppendLine();
            content.AppendLine("This PM4 file contains hierarchical WMO (World Map Object) data where:");
            content.AppendLine("- **Primary Structures**: Self-referencing root nodes that define main model assemblies");
            content.AppendLine("- **Complex Assemblies**: Multi-component groups with parent-child relationships");
            content.AppendLine("- **Sub-Components**: Individual geometry pieces that combine to form larger structures");
            content.AppendLine();

            content.AppendLine("### Assembly Details");
            content.AppendLine();

            foreach (var assembly in assemblies)
            {
                content.AppendLine($"#### {assembly.AssemblyName}");
                content.AppendLine();
                content.AppendLine($"- **Type**: {assembly.AssemblyType}");
                content.AppendLine($"- **Root Node**: {assembly.RootNodeIndex}");
                content.AppendLine($"- **Components**: {assembly.Components.Count}");
                content.AppendLine($"- **Geometry Nodes**: {assembly.Components.Count(c => c.Type == ComponentType.GeometryNode)}");
                content.AppendLine($"- **Doodad Nodes**: {assembly.Components.Count(c => c.Type == ComponentType.DoodadNode)}");
                content.AppendLine();

                if (assembly.Components.Any())
                {
                    content.AppendLine("**Component Tree:**");
                    foreach (var component in assembly.Components.Take(1)) // Show structure for first component
                    {
                        WriteMarkdownComponentTree(content, component, 0);
                    }
                    if (assembly.Components.Count > 1)
                    {
                        content.AppendLine($"  ... and {assembly.Components.Count - 1} more components");
                    }
                    content.AppendLine();
                }
            }

            content.AppendLine("## Usage Instructions");
            content.AppendLine();
            content.AppendLine("1. **Individual Analysis**: Use the `individual_objects/` folder for analyzing single components");
            content.AppendLine("2. **Assembly Analysis**: Use the `model_assemblies/` folder to understand hierarchical relationships");
            content.AppendLine("3. **Component Relationships**: Check `assembly_manifest.txt` files for detailed hierarchy information");
            content.AppendLine("4. **3D Reconstruction**: Import component OBJ files using hierarchy data to reconstruct full models");

            File.WriteAllText(overviewPath, content.ToString());
        }

        private void WriteMarkdownComponentTree(StringBuilder content, SubComponent component, int depth)
        {
            var indent = new string(' ', depth * 2);
            content.AppendLine($"{indent}- **{component.ComponentName}** ({component.Type})");
            content.AppendLine($"{indent}  - File: `{component.ObjFileName}`");

            foreach (var child in component.Children.Take(3)) // Limit to prevent huge output
            {
                WriteMarkdownComponentTree(content, child, depth + 1);
            }
            if (component.Children.Count > 3)
            {
                content.AppendLine($"{indent}  ... and {component.Children.Count - 3} more child components");
            }
        }
    }
}
