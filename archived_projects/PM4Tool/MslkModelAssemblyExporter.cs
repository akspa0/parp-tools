using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
            public BoundingBox3D BoundingBox { get; set; }
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
            public BoundingBox3D LocalBounds { get; set; }
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
        public List<ModelAssembly> CreateModelAssemblies(MslkHierarchyResult hierarchyResult, string baseFileName)
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

            // Group remaining nodes into logical assemblies
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

            // Export each model assembly
            foreach (var assembly in assemblies)
            {
                var assemblySubDir = Path.Combine(assemblyDir, assembly.AssemblyName);
                Directory.CreateDirectory(assemblySubDir);

                // Export assembly manifest
                ExportAssemblyManifest(assembly, assemblySubDir);

                // Export component hierarchy
                ExportComponentHierarchy(assembly, assemblySubDir);

                // Export combined assembly OBJ
                ExportCombinedAssembly(assembly, pm4File, objectMeshExporter, assemblySubDir, baseFileName);

                // Export individual sub-components
                ExportSubComponents(assembly, pm4File, objectMeshExporter, assemblySubDir, baseFileName);

                Console.WriteLine($"📦 Model Assembly: {assembly.AssemblyName} ({assembly.Components.Count} components)");
            }

            // Export assembly overview
            ExportAssemblyOverview(assemblies, assemblyDir, baseFileName);
        }

        private List<HierarchyNode> IdentifyRootAssemblies(MslkHierarchyResult hierarchyResult)
        {
            return hierarchyResult.AllNodes
                .Where(node => IsSelfReferencing(node) || IsLikelyRoot(node))
                .ToList();
        }

        private bool IsSelfReferencing(HierarchyNode node)
        {
            return node.ParentIndex == node.Index;
        }

        private bool IsLikelyRoot(HierarchyNode node)
        {
            // Identify potential root nodes based on patterns
            return node.Flags == 0x02 && node.MslkEntry.Unknown_0x01 == 0;
        }

        private ModelAssembly CreatePrimaryAssembly(HierarchyNode rootNode, MslkHierarchyResult hierarchyResult, string baseFileName)
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
            CalculateAssemblyStats(assembly);

            return assembly;
        }

        private void BuildComponentTree(SubComponent parentComponent, HierarchyNode parentNode, MslkHierarchyResult hierarchyResult, int depth)
        {
            var children = hierarchyResult.AllNodes
                .Where(node => node.ParentIndex == parentNode.Index && node.Index != parentNode.Index)
                .ToList();

            foreach (var childNode in children)
            {
                var childComponent = CreateSubComponent(childNode, depth);
                parentComponent.Children.Add(childComponent);

                // Recursively build deeper levels (limit depth to prevent cycles)
                if (depth < 10)
                {
                    BuildComponentTree(childComponent, childNode, hierarchyResult, depth + 1);
                }
            }
        }

        private SubComponent CreateSubComponent(HierarchyNode node, int depth)
        {
            return new SubComponent
            {
                NodeIndex = node.Index,
                ComponentName = $"Component_{node.Index:D3}",
                ObjFileName = $"geom_{node.Index:D3}.obj",
                ParentIndex = node.ParentIndex,
                Depth = depth,
                Type = node.IsGeometryNode ? ComponentType.GeometryNode : ComponentType.DoodadNode,
                LocalBounds = node.BoundingBox
            };
        }

        private List<HierarchyNode> GetUngroupedNodes(MslkHierarchyResult hierarchyResult, List<HierarchyNode> rootNodes)
        {
            var rootIndices = rootNodes.Select(n => n.Index).ToHashSet();
            return hierarchyResult.AllNodes
                .Where(node => !rootIndices.Contains(node.Index) && !HasRootParent(node, rootIndices))
                .ToList();
        }

        private bool HasRootParent(HierarchyNode node, HashSet<int> rootIndices)
        {
            return rootIndices.Contains(node.ParentIndex);
        }

        private List<ModelAssembly> CreateSecondaryAssemblies(List<HierarchyNode> ungroupedNodes, MslkHierarchyResult hierarchyResult, string baseFileName)
        {
            var assemblies = new List<ModelAssembly>();

            // Group nodes by common parents or patterns
            var groupedNodes = GroupNodesByPattern(ungroupedNodes);

            foreach (var group in groupedNodes)
            {
                var assembly = new ModelAssembly
                {
                    RootNodeIndex = group.Key,
                    AssemblyName = $"{baseFileName}_Group_{group.Key:D3}",
                    AssemblyType = ModelAssemblyType.ComplexAssembly
                };

                foreach (var node in group.Value)
                {
                    var component = CreateSubComponent(node, 0);
                    assembly.Components.Add(component);
                }

                CalculateAssemblyStats(assembly);
                assemblies.Add(assembly);
            }

            return assemblies;
        }

        private Dictionary<int, List<HierarchyNode>> GroupNodesByPattern(List<HierarchyNode> nodes)
        {
            return nodes
                .GroupBy(node => node.ParentIndex)
                .Where(group => group.Count() > 1) // Only groups with multiple children
                .ToDictionary(group => group.Key, group => group.ToList());
        }

        private void CalculateAssemblyStats(ModelAssembly assembly)
        {
            assembly.TotalVertices = assembly.Components.Sum(c => c.VertexCount);
            assembly.TotalTriangles = assembly.Components.Sum(c => c.TriangleCount);

            // Calculate combined bounding box
            if (assembly.Components.Any())
            {
                var bounds = assembly.Components.First().LocalBounds;
                foreach (var component in assembly.Components.Skip(1))
                {
                    bounds = CombineBounds(bounds, component.LocalBounds);
                }
                assembly.BoundingBox = bounds;
            }
        }

        private BoundingBox3D CombineBounds(BoundingBox3D a, BoundingBox3D b)
        {
            return new BoundingBox3D
            {
                Min = new Vector3D
                {
                    X = Math.Min(a.Min.X, b.Min.X),
                    Y = Math.Min(a.Min.Y, b.Min.Y),
                    Z = Math.Min(a.Min.Z, b.Min.Z)
                },
                Max = new Vector3D
                {
                    X = Math.Max(a.Max.X, b.Max.X),
                    Y = Math.Max(a.Max.Y, b.Max.Y),
                    Z = Math.Max(a.Max.Z, b.Max.Z)
                }
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
            content.AppendLine();

            content.AppendLine("=== COMPONENT HIERARCHY ===");
            foreach (var component in assembly.Components)
            {
                WriteComponentHierarchy(content, component, 0);
            }

            File.WriteAllText(manifestPath, content.ToString());
        }

        private void WriteComponentHierarchy(StringBuilder content, SubComponent component, int indent)
        {
            var indentStr = new string(' ', indent * 2);
            content.AppendLine($"{indentStr}├─ {component.ComponentName} ({component.Type})");
            content.AppendLine($"{indentStr}   └─ OBJ: {component.ObjFileName}");
            content.AppendLine($"{indentStr}   └─ Parent: {component.ParentIndex}");

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
            content.AppendLine("```mermaid");
            content.AppendLine("graph TD");

            foreach (var component in assembly.Components)
            {
                WriteMermaidComponent(content, component);
            }

            content.AppendLine("```");

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

        private void ExportCombinedAssembly(ModelAssembly assembly, PM4File pm4File, MslkObjectMeshExporter objectMeshExporter, string outputDir, string baseFileName)
        {
            // Create a combined OBJ that references all sub-components
            var combinedPath = Path.Combine(outputDir, $"{assembly.AssemblyName}_combined.obj");
            var content = new StringBuilder();

            content.AppendLine($"# Model Assembly: {assembly.AssemblyName}");
            content.AppendLine($"# Combined model with {assembly.Components.Count} sub-components");
            content.AppendLine($"# Generated: {DateTime.Now}");
            content.AppendLine();

            var vertexOffset = 1;
            foreach (var component in assembly.Components)
            {
                if (component.Type == ComponentType.GeometryNode)
                {
                    content.AppendLine($"# Component: {component.ComponentName}");
                    content.AppendLine($"# Source: {component.ObjFileName}");
                    content.AppendLine($"g {component.ComponentName}");
                    content.AppendLine();

                    // Export component geometry inline or reference external file
                    content.AppendLine($"# usemtl component_{component.NodeIndex}");
                    content.AppendLine($"# Vertices and faces would be included here");
                    content.AppendLine($"# Or use 'call {component.ObjFileName}' for external reference");
                    content.AppendLine();
                }
            }

            File.WriteAllText(combinedPath, content.ToString());
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
                        var segmentResult = new SegmentedObject
                        {
                            RootIndex = component.NodeIndex,
                            GeometryNodeIndices = new List<int> { component.NodeIndex },
                            DoodadNodeIndices = new List<int>(),
                            SegmentationType = SegmentationType.IndividualGeometry
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
            content.AppendLine($"Generated: {DateTime.Now}");
            content.AppendLine($"Total Assemblies: {assemblies.Count}");
            content.AppendLine();

            content.AppendLine("## Assembly Summary");
            content.AppendLine();
            content.AppendLine("| Assembly | Type | Components | Vertices | Triangles |");
            content.AppendLine("|----------|------|------------|----------|-----------|");

            foreach (var assembly in assemblies)
            {
                content.AppendLine($"| {assembly.AssemblyName} | {assembly.AssemblyType} | {assembly.Components.Count} | {assembly.TotalVertices} | {assembly.TotalTriangles} |");
            }

            content.AppendLine();
            content.AppendLine("## Hierarchical Structure");
            content.AppendLine();

            foreach (var assembly in assemblies)
            {
                content.AppendLine($"### {assembly.AssemblyName}");
                content.AppendLine();
                content.AppendLine($"- **Type**: {assembly.AssemblyType}");
                content.AppendLine($"- **Root Node**: {assembly.RootNodeIndex}");
                content.AppendLine($"- **Components**: {assembly.Components.Count}");
                content.AppendLine();

                content.AppendLine("**Component Tree:**");
                foreach (var component in assembly.Components)
                {
                    WriteMarkdownComponentTree(content, component, 0);
                }
                content.AppendLine();
            }

            File.WriteAllText(overviewPath, content.ToString());
        }

        private void WriteMarkdownComponentTree(StringBuilder content, SubComponent component, int depth)
        {
            var indent = new string(' ', depth * 2);
            content.AppendLine($"{indent}- **{component.ComponentName}** ({component.Type})");
            content.AppendLine($"{indent}  - File: `{component.ObjFileName}`");
            content.AppendLine($"{indent}  - Parent: {component.ParentIndex}");

            foreach (var child in component.Children)
            {
                WriteMarkdownComponentTree(content, child, depth + 1);
            }
        }
    }
} 