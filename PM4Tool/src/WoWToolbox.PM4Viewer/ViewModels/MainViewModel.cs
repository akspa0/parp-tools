using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HelixToolkit.Wpf;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using WoWToolbox.PM4Viewer.Services;

namespace WoWToolbox.PM4Viewer.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private readonly PM4StructuralAnalyzer _analyzer;
        
        [ObservableProperty]
        private string? _loadedFileName;
        
        [ObservableProperty] 
        private PM4File? _pm4File;
        
        [ObservableProperty]
        private Model3DGroup _sceneModel = new();
        
        [ObservableProperty]
        private ObservableCollection<ChunkVisualizationItem> _chunkItems = new();
        
        [ObservableProperty]
        private ObservableCollection<StructuralInsight> _structuralInsights = new();
        
        [ObservableProperty]
        private string _analysisOutput = string.Empty;
        
        [ObservableProperty]
        private bool _showMSVTVertices = true;
        
        [ObservableProperty]
        private bool _showMSCNPoints = true;
        
        [ObservableProperty]
        private bool _showMSPVVertices = true;
        
        [ObservableProperty]
        private bool _showConnections = false;
        
        [ObservableProperty]
        private bool _showNodeHierarchy = false;

        public MainViewModel()
        {
            _analyzer = new PM4StructuralAnalyzer();
            LoadFileCommand = new AsyncRelayCommand(LoadFileAsync);
            ExportAnalysisCommand = new AsyncRelayCommand(ExportAnalysisAsync);
            ToggleChunkVisibilityCommand = new RelayCommand<string>(ToggleChunkVisibility);
        }

        public IAsyncRelayCommand LoadFileCommand { get; }
        public IAsyncRelayCommand ExportAnalysisCommand { get; }
        public IRelayCommand<string> ToggleChunkVisibilityCommand { get; }

        private async Task LoadFileAsync()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "PM4 Files (*.pm4)|*.pm4|All Files (*.*)|*.*",
                Title = "Select PM4 File"
            };

            if (dialog.ShowDialog() == true)
            {
                try
                {
                    await LoadPM4FileAsync(dialog.FileName);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error loading file: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private async Task LoadPM4FileAsync(string filePath)
        {
            LoadedFileName = Path.GetFileName(filePath);
            
            try
            {
                // Load PM4 file
                Pm4File = PM4File.FromFile(filePath);
                
                // Update visualization first
                UpdateVisualization();
                
                // Perform structural analysis with error handling
                var analysisResult = await Task.Run(() => 
                {
                    try
                    {
                        return _analyzer.AnalyzeFile(filePath);
                    }
                    catch (Exception ex)
                    {
                        // Create error result for debugging
                        var errorResult = new PM4StructuralAnalyzer.StructuralAnalysisResult
                        {
                            FileName = Path.GetFileName(filePath)
                        };
                        errorResult.Metadata["AnalysisError"] = ex.Message;
                        errorResult.Metadata["StackTrace"] = ex.StackTrace ?? "No stack trace";
                        return errorResult;
                    }
                });
                
                // Update analysis insights
                UpdateStructuralInsights(analysisResult);
                
                // Generate analysis report
                GenerateAnalysisReport(analysisResult);
            }
            catch (Exception ex)
            {
                // Create basic analysis with error info
                var errorResult = new PM4StructuralAnalyzer.StructuralAnalysisResult
                {
                    FileName = Path.GetFileName(filePath)
                };
                errorResult.Metadata["LoadError"] = ex.Message;
                
                UpdateStructuralInsights(errorResult);
                GenerateAnalysisReport(errorResult);
            }
        }

        private void UpdateVisualization()
        {
            if (Pm4File == null) return;

            var newScene = new Model3DGroup();
            ChunkItems.Clear();

            // Visualize MSVT render vertices
            if (Pm4File.MSVT?.Vertices != null && ShowMSVTVertices)
            {
                var msvtModel = CreateVertexVisualization(
                    Pm4File.MSVT.Vertices.Select(v => new Point3D(v.Y, v.X, v.Z)),
                    Colors.Blue,
                    "MSVT Render Vertices"
                );
                newScene.Children.Add(msvtModel);
                
                ChunkItems.Add(new ChunkVisualizationItem
                {
                    Name = "MSVT Render Vertices",
                    Count = Pm4File.MSVT.Vertices.Count,
                    Color = Colors.Blue,
                    IsVisible = true
                });
            }

            // Visualize MSCN collision points
            if (Pm4File.MSCN?.ExteriorVertices != null && ShowMSCNPoints)
            {
                var mscnModel = CreateVertexVisualization(
                    Pm4File.MSCN.ExteriorVertices.Select(v => new Point3D(v.X, -v.Y, v.Z)),
                    Colors.Red,
                    "MSCN Collision Points"
                );
                newScene.Children.Add(mscnModel);
                
                ChunkItems.Add(new ChunkVisualizationItem
                {
                    Name = "MSCN Collision Points", 
                    Count = Pm4File.MSCN.ExteriorVertices.Count,
                    Color = Colors.Red,
                    IsVisible = true
                });
            }

            // Visualize MSPV structure vertices
            if (Pm4File.MSPV?.Vertices != null && ShowMSPVVertices)
            {
                var mspvModel = CreateVertexVisualization(
                    Pm4File.MSPV.Vertices.Select(v => new Point3D(v.X, v.Y, v.Z)),
                    Colors.Green,
                    "MSPV Structure Vertices"
                );
                newScene.Children.Add(mspvModel);
                
                ChunkItems.Add(new ChunkVisualizationItem
                {
                    Name = "MSPV Structure Vertices",
                    Count = Pm4File.MSPV.Vertices.Count,
                    Color = Colors.Green,
                    IsVisible = true
                });
            }

            // Visualize faces if available
            if (Pm4File.MSVT?.Vertices != null && Pm4File.MSUR?.Entries != null && Pm4File.MSVI?.Indices != null)
            {
                var facesModel = CreateFaceVisualization();
                if (facesModel != null)
                {
                    newScene.Children.Add(facesModel);
                }
            }

            // Visualize node hierarchy connections
            if (ShowNodeHierarchy)
            {
                var hierarchyModel = CreateNodeHierarchyVisualization();
                if (hierarchyModel != null)
                {
                    newScene.Children.Add(hierarchyModel);
                }
            }

            SceneModel = newScene;
        }

        private GeometryModel3D CreateVertexVisualization(IEnumerable<Point3D> points, Color color, string name)
        {
            var geometry = new MeshGeometry3D();
            var material = new DiffuseMaterial(new SolidColorBrush(color));

            foreach (var point in points)
            {
                // Create small spheres for each vertex
                var sphere = new SphereVisual3D { Center = point, Radius = 0.5 };
                // Note: In a real implementation, you'd add these to a combined mesh for performance
                // For now, we'll create point representations
                
                // Add point as tiny cube for performance
                var cube = CreateCube(point, 1.0);
                geometry.Positions.Add(cube.positions[0]);
                geometry.Positions.Add(cube.positions[1]);
                geometry.Positions.Add(cube.positions[2]);
                geometry.Positions.Add(cube.positions[3]);
                
                // Add triangle indices for cube faces
                foreach (var index in cube.indices)
                {
                    geometry.TriangleIndices.Add(index);
                }
            }

            return new GeometryModel3D(geometry, material);
        }

        private (Point3D[] positions, int[] indices) CreateCube(Point3D center, double size)
        {
            var half = size / 2;
            var positions = new Point3D[]
            {
                new(center.X - half, center.Y - half, center.Z - half),
                new(center.X + half, center.Y - half, center.Z - half),
                new(center.X + half, center.Y + half, center.Z - half),
                new(center.X - half, center.Y + half, center.Z - half),
                new(center.X - half, center.Y - half, center.Z + half),
                new(center.X + half, center.Y - half, center.Z + half),
                new(center.X + half, center.Y + half, center.Z + half),
                new(center.X - half, center.Y + half, center.Z + half)
            };

            var indices = new int[]
            {
                // Front face
                0, 1, 2, 0, 2, 3,
                // Back face  
                4, 6, 5, 4, 7, 6,
                // Left face
                0, 3, 7, 0, 7, 4,
                // Right face
                1, 5, 6, 1, 6, 2,
                // Top face
                3, 2, 6, 3, 6, 7,
                // Bottom face
                0, 4, 5, 0, 5, 1
            };

            return (positions, indices);
        }

        private GeometryModel3D? CreateFaceVisualization()
        {
            if (Pm4File?.MSVT?.Vertices == null || Pm4File.MSUR?.Entries == null || Pm4File.MSVI?.Indices == null)
                return null;

            var geometry = new MeshGeometry3D();
            var material = new DiffuseMaterial(new SolidColorBrush(Color.FromArgb(128, 255, 255, 255))); // Semi-transparent

            // Convert MSVT vertices to Point3D array for indexing
            var vertices = Pm4File.MSVT.Vertices.Select(v => new Point3D(v.Y, v.X, v.Z)).ToArray();

            foreach (var surface in Pm4File.MSUR.Entries)
            {
                if (surface.MsviFirstIndex + surface.IndexCount <= Pm4File.MSVI.Indices.Count && surface.IndexCount >= 3)
                {
                    // Get surface indices
                    var surfaceIndices = new List<uint>();
                    for (int i = 0; i < surface.IndexCount; i++)
                    {
                        uint vertexIndex = Pm4File.MSVI.Indices[(int)surface.MsviFirstIndex + i];
                        if (vertexIndex < vertices.Length)
                        {
                            surfaceIndices.Add(vertexIndex);
                        }
                    }

                    // Create triangle fan
                    if (surfaceIndices.Count >= 3)
                    {
                        var baseIndex = geometry.Positions.Count;
                        
                        // Add vertices to geometry
                        foreach (var index in surfaceIndices)
                        {
                            geometry.Positions.Add(vertices[index]);
                        }

                        // Create triangle fan indices
                        for (int i = 1; i < surfaceIndices.Count - 1; i++)
                        {
                            geometry.TriangleIndices.Add(baseIndex);
                            geometry.TriangleIndices.Add(baseIndex + i);
                            geometry.TriangleIndices.Add(baseIndex + i + 1);
                        }
                    }
                }
            }

            return geometry.Positions.Count > 0 ? new GeometryModel3D(geometry, material) : null;
        }

        private GeometryModel3D? CreateNodeHierarchyVisualization()
        {
            if (Pm4File?.MSLK?.Entries == null) return null;

            var geometry = new MeshGeometry3D();
            var material = new DiffuseMaterial(new SolidColorBrush(Colors.Yellow));

            // Visualize MSLK hierarchical connections
            var groupedEntries = Pm4File.MSLK.Entries.GroupBy(e => e.Unknown_0x04).ToList();
            
            foreach (var group in groupedEntries.Where(g => g.Count() > 1))
            {
                var entries = group.ToList();
                
                // Connect entries within the same group
                for (int i = 0; i < entries.Count - 1; i++)
                {
                    var entry1 = entries[i];
                    var entry2 = entries[i + 1];
                    
                    // Try to get 3D positions for connection visualization
                    var pos1 = TryGetEntryPosition(entry1);
                    var pos2 = TryGetEntryPosition(entry2);
                    
                    if (pos1.HasValue && pos2.HasValue)
                    {
                        // Create line between connected nodes
                        AddLine(geometry, pos1.Value, pos2.Value);
                    }
                }
            }

            return geometry.Positions.Count > 0 ? new GeometryModel3D(geometry, material) : null;
        }

        private Point3D? TryGetEntryPosition(MSLKEntry entry)
        {
            // Try to get position from MSVI reference
            if (entry.Unknown_0x10 < (Pm4File?.MSVI?.Indices.Count ?? 0))
            {
                var msviIndex = Pm4File!.MSVI!.Indices[entry.Unknown_0x10];
                if (msviIndex < (Pm4File.MSVT?.Vertices.Count ?? 0))
                {
                    var vertex = Pm4File.MSVT!.Vertices[(int)msviIndex];
                    return new Point3D(vertex.Y, vertex.X, vertex.Z);
                }
            }
            
            return null;
        }

        private void AddLine(MeshGeometry3D geometry, Point3D start, Point3D end)
        {
            var baseIndex = geometry.Positions.Count;
            var thickness = 0.1;
            
            // Create a thin cylinder between two points (simplified)
            geometry.Positions.Add(new Point3D(start.X - thickness, start.Y, start.Z));
            geometry.Positions.Add(new Point3D(start.X + thickness, start.Y, start.Z));
            geometry.Positions.Add(new Point3D(end.X + thickness, end.Y, end.Z));
            geometry.Positions.Add(new Point3D(end.X - thickness, end.Y, end.Z));
            
            // Add quad indices
            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 1);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex + 3);
        }

        private void UpdateStructuralInsights(PM4StructuralAnalyzer.StructuralAnalysisResult result)
        {
            StructuralInsights.Clear();

            // Add basic file info insight first
            StructuralInsights.Add(new StructuralInsight
            {
                Type = "File Analysis",
                Description = $"Analyzing file: {result.FileName}",
                Significance = "Analysis pipeline activated",
                DataPreview = $"Timestamp: {DateTime.Now:HH:mm:ss}"
            });

            // Check for errors
            if (result.Metadata.ContainsKey("AnalysisError") || result.Metadata.ContainsKey("LoadError"))
            {
                var errorKey = result.Metadata.ContainsKey("AnalysisError") ? "AnalysisError" : "LoadError";
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Error",
                    Description = $"Analysis error occurred",
                    Significance = result.Metadata[errorKey]?.ToString() ?? "Unknown error",
                    DataPreview = result.Metadata.ContainsKey("StackTrace") ? result.Metadata["StackTrace"]?.ToString() ?? "" : "No stack trace"
                });
                return; // Don't process further if there's an error
            }

            // Add insights from padding analysis
            foreach (var padding in result.PaddingAnalysis.Where(p => p.HasNonZeroPadding))
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Padding Analysis",
                    Description = $"{padding.ChunkType} has {padding.PaddingBytes} bytes of non-zero padding",
                    Significance = "May contain hidden metadata",
                    DataPreview = string.Join(" ", padding.PaddingData.Take(16).Select(b => b.ToString("X2")))
                });
            }

            // Add chunk counts insight
            if (result.Metadata.TryGetValue("ChunkCounts", out var countsObj) && countsObj is Dictionary<string, int> counts)
            {
                var totalVertices = counts.GetValueOrDefault("MSVT_Vertices") + 
                                  counts.GetValueOrDefault("MSCN_Points") + 
                                  counts.GetValueOrDefault("MSPV_Vertices");
                
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Geometry Statistics",
                    Description = $"Total vertices: {totalVertices:N0}",
                    Significance = $"MSVT: {counts.GetValueOrDefault("MSVT_Vertices"):N0}, MSCN: {counts.GetValueOrDefault("MSCN_Points"):N0}, MSPV: {counts.GetValueOrDefault("MSPV_Vertices"):N0}",
                    DataPreview = $"MSLK entries: {counts.GetValueOrDefault("MSLK_Entries")}, MSUR surfaces: {counts.GetValueOrDefault("MSUR_Surfaces")}"
                });
            }

            // Add insights from unknown fields
            foreach (var field in result.UnknownFields.Where(f => f.LooksLikeIndex || f.LooksLikeFlags))
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Field Pattern",
                    Description = $"{field.ChunkType}.{field.FieldName} appears to be {(field.LooksLikeIndex ? "index data" : "flag data")}",
                    Significance = field.LooksLikeIndex ? "Potential hierarchical reference" : "Potential state/type flags",
                    DataPreview = $"Range: {field.ValueRange}, Unique values: {field.UniqueValues.Count}"
                });
            }

            // Add hierarchical relationship insights
            foreach (var hierarchy in result.Hierarchies.Where(h => h.ConfidenceScore > 0.7f))
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Hierarchy",
                    Description = $"{hierarchy.ParentChunk} → {hierarchy.ChildChunk} ({hierarchy.RelationshipType})",
                    Significance = hierarchy.Evidence,
                    DataPreview = $"Confidence: {hierarchy.ConfidenceScore:P0}"
                });
            }

            // Add node structure insights
            foreach (var nodeStructure in result.NodeStructures.Where(n => n.HasHierarchicalStructure))
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Node Structure",
                    Description = $"{nodeStructure.ChunkType} contains {nodeStructure.NodeGroups.Count} node groups",
                    Significance = "Potential hierarchical object system",
                    DataPreview = string.Join(", ", nodeStructure.NodeGroups.Take(5).Select(g => $"Group {g.GroupId}: {g.EntryCount} entries"))
                });
            }
        }

        private void GenerateAnalysisReport(PM4StructuralAnalyzer.StructuralAnalysisResult result)
        {
            var report = new System.Text.StringBuilder();
            
            report.AppendLine($"=== PM4 STRUCTURAL ANALYSIS REPORT ===");
            report.AppendLine($"File: {result.FileName}");
            report.AppendLine($"Generated: {DateTime.Now}");
            report.AppendLine();
            
            // Chunk presence summary
            if (result.Metadata.TryGetValue("ChunkPresence", out var presenceObj) && presenceObj is Dictionary<string, bool> presence)
            {
                report.AppendLine("CHUNK PRESENCE:");
                foreach (var (chunk, present) in presence.Where(kvp => kvp.Value))
                {
                    report.AppendLine($"  ✓ {chunk}");
                }
                report.AppendLine();
            }

            // Padding analysis
            if (result.PaddingAnalysis.Any(p => p.HasNonZeroPadding))
            {
                report.AppendLine("NON-ZERO PADDING DETECTED:");
                foreach (var padding in result.PaddingAnalysis.Where(p => p.HasNonZeroPadding))
                {
                    report.AppendLine($"  {padding.ChunkType}: {padding.PaddingBytes} bytes");
                    report.AppendLine($"    Data: {string.Join(" ", padding.PaddingData.Take(32).Select(b => b.ToString("X2")))}");
                }
                report.AppendLine();
            }

            // Unknown field patterns
            report.AppendLine("UNKNOWN FIELD ANALYSIS:");
            foreach (var field in result.UnknownFields)
            {
                report.AppendLine($"  {field.ChunkType}.{field.FieldName}:");
                report.AppendLine($"    Range: {field.ValueRange}");
                report.AppendLine($"    Unique values: {field.UniqueValues.Count}");
                report.AppendLine($"    Patterns: Index={field.LooksLikeIndex}, Flags={field.LooksLikeFlags}, Count={field.LooksLikeCount}");
            }
            report.AppendLine();

            // Hierarchical relationships
            report.AppendLine("HIERARCHICAL RELATIONSHIPS:");
            foreach (var hierarchy in result.Hierarchies.OrderByDescending(h => h.ConfidenceScore))
            {
                report.AppendLine($"  {hierarchy.ParentChunk} → {hierarchy.ChildChunk} ({hierarchy.ConfidenceScore:P0})");
                report.AppendLine($"    Type: {hierarchy.RelationshipType}");
                report.AppendLine($"    Evidence: {hierarchy.Evidence}");
            }

            AnalysisOutput = report.ToString();
        }

        private void ToggleChunkVisibility(string? chunkName)
        {
            if (string.IsNullOrEmpty(chunkName)) return;

            switch (chunkName)
            {
                case "MSVT":
                    ShowMSVTVertices = !ShowMSVTVertices;
                    break;
                case "MSCN":
                    ShowMSCNPoints = !ShowMSCNPoints;
                    break;
                case "MSPV":
                    ShowMSPVVertices = !ShowMSPVVertices;
                    break;
            }

            UpdateVisualization();
        }

        private async Task ExportAnalysisAsync()
        {
            if (string.IsNullOrEmpty(AnalysisOutput)) return;

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*",
                FileName = $"PM4_Analysis_{LoadedFileName}_{DateTime.Now:yyyyMMdd_HHmmss}.txt"
            };

            if (dialog.ShowDialog() == true)
            {
                await File.WriteAllTextAsync(dialog.FileName, AnalysisOutput);
                MessageBox.Show($"Analysis exported to: {dialog.FileName}", "Export Complete", 
                    MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }
    }

    public class ChunkVisualizationItem
    {
        public string Name { get; set; } = string.Empty;
        public int Count { get; set; }
        public Color Color { get; set; }
        public bool IsVisible { get; set; }
    }

    public class StructuralInsight
    {
        public string Type { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Significance { get; set; } = string.Empty;
        public string DataPreview { get; set; } = string.Empty;
    }
} 