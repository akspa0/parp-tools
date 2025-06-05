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
using System.Threading;

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
        
        [ObservableProperty]
        private bool _colorByUnknown0x04 = false;
        
        [ObservableProperty]
        private ObservableCollection<Unknown0x04Group> _unknown0x04Groups = new();
        
        [ObservableProperty]
        private Unknown0x04Group? _selectedUnknown0x04Group;
        
        [ObservableProperty]
        private bool _showOnlySelectedGroup = false;

        [ObservableProperty]
        private bool _isLoading = false;
        
        [ObservableProperty]
        private double _loadingProgress = 0;
        
        [ObservableProperty]
        private string _loadingOperation = string.Empty;
        
        [ObservableProperty]
        private string _loadingSubOperation = string.Empty;
        
        [ObservableProperty]
        private bool _canCancelLoading = true;
        
        private CancellationTokenSource? _loadingCancellationTokenSource;

        [ObservableProperty]
        private ObservableCollection<Unknown0x04Group> _filteredUnknown0x04Groups = new();
        
        [ObservableProperty]
        private int _maxGroupsToShow = 50;
        
        [ObservableProperty]
        private string _groupsFilterText = string.Empty;
        
        [ObservableProperty]
        private bool _showAllGroups = false;
        
        [ObservableProperty]
        private string _groupsSummary = string.Empty;

        public MainViewModel()
        {
            _analyzer = new PM4StructuralAnalyzer();
            LoadFileCommand = new AsyncRelayCommand(LoadFileAsync);
            ExportAnalysisCommand = new AsyncRelayCommand(ExportAnalysisAsync);
            ToggleChunkVisibilityCommand = new RelayCommand<string>(ToggleChunkVisibility);
            SelectGroupCommand = new RelayCommand<Unknown0x04Group>(SelectGroup);
            CancelLoadingCommand = new RelayCommand(CancelLoading);
        }

        public IAsyncRelayCommand LoadFileCommand { get; }
        public IAsyncRelayCommand ExportAnalysisCommand { get; }
        public IRelayCommand<string> ToggleChunkVisibilityCommand { get; }
        public IRelayCommand<Unknown0x04Group> SelectGroupCommand { get; }
        public IRelayCommand CancelLoadingCommand { get; }

        partial void OnGroupsFilterTextChanged(string value)
        {
            FilterUnknown0x04Groups();
        }
        
        partial void OnShowAllGroupsChanged(bool value)
        {
            FilterUnknown0x04Groups();
        }
        
        partial void OnMaxGroupsToShowChanged(int value)
        {
            FilterUnknown0x04Groups();
        }
        
        partial void OnColorByUnknown0x04Changed(bool value)
        {
            UpdateVisualization();
        }
        
        partial void OnShowOnlySelectedGroupChanged(bool value)
        {
            UpdateVisualization();
        }

        private async Task LoadFileAsync()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "PM4 Files (*.pm4)|*.pm4|All Files (*.*)|*.*",
                Title = "Select PM4 File"
            };

            if (dialog.ShowDialog() == true)
            {
                _loadingCancellationTokenSource?.Cancel();
                _loadingCancellationTokenSource = new CancellationTokenSource();
                
                IsLoading = true;
                LoadingProgress = 0;
                LoadingOperation = "Starting...";
                LoadingSubOperation = "";
                
                try
                {
                    await LoadPM4FileAsync(dialog.FileName, _loadingCancellationTokenSource.Token);
                }
                catch (OperationCanceledException)
                {
                    LoadingOperation = "Cancelled";
                    LoadingSubOperation = "";
                }
                catch (Exception ex)
                {
                    LoadingOperation = "Error";
                    LoadingSubOperation = ex.Message;
                    MessageBox.Show($"Error loading file: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                finally
                {
                    IsLoading = false;
                    LoadingProgress = 0;
                }
            }
        }

        private async Task LoadPM4FileAsync(string filePath, CancellationToken cancellationToken = default)
        {
            LoadedFileName = Path.GetFileName(filePath);
            
            try
            {
                // Stage 1: Initial file reading (5%)
                LoadingProgress = 5;
                LoadingOperation = "Reading PM4 file";
                LoadingSubOperation = $"Loading {Path.GetFileName(filePath)} ({new FileInfo(filePath).Length / 1024:N0} KB)";
                await Task.Delay(100, cancellationToken);
                
                // Stage 2: Parse PM4 structure with detailed progress (40%)
                LoadingProgress = 10;
                LoadingOperation = "Parsing PM4 structure";
                LoadingSubOperation = "Detecting chunk headers...";
                await Task.Delay(100, cancellationToken);
                
                Pm4File = await Task.Run(() => PM4File.FromFile(filePath), cancellationToken);
                
                // Report what we found
                LoadingProgress = 25;
                LoadingSubOperation = $"Found chunks: {GetChunkSummary(Pm4File)}";
                await Task.Delay(100, cancellationToken);
                
                // Stage 3: Detailed chunk analysis (25%)
                LoadingProgress = 50;
                LoadingOperation = "Analyzing PM4 chunks";
                LoadingSubOperation = await AnalyzeChunksAsync(cancellationToken);
                await Task.Delay(100, cancellationToken);
                
                // Stage 4: Build 3D visualization (15%)
                LoadingProgress = 70;
                LoadingOperation = "Building 3D visualization";
                LoadingSubOperation = "Processing vertex data...";
                await Task.Delay(100, cancellationToken);
                
                // Update visualization on UI thread
                UpdateVisualization();
                
                LoadingProgress = 75;
                LoadingSubOperation = $"Rendered {GetVisualizationSummary()} geometry";
                await Task.Delay(100, cancellationToken);
                
                // Stage 5: Structural analysis (10%)
                LoadingProgress = 85;
                LoadingOperation = "Running structural analysis";
                LoadingSubOperation = "Analyzing unknown fields and patterns...";
                await Task.Delay(100, cancellationToken);
                
                // Run analysis in background but capture result
                var analysisResult = await Task.Run(() => 
                {
                    try
                    {
                        return _analyzer.AnalyzeFile(filePath);
                    }
                    catch (Exception ex)
                    {
                        var errorResult = new PM4StructuralAnalyzer.StructuralAnalysisResult
                        {
                            FileName = Path.GetFileName(filePath)
                        };
                        errorResult.Metadata["AnalysisError"] = ex.Message;
                        errorResult.Metadata["StackTrace"] = ex.StackTrace ?? "No stack trace";
                        return errorResult;
                    }
                }, cancellationToken);
                
                // Stage 6: Generate insights on UI thread (5%)
                LoadingProgress = 95;
                LoadingOperation = "Generating insights";
                LoadingSubOperation = "Creating analysis reports...";
                await Task.Delay(100, cancellationToken);
                
                // Update UI collections on UI thread
                UpdateStructuralInsights(analysisResult);
                GenerateAnalysisReport(analysisResult);
                
                // Final completion
                LoadingProgress = 100;
                LoadingOperation = "Analysis complete";
                LoadingSubOperation = $"Ready! {Unknown0x04Groups.Count} groups • {StructuralInsights.Count} insights • {GetFinalSummary()}";
                await Task.Delay(750, cancellationToken); // Show completion longer
            }
            catch (Exception ex)
            {
                var errorResult = new PM4StructuralAnalyzer.StructuralAnalysisResult
                {
                    FileName = Path.GetFileName(filePath)
                };
                errorResult.Metadata["LoadError"] = ex.Message;
                
                UpdateStructuralInsights(errorResult);
                GenerateAnalysisReport(errorResult);
                
                LoadingOperation = "Error occurred";
                LoadingSubOperation = ex.Message;
            }
        }
        
        private string GetChunkSummary(PM4File pm4)
        {
            var chunks = new List<string>();
            if (pm4.MSLK?.Entries?.Count > 0) chunks.Add($"MSLK({pm4.MSLK.Entries.Count})");
            if (pm4.MSVT?.Vertices?.Count > 0) chunks.Add($"MSVT({pm4.MSVT.Vertices.Count})");
            if (pm4.MSCN?.ExteriorVertices?.Count > 0) chunks.Add($"MSCN({pm4.MSCN.ExteriorVertices.Count})");
            if (pm4.MSPV?.Vertices?.Count > 0) chunks.Add($"MSPV({pm4.MSPV.Vertices.Count})");
            if (pm4.MSVI?.Indices?.Count > 0) chunks.Add($"MSVI({pm4.MSVI.Indices.Count})");
            if (pm4.MSPI?.Indices?.Count > 0) chunks.Add($"MSPI({pm4.MSPI.Indices.Count})");
            if (pm4.MSUR?.Entries?.Count > 0) chunks.Add($"MSUR({pm4.MSUR.Entries.Count})");
            
            return chunks.Count > 0 ? string.Join(", ", chunks.ToArray()) : "No recognized chunks";
        }
        
        private async Task<string> AnalyzeChunksAsync(CancellationToken cancellationToken)
        {
            if (Pm4File == null) return "No PM4 data";
            
            var details = new List<string>();
            
            // Analyze MSLK (navigation links)
            if (Pm4File.MSLK?.Entries?.Count > 0)
            {
                details.Add($"MSLK: {Pm4File.MSLK.Entries.Count} nav entries");
                await Task.Delay(50, cancellationToken);
            }
            
            // Analyze vertex chunks
            var totalVertices = 0;
            if (Pm4File.MSVT?.Vertices?.Count > 0) totalVertices += Pm4File.MSVT.Vertices.Count;
            if (Pm4File.MSCN?.ExteriorVertices?.Count > 0) totalVertices += Pm4File.MSCN.ExteriorVertices.Count;
            if (Pm4File.MSPV?.Vertices?.Count > 0) totalVertices += Pm4File.MSPV.Vertices.Count;
            
            if (totalVertices > 0)
            {
                details.Add($"{totalVertices:N0} total vertices");
                await Task.Delay(50, cancellationToken);
            }
            
            // Analyze face data
            var totalFaces = 0;
            if (Pm4File.MSVI?.Indices?.Count > 0) totalFaces += Pm4File.MSVI.Indices.Count / 3;
            if (totalFaces > 0)
            {
                details.Add($"{totalFaces:N0} faces");
                await Task.Delay(50, cancellationToken);
            }
            
            return details.Count > 0 ? string.Join(" • ", details.ToArray()) : "Basic structure detected";
        }
        
        private string GetVisualizationSummary()
        {
            var parts = new List<string>();
            
            if (Pm4File?.MSVT?.Vertices?.Count > 0) parts.Add($"{Pm4File.MSVT.Vertices.Count} render vertices");
            if (Pm4File?.MSCN?.ExteriorVertices?.Count > 0) parts.Add($"{Pm4File.MSCN.ExteriorVertices.Count} collision points");
            if (Pm4File?.MSPV?.Vertices?.Count > 0) parts.Add($"{Pm4File.MSPV.Vertices.Count} structure vertices");
            
            return parts.Count > 0 ? string.Join(" • ", parts.ToArray()) : "No geometry";
        }
        
        private string GetFinalSummary()
        {
            var summary = new List<string>();
            
            if (ChunkItems.Count > 0) summary.Add($"{ChunkItems.Count} chunk types");
            
            var totalVertices = ChunkItems.Sum(c => c.Count);
            if (totalVertices > 0) summary.Add($"{totalVertices:N0} vertices");
            
            return summary.Count > 0 ? string.Join(" • ", summary.ToArray()) : "Analysis complete";
        }

        private void UpdateVisualization()
        {
            if (Pm4File == null) return;

            var newScene = new Model3DGroup();
            ChunkItems.Clear();

            // Analyze Unknown_0x04 groups first
            AnalyzeUnknown0x04Groups();

            // Visualize MSVT render vertices
            if (Pm4File.MSVT?.Vertices != null && ShowMSVTVertices)
            {
                var msvtColor = ColorByUnknown0x04 ? Colors.LightBlue : Colors.Blue;
                var msvtModel = CreateVertexVisualization(
                    Pm4File.MSVT.Vertices.Select(v => new Point3D(v.Y, v.X, v.Z)),
                    msvtColor,
                    "MSVT Render Vertices"
                );
                newScene.Children.Add(msvtModel);
                
                ChunkItems.Add(new ChunkVisualizationItem
                {
                    Name = "MSVT Render Vertices",
                    Count = Pm4File.MSVT.Vertices.Count,
                    Color = msvtColor,
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

            // Visualize Unknown_0x04 groups if enabled
            if (ColorByUnknown0x04 || ShowOnlySelectedGroup)
            {
                CreateUnknown0x04GroupVisualization(newScene);
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

        private void CreateUnknown0x04GroupVisualization(Model3DGroup scene)
        {
            if (Pm4File?.MSLK?.Entries == null) return;

            var groupsToShow = Unknown0x04Groups.AsEnumerable();
            
            // Filter to only selected group if that option is enabled
            if (ShowOnlySelectedGroup && SelectedUnknown0x04Group != null)
            {
                groupsToShow = groupsToShow.Where(g => g.GroupValue == SelectedUnknown0x04Group.GroupValue);
            }

            foreach (var group in groupsToShow)
            {
                // Get MSLK entries for this group
                var groupEntries = group.EntryIndices
                    .Where(idx => idx < Pm4File.MSLK.Entries.Count)
                    .Select(idx => Pm4File.MSLK.Entries[idx])
                    .ToList();

                if (!groupEntries.Any()) continue;

                // Create visualization for group vertices
                var groupVertices = ExtractGroupVerticesAsPoints(groupEntries);
                if (groupVertices.Any())
                {
                    var groupModel = CreateVertexVisualization(groupVertices, group.Color, $"Group 0x{group.GroupValue:X8}");
                    scene.Children.Add(groupModel);
                    
                    // Add to chunk items for visibility control
                    ChunkItems.Add(new ChunkVisualizationItem
                    {
                        Name = $"Group 0x{group.GroupValue:X8} ({group.EntryCount} entries)",
                        Count = groupVertices.Count(),
                        Color = group.Color,
                        IsVisible = true
                    });
                }
            }
        }

        private IEnumerable<Point3D> ExtractGroupVerticesAsPoints(List<MSLKEntry> entries)
        {
            var points = new List<Point3D>();
            
            if (Pm4File?.MSPI?.Indices == null || Pm4File.MSPV?.Vertices == null)
                return points;

            foreach (var entry in entries)
            {
                if (entry.MspiFirstIndex >= 0 && 
                    entry.MspiFirstIndex + entry.MspiIndexCount <= Pm4File.MSPI.Indices.Count)
                {
                    for (int i = 0; i < entry.MspiIndexCount; i++)
                    {
                        var mspiIndex = Pm4File.MSPI.Indices[entry.MspiFirstIndex + i];
                        if (mspiIndex < Pm4File.MSPV.Vertices.Count)
                        {
                            var vertex = Pm4File.MSPV.Vertices[(int)mspiIndex];
                            points.Add(new Point3D(vertex.X, vertex.Y, vertex.Z));
                        }
                    }
                }
            }
            
            return points;
        }

        private void UpdateStructuralInsights(PM4StructuralAnalyzer.StructuralAnalysisResult result)
        {
            // Ensure we're on the UI thread for collection updates
            if (System.Windows.Application.Current.Dispatcher.CheckAccess())
            {
                UpdateStructuralInsightsImpl(result);
            }
            else
            {
                System.Windows.Application.Current.Dispatcher.Invoke(() => UpdateStructuralInsightsImpl(result));
            }
        }
        
        private void UpdateStructuralInsightsImpl(PM4StructuralAnalyzer.StructuralAnalysisResult result)
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

        private void AnalyzeUnknown0x04Groups()
        {
            // Ensure we're on the UI thread for collection updates
            if (System.Windows.Application.Current.Dispatcher.CheckAccess())
            {
                AnalyzeUnknown0x04GroupsImpl();
            }
            else
            {
                System.Windows.Application.Current.Dispatcher.Invoke(AnalyzeUnknown0x04GroupsImpl);
            }
        }
        
        private void AnalyzeUnknown0x04GroupsImpl()
        {
            Unknown0x04Groups.Clear();
            
            if (Pm4File?.MSLK?.Entries == null || Pm4File.MSLK.Entries.Count == 0)
            {
                GroupsSummary = "No MSLK data available";
                FilterUnknown0x04Groups();
                return;
            }

            // Group MSLK entries by Unknown_0x04 value
            var groupedEntries = Pm4File.MSLK.Entries
                .Select((entry, index) => new { Entry = entry, Index = index })
                .GroupBy(x => x.Entry.Unknown_0x04)
                .OrderByDescending(g => g.Count()) // Show largest groups first
                .ToList();

            // Generate distinct colors for the groups we'll actually show
            var maxColors = Math.Min(groupedEntries.Count, 100); // Limit color generation
            var colors = GenerateDistinctColors(maxColors);
            
            for (int i = 0; i < groupedEntries.Count; i++)
            {
                var group = groupedEntries[i];
                var entries = group.Select(x => x.Entry).ToList();
                var color = i < maxColors ? colors[i] : Colors.Gray;
                
                var groupInfo = new Unknown0x04Group
                {
                    GroupValue = group.Key,
                    EntryCount = group.Count(),
                    EntryIndices = group.Select(x => x.Index).ToList(),
                    GroupColor = color,
                    Color = color,
                    Description = AnalyzeGroupPattern(entries),
                    AssociatedVertices = ExtractGroupVertices(entries),
                    AverageUnknown0x0C = entries.Count > 0 ? (float)entries.Average(e => e.Unknown_0x0C) : 0f,
                    MinUnknown0x0C = entries.Count > 0 ? entries.Min(e => e.Unknown_0x0C) : 0f,
                    MaxUnknown0x0C = entries.Count > 0 ? entries.Max(e => e.Unknown_0x0C) : 0f
                };
                
                Unknown0x04Groups.Add(groupInfo);
            }
            
            // Update summary
            var totalEntries = groupedEntries.Sum(g => g.Count());
            var topGroups = groupedEntries.Take(5).ToList();
            var topGroupsInfo = string.Join(", ", topGroups.Select(g => $"0x{g.Key:X8}({g.Count()})"));
            
            GroupsSummary = $"Total: {groupedEntries.Count} groups, {totalEntries} entries. Top: {topGroupsInfo}";
            
            // Apply filtering
            FilterUnknown0x04Groups();
        }
        
        private void FilterUnknown0x04Groups()
        {
            // Ensure we're on the UI thread for collection updates
            if (System.Windows.Application.Current.Dispatcher.CheckAccess())
            {
                FilterUnknown0x04GroupsImpl();
            }
            else
            {
                System.Windows.Application.Current.Dispatcher.Invoke(FilterUnknown0x04GroupsImpl);
            }
        }
        
        private void FilterUnknown0x04GroupsImpl()
        {
            if (Unknown0x04Groups.Count == 0)
            {
                FilteredUnknown0x04Groups.Clear();
                return;
            }
            
            var filtered = Unknown0x04Groups.AsEnumerable();
            
            // Apply text filter
            if (!string.IsNullOrWhiteSpace(GroupsFilterText))
            {
                var filterLower = GroupsFilterText.ToLower();
                filtered = filtered.Where(g => 
                    g.GroupValue.ToString("X8").ToLower().Contains(filterLower) ||
                    g.Description.ToLower().Contains(filterLower) ||
                    g.EntryCount.ToString().Contains(filterLower));
            }
            
            // Apply count limit
            if (!ShowAllGroups)
            {
                filtered = filtered.Take(MaxGroupsToShow);
            }
            
            FilteredUnknown0x04Groups.Clear();
            foreach (var group in filtered)
            {
                FilteredUnknown0x04Groups.Add(group);
            }
        }

        private List<Color> GenerateDistinctColors(int count)
        {
            var colors = new List<Color>();
            if (count == 0) return colors;

            for (int i = 0; i < count; i++)
            {
                float hue = (float)i / count * 360f;
                var color = HSVToRGB(hue, 0.8f, 0.9f);
                colors.Add(color);
            }
            
            return colors;
        }

        private Color HSVToRGB(float h, float s, float v)
        {
            h = h / 60f;
            int i = (int)Math.Floor(h);
            float f = h - i;
            float p = v * (1 - s);
            float q = v * (1 - s * f);
            float t = v * (1 - s * (1 - f));

            float r, g, b;
            switch (i % 6)
            {
                case 0: r = v; g = t; b = p; break;
                case 1: r = q; g = v; b = p; break;
                case 2: r = p; g = v; b = t; break;
                case 3: r = p; g = q; b = v; break;
                case 4: r = t; g = p; b = v; break;
                default: r = v; g = p; b = q; break;
            }

            return Color.FromRgb((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
        }

        private string AnalyzeGroupPattern(List<MSLKEntry> entries)
        {
            if (entries.Count == 1)
                return "Single entry";
                
            // Analyze patterns in the group
            var hasGeometry = entries.Any(e => e.MspiFirstIndex >= 0 && e.MspiIndexCount > 0);
            var hasDoodads = entries.Any(e => e.MspiFirstIndex == -1);
            var avgUnknown0x0C = entries.Average(e => e.Unknown_0x0C);
            
            var pattern = new List<string>();
            if (hasGeometry) pattern.Add("Geometry");
            if (hasDoodads) pattern.Add("Doodads");
            
            var description = pattern.Count > 0 ? string.Join(" + ", pattern) : "Unknown";
            return $"{description} (avg 0x0C: {avgUnknown0x0C:F1})";
        }

        private List<Vector3> ExtractGroupVertices(List<MSLKEntry> entries)
        {
            var vertices = new List<Vector3>();
            
            if (Pm4File?.MSPI?.Indices == null || Pm4File.MSPV?.Vertices == null)
                return vertices;

            foreach (var entry in entries)
            {
                if (entry.MspiFirstIndex >= 0 && 
                    entry.MspiFirstIndex + entry.MspiIndexCount <= Pm4File.MSPI.Indices.Count)
                {
                    for (int i = 0; i < entry.MspiIndexCount; i++)
                    {
                        var mspiIndex = Pm4File.MSPI.Indices[entry.MspiFirstIndex + i];
                        if (mspiIndex < Pm4File.MSPV.Vertices.Count)
                        {
                            var vertex = Pm4File.MSPV.Vertices[(int)mspiIndex];
                            vertices.Add(new Vector3(vertex.X, vertex.Y, vertex.Z));
                        }
                    }
                }
            }
            
            return vertices;
        }

        private void SelectGroup(Unknown0x04Group? group)
        {
            SelectedUnknown0x04Group = group;
            
            // Update visualization when group is selected
            if (ShowOnlySelectedGroup || ColorByUnknown0x04)
            {
                UpdateVisualization();
            }
        }

        private void CancelLoading()
        {
            _loadingCancellationTokenSource?.Cancel();
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

    public class Unknown0x04Group
    {
        public uint GroupValue { get; set; }
        public int EntryCount { get; set; }
        public List<int> EntryIndices { get; set; } = new();
        public Color GroupColor { get; set; }
        public Color Color { get; set; }
        public string Description { get; set; } = string.Empty;
        public List<Vector3> AssociatedVertices { get; set; } = new();
        public float AverageUnknown0x0C { get; set; }
        public float MinUnknown0x0C { get; set; }
        public float MaxUnknown0x0C { get; set; }
    }

    public class LoadingProgress
    {
        public double Percentage { get; set; }
        public string CurrentOperation { get; set; } = string.Empty;
        public string SubOperation { get; set; } = string.Empty;
        public bool IsIndeterminate { get; set; }
        public bool CanCancel { get; set; } = true;
    }

    public class PM4AnalysisResult
    {
        public List<Unknown0x04Group> Groups { get; set; } = new();
        public List<StructuralInsight> Insights { get; set; } = new();
        public Dictionary<string, object> RawAnalysis { get; set; } = new();
        public string Summary { get; set; } = string.Empty;
    }
} 