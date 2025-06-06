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
        
        // PM4 Coordinate System Constants
        private const float PM4_COORDINATE_BOUND = 17066.666f;
        private const float PM4_MIN_COORDINATE = -PM4_COORDINATE_BOUND;
        private const float PM4_MAX_COORDINATE = PM4_COORDINATE_BOUND;
        
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
        private string _groupsSummary = "No analysis available";

        [ObservableProperty]
        private bool _colorByUnknown0x0C = false;
        
        [ObservableProperty]
        private ObservableCollection<Unknown0x0CGroup> _unknown0x0CGroups = new();
        
        [ObservableProperty]
        private ObservableCollection<Unknown0x0CGroup> _filteredUnknown0x0CGroups = new();

        // Enhanced Visualization Controls
        [ObservableProperty]
        private bool _showGroundGrid = true;
        
        [ObservableProperty]
        private bool _showCoordinateBounds = true;
        
        [ObservableProperty]
        private bool _showCoordinateAxes = true;
        
        [ObservableProperty]
        private bool _showVertexIndices = false;
        
        [ObservableProperty]
        private bool _showGroupConnections = false;
        
        [ObservableProperty]
        private bool _showHierarchyLines = false;
        
        // Legend System
        [ObservableProperty]
        private ObservableCollection<LegendItem> _legendItems = new();
        
        // Batch Loading Properties
        [ObservableProperty]
        private ObservableCollection<PM4File> _loadedFiles = new();
        
        [ObservableProperty]
        private bool _showBatchSummary = false;
        
        [ObservableProperty]
        private string _batchSummary = "No batch data loaded";
        
        [ObservableProperty]
        private int _totalFilesLoaded = 0;
        
        [ObservableProperty]
        private long _totalVerticesLoaded = 0;
        
        [ObservableProperty]
        private long _totalEntriesLoaded = 0;
        
        // Camera Controls
        [ObservableProperty]
        private Point3D _cameraPosition = new(100, 100, 100);
        
        [ObservableProperty]
        private Vector3D _cameraLookDirection = new(-1, -1, -1);
        
        [ObservableProperty]
        private string _cameraInfo = "Position: (100, 100, 100)";
        
        // Data Navigation
        [ObservableProperty]
        private string _selectedDataInfo = "No data selected";
        
        [ObservableProperty]
        private bool _autoFitCamera = true;

        // Hierarchy Visualization Properties
        [ObservableProperty]
        private HierarchyAnalysisResult? _currentHierarchyAnalysis;

        [ObservableProperty]
        private bool _showHierarchyTree = false;

        [ObservableProperty]
        private bool _showRootNodes = true;

        [ObservableProperty]
        private bool _showLeafNodes = true;

        [ObservableProperty]
        private bool _showParentChildConnections = true;

        [ObservableProperty]
        private bool _showCrossReferences = false;

        [ObservableProperty]
        private int _hierarchyLevelFilter = 0; // 0 = show all levels

        public MainViewModel()
        {
            _analyzer = new PM4StructuralAnalyzer();
            LoadFileCommand = new AsyncRelayCommand(LoadFileAsync);
            ExportAnalysisCommand = new AsyncRelayCommand(ExportAnalysisAsync);
            ToggleChunkVisibilityCommand = new RelayCommand<string>(ToggleChunkVisibility);
            SelectGroupCommand = new RelayCommand<Unknown0x04Group>(SelectGroup);
            CancelLoadingCommand = new RelayCommand(CancelLoading);
            
            // Enhanced Controls
            FitCameraToDataCommand = new RelayCommand(FitCameraToData);
            ResetCameraCommand = new RelayCommand(ResetCamera);
            JumpToChunkCommand = new RelayCommand<string>(JumpToChunk);
            RefreshLegendCommand = new RelayCommand(RefreshLegend);
            
            // Enhanced Export Commands
            ExportInsightsCommand = new AsyncRelayCommand(ExportInsightsAsync);
            ExportDetailedAnalysisCommand = new AsyncRelayCommand(ExportDetailedAnalysisAsync);
            ExportGroupAnalysisCommand = new AsyncRelayCommand(ExportGroupAnalysisAsync);
            PerformDeepAnalysisCommand = new AsyncRelayCommand(PerformDeepAnalysisAsync);
            InvestigatePaddingCommand = new AsyncRelayCommand(InvestigatePaddingAsync);
            
            // Batch Loading Commands
            LoadBatchFilesCommand = new AsyncRelayCommand(LoadBatchFilesAsync);
            LoadDirectoryCommand = new AsyncRelayCommand(LoadDirectoryAsync);
            
            InitializeLegend();
        }

        public IAsyncRelayCommand LoadFileCommand { get; }
        public IAsyncRelayCommand ExportAnalysisCommand { get; }
        public IRelayCommand<string> ToggleChunkVisibilityCommand { get; }
        public IRelayCommand<Unknown0x04Group> SelectGroupCommand { get; }
        public IRelayCommand CancelLoadingCommand { get; }
        
        // Enhanced Commands
        public IRelayCommand FitCameraToDataCommand { get; }
        public IRelayCommand ResetCameraCommand { get; }
        public IRelayCommand<string> JumpToChunkCommand { get; }
        public IRelayCommand RefreshLegendCommand { get; }

        // Enhanced Export Commands
        public IAsyncRelayCommand ExportInsightsCommand { get; }
        public IAsyncRelayCommand ExportDetailedAnalysisCommand { get; }
        public IAsyncRelayCommand ExportGroupAnalysisCommand { get; }
        public IAsyncRelayCommand PerformDeepAnalysisCommand { get; }
        public IAsyncRelayCommand InvestigatePaddingCommand { get; }
        
        // Batch Loading Commands
        public IAsyncRelayCommand LoadBatchFilesCommand { get; }
        public IAsyncRelayCommand LoadDirectoryCommand { get; }

        partial void OnGroupsFilterTextChanged(string value)
        {
            FilterUnknown0x04Groups();
            FilterUnknown0x0CGroups();
        }
        
        partial void OnMaxGroupsToShowChanged(int value)
        {
            FilterUnknown0x04Groups();
            FilterUnknown0x0CGroups();
        }
        
        partial void OnShowAllGroupsChanged(bool value)
        {
            FilterUnknown0x04Groups();
            FilterUnknown0x0CGroups();
        }
        
        partial void OnColorByUnknown0x04Changed(bool value)
        {
            UpdateVisualization();
        }
        
        partial void OnColorByUnknown0x0CChanged(bool value)
        {
            UpdateVisualization();
        }
        
        partial void OnShowOnlySelectedGroupChanged(bool value)
        {
            UpdateVisualization();
        }
        
        // Enhanced Visualization Property Changes
        partial void OnShowGroundGridChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }
        
        partial void OnShowCoordinateBoundsChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }
        
        partial void OnShowCoordinateAxesChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }
        
        partial void OnShowVertexIndicesChanged(bool value)
        {
            UpdateVisualization();
        }
        
        partial void OnShowGroupConnectionsChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }
        
        partial void OnShowHierarchyLinesChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }
        
        partial void OnShowMSVTVerticesChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }
        
        partial void OnShowMSCNPointsChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }
        
        partial void OnShowMSPVVerticesChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }

        // Hierarchy Visualization Property Changes
        partial void OnShowHierarchyTreeChanged(bool value)
        {
            UpdateVisualization();
            RefreshLegend();
        }

        partial void OnShowRootNodesChanged(bool value)
        {
            UpdateVisualization();
        }

        partial void OnShowLeafNodesChanged(bool value)
        {
            UpdateVisualization();
        }

        partial void OnShowParentChildConnectionsChanged(bool value)
        {
            UpdateVisualization();
        }

        partial void OnShowCrossReferencesChanged(bool value)
        {
            UpdateVisualization();
        }

        partial void OnHierarchyLevelFilterChanged(int value)
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
                LoadingOperation = "Structural analysis";
                LoadingSubOperation = "Analyzing Unknown_0x04 patterns...";
                await Task.Delay(100, cancellationToken);
                
                AnalyzeUnknown0x0CGroups();
                AnalyzeUnknown0x04Groups();
                
                LoadingProgress = 90;
                LoadingSubOperation = "Analyzing Unknown_0x0C patterns...";
                await Task.Delay(100, cancellationToken);
                
                AnalyzeUnknown0x0CGroups();
                
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

            // Add coordinate system visualization first
            AddCoordinateSystemVisualization(newScene);

            // Analyze Unknown_0x04 groups first
            AnalyzeUnknown0x04Groups();

            // Visualize MSVT render vertices
            if (Pm4File.MSVT?.Vertices != null && ShowMSVTVertices)
            {
                var msvtColor = ColorByUnknown0x04 ? Colors.LightBlue : Colors.Blue;
                var msvtPoints = Pm4File.MSVT.Vertices.Select(v => new Point3D(v.Y, v.X, v.Z)).ToList();
                var boundsInfo = ValidateCoordinateBounds(msvtPoints, "MSVT");
                
                var msvtModel = CreateVertexVisualization(msvtPoints, msvtColor, "MSVT Render Vertices");
                newScene.Children.Add(msvtModel);
                
                ChunkItems.Add(new ChunkVisualizationItem
                {
                    Name = $"MSVT Render Vertices{boundsInfo}",
                    Count = Pm4File.MSVT.Vertices.Count,
                    Color = msvtColor,
                    IsVisible = true
                });
            }

            // Visualize MSCN collision points
            if (Pm4File.MSCN?.ExteriorVertices != null && ShowMSCNPoints)
            {
                var mscnPoints = Pm4File.MSCN.ExteriorVertices.Select(v => new Point3D(v.X, -v.Y, v.Z)).ToList();
                var boundsInfo = ValidateCoordinateBounds(mscnPoints, "MSCN");
                
                var mscnModel = CreateVertexVisualization(mscnPoints, Colors.Red, "MSCN Collision Points");
                newScene.Children.Add(mscnModel);
                
                ChunkItems.Add(new ChunkVisualizationItem
                {
                    Name = $"MSCN Collision Points{boundsInfo}", 
                    Count = Pm4File.MSCN.ExteriorVertices.Count,
                    Color = Colors.Red,
                    IsVisible = true
                });
            }

            // Visualize MSPV structure vertices
            if (Pm4File.MSPV?.Vertices != null && ShowMSPVVertices)
            {
                var mspvPoints = Pm4File.MSPV.Vertices.Select(v => new Point3D(v.X, v.Y, v.Z)).ToList();
                var boundsInfo = ValidateCoordinateBounds(mspvPoints, "MSPV");
                
                var mspvModel = CreateVertexVisualization(mspvPoints, Colors.Green, "MSPV Structure Vertices");
                newScene.Children.Add(mspvModel);
                
                ChunkItems.Add(new ChunkVisualizationItem
                {
                    Name = $"MSPV Structure Vertices{boundsInfo}",
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
            if (ShowNodeHierarchy || ShowHierarchyTree)
            {
                var hierarchyModel = CreateNodeHierarchyVisualization();
                if (hierarchyModel != null)
                {
                    newScene.Children.Add(hierarchyModel);
                }
                
                // If showing hierarchy tree, add the enhanced visualizations directly to scene
                if (ShowHierarchyTree && CurrentHierarchyAnalysis != null)
                {
                    CreateHierarchyTreeVisualization(newScene);
                }
            }

            // Visualize Unknown_0x04 groups if enabled
            if (ColorByUnknown0x04 || ShowOnlySelectedGroup)
            {
                CreateUnknown0x04GroupVisualization(newScene);
            }
            
            // Visualize Unknown_0x0C groups if enabled
            if (ColorByUnknown0x0C)
            {
                CreateUnknown0x0CGroupVisualization(newScene);
            }

            SceneModel = newScene;
            
            // Update legend to reflect current visualization state
            RefreshLegend();
            
            // Auto-fit camera if enabled and data loaded
            if (AutoFitCamera && Pm4File != null)
            {
                FitCameraToData();
            }
        }

        private void AddCoordinateSystemVisualization(Model3DGroup scene)
        {
            // Add ground plane grid at Y=0 within PM4 coordinate bounds
            if (ShowGroundGrid)
            {
                var groundPlane = CreateGroundPlaneGrid();
                scene.Children.Add(groundPlane);
            }
            
            // Add coordinate boundary visualization
            if (ShowCoordinateBounds)
            {
                var coordinateBounds = CreateCoordinateBounds();
                scene.Children.Add(coordinateBounds);
            }
            
            // Add coordinate axes
            if (ShowCoordinateAxes)
            {
                var coordinateAxes = CreateCoordinateAxes();
                scene.Children.Add(coordinateAxes);
            }
        }

        private GeometryModel3D CreateGroundPlaneGrid()
        {
            var geometry = new MeshGeometry3D();
            var material = new DiffuseMaterial(new SolidColorBrush(Color.FromArgb(64, 128, 128, 128))); // Semi-transparent gray

            var gridSize = PM4_COORDINATE_BOUND;
            var gridSpacing = gridSize / 20f; // 20 divisions
            
            // Create grid lines parallel to X-axis (running in Z direction)
            for (int i = 0; i <= 40; i++)
            {
                var x = PM4_MIN_COORDINATE + (i * gridSpacing);
                AddGridLine(geometry, 
                    new Point3D(x, 0, PM4_MIN_COORDINATE), 
                    new Point3D(x, 0, PM4_MAX_COORDINATE));
            }
            
            // Create grid lines parallel to Z-axis (running in X direction)
            for (int i = 0; i <= 40; i++)
            {
                var z = PM4_MIN_COORDINATE + (i * gridSpacing);
                AddGridLine(geometry, 
                    new Point3D(PM4_MIN_COORDINATE, 0, z), 
                    new Point3D(PM4_MAX_COORDINATE, 0, z));
            }

            return new GeometryModel3D(geometry, material);
        }

        private GeometryModel3D CreateCoordinateBounds()
        {
            var geometry = new MeshGeometry3D();
            var material = new DiffuseMaterial(new SolidColorBrush(Color.FromArgb(128, 255, 255, 0))); // Semi-transparent yellow

            // Create wireframe bounding box at PM4 coordinate limits
            var corners = new Point3D[]
            {
                // Bottom face
                new(PM4_MIN_COORDINATE, PM4_MIN_COORDINATE, PM4_MIN_COORDINATE),
                new(PM4_MAX_COORDINATE, PM4_MIN_COORDINATE, PM4_MIN_COORDINATE),
                new(PM4_MAX_COORDINATE, PM4_MIN_COORDINATE, PM4_MAX_COORDINATE),
                new(PM4_MIN_COORDINATE, PM4_MIN_COORDINATE, PM4_MAX_COORDINATE),
                // Top face
                new(PM4_MIN_COORDINATE, PM4_MAX_COORDINATE, PM4_MIN_COORDINATE),
                new(PM4_MAX_COORDINATE, PM4_MAX_COORDINATE, PM4_MIN_COORDINATE),
                new(PM4_MAX_COORDINATE, PM4_MAX_COORDINATE, PM4_MAX_COORDINATE),
                new(PM4_MIN_COORDINATE, PM4_MAX_COORDINATE, PM4_MAX_COORDINATE)
            };

            // Add wireframe edges
            var wireframeEdges = new (int start, int end)[]
            {
                // Bottom face edges
                (0, 1), (1, 2), (2, 3), (3, 0),
                // Top face edges
                (4, 5), (5, 6), (6, 7), (7, 4),
                // Vertical edges
                (0, 4), (1, 5), (2, 6), (3, 7)
            };

            foreach (var (start, end) in wireframeEdges)
            {
                AddWireframeLine(geometry, corners[start], corners[end]);
            }

            return new GeometryModel3D(geometry, material);
        }

        private GeometryModel3D CreateCoordinateAxes()
        {
            var geometry = new MeshGeometry3D();
            var material = new DiffuseMaterial(new SolidColorBrush(Colors.White));

            var axisLength = PM4_COORDINATE_BOUND * 0.1f; // 10% of coordinate space for axis length
            var origin = new Point3D(0, 0, 0);

            // X-axis (Red)
            AddAxisLine(geometry, origin, new Point3D(axisLength, 0, 0), Colors.Red);
            // Y-axis (Green) 
            AddAxisLine(geometry, origin, new Point3D(0, axisLength, 0), Colors.Green);
            // Z-axis (Blue)
            AddAxisLine(geometry, origin, new Point3D(0, 0, axisLength), Colors.Blue);

            return new GeometryModel3D(geometry, material);
        }

        private void AddGridLine(MeshGeometry3D geometry, Point3D start, Point3D end)
        {
            var thickness = 1.0f;
            var baseIndex = geometry.Positions.Count;

            // Create thin line as a rectangle
            var direction = new Vector3D(end.X - start.X, end.Y - start.Y, end.Z - start.Z);
            direction.Normalize();
            
            var perpendicular = new Vector3D(-direction.Y, direction.X, 0);
            if (perpendicular.Length < 0.01) perpendicular = new Vector3D(0, 0, 1);
            perpendicular.Normalize();
            perpendicular *= thickness;

            geometry.Positions.Add(new Point3D(start.X - perpendicular.X, start.Y - perpendicular.Y, start.Z - perpendicular.Z));
            geometry.Positions.Add(new Point3D(start.X + perpendicular.X, start.Y + perpendicular.Y, start.Z + perpendicular.Z));
            geometry.Positions.Add(new Point3D(end.X + perpendicular.X, end.Y + perpendicular.Y, end.Z + perpendicular.Z));
            geometry.Positions.Add(new Point3D(end.X - perpendicular.X, end.Y - perpendicular.Y, end.Z - perpendicular.Z));

            // Add triangle indices for the line quad
            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 1);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex + 3);
        }

        private void AddWireframeLine(MeshGeometry3D geometry, Point3D start, Point3D end)
        {
            var thickness = 10.0f; // Thicker for visibility
            var baseIndex = geometry.Positions.Count;

            var direction = new Vector3D(end.X - start.X, end.Y - start.Y, end.Z - start.Z);
            direction.Normalize();
            
            var perpendicular = new Vector3D(-direction.Y, direction.X, 0);
            if (perpendicular.Length < 0.01) perpendicular = new Vector3D(0, 0, 1);
            perpendicular.Normalize();
            perpendicular *= thickness;

            geometry.Positions.Add(new Point3D(start.X - perpendicular.X, start.Y - perpendicular.Y, start.Z - perpendicular.Z));
            geometry.Positions.Add(new Point3D(start.X + perpendicular.X, start.Y + perpendicular.Y, start.Z + perpendicular.Z));
            geometry.Positions.Add(new Point3D(end.X + perpendicular.X, end.Y + perpendicular.Y, end.Z + perpendicular.Z));
            geometry.Positions.Add(new Point3D(end.X - perpendicular.X, end.Y - perpendicular.Y, end.Z - perpendicular.Z));

            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 1);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex + 3);
        }

        private void AddAxisLine(MeshGeometry3D geometry, Point3D start, Point3D end, Color color)
        {
            var thickness = 20.0f; // Thick for visibility
            var baseIndex = geometry.Positions.Count;

            var direction = new Vector3D(end.X - start.X, end.Y - start.Y, end.Z - start.Z);
            direction.Normalize();
            
            var perpendicular = new Vector3D(-direction.Y, direction.X, 0);
            if (perpendicular.Length < 0.01) perpendicular = new Vector3D(0, 0, 1);
            perpendicular.Normalize();
            perpendicular *= thickness;

            geometry.Positions.Add(new Point3D(start.X - perpendicular.X, start.Y - perpendicular.Y, start.Z - perpendicular.Z));
            geometry.Positions.Add(new Point3D(start.X + perpendicular.X, start.Y + perpendicular.Y, start.Z + perpendicular.Z));
            geometry.Positions.Add(new Point3D(end.X + perpendicular.X, end.Y + perpendicular.Y, end.Z + perpendicular.Z));
            geometry.Positions.Add(new Point3D(end.X - perpendicular.X, end.Y - perpendicular.Y, end.Z - perpendicular.Z));

            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 1);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex);
            geometry.TriangleIndices.Add(baseIndex + 2);
            geometry.TriangleIndices.Add(baseIndex + 3);
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
            if (!ShowHierarchyTree || CurrentHierarchyAnalysis == null || Pm4File?.MSLK?.Entries == null) return null;

            var hierarchyScene = new Model3DGroup();

            // Create enhanced hierarchy visualization
            CreateHierarchyTreeVisualization(hierarchyScene);

            if (hierarchyScene.Children.Count > 0)
            {
                return new GeometryModel3D(null, null) { Geometry = null, Material = null, BackMaterial = null };
            }

            return null;
        }

        private void CreateHierarchyTreeVisualization(Model3DGroup scene)
        {
            if (CurrentHierarchyAnalysis == null) return;

            // 1. Create hierarchy nodes with different visuals for root, intermediate, and leaf nodes
            foreach (var kvp in CurrentHierarchyAnalysis.GroupHierarchy)
            {
                var groupValue = kvp.Key;
                var hierarchyInfo = kvp.Value;

                // Filter by hierarchy level if specified
                if (HierarchyLevelFilter > 0 && hierarchyInfo.HierarchyLevel != HierarchyLevelFilter)
                    continue;

                // Get positions for this group
                var group = Unknown0x04Groups.FirstOrDefault(g => g.GroupValue == groupValue);
                if (group?.AssociatedVertices == null || !group.AssociatedVertices.Any())
                    continue;

                // Calculate center position for the group
                var centerPos = CalculateGroupCenter(group.AssociatedVertices);

                // Create node visualization based on hierarchy type
                CreateHierarchyNode(scene, centerPos, hierarchyInfo, groupValue);
            }

            // 2. Create parent-child connections
            if (ShowParentChildConnections)
            {
                CreateParentChildConnections(scene);
            }

            // 3. Create cross-reference connections
            if (ShowCrossReferences)
            {
                CreateCrossReferenceConnections(scene);
            }
        }

        private Point3D CalculateGroupCenter(List<Vector3> vertices)
        {
            if (vertices.Count == 0) return new Point3D();

            var centerX = vertices.Average(v => v.X);
            var centerY = vertices.Average(v => v.Y);
            var centerZ = vertices.Average(v => v.Z);

            return new Point3D(centerX, centerY, centerZ);
        }

        private void CreateHierarchyNode(Model3DGroup scene, Point3D position, GroupHierarchyInfo hierarchyInfo, uint groupValue)
        {
            var geometry = new MeshGeometry3D();
            Color nodeColor;
            double nodeSize;
            
            // Special handling for the root node (0x00000000)
            if (groupValue == 0x00000000 && hierarchyInfo.IsRootNode && ShowRootNodes)
            {
                nodeColor = Colors.Gold;
                nodeSize = 4.0; // Extra large for root
                CreateNodeShape(geometry, position, nodeSize, NodeShape.Diamond);
                
                // Add a glowing effect with a larger transparent diamond
                var glowGeometry = new MeshGeometry3D();
                CreateNodeShape(glowGeometry, position, nodeSize * 1.5, NodeShape.Diamond);
                var glowMaterial = new DiffuseMaterial(new SolidColorBrush(Color.FromArgb(64, 255, 215, 0))); // Semi-transparent gold
                var glowModel = new GeometryModel3D(glowGeometry, glowMaterial);
                scene.Children.Add(glowModel);
            }
            // Other root nodes
            else if (hierarchyInfo.IsRootNode && ShowRootNodes)
            {
                nodeColor = Colors.Orange;
                nodeSize = 3.0;
                CreateNodeShape(geometry, position, nodeSize, NodeShape.Diamond);
            }
            // Leaf nodes with enhanced colors
            else if (hierarchyInfo.IsLeafNode && ShowLeafNodes)
            {
                // Color leaf nodes based on their level depth
                var leafHue = 120f + (hierarchyInfo.HierarchyLevel * 10f); // Green spectrum
                nodeColor = HSVToRGB(leafHue / 360f, 0.7f, 0.9f);
                nodeSize = 2.0;
                CreateNodeShape(geometry, position, nodeSize, NodeShape.Triangle);
            }
            else
            {
                // Intermediate nodes - enhanced level-based coloring for 13 levels
                var maxLevels = 13f;
                var levelRatio = Math.Min(hierarchyInfo.HierarchyLevel / maxLevels, 1.0f);
                
                // Use a rainbow spectrum across the 13 levels
                var hue = levelRatio * 300f; // 0° (red) to 300° (magenta), avoiding green which is for leaves
                var saturation = 0.8f + (levelRatio * 0.2f); // Increase saturation with depth
                var brightness = 1.0f - (levelRatio * 0.3f); // Slightly dimmer for deeper levels
                
                nodeColor = HSVToRGB(hue / 360f, saturation, brightness);
                nodeSize = 2.5 - (levelRatio * 0.5); // Smaller nodes for deeper levels
                CreateNodeShape(geometry, position, nodeSize, NodeShape.Cube);
            }

            var material = new DiffuseMaterial(new SolidColorBrush(nodeColor));
            var model = new GeometryModel3D(geometry, material);
            scene.Children.Add(model);
        }

        private void CreateParentChildConnections(Model3DGroup scene)
        {
            if (CurrentHierarchyAnalysis == null) return;

            // Create different geometries for different hierarchy levels
            var connectionsByLevel = new Dictionary<int, MeshGeometry3D>();

            foreach (var kvp in CurrentHierarchyAnalysis.GroupHierarchy)
            {
                var childInfo = kvp.Value;
                if (!childInfo.ParentValue.HasValue) continue;

                // Skip if level filtering is active and this level doesn't match
                if (HierarchyLevelFilter > 0 && childInfo.HierarchyLevel != HierarchyLevelFilter)
                    continue;

                // Find parent and child positions
                var childGroup = Unknown0x04Groups.FirstOrDefault(g => g.GroupValue == childInfo.GroupValue);
                var parentGroup = Unknown0x04Groups.FirstOrDefault(g => g.GroupValue == childInfo.ParentValue.Value);

                if (childGroup?.AssociatedVertices.Any() == true && parentGroup?.AssociatedVertices.Any() == true)
                {
                    var childPos = CalculateGroupCenter(childGroup.AssociatedVertices);
                    var parentPos = CalculateGroupCenter(parentGroup.AssociatedVertices);

                    // Get or create geometry for this hierarchy level
                    if (!connectionsByLevel.ContainsKey(childInfo.HierarchyLevel))
                        connectionsByLevel[childInfo.HierarchyLevel] = new MeshGeometry3D();

                    // Make connections thicker for higher levels (closer to root)
                    var thickness = Math.Max(0.2f, 1.0f - (childInfo.HierarchyLevel * 0.06f));
                    
                    CreateConnection(connectionsByLevel[childInfo.HierarchyLevel], parentPos, childPos, thickness, Colors.Orange);
                }
            }

            // Add geometries to scene with level-based colors
            foreach (var kvp in connectionsByLevel)
            {
                if (kvp.Value.Positions.Count == 0) continue;

                var level = kvp.Key;
                var geometry = kvp.Value;

                // Color connections based on hierarchy level
                var levelRatio = Math.Min(level / 13f, 1.0f);
                var hue = 20f + (levelRatio * 40f); // Orange to red spectrum
                var connectionColor = HSVToRGB(hue / 360f, 0.9f, 1.0f);

                var material = new DiffuseMaterial(new SolidColorBrush(connectionColor));
                var model = new GeometryModel3D(geometry, material);
                scene.Children.Add(model);
            }
        }

        private void CreateCrossReferenceConnections(Model3DGroup scene)
        {
            if (CurrentHierarchyAnalysis?.CrossReferenceNetwork == null) return;

            var crossRefGeometry = new MeshGeometry3D();
            var highVolumeGeometry = new MeshGeometry3D(); // For nodes with many references

            foreach (var kvp in CurrentHierarchyAnalysis.CrossReferenceNetwork)
            {
                var sourceGroup = Unknown0x04Groups.FirstOrDefault(g => g.GroupValue == kvp.Key);
                if (sourceGroup?.AssociatedVertices.Any() != true) continue;

                var sourcePos = CalculateGroupCenter(sourceGroup.AssociatedVertices);
                var referenceCount = kvp.Value.Count;

                // Special highlighting for high-volume cross-reference nodes
                var isHighVolume = referenceCount > 5; // More than 5 references
                var targetGeometry = isHighVolume ? highVolumeGeometry : crossRefGeometry;
                var thickness = isHighVolume ? 0.4f : 0.2f;

                foreach (var targetGroupValue in kvp.Value)
                {
                    var targetGroup = Unknown0x04Groups.FirstOrDefault(g => g.GroupValue == targetGroupValue);
                    if (targetGroup?.AssociatedVertices.Any() == true)
                    {
                        var targetPos = CalculateGroupCenter(targetGroup.AssociatedVertices);
                        
                        // Create dashed-style connection for cross-references
                        CreateDashedConnection(targetGeometry, sourcePos, targetPos, thickness, Colors.Cyan);
                    }
                }
            }

            // Add regular cross-reference connections
            if (crossRefGeometry.Positions.Count > 0)
            {
                var material = new DiffuseMaterial(new SolidColorBrush(Colors.Cyan));
                var model = new GeometryModel3D(crossRefGeometry, material);
                scene.Children.Add(model);
            }

            // Add high-volume cross-reference connections with enhanced visibility
            if (highVolumeGeometry.Positions.Count > 0)
            {
                var material = new DiffuseMaterial(new SolidColorBrush(Colors.DeepSkyBlue));
                var model = new GeometryModel3D(highVolumeGeometry, material);
                scene.Children.Add(model);
            }
        }

        private void CreateNodeShape(MeshGeometry3D geometry, Point3D center, double size, NodeShape shape)
        {
            switch (shape)
            {
                case NodeShape.Cube:
                    AddVertexCube(geometry, center, size);
                    break;
                case NodeShape.Diamond:
                    CreateDiamond(geometry, center, size);
                    break;
                case NodeShape.Triangle:
                    CreateTriangle(geometry, center, size);
                    break;
            }
        }

        private void CreateDiamond(MeshGeometry3D geometry, Point3D center, double size)
        {
            var half = size / 2;
            var baseIndex = geometry.Positions.Count;

            // Diamond vertices (6 points)
            var vertices = new Point3D[]
            {
                new(center.X, center.Y, center.Z + half),     // top
                new(center.X, center.Y, center.Z - half),     // bottom
                new(center.X + half, center.Y, center.Z),     // right
                new(center.X - half, center.Y, center.Z),     // left
                new(center.X, center.Y + half, center.Z),     // front
                new(center.X, center.Y - half, center.Z)      // back
            };

            foreach (var vertex in vertices)
                geometry.Positions.Add(vertex);

            // Diamond faces (8 triangular faces)
            var indices = new int[]
            {
                0, 2, 4,  0, 4, 3,  0, 3, 5,  0, 5, 2,  // top pyramid
                1, 4, 2,  1, 3, 4,  1, 5, 3,  1, 2, 5   // bottom pyramid
            };

            foreach (var index in indices)
                geometry.TriangleIndices.Add(baseIndex + index);
        }

        private void CreateTriangle(MeshGeometry3D geometry, Point3D center, double size)
        {
            var half = size / 2;
            var height = size * 0.866; // sqrt(3)/2
            var baseIndex = geometry.Positions.Count;

            // Triangle vertices (3 points in XY plane, elevated in Z)
            var vertices = new Point3D[]
            {
                new(center.X, center.Y + height/2, center.Z),           // top
                new(center.X - half, center.Y - height/2, center.Z),    // bottom left
                new(center.X + half, center.Y - height/2, center.Z),    // bottom right
                new(center.X, center.Y, center.Z + size/3),             // elevated center
            };

            foreach (var vertex in vertices)
                geometry.Positions.Add(vertex);

            // Triangle faces
            var indices = new int[] { 0, 1, 3,  1, 2, 3,  2, 0, 3,  0, 2, 1 };

            foreach (var index in indices)
                geometry.TriangleIndices.Add(baseIndex + index);
        }

        private void CreateConnection(MeshGeometry3D geometry, Point3D start, Point3D end, float thickness, Color color)
        {
            var direction = end - start;
            var length = direction.Length;
            if (length < 0.001) return;

            direction.Normalize();
            
            // Create cylinder between points
            var perpendicular = Vector3D.CrossProduct(direction, new Vector3D(0, 0, 1));
            if (perpendicular.Length < 0.001)
                perpendicular = Vector3D.CrossProduct(direction, new Vector3D(1, 0, 0));
            perpendicular.Normalize();

            var baseIndex = geometry.Positions.Count;
            var segments = 8;

            // Create cylinder vertices
            for (int i = 0; i <= segments; i++)
            {
                var angle = (double)i / segments * 2 * Math.PI;
                var offset = perpendicular * thickness * Math.Cos(angle) + 
                           Vector3D.CrossProduct(direction, perpendicular) * thickness * Math.Sin(angle);

                geometry.Positions.Add(start + offset);
                geometry.Positions.Add(end + offset);
            }

            // Create cylinder faces
            for (int i = 0; i < segments; i++)
            {
                var current = baseIndex + i * 2;
                var next = baseIndex + ((i + 1) % segments) * 2;

                // Two triangles per face
                geometry.TriangleIndices.Add(current);
                geometry.TriangleIndices.Add(next);
                geometry.TriangleIndices.Add(current + 1);

                geometry.TriangleIndices.Add(current + 1);
                geometry.TriangleIndices.Add(next);
                geometry.TriangleIndices.Add(next + 1);
            }
        }

        private void CreateDashedConnection(MeshGeometry3D geometry, Point3D start, Point3D end, float thickness, Color color)
        {
            var direction = end - start;
            var length = direction.Length;
            var dashLength = 2.0;
            var gapLength = 1.0;
            var totalLength = dashLength + gapLength;

            var normalizedDirection = direction;
            normalizedDirection.Normalize();

            var numDashes = (int)(length / totalLength);
            
            for (int i = 0; i < numDashes; i++)
            {
                var dashStart = start + normalizedDirection * (i * totalLength);
                var dashEnd = start + normalizedDirection * (i * totalLength + dashLength);
                
                CreateConnection(geometry, dashStart, dashEnd, thickness, color);
            }
        }

        private enum NodeShape
        {
            Cube,
            Diamond, 
            Triangle
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
            if (Pm4File?.MSLK?.Entries == null || Unknown0x04Groups.Count == 0)
                return;

            foreach (var group in Unknown0x04Groups.Take(100)) // Limit for performance
            {
                var vertices = ExtractGroupVerticesAsPoints(group.AssociatedVertices);
                if (vertices.Count() == 0) continue;

                var geometry = new MeshGeometry3D();
                
                foreach (var vertex in vertices)
                {
                    AddVertexCube(geometry, vertex, 1.0);
                }

                var material = new DiffuseMaterial(new SolidColorBrush(group.Color));
                var model = new GeometryModel3D(geometry, material);
                
                scene.Children.Add(model);
            }
        }
        
        private void CreateUnknown0x0CGroupVisualization(Model3DGroup scene)
        {
            if (Pm4File?.MSLK?.Entries == null || Unknown0x0CGroups.Count == 0)
                return;

            foreach (var group in Unknown0x0CGroups.Take(100)) // Limit for performance
            {
                var vertices = ExtractGroupVerticesAsPoints(group.AssociatedVertices);
                if (vertices.Count() == 0) continue;

                var geometry = new MeshGeometry3D();
                
                foreach (var vertex in vertices)
                {
                    AddVertexCube(geometry, vertex, 0.8); // Slightly smaller to distinguish
                }

                var material = new DiffuseMaterial(new SolidColorBrush(group.Color));
                var model = new GeometryModel3D(geometry, material);
                
                scene.Children.Add(model);
            }
        }

        private void AddVertexCube(MeshGeometry3D geometry, Point3D point, double size)
        {
            var half = size / 2;
            var positions = new Point3D[]
            {
                new(point.X - half, point.Y - half, point.Z - half),
                new(point.X + half, point.Y - half, point.Z - half),
                new(point.X + half, point.Y + half, point.Z - half),
                new(point.X - half, point.Y + half, point.Z - half),
                new(point.X - half, point.Y - half, point.Z + half),
                new(point.X + half, point.Y - half, point.Z + half),
                new(point.X + half, point.Y + half, point.Z + half),
                new(point.X - half, point.Y + half, point.Z + half)
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

            foreach (var position in positions)
            {
                geometry.Positions.Add(position);
            }

            foreach (var index in indices)
            {
                geometry.TriangleIndices.Add(index);
            }
        }

        private IEnumerable<Point3D> ExtractGroupVerticesAsPoints(List<Vector3> vertices)
        {
            return vertices.Select(v => new Point3D(v.X, v.Y, v.Z));
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
        
        private void AnalyzeUnknown0x0CGroups()
        {
            // Ensure we're on the UI thread for collection updates
            if (System.Windows.Application.Current.Dispatcher.CheckAccess())
            {
                AnalyzeUnknown0x0CGroupsImpl();
            }
            else
            {
                System.Windows.Application.Current.Dispatcher.Invoke(AnalyzeUnknown0x0CGroupsImpl);
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
                .OrderByDescending(g => g.Count())
                .ToList();

            // Perform deep hierarchical analysis
            var hierarchyAnalysis = PerformDeepHierarchicalAnalysis(groupedEntries.Cast<IGrouping<uint, dynamic>>().ToList());

            // Generate distinct colors for groups using HSV
            int maxColors = Math.Min(groupedEntries.Count, 360); // One color per degree
            var colors = GenerateDistinctColors(maxColors);
            
            for (int i = 0; i < groupedEntries.Count; i++)
            {
                var group = groupedEntries[i];
                var entries = group.Select(x => x.Entry).ToList();
                
                // Calculate Unknown_0x0C statistics for this group
                var unknown0x0CValues = entries.Select(e => (float)e.Unknown_0x0C).ToList();
                var avgUnknown0x0C = unknown0x0CValues.Count > 0 ? unknown0x0CValues.Average() : 0f;
                var minUnknown0x0C = unknown0x0CValues.Count > 0 ? unknown0x0CValues.Min() : 0f;
                var maxUnknown0x0C = unknown0x0CValues.Count > 0 ? unknown0x0CValues.Max() : 0f;
                
                // Get hierarchical info for this group
                var hierarchyInfo = hierarchyAnalysis.GroupHierarchy.GetValueOrDefault(group.Key, new GroupHierarchyInfo());
                
                var groupInfo = new Unknown0x04Group
                {
                    GroupValue = group.Key,
                    EntryCount = group.Count(),
                    EntryIndices = group.Select(x => x.Index).ToList(),
                    GroupColor = i < maxColors ? colors[i] : Colors.Gray,
                    Color = i < maxColors ? colors[i] : Colors.Gray,
                    Description = AnalyzeGroupPattern(entries, hierarchyInfo),
                    AssociatedVertices = ExtractGroupVertices(entries),
                    AverageUnknown0x0C = avgUnknown0x0C,
                    MinUnknown0x0C = minUnknown0x0C,
                    MaxUnknown0x0C = maxUnknown0x0C,
                    HierarchyInfo = hierarchyInfo,
                    ChunkOwnership = AnalyzeChunkOwnership(entries)
                };
                
                Unknown0x04Groups.Add(groupInfo);
            }

            // Generate enhanced summary with hierarchy insights
            var topGroups = Unknown0x04Groups.Take(5).ToList();
            var hierarchyDepth = hierarchyAnalysis.MaxHierarchyDepth;
            var totalConnections = hierarchyAnalysis.TotalConnections;
            
            var summary = $"Found {Unknown0x04Groups.Count} groups (depth: {hierarchyDepth}, connections: {totalConnections}). " +
                $"Top 5: {string.Join(", ", topGroups.Select(g => $"0x{g.GroupValue:X8}({g.EntryCount})"))}";
            GroupsSummary = summary;

            // Add hierarchy insights to structural insights
            AddHierarchyInsights(hierarchyAnalysis);

            FilterUnknown0x04Groups();
        }
        
        private void AnalyzeUnknown0x0CGroupsImpl()
        {
            Unknown0x0CGroups.Clear();
            
            if (Pm4File?.MSLK?.Entries == null || Pm4File.MSLK.Entries.Count == 0)
            {
                FilterUnknown0x0CGroups();
                return;
            }

            // Group MSLK entries by Unknown_0x0C value
            var groupedEntries = Pm4File.MSLK.Entries
                .Select((entry, index) => new { Entry = entry, Index = index })
                .GroupBy(x => x.Entry.Unknown_0x0C)
                .OrderByDescending(g => g.Count())
                .ToList();

            // Generate distinct colors for groups using HSV
            int maxColors = Math.Min(groupedEntries.Count, 360); // One color per degree
            var colors = GenerateDistinctColors(maxColors);
            
            for (int i = 0; i < groupedEntries.Count; i++)
            {
                var group = groupedEntries[i];
                var entries = group.Select(x => x.Entry).ToList();
                
                // Calculate Unknown_0x04 statistics for this group
                var unknown0x04Values = entries.Select(e => (float)e.Unknown_0x04).ToList();
                var avgUnknown0x04 = unknown0x04Values.Count > 0 ? unknown0x04Values.Average() : 0f;
                var minUnknown0x04 = unknown0x04Values.Count > 0 ? unknown0x04Values.Min() : 0f;
                var maxUnknown0x04 = unknown0x04Values.Count > 0 ? unknown0x04Values.Max() : 0f;
                
                var groupInfo = new Unknown0x0CGroup
                {
                    GroupValue = group.Key,
                    EntryCount = group.Count(),
                    EntryIndices = group.Select(x => x.Index).ToList(),
                    GroupColor = i < maxColors ? colors[i] : Colors.Gray,
                    Color = i < maxColors ? colors[i] : Colors.Gray,
                    Description = AnalyzeGroupPattern(entries, new GroupHierarchyInfo()), // Use empty hierarchy info for 0x0C groups
                    AssociatedVertices = ExtractGroupVertices(entries),
                    AverageUnknown0x04 = avgUnknown0x04,
                    MinUnknown0x04 = minUnknown0x04,
                    MaxUnknown0x04 = maxUnknown0x04
                };
                
                Unknown0x0CGroups.Add(groupInfo);
            }

            FilterUnknown0x0CGroups();
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
        
        private void FilterUnknown0x0CGroups()
        {
            // Ensure we're on the UI thread for collection updates
            if (System.Windows.Application.Current.Dispatcher.CheckAccess())
            {
                FilterUnknown0x0CGroupsImpl();
            }
            else
            {
                System.Windows.Application.Current.Dispatcher.Invoke(FilterUnknown0x0CGroupsImpl);
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
        
        private void FilterUnknown0x0CGroupsImpl()
        {
            if (Unknown0x0CGroups.Count == 0)
            {
                FilteredUnknown0x0CGroups.Clear();
                return;
            }
            
            var filtered = Unknown0x0CGroups.AsEnumerable();
            
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
            
            FilteredUnknown0x0CGroups.Clear();
            foreach (var group in filtered)
            {
                FilteredUnknown0x0CGroups.Add(group);
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

        private string AnalyzeGroupPattern(List<MSLKEntry> entries, GroupHierarchyInfo hierarchyInfo)
        {
            if (entries.Count == 1)
                return $"Single entry{(hierarchyInfo.IsLeafNode ? " [Leaf]" : "")}{(hierarchyInfo.IsRootNode ? " [Root]" : "")}";
                
            // Analyze patterns in the group
            var hasGeometry = entries.Any(e => e.MspiFirstIndex >= 0 && e.MspiIndexCount > 0);
            var hasDoodads = entries.Any(e => e.MspiFirstIndex == -1);
            var avgUnknown0x0C = entries.Average(e => e.Unknown_0x0C);
            
            var pattern = new List<string>();
            if (hasGeometry) pattern.Add("Geometry");
            if (hasDoodads) pattern.Add("Doodads");
            
            // Add hierarchy information
            var hierarchyTags = new List<string>();
            if (hierarchyInfo.IsRootNode) hierarchyTags.Add("Root");
            if (hierarchyInfo.IsLeafNode) hierarchyTags.Add("Leaf");
            if (hierarchyInfo.ChildCount > 0) hierarchyTags.Add($"{hierarchyInfo.ChildCount} children");
            if (hierarchyInfo.HierarchyLevel > 0) hierarchyTags.Add($"L{hierarchyInfo.HierarchyLevel}");
            
            var description = pattern.Count > 0 ? string.Join(" + ", pattern) : "Unknown";
            var hierarchyDesc = hierarchyTags.Count > 0 ? $" [{string.Join(", ", hierarchyTags)}]" : "";
            
            return $"{description} (avg 0x0C: {avgUnknown0x0C:F1}){hierarchyDesc}";
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
            UpdateCameraInfo();
            RefreshLegend();
            
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

        #region Deep Hierarchical Analysis

        private HierarchyAnalysisResult PerformDeepHierarchicalAnalysis(List<IGrouping<uint, dynamic>> groupedEntries, CancellationToken cancellationToken = default)
        {
            var result = new HierarchyAnalysisResult();
            
            if (Pm4File?.MSLK?.Entries == null)
                return result;

            cancellationToken.ThrowIfCancellationRequested();
            
            // 1. Analyze numerical hierarchy patterns in Unknown_0x04 values
            AnalyzeNumericalHierarchy(groupedEntries, result, cancellationToken);
            
            cancellationToken.ThrowIfCancellationRequested();
            
            // 2. Analyze Unknown_0x10 cross-reference relationships
            AnalyzeUnknown0x10Relationships(result, cancellationToken);
            
            cancellationToken.ThrowIfCancellationRequested();
            
            // 3. Analyze chunk ownership patterns
            AnalyzeChunkOwnershipHierarchy(groupedEntries, result, cancellationToken);
            
            cancellationToken.ThrowIfCancellationRequested();
            
            // 4. Analyze spatial hierarchy patterns
            AnalyzeSpatialHierarchy(groupedEntries, result, cancellationToken);
            
            cancellationToken.ThrowIfCancellationRequested();
            
            // 5. Calculate hierarchy metrics
            CalculateHierarchyMetrics(result, cancellationToken);
            
            return result;
        }

        private void AnalyzeNumericalHierarchy(List<IGrouping<uint, dynamic>> groupedEntries, HierarchyAnalysisResult result, CancellationToken cancellationToken = default)
        {
            var groupValues = groupedEntries.Select(g => g.Key).OrderBy(x => x).ToList();
            
            foreach (var groupValue in groupValues)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var hierarchyInfo = new GroupHierarchyInfo
                {
                    GroupValue = groupValue,
                    IsRootNode = true,
                    IsLeafNode = true,
                    HierarchyLevel = 0
                };

                // Check for numerical parent-child relationships
                // Pattern 1: Bit masking hierarchy (0x1000 parent of 0x1001, 0x1002, etc.)
                var potentialParents = groupValues.Where(v => v < groupValue && (groupValue & v) == v).ToList();
                if (potentialParents.Any())
                {
                    var closestParent = potentialParents.Max();
                    hierarchyInfo.ParentValue = closestParent;
                    hierarchyInfo.IsRootNode = false;
                    
                    if (result.GroupHierarchy.ContainsKey(closestParent))
                    {
                        result.GroupHierarchy[closestParent].IsLeafNode = false;
                        result.GroupHierarchy[closestParent].ChildCount++;
                    }
                }
                
                // Pattern 2: Sequential hierarchy (0x1000, 0x1100, 0x1200 as siblings under common parent)
                var baseValue = groupValue & 0xFFF00000; // Check upper bits
                if (baseValue != 0 && baseValue != groupValue)
                {
                    var sequentialParents = groupValues.Where(v => (v & 0xFFF00000) == baseValue && v < groupValue).ToList();
                    if (sequentialParents.Any() && !hierarchyInfo.ParentValue.HasValue)
                    {
                        hierarchyInfo.ParentValue = baseValue;
                        hierarchyInfo.IsRootNode = false;
                    }
                }

                result.GroupHierarchy[groupValue] = hierarchyInfo;
            }
        }

        private void AnalyzeUnknown0x10Relationships(HierarchyAnalysisResult result, CancellationToken cancellationToken = default)
        {
            if (Pm4File?.MSLK?.Entries == null) return;

            // Analyze Unknown_0x10 as potential cross-references creating hierarchies
            var unknown0x10References = new Dictionary<uint, List<uint>>();
            
            foreach (var entry in Pm4File.MSLK.Entries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                // Check if Unknown_0x10 points to valid indices in other structures
                if (entry.Unknown_0x10 < Pm4File.MSLK.Entries.Count)
                {
                    var referencedEntry = Pm4File.MSLK.Entries[entry.Unknown_0x10];
                    var sourceGroup = entry.Unknown_0x04;
                    var targetGroup = referencedEntry.Unknown_0x04;
                    
                    if (sourceGroup != targetGroup)
                    {
                        if (!unknown0x10References.ContainsKey(sourceGroup))
                            unknown0x10References[sourceGroup] = new List<uint>();
                        
                        unknown0x10References[sourceGroup].Add(targetGroup);
                        result.TotalConnections++;
                        
                        // Update hierarchy info with cross-references
                        if (result.GroupHierarchy.ContainsKey(sourceGroup))
                        {
                            result.GroupHierarchy[sourceGroup].CrossReferences.Add(targetGroup);
                        }
                    }
                }
            }

            result.CrossReferenceNetwork = unknown0x10References;
        }

        private void AnalyzeChunkOwnershipHierarchy(List<IGrouping<uint, dynamic>> groupedEntries, HierarchyAnalysisResult result, CancellationToken cancellationToken = default)
        {
            if (Pm4File?.MSUR?.Entries == null || Pm4File?.MSVI?.Indices == null) return;

            // Analyze which groups "own" which MSUR surfaces and vertex ranges
            foreach (var group in groupedEntries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var entries = group.Select(x => x.Entry).Cast<MSLKEntry>().ToList();
                var ownership = new ChunkOwnershipInfo();
                
                // Analyze MSVI index ownership
                var msviIndices = new HashSet<uint>();
                foreach (var entry in entries)
                {
                    if (entry.Unknown_0x10 < (Pm4File.MSVI?.Indices.Count ?? 0))
                    {
                        msviIndices.Add((uint)entry.Unknown_0x10);
                    }
                }
                ownership.OwnedMSVIIndices = msviIndices.ToList();
                
                // Analyze MSVT vertex range ownership through MSVI
                var msvtVertices = new HashSet<uint>();
                foreach (var msviIndex in msviIndices)
                {
                    if (msviIndex < (Pm4File.MSVI?.Indices.Count ?? 0))
                    {
                        var vertexIndex = Pm4File.MSVI!.Indices[(int)msviIndex];
                        if (vertexIndex < (Pm4File.MSVT?.Vertices.Count ?? 0))
                        {
                            msvtVertices.Add(vertexIndex);
                        }
                    }
                }
                ownership.OwnedMSVTVertices = msvtVertices.ToList();
                
                // Calculate ownership statistics
                ownership.VertexRangeStart = msvtVertices.Count > 0 ? msvtVertices.Min() : 0;
                ownership.VertexRangeEnd = msvtVertices.Count > 0 ? msvtVertices.Max() : 0;
                ownership.VertexRangeSize = msvtVertices.Count;
                
                result.ChunkOwnership[group.Key] = ownership;
            }
        }

        private void AnalyzeSpatialHierarchy(List<IGrouping<uint, dynamic>> groupedEntries, HierarchyAnalysisResult result, CancellationToken cancellationToken = default)
        {
            // Analyze spatial clustering to detect hierarchical patterns
            foreach (var group in groupedEntries)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var entries = group.Select(x => x.Entry).Cast<MSLKEntry>().ToList();
                var vertices = ExtractGroupVertices(entries);
                
                if (vertices.Count > 0)
                {
                    // Calculate spatial statistics
                    var centerX = vertices.Average(v => v.X);
                    var centerY = vertices.Average(v => v.Y);
                    var centerZ = vertices.Average(v => v.Z);
                    
                    var maxDistance = vertices.Max(v => 
                        Math.Sqrt(Math.Pow(v.X - centerX, 2) + Math.Pow(v.Y - centerY, 2) + Math.Pow(v.Z - centerZ, 2)));
                    
                    if (result.GroupHierarchy.ContainsKey(group.Key))
                    {
                        result.GroupHierarchy[group.Key].SpatialCenter = new Vector3((float)centerX, (float)centerY, (float)centerZ);
                        result.GroupHierarchy[group.Key].SpatialRadius = (float)maxDistance;
                    }
                }
            }
        }

        private void CalculateHierarchyMetrics(HierarchyAnalysisResult result, CancellationToken cancellationToken = default)
        {
            // Calculate hierarchy depth
            foreach (var group in result.GroupHierarchy.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                CalculateHierarchyLevel(group, result.GroupHierarchy);
            }
            
            result.MaxHierarchyDepth = result.GroupHierarchy.Values.Any() ? result.GroupHierarchy.Values.Max(g => g.HierarchyLevel) : 0;
            result.RootNodes = result.GroupHierarchy.Values.Count(g => g.IsRootNode);
            result.LeafNodes = result.GroupHierarchy.Values.Count(g => g.IsLeafNode);
        }

        private void CalculateHierarchyLevel(GroupHierarchyInfo group, Dictionary<uint, GroupHierarchyInfo> hierarchy)
        {
            if (group.HierarchyLevel > 0) return; // Already calculated
            
            if (group.IsRootNode)
            {
                group.HierarchyLevel = 1;
            }
            else if (group.ParentValue.HasValue && hierarchy.ContainsKey(group.ParentValue.Value))
            {
                var parent = hierarchy[group.ParentValue.Value];
                CalculateHierarchyLevel(parent, hierarchy);
                group.HierarchyLevel = parent.HierarchyLevel + 1;
            }
            else
            {
                group.HierarchyLevel = 1; // Treat as root if parent not found
            }
        }

        private ChunkOwnershipInfo AnalyzeChunkOwnership(List<MSLKEntry> entries)
        {
            var ownership = new ChunkOwnershipInfo();
            
            if (Pm4File?.MSVI?.Indices == null) return ownership;
            
            // Analyze what chunks this group references
            var msviIndices = new HashSet<uint>();
            var msvtVertices = new HashSet<uint>();
            
            foreach (var entry in entries)
            {
                if (entry.Unknown_0x10 < Pm4File.MSVI.Indices.Count)
                {
                    msviIndices.Add((uint)entry.Unknown_0x10);
                    
                    var vertexIndex = Pm4File.MSVI.Indices[entry.Unknown_0x10];
                    if (vertexIndex < (Pm4File.MSVT?.Vertices.Count ?? 0))
                    {
                        msvtVertices.Add(vertexIndex);
                    }
                }
            }
            
            ownership.OwnedMSVIIndices = msviIndices.ToList();
            ownership.OwnedMSVTVertices = msvtVertices.ToList();
            ownership.VertexRangeSize = msvtVertices.Count;
            
            if (msvtVertices.Count > 0)
            {
                ownership.VertexRangeStart = msvtVertices.Min();
                ownership.VertexRangeEnd = msvtVertices.Max();
            }
            
            return ownership;
        }

        private void AddHierarchyInsights(HierarchyAnalysisResult analysis)
        {
            // Add hierarchy insights to structural insights collection
            StructuralInsights.Add(new StructuralInsight
            {
                Type = "Hierarchy Analysis",
                Description = $"Detected {analysis.GroupHierarchy.Count} groups with {analysis.MaxHierarchyDepth} levels deep",
                Significance = $"Root nodes: {analysis.RootNodes}, Leaf nodes: {analysis.LeafNodes}, Cross-connections: {analysis.TotalConnections}",
                DataPreview = $"Max spatial radius: {(analysis.GroupHierarchy.Values.Where(g => g.SpatialRadius > 0).Any() ? analysis.GroupHierarchy.Values.Where(g => g.SpatialRadius > 0).Max(g => g.SpatialRadius) : 0):F1}"
            });

            // Add numerical hierarchy patterns
            var numericalPatterns = analysis.GroupHierarchy.Values.Where(g => g.ParentValue.HasValue).ToList();
            if (numericalPatterns.Any())
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Numerical Hierarchy",
                    Description = $"Found {numericalPatterns.Count} parent-child relationships in Unknown_0x04 values",
                    Significance = "Bit masking or sequential hierarchy patterns detected",
                    DataPreview = string.Join(", ", numericalPatterns.Take(3).Select(p => $"0x{p.GroupValue:X8}→0x{p.ParentValue:X8}"))
                });
            }

            // Add cross-reference insights
            if (analysis.CrossReferenceNetwork.Any())
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Cross-Reference Network",
                    Description = $"Unknown_0x10 creates {analysis.TotalConnections} cross-group connections",
                    Significance = "Potential navigation or dependency relationships",
                    DataPreview = string.Join(", ", analysis.CrossReferenceNetwork.Take(3).Select(kvp => $"0x{kvp.Key:X8}→[{kvp.Value.Count} refs]"))
                });
            }

            // Add chunk ownership insights
            var exclusiveOwnership = analysis.ChunkOwnership.Values.Where(o => o.VertexRangeSize > 0).ToList();
            if (exclusiveOwnership.Any())
            {
                var totalVerticesOwned = exclusiveOwnership.Sum(o => o.VertexRangeSize);
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Chunk Ownership",
                    Description = $"Groups have exclusive ownership of {totalVerticesOwned} vertices across {exclusiveOwnership.Count} ranges",
                    Significance = "Groups control distinct geometry regions",
                    DataPreview = $"Largest range: {exclusiveOwnership.Max(o => o.VertexRangeSize)} vertices"
                });
            }
        }

        #endregion

        #region Coordinate Bounds Validation

        private string ValidateCoordinateBounds(List<Point3D> points, string chunkType)
        {
            if (!points.Any()) return "";

            var outOfBounds = points.Where(p => 
                p.X < PM4_MIN_COORDINATE || p.X > PM4_MAX_COORDINATE ||
                p.Y < PM4_MIN_COORDINATE || p.Y > PM4_MAX_COORDINATE ||
                p.Z < PM4_MIN_COORDINATE || p.Z > PM4_MAX_COORDINATE).ToList();

            if (outOfBounds.Any())
            {
                var percentage = (outOfBounds.Count * 100.0) / points.Count;
                
                // Add insight about out-of-bounds vertices
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Coordinate Bounds",
                    Description = $"{chunkType} has {outOfBounds.Count} vertices outside PM4 coordinate bounds",
                    Significance = $"{percentage:F1}% of vertices exceed ±{PM4_COORDINATE_BOUND:F0} limits",
                    DataPreview = $"Range: X[{points.Min(p => p.X):F0}, {points.Max(p => p.X):F0}] Y[{points.Min(p => p.Y):F0}, {points.Max(p => p.Y):F0}] Z[{points.Min(p => p.Z):F0}, {points.Max(p => p.Z):F0}]"
                });

                return $" ⚠️{outOfBounds.Count}OOB";
            }

            // Calculate actual bounds for information
            var minX = points.Min(p => p.X);
            var maxX = points.Max(p => p.X);
            var minY = points.Min(p => p.Y);
            var maxY = points.Max(p => p.Y);
            var minZ = points.Min(p => p.Z);
            var maxZ = points.Max(p => p.Z);

            // Add insight about coordinate usage
            var rangeX = maxX - minX;
            var rangeY = maxY - minY;
            var rangeZ = maxZ - minZ;
            var percentageUsedX = (rangeX / (2 * PM4_COORDINATE_BOUND)) * 100;
            var percentageUsedY = (rangeY / (2 * PM4_COORDINATE_BOUND)) * 100;
            var percentageUsedZ = (rangeZ / (2 * PM4_COORDINATE_BOUND)) * 100;

            if (points.Count > 100) // Only add insight for significant chunks
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Coordinate Usage",
                    Description = $"{chunkType} uses {percentageUsedX:F1}%×{percentageUsedY:F1}%×{percentageUsedZ:F1}% of PM4 coordinate space",
                    Significance = $"Spatial distribution across coordinate bounds",
                    DataPreview = $"Center: ({(minX + maxX) / 2:F0}, {(minY + maxY) / 2:F0}, {(minZ + maxZ) / 2:F0})"
                });
            }

            return " ✓";
        }

        #endregion

        #region Enhanced Control Methods

        private void InitializeLegend()
        {
            RefreshLegend();
        }

        private void RefreshLegend()
        {
            LegendItems.Clear();
            
            // Coordinate system items
            if (ShowGroundGrid)
            {
                LegendItems.Add(new LegendItem("Ground Grid", Colors.Gray, "Reference grid at Y=0", "🌐"));
            }
            
            if (ShowCoordinateBounds)
            {
                LegendItems.Add(new LegendItem("PM4 Bounds", Colors.Yellow, "±17066.666 coordinate limits", "📦"));
            }
            
            if (ShowCoordinateAxes)
            {
                LegendItems.Add(new LegendItem("Coordinate Axes", Colors.White, "Red=X, Green=Y, Blue=Z", "📐"));
            }
            
            // Data type items
            if (Pm4File != null)
            {
                if (ShowMSVTVertices && Pm4File.MSVT?.Vertices?.Count > 0)
                {
                    LegendItems.Add(new LegendItem("MSVT Vertices", Colors.Blue, 
                        $"Render vertices ({Pm4File.MSVT.Vertices.Count:N0})", "🔵"));
                }
                
                if (ShowMSCNPoints && Pm4File.MSCN?.ExteriorVertices?.Count > 0)
                {
                    LegendItems.Add(new LegendItem("MSCN Points", Colors.Red, 
                        $"Collision points ({Pm4File.MSCN.ExteriorVertices.Count:N0})", "🔴"));
                }
                
                if (ShowMSPVVertices && Pm4File.MSPV?.Vertices?.Count > 0)
                {
                    LegendItems.Add(new LegendItem("MSPV Vertices", Colors.Green, 
                        $"Structure vertices ({Pm4File.MSPV.Vertices.Count:N0})", "🟢"));
                }
                
                if (ShowGroupConnections)
                {
                    LegendItems.Add(new LegendItem("Group Connections", Colors.Orange, 
                        "Links between Unknown_0x04 groups", "🔗"));
                }
                
                if (ShowHierarchyLines)
                {
                    LegendItems.Add(new LegendItem("Hierarchy Lines", Colors.Purple, 
                        "Parent-child relationships", "🌳"));
                }
                
                if (ShowHierarchyTree && CurrentHierarchyAnalysis != null)
                {
                    // Special highlighting for the master root node
                    LegendItems.Add(new LegendItem("🌟 Master Root (0x00000000)", Colors.Gold, 
                        "Golden diamond with glow effect", "💎"));
                    
                    if (CurrentHierarchyAnalysis.RootNodes > 1)
                    {
                        LegendItems.Add(new LegendItem("Other Root Nodes", Colors.Orange, 
                            $"{CurrentHierarchyAnalysis.RootNodes - 1} additional roots", "🔶"));
                    }
                    
                    LegendItems.Add(new LegendItem("Leaf Nodes", Colors.LightGreen, 
                        $"{CurrentHierarchyAnalysis.LeafNodes} terminal triangles (green spectrum)", "🔺"));
                    
                    LegendItems.Add(new LegendItem("Level 1-13 Intermediate", Colors.Purple, 
                        "Rainbow spectrum by hierarchy depth", "🔲"));
                    
                    if (ShowParentChildConnections)
                    {
                        LegendItems.Add(new LegendItem("Parent-Child Links", Colors.OrangeRed, 
                            $"{CurrentHierarchyAnalysis.GroupHierarchy.Values.Count(g => g.ParentValue.HasValue)} hierarchical connections", "🔗"));
                    }
                    
                    if (ShowCrossReferences)
                    {
                        LegendItems.Add(new LegendItem("Cross-References", Colors.Cyan, 
                            $"{CurrentHierarchyAnalysis.TotalConnections} Unknown_0x10 connections", "⚡"));
                        LegendItems.Add(new LegendItem("High-Volume Cross-Refs", Colors.DeepSkyBlue, 
                            "Nodes with >5 references (thicker lines)", "⚡"));
                    }
                    
                    LegendItems.Add(new LegendItem("📊 Hierarchy Stats", Colors.White, 
                        $"{CurrentHierarchyAnalysis.MaxHierarchyDepth} levels, {CurrentHierarchyAnalysis.GroupHierarchy.Count} total groups", "📈"));
                }
                
                // Unknown_0x04 group colors
                if (ColorByUnknown0x04 && Unknown0x04Groups.Count > 0)
                {
                    var visibleGroups = ShowOnlySelectedGroup && SelectedUnknown0x04Group != null
                        ? new[] { SelectedUnknown0x04Group }
                        : Unknown0x04Groups.Take(10); // Show first 10 in legend
                    
                    foreach (var group in visibleGroups)
                    {
                        LegendItems.Add(new LegendItem($"Group 0x{group.GroupValue:X8}", 
                            group.GroupColor, $"{group.EntryCount} entries", "⬛"));
                    }
                    
                    if (Unknown0x04Groups.Count > 10)
                    {
                        LegendItems.Add(new LegendItem("... more groups", Colors.LightGray, 
                            $"{Unknown0x04Groups.Count - 10} additional groups", "⬜"));
                    }
                }
            }
        }

        private void FitCameraToData()
        {
            if (Pm4File == null) return;
            
            var allPoints = new List<Point3D>();
            
            // Collect all visible data points
            if (ShowMSVTVertices && Pm4File.MSVT?.Vertices != null)
            {
                allPoints.AddRange(Pm4File.MSVT.Vertices.Select(v => new Point3D(v.Y, v.X, v.Z)));
            }
            
            if (ShowMSCNPoints && Pm4File.MSCN?.ExteriorVertices != null)
            {
                allPoints.AddRange(Pm4File.MSCN.ExteriorVertices.Select(v => new Point3D(v.X, -v.Y, v.Z)));
            }
            
            if (ShowMSPVVertices && Pm4File.MSPV?.Vertices != null)
            {
                allPoints.AddRange(Pm4File.MSPV.Vertices.Select(v => new Point3D(v.X, v.Y, v.Z)));
            }
            
            if (allPoints.Count == 0)
            {
                ResetCamera();
                return;
            }
            
            // Calculate bounding box
            var minX = allPoints.Min(p => p.X);
            var maxX = allPoints.Max(p => p.X);
            var minY = allPoints.Min(p => p.Y);
            var maxY = allPoints.Max(p => p.Y);
            var minZ = allPoints.Min(p => p.Z);
            var maxZ = allPoints.Max(p => p.Z);
            
            var center = new Point3D(
                (minX + maxX) / 2,
                (minY + maxY) / 2,
                (minZ + maxZ) / 2
            );
            
            var size = Math.Max(Math.Max(maxX - minX, maxY - minY), maxZ - minZ);
            var distance = size * 2; // Move camera back enough to see everything
            
            CameraPosition = new Point3D(center.X + distance, center.Y + distance, center.Z + distance);
            CameraLookDirection = new Vector3D(center.X - CameraPosition.X, center.Y - CameraPosition.Y, center.Z - CameraPosition.Z);
            
            UpdateCameraInfo();
        }

        private void ResetCamera()
        {
            CameraPosition = new Point3D(100, 100, 100);
            CameraLookDirection = new Vector3D(-1, -1, -1);
            UpdateCameraInfo();
        }

        private void JumpToChunk(string? chunkType)
        {
            if (Pm4File == null || string.IsNullOrEmpty(chunkType)) return;
            
            List<Point3D> points = new();
            
            switch (chunkType?.ToUpper())
            {
                case "MSVT":
                    if (Pm4File.MSVT?.Vertices != null)
                        points.AddRange(Pm4File.MSVT.Vertices.Select(v => new Point3D(v.Y, v.X, v.Z)));
                    break;
                case "MSCN":
                    if (Pm4File.MSCN?.ExteriorVertices != null)
                        points.AddRange(Pm4File.MSCN.ExteriorVertices.Select(v => new Point3D(v.X, -v.Y, v.Z)));
                    break;
                case "MSPV":
                    if (Pm4File.MSPV?.Vertices != null)
                        points.AddRange(Pm4File.MSPV.Vertices.Select(v => new Point3D(v.X, v.Y, v.Z)));
                    break;
            }
            
            if (points.Count == 0) return;
            
            var center = new Point3D(
                points.Average(p => p.X),
                points.Average(p => p.Y),
                points.Average(p => p.Z)
            );
            
            var size = Math.Max(Math.Max(
                points.Max(p => p.X) - points.Min(p => p.X),
                points.Max(p => p.Y) - points.Min(p => p.Y)),
                points.Max(p => p.Z) - points.Min(p => p.Z));
            
            var distance = size * 1.5;
            
            CameraPosition = new Point3D(center.X + distance, center.Y + distance, center.Z + distance);
            CameraLookDirection = new Vector3D(center.X - CameraPosition.X, center.Y - CameraPosition.Y, center.Z - CameraPosition.Z);
            
            UpdateCameraInfo();
        }

        private void UpdateCameraInfo()
        {
            CameraInfo = $"Position: ({CameraPosition.X:F1}, {CameraPosition.Y:F1}, {CameraPosition.Z:F1})";
            
            // Update selected data info if we have any
            if (SelectedUnknown0x04Group != null)
            {
                SelectedDataInfo = $"Selected: Group 0x{SelectedUnknown0x04Group.GroupValue:X8} " +
                                   $"({SelectedUnknown0x04Group.EntryCount} entries, {SelectedUnknown0x04Group.AssociatedVertices.Count} vertices)";
            }
            else
            {
                var totalVertices = 0;
                if (Pm4File?.MSVT?.Vertices != null) totalVertices += Pm4File.MSVT.Vertices.Count;
                if (Pm4File?.MSCN?.ExteriorVertices != null) totalVertices += Pm4File.MSCN.ExteriorVertices.Count;
                if (Pm4File?.MSPV?.Vertices != null) totalVertices += Pm4File.MSPV.Vertices.Count;
                
                SelectedDataInfo = $"Total: {totalVertices:N0} vertices across {ChunkItems.Count} chunk types";
            }
        }

        #endregion

        #region Batch Loading Methods

        private async Task LoadBatchFilesAsync()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "PM4 Files (*.pm4)|*.pm4|All Files (*.*)|*.*",
                Title = "Select PM4 Files for Batch Loading",
                Multiselect = true
            };

            if (dialog.ShowDialog() == true)
            {
                await LoadMultipleFilesAsync(dialog.FileNames);
            }
        }

        private async Task LoadDirectoryAsync()
        {
            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "Select Directory Containing PM4 Files"
            };

            if (dialog.ShowDialog() == true)
            {
                var pm4Files = Directory.GetFiles(dialog.FolderName, "*.pm4", SearchOption.AllDirectories);
                if (pm4Files.Length == 0)
                {
                    MessageBox.Show("No PM4 files found in the selected directory.", "No Files Found", MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                var result = MessageBox.Show($"Found {pm4Files.Length} PM4 files. This may take a long time and use significant memory. Continue?", 
                                           "Batch Load Confirmation", MessageBoxButton.YesNo, MessageBoxImage.Warning);
                
                if (result == MessageBoxResult.Yes)
                {
                    await LoadMultipleFilesAsync(pm4Files);
                }
            }
        }

        private async Task LoadMultipleFilesAsync(string[] filePaths)
        {
            IsLoading = true;
            LoadingOperation = "Batch Loading PM4 Files";
            
            var loadedCount = 0;
            var totalVertices = 0L;
            var totalEntries = 0L;
            var errors = new List<string>();

            try
            {
                LoadedFiles.Clear();
                
                foreach (var filePath in filePaths)
                {
                    try
                    {
                        LoadingSubOperation = $"Loading {Path.GetFileName(filePath)} ({loadedCount + 1}/{filePaths.Length})";
                        await Task.Delay(10); // Allow UI to update

                        var file = await Task.Run(() =>
                        {
                            var fileBytes = File.ReadAllBytes(filePath);
                            return new PM4File(fileBytes);
                        });

                        LoadedFiles.Add(file);
                        loadedCount++;
                        
                        // Accumulate statistics
                        totalVertices += file.MSVT?.Vertices?.Count ?? 0;
                        totalEntries += file.MSLK?.Entries?.Count ?? 0;
                        
                        // Update progress every 10 files
                        if (loadedCount % 10 == 0)
                        {
                            Application.Current.Dispatcher.Invoke(() =>
                            {
                                TotalFilesLoaded = loadedCount;
                                TotalVerticesLoaded = totalVertices;
                                TotalEntriesLoaded = totalEntries;
                                UpdateBatchSummary();
                            });
                        }
                    }
                    catch (Exception ex)
                    {
                        errors.Add($"{Path.GetFileName(filePath)}: {ex.Message}");
                    }
                }

                // Final update
                TotalFilesLoaded = loadedCount;
                TotalVerticesLoaded = totalVertices;
                TotalEntriesLoaded = totalEntries;
                UpdateBatchSummary();
                ShowBatchSummary = true;

                // Show results
                var message = $"Batch loading complete!\n\n" +
                             $"✅ Loaded: {loadedCount}/{filePaths.Length} files\n" +
                             $"📊 Total Vertices: {totalVertices:N0}\n" +
                             $"📋 Total MSLK Entries: {totalEntries:N0}";

                if (errors.Any())
                {
                    message += $"\n\n⚠️ Errors: {errors.Count}\n" +
                              string.Join("\n", errors.Take(5));
                    if (errors.Count > 5)
                        message += $"\n... and {errors.Count - 5} more errors";
                }

                MessageBox.Show(message, "Batch Load Complete", MessageBoxButton.OK, MessageBoxImage.Information);

                // Perform batch analysis if files were loaded
                if (loadedCount > 0)
                {
                    await PerformBatchAnalysisAsync();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during batch loading: {ex.Message}", "Batch Load Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                IsLoading = false;
            }
        }

        private void UpdateBatchSummary()
        {
            BatchSummary = $"Batch Loaded: {TotalFilesLoaded} files | {TotalVerticesLoaded:N0} vertices | {TotalEntriesLoaded:N0} MSLK entries";
        }

        private async Task PerformBatchAnalysisAsync()
        {
            LoadingOperation = "Performing Batch Analysis";
            LoadingSubOperation = "Analyzing patterns across all files...";

            try
            {
                await Task.Run(() =>
                {
                    // Collect all Unknown_0x04 values across all files
                    var allUnknown04Values = new List<uint>();
                    var allUnknown0CValues = new List<uint>();
                    var fileGroupCounts = new Dictionary<string, int>();

                    foreach (var file in LoadedFiles)
                    {
                        if (file.MSLK?.Entries != null)
                        {
                            var fileName = "Unknown"; // We'd need to store the filename with the PM4File
                            
                            var unknown04Values = file.MSLK.Entries.Select(e => e.Unknown_0x04).ToList();
                            var unknown0CValues = file.MSLK.Entries.Select(e => e.Unknown_0x0C).ToList();
                            
                            allUnknown04Values.AddRange(unknown04Values);
                            allUnknown0CValues.AddRange(unknown0CValues);
                            
                            var groupCount = unknown04Values.Distinct().Count();
                            fileGroupCounts[fileName] = groupCount;
                        }
                    }

                    // Add batch insights
                    StructuralInsights.Add(new StructuralInsight
                    {
                        Type = "Batch Analysis",
                        Description = $"Cross-file pattern analysis of {LoadedFiles.Count} PM4 files",
                        Significance = $"Found {allUnknown04Values.Distinct().Count()} unique Unknown_0x04 values across all files",
                        DataPreview = $"Total entries analyzed: {allUnknown04Values.Count:N0}"
                    });

                    StructuralInsights.Add(new StructuralInsight
                    {
                        Type = "Global Unknown_0x04 Distribution",
                        Description = "Distribution of Unknown_0x04 values across all loaded files",
                        Significance = allUnknown04Values.GroupBy(x => x)
                            .OrderByDescending(g => g.Count())
                            .Take(5)
                            .Select(g => $"0x{g.Key:X8}({g.Count()})")
                            .Aggregate("Top values: ", (acc, val) => acc + val + " "),
                        DataPreview = $"Range: 0x{allUnknown04Values.Min():X8} - 0x{allUnknown04Values.Max():X8}"
                    });
                });

                MessageBox.Show($"Batch analysis complete! Check the Insights tab for cross-file patterns.", 
                              "Batch Analysis Complete", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during batch analysis: {ex.Message}", "Analysis Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        #endregion

        #region Enhanced Analysis and Export Methods

        private async Task ExportInsightsAsync()
        {
            if (StructuralInsights.Count == 0)
            {
                MessageBox.Show("No insights available to export. Please load a PM4 file first.", "No Data", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "CSV Files (*.csv)|*.csv|JSON Files (*.json)|*.json|Text Files (*.txt)|*.txt|All Files (*.*)|*.*",
                Title = "Export Structural Insights",
                FileName = $"PM4_Insights_{DateTime.Now:yyyyMMdd_HHmmss}"
            };

            if (dialog.ShowDialog() == true)
            {
                IsLoading = true;
                LoadingOperation = "Exporting insights";
                LoadingSubOperation = "Generating export file...";

                try
                {
                    await Task.Run(() =>
                    {
                        var extension = Path.GetExtension(dialog.FileName).ToLower();
                        switch (extension)
                        {
                            case ".csv":
                                ExportInsightsToCSV(dialog.FileName);
                                break;
                            case ".json":
                                ExportInsightsToJSON(dialog.FileName);
                                break;
                            default:
                                ExportInsightsToText(dialog.FileName);
                                break;
                        }
                    });

                    MessageBox.Show($"Insights exported successfully to:\n{dialog.FileName}", "Export Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error exporting insights: {ex.Message}", "Export Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                finally
                {
                    IsLoading = false;
                }
            }
        }

        private async Task ExportDetailedAnalysisAsync()
        {
            if (Pm4File == null)
            {
                MessageBox.Show("No PM4 file loaded. Please load a file first.", "No Data", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "HTML Report (*.html)|*.html|JSON Analysis (*.json)|*.json|CSV Data (*.csv)|*.csv|All Files (*.*)|*.*",
                Title = "Export Detailed Analysis Report",
                FileName = $"PM4_DetailedAnalysis_{DateTime.Now:yyyyMMdd_HHmmss}"
            };

            if (dialog.ShowDialog() == true)
            {
                IsLoading = true;
                LoadingOperation = "Generating detailed analysis";
                LoadingSubOperation = "Processing PM4 structure...";

                try
                {
                    await Task.Run(async () =>
                    {
                        var analysis = await GenerateDetailedAnalysisAsync();
                        var extension = Path.GetExtension(dialog.FileName).ToLower();
                        
                        switch (extension)
                        {
                            case ".html":
                                await ExportAnalysisToHTMLAsync(dialog.FileName, analysis);
                                break;
                            case ".json":
                                await ExportAnalysisToJSONAsync(dialog.FileName, analysis);
                                break;
                            case ".csv":
                                await ExportAnalysisToCSVAsync(dialog.FileName, analysis);
                                break;
                            default:
                                await ExportAnalysisToHTMLAsync(dialog.FileName, analysis);
                                break;
                        }
                    });

                    MessageBox.Show($"Detailed analysis exported successfully to:\n{dialog.FileName}", "Export Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error exporting detailed analysis: {ex.Message}", "Export Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                finally
                {
                    IsLoading = false;
                }
            }
        }

        private async Task ExportGroupAnalysisAsync()
        {
            if (Unknown0x04Groups.Count == 0)
            {
                MessageBox.Show("No Unknown_0x04 groups analyzed yet. Please load a PM4 file first.", "No Data", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "CSV Files (*.csv)|*.csv|JSON Files (*.json)|*.json|Excel Files (*.xlsx)|*.xlsx|All Files (*.*)|*.*",
                Title = "Export Group Analysis",
                FileName = $"PM4_GroupAnalysis_{DateTime.Now:yyyyMMdd_HHmmss}"
            };

            if (dialog.ShowDialog() == true)
            {
                IsLoading = true;
                LoadingOperation = "Exporting group analysis";
                LoadingSubOperation = "Processing group data...";

                try
                {
                    await Task.Run(() =>
                    {
                        var extension = Path.GetExtension(dialog.FileName).ToLower();
                        switch (extension)
                        {
                            case ".csv":
                                ExportGroupsToCSV(dialog.FileName);
                                break;
                            case ".json":
                                ExportGroupsToJSON(dialog.FileName);
                                break;
                            default:
                                ExportGroupsToCSV(dialog.FileName);
                                break;
                        }
                    });

                    MessageBox.Show($"Group analysis exported successfully to:\n{dialog.FileName}", "Export Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error exporting group analysis: {ex.Message}", "Export Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                finally
                {
                    IsLoading = false;
                }
            }
        }

        private async Task PerformDeepAnalysisAsync()
        {
            if (Pm4File == null)
            {
                MessageBox.Show("No PM4 file loaded. Please load a file first.", "No Data", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            // Create a new cancellation token for this operation
            _loadingCancellationTokenSource?.Cancel();
            _loadingCancellationTokenSource = new CancellationTokenSource();
            var cancellationToken = _loadingCancellationTokenSource.Token;

            IsLoading = true;
            LoadingOperation = "Performing Deep Structure Analysis";
            LoadingSubOperation = "Initializing comprehensive analysis...";

            try
            {
                var analysis = await Task.Run(async () =>
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    
                    LoadingSubOperation = "Analyzing unknown field patterns...";
                    await Task.Delay(100, cancellationToken);
                    
                    var fieldAnalysis = await AnalyzeUnknownFieldPatternsAsync(cancellationToken);
                    
                    cancellationToken.ThrowIfCancellationRequested();
                    LoadingSubOperation = "Detecting hierarchical structures...";
                    await Task.Delay(100, cancellationToken);
                    
                    var hierarchyAnalysis = await AnalyzeHierarchicalStructuresAsync(cancellationToken);
                    
                    cancellationToken.ThrowIfCancellationRequested();
                    LoadingSubOperation = "Cross-referencing chunk relationships...";
                    await Task.Delay(100, cancellationToken);
                    
                    var crossRefAnalysis = await AnalyzeCrossChunkReferencesAsync(cancellationToken);
                    
                    cancellationToken.ThrowIfCancellationRequested();
                    LoadingSubOperation = "Generating comprehensive report...";
                    await Task.Delay(100, cancellationToken);
                    
                    return new DeepAnalysisResult
                    {
                        FieldPatterns = fieldAnalysis,
                        HierarchyStructures = hierarchyAnalysis,
                        CrossReferences = crossRefAnalysis,
                        Timestamp = DateTime.Now
                    };
                }, cancellationToken);

                // Perform detailed hierarchy analysis
                cancellationToken.ThrowIfCancellationRequested();
                LoadingSubOperation = "Performing detailed hierarchy analysis...";
                
                if (Pm4File?.MSLK?.Entries != null)
                {
                    var groupedEntries = Pm4File.MSLK.Entries
                        .Select((entry, index) => new { Entry = entry, Index = index })
                        .GroupBy(x => x.Entry.Unknown_0x04)
                        .ToList();

                    CurrentHierarchyAnalysis = await Task.Run(() => PerformDeepHierarchicalAnalysis(groupedEntries.Cast<IGrouping<uint, dynamic>>().ToList(), cancellationToken), cancellationToken);

                    cancellationToken.ThrowIfCancellationRequested();
                    
                    // Update the Unknown0x04Groups with hierarchy information
                    foreach (var group in Unknown0x04Groups)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        
                        if (CurrentHierarchyAnalysis.GroupHierarchy.TryGetValue(group.GroupValue, out var hierarchyInfo))
                        {
                            group.HierarchyInfo = hierarchyInfo;
                        }
                        if (CurrentHierarchyAnalysis.ChunkOwnership.TryGetValue(group.GroupValue, out var chunkOwnership))
                        {
                            group.ChunkOwnership = chunkOwnership;
                        }
                    }

                    // Add hierarchy insights
                    AddHierarchyInsights(CurrentHierarchyAnalysis);
                }

                // Update insights with deep analysis results
                AddDeepAnalysisInsights(analysis);

                // Enable hierarchy visualization
                ShowHierarchyTree = true;
                UpdateVisualization();
                
                MessageBox.Show($"Deep analysis complete! Found:\n" +
                              $"• {analysis.FieldPatterns.Count} field patterns\n" +
                              $"• {analysis.HierarchyStructures.Count} hierarchy structures\n" +
                              $"• {analysis.CrossReferences.Count} cross-references\n" +
                              (CurrentHierarchyAnalysis != null ? 
                                $"• {CurrentHierarchyAnalysis.MaxHierarchyDepth} hierarchy levels\n" +
                                $"• {CurrentHierarchyAnalysis.RootNodes} root nodes, {CurrentHierarchyAnalysis.LeafNodes} leaf nodes\n" +
                                $"• {CurrentHierarchyAnalysis.TotalConnections} total connections\n" : "") +
                              $"\n🌳 Hierarchy Tree visualization enabled! Check the Insights tab for detailed results.", 
                              "Deep Analysis Complete", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (OperationCanceledException)
            {
                MessageBox.Show("Deep analysis was cancelled.", "Analysis Cancelled", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during deep analysis: {ex.Message}", "Analysis Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                IsLoading = false;
                _loadingCancellationTokenSource = null;
            }
        }

        private async Task InvestigatePaddingAsync()
        {
            if (Pm4File == null)
            {
                MessageBox.Show("No PM4 file loaded. Please load a file first.", "No Data", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            IsLoading = true;
            LoadingOperation = "Investigating Padding & Hidden Data";
            LoadingSubOperation = "Scanning for non-zero padding...";

            try
            {
                var investigation = await Task.Run(async () =>
                {
                    LoadingSubOperation = "Analyzing padding patterns...";
                    await Task.Delay(100);
                    
                    var paddingAnalysis = AnalyzePaddingPatterns();
                    
                    LoadingSubOperation = "Detecting hidden metadata...";
                    await Task.Delay(100);
                    
                    var hiddenData = DetectHiddenMetadata();
                    
                    LoadingSubOperation = "Frequency analysis of bytes...";
                    await Task.Delay(100);
                    
                    var byteFrequency = AnalyzeByteFrequency();
                    
                    LoadingSubOperation = "Generating investigation report...";
                    await Task.Delay(100);
                    
                    return new PaddingInvestigationResult
                    {
                        PaddingPatterns = paddingAnalysis,
                        HiddenData = hiddenData,
                        ByteFrequency = byteFrequency,
                        Timestamp = DateTime.Now
                    };
                });

                // Update insights with padding investigation results
                AddPaddingInvestigationInsights(investigation);
                
                MessageBox.Show($"Padding investigation complete! Found:\n" +
                              $"• {investigation.PaddingPatterns.Count} padding patterns\n" +
                              $"• {investigation.HiddenData.Count} potential hidden data regions\n" +
                              $"• Byte frequency analysis complete\n\n" +
                              $"Check the Insights tab for detailed results.", 
                              "Padding Investigation Complete", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error during padding investigation: {ex.Message}", "Investigation Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                IsLoading = false;
            }
        }

        // Export Implementation Methods
        private void ExportInsightsToCSV(string filePath)
        {
            var csv = new System.Text.StringBuilder();
            csv.AppendLine("Type,Description,Significance,DataPreview");
            
            foreach (var insight in StructuralInsights)
            {
                csv.AppendLine($"\"{insight.Type}\",\"{insight.Description}\",\"{insight.Significance}\",\"{insight.DataPreview}\"");
            }
            
            File.WriteAllText(filePath, csv.ToString());
        }

        private void ExportInsightsToJSON(string filePath)
        {
            var json = System.Text.Json.JsonSerializer.Serialize(StructuralInsights, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            File.WriteAllText(filePath, json);
        }

        private void ExportInsightsToText(string filePath)
        {
            var text = new System.Text.StringBuilder();
            text.AppendLine($"PM4 Structural Insights Report");
            text.AppendLine($"Generated: {DateTime.Now}");
            text.AppendLine($"File: {LoadedFileName}");
            text.AppendLine(new string('=', 80));
            text.AppendLine();

            foreach (var insight in StructuralInsights)
            {
                text.AppendLine($"[{insight.Type}]");
                text.AppendLine($"Description: {insight.Description}");
                text.AppendLine($"Significance: {insight.Significance}");
                text.AppendLine($"Data Preview: {insight.DataPreview}");
                text.AppendLine(new string('-', 40));
                text.AppendLine();
            }
            
            File.WriteAllText(filePath, text.ToString());
        }

        private void ExportGroupsToCSV(string filePath)
        {
            var csv = new System.Text.StringBuilder();
            csv.AppendLine("GroupValue,EntryCount,Description,AssociatedVertices,AverageUnknown0x0C,MinUnknown0x0C,MaxUnknown0x0C,HierarchyLevel,IsRootNode,IsLeafNode,ChildCount");
            
            foreach (var group in Unknown0x04Groups)
            {
                csv.AppendLine($"0x{group.GroupValue:X8},{group.EntryCount},\"{group.Description}\",{group.AssociatedVertices.Count}," +
                              $"{group.AverageUnknown0x0C},{group.MinUnknown0x0C},{group.MaxUnknown0x0C}," +
                              $"{group.HierarchyInfo.HierarchyLevel},{group.HierarchyInfo.IsRootNode},{group.HierarchyInfo.IsLeafNode},{group.HierarchyInfo.ChildCount}");
            }
            
            File.WriteAllText(filePath, csv.ToString());
        }

        private void ExportGroupsToJSON(string filePath)
        {
            var exportData = Unknown0x04Groups.Select(g => new 
            {
                GroupValue = $"0x{g.GroupValue:X8}",
                g.EntryCount,
                g.Description,
                AssociatedVerticesCount = g.AssociatedVertices.Count,
                g.AverageUnknown0x0C,
                g.MinUnknown0x0C,
                g.MaxUnknown0x0C,
                Hierarchy = new 
                {
                    g.HierarchyInfo.HierarchyLevel,
                    g.HierarchyInfo.IsRootNode,
                    g.HierarchyInfo.IsLeafNode,
                    g.HierarchyInfo.ChildCount,
                    ParentValue = g.HierarchyInfo.ParentValue?.ToString("X8"),
                    ChildValues = g.HierarchyInfo.ChildValues.Select(v => v.ToString("X8")).ToArray()
                },
                ChunkOwnership = new 
                {
                    g.ChunkOwnership.VertexRangeStart,
                    g.ChunkOwnership.VertexRangeEnd,
                    g.ChunkOwnership.VertexRangeSize,
                    g.ChunkOwnership.HasExclusiveGeometry,
                    g.ChunkOwnership.GeometryComplexity
                }
            }).ToArray();
            
            var json = System.Text.Json.JsonSerializer.Serialize(exportData, new System.Text.Json.JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            File.WriteAllText(filePath, json);
        }

        // Deep Analysis Implementation Methods
        // Async versions with cancellation support
        private async Task<List<FieldPattern>> AnalyzeUnknownFieldPatternsAsync(CancellationToken cancellationToken)
        {
            var patterns = new List<FieldPattern>();
            
            if (Pm4File?.MSLK?.Entries == null) return patterns;

            cancellationToken.ThrowIfCancellationRequested();
            
            // Analyze Unknown_0x04 patterns
            var unknown04Values = Pm4File.MSLK.Entries.Select(e => e.Unknown_0x04).ToList();
            patterns.Add(await AnalyzeFieldForPatternsAsync("Unknown_0x04", unknown04Values, cancellationToken));

            cancellationToken.ThrowIfCancellationRequested();
            
            // Analyze Unknown_0x02 patterns  
            var unknown02Values = Pm4File.MSLK.Entries.Select(e => (uint)e.Unknown_0x02).ToList();
            patterns.Add(await AnalyzeFieldForPatternsAsync("Unknown_0x02", unknown02Values, cancellationToken));

            cancellationToken.ThrowIfCancellationRequested();
            
            // Analyze Unknown_0x0C patterns
            var unknown0CValues = Pm4File.MSLK.Entries.Select(e => e.Unknown_0x0C).ToList();
            patterns.Add(await AnalyzeFieldForPatternsAsync("Unknown_0x0C", unknown0CValues, cancellationToken));

            cancellationToken.ThrowIfCancellationRequested();
            
            // Analyze Unknown_0x10 patterns (potential indices)
            var unknown10Values = Pm4File.MSLK.Entries.Select(e => (uint)e.Unknown_0x10).ToList();
            patterns.Add(await AnalyzeFieldForPatternsAsync("Unknown_0x10", unknown10Values, cancellationToken));

            return patterns.Where(p => p.Confidence > 0.5).ToList();
        }

        private List<FieldPattern> AnalyzeUnknownFieldPatterns()
        {
            var patterns = new List<FieldPattern>();
            
            if (Pm4File?.MSLK?.Entries == null) return patterns;

            // Analyze Unknown_0x04 patterns
            var unknown04Values = Pm4File.MSLK.Entries.Select(e => e.Unknown_0x04).ToList();
            patterns.Add(AnalyzeFieldForPatterns("Unknown_0x04", unknown04Values));

            // Analyze Unknown_0x02 patterns  
            var unknown02Values = Pm4File.MSLK.Entries.Select(e => (uint)e.Unknown_0x02).ToList();
            patterns.Add(AnalyzeFieldForPatterns("Unknown_0x02", unknown02Values));

            // Analyze Unknown_0x0C patterns
            var unknown0CValues = Pm4File.MSLK.Entries.Select(e => e.Unknown_0x0C).ToList();
            patterns.Add(AnalyzeFieldForPatterns("Unknown_0x0C", unknown0CValues));

            // Analyze Unknown_0x10 patterns (potential indices)
            var unknown10Values = Pm4File.MSLK.Entries.Select(e => (uint)e.Unknown_0x10).ToList();
            patterns.Add(AnalyzeFieldForPatterns("Unknown_0x10", unknown10Values));

            return patterns.Where(p => p.Confidence > 0.5).ToList();
        }

        private async Task<FieldPattern> AnalyzeFieldForPatternsAsync(string fieldName, List<uint> values, CancellationToken cancellationToken)
        {
            var pattern = new FieldPattern { FieldName = fieldName, Values = values };
            
            if (!values.Any()) return pattern;

            cancellationToken.ThrowIfCancellationRequested();
            
            var uniqueValues = values.Distinct().ToList();
            var minValue = values.Min();
            var maxValue = values.Max();
            var range = maxValue - minValue;
            
            cancellationToken.ThrowIfCancellationRequested();
            
            // Check for sequential patterns
            var sortedUnique = uniqueValues.OrderBy(x => x).ToList();
            var sequential = true;
            for (int i = 1; i < sortedUnique.Count && sequential; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (sortedUnique[i] - sortedUnique[i-1] != 1)
                    sequential = false;
            }

            if (sequential && uniqueValues.Count > 3)
            {
                pattern.PatternType = "Sequential";
                pattern.Description = $"Sequential values from {minValue} to {maxValue}";
                pattern.Confidence = 0.9;
                return pattern;
            }

            cancellationToken.ThrowIfCancellationRequested();
            
            // Check for power-of-2 patterns (flags)
            var powerOf2Count = uniqueValues.Count(v => v > 0 && (v & (v - 1)) == 0);
            if (powerOf2Count > uniqueValues.Count * 0.7)
            {
                pattern.PatternType = "Flags";
                pattern.Description = $"Flag-like values (powers of 2): {powerOf2Count}/{uniqueValues.Count} are powers of 2";
                pattern.Confidence = 0.8;
                return pattern;
            }

            cancellationToken.ThrowIfCancellationRequested();
            
            // Check for bounded index patterns
            if (fieldName == "Unknown_0x10" && maxValue < (Pm4File?.MSLK?.Entries?.Count ?? 0) * 2)
            {
                pattern.PatternType = "Index";
                pattern.Description = $"Potential index values (max: {maxValue}, entries: {Pm4File?.MSLK?.Entries?.Count ?? 0})";
                pattern.Confidence = 0.7;
                return pattern;
            }

            cancellationToken.ThrowIfCancellationRequested();
            
            // Check for grouping patterns
            var groups = values.GroupBy(x => x).Where(g => g.Count() > 1).ToList();
            if (groups.Count < uniqueValues.Count * 0.5)
            {
                pattern.PatternType = "Grouping";
                pattern.Description = $"Values form {groups.Count} groups with repeats";
                pattern.Confidence = 0.6;
                return pattern;
            }

            pattern.PatternType = "Random";
            pattern.Description = $"No clear pattern detected in {uniqueValues.Count} unique values";
            pattern.Confidence = 0.1;
            return pattern;
        }

        private FieldPattern AnalyzeFieldForPatterns(string fieldName, List<uint> values)
        {
            var pattern = new FieldPattern { FieldName = fieldName, Values = values };
            
            if (!values.Any()) return pattern;

            var uniqueValues = values.Distinct().ToList();
            var minValue = values.Min();
            var maxValue = values.Max();
            var range = maxValue - minValue;
            
            // Check for sequential patterns
            var sortedUnique = uniqueValues.OrderBy(x => x).ToList();
            var sequential = true;
            for (int i = 1; i < sortedUnique.Count && sequential; i++)
            {
                if (sortedUnique[i] - sortedUnique[i-1] != 1)
                    sequential = false;
            }

            if (sequential && uniqueValues.Count > 3)
            {
                pattern.PatternType = "Sequential";
                pattern.Description = $"Sequential values from {minValue} to {maxValue}";
                pattern.Confidence = 0.9;
                return pattern;
            }

            // Check for power-of-2 patterns (flags)
            var powerOf2Count = uniqueValues.Count(v => v > 0 && (v & (v - 1)) == 0);
            if (powerOf2Count > uniqueValues.Count * 0.7)
            {
                pattern.PatternType = "Flags";
                pattern.Description = $"Flag-like values (powers of 2): {powerOf2Count}/{uniqueValues.Count} are powers of 2";
                pattern.Confidence = 0.8;
                return pattern;
            }

            // Check for bounded index patterns
            if (fieldName == "Unknown_0x10" && maxValue < (Pm4File?.MSLK?.Entries?.Count ?? 0) * 2)
            {
                pattern.PatternType = "Index";
                pattern.Description = $"Potential index values (max: {maxValue}, entries: {Pm4File?.MSLK?.Entries?.Count ?? 0})";
                pattern.Confidence = 0.7;
                return pattern;
            }

            // Check for grouping patterns
            var groups = values.GroupBy(x => x).Where(g => g.Count() > 1).ToList();
            if (groups.Count < uniqueValues.Count * 0.5)
            {
                pattern.PatternType = "Grouping";
                pattern.Description = $"Values form {groups.Count} groups with repeats";
                pattern.Confidence = 0.6;
                return pattern;
            }

            pattern.PatternType = "Random";
            pattern.Description = $"No clear pattern detected in {uniqueValues.Count} unique values";
            pattern.Confidence = 0.1;
            return pattern;
        }

        private async Task<List<HierarchyStructure>> AnalyzeHierarchicalStructuresAsync(CancellationToken cancellationToken)
        {
            var structures = new List<HierarchyStructure>();
            
            if (Pm4File?.MSLK?.Entries == null) return structures;

            cancellationToken.ThrowIfCancellationRequested();
            
            // Analyze Unknown_0x04 based hierarchy
            var unknown04Groups = Pm4File.MSLK.Entries
                .GroupBy(e => e.Unknown_0x04)
                .ToDictionary(g => g.Key, g => g.ToList());

            var hierarchy = new HierarchyStructure
            {
                StructureType = "Unknown_0x04 Groups",
                RootNodes = new List<uint>(),
                Relationships = new Dictionary<uint, List<uint>>()
            };

            // Find potential parent-child relationships through bit masking
            foreach (var group in unknown04Groups.Keys.OrderBy(x => x))
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                var potentialChildren = unknown04Groups.Keys
                    .Where(other => other > group && (other & group) == group && other != group)
                    .ToList();

                if (potentialChildren.Any())
                {
                    hierarchy.Relationships[group] = potentialChildren;
                    if (!unknown04Groups.Keys.Any(parent => parent < group && (group & parent) == parent))
                    {
                        hierarchy.RootNodes.Add(group);
                    }
                }
                else if (!unknown04Groups.Keys.Any(parent => parent < group && (group & parent) == parent))
                {
                    hierarchy.RootNodes.Add(group);
                }
            }

            cancellationToken.ThrowIfCancellationRequested();
            hierarchy.Depth = await CalculateHierarchyDepthAsync(hierarchy.Relationships, hierarchy.RootNodes, cancellationToken);
            structures.Add(hierarchy);

            return structures;
        }

        private List<HierarchyStructure> AnalyzeHierarchicalStructures()
        {
            var structures = new List<HierarchyStructure>();
            
            if (Pm4File?.MSLK?.Entries == null) return structures;

            // Analyze Unknown_0x04 based hierarchy
            var unknown04Groups = Pm4File.MSLK.Entries
                .GroupBy(e => e.Unknown_0x04)
                .ToDictionary(g => g.Key, g => g.ToList());

            var hierarchy = new HierarchyStructure
            {
                StructureType = "Unknown_0x04 Groups",
                RootNodes = new List<uint>(),
                Relationships = new Dictionary<uint, List<uint>>()
            };

            // Find potential parent-child relationships through bit masking
            foreach (var group in unknown04Groups.Keys.OrderBy(x => x))
            {
                var potentialChildren = unknown04Groups.Keys
                    .Where(other => other > group && (other & group) == group && other != group)
                    .ToList();

                if (potentialChildren.Any())
                {
                    hierarchy.Relationships[group] = potentialChildren;
                    if (!unknown04Groups.Keys.Any(parent => parent < group && (group & parent) == parent))
                    {
                        hierarchy.RootNodes.Add(group);
                    }
                }
                else if (!unknown04Groups.Keys.Any(parent => parent < group && (group & parent) == parent))
                {
                    hierarchy.RootNodes.Add(group);
                }
            }

            hierarchy.Depth = CalculateHierarchyDepth(hierarchy.Relationships, hierarchy.RootNodes);
            structures.Add(hierarchy);

            return structures;
        }

        private async Task<int> CalculateHierarchyDepthAsync(Dictionary<uint, List<uint>> relationships, List<uint> rootNodes, CancellationToken cancellationToken)
        {
            int maxDepth = 0;
            foreach (var root in rootNodes)
            {
                cancellationToken.ThrowIfCancellationRequested();
                maxDepth = Math.Max(maxDepth, await CalculateNodeDepthAsync(root, relationships, 0, cancellationToken, new HashSet<uint>()));
            }
            return maxDepth;
        }

        private async Task<int> CalculateNodeDepthAsync(uint node, Dictionary<uint, List<uint>> relationships, int currentDepth, CancellationToken cancellationToken, HashSet<uint> visited)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            // Prevent infinite recursion
            if (visited.Contains(node) || currentDepth > 20) // Max depth safety check
                return currentDepth;
                
            visited.Add(node);

            if (!relationships.ContainsKey(node) || !relationships[node].Any())
            {
                visited.Remove(node);
                return currentDepth;
            }

            int maxChildDepth = currentDepth;
            foreach (var child in relationships[node])
            {
                cancellationToken.ThrowIfCancellationRequested();
                maxChildDepth = Math.Max(maxChildDepth, await CalculateNodeDepthAsync(child, relationships, currentDepth + 1, cancellationToken, visited));
            }
            
            visited.Remove(node);
            return maxChildDepth;
        }

        private int CalculateHierarchyDepth(Dictionary<uint, List<uint>> relationships, List<uint> rootNodes)
        {
            int maxDepth = 0;
            foreach (var root in rootNodes)
            {
                maxDepth = Math.Max(maxDepth, CalculateNodeDepth(root, relationships, 0));
            }
            return maxDepth;
        }

        private int CalculateNodeDepth(uint node, Dictionary<uint, List<uint>> relationships, int currentDepth)
        {
            if (!relationships.ContainsKey(node) || !relationships[node].Any())
                return currentDepth;

            int maxChildDepth = currentDepth;
            foreach (var child in relationships[node])
            {
                maxChildDepth = Math.Max(maxChildDepth, CalculateNodeDepth(child, relationships, currentDepth + 1));
            }
            return maxChildDepth;
        }

        private async Task<List<CrossReference>> AnalyzeCrossChunkReferencesAsync(CancellationToken cancellationToken)
        {
            var crossRefs = new List<CrossReference>();
            
            if (Pm4File == null) return crossRefs;

            cancellationToken.ThrowIfCancellationRequested();

            // MSLK to MSVI references
            if (Pm4File.MSLK?.Entries != null && Pm4File.MSVI?.Indices != null)
            {
                var mslkToMsvi = new CrossReference
                {
                    SourceChunk = "MSLK",
                    TargetChunk = "MSVI", 
                    ReferenceType = "Index",
                    References = new List<(uint, uint)>()
                };

                foreach (var entry in Pm4File.MSLK.Entries)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    
                    if (entry.Unknown_0x10 < Pm4File.MSVI.Indices.Count)
                    {
                        mslkToMsvi.References.Add(((uint)entry.Unknown_0x10, Pm4File.MSVI.Indices[entry.Unknown_0x10]));
                    }
                }

                crossRefs.Add(mslkToMsvi);
            }

            cancellationToken.ThrowIfCancellationRequested();

            // MSVI to MSVT references
            if (Pm4File.MSVI?.Indices != null && Pm4File.MSVT?.Vertices != null)
            {
                var msviToMsvt = new CrossReference
                {
                    SourceChunk = "MSVI",
                    TargetChunk = "MSVT",
                    ReferenceType = "VertexIndex",
                    References = new List<(uint, uint)>()
                };

                for (int i = 0; i < Pm4File.MSVI.Indices.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    
                    var vertexIndex = Pm4File.MSVI.Indices[i];
                    if (vertexIndex < Pm4File.MSVT.Vertices.Count)
                    {
                        msviToMsvt.References.Add(((uint)i, vertexIndex));
                    }
                }

                crossRefs.Add(msviToMsvt);
            }

            return crossRefs;
        }

        private List<PaddingPattern> AnalyzePaddingPatterns()
        {
            var patterns = new List<PaddingPattern>();
            
            // This would require access to raw file bytes - placeholder implementation
            // In a real implementation, you'd analyze the raw PM4 file for padding between structures
            
            return patterns;
        }

        private List<HiddenDataRegion> DetectHiddenMetadata()
        {
            var hiddenRegions = new List<HiddenDataRegion>();
            
            // This would require access to raw file bytes - placeholder implementation
            // In a real implementation, you'd look for non-zero padding, unexpected data patterns
            
            return hiddenRegions;
        }

        private Dictionary<byte, int> AnalyzeByteFrequency()
        {
            var frequency = new Dictionary<byte, int>();
            
            // This would require access to raw file bytes - placeholder implementation
            // In a real implementation, you'd count frequency of each byte value
            
            return frequency;
        }

        private void AddDeepAnalysisInsights(DeepAnalysisResult analysis)
        {
            foreach (var pattern in analysis.FieldPatterns)
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Field Pattern",
                    Description = $"{pattern.FieldName}: {pattern.PatternType} pattern detected",
                    Significance = $"Confidence: {pattern.Confidence:P1} - {pattern.Description}",
                    DataPreview = pattern.Values.Count > 5 
                        ? $"Sample: {string.Join(", ", pattern.Values.Take(5).Select(v => $"0x{v:X8}"))}, ..."
                        : $"Values: {string.Join(", ", pattern.Values.Select(v => $"0x{v:X8}"))}"
                });
            }

            foreach (var hierarchy in analysis.HierarchyStructures)
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Hierarchy Structure",
                    Description = $"{hierarchy.StructureType} with {hierarchy.RootNodes.Count} root nodes",
                    Significance = $"Depth: {hierarchy.Depth} levels, {hierarchy.Relationships.Count} parent-child relationships",
                    DataPreview = $"Roots: {string.Join(", ", hierarchy.RootNodes.Take(3).Select(r => $"0x{r:X8}"))}"
                });
            }

            foreach (var crossRef in analysis.CrossReferences)
            {
                StructuralInsights.Add(new StructuralInsight
                {
                    Type = "Cross Reference",
                    Description = $"{crossRef.SourceChunk} → {crossRef.TargetChunk} references",
                    Significance = $"{crossRef.References.Count} {crossRef.ReferenceType} connections found",
                    DataPreview = crossRef.References.Take(3).Any() 
                        ? $"Sample: {string.Join(", ", crossRef.References.Take(3).Select(r => $"{r.source}→{r.target}"))}"
                        : "No valid references"
                });
            }
        }

        private void AddPaddingInvestigationInsights(PaddingInvestigationResult investigation)
        {
            StructuralInsights.Add(new StructuralInsight
            {
                Type = "Padding Investigation",
                Description = $"Found {investigation.PaddingPatterns.Count} padding patterns and {investigation.HiddenData.Count} hidden data regions",
                Significance = "Byte frequency analysis completed for potential metadata detection",
                DataPreview = investigation.ByteFrequency.Any() 
                    ? $"Most common bytes: {string.Join(", ", investigation.ByteFrequency.OrderByDescending(kvp => kvp.Value).Take(3).Select(kvp => $"0x{kvp.Key:X2}({kvp.Value})"))}"
                    : "No byte frequency data available"
            });
        }

        // Placeholder methods for missing HTML/JSON export functionality
        private Task<object> GenerateDetailedAnalysisAsync()
        {
            return Task.FromResult<object>(new 
            {
                FileName = LoadedFileName,
                Timestamp = DateTime.Now,
                Insights = StructuralInsights.ToArray(),
                Groups = Unknown0x04Groups.ToArray(),
                Summary = new 
                {
                    TotalInsights = StructuralInsights.Count,
                    TotalGroups = Unknown0x04Groups.Count,
                    ChunkTypes = ChunkItems.Count
                }
            });
        }

        private Task ExportAnalysisToHTMLAsync(string filePath, object analysis)
        {
            return Task.Run(() =>
            {
                // Placeholder HTML export - would generate a comprehensive HTML report
                var html = $@"
                <!DOCTYPE html>
                <html>
                <head><title>PM4 Analysis Report</title></head>
                <body>
                    <h1>PM4 Detailed Analysis Report</h1>
                    <p>Generated: {DateTime.Now}</p>
                    <p>File: {LoadedFileName}</p>
                    <h2>Summary</h2>
                    <p>Insights: {StructuralInsights.Count}</p>
                    <p>Groups: {Unknown0x04Groups.Count}</p>
                </body>
                </html>";
                File.WriteAllText(filePath, html);
            });
        }

        private Task ExportAnalysisToJSONAsync(string filePath, object analysis)
        {
            return Task.Run(() =>
            {
                var json = System.Text.Json.JsonSerializer.Serialize(analysis, new System.Text.Json.JsonSerializerOptions 
                { 
                    WriteIndented = true 
                });
                File.WriteAllText(filePath, json);
            });
        }

        private Task ExportAnalysisToCSVAsync(string filePath, object analysis)
        {
            return Task.Run(() =>
            {
                // Export a comprehensive CSV with all analysis data
                ExportInsightsToCSV(filePath);
            });
        }

        #endregion
    }

    // Enhanced Analysis Result Classes
    public class DeepAnalysisResult
    {
        public List<FieldPattern> FieldPatterns { get; set; } = new();
        public List<HierarchyStructure> HierarchyStructures { get; set; } = new();
        public List<CrossReference> CrossReferences { get; set; } = new();
        public DateTime Timestamp { get; set; }
    }

    public class PaddingInvestigationResult
    {
        public List<PaddingPattern> PaddingPatterns { get; set; } = new();
        public List<HiddenDataRegion> HiddenData { get; set; } = new();
        public Dictionary<byte, int> ByteFrequency { get; set; } = new();
        public DateTime Timestamp { get; set; }
    }

    public class FieldPattern
    {
        public string FieldName { get; set; } = string.Empty;
        public string PatternType { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public List<uint> Values { get; set; } = new();
        public double Confidence { get; set; }
    }

    public class HierarchyStructure
    {
        public string StructureType { get; set; } = string.Empty;
        public List<uint> RootNodes { get; set; } = new();
        public Dictionary<uint, List<uint>> Relationships { get; set; } = new();
        public int Depth { get; set; }
    }

    public class CrossReference
    {
        public string SourceChunk { get; set; } = string.Empty;
        public string TargetChunk { get; set; } = string.Empty;
        public string ReferenceType { get; set; } = string.Empty;
        public List<(uint source, uint target)> References { get; set; } = new();
    }

    public class PaddingPattern
    {
        public int Offset { get; set; }
        public int Length { get; set; }
        public byte[] Pattern { get; set; } = Array.Empty<byte>();
        public string Description { get; set; } = string.Empty;
    }

    public class HiddenDataRegion
    {
        public int Offset { get; set; }
        public int Length { get; set; }
        public byte[] Data { get; set; } = Array.Empty<byte>();
        public string PotentialType { get; set; } = string.Empty;
    }

    public class LegendItem
    {
        public string Name { get; set; } = string.Empty;
        public Color Color { get; set; }
        public string Description { get; set; } = string.Empty;
        public string Icon { get; set; } = string.Empty;
        public bool IsVisible { get; set; } = true;

        public LegendItem() { }

        public LegendItem(string name, Color color, string description, string icon = "")
        {
            Name = name;
            Color = color;
            Description = description;
            Icon = icon;
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
        public GroupHierarchyInfo HierarchyInfo { get; set; } = new();
        public ChunkOwnershipInfo ChunkOwnership { get; set; } = new();
    }

    public class Unknown0x0CGroup
    {
        public uint GroupValue { get; set; }
        public int EntryCount { get; set; }
        public List<int> EntryIndices { get; set; } = new();
        public Color GroupColor { get; set; }
        public Color Color { get; set; }
        public string Description { get; set; } = string.Empty;
        public List<Vector3> AssociatedVertices { get; set; } = new();
        public float AverageUnknown0x04 { get; set; }
        public float MinUnknown0x04 { get; set; }
        public float MaxUnknown0x04 { get; set; }
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

    public class HierarchyAnalysisResult
    {
        public Dictionary<uint, GroupHierarchyInfo> GroupHierarchy { get; set; } = new();
        public Dictionary<uint, List<uint>> CrossReferenceNetwork { get; set; } = new();
        public Dictionary<uint, ChunkOwnershipInfo> ChunkOwnership { get; set; } = new();
        public int MaxHierarchyDepth { get; set; }
        public int RootNodes { get; set; }
        public int LeafNodes { get; set; }
        public int TotalConnections { get; set; }
    }

    public class GroupHierarchyInfo
    {
        public uint GroupValue { get; set; }
        public uint? ParentValue { get; set; }
        public List<uint> ChildValues { get; set; } = new();
        public List<uint> CrossReferences { get; set; } = new();
        public bool IsRootNode { get; set; }
        public bool IsLeafNode { get; set; }
        public int HierarchyLevel { get; set; }
        public int ChildCount { get; set; }
        public Vector3? SpatialCenter { get; set; }
        public float SpatialRadius { get; set; }
    }

    public class ChunkOwnershipInfo
    {
        public List<uint> OwnedMSVIIndices { get; set; } = new();
        public List<uint> OwnedMSVTVertices { get; set; } = new();
        public List<uint> OwnedMSURSurfaces { get; set; } = new();
        public uint VertexRangeStart { get; set; }
        public uint VertexRangeEnd { get; set; }
        public int VertexRangeSize { get; set; }
        public bool HasExclusiveGeometry { get; set; }
        public float GeometryComplexity { get; set; }
    }
} 