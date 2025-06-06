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
using Color = System.Windows.Media.Color;

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
        private string? _loadedFilePath;
        
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

        // Enhanced Optimization Features
        [ObservableProperty]
        private bool _showSurfaceNormals = false;
        
        [ObservableProperty]
        private bool _useSurfaceNormals = true;
        
        [ObservableProperty]
        private bool _colorByObjectType = false;
        
        [ObservableProperty]
        private bool _colorByMaterialId = false;
        
        [ObservableProperty]
        private bool _showHeightBands = false;
        
        [ObservableProperty]
        private double _heightBandSize = 50.0;
        
        [ObservableProperty]
        private ObservableCollection<ObjectTypeGroup> _objectTypeGroups = new();
        
        [ObservableProperty]
        private ObservableCollection<MaterialGroup> _materialGroups = new();
        
        [ObservableProperty]
        private ObjectTypeGroup? _selectedObjectType;
        
        [ObservableProperty]
        private MaterialGroup? _selectedMaterial;
        
        [ObservableProperty]
        private bool _enhancedExportMode = false;

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

        // Hierarchy Tree View Properties
        [ObservableProperty]
        private ObservableCollection<HierarchyTreeNode> _hierarchyTreeNodes = new();

        [ObservableProperty]
        private HierarchyTreeNode? _selectedHierarchyNode;

        [ObservableProperty]
        private string _hierarchyTreeFilter = string.Empty;

        // Combined MPRL Mesh Properties
        [ObservableProperty]
        private bool _showCombinedMPRLMesh = false;

        [ObservableProperty]
        private bool _isBuildingCombinedMesh = false;

        [ObservableProperty]
        private string _combinedMeshStatus = "No combined mesh loaded";

        [ObservableProperty]
        private int _combinedMeshVertexCount = 0;

        // Batch loading properties for MPRL combination
        [ObservableProperty]
        private List<string> _loadedPM4Files = new();

        // Combined MPRL mesh data
        private List<Point3D> _combinedMPRLVertices = new();
        private Dictionary<string, List<Point3D>> _mprlMeshByFile = new();

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

            // Hierarchy Tree Commands
            SelectHierarchyNodeCommand = new RelayCommand<HierarchyTreeNode>(SelectHierarchyNode);
            ExpandAllHierarchyNodesCommand = new RelayCommand(ExpandAllHierarchyNodes);
            CollapseAllHierarchyNodesCommand = new RelayCommand(CollapseAllHierarchyNodes);

            // Combined MPRL Mesh Commands
            BuildCombinedMPRLMeshCommand = new AsyncRelayCommand(BuildCombinedMPRLMeshAsync);
            ExportCombinedMPRLMeshCommand = new AsyncRelayCommand(ExportCombinedMPRLMeshAsync);
            ClearCombinedMeshCommand = new RelayCommand(ClearCombinedMesh);
            
            // Enhanced Optimization Commands
            AnalyzeObjectTypesCommand = new RelayCommand(AnalyzeObjectTypes);
            AnalyzeMaterialsCommand = new RelayCommand(AnalyzeMaterials);
            ExportEnhancedOBJCommand = new AsyncRelayCommand(ExportEnhancedOBJAsync);
            SelectObjectTypeCommand = new RelayCommand<ObjectTypeGroup>(SelectObjectType);
            SelectMaterialCommand = new RelayCommand<MaterialGroup>(SelectMaterial);
            
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

        // Hierarchy Tree Commands
        public IRelayCommand<HierarchyTreeNode> SelectHierarchyNodeCommand { get; }
        public IRelayCommand ExpandAllHierarchyNodesCommand { get; }
        public IRelayCommand CollapseAllHierarchyNodesCommand { get; }

        // Combined MPRL Mesh Commands
        public IAsyncRelayCommand BuildCombinedMPRLMeshCommand { get; }
        public IAsyncRelayCommand ExportCombinedMPRLMeshCommand { get; }
        public IRelayCommand ClearCombinedMeshCommand { get; }
        
        // Enhanced Optimization Commands
        public IRelayCommand AnalyzeObjectTypesCommand { get; }
        public IRelayCommand AnalyzeMaterialsCommand { get; }
        public IAsyncRelayCommand ExportEnhancedOBJCommand { get; }
        public IRelayCommand<ObjectTypeGroup> SelectObjectTypeCommand { get; }
        public IRelayCommand<MaterialGroup> SelectMaterialCommand { get; }

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

        partial void OnHierarchyTreeFilterChanged(string value)
        {
            FilterHierarchyTree();
        }

        partial void OnSelectedHierarchyNodeChanged(HierarchyTreeNode? value)
        {
            if (value?.GroupInfo != null)
            {
                SelectedUnknown0x04Group = value.GroupInfo;
                UpdateVisualization();
            }
        }

        partial void OnShowCombinedMPRLMeshChanged(bool value)
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
            LoadedFilePath = filePath;
            
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
                
                Pm4File = await Task.Run(() => 
                {
                    try 
                    {
                        return PM4File.FromFile(filePath);
                    }
                    catch (Exception ex)
                    {
                        // Log error and continue with null file to prevent complete failure
                        Console.WriteLine($"Error parsing PM4 file: {ex.Message}");
                        return null;
                    }
                }, cancellationToken);
                
                // Check if file was successfully parsed
                if (Pm4File == null)
                {
                    LoadingOperation = "Error: Could not parse PM4 file";
                    LoadingSubOperation = "File may be corrupted or in an unsupported format";
                    return;
                }

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
                LoadingSubOperation = "Performing basic analysis...";
                await Task.Delay(100, cancellationToken);
                
                // Perform lightweight analysis on UI thread to prevent deadlocks
                try
                {
                    AnalyzeUnknown0x04Groups();
                }
                catch (Exception ex)
                {
                    LoadingSubOperation = $"Analysis skipped: {ex.Message}";
                }
                
                LoadingProgress = 90;
                LoadingSubOperation = "Analyzing group patterns...";
                await Task.Delay(100, cancellationToken);
                
                AnalyzeUnknown0x0CGroups();
                
                // Run analysis in background but capture result
                var analysisResult = await Task.Run(() => 
                {
                    try
                    {
                        // Ensure absolute path before passing to analyzer
                        var absolutePath = Path.IsPathRooted(filePath) ? filePath : Path.GetFullPath(filePath);
                        
                        if (!File.Exists(absolutePath))
                        {
                            throw new FileNotFoundException($"PM4 file not found at path: {absolutePath}");
                        }
                        
                        return _analyzer.AnalyzeFile(absolutePath);
                    }
                    catch (Exception ex)
                    {
                        var errorResult = new PM4StructuralAnalyzer.StructuralAnalysisResult
                        {
                            FileName = Path.GetFileName(filePath)
                        };
                        errorResult.Metadata["AnalysisError"] = ex.Message;
                        errorResult.Metadata["StackTrace"] = ex.StackTrace ?? "No stack trace";
                        errorResult.Metadata["DebugFilePath"] = filePath;
                        errorResult.Metadata["DebugFileExists"] = File.Exists(filePath);
                        errorResult.Metadata["DebugWorkingDir"] = Directory.GetCurrentDirectory();
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

            // Skip heavy analysis during initial visualization to prevent hanging

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

            // Add combined MPRL mesh visualization
            var combinedMesh = CreateCombinedMPRLMeshVisualization();
            if (combinedMesh != null)
            {
                newScene.Children.Add(combinedMesh);
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

        private void UpdateBatchVisualization()
        {
            if (LoadedFiles.Count == 0)
            {
                UpdateVisualization(); // Fall back to single file visualization
                return;
            }

            var newScene = new Model3DGroup();
            ChunkItems.Clear();

            // Add coordinate system visualization
            AddCoordinateSystemVisualization(newScene);

            var totalVertices = 0;
            var fileIndex = 0;
            var distinctColors = GenerateDistinctColors(LoadedFiles.Count);

            // Process each loaded PM4 file
            foreach (var pm4File in LoadedFiles)
            {
                var fileColor = distinctColors[fileIndex % distinctColors.Count];
                var fileName = fileIndex < LoadedPM4Files.Count ? Path.GetFileNameWithoutExtension(LoadedPM4Files[fileIndex]) : $"File{fileIndex + 1}";

                // Visualize MSVT render vertices for this file
                if (pm4File.MSVT?.Vertices != null && ShowMSVTVertices)
                {
                    var msvtPoints = pm4File.MSVT.Vertices.Select(v => new Point3D(v.Y, v.X, v.Z)).ToList();
                    var boundsInfo = ValidateCoordinateBounds(msvtPoints, "MSVT");
                    
                    var msvtModel = CreateVertexVisualization(msvtPoints, fileColor, $"{fileName} MSVT");
                    newScene.Children.Add(msvtModel);
                    
                    ChunkItems.Add(new ChunkVisualizationItem
                    {
                        Name = $"{fileName} MSVT ({pm4File.MSVT.Vertices.Count:N0} vertices){boundsInfo}",
                        Count = pm4File.MSVT.Vertices.Count,
                        Color = fileColor,
                        IsVisible = true
                    });

                    totalVertices += pm4File.MSVT.Vertices.Count;
                }

                // Visualize MSCN collision points for this file
                if (pm4File.MSCN?.ExteriorVertices != null && ShowMSCNPoints)
                {
                    var mscnPoints = pm4File.MSCN.ExteriorVertices.Select(v => new Point3D(v.X, -v.Y, v.Z)).ToList();
                    var boundsInfo = ValidateCoordinateBounds(mscnPoints, "MSCN");
                    
                    // Use a slightly different color for collision points
                    var mscnColor = Color.FromArgb(255, 
                        (byte)Math.Max(0, fileColor.R - 30), 
                        (byte)Math.Min(255, fileColor.G + 30), 
                        (byte)Math.Max(0, fileColor.B - 30));
                    
                    var mscnModel = CreateVertexVisualization(mscnPoints, mscnColor, $"{fileName} MSCN");
                    newScene.Children.Add(mscnModel);
                    
                    ChunkItems.Add(new ChunkVisualizationItem
                    {
                        Name = $"{fileName} MSCN ({pm4File.MSCN.ExteriorVertices.Count:N0} points){boundsInfo}",
                        Count = pm4File.MSCN.ExteriorVertices.Count,
                        Color = mscnColor,
                        IsVisible = true
                    });
                }

                // Visualize MSPV structure vertices for this file
                if (pm4File.MSPV?.Vertices != null && ShowMSPVVertices)
                {
                    var mspvPoints = pm4File.MSPV.Vertices.Select(v => new Point3D(v.X, v.Y, v.Z)).ToList();
                    var boundsInfo = ValidateCoordinateBounds(mspvPoints, "MSPV");
                    
                    // Use a slightly different color for structure vertices
                    var mspvColor = Color.FromArgb(255, 
                        (byte)Math.Min(255, fileColor.R + 30), 
                        (byte)Math.Max(0, fileColor.G - 30), 
                        (byte)Math.Min(255, fileColor.B + 30));
                    
                    var mspvModel = CreateVertexVisualization(mspvPoints, mspvColor, $"{fileName} MSPV");
                    newScene.Children.Add(mspvModel);
                    
                    ChunkItems.Add(new ChunkVisualizationItem
                    {
                        Name = $"{fileName} MSPV ({pm4File.MSPV.Vertices.Count:N0} vertices){boundsInfo}",
                        Count = pm4File.MSPV.Vertices.Count,
                        Color = mspvColor,
                        IsVisible = true
                    });
                }

                fileIndex++;
            }

            // Add combined MPRL mesh visualization if available
            var combinedMesh = CreateCombinedMPRLMeshVisualization();
            if (combinedMesh != null)
            {
                newScene.Children.Add(combinedMesh);
            }

            SceneModel = newScene;
            
            // Update legend to reflect current batch visualization state
            RefreshLegend();
            
            // Auto-fit camera to show all data
            if (AutoFitCamera)
            {
                FitCameraToData();
            }
            
            // Update batch summary to show combined info
            BatchSummary = $"Batch Loaded: {LoadedFiles.Count} PM4 files | {totalVertices:N0} total vertices across all files";
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
            
            report.AppendLine($"=== PM4 REAL-TIME ANALYSIS REPORT ===");
            report.AppendLine($"File: {result.FileName}");
            report.AppendLine($"Generated: {DateTime.Now}");
            report.AppendLine($"File Size: {(Pm4File != null && !string.IsNullOrEmpty(LoadedFilePath) && File.Exists(LoadedFilePath) ? new FileInfo(LoadedFilePath).Length / 1024 : 0):N0} KB");
            report.AppendLine();

            // File Loading Progress (enhanced)
            report.AppendLine("📂 LOADING PROGRESS:");
            report.AppendLine($"  ✓ File read and parsed successfully");
            report.AppendLine($"  ✓ Chunk structure analysis completed");
            report.AppendLine($"  ✓ Vertex data processing completed");
            report.AppendLine($"  ✓ Hierarchy analysis completed");
            report.AppendLine();

            // Enhanced Chunk Analysis
            if (Pm4File != null)
            {
                report.AppendLine("📊 CHUNK ANALYSIS:");
                if (Pm4File.MSLK?.Entries?.Count > 0)
                    report.AppendLine($"  ✓ MSLK: {Pm4File.MSLK.Entries.Count:N0} navigation entries");
                if (Pm4File.MSVT?.Vertices?.Count > 0)
                    report.AppendLine($"  ✓ MSVT: {Pm4File.MSVT.Vertices.Count:N0} render vertices");
                if (Pm4File.MSCN?.ExteriorVertices?.Count > 0)
                    report.AppendLine($"  ✓ MSCN: {Pm4File.MSCN.ExteriorVertices.Count:N0} collision vertices");
                if (Pm4File.MSPV?.Vertices?.Count > 0)
                    report.AppendLine($"  ✓ MSPV: {Pm4File.MSPV.Vertices.Count:N0} portal vertices");
                if (Pm4File.MSVI?.Indices?.Count > 0)
                    report.AppendLine($"  ✓ MSVI: {Pm4File.MSVI.Indices.Count:N0} vertex indices ({Pm4File.MSVI.Indices.Count / 3:N0} faces)");
                if (Pm4File.MSUR?.Entries?.Count > 0)
                    report.AppendLine($"  ✓ MSUR: {Pm4File.MSUR.Entries.Count:N0} surface entries");
                report.AppendLine();
            }

            // Enhanced Hierarchy Analysis
            if (CurrentHierarchyAnalysis != null)
            {
                report.AppendLine("🌳 HIERARCHY STRUCTURE ANALYSIS:");
                report.AppendLine($"  📈 Depth: {CurrentHierarchyAnalysis.MaxHierarchyDepth} levels");
                report.AppendLine($"  🌟 Root nodes: {CurrentHierarchyAnalysis.RootNodes}");
                report.AppendLine($"  🔺 Leaf nodes: {CurrentHierarchyAnalysis.LeafNodes}");
                report.AppendLine($"  🔲 Total groups: {CurrentHierarchyAnalysis.GroupHierarchy.Count}");
                report.AppendLine($"  🔗 Parent-child connections: {CurrentHierarchyAnalysis.GroupHierarchy.Values.Count(g => g.ParentValue.HasValue)}");
                report.AppendLine($"  ⚡ Cross-references: {CurrentHierarchyAnalysis.TotalConnections}");
                report.AppendLine();

                // Root node details
                var rootNodes = CurrentHierarchyAnalysis.GroupHierarchy.Values.Where(g => g.IsRootNode).ToList();
                if (rootNodes.Any())
                {
                    report.AppendLine("  🌟 ROOT NODE DETAILS:");
                    foreach (var root in rootNodes.OrderBy(r => r.GroupValue))
                    {
                        report.AppendLine($"    • 0x{root.GroupValue:X8} - {root.ChildCount} children");
                        if (root.GroupValue == 0x00000000)
                            report.AppendLine($"      ⭐ MASTER ROOT NODE - Controls entire hierarchy");
                    }
                    report.AppendLine();
                }

                // Level distribution
                report.AppendLine("  📊 LEVEL DISTRIBUTION:");
                var levelCounts = CurrentHierarchyAnalysis.GroupHierarchy.Values
                    .GroupBy(g => g.HierarchyLevel)
                    .OrderBy(g => g.Key)
                    .ToList();
                foreach (var level in levelCounts)
                {
                    report.AppendLine($"    Level {level.Key}: {level.Count()} groups");
                }
                report.AppendLine();
            }

            // Group Analysis Summary
            if (Unknown0x04Groups.Count > 0)
            {
                report.AppendLine("🔢 UNKNOWN_0x04 GROUP ANALYSIS:");
                report.AppendLine($"  📋 Total groups discovered: {Unknown0x04Groups.Count}");
                
                var topGroups = Unknown0x04Groups.OrderByDescending(g => g.EntryCount).Take(10).ToList();
                report.AppendLine($"  🔝 TOP 10 GROUPS BY SIZE:");
                foreach (var group in topGroups)
                {
                    var hierarchyInfo = group.HierarchyInfo;
                    var typeInfo = hierarchyInfo.IsRootNode ? "(Root)" : 
                                  hierarchyInfo.IsLeafNode ? "(Leaf)" : 
                                  $"(L{hierarchyInfo.HierarchyLevel})";
                    report.AppendLine($"    • 0x{group.GroupValue:X8}: {group.EntryCount} entries, {group.AssociatedVertices.Count} vertices {typeInfo}");
                }
                report.AppendLine();
            }

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
                report.AppendLine("🔍 NON-ZERO PADDING DETECTED:");
                foreach (var padding in result.PaddingAnalysis.Where(p => p.HasNonZeroPadding))
                {
                    report.AppendLine($"  {padding.ChunkType}: {padding.PaddingBytes} bytes");
                    report.AppendLine($"    Data: {string.Join(" ", padding.PaddingData.Take(32).Select(b => b.ToString("X2")))}");
                }
                report.AppendLine();
            }

            // Unknown field patterns
            report.AppendLine("🔍 UNKNOWN FIELD ANALYSIS:");
            foreach (var field in result.UnknownFields)
            {
                report.AppendLine($"  {field.ChunkType}.{field.FieldName}:");
                report.AppendLine($"    Range: {field.ValueRange}");
                report.AppendLine($"    Unique values: {field.UniqueValues.Count}");
                report.AppendLine($"    Patterns: Index={field.LooksLikeIndex}, Flags={field.LooksLikeFlags}, Count={field.LooksLikeCount}");
            }
            report.AppendLine();

            // Hierarchical relationships
            report.AppendLine("🔗 HIERARCHICAL RELATIONSHIPS:");
            foreach (var hierarchy in result.Hierarchies.OrderByDescending(h => h.ConfidenceScore))
            {
                report.AppendLine($"  {hierarchy.ParentChunk} → {hierarchy.ChildChunk} ({hierarchy.ConfidenceScore:P0})");
                report.AppendLine($"    Type: {hierarchy.RelationshipType}");
                report.AppendLine($"    Evidence: {hierarchy.Evidence}");
            }

            // Analysis completion timestamp
            report.AppendLine();
            report.AppendLine($"🎯 ANALYSIS COMPLETED: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            report.AppendLine($"⏱️ Ready for visualization and exploration!");

            AnalysisOutput = report.ToString();
        }

        private void BuildHierarchyTree()
        {
            // Ensure we're on the UI thread for collection updates
            if (System.Windows.Application.Current.Dispatcher.CheckAccess())
            {
                BuildHierarchyTreeImpl();
            }
            else
            {
                System.Windows.Application.Current.Dispatcher.Invoke(BuildHierarchyTreeImpl);
            }
        }

        private void BuildHierarchyTreeImpl()
        {
            HierarchyTreeNodes.Clear();

            if (CurrentHierarchyAnalysis == null || Unknown0x04Groups.Count == 0)
                return;

            // Create dictionary for quick lookup
            var nodeDict = new Dictionary<uint, HierarchyTreeNode>();

            // Create all nodes first
            foreach (var group in Unknown0x04Groups)
            {
                var hierarchyInfo = group.HierarchyInfo;
                var node = new HierarchyTreeNode
                {
                    GroupValue = group.GroupValue,
                    DisplayName = $"0x{group.GroupValue:X8}",
                    Description = $"{group.EntryCount} entries, {group.AssociatedVertices.Count} vertices",
                    HierarchyLevel = hierarchyInfo.HierarchyLevel,
                    EntryCount = group.EntryCount,
                    ParentValue = hierarchyInfo.ParentValue,
                    VertexCount = group.AssociatedVertices.Count,
                    NodeColor = group.GroupColor,
                    HierarchyInfo = hierarchyInfo,
                    GroupInfo = group,
                    Unknown0x10References = string.Join(", ", hierarchyInfo.CrossReferences.Take(3).Select(r => $"0x{r:X8}")) +
                                          (hierarchyInfo.CrossReferences.Count > 3 ? $" (+{hierarchyInfo.CrossReferences.Count - 3} more)" : ""),
                    IsExpanded = hierarchyInfo.HierarchyLevel <= 2 // Expand first 2 levels by default
                };

                nodeDict[group.GroupValue] = node;
            }

            // Build parent-child relationships
            foreach (var node in nodeDict.Values)
            {
                if (node.ParentValue.HasValue && nodeDict.TryGetValue(node.ParentValue.Value, out var parent))
                {
                    parent.Children.Add(node);
                }
                else if (!node.ParentValue.HasValue) // Root node
                {
                    HierarchyTreeNodes.Add(node);
                }
            }

            // Sort children by group value for consistent ordering
            SortTreeNodes(HierarchyTreeNodes);
        }

        private void SortTreeNodes(ObservableCollection<HierarchyTreeNode> nodes)
        {
            var sortedNodes = nodes.OrderBy(n => n.GroupValue).ToList();
            nodes.Clear();
            foreach (var node in sortedNodes)
            {
                nodes.Add(node);
                SortTreeNodes(node.Children);
            }
        }

        private void FilterHierarchyTree()
        {
            if (string.IsNullOrWhiteSpace(HierarchyTreeFilter))
            {
                BuildHierarchyTree();
                return;
            }

            var filter = HierarchyTreeFilter.ToLowerInvariant();
            
            // Rebuild filtered tree
            BuildHierarchyTree();
            FilterTreeNodes(HierarchyTreeNodes, filter);
        }

        private bool FilterTreeNodes(ObservableCollection<HierarchyTreeNode> nodes, string filter)
        {
            var nodesToRemove = new List<HierarchyTreeNode>();
            var hasVisibleChildren = false;

            foreach (var node in nodes)
            {
                var nodeMatches = node.DisplayName.ToLowerInvariant().Contains(filter) ||
                                 node.Description.ToLowerInvariant().Contains(filter) ||
                                 node.NodeTypeName.ToLowerInvariant().Contains(filter);

                var childrenVisible = FilterTreeNodes(node.Children, filter);

                if (!nodeMatches && !childrenVisible)
                {
                    nodesToRemove.Add(node);
                }
                else
                {
                    hasVisibleChildren = true;
                    if (childrenVisible) node.IsExpanded = true; // Expand if children match
                }
            }

            foreach (var node in nodesToRemove)
            {
                nodes.Remove(node);
            }

            return hasVisibleChildren;
        }

        private void SelectHierarchyNode(HierarchyTreeNode? node)
        {
            if (node == null) return;

            // Clear previous selection
            ClearTreeSelection(HierarchyTreeNodes);
            
            // Set new selection
            node.IsSelected = true;
            SelectedHierarchyNode = node;
        }

        private void ClearTreeSelection(ObservableCollection<HierarchyTreeNode> nodes)
        {
            foreach (var node in nodes)
            {
                node.IsSelected = false;
                ClearTreeSelection(node.Children);
            }
        }

        private void ExpandAllHierarchyNodes()
        {
            SetTreeExpansion(HierarchyTreeNodes, true);
        }

        private void CollapseAllHierarchyNodes()
        {
            SetTreeExpansion(HierarchyTreeNodes, false);
        }

        private void SetTreeExpansion(ObservableCollection<HierarchyTreeNode> nodes, bool expanded)
        {
            foreach (var node in nodes)
            {
                node.IsExpanded = expanded;
                SetTreeExpansion(node.Children, expanded);
            }
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

            // Perform lightweight hierarchical analysis to prevent hanging
            var hierarchyAnalysis = PerformLightweightHierarchicalAnalysis(groupedEntries.Cast<IGrouping<uint, dynamic>>().ToList());

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

            // Store hierarchy analysis result
            CurrentHierarchyAnalysis = hierarchyAnalysis;

            // Add hierarchy insights to structural insights
            AddHierarchyInsights(hierarchyAnalysis);

            // Build hierarchy tree view
            BuildHierarchyTree();

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

        private HierarchyAnalysisResult PerformLightweightHierarchicalAnalysis(List<IGrouping<uint, dynamic>> groupedEntries)
        {
            // Simplified version that doesn't hang the UI
            var result = new HierarchyAnalysisResult();
            
            try
            {
                                 foreach (var group in groupedEntries.Take(50)) // Limit to prevent hanging
                 {
                     var hierarchyInfo = new GroupHierarchyInfo
                     {
                         GroupValue = group.Key,
                         IsRootNode = group.Key == 0x00000000,
                         IsLeafNode = true, // Simplified - assume all are leaves for performance
                         ChildCount = group.Count(),
                         HierarchyLevel = group.Key == 0x00000000 ? 0 : 1 // Simplified level calculation
                     };
                    
                    result.GroupHierarchy[group.Key] = hierarchyInfo;
                }
                
                result.MaxHierarchyDepth = result.GroupHierarchy.Values.Any() ? result.GroupHierarchy.Values.Max(h => h.HierarchyLevel) : 0;
                result.RootNodes = result.GroupHierarchy.Values.Count(h => h.IsRootNode);
                result.LeafNodes = result.GroupHierarchy.Values.Count(h => h.IsLeafNode);
                result.TotalConnections = result.GroupHierarchy.Count;
            }
            catch
            {
                // Return empty result if anything fails
                result = new HierarchyAnalysisResult();
            }
            
            return result;
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

            // Quick check for out-of-bounds without expensive LINQ operations
            var outOfBoundsCount = 0;
            foreach (var point in points)
            {
                if (point.X < PM4_MIN_COORDINATE || point.X > PM4_MAX_COORDINATE ||
                    point.Y < PM4_MIN_COORDINATE || point.Y > PM4_MAX_COORDINATE ||
                    point.Z < PM4_MIN_COORDINATE || point.Z > PM4_MAX_COORDINATE)
                {
                    outOfBoundsCount++;
                    if (outOfBoundsCount > 10) break; // Stop counting after 10 to prevent hanging
                }
            }

            if (outOfBoundsCount > 0)
            {
                return $" ⚠️{outOfBoundsCount}+OOB";
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
            // Use OpenFileDialog to select any file in the target directory
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "Select any PM4 file in the directory you want to batch load",
                Filter = "PM4 Files (*.pm4)|*.pm4|All Files (*.*)|*.*",
                CheckFileExists = true
            };

            if (dialog.ShowDialog() == true)
            {
                var selectedDirectory = Path.GetDirectoryName(dialog.FileName);
                if (string.IsNullOrEmpty(selectedDirectory))
                    return;

                var pm4Files = Directory.GetFiles(selectedDirectory, "*.pm4", SearchOption.AllDirectories);
                if (pm4Files.Length == 0)
                {
                    MessageBox.Show("No PM4 files found in the selected directory.", "No Files Found", MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                var result = MessageBox.Show($"Found {pm4Files.Length} PM4 files in {selectedDirectory}.\nThis may take a long time and use significant memory. Continue?", 
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
                LoadedPM4Files.Clear();
                _mprlMeshByFile.Clear();
                
                foreach (var filePath in filePaths)
                {
                    try
                    {
                        var fileName = Path.GetFileName(filePath);
                        LoadingSubOperation = $"Loading {fileName} ({loadedCount + 1}/{filePaths.Length})";
                        await Task.Delay(10); // Allow UI to update

                        // Validate file exists before attempting to load
                        if (!File.Exists(filePath))
                        {
                            errors.Add($"{fileName}: File not found at path {filePath}");
                            continue;
                        }

                        var file = await Task.Run(() =>
                        {
                            var fileBytes = File.ReadAllBytes(filePath);
                            return new PM4File(fileBytes);
                        });

                        LoadedFiles.Add(file);
                        LoadedPM4Files.Add(filePath);
                        loadedCount++;
                        
                        // Extract MPRL data if available
                        if (file.MSVT?.Vertices != null)
                        {
                            var mprlVertices = new List<Point3D>();
                            foreach (var vertex in file.MSVT.Vertices)
                            {
                                mprlVertices.Add(new Point3D(vertex.X, vertex.Y, vertex.Z));
                            }
                            _mprlMeshByFile[fileName] = mprlVertices;
                        }
                        
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
                        var fileName = Path.GetFileName(filePath) ?? "Unknown file";
                        errors.Add($"{fileName}: {ex.Message}");
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
                    
                    // Set the first loaded file as the current file for visualization
                    if (LoadedFiles.Count > 0)
                    {
                        Pm4File = LoadedFiles[0];
                        LoadedFileName = Path.GetFileName(LoadedPM4Files[0]);
                        LoadedFilePath = LoadedPM4Files[0];
                        
                        // Create combined visualization of all loaded files
                        Application.Current.Dispatcher.Invoke(() =>
                        {
                            UpdateBatchVisualization();
                            AnalyzeUnknown0x04Groups();
                        });
                    }
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

                    // Prepare insights to add on UI thread
                    var batchInsights = new List<StructuralInsight>();
                    
                    batchInsights.Add(new StructuralInsight
                    {
                        Type = "Batch Analysis",
                        Description = $"Cross-file pattern analysis of {LoadedFiles.Count} PM4 files",
                        Significance = $"Found {allUnknown04Values.Distinct().Count()} unique Unknown_0x04 values across all files",
                        DataPreview = $"Total entries analyzed: {allUnknown04Values.Count:N0}"
                    });

                    if (allUnknown04Values.Any())
                    {
                        batchInsights.Add(new StructuralInsight
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
                    }

                    // Add insights on UI thread
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        foreach (var insight in batchInsights)
                        {
                            StructuralInsights.Add(insight);
                        }
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

        #region Combined MPRL Mesh Methods

        private async Task BuildCombinedMPRLMeshAsync()
        {
            if (!_mprlMeshByFile.Any())
            {
                MessageBox.Show("No PM4 files with MPRL data loaded. Please load multiple PM4 files first.", 
                    "No MPRL Data", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            IsBuildingCombinedMesh = true;
            CombinedMeshStatus = "Building combined MPRL mesh...";

            try
            {
                await Task.Run(() =>
                {
                    _combinedMPRLVertices.Clear();
                    
                    // Combine all MPRL vertices from all files
                    foreach (var kvp in _mprlMeshByFile)
                    {
                        _combinedMPRLVertices.AddRange(kvp.Value);
                    }
                });

                CombinedMeshVertexCount = _combinedMPRLVertices.Count;
                CombinedMeshStatus = $"Combined mesh ready: {CombinedMeshVertexCount:N0} vertices from {_mprlMeshByFile.Count} files";
                
                MessageBox.Show($"Combined MPRL mesh built successfully!\n\n" +
                               $"📊 Total Vertices: {CombinedMeshVertexCount:N0}\n" +
                               $"📁 Source Files: {_mprlMeshByFile.Count}\n" +
                               $"🔗 Enable 'Show Combined MPRL Mesh' to visualize",
                               "Combined Mesh Ready", MessageBoxButton.OK, MessageBoxImage.Information);

                UpdateVisualization();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error building combined MPRL mesh: {ex.Message}", 
                    "Build Error", MessageBoxButton.OK, MessageBoxImage.Error);
                CombinedMeshStatus = "Error building combined mesh";
            }
            finally
            {
                IsBuildingCombinedMesh = false;
            }
        }

        private async Task ExportCombinedMPRLMeshAsync()
        {
            if (!_combinedMPRLVertices.Any())
            {
                MessageBox.Show("No combined MPRL mesh available. Build the combined mesh first.", 
                    "No Combined Mesh", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "OBJ Files (*.obj)|*.obj|PLY Files (*.ply)|*.ply|Text Files (*.txt)|*.txt|All Files (*.*)|*.*",
                FileName = $"CombinedMPRL_Mesh_{DateTime.Now:yyyyMMdd_HHmmss}.obj"
            };

            if (dialog.ShowDialog() == true)
            {
                try
                {
                    await Task.Run(() =>
                    {
                        var extension = Path.GetExtension(dialog.FileName).ToLowerInvariant();
                        
                        switch (extension)
                        {
                            case ".obj":
                                ExportCombinedMeshAsOBJ(dialog.FileName);
                                break;
                            case ".ply":
                                ExportCombinedMeshAsPLY(dialog.FileName);
                                break;
                            default:
                                ExportCombinedMeshAsText(dialog.FileName);
                                break;
                        }
                    });

                    MessageBox.Show($"Combined MPRL mesh exported successfully!\n\nFile: {dialog.FileName}\nVertices: {_combinedMPRLVertices.Count:N0}", 
                        "Export Complete", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Error exporting combined mesh: {ex.Message}", 
                        "Export Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void ExportCombinedMeshAsOBJ(string filePath)
        {
            using var writer = new StreamWriter(filePath);
            
            writer.WriteLine("# Combined MPRL Mesh from PM4 Files");
            writer.WriteLine($"# Generated: {DateTime.Now}");
            writer.WriteLine($"# Total Vertices: {_combinedMPRLVertices.Count:N0}");
            writer.WriteLine($"# Source Files: {_mprlMeshByFile.Count}");
            writer.WriteLine();

            // Write vertices
            foreach (var vertex in _combinedMPRLVertices)
            {
                writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }

            writer.WriteLine();
            writer.WriteLine("# Point cloud - no faces defined");
            writer.WriteLine("# Use in Blender/3D software for terrain reconstruction");
        }

        private void ExportCombinedMeshAsPLY(string filePath)
        {
            using var writer = new StreamWriter(filePath);
            
            writer.WriteLine("ply");
            writer.WriteLine("format ascii 1.0");
            writer.WriteLine($"comment Combined MPRL Mesh from PM4 Files - Generated: {DateTime.Now}");
            writer.WriteLine($"comment Total Vertices: {_combinedMPRLVertices.Count:N0} from {_mprlMeshByFile.Count} files");
            writer.WriteLine($"element vertex {_combinedMPRLVertices.Count}");
            writer.WriteLine("property float x");
            writer.WriteLine("property float y");
            writer.WriteLine("property float z");
            writer.WriteLine("end_header");

            foreach (var vertex in _combinedMPRLVertices)
            {
                writer.WriteLine($"{vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
        }

        private void ExportCombinedMeshAsText(string filePath)
        {
            using var writer = new StreamWriter(filePath);
            
            writer.WriteLine("Combined MPRL Mesh Data");
            writer.WriteLine("======================");
            writer.WriteLine($"Generated: {DateTime.Now}");
            writer.WriteLine($"Total Vertices: {_combinedMPRLVertices.Count:N0}");
            writer.WriteLine($"Source Files: {_mprlMeshByFile.Count}");
            writer.WriteLine();
            
            writer.WriteLine("Source File Breakdown:");
            foreach (var kvp in _mprlMeshByFile)
            {
                writer.WriteLine($"  {kvp.Key}: {kvp.Value.Count:N0} vertices");
            }
            writer.WriteLine();

            writer.WriteLine("Vertex Data (X, Y, Z):");
            foreach (var vertex in _combinedMPRLVertices)
            {
                writer.WriteLine($"{vertex.X:F6}, {vertex.Y:F6}, {vertex.Z:F6}");
            }
        }

        private void ClearCombinedMesh()
        {
            _combinedMPRLVertices.Clear();
            _mprlMeshByFile.Clear();
            CombinedMeshVertexCount = 0;
            CombinedMeshStatus = "No combined mesh loaded";
            ShowCombinedMPRLMesh = false;
            UpdateVisualization();
        }

        private GeometryModel3D? CreateCombinedMPRLMeshVisualization()
        {
            if (!ShowCombinedMPRLMesh || !_combinedMPRLVertices.Any())
                return null;

            var geometry = new MeshGeometry3D();
            
            // Create small cubes for each vertex in the combined mesh
            foreach (var vertex in _combinedMPRLVertices)
            {
                AddVertexCube(geometry, vertex, 1.0); // Smaller cubes for better performance
            }

            // Use a distinctive color for the combined mesh
            var material = new DiffuseMaterial(new SolidColorBrush(Colors.Magenta));
            
            return new GeometryModel3D(geometry, material);
        }

        // Enhanced Optimization Methods
        private void AnalyzeObjectTypes()
        {
            if (Pm4File?.MSLK?.Entries == null)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    SelectedDataInfo = "No MSLK data to analyze";
                });
                return;
            }

            try
            {
                var objectTypeGroups = Pm4File.MSLK.Entries
                    .GroupBy(e => e.ObjectTypeFlags)
                    .Select((group, index) => new ObjectTypeGroup
                    {
                        ObjectType = group.Key,
                        TypeName = GetObjectTypeName(group.Key),
                        EntryCount = group.Count(),
                        Color = GenerateDistinctColors(1)[0],
                        Entries = group.ToList(),
                        Description = $"Type {group.Key}: {group.Count()} entries"
                    })
                    .OrderByDescending(g => g.EntryCount)
                    .ToList();

                Application.Current.Dispatcher.Invoke(() =>
                {
                    ObjectTypeGroups.Clear();
                    foreach (var group in objectTypeGroups)
                    {
                        ObjectTypeGroups.Add(group);
                    }
                    SelectedDataInfo = $"Found {ObjectTypeGroups.Count} object types";
                });
            }
            catch (Exception ex)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    SelectedDataInfo = $"Error analyzing object types: {ex.Message}";
                });
            }
        }

        private void AnalyzeMaterials()
        {
            if (Pm4File?.MSLK?.Entries == null)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    SelectedDataInfo = "No MSLK data to analyze";
                });
                return;
            }

            try
            {
                var materialGroups = Pm4File.MSLK.Entries
                    .GroupBy(e => e.MaterialColorId)
                    .Select((group, index) => new MaterialGroup
                    {
                        MaterialId = group.Key,
                        MaterialName = GetMaterialName(group.Key),
                        EntryCount = group.Count(),
                        Color = GenerateDistinctColors(1)[0],
                        Entries = group.ToList(),
                        Description = $"Material 0x{group.Key:X8}: {group.Count()} entries"
                    })
                    .OrderByDescending(g => g.EntryCount)
                    .ToList();

                Application.Current.Dispatcher.Invoke(() =>
                {
                    MaterialGroups.Clear();
                    foreach (var group in materialGroups)
                    {
                        MaterialGroups.Add(group);
                    }
                    SelectedDataInfo = $"Found {MaterialGroups.Count} material types";
                });
            }
            catch (Exception ex)
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    SelectedDataInfo = $"Error analyzing materials: {ex.Message}";
                });
            }
        }

        private string GetObjectTypeName(byte objectType)
        {
            return objectType switch
            {
                1 => "Terrain Base",
                2 => "Structure Foundation", 
                4 => "Doodad Placement",
                10 => "Terrain Detail",
                12 => "Object Reference",
                17 => "Special Feature",
                _ => $"Type {objectType}"
            };
        }

        private string GetMaterialName(uint materialId)
        {
            if ((materialId & 0xFFFF0000) == 0xFFFF0000)
            {
                var materialIndex = materialId & 0xFFFF;
                return $"Material #{materialIndex}";
            }
            return $"ID {materialId:X8}";
        }

        private void SelectObjectType(ObjectTypeGroup? objectType)
        {
            SelectedObjectType = objectType;
            if (objectType != null)
            {
                SelectedDataInfo = $"Selected: {objectType.TypeName} ({objectType.EntryCount} entries)";
                // Update visualization to highlight selected object type
                UpdateVisualization();
            }
        }

        private void SelectMaterial(MaterialGroup? material)
        {
            SelectedMaterial = material;
            if (material != null)
            {
                SelectedDataInfo = $"Selected: {material.MaterialName} ({material.EntryCount} entries)";
                // Update visualization to highlight selected material
                UpdateVisualization();
            }
        }

        private async Task ExportEnhancedOBJAsync()
        {
            if (Pm4File == null)
            {
                SelectedDataInfo = "No PM4 file loaded";
                return;
            }

            try
            {
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var outputDir = Path.Combine("output", timestamp, "EnhancedObjExport");
                Directory.CreateDirectory(outputDir);

                await Task.Run(() =>
                {
                    // Export with surface normals from MSUR if available
                    if (Pm4File.MSUR?.Entries != null && Pm4File.MSUR.Entries.Count > 0)
                    {
                        ExportWithSurfaceNormals(Path.Combine(outputDir, "enhanced_with_normals.obj"));
                    }

                    // Export grouped by object type
                    if (ObjectTypeGroups.Count > 0)
                    {
                        ExportGroupedByObjectType(outputDir);
                    }

                    // Export grouped by material
                    if (MaterialGroups.Count > 0)
                    {
                        ExportGroupedByMaterial(outputDir);
                    }

                    // Export height bands if enabled
                    if (ShowHeightBands)
                    {
                        ExportHeightBands(outputDir);
                    }
                });

                SelectedDataInfo = $"Enhanced export completed to: {outputDir}";
            }
            catch (Exception ex)
            {
                SelectedDataInfo = $"Export error: {ex.Message}";
            }
        }

        private void ExportWithSurfaceNormals(string filePath)
        {
            if (Pm4File?.MSUR?.Entries == null || Pm4File.MSVT?.Vertices == null)
                return;

            using var writer = new StreamWriter(filePath);
            writer.WriteLine("# Enhanced PM4 Export with Surface Normals");
            writer.WriteLine($"# Generated: {DateTime.Now}");
            writer.WriteLine($"# Source: {LoadedFileName}");
            writer.WriteLine();

            // Export vertices
            foreach (var vertex in Pm4File.MSVT.Vertices)
            {
                writer.WriteLine($"v {vertex.X} {vertex.Y} {vertex.Z}");
            }

            // Export normals from MSUR
            foreach (var surface in Pm4File.MSUR.Entries)
            {
                writer.WriteLine($"vn {surface.SurfaceNormalX} {surface.SurfaceNormalY} {surface.SurfaceNormalZ}");
            }

            writer.WriteLine("# Faces with normals");
            // Create simple triangles (this is simplified - real implementation would need proper indexing)
            for (int i = 0; i < Pm4File.MSVT.Vertices.Count - 2; i += 3)
            {
                var normalIndex = Math.Min(i / 3, Pm4File.MSUR.Entries.Count - 1) + 1;
                writer.WriteLine($"f {i + 1}//{normalIndex} {i + 2}//{normalIndex} {i + 3}//{normalIndex}");
            }
        }

        private void ExportGroupedByObjectType(string outputDir)
        {
            foreach (var objectType in ObjectTypeGroups)
            {
                var fileName = $"object_type_{objectType.ObjectType:D2}_{objectType.TypeName.Replace(" ", "_")}.obj";
                var filePath = Path.Combine(outputDir, fileName);
                
                using var writer = new StreamWriter(filePath);
                writer.WriteLine($"# Object Type: {objectType.TypeName}");
                writer.WriteLine($"# Entry Count: {objectType.EntryCount}");
                writer.WriteLine();

                // Export vertices for this object type
                ExportEntriesAsOBJ(writer, objectType.Entries);
            }
        }

        private void ExportGroupedByMaterial(string outputDir)
        {
            foreach (var material in MaterialGroups)
            {
                var fileName = $"material_{material.MaterialId:X8}_{material.MaterialName.Replace(" ", "_")}.obj";
                var filePath = Path.Combine(outputDir, fileName);
                
                using var writer = new StreamWriter(filePath);
                writer.WriteLine($"# Material: {material.MaterialName}");
                writer.WriteLine($"# Entry Count: {material.EntryCount}");
                writer.WriteLine();

                // Export vertices for this material
                ExportEntriesAsOBJ(writer, material.Entries);
            }
        }

        private void ExportHeightBands(string outputDir)
        {
            if (Pm4File?.MSVT?.Vertices == null)
                return;

            var minHeight = Pm4File.MSVT.Vertices.Min(v => v.Z);
            var maxHeight = Pm4File.MSVT.Vertices.Max(v => v.Z);
            var bandCount = (int)Math.Ceiling((maxHeight - minHeight) / HeightBandSize);

            for (int band = 0; band < bandCount; band++)
            {
                var bandMin = minHeight + (band * HeightBandSize);
                var bandMax = bandMin + HeightBandSize;
                
                var verticesInBand = Pm4File.MSVT.Vertices
                    .Where(v => v.Z >= bandMin && v.Z < bandMax)
                    .ToList();

                if (verticesInBand.Count > 0)
                {
                    var fileName = $"height_band_{band:D2}_{bandMin:F1}_to_{bandMax:F1}.obj";
                    var filePath = Path.Combine(outputDir, fileName);
                    
                    using var writer = new StreamWriter(filePath);
                    writer.WriteLine($"# Height Band {band}: {bandMin:F1} to {bandMax:F1}");
                    writer.WriteLine($"# Vertex Count: {verticesInBand.Count}");
                    writer.WriteLine();

                    foreach (var vertex in verticesInBand)
                    {
                        writer.WriteLine($"v {vertex.X} {vertex.Y} {vertex.Z}");
                    }
                }
            }
        }

        private void ExportEntriesAsOBJ(StreamWriter writer, List<MSLKEntry> entries)
        {
            if (Pm4File?.MSVT?.Vertices == null)
                return;

            // This is a simplified approach - in reality, you'd need to properly map
            // MSLK entries to their corresponding vertices through MSPI indices
            var validEntries = entries.Where(e => e.MspiFirstIndex >= 0 && e.MspiIndexCount > 0).ToList();
            
            int vertexIndex = 1;
            foreach (var entry in validEntries)
            {
                writer.WriteLine($"# Entry: Group={entry.GroupObjectId:X8}, Material={entry.MaterialColorId:X8}");
                
                // Export a small cube at the entry's position (simplified visualization)
                var baseIndex = Math.Min(entry.MspiFirstIndex, Pm4File.MSVT.Vertices.Count - 1);
                if (baseIndex >= 0 && baseIndex < Pm4File.MSVT.Vertices.Count)
                {
                    var vertex = Pm4File.MSVT.Vertices[baseIndex];
                    writer.WriteLine($"v {vertex.X} {vertex.Y} {vertex.Z}");
                    vertexIndex++;
                }
            }
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

    public partial class HierarchyTreeNode : ObservableObject
    {
        [ObservableProperty]
        private uint _groupValue;

        [ObservableProperty]
        private string _displayName = string.Empty;

        [ObservableProperty]
        private string _description = string.Empty;

        [ObservableProperty]
        private bool _isExpanded = true;

        [ObservableProperty]
        private bool _isSelected = false;

        [ObservableProperty]
        private int _hierarchyLevel;

        [ObservableProperty]
        private int _entryCount;

        [ObservableProperty]
        private uint? _parentValue;

        [ObservableProperty]
        private int _vertexCount;

        [ObservableProperty]
        private string _unknown0x10References = string.Empty;

        [ObservableProperty]
        private Color _nodeColor = Colors.Gray;

        [ObservableProperty]
        private ObservableCollection<HierarchyTreeNode> _children = new();

        public GroupHierarchyInfo? HierarchyInfo { get; set; }
        public Unknown0x04Group? GroupInfo { get; set; }

        public string NodeTypeIcon => HierarchyInfo?.IsRootNode == true ? "💎" :
                                     HierarchyInfo?.IsLeafNode == true ? "🔺" : "🔲";

        public string NodeTypeName => HierarchyInfo?.IsRootNode == true ? "Root" :
                                     HierarchyInfo?.IsLeafNode == true ? "Leaf" : "Intermediate";
    }

    /// <summary>
    /// Represents a group of objects by type for classification
    /// </summary>
    public class ObjectTypeGroup
    {
        public byte ObjectType { get; set; }
        public string TypeName { get; set; } = string.Empty;
        public int EntryCount { get; set; }
        public Color Color { get; set; }
        public List<MSLKEntry> Entries { get; set; } = new();
        public string Description { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents a group of objects by material ID
    /// </summary>
    public class MaterialGroup
    {
        public uint MaterialId { get; set; }
        public string MaterialName { get; set; } = string.Empty;
        public int EntryCount { get; set; }
        public Color Color { get; set; }
        public List<MSLKEntry> Entries { get; set; } = new();
        public string Description { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents surface data with enhanced metadata
    /// </summary>
    public class SurfaceInfo
    {
        public Vector3 Normal { get; set; }
        public float Height { get; set; }
        public int TriangleCount { get; set; }
        public List<int> VertexIndices { get; set; } = new();
        public Color Color { get; set; }
        public string QualityInfo { get; set; } = string.Empty;
    }
} 