using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Correlates data across PM4 chunks based on special values to identify asset properties.
    /// </summary>
    public class SpecialValueCorrelator
    {
        private readonly ILogger<SpecialValueCorrelator> _logger;
        private string _csvDirectory = string.Empty;
        
        // Metadata dictionaries - smaller footprint than full data
        private readonly Dictionary<string, HashSet<uint>> _fileSpecialValues = new();
        private readonly Dictionary<uint, List<string>> _filesBySpecialValue = new();
        private readonly Dictionary<uint, SpecialValueMetadata> _specialValueMetadata = new();
        
        // Batch size for processing
        private const int DEFAULT_BATCH_SIZE = 100;
        private const int MAX_ITEMS_PER_CATEGORY = 20; // Maximum items to include in detailed reports
        
        /// <summary>
        /// Initializes a new instance of the <see cref="SpecialValueCorrelator"/> class.
        /// </summary>
        /// <param name="logger">Optional logger for logging operations.</param>
        public SpecialValueCorrelator(ILogger<SpecialValueCorrelator>? logger = null)
        {
            _logger = logger ?? NullLogger<SpecialValueCorrelator>.Instance;
        }
        
        /// <summary>
        /// Analyzes CSV files to correlate special values with all available data.
        /// </summary>
        /// <param name="csvDirectory">Directory containing the CSV files.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task AnalyzeDataAsync(string csvDirectory)
        {
            _logger.LogInformation("Starting special value correlation analysis from {Directory}", csvDirectory);
            _csvDirectory = csvDirectory;
            
            // Clear any previous data
            _fileSpecialValues.Clear();
            _filesBySpecialValue.Clear();
            _specialValueMetadata.Clear();
            
            // Only load the metadata from positions.csv to get special values
            await LoadSpecialValueMetadataAsync(Path.Combine(csvDirectory, "positions.csv"));
            
            _logger.LogInformation("Completed metadata analysis. Found {Count} unique special values.", _specialValueMetadata.Count);
        }
        
        /// <summary>
        /// Generates a comprehensive report for all special values.
        /// </summary>
        /// <param name="outputPath">Path to write the report.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task GenerateReportAsync(string outputPath)
        {
            _logger.LogInformation("Starting report generation to {OutputPath}", outputPath);
            
            // Process in batches to reduce memory usage
            using var writer = new StreamWriter(outputPath, false);
            
            // Write header
            await writer.WriteLineAsync("# Special Value Correlation Report");
            await writer.WriteLineAsync($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            await writer.WriteLineAsync();
            await writer.WriteLineAsync($"Total Unique Special Values: {_specialValueMetadata.Count}");
            await writer.WriteLineAsync();
            
            // Summary table for all special values
            await writer.WriteLineAsync("## Special Value Summary");
            await writer.WriteLineAsync("| Special Value | Files | Position Entries | Vertex Count | Triangle Count | Normal Count |");
            await writer.WriteLineAsync("|--------------|-------|------------------|--------------|----------------|--------------|");
            
            _logger.LogInformation("Generating summary table for {Count} special values", _specialValueMetadata.Count);
            
            // Process special values in batches
            var specialValues = _specialValueMetadata.Keys.OrderBy(k => k).ToList();
            int batchesProcessed = 0;
            int batchSize = Math.Min(DEFAULT_BATCH_SIZE, specialValues.Count);
            
            for (int i = 0; i < specialValues.Count; i += batchSize)
            {
                var batchValues = specialValues.Skip(i).Take(batchSize).ToList();
                _logger.LogInformation("Processing batch {Batch} of special values ({Start}-{End} of {Total})",
                    batchesProcessed + 1, i + 1, Math.Min(i + batchSize, specialValues.Count), specialValues.Count);
                
                foreach (var specialValue in batchValues)
                {
                    var metadata = _specialValueMetadata[specialValue];
                    var files = _filesBySpecialValue.TryGetValue(specialValue, out var fileList) ? fileList.Count : 0;
                    
                    await writer.WriteLineAsync($"| {specialValue} | {files} | {metadata.PositionCount} | {metadata.VertexCount} | {metadata.TriangleCount} | {metadata.NormalCount} |");
                }
                
                batchesProcessed++;
                _logger.LogInformation("Completed summary for batch {Batch} ({PercentComplete:F1}% complete)",
                    batchesProcessed, (float)Math.Min(i + batchSize, specialValues.Count) / specialValues.Count * 100);
                
                // Explicitly force garbage collection between batches to release memory
                if (batchesProcessed % 5 == 0)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
            }
            
            await writer.WriteLineAsync();
            await writer.WriteLineAsync("## Detailed Special Value Reports");
            
            // Generate detailed reports in batches
            batchesProcessed = 0;
            
            for (int i = 0; i < specialValues.Count; i += batchSize)
            {
                var batchValues = specialValues.Skip(i).Take(batchSize).ToList();
                _logger.LogInformation("Generating detailed reports for batch {Batch} of special values ({Start}-{End} of {Total})",
                    batchesProcessed + 1, i + 1, Math.Min(i + batchSize, specialValues.Count), specialValues.Count);
                
                foreach (var specialValue in batchValues)
                {
                    await GenerateDetailedReportForSpecialValueAsync(writer, specialValue);
                }
                
                batchesProcessed++;
                _logger.LogInformation("Completed detailed reports for batch {Batch} ({PercentComplete:F1}% complete)",
                    batchesProcessed, (float)Math.Min(i + batchSize, specialValues.Count) / specialValues.Count * 100);
                
                // Explicitly force garbage collection between batches
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
            
            _logger.LogInformation("Report generation completed successfully");
        }
        
        /// <summary>
        /// Generate a detailed report for a single special value
        /// </summary>
        private async Task GenerateDetailedReportForSpecialValueAsync(TextWriter writer, uint specialValue)
        {
            _logger.LogDebug("Generating detailed report for special value {SpecialValue}", specialValue);
            
            await writer.WriteLineAsync($"### Special Value: {specialValue}");
            
            // Files containing this special value
            if (_filesBySpecialValue.TryGetValue(specialValue, out var files))
            {
                await writer.WriteLineAsync("#### Files");
                foreach (var file in files.OrderBy(f => f))
                {
                    await writer.WriteLineAsync($"- {file}");
                }
                await writer.WriteLineAsync();
            }
            
            // Position entries - stream from CSV file
            await LoadAndWritePositionDataAsync(writer, specialValue);
            
            // Vertex entries - stream from CSV file
            await LoadAndWriteVertexDataAsync(writer, specialValue);
            
            // Normal entries - stream from CSV file
            await LoadAndWriteNormalDataAsync(writer, specialValue);
            
            // Triangle entries - stream from CSV file
            await LoadAndWriteTriangleDataAsync(writer, specialValue);
            
            await writer.WriteLineAsync("---");
        }
        
        /// <summary>
        /// Load and stream position data for a specific special value
        /// </summary>
        private async Task LoadAndWritePositionDataAsync(TextWriter writer, uint specialValue)
        {
            var filePath = Path.Combine(_csvDirectory, "positions.csv");
            if (!File.Exists(filePath))
            {
                return;
            }
            
            var positions = new List<PositionData>();
            
            // First pass - collect position data
            using (var reader = new StreamReader(new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)))
            {
                // Skip header
                await reader.ReadLineAsync();
                
                string? line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    var parts = line.Split(',');
                    if (parts.Length < 8) continue;
                    
                    var fileName = parts[0];
                    
                    if (!int.TryParse(parts[1], out var index)) continue;
                    var type = parts[2];
                    
                    if (!float.TryParse(parts[3], CultureInfo.InvariantCulture, out var x)) continue;
                    if (!float.TryParse(parts[4], CultureInfo.InvariantCulture, out var y)) continue;
                    if (!float.TryParse(parts[5], CultureInfo.InvariantCulture, out var z)) continue;
                    
                    var isSpecial = type == "Special";
                    
                    if (isSpecial && uint.TryParse(parts[7], out var value) && value == specialValue)
                    {
                        positions.Add(new PositionData
                        {
                            FileName = fileName,
                            Index = index,
                            IsSpecial = true,
                            X = x,
                            Y = y,
                            Z = z
                        });
                    }
                    else if (!isSpecial && index > 0)
                    {
                        // Check if previous entry was special with this specialValue
                        var previousEntry = positions.FirstOrDefault(p => 
                            p.FileName == fileName && p.Index == index - 1 && p.IsSpecial);
                            
                        if (previousEntry != null)
                        {
                            positions.Add(new PositionData
                            {
                                FileName = fileName,
                                Index = index,
                                IsSpecial = false,
                                X = x,
                                Y = y,
                                Z = z
                            });
                        }
                    }
                }
            }
            
            // Only write if we found positions
            if (positions.Count > 0)
            {
                await writer.WriteLineAsync("#### Position Entries");
                await writer.WriteLineAsync("| File | Index | Position |");
                await writer.WriteLineAsync("|------|-------|----------|");
                
                foreach (var position in positions.OrderBy(p => p.FileName).ThenBy(p => p.Index))
                {
                    await writer.WriteLineAsync($"| {position.FileName} | {position.Index} | ({position.X:F3}, {position.Y:F3}, {position.Z:F3}) |");
                }
                
                await writer.WriteLineAsync();
            }
            
            // Clear list to free memory
            positions.Clear();
        }
        
        /// <summary>
        /// Load and stream vertex data for files containing a specific special value
        /// </summary>
        private async Task LoadAndWriteVertexDataAsync(TextWriter writer, uint specialValue)
        {
            if (!_filesBySpecialValue.TryGetValue(specialValue, out var files) || files.Count == 0)
            {
                return;
            }
            
            var filePath = Path.Combine(_csvDirectory, "vertices.csv");
            if (!File.Exists(filePath))
            {
                return;
            }
            
            var vertices = new List<VertexData>();
            var filesSet = new HashSet<string>(files);
            
            // First pass - collect vertex data from files that contain the special value
            using (var reader = new StreamReader(new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)))
            {
                // Skip header
                await reader.ReadLineAsync();
                
                string? line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    var parts = line.Split(',');
                    if (parts.Length < 5) continue;
                    
                    var fileName = parts[0];
                    
                    // Only process if this file contains our special value
                    if (!filesSet.Contains(fileName)) continue;
                    
                    if (!int.TryParse(parts[1], out var index)) continue;
                    if (!float.TryParse(parts[2], CultureInfo.InvariantCulture, out var x)) continue;
                    if (!float.TryParse(parts[3], CultureInfo.InvariantCulture, out var y)) continue;
                    if (!float.TryParse(parts[4], CultureInfo.InvariantCulture, out var z)) continue;
                    
                    vertices.Add(new VertexData
                    {
                        FileName = fileName,
                        Index = index,
                        X = x,
                        Y = y,
                        Z = z
                    });
                    
                    // Only keep up to MAX_ITEMS_PER_CATEGORY vertices per file
                    var verticesPerFile = vertices.GroupBy(v => v.FileName)
                        .ToDictionary(g => g.Key, g => g.Count());
                    
                    if (verticesPerFile.TryGetValue(fileName, out var count) && count >= MAX_ITEMS_PER_CATEGORY)
                    {
                        break;
                    }
                }
            }
            
            // Only write if we found vertices
            if (vertices.Count > 0)
            {
                await writer.WriteLineAsync("#### Vertex Entries");
                await writer.WriteLineAsync("| File | Index | Position |");
                await writer.WriteLineAsync("|------|-------|----------|");
                
                // Limit to MAX_ITEMS_PER_CATEGORY vertices for readability
                var displayVertices = vertices.OrderBy(v => v.FileName).ThenBy(v => v.Index).Take(MAX_ITEMS_PER_CATEGORY).ToList();
                foreach (var vertex in displayVertices)
                {
                    await writer.WriteLineAsync($"| {vertex.FileName} | {vertex.Index} | ({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3}) |");
                }
                
                if (vertices.Count > MAX_ITEMS_PER_CATEGORY)
                {
                    await writer.WriteLineAsync($"*...and {vertices.Count - MAX_ITEMS_PER_CATEGORY} more vertices...*");
                }
                
                await writer.WriteLineAsync();
            }
            
            // Clear list to free memory
            vertices.Clear();
        }
        
        /// <summary>
        /// Load and stream normal data for files containing a specific special value
        /// </summary>
        private async Task LoadAndWriteNormalDataAsync(TextWriter writer, uint specialValue)
        {
            if (!_filesBySpecialValue.TryGetValue(specialValue, out var files) || files.Count == 0)
            {
                return;
            }
            
            var filePath = Path.Combine(_csvDirectory, "normals.csv");
            if (!File.Exists(filePath))
            {
                return;
            }
            
            var normals = new List<NormalData>();
            var filesSet = new HashSet<string>(files);
            
            // First pass - collect normal data from files that contain the special value
            using (var reader = new StreamReader(new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)))
            {
                // Skip header
                await reader.ReadLineAsync();
                
                string? line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    var parts = line.Split(',');
                    if (parts.Length < 5) continue;
                    
                    var fileName = parts[0];
                    
                    // Only process if this file contains our special value
                    if (!filesSet.Contains(fileName)) continue;
                    
                    if (!int.TryParse(parts[1], out var index)) continue;
                    if (!float.TryParse(parts[2], CultureInfo.InvariantCulture, out var x)) continue;
                    if (!float.TryParse(parts[3], CultureInfo.InvariantCulture, out var y)) continue;
                    if (!float.TryParse(parts[4], CultureInfo.InvariantCulture, out var z)) continue;
                    
                    normals.Add(new NormalData
                    {
                        FileName = fileName,
                        Index = index,
                        X = x,
                        Y = y,
                        Z = z
                    });
                    
                    // Only keep up to MAX_ITEMS_PER_CATEGORY normals per file
                    var normalsPerFile = normals.GroupBy(n => n.FileName)
                        .ToDictionary(g => g.Key, g => g.Count());
                    
                    if (normalsPerFile.TryGetValue(fileName, out var count) && count >= MAX_ITEMS_PER_CATEGORY)
                    {
                        break;
                    }
                }
            }
            
            // Only write if we found normals
            if (normals.Count > 0)
            {
                await writer.WriteLineAsync("#### Normal Entries");
                await writer.WriteLineAsync("| File | Index | Normal |");
                await writer.WriteLineAsync("|------|-------|--------|");
                
                // Limit to MAX_ITEMS_PER_CATEGORY normals for readability
                var displayNormals = normals.OrderBy(n => n.FileName).ThenBy(n => n.Index).Take(MAX_ITEMS_PER_CATEGORY).ToList();
                foreach (var normal in displayNormals)
                {
                    await writer.WriteLineAsync($"| {normal.FileName} | {normal.Index} | ({normal.X:F3}, {normal.Y:F3}, {normal.Z:F3}) |");
                }
                
                if (normals.Count > MAX_ITEMS_PER_CATEGORY)
                {
                    await writer.WriteLineAsync($"*...and {normals.Count - MAX_ITEMS_PER_CATEGORY} more normals...*");
                }
                
                await writer.WriteLineAsync();
            }
            
            // Clear list to free memory
            normals.Clear();
        }
        
        /// <summary>
        /// Load and stream triangle data for files containing a specific special value
        /// </summary>
        private async Task LoadAndWriteTriangleDataAsync(TextWriter writer, uint specialValue)
        {
            if (!_filesBySpecialValue.TryGetValue(specialValue, out var files) || files.Count == 0)
            {
                return;
            }
            
            var filePath = Path.Combine(_csvDirectory, "triangles.csv");
            if (!File.Exists(filePath))
            {
                return;
            }
            
            var triangles = new List<TriangleData>();
            var filesSet = new HashSet<string>(files);
            
            // First pass - collect triangle data from files that contain the special value
            using (var reader = new StreamReader(new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)))
            {
                // Skip header
                await reader.ReadLineAsync();
                
                string? line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    var parts = line.Split(',');
                    if (parts.Length < 5) continue;
                    
                    var fileName = parts[0];
                    
                    // Only process if this file contains our special value
                    if (!filesSet.Contains(fileName)) continue;
                    
                    if (!int.TryParse(parts[1], out var index)) continue;
                    if (!uint.TryParse(parts[2], out var v1)) continue;
                    if (!uint.TryParse(parts[3], out var v2)) continue;
                    if (!uint.TryParse(parts[4], out var v3)) continue;
                    
                    triangles.Add(new TriangleData
                    {
                        FileName = fileName,
                        Index = index,
                        V1 = v1,
                        V2 = v2,
                        V3 = v3
                    });
                    
                    // Only keep up to MAX_ITEMS_PER_CATEGORY triangles per file
                    var trianglesPerFile = triangles.GroupBy(t => t.FileName)
                        .ToDictionary(g => g.Key, g => g.Count());
                    
                    if (trianglesPerFile.TryGetValue(fileName, out var count) && count >= MAX_ITEMS_PER_CATEGORY)
                    {
                        break;
                    }
                }
            }
            
            // Only write if we found triangles
            if (triangles.Count > 0)
            {
                await writer.WriteLineAsync("#### Triangle Entries");
                await writer.WriteLineAsync("| File | Index | Vertices |");
                await writer.WriteLineAsync("|------|-------|----------|");
                
                // Limit to MAX_ITEMS_PER_CATEGORY triangles for readability
                var displayTriangles = triangles.OrderBy(t => t.FileName).ThenBy(t => t.Index).Take(MAX_ITEMS_PER_CATEGORY).ToList();
                foreach (var triangle in displayTriangles)
                {
                    await writer.WriteLineAsync($"| {triangle.FileName} | {triangle.Index} | ({triangle.V1}, {triangle.V2}, {triangle.V3}) |");
                }
                
                if (triangles.Count > MAX_ITEMS_PER_CATEGORY)
                {
                    await writer.WriteLineAsync($"*...and {triangles.Count - MAX_ITEMS_PER_CATEGORY} more triangles...*");
                }
                
                await writer.WriteLineAsync();
            }
            
            // Clear list to free memory
            triangles.Clear();
        }
        
        /// <summary>
        /// Load only the metadata about special values from positions.csv
        /// </summary>
        private async Task LoadSpecialValueMetadataAsync(string filePath)
        {
            if (!File.Exists(filePath))
            {
                _logger.LogWarning("Positions file not found: {FilePath}", filePath);
                return;
            }
            
            _logger.LogInformation("Loading special value metadata from {FilePath}", filePath);
            
            using var reader = new StreamReader(new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite));
            
            // Skip header
            await reader.ReadLineAsync();
            
            string? line;
            int specialCount = 0;
            int normalCount = 0;
            
            while ((line = await reader.ReadLineAsync()) != null)
            {
                var parts = line.Split(',');
                if (parts.Length < 8) continue;
                
                var fileName = parts[0];
                
                if (!int.TryParse(parts[1], out var index)) continue;
                var type = parts[2];
                
                var isSpecial = type == "Special";
                
                if (isSpecial && uint.TryParse(parts[7], out var specialValue))
                {
                    specialCount++;
                    
                    // Add to file-to-specialValues mapping
                    if (!_fileSpecialValues.TryGetValue(fileName, out var fileValues))
                    {
                        fileValues = new HashSet<uint>();
                        _fileSpecialValues[fileName] = fileValues;
                    }
                    
                    fileValues.Add(specialValue);
                    
                    // Add to specialValue-to-files mapping
                    if (!_filesBySpecialValue.TryGetValue(specialValue, out var valueFiles))
                    {
                        valueFiles = new List<string>();
                        _filesBySpecialValue[specialValue] = valueFiles;
                    }
                    
                    if (!valueFiles.Contains(fileName))
                    {
                        valueFiles.Add(fileName);
                    }
                    
                    // Add/update metadata
                    if (!_specialValueMetadata.TryGetValue(specialValue, out var metadata))
                    {
                        metadata = new SpecialValueMetadata();
                        _specialValueMetadata[specialValue] = metadata;
                    }
                    
                    metadata.PositionCount++;
                }
                else if (!isSpecial && index > 0)
                {
                    normalCount++;
                    
                    // For normal entries, we need to find the associated special value
                    // In the positions.csv file, normal entries follow their special entries
                    
                    // Calculate the special entry index (usually index - 1)
                    var specialIndex = index - 1;
                    
                    // Check if there's a special value from this file at the previous index
                    if (_fileSpecialValues.TryGetValue(fileName, out var specialValues) && 
                        specialValues.Count > 0)
                    {
                        // This isn't perfect as we can't directly know which special value this normal entry belongs to
                        // In a proper implementation, we'd need to track which special value each normal entry is associated with
                        
                        // For this simplified version, we'll just increment position count for all special values in this file
                        // This is a compromise for memory efficiency vs. accuracy
                        foreach (var sv in specialValues)
                        {
                            if (_specialValueMetadata.TryGetValue(sv, out var metadata))
                            {
                                metadata.PositionCount++;
                            }
                        }
                    }
                }
                
                // Periodically log progress and free memory
                if ((specialCount + normalCount) % 1000000 == 0)
                {
                    _logger.LogInformation("Processed {Count} entries ({Special} special, {Normal} normal)", 
                        specialCount + normalCount, specialCount, normalCount);
                    GC.Collect();
                }
            }
            
            // Now count other data types for each special value
            await CountDataTypesBySpecialValueAsync();
            
            _logger.LogInformation("Loaded metadata for {Count} unique special values from {Special} special entries and {Normal} normal entries", 
                _specialValueMetadata.Count, specialCount, normalCount);
        }
        
        /// <summary>
        /// Count the number of vertices, normals, and triangles for each special value
        /// </summary>
        private async Task CountDataTypesBySpecialValueAsync()
        {
            await CountVerticesAsync();
            await CountNormalsAsync();
            await CountTrianglesAsync();
        }
        
        private async Task CountVerticesAsync()
        {
            var verticesPath = Path.Combine(_csvDirectory, "vertices.csv");
            if (!File.Exists(verticesPath))
            {
                _logger.LogWarning("Vertices file not found: {FilePath}", verticesPath);
                return;
            }
            
            using var reader = new StreamReader(new FileStream(verticesPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite));
            
            // Skip header
            await reader.ReadLineAsync();
            
            string? line;
            int lineCount = 0;
            
            while ((line = await reader.ReadLineAsync()) != null)
            {
                lineCount++;
                
                var parts = line.Split(',');
                if (parts.Length < 2) continue;
                
                var fileName = parts[0];
                
                // For each special value in this file, increment vertex count
                if (_fileSpecialValues.TryGetValue(fileName, out var specialValues))
                {
                    foreach (var sv in specialValues)
                    {
                        if (_specialValueMetadata.TryGetValue(sv, out var metadata))
                        {
                            metadata.VertexCount++;
                        }
                    }
                }
                
                // Log progress periodically
                if (lineCount % 1000000 == 0)
                {
                    _logger.LogInformation("Processed {Count} vertex entries", lineCount);
                    GC.Collect();
                }
            }
        }
        
        private async Task CountNormalsAsync()
        {
            var normalsPath = Path.Combine(_csvDirectory, "normals.csv");
            if (!File.Exists(normalsPath))
            {
                _logger.LogWarning("Normals file not found: {FilePath}", normalsPath);
                return;
            }
            
            using var reader = new StreamReader(new FileStream(normalsPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite));
            
            // Skip header
            await reader.ReadLineAsync();
            
            string? line;
            int lineCount = 0;
            
            while ((line = await reader.ReadLineAsync()) != null)
            {
                lineCount++;
                
                var parts = line.Split(',');
                if (parts.Length < 2) continue;
                
                var fileName = parts[0];
                
                // For each special value in this file, increment normal count
                if (_fileSpecialValues.TryGetValue(fileName, out var specialValues))
                {
                    foreach (var sv in specialValues)
                    {
                        if (_specialValueMetadata.TryGetValue(sv, out var metadata))
                        {
                            metadata.NormalCount++;
                        }
                    }
                }
                
                // Log progress periodically
                if (lineCount % 1000000 == 0)
                {
                    _logger.LogInformation("Processed {Count} normal entries", lineCount);
                    GC.Collect();
                }
            }
        }
        
        private async Task CountTrianglesAsync()
        {
            var trianglesPath = Path.Combine(_csvDirectory, "triangles.csv");
            if (!File.Exists(trianglesPath))
            {
                _logger.LogWarning("Triangles file not found: {FilePath}", trianglesPath);
                return;
            }
            
            using var reader = new StreamReader(new FileStream(trianglesPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite));
            
            // Skip header
            await reader.ReadLineAsync();
            
            string? line;
            int lineCount = 0;
            
            while ((line = await reader.ReadLineAsync()) != null)
            {
                lineCount++;
                
                var parts = line.Split(',');
                if (parts.Length < 2) continue;
                
                var fileName = parts[0];
                
                // For each special value in this file, increment triangle count
                if (_fileSpecialValues.TryGetValue(fileName, out var specialValues))
                {
                    foreach (var sv in specialValues)
                    {
                        if (_specialValueMetadata.TryGetValue(sv, out var metadata))
                        {
                            metadata.TriangleCount++;
                        }
                    }
                }
                
                // Log progress periodically
                if (lineCount % 1000000 == 0)
                {
                    _logger.LogInformation("Processed {Count} triangle entries", lineCount);
                    GC.Collect();
                }
            }
        }
        
        #region Data Classes
        
        /// <summary>
        /// Metadata about a special value, used to reduce memory usage
        /// </summary>
        private class SpecialValueMetadata
        {
            public int PositionCount { get; set; }
            public int VertexCount { get; set; }
            public int NormalCount { get; set; }
            public int TriangleCount { get; set; }
        }
        
        /// <summary>
        /// Represents a position entry from the positions.csv file.
        /// </summary>
        private class PositionData
        {
            public string FileName { get; set; } = string.Empty;
            public int Index { get; set; }
            public bool IsSpecial { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }
        }
        
        /// <summary>
        /// Represents a vertex entry from the vertices.csv file.
        /// </summary>
        private class VertexData
        {
            public string FileName { get; set; } = string.Empty;
            public int Index { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }
        }
        
        /// <summary>
        /// Represents a normal entry from the normals.csv file.
        /// </summary>
        private class NormalData
        {
            public string FileName { get; set; } = string.Empty;
            public int Index { get; set; }
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }
        }
        
        /// <summary>
        /// Represents a triangle entry from the triangles.csv file.
        /// </summary>
        private class TriangleData
        {
            public string FileName { get; set; } = string.Empty;
            public int Index { get; set; }
            public uint V1 { get; set; }
            public uint V2 { get; set; }
            public uint V3 { get; set; }
        }
        
        #endregion
    }
} 