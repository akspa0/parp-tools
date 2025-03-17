using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Warcraft.NET.Extensions;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;
using WCAnalyzer.Core.Services;
using System.Linq;
using System.Text;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for parsing PM4 files and generating analysis results.
    /// </summary>
    public class PM4Parser
    {
        private readonly ILogger<PM4Parser> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4Parser"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        public PM4Parser(ILogger<PM4Parser>? logger = null)
        {
            _logger = logger ?? NullLogger<PM4Parser>.Instance;
        }

        /// <summary>
        /// Parses a PM4 file and returns an analysis result.
        /// </summary>
        /// <param name="filePath">Path to the PM4 file.</param>
        /// <returns>A PM4AnalysisResult containing the analysis data.</returns>
        public PM4AnalysisResult ParseFile(string filePath)
        {
            try
            {
                _logger?.LogInformation("Starting to parse PM4 file: {FilePath}", filePath);

                if (!File.Exists(filePath))
                {
                    _logger?.LogError("PM4 file not found: {FilePath}", filePath);
                    var errorResult = new PM4AnalysisResult
                    {
                        FileName = Path.GetFileName(filePath),
                        FilePath = filePath,
                        Errors = new List<string> { $"File not found: {filePath}" },
                        Success = false
                    };
                    return errorResult;
                }

                // Read the file as a binary stream
                using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                using var reader = new BinaryReader(stream);
                
                var pm4Data = new PM4Data();
                
                try
                {
                    pm4Data = ParsePM4Data(reader);
                }
                catch (Exception ex)
                {
                    _logger?.LogError(ex, "Error parsing PM4 data: {Message}", ex.Message);
                    pm4Data.Errors.Add(ex.Message);
                }
                
                var result = new PM4AnalysisResult
                {
                    FileName = Path.GetFileName(filePath),
                    FilePath = filePath,
                    Errors = pm4Data.Errors,
                    PM4Data = pm4Data,
                    Success = !pm4Data.Errors.Any()
                };
                
                _logger?.LogInformation("Successfully parsed PM4 file: {FilePath}", filePath);
                
                return result;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PM4 file: {FilePath}", filePath);
                var errorResult = new PM4AnalysisResult
                {
                    FileName = Path.GetFileName(filePath),
                    FilePath = filePath,
                    Errors = new List<string> { $"Error parsing PM4 file: {ex.Message}" },
                    Success = false
                };
                return errorResult;
            }
        }
        
        /// <summary>
        /// Parses the PM4 file data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        /// <returns>A dictionary of chunk data.</returns>
        private PM4Data ParsePM4Data(BinaryReader reader)
        {
            var data = new PM4Data();
            data.UnknownChunks = new List<UnknownChunk>();
            
            try
            {
                // Read the file header (MVER chunk) - note that all chunk names are reversed
                var magicBytes = reader.ReadBytes(4);
                Array.Reverse(magicBytes);
                var magic = Encoding.ASCII.GetString(magicBytes);
                
                if (magic != "MVER")
                {
                    _logger?.LogError("Invalid PM4 file format. Expected MVER (reversed), got {Magic}", magic);
                    throw new InvalidDataException($"Invalid PM4 file format. Expected MVER, got {magic}");
                }
                
                // Read the rest of the MVER chunk
                var mverSize = reader.ReadInt32();
                var version = reader.ReadInt32();
                
                _logger?.LogDebug("PM4 version: {Version}", version);
                data.Version = version;
                
                // Process chunks until the end of the file
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    // Read the chunk name and reverse it since all chunk names are reversed
                    var chunkNameBytes = reader.ReadBytes(4);
                    var originalChunkName = Encoding.ASCII.GetString(chunkNameBytes);
                    Array.Reverse(chunkNameBytes);
                    var chunkName = Encoding.ASCII.GetString(chunkNameBytes);
                    
                    var chunkSize = reader.ReadInt32();
                    var chunkPosition = reader.BaseStream.Position;
                    
                    _logger?.LogInformation("Found chunk: {ChunkName} (original: {OriginalChunkName}), Size: {ChunkSize} bytes at position {Position}", 
                        chunkName, originalChunkName, chunkSize, chunkPosition);
                    
                    // Process the chunk based on its type
                    bool chunkProcessed = false;
                    switch (chunkName)
                    {
                        case "MSPV": // MS Vertex Positions
                            data.VertexPositions = ReadVertexPositions(reader, chunkSize);
                            chunkProcessed = true;
                            break;
                        case "MSPI": // MS Vertex Indices
                            data.VertexIndices = ReadVertexIndices(reader, chunkSize);
                            chunkProcessed = true;
                            break;
                        case "MSVI": // MS Vertex Indices for MSVT vertices (interior models)
                            data.MsviIndices = ReadMsviIndices(reader, chunkSize);
                            chunkProcessed = true;
                            break;
                        case "MSLK": // MS Links
                            data.Links = ReadLinkData(reader, chunkSize);
                            chunkProcessed = true;
                            break;
                        case "MSVT": // MS Vertex Data
                            data.VertexData = ReadVertexData(reader, chunkSize);
                            chunkProcessed = true;
                            break;
                        case "MPRL": // MP Position Data
                            data.Positions = ReadPositionData(reader, chunkSize);
                            chunkProcessed = true;
                            break;
                        case "MPRR": // MP Position References
                            data.PositionReferences = ReadPositionReferences(reader, chunkSize);
                            chunkProcessed = true;
                            break;
                        // Add the other implemented chunks
                        case "MSHD": // MS Header
                        case "MSCN": // MS Connections
                        case "MSUR": // MS Unknown References
                        case "MDBH": // Database Header
                        case "MDBI": // Database IDs
                        case "MDBF": // Database Filenames
                        case "MDOS": // Database Object Set
                        case "MDSF": // Database Something File
                            // TODO: Implement parsing for these chunks when needed
                            _logger?.LogInformation("Recognized chunk {ChunkName} but parser not yet implemented", chunkName);
                            reader.BaseStream.Position += chunkSize;
                            chunkProcessed = true;
                            break;
                    }
                    
                    if (!chunkProcessed)
                    {
                        // For unknown chunks, store basic info and first bytes for debugging
                        var unknownChunk = new UnknownChunk
                        {
                            ChunkName = chunkName,
                            OriginalChunkName = originalChunkName,
                            Size = chunkSize,
                            Position = chunkPosition
                        };
                        
                        // Read first 16 bytes of data as hex (or less if chunk is smaller)
                        int bytesToRead = Math.Min(16, chunkSize);
                        if (bytesToRead > 0)
                        {
                            var previewBytes = reader.ReadBytes(bytesToRead);
                            unknownChunk.HexPreview = BitConverter.ToString(previewBytes).Replace("-", " ");
                            
                            // Reset position to read the rest or skip
                            reader.BaseStream.Position = chunkPosition;
                        }
                        
                        data.UnknownChunks.Add(unknownChunk);
                        
                        // Skip unknown chunks
                        _logger?.LogDebug("Skipping unknown chunk: {ChunkName}", chunkName);
                        reader.BaseStream.Position += chunkSize;
                    }
                    
                    // Ensure we're at the right position after reading the chunk
                    if (reader.BaseStream.Position != chunkPosition + chunkSize)
                    {
                        _logger?.LogWarning("Incorrect chunk read position for {ChunkName}, adjusting", chunkName);
                        reader.BaseStream.Position = chunkPosition + chunkSize;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PM4 data");
                data.Errors.Add($"Error parsing PM4 data: {ex.Message}");
            }
            
            return data;
        }
        
        /// <summary>
        /// Reads vertex position data from the binary reader.
        /// </summary>
        private List<Vector3> ReadVertexPositions(BinaryReader reader, int chunkSize)
        {
            var vertexCount = chunkSize / 12; // 3 floats (x,y,z) * 4 bytes each
            var positions = new List<Vector3>(vertexCount);
            
            for (int i = 0; i < vertexCount; i++)
            {
                positions.Add(new Vector3(
                    reader.ReadSingle(),
                    reader.ReadSingle(),
                    reader.ReadSingle()
                ));
            }
            
            return positions;
        }
        
        /// <summary>
        /// Reads vertex index data from the binary reader.
        /// </summary>
        private List<int> ReadVertexIndices(BinaryReader reader, int chunkSize)
        {
            var indexCount = chunkSize / 2; // 2 bytes per index (16-bit)
            var indices = new List<int>(indexCount);
            
            for (int i = 0; i < indexCount; i++)
            {
                indices.Add(reader.ReadUInt16());
            }
            
            return indices;
        }
        
        /// <summary>
        /// Reads MSVI indices from the binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        /// <param name="chunkSize">The size of the chunk in bytes.</param>
        /// <returns>A list of MSVI indices.</returns>
        private List<int> ReadMsviIndices(BinaryReader reader, int chunkSize)
        {
            var indices = new List<int>();
            
            // MSVI chunk likely contains uint32 indices that reference into MSVT vertices
            var entryCount = chunkSize / 4; // 4 bytes per index (32-bit)
            
            _logger?.LogInformation("Reading {Count} MSVI indices for interior model faces", entryCount);
            
            for (int i = 0; i < entryCount; i++)
            {
                indices.Add((int)reader.ReadUInt32());
            }
            
            return indices;
        }
        
        /// <summary>
        /// Reads link data from the binary reader.
        /// </summary>
        private List<LinkData> ReadLinkData(BinaryReader reader, int chunkSize)
        {
            var linkCount = chunkSize / 8; // 2 integers per link
            var links = new List<LinkData>(linkCount);
            
            for (int i = 0; i < linkCount; i++)
            {
                var linkData = new LinkData
                {
                    Value0x00 = reader.ReadInt32(),
                    Value0x04 = reader.ReadInt32(),
                    Value0x08 = 0,
                    Value0x0C = 0
                };
                
                // Set the simplified properties
                linkData.SourceIndex = linkData.Value0x00;
                linkData.TargetIndex = linkData.Value0x04;
                
                links.Add(linkData);
            }
            
            return links;
        }
        
        /// <summary>
        /// Reads position data from the binary reader.
        /// </summary>
        private List<PositionData> ReadPositionData(BinaryReader reader, int chunkSize)
        {
            var entrySize = 20; // Each position data entry is 20 bytes
            var entryCount = chunkSize / entrySize;
            var positions = new List<PositionData>(entryCount);
            
            for (int i = 0; i < entryCount; i++)
            {
                positions.Add(new PositionData
                {
                    Value0x00 = reader.ReadInt32(),
                    Value0x04 = reader.ReadInt32(),
                    Value0x08 = reader.ReadSingle(),
                    Value0x0C = reader.ReadSingle(),
                    Value0x10 = reader.ReadInt32()
                });
            }
            
            return positions;
        }
        
        /// <summary>
        /// Reads position references from the binary reader.
        /// </summary>
        private List<PositionReference> ReadPositionReferences(BinaryReader reader, int chunkSize)
        {
            var entrySize = 4; // Each position reference entry is 4 bytes (2 uint16 values)
            var entryCount = chunkSize / entrySize;
            var references = new List<PositionReference>(entryCount);
            
            for (int i = 0; i < entryCount; i++)
            {
                var value1 = reader.ReadUInt16();
                var value2 = reader.ReadUInt16();
                
                var reference = new PositionReference
                {
                    Value0x00 = value1, // Store the uint16 in the int property
                    Value0x04 = value2, // Store the uint16 in the int property
                    Value0x08 = 0,
                    Value0x0C = 0,
                    Value1 = value1,
                    Value2 = value2
                };
                
                references.Add(reference);
            }
            
            return references;
        }

        /// <summary>
        /// Reads vertex data from the binary reader for MSVT chunk.
        /// The data is ordered YXZ as per documentation and requires specific transformations.
        /// </summary>
        private List<VertexData> ReadVertexData(BinaryReader reader, int chunkSize)
        {
            // Constants for coordinate transformations as per documentation
            const float CoordinateOffset = 17066.666f;
            const float HeightConversion = 36.0f; // Convert internal inch height to yards
            
            var entrySize = 12; // 3 floats x 4 bytes each
            var entryCount = chunkSize / entrySize;
            var vertices = new List<VertexData>(entryCount);
            
            for (int i = 0; i < entryCount; i++)
            {
                // Read the raw values in YXZ order as per documentation
                float rawY = reader.ReadSingle();
                float rawX = reader.ReadSingle();
                float rawZ = reader.ReadSingle();
                
                // Apply the transformation formulas from the documentation
                float worldY = CoordinateOffset - rawY;
                float worldX = CoordinateOffset - rawX;
                float worldZ = rawZ / HeightConversion;
                
                var vertex = new VertexData
                {
                    // Store both raw and transformed values
                    RawY = rawY,
                    RawX = rawX,
                    RawZ = rawZ,
                    
                    // Store transformed world coordinates
                    WorldX = worldX,
                    WorldY = worldY,
                    WorldZ = worldZ
                };
                
                vertices.Add(vertex);
            }
            
            return vertices;
        }
        
        /// <summary>
        /// Parses multiple PM4 files asynchronously and returns analysis results.
        /// </summary>
        /// <param name="filePaths">List of paths to PM4 files.</param>
        /// <returns>A list of PM4AnalysisResult containing the analysis data for each file.</returns>
        public async Task<List<PM4AnalysisResult>> ParseFilesAsync(IEnumerable<string> filePaths)
        {
            var results = new List<PM4AnalysisResult>();
            
            foreach (var filePath in filePaths)
            {
                // Use Task.Run to offload CPU-bound work to a thread pool thread
                var result = await Task.Run(() => ParseFile(filePath));
                results.Add(result);
            }
            
            _logger?.LogInformation("Completed parsing {Count} PM4 files", results.Count);
            return results;
        }

        /// <summary>
        /// Parses all PM4 files in a directory and returns analysis results.
        /// </summary>
        /// <param name="directoryPath">Path to the directory containing PM4 files.</param>
        /// <param name="searchPattern">File search pattern (default: *.pm4).</param>
        /// <param name="searchOption">Directory search options (default: AllDirectories).</param>
        /// <returns>A list of PM4AnalysisResult containing the analysis data for each file.</returns>
        public async Task<List<PM4AnalysisResult>> ParseDirectoryAsync(
            string directoryPath, 
            string searchPattern = "*.pm4", 
            SearchOption searchOption = SearchOption.AllDirectories)
        {
            try
            {
                _logger?.LogInformation("Searching for PM4 files in directory: {DirectoryPath}", directoryPath);
                
                if (!Directory.Exists(directoryPath))
                {
                    _logger?.LogError("Directory not found: {DirectoryPath}", directoryPath);
                    throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");
                }

                var filePaths = Directory.GetFiles(directoryPath, searchPattern, searchOption);
                _logger?.LogInformation("Found {Count} PM4 files in directory", filePaths.Length);

                return await ParseFilesAsync(filePaths);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PM4 files in directory: {DirectoryPath}", directoryPath);
                throw;
            }
        }

        /// <summary>
        /// Combines multiple PM4 analysis results into a single summary.
        /// </summary>
        /// <param name="results">The list of PM4 analysis results to combine.</param>
        /// <returns>A string containing the combined summary.</returns>
        public string GenerateSummary(IEnumerable<PM4AnalysisResult> results)
        {
            var summary = new System.Text.StringBuilder();
            var resultsList = results as List<PM4AnalysisResult> ?? new List<PM4AnalysisResult>(results);
            
            summary.AppendLine($"PM4 Analysis Summary");
            summary.AppendLine($"===================");
            summary.AppendLine($"Files Analyzed: {resultsList.Count}");
            
            int filesWithErrors = resultsList.Count(r => r.Errors.Count > 0);
            summary.AppendLine($"Files with Errors: {filesWithErrors}");
            
            summary.AppendLine("\nChunk Statistics:");
            summary.AppendLine($"- Files with Shadow Data: {resultsList.Count(r => r.HasShadowData)}");
            summary.AppendLine($"- Files with Vertex Positions: {resultsList.Count(r => r.HasVertexPositions)}");
            summary.AppendLine($"- Files with Vertex Indices: {resultsList.Count(r => r.HasVertexIndices)}");
            summary.AppendLine($"- Files with Normal Coordinates: {resultsList.Count(r => r.HasNormalCoordinates)}");
            summary.AppendLine($"- Files with Links: {resultsList.Count(r => r.HasLinks)}");
            summary.AppendLine($"- Files with Vertex Data: {resultsList.Count(r => r.HasVertexData)}");
            summary.AppendLine($"- Files with Vertex Info: {resultsList.Count(r => r.HasVertexInfo)}");
            summary.AppendLine($"- Files with Surface Data: {resultsList.Count(r => r.HasSurfaceData)}");
            summary.AppendLine($"- Files with Position Data: {resultsList.Count(r => r.HasPositionData)}");
            summary.AppendLine($"- Files with Position Reference: {resultsList.Count(r => r.HasPositionReference)}");
            summary.AppendLine($"- Files with Destructible Building Header: {resultsList.Count(r => r.HasDestructibleBuildingHeader)}");
            summary.AppendLine($"- Files with Object Data: {resultsList.Count(r => r.HasObjectData)}");
            summary.AppendLine($"- Files with Server Flag Data: {resultsList.Count(r => r.HasServerFlagData)}");
            
            if (filesWithErrors > 0)
            {
                summary.AppendLine("\nFiles with Errors:");
                foreach (var result in resultsList.Where(r => r.Errors.Count > 0))
                {
                    summary.AppendLine($"- {result.FileName}:");
                    foreach (var error in result.Errors)
                    {
                        summary.AppendLine($"  - {error}");
                    }
                }
            }
            
            return summary.ToString();
        }

        /// <summary>
        /// Resolves file references in the PM4 analysis result using a listfile.
        /// </summary>
        /// <param name="result">The PM4 analysis result to enhance.</param>
        /// <param name="listfile">Dictionary mapping FileDataIDs to file names.</param>
        public void ResolveReferences(PM4AnalysisResult result, Dictionary<uint, string> listfile)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));
            
            if (listfile == null || listfile.Count == 0)
            {
                _logger?.LogInformation("No listfile provided for reference resolution");
                return;
            }
            
            // PM4 files don't actually contain model FileDataIDs to resolve
            // The structure previously assumed to have FileDataIDs actually contains vertex coordinates and flag data
            
            _logger?.LogInformation("PM4 files do not contain FileDataIDs to resolve - this is server-side collision mesh data");
            
            // The file contains server-side collision and navigation data, not model references
            result.ResolvedFileNames.Clear();
        }

        /// <summary>
        /// Asynchronously parses PM4 files from a directory and resolves references using a listfile.
        /// </summary>
        /// <param name="directoryPath">The directory containing PM4 files.</param>
        /// <param name="listfilePath">Path to the listfile for reference resolution.</param>
        /// <param name="searchPattern">File search pattern.</param>
        /// <param name="searchOption">Directory search options.</param>
        /// <returns>A list of PM4 analysis results.</returns>
        public async Task<List<PM4AnalysisResult>> ParseDirectoryWithReferencesAsync(
            string directoryPath,
            string listfilePath,
            string searchPattern = "*.pm4",
            SearchOption searchOption = SearchOption.AllDirectories)
        {
            // First load the listfile
            Dictionary<uint, string> listfile = new Dictionary<uint, string>();
            try
            {
                if (File.Exists(listfilePath))
                {
                    _logger?.LogInformation("Loading listfile from {ListfilePath}", listfilePath);
                    
                    var lines = await File.ReadAllLinesAsync(listfilePath);
                    foreach (var line in lines)
                    {
                        var parts = line.Split(';');
                        if (parts.Length >= 2 && uint.TryParse(parts[0], out uint fileId))
                        {
                            listfile[fileId] = parts[1];
                        }
                    }
                    
                    _logger?.LogInformation("Loaded {Count} entries from listfile", listfile.Count);
                }
                else
                {
                    _logger?.LogWarning("Listfile not found at {ListfilePath}", listfilePath);
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error loading listfile from {ListfilePath}", listfilePath);
            }

            // Parse the PM4 files
            var results = await ParseDirectoryAsync(directoryPath, searchPattern, searchOption);

            // Resolve references for each result
            if (listfile.Count > 0)
            {
                foreach (var result in results)
                {
                    ResolveReferences(result, listfile);
                }
            }

            return results;
        }
    }
} 