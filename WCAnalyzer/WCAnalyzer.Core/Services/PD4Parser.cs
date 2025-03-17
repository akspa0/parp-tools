using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Warcraft.NET.Extensions;
using WCAnalyzer.Core.Models.PD4;
using System.Linq;
using System.Text;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for parsing PD4 files and generating analysis results.
    /// </summary>
    public class PD4Parser
    {
        private readonly ILogger<PD4Parser>? _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PD4Parser"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        public PD4Parser(ILogger<PD4Parser>? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Parses a PD4 file and returns an analysis result.
        /// </summary>
        /// <param name="filePath">Path to the PD4 file.</param>
        /// <returns>A PD4AnalysisResult containing the analysis data.</returns>
        public PD4AnalysisResult ParseFile(string filePath)
        {
            try
            {
                _logger?.LogInformation("Starting to parse PD4 file: {FilePath}", filePath);

                if (!File.Exists(filePath))
                {
                    _logger?.LogError("PD4 file not found: {FilePath}", filePath);
                    var errorResult = new PD4AnalysisResult
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
                
                var result = new PD4AnalysisResult
                {
                    FileName = Path.GetFileName(filePath),
                    FilePath = filePath,
                    Errors = new List<string>(),
                    PD4Data = ParsePD4Data(reader)
                };
                
                _logger?.LogInformation("Successfully parsed PD4 file: {FilePath}", filePath);
                
                return result;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PD4 file: {FilePath}", filePath);
                var errorResult = new PD4AnalysisResult
                {
                    FileName = Path.GetFileName(filePath),
                    FilePath = filePath,
                    Errors = new List<string> { $"Error parsing PD4 file: {ex.Message}" },
                    Success = false
                };
                return errorResult;
            }
        }
        
        /// <summary>
        /// Parses the PD4 file data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        /// <returns>A PD4Data object containing the parsed data.</returns>
        private PD4Data ParsePD4Data(BinaryReader reader)
        {
            var data = new PD4Data();
            
            try
            {
                // Read the file header (MVER chunk) - note that all chunk names are reversed
                var magicBytes = reader.ReadBytes(4);
                Array.Reverse(magicBytes);
                var magic = Encoding.ASCII.GetString(magicBytes);
                
                if (magic != "MVER")
                {
                    _logger?.LogError("Invalid PD4 file format. Expected MVER (reversed), got {Magic}", magic);
                    data.Errors.Add($"Invalid PD4 file format. Expected MVER, got {magic}");
                    return data;
                }
                
                var mverSize = reader.ReadInt32();
                var mverVersion = reader.ReadInt32();
                data.Version = mverVersion;
                
                // Read each chunk until end of file
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    try
                    {
                        // Read the chunk name and reverse it since all chunk names are reversed
                        var chunkNameBytes = reader.ReadBytes(4);
                        Array.Reverse(chunkNameBytes);
                        var chunkName = Encoding.ASCII.GetString(chunkNameBytes);
                        
                        var chunkSize = reader.ReadInt32();
                        var chunkStartPosition = reader.BaseStream.Position;
                        
                        // Process based on chunk type
                        switch (chunkName)
                        {
                            case "MPVD":
                                data.VertexPositions = ReadVertexPositions(reader, chunkSize);
                                break;
                                
                            case "MVIX":
                                data.VertexIndices = ReadVertexIndices(reader, chunkSize);
                                break;
                                
                            case "MTEX":
                                data.TextureNames = ReadTextureNames(reader, chunkSize);
                                break;
                                
                            case "MMTX":
                                data.MaterialData = ReadMaterialData(reader, chunkSize);
                                break;
                                
                            default:
                                _logger?.LogDebug("Skipping unknown chunk type: {ChunkType}", chunkName);
                                break;
                        }
                        
                        // Skip to the end of the chunk if needed
                        if (reader.BaseStream.Position < chunkStartPosition + chunkSize)
                        {
                            reader.BaseStream.Seek(chunkStartPosition + chunkSize, SeekOrigin.Begin);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, "Error reading chunk at position {Position}", reader.BaseStream.Position);
                        data.Errors.Add($"Error reading chunk at position {reader.BaseStream.Position}: {ex.Message}");
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PD4 data");
                data.Errors.Add($"Error parsing PD4 data: {ex.Message}");
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
        /// Reads texture names from the binary reader.
        /// </summary>
        private List<string> ReadTextureNames(BinaryReader reader, int chunkSize)
        {
            var startPosition = reader.BaseStream.Position;
            var endPosition = startPosition + chunkSize;
            var textureNames = new List<string>();
            
            while (reader.BaseStream.Position < endPosition)
            {
                var nameBytes = new List<byte>();
                byte b;
                while ((b = reader.ReadByte()) != 0 && reader.BaseStream.Position < endPosition)
                {
                    nameBytes.Add(b);
                }
                
                if (nameBytes.Count > 0)
                {
                    textureNames.Add(Encoding.ASCII.GetString(nameBytes.ToArray()));
                }
            }
            
            return textureNames;
        }
        
        /// <summary>
        /// Reads material data from the binary reader.
        /// </summary>
        private List<MaterialData> ReadMaterialData(BinaryReader reader, int chunkSize)
        {
            var materialCount = chunkSize / 16; // Each material entry is 16 bytes
            var materials = new List<MaterialData>(materialCount);
            
            for (int i = 0; i < materialCount; i++)
            {
                materials.Add(new MaterialData
                {
                    TextureIndex = reader.ReadInt32(),
                    Flags = reader.ReadInt32(),
                    Value1 = reader.ReadInt32(),
                    Value2 = reader.ReadInt32()
                });
            }
            
            return materials;
        }
        
        /// <summary>
        /// Parses multiple PD4 files asynchronously and returns analysis results.
        /// </summary>
        /// <param name="filePaths">List of paths to PD4 files.</param>
        /// <returns>A list of PD4AnalysisResult containing the analysis data for each file.</returns>
        public async Task<List<PD4AnalysisResult>> ParseFilesAsync(IEnumerable<string> filePaths)
        {
            var results = new List<PD4AnalysisResult>();
            
            foreach (var filePath in filePaths)
            {
                // Use Task.Run to offload CPU-bound work to a thread pool thread
                var result = await Task.Run(() => ParseFile(filePath));
                results.Add(result);
            }
            
            _logger?.LogInformation("Completed parsing {Count} PD4 files", results.Count);
            return results;
        }

        /// <summary>
        /// Parses all PD4 files in a directory and returns analysis results.
        /// </summary>
        /// <param name="directoryPath">Path to the directory containing PD4 files.</param>
        /// <param name="searchPattern">File search pattern (default: *.pd4).</param>
        /// <param name="searchOption">Directory search options (default: AllDirectories).</param>
        /// <returns>A list of PD4AnalysisResult containing the analysis data for each file.</returns>
        public async Task<List<PD4AnalysisResult>> ParseDirectoryAsync(
            string directoryPath, 
            string searchPattern = "*.pd4", 
            SearchOption searchOption = SearchOption.AllDirectories)
        {
            try
            {
                _logger?.LogInformation("Searching for PD4 files in directory: {DirectoryPath}", directoryPath);
                
                if (!Directory.Exists(directoryPath))
                {
                    _logger?.LogError("Directory not found: {DirectoryPath}", directoryPath);
                    throw new DirectoryNotFoundException($"Directory not found: {directoryPath}");
                }

                var filePaths = Directory.GetFiles(directoryPath, searchPattern, searchOption);
                _logger?.LogInformation("Found {Count} PD4 files in directory", filePaths.Length);

                return await ParseFilesAsync(filePaths);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error parsing PD4 files in directory: {DirectoryPath}", directoryPath);
                throw;
            }
        }
    }
} 