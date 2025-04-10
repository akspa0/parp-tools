using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Numerics;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for parsing ADT files using Warcraft.NET.
    /// </summary>
    public class AdtParser
    {
        private readonly ILogger<AdtParser> _logger;
        private static readonly Regex CoordinateRegex = new Regex(@"_(\d+)_(\d+)", RegexOptions.Compiled);

        /// <summary>
        /// Creates a new instance of the AdtParser class.
        /// </summary>
        /// <param name="logger">The logging service to use.</param>
        public AdtParser(ILogger<AdtParser> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Parses an ADT file.
        /// </summary>
        /// <param name="filePath">The path to the ADT file.</param>
        /// <returns>The ADT analysis result.</returns>
        public async Task<AdtAnalysisResult> ParseAsync(string filePath)
        {
            _logger.LogInformation("Parsing ADT file: {FilePath}", Path.GetFileName(filePath));

            try
            {
                byte[] fileData = await File.ReadAllBytesAsync(filePath);

                // Parse the ADT file manually
                var adtInfo = ParseAdtFile(fileData);
                adtInfo.FilePath = filePath;
                adtInfo.FileName = Path.GetFileName(filePath);

                // Convert to AdtAnalysisResult
                var result = ConvertToAdtAnalysisResult(adtInfo, filePath);
                
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing ADT file {FilePath}: {Message}", filePath, ex.Message);
                
                // Return a minimal result with error information
                var result = new AdtAnalysisResult
                {
                    FilePath = filePath,
                    FileName = Path.GetFileName(filePath),
                    Errors = new List<string> { ex.Message }
                };
                
                // Try to extract coordinates from filename
                ExtractCoordinatesFromFilename(result);
                
                return result;
            }
        }

        private AdtInfo ParseAdtFile(byte[] fileData)
        {
            // Create an AdtInfo object to store the parsing results
            var adtInfo = new AdtInfo();
            
            // Add a property to track whether the ADT file is using FileDataIDs
            bool isUsingFileDataId = false;
            
            // Check if this is a tex0 or obj0 file based on file size and initial bytes
            bool isSpecialFile = false;
            bool isMonolithicAdt = true; // Default assumption is that this is a standard monolithic ADT
            
            if (fileData.Length < 1024) // Small files are likely not standard ADT files
            {
                isSpecialFile = true;
                isMonolithicAdt = false;
                _logger.LogDebug("File appears to be a special/auxiliary ADT file (small size: {Size} bytes)", fileData.Length);
            }
            
            using (MemoryStream ms = new MemoryStream(fileData))
            using (BinaryReader br = new BinaryReader(ms))
            {
                // Check for reversed chunk IDs (common in some ADT files)
                byte[] firstFourBytes = br.ReadBytes(4);
                string firstToken = Encoding.ASCII.GetString(firstFourBytes);
                _logger.LogDebug("First 4 bytes of file: {FirstToken}", firstToken);

                bool reversedChunks = false;
                if (firstToken == "REVM") // Reversed "MVER"
                {
                    _logger.LogDebug("File has reversed chunk IDs, correcting...");
                    reversedChunks = true;
                    ms.Position = 0; // Go back to start
                }
                else if (firstToken != "MVER")
                {
                    // If it doesn't start with MVER or REVM, it's probably a special file
                    isSpecialFile = true;
                    isMonolithicAdt = false;
                    _logger.LogDebug("File does not start with standard ADT header (MVER)");
                }

                // Create a helper to read 4-byte chunk identifiers
                Func<string> readChunkId = () =>
                {
                    try
                    {
                        byte[] bytes = br.ReadBytes(4);
                        if (bytes.Length < 4)
                        {
                            _logger.LogDebug("Reached end of file while reading chunk ID");
                            return string.Empty;
                        }
                        
                        if (reversedChunks)
                        {
                            Array.Reverse(bytes);
                        }
                        return Encoding.ASCII.GetString(bytes);
                    }
                    catch (EndOfStreamException)
                    {
                        _logger.LogDebug("End of stream reached while trying to read chunk ID");
                        return string.Empty;
                    }
                };

                // If it's a special file, use a simplified parsing approach
                if (isSpecialFile)
                {
                    _logger.LogDebug("Using simplified parsing for special ADT file format");
                    
                    try
                    {
                        // Reset position to start
                        ms.Position = 0;
                        
                        // Just scan for known chunk IDs without expecting the regular ADT structure
                        while (ms.Position < ms.Length - 8)
                        {
                            long currentPos = ms.Position;
                            string possibleChunkId = readChunkId();
                            
                            if (string.IsNullOrEmpty(possibleChunkId))
                                break;
                            
                            // Try to read the chunk size
                            uint chunkSize;
                            try
                            {
                                chunkSize = br.ReadUInt32();
                            }
                            catch (EndOfStreamException)
                            {
                                break;
                            }
                            
                            // Process known chunks as we would in the regular parser
                            switch (possibleChunkId)
                            {
                                case "MTEX": // Texture names chunk
                                    _logger.LogDebug("Found MTEX chunk in special file, size: {Size}", chunkSize);
                                    long endPos = ms.Position + chunkSize;
                                    while (ms.Position < endPos)
                                    {
                                        try
                                        {
                                            string textureName = ReadNullTerminatedString(br);
                                            if (!string.IsNullOrWhiteSpace(textureName))
                                            {
                                                adtInfo.TextureNames.Add(textureName);
                                            }
                                        }
                                        catch (EndOfStreamException)
                                        {
                                            _logger.LogDebug("End of stream reached while reading texture names");
                                            break;
                                        }
                                        
                                        // Break if we've reached the end of the chunk
                                        if (ms.Position >= endPos)
                                            break;
                                    }
                                    _logger.LogDebug("Found {Count} texture names in special file", adtInfo.TextureNames.Count);
                                    break;
                                    
                                case "MMDX": // Model names chunk
                                    _logger.LogDebug("Found MMDX chunk in special file, size: {Size}", chunkSize);
                                    endPos = ms.Position + chunkSize;
                                    while (ms.Position < endPos)
                                    {
                                        try
                                        {
                                            string modelName = ReadNullTerminatedString(br);
                                            if (!string.IsNullOrWhiteSpace(modelName))
                                            {
                                                adtInfo.ModelNames.Add(modelName);
                                            }
                                        }
                                        catch (EndOfStreamException)
                                        {
                                            _logger.LogDebug("End of stream reached while reading model names");
                                            break;
                                        }
                                        
                                        // Break if we've reached the end of the chunk
                                        if (ms.Position >= endPos)
                                            break;
                                    }
                                    _logger.LogDebug("Found {Count} model names in special file", adtInfo.ModelNames.Count);
                                    break;
                                    
                                case "MODF": // WMO placement chunk (might exist in special files)
                                    try
                                    {
                                        var wmoPlacementCount = chunkSize / 64; // Each WMO placement is 64 bytes
                                        _logger.LogDebug("Found MODF chunk with {Count} WMO placements", wmoPlacementCount);
                                        
                                        adtInfo.WmoPlacements = (int)wmoPlacementCount;
                                        
                                        for (int i = 0; i < wmoPlacementCount; i++)
                                        {
                                            // Make sure we always process WMO placements regardless of file type
                                            // Parse WMO placement data here and add to adtInfo
                                            uint nameId = br.ReadUInt32();
                                            uint uniqueId = br.ReadUInt32();
                                            float pos_x = br.ReadSingle();
                                            float pos_y = br.ReadSingle();
                                            float pos_z = br.ReadSingle();
                                            float rot_x = br.ReadSingle();
                                            float rot_y = br.ReadSingle();
                                            float rot_z = br.ReadSingle();
                                            // Skip bounding box (6 floats) and flags (2 uints)
                                            ms.Position += 32;
                                            
                                            // Determine if we're using a FileDataID based on flags
                                            bool nameIdIsFileDataId = false; // Default assumption
                                            
                                            // Get the WMO name from index or store FileDataID
                                            string wmoName = string.Empty;
                                            if (!nameIdIsFileDataId && nameId < adtInfo.WmoNames.Count)
                                            {
                                                wmoName = adtInfo.WmoNames[(int)nameId];
                                            }
                                            else
                                            {
                                                wmoName = $"<unknown WMO {nameId}>";
                                            }
                                            
                                            // Store the placement information
                                            var wmoPlacement = new WmoPlacementInfo
                                            {
                                                NameId = nameId,
                                                UniqueId = (int)uniqueId,
                                                Position = new System.Numerics.Vector3(pos_x, pos_y, pos_z),
                                                Rotation = new System.Numerics.Vector3(rot_x, rot_y, rot_z),
                                                NameIdIsFileDataId = nameIdIsFileDataId
                                            };
                                            
                                            adtInfo.WmoPlacementDetails.Add(wmoPlacement);
                                            
                                            _logger.LogDebug("WMO Placement: {UniqueId}, Position: ({X}, {Y}, {Z}), NameIdIsFileDataId: {UsesFileDataId}",
                                                uniqueId, pos_x, pos_y, pos_z, nameIdIsFileDataId);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        _logger.LogWarning(ex, "Error parsing MODF chunk in special file");
                                        ms.Position += chunkSize;
                                    }
                                    break;
                                
                                case "MDDF": // Model placement chunk (might exist in special files)
                                    try
                                    {
                                        var modelPlacementCount = chunkSize / 36; // Each model placement is 36 bytes
                                        _logger.LogDebug("Found MDDF chunk with {Count} model placements", modelPlacementCount);
                                        
                                        adtInfo.ModelPlacements = (int)modelPlacementCount;
                                        
                                        for (int i = 0; i < modelPlacementCount; i++)
                                        {
                                            // Make sure we always process model placements regardless of file type
                                            // Parse model placement data here and add to adtInfo
                                            uint nameId = br.ReadUInt32();
                                            uint uniqueId = br.ReadUInt32();
                                            float pos_x = br.ReadSingle();
                                            float pos_y = br.ReadSingle();
                                            float pos_z = br.ReadSingle();
                                            float rot_x = br.ReadSingle();
                                            float rot_y = br.ReadSingle();
                                            float rot_z = br.ReadSingle();
                                            float scale = br.ReadSingle();
                                            // Skip flags (1 uint)
                                            ms.Position += 4;
                                            
                                            // Determine if we're using a FileDataID based on flags
                                            bool nameIdIsFileDataId = false; // Default assumption
                                            
                                            // Get the model name from index or store FileDataID
                                            string modelName = string.Empty;
                                            if (!nameIdIsFileDataId && nameId < adtInfo.ModelNames.Count)
                                            {
                                                modelName = adtInfo.ModelNames[(int)nameId];
                                            }
                                            else
                                            {
                                                modelName = $"<unknown Model {nameId}>";
                                            }
                                            
                                            // Store the placement information
                                            var modelPlacement = new ModelPlacementInfo
                                            {
                                                NameId = nameId,
                                                UniqueId = (int)uniqueId,
                                                Position = new System.Numerics.Vector3(pos_x, pos_y, pos_z),
                                                Rotation = new System.Numerics.Vector3(rot_x, rot_y, rot_z),
                                                Scale = scale,
                                                NameIdIsFileDataId = nameIdIsFileDataId
                                            };
                                            
                                            adtInfo.ModelPlacementDetails.Add(modelPlacement);
                                            
                                            _logger.LogDebug("Model Placement: {UniqueId}, Position: ({X}, {Y}, {Z}), Scale: {Scale}, NameIdIsFileDataId: {UsesFileDataId}",
                                                uniqueId, pos_x, pos_y, pos_z, scale, nameIdIsFileDataId);
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        _logger.LogWarning(ex, "Error parsing MDDF chunk in special file");
                                        ms.Position += chunkSize;
                                    }
                                    break;
                                
                                default:
                                    // Skip unrecognized chunks in special files
                                    _logger.LogDebug("Skipping unrecognized chunk {ChunkId} in special file, size: {Size}", possibleChunkId, chunkSize);
                                    try
                                    {
                                        ms.Position += chunkSize;
                                    }
                                    catch (IOException)
                                    {
                                        // If we can't seek, we've likely reached the end of the file
                                        _logger.LogDebug("Unable to seek {ChunkSize} bytes forward, likely reached end of file", chunkSize);
                                        break;
                                    }
                                    break;
                            }
                        }
                        
                        return adtInfo;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "Error during simplified parsing of special ADT file");
                        return adtInfo; // Return whatever we could parse
                    }
                }

                // For standard ADT files, continue with normal parsing
                // Log all chunk IDs found in the file for debugging
                _logger.LogDebug("Beginning chunk scan for debugging...");
                long originalPosition = ms.Position;
                
                try {
                    // Quick scan through file to identify all chunks
                    ms.Position = 0;
                    var chunkPositions = new Dictionary<string, List<long>>();
                    
                    while (ms.Position < ms.Length - 8) // Need at least 8 bytes for chunk header
                    {
                        long chunkStart = ms.Position;
                        string chunkId = readChunkId();
                        uint chunkSize = br.ReadUInt32();
                        
                        if (!chunkPositions.ContainsKey(chunkId))
                        {
                            chunkPositions[chunkId] = new List<long>();
                        }
                        chunkPositions[chunkId].Add(chunkStart);
                        
                        // Skip to next chunk
                        ms.Position = chunkStart + 8 + chunkSize;
                        
                        // Check for potential MODF or MWMO chunks specifically
                        if (chunkId == "MODF" || chunkId == "MWMO")
                        {
                            _logger.LogDebug("Found {ChunkId} chunk at position {Position} with size {Size}", 
                                chunkId, chunkStart, chunkSize);
                        }
                    }
                    
                    // Log summary of chunk types found
                    _logger.LogDebug("Chunk types found in file:");
                    foreach (var entry in chunkPositions)
                    {
                        _logger.LogDebug("  {ChunkId}: {Count} instances", entry.Key, entry.Value.Count);
                        
                        // Special handling for WMO-related chunks
                        if (entry.Key == "MWMO" || entry.Key == "MODF")
                        {
                            foreach (var position in entry.Value)
                            {
                                _logger.LogDebug("    Instance at position: {Position}", position);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error during chunk scanning for debugging");
                }
                
                // Reset position to continue with normal parsing
                ms.Position = originalPosition;
                _logger.LogDebug("Resuming normal parsing...");

                // Parse chunks
                while (ms.Position < ms.Length - 8) // Need at least 8 bytes for chunk header
                {
                    // Read chunk ID and size
                    string chunkId = readChunkId();
                    
                    // Exit the loop if we can't read a valid chunk ID
                    if (string.IsNullOrEmpty(chunkId))
                    {
                        _logger.LogDebug("Unable to read valid chunk ID, ending parsing");
                        break;
                    }
                    
                    // Try to read chunk size
                    uint chunkSize;
                    try
                    {
                        chunkSize = br.ReadUInt32();
                        
                        // Sanity check the chunk size
                        if (chunkSize > ms.Length - ms.Position)
                        {
                            _logger.LogWarning("Chunk {ChunkId} has invalid size {Size} larger than remaining file size {Remaining}", 
                                chunkId, chunkSize, ms.Length - ms.Position);
                            // Use the remaining file size as the chunk size, but be cautious
                            chunkSize = (uint)(ms.Length - ms.Position);
                        }
                    }
                    catch (EndOfStreamException)
                    {
                        _logger.LogDebug("End of stream reached while reading chunk size");
                        break;
                    }

                    // Process chunk based on ID
                    try
                    {
                        switch (chunkId)
                        {
                            case "MVER": // Version chunk
                                try
                                {
                                    adtInfo.Version = br.ReadInt32();
                                    _logger.LogDebug("Found MVER chunk, version: {Version}", adtInfo.Version);
                                }
                                catch (EndOfStreamException)
                                {
                                    _logger.LogWarning("Unable to read MVER chunk data");
                                    ms.Position += chunkSize; // Skip the rest of the chunk
                                }
                                break;
                                
                            case "MHDR": // Header chunk
                                try
                                {
                                    adtInfo.Flags = br.ReadUInt32();
                                    _logger.LogDebug("Found MHDR chunk, flags: {Flags}", adtInfo.Flags);
                                    // Skip the rest of the header
                                    ms.Position += chunkSize - 4;
                                }
                                catch (EndOfStreamException)
                                {
                                    _logger.LogWarning("Unable to read MHDR chunk data");
                                    ms.Position += chunkSize; // Skip the rest of the chunk
                                }
                                break;
                                
                            case "MCNK": // Terrain chunk
                                _logger.LogDebug("Found MCNK chunk, size: {Size}", chunkSize);
                                
                                // Read terrain chunk header
                                var terrainChunk = new TerrainChunkInfo();
                                
                                // Note: In WoW ADT files, the MCNK chunk size declared in the header
                                // typically only refers to the size of the header, not including its sub-chunks.
                                // The sub-chunks that follow are part of the MCNK's data but may extend
                                // beyond the declared size.
                                long mcnkHeaderEndPos = ms.Position + chunkSize;
                                
                                try
                                {
                                    // Read key data from terrain chunk header
                                    terrainChunk.AreaId = br.ReadInt32();
                                    
                                    // Add to list of area IDs
                                    if (!adtInfo.AreaIds.Contains(terrainChunk.AreaId))
                                    {
                                        adtInfo.AreaIds.Add(terrainChunk.AreaId);
                                    }
                                    
                                    // Skip to flags (12 bytes ahead)
                                    ms.Position += 12;
                                    
                                    // Read flags
                                    terrainChunk.Flags = br.ReadInt32();
                                    
                                    // Skip to position (4 bytes ahead)
                                    ms.Position += 4;
                                    
                                    try
                                    {
                                        // Read position
                                        terrainChunk.X = br.ReadSingle();
                                        terrainChunk.Y = br.ReadSingle();
                                        terrainChunk.Z = br.ReadSingle();
                                        
                                        // Skip to the end of the MCNK header
                                        ms.Position = mcnkHeaderEndPos;
                                        
                                        // Parse sub-chunks that follow the MCNK header
                                        // Continue reading until we find the next main chunk or reach the end of the file
                                        bool foundNextMainChunk = false;
                                        while (!foundNextMainChunk && ms.Position < ms.Length - 8)
                                        {
                                            long currentPos = ms.Position;
                                            string peekId = readChunkId();
                                            
                                            // If we've reached a new main chunk, break
                                            if (peekId == "MCNK" || peekId == "MMDX" || peekId == "MTEX" || 
                                                peekId == "MWMO" || peekId == "MDDF" || peekId == "MODF" ||
                                                peekId == "MVER" || peekId == "MHDR" || string.IsNullOrEmpty(peekId))
                                            {
                                                // This is a main chunk, go back and let the main loop handle it
                                                ms.Position = currentPos;
                                                foundNextMainChunk = true;
                                                break;
                                            }
                                            
                                            // Otherwise, read the sub-chunk size
                                            uint subChunkSize;
                                            try
                                            {
                                                subChunkSize = br.ReadUInt32();
                                                _logger.LogDebug("Processing {SubChunkId} sub-chunk, size: {Size}", 
                                                    peekId, subChunkSize);
                                            }
                                            catch (EndOfStreamException)
                                            {
                                                _logger.LogDebug("End of stream reached while reading sub-chunk size");
                                                break;
                                            }
                                            
                                            // Process the sub-chunk based on ID
                                            long subChunkEndPos = ms.Position + subChunkSize;
                                            
                                            switch (peekId)
                                            {
                                                case "MCVT": // Height map
                                                    _logger.LogDebug("Found MCVT sub-chunk, size: {Size}", subChunkSize);
                                                    try
                                                    {
                                                        // Heights are stored as a 9x9 grid of floats (145 values total)
                                                        terrainChunk.Heights = new float[145];
                                                        for (int i = 0; i < 145 && ms.Position < subChunkEndPos; i++)
                                                        {
                                                            terrainChunk.Heights[i] = br.ReadSingle();
                                                        }
                                                        _logger.LogDebug("Read {Count} height values", terrainChunk.Heights.Length);
                                                    }
                                                    catch (EndOfStreamException ex)
                                                    {
                                                        _logger.LogWarning(ex, "Error reading MCVT height data");
                                                    }
                                                    break;
                                                    
                                                case "MCNR": // Normal vectors
                                                    _logger.LogDebug("Found MCNR sub-chunk, size: {Size}", subChunkSize);
                                                    try
                                                    {
                                                        // Normals are stored as a 9x9 grid of 3-byte values (145 values total)
                                                        terrainChunk.Normals = new Vector3[145];
                                                        for (int i = 0; i < 145 && ms.Position < subChunkEndPos; i++)
                                                        {
                                                            // Normals are stored as signed bytes (-127 to 127) and need to be normalized
                                                            float nx = br.ReadSByte() / 127.0f;
                                                            float ny = br.ReadSByte() / 127.0f;
                                                            float nz = br.ReadSByte() / 127.0f;
                                                            terrainChunk.Normals[i] = new Vector3(nx, ny, nz);
                                                        }
                                                        _logger.LogDebug("Read {Count} normal vectors", terrainChunk.Normals.Length);
                                                    }
                                                    catch (EndOfStreamException ex)
                                                    {
                                                        _logger.LogWarning(ex, "Error reading MCNR normal data");
                                                    }
                                                    break;
                                                    
                                                case "MCLY": // Texture layers
                                                    _logger.LogDebug("Found MCLY sub-chunk, size: {Size}", subChunkSize);
                                                    try
                                                    {
                                                        // Each texture layer is 16 bytes
                                                        int layerCount = (int)(subChunkSize / 16);
                                                        terrainChunk.TextureLayers = new List<TextureLayerInfo>(layerCount);
                                                        
                                                        for (int i = 0; i < layerCount && ms.Position < subChunkEndPos; i++)
                                                        {
                                                            var layer = new TextureLayerInfo
                                                            {
                                                                TextureId = br.ReadUInt32(),
                                                                Flags = br.ReadUInt32(),
                                                                OffsetInMCAL = br.ReadUInt32(),
                                                                EffectId = br.ReadInt32()
                                                            };
                                                            terrainChunk.TextureLayers.Add(layer);
                                                        }
                                                        _logger.LogDebug("Read {Count} texture layers", terrainChunk.TextureLayers.Count);
                                                    }
                                                    catch (EndOfStreamException ex)
                                                    {
                                                        _logger.LogWarning(ex, "Error reading MCLY texture layer data");
                                                    }
                                                    break;
                                                    
                                                case "MCAL": // Alpha maps
                                                    _logger.LogDebug("Found MCAL sub-chunk, size: {Size}", subChunkSize);
                                                    try
                                                    {
                                                        // Store the alpha map data for later processing
                                                        terrainChunk.AlphaMapData = new byte[subChunkSize];
                                                        int bytesRead = br.Read(terrainChunk.AlphaMapData, 0, (int)subChunkSize);
                                                        _logger.LogDebug("Read {BytesRead} bytes of alpha map data", bytesRead);
                                                    }
                                                    catch (EndOfStreamException ex)
                                                    {
                                                        _logger.LogWarning(ex, "Error reading MCAL alpha map data");
                                                    }
                                                    break;
                                                    
                                                case "MCSH": // Shadow map
                                                    _logger.LogDebug("Found MCSH sub-chunk, size: {Size}", subChunkSize);
                                                    try
                                                    {
                                                        // Store the shadow map data
                                                        terrainChunk.ShadowMapData = new byte[subChunkSize];
                                                        int bytesRead = br.Read(terrainChunk.ShadowMapData, 0, (int)subChunkSize);
                                                        _logger.LogDebug("Read {BytesRead} bytes of shadow map data", bytesRead);
                                                    }
                                                    catch (EndOfStreamException ex)
                                                    {
                                                        _logger.LogWarning(ex, "Error reading MCSH shadow map data");
                                                    }
                                                    break;
                                                    
                                                default:
                                                    // Skip unknown sub-chunks
                                                    _logger.LogDebug("Skipping unknown MCNK sub-chunk {SubChunkId}, size: {Size}", peekId, subChunkSize);
                                                    break;
                                            }
                                            
                                            // Move to the end of the sub-chunk
                                            try
                                            {
                                                ms.Position = subChunkEndPos;
                                            }
                                            catch (IOException)
                                            {
                                                _logger.LogDebug("Unable to seek to next sub-chunk position at {Position}, likely reached end of file", subChunkEndPos);
                                                break;
                                            }
                                        }
                                    }
                                    catch (EndOfStreamException ex)
                                    {
                                        // Handle special case for _tex0.adt files that might be truncated
                                        _logger.LogDebug(ex, "Reached end of stream while reading terrain chunk position data");
                                        
                                        // Use default values
                                        terrainChunk.X = 0;
                                        terrainChunk.Y = 0;
                                        terrainChunk.Z = 0;
                                    }
                                }
                                catch (EndOfStreamException ex)
                                {
                                    _logger.LogDebug(ex, "Reached end of stream while reading terrain chunk header data");
                                }
                                
                                // Store terrain chunk even with partial data
                                adtInfo.TerrainChunkDetails.Add(terrainChunk);
                                adtInfo.TerrainChunks++;
                                break;
                                
                            case "MTEX": // Texture names chunk
                                _logger.LogDebug("Found MTEX chunk, size: {Size}", chunkSize);
                                long endPos = ms.Position + chunkSize;
                                while (ms.Position < endPos)
                                {
                                    string textureName = ReadNullTerminatedString(br);
                                    if (!string.IsNullOrWhiteSpace(textureName))
                                    {
                                        adtInfo.TextureNames.Add(textureName);
                                    }
                                    
                                    // Break if we've reached the end of the chunk
                                    if (ms.Position >= endPos)
                                        break;
                                }
                                _logger.LogDebug("Found {Count} texture names", adtInfo.TextureNames.Count);
                                break;
                                
                            case "MMDX": // Model names chunk
                                _logger.LogDebug("Found MMDX chunk, size: {Size}", chunkSize);
                                endPos = ms.Position + chunkSize;
                                while (ms.Position < endPos)
                                {
                                    string modelName = ReadNullTerminatedString(br);
                                    if (!string.IsNullOrWhiteSpace(modelName))
                                    {
                                        adtInfo.ModelNames.Add(modelName);
                                    }
                                    
                                    // Break if we've reached the end of the chunk
                                    if (ms.Position >= endPos)
                                        break;
                                }
                                _logger.LogDebug("Found {Count} model names", adtInfo.ModelNames.Count);
                                break;
                                
                            case "MWMO": // WMO names chunk
                                _logger.LogDebug("Found MWMO chunk, size: {Size}", chunkSize);
                                endPos = ms.Position + chunkSize;
                                while (ms.Position < endPos)
                                {
                                    string wmoName = ReadNullTerminatedString(br);
                                    if (!string.IsNullOrWhiteSpace(wmoName))
                                    {
                                        adtInfo.WmoNames.Add(wmoName);
                                    }
                                    
                                    // Break if we've reached the end of the chunk
                                    if (ms.Position >= endPos)
                                        break;
                                }
                                _logger.LogDebug("Found {Count} WMO names", adtInfo.WmoNames.Count);
                                break;
                                
                            case "MODF": // WMO placements chunk
                                _logger.LogDebug("Found MODF chunk, size: {Size}", chunkSize);
                                int wmoPlacementCount = (int)(chunkSize / 64); // Each entry is 64 bytes
                                adtInfo.WmoPlacements = wmoPlacementCount;
                                
                                _logger.LogDebug("MODF chunk details: Position={Position}, Size={Size}, WMO Placement Count={Count}",
                                    ms.Position - 8, chunkSize, wmoPlacementCount);
                                
                                if (wmoPlacementCount == 0)
                                {
                                    _logger.LogWarning("MODF chunk found but contains no WMO placements (size might be wrong)");
                                    break;
                                }
                                
                                // Dump the first few bytes for debugging if there are placements
                                if (wmoPlacementCount > 0)
                                {
                                    long startPos = ms.Position;
                                    byte[] sampleBytes = br.ReadBytes(Math.Min(32, (int)chunkSize));
                                    ms.Position = startPos; // Reset position
                                    
                                    StringBuilder hexDump = new StringBuilder();
                                    foreach (byte b in sampleBytes)
                                    {
                                        hexDump.Append(b.ToString("X2")).Append(" ");
                                    }
                                    _logger.LogDebug("MODF chunk first bytes: {HexDump}", hexDump.ToString());
                                }
                                
                                // Read actual WMO placement data
                                for (int i = 0; i < wmoPlacementCount; i++)
                                {
                                    var placement = new WmoPlacementInfo();
                                    
                                    // Read basic info - using Warcraft.NET's exact format
                                    placement.NameId = br.ReadUInt32();  // Changed from ReadInt32 to ReadUInt32
                                    placement.UniqueId = br.ReadInt32();
                                    
                                    // Read position
                                    placement.Position = new Vector3(
                                        br.ReadSingle(),
                                        br.ReadSingle(),
                                        br.ReadSingle()
                                    );
                                    
                                    // Read rotation
                                    placement.Rotation = new Vector3(
                                        br.ReadSingle(),
                                        br.ReadSingle(),
                                        br.ReadSingle()
                                    );
                                    
                                    // Read bounds
                                    placement.BoundingBox1 = new Vector3(
                                        br.ReadSingle(),
                                        br.ReadSingle(),
                                        br.ReadSingle()
                                    );
                                    
                                    placement.BoundingBox2 = new Vector3(
                                        br.ReadSingle(),
                                        br.ReadSingle(),
                                        br.ReadSingle()
                                    );
                                    
                                    // Read flags and doodad set - using Warcraft.NET's exact format
                                    placement.Flags = br.ReadUInt16();   // Changed from ReadInt16 to ReadUInt16
                                    placement.DoodadSet = br.ReadUInt16(); // Changed from ReadInt16 to ReadUInt16
                                    placement.NameSet = br.ReadUInt16();   // Changed from ReadInt16 to ReadUInt16
                                    placement.Scale = br.ReadUInt16();     // Changed from Unknown to Scale and ReadInt16 to ReadUInt16
                                    
                                    // Check if the flags indicate that NameId is a FileDataID
                                    bool nameIdIsFileDataId = (placement.Flags & 0x8) != 0; // 0x8 is MODFFlags.NameIdIsFiledataId
                                    placement.NameIdIsFileDataId = nameIdIsFileDataId;
                                    
                                    // Validate data to ensure it's reasonable
                                    bool isValid = true;
                                    
                                    // Check for unreasonably large IDs - unless it's a FileDataID which can be large
                                    if (!nameIdIsFileDataId && placement.NameId > 10000)
                                    {
                                        _logger.LogWarning("Skipping WMO placement with suspicious NameId: {NameId}", placement.NameId);
                                        isValid = false;
                                    }
                                    
                                    // Validate position and rotation to ensure they're within reasonable ranges
                                    if (IsInvalidVector(placement.Position) || IsInvalidVector(placement.Rotation) ||
                                        IsInvalidVector(placement.BoundingBox1) || IsInvalidVector(placement.BoundingBox2))
                                    {
                                        _logger.LogWarning("Skipping WMO placement with invalid position, rotation, or bounding box: Position={Position}, Rotation={Rotation}",
                                            placement.Position, placement.Rotation);
                                        isValid = false;
                                    }
                                    
                                    if (isValid)
                                    {
                                        // Add to list
                                        adtInfo.WmoPlacementDetails.Add(placement);
                                        
                                        // Add unique ID
                                        if (placement.UniqueId != 0 && !adtInfo.UniqueIds.Contains(placement.UniqueId))
                                        {
                                            adtInfo.UniqueIds.Add(placement.UniqueId);
                                        }

                                        // Log each WMO placement for debugging
                                        _logger.LogDebug("WMO Placement {Index}: NameId={NameId}, UniqueId={UniqueId}, Position=({X}, {Y}, {Z}), FileDataID={IsFileDataId}",
                                            i, placement.NameId, placement.UniqueId, placement.Position.X, placement.Position.Y, placement.Position.Z, nameIdIsFileDataId);
                                        
                                        // Log if this is using a FileDataID
                                        if (nameIdIsFileDataId)
                                        {
                                            _logger.LogDebug("WMO placement using FileDataID: {FileDataID}", placement.NameId);
                                            isUsingFileDataId = true;
                                            
                                            // Store the FileDataID as a reference
                                            if (placement.NameId > 0 && !adtInfo.ReferencedWmos.Contains(placement.NameId))
                                            {
                                                adtInfo.ReferencedWmos.Add(placement.NameId);
                                                
                                                // Also ensure we have a property to store WMO FileDataIDs
                                                if (!adtInfo.Properties.TryGetValue("WmoFileDataIds", out var wmoFileDataIdsObj))
                                                {
                                                    adtInfo.Properties["WmoFileDataIds"] = new List<uint>();
                                                    wmoFileDataIdsObj = adtInfo.Properties["WmoFileDataIds"];
                                                }
                                                
                                                // Add to the list if it doesn't exist
                                                if (wmoFileDataIdsObj is List<uint> wmoFileDataIds && !wmoFileDataIds.Contains(placement.NameId))
                                                {
                                                    wmoFileDataIds.Add(placement.NameId);
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                _logger.LogDebug("Read {Count} WMO placements", adtInfo.WmoPlacementDetails.Count);
                                break;
                            
                            case "MDDF": // Model placement chunk
                                _logger.LogDebug("Found MDDF chunk, size: {Size}", chunkSize);
                                int modelPlacementCount = (int)(chunkSize / 36); // Each entry is 36 bytes
                                adtInfo.ModelPlacements = modelPlacementCount;
                                
                                // Read actual model placement data
                                for (int i = 0; i < modelPlacementCount; i++)
                                {
                                    var placement = new ModelPlacementInfo();
                                    
                                    // Read basic info - using Warcraft.NET's exact format
                                    placement.NameId = br.ReadUInt32();
                                    placement.UniqueId = (int)br.ReadUInt32(); // UniqueID is uint in Warcraft.NET's MDDFEntry
                                    
                                    // Read position
                                    placement.Position = new Vector3(
                                        br.ReadSingle(),
                                        br.ReadSingle(),
                                        br.ReadSingle()
                                    );
                                    
                                    // Read rotation
                                    placement.Rotation = new Vector3(
                                        br.ReadSingle(),
                                        br.ReadSingle(),
                                        br.ReadSingle()
                                    );
                                    
                                    // Read scale - IMPORTANT: In Warcraft.NET, this is a ushort ScalingFactor, not a float
                                    // 1024 = 1.0f scale, so we need to convert
                                    ushort scalingFactor = br.ReadUInt16();
                                    placement.Scale = scalingFactor / 1024.0f; // Convert from ushort (1024=1.0f) to float
                                    
                                    // Read flags
                                    ushort flags = br.ReadUInt16(); // In Warcraft.NET MDDFEntry, Flags is MDDFFlags enum (ushort)
                                    placement.Flags = flags;
                                    
                                    // Check if the flags indicate that NameId is a FileDataID
                                    // In MDDFFlags, NameIdIsFiledataId = 0x40
                                    bool nameIdIsFileDataId = (flags & 0x40) != 0;
                                    placement.NameIdIsFileDataId = nameIdIsFileDataId;
                                    
                                    // Validate data to ensure it's reasonable
                                    bool isValid = true;
                                    
                                    // Check for unreasonably large IDs - unless it's a FileDataID which can be large
                                    if (!nameIdIsFileDataId && placement.NameId > 10000)
                                    {
                                        _logger.LogWarning("Skipping model placement with suspicious NameId: {NameId}", placement.NameId);
                                        isValid = false;
                                    }
                                    
                                    // Validate position and rotation to ensure they're within reasonable ranges
                                    if (IsInvalidVector(placement.Position) || IsInvalidVector(placement.Rotation))
                                    {
                                        _logger.LogWarning("Skipping model placement with invalid position or rotation: Position={Position}, Rotation={Rotation}",
                                            placement.Position, placement.Rotation);
                                        isValid = false;
                                    }
                                    
                                    if (isValid)
                                    {
                                        // Add to list
                                        adtInfo.ModelPlacementDetails.Add(placement);
                                        
                                        // Add unique ID
                                        if (placement.UniqueId != 0 && !adtInfo.UniqueIds.Contains(placement.UniqueId))
                                        {
                                            adtInfo.UniqueIds.Add(placement.UniqueId);
                                        }
                                        
                                        // Log details for debugging
                                        _logger.LogDebug("Model Placement {Index}: NameId={NameId}, UniqueId={UniqueId}, Position=({X}, {Y}, {Z}), Scale={Scale}, FileDataID={IsFileDataId}",
                                            i, placement.NameId, placement.UniqueId, 
                                            placement.Position.X, placement.Position.Y, placement.Position.Z, 
                                            placement.Scale, nameIdIsFileDataId);
                                        
                                        // Log if this is using a FileDataID
                                        if (nameIdIsFileDataId)
                                        {
                                            _logger.LogDebug("Model placement using FileDataID: {FileDataID}", placement.NameId);
                                            isUsingFileDataId = true;
                                            
                                            // Store the FileDataID as a reference
                                            if (placement.NameId > 0 && !adtInfo.ReferencedModels.Contains(placement.NameId))
                                            {
                                                adtInfo.ReferencedModels.Add(placement.NameId);
                                                
                                                // Also ensure we have a property to store model FileDataIDs
                                                if (!adtInfo.Properties.TryGetValue("ModelFileDataIds", out var modelFileDataIdsObj))
                                                {
                                                    adtInfo.Properties["ModelFileDataIds"] = new List<uint>();
                                                    modelFileDataIdsObj = adtInfo.Properties["ModelFileDataIds"];
                                                }
                                                
                                                // Add to the list if it doesn't exist
                                                if (modelFileDataIdsObj is List<uint> modelFileDataIds && !modelFileDataIds.Contains(placement.NameId))
                                                {
                                                    modelFileDataIds.Add(placement.NameId);
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                _logger.LogDebug("Read {Count} model placements", adtInfo.ModelPlacementDetails.Count);
                                break;
                            
                            default:
                                // Check for ML chunks (Terrain LOD chunks)
                                if (chunkId.StartsWith("ML"))
                                {
                                    // Initialize TerrainLod if not already created
                                    if (adtInfo.TerrainLod == null)
                                    {
                                        adtInfo.TerrainLod = new Models.TerrainLod();
                                        _logger.LogDebug("Created TerrainLod object for ML chunks");
                                    }

                                    // Process specific ML chunks
                                    switch (chunkId)
                                    {
                                        case "MLHD": // LOD Header
                                            try
                                            {
                                                var header = new Models.TerrainLodHeader();
                                                header.Flags = br.ReadUInt32();
                                                
                                                // Read bounding box
                                                var bbox = new Models.BoundingBox();
                                                bbox.Min = new System.Numerics.Vector3(
                                                    br.ReadSingle(), 
                                                    br.ReadSingle(), 
                                                    br.ReadSingle());
                                                bbox.Max = new System.Numerics.Vector3(
                                                    br.ReadSingle(), 
                                                    br.ReadSingle(),
                                                    br.ReadSingle());
                                                header.BoundingBox = bbox;
                                                
                                                adtInfo.TerrainLod.Header = header;
                                                _logger.LogDebug("Parsed MLHD chunk with flags: {Flags}", header.Flags);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLHD chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLVH": // LOD Heightmap
                                            try
                                            {
                                                int heightCount = (int)(chunkSize / sizeof(float));
                                                var heightData = new float[heightCount];
                                                
                                                for (int i = 0; i < heightCount; i++)
                                                {
                                                    heightData[i] = br.ReadSingle();
                                                }
                                                
                                                adtInfo.TerrainLod.HeightData = heightData;
                                                _logger.LogDebug("Parsed MLVH chunk with {Count} height values", heightCount);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLVH chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLLL": // LOD Levels
                                            try
                                            {
                                                // Each MLLLEntry is 20 bytes
                                                int entrySize = 20;
                                                int entryCount = (int)(chunkSize / entrySize);
                                                
                                                for (int i = 0; i < entryCount; i++)
                                                {
                                                    var level = new Models.TerrainLodLevel
                                                    {
                                                        LodBands = br.ReadSingle(),
                                                        HeightLength = br.ReadUInt32(),
                                                        HeightIndex = br.ReadUInt32(),
                                                        MapAreaLowLength = br.ReadUInt32(),
                                                        MapAreaLowIndex = br.ReadUInt32()
                                                    };
                                                    
                                                    adtInfo.TerrainLod.Levels.Add(level);
                                                }
                                                
                                                _logger.LogDebug("Parsed MLLL chunk with {Count} level entries", entryCount);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLLL chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLND": // LOD Node data
                                            try
                                            {
                                                // Each MLNDEntry is 20 bytes
                                                int entrySize = 20;
                                                int entryCount = (int)(chunkSize / entrySize);
                                                
                                                for (int i = 0; i < entryCount; i++)
                                                {
                                                    var node = new Models.TerrainLodNode
                                                    {
                                                        VertexIndicesOffset = br.ReadUInt32(),
                                                        VertexIndicesLength = br.ReadUInt32(),
                                                        Unknown1 = br.ReadUInt32(),
                                                        Unknown2 = br.ReadUInt32()
                                                    };
                                                    
                                                    // Read 4 child indices (2 bytes each)
                                                    node.ChildIndices = new ushort[4];
                                                    for (int j = 0; j < 4; j++)
                                                    {
                                                        node.ChildIndices[j] = br.ReadUInt16();
                                                    }
                                                    
                                                    adtInfo.TerrainLod.Nodes.Add(node);
                                                }
                                                
                                                _logger.LogDebug("Parsed MLND chunk with {Count} node entries", entryCount);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLND chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLVI": // LOD Vertex Indices
                                            try
                                            {
                                                int indexCount = (int)(chunkSize / sizeof(ushort));
                                                var indices = new ushort[indexCount];
                                                
                                                for (int i = 0; i < indexCount; i++)
                                                {
                                                    indices[i] = br.ReadUInt16();
                                                }
                                                
                                                adtInfo.TerrainLod.VertexIndices = indices;
                                                _logger.LogDebug("Parsed MLVI chunk with {Count} vertex indices", indexCount);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLVI chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLSI": // LOD Skirt Indices
                                            try
                                            {
                                                int indexCount = (int)(chunkSize / sizeof(ushort));
                                                var indices = new ushort[indexCount];
                                                
                                                for (int i = 0; i < indexCount; i++)
                                                {
                                                    indices[i] = br.ReadUInt16();
                                                }
                                                
                                                adtInfo.TerrainLod.SkirtIndices = indices;
                                                _logger.LogDebug("Parsed MLSI chunk with {Count} skirt indices", indexCount);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLSI chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLLD": // LOD Liquid Data
                                            try
                                            {
                                                var liquidData = new Models.TerrainLodLiquidData();
                                                liquidData.Flags = br.ReadUInt32();
                                                liquidData.DepthChunkSize = br.ReadUInt16();
                                                liquidData.AlphaChunkSize = br.ReadUInt16();
                                                
                                                // Read depth chunk data
                                                if (liquidData.DepthChunkSize > 0)
                                                {
                                                    liquidData.DepthChunkData = br.ReadBytes(liquidData.DepthChunkSize);
                                                }
                                                
                                                // Read alpha chunk data
                                                if (liquidData.AlphaChunkSize > 0)
                                                {
                                                    liquidData.AlphaChunkData = br.ReadBytes(liquidData.AlphaChunkSize);
                                                }
                                                
                                                adtInfo.TerrainLod.LiquidData = liquidData;
                                                _logger.LogDebug("Parsed MLLD chunk with flags: {Flags}", liquidData.Flags);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLLD chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLLN": // LOD Liquid Node
                                            try
                                            {
                                                var liquidNode = new Models.TerrainLodLiquidNode();
                                                liquidNode.Unknown1 = br.ReadUInt32();
                                                liquidNode.MlliLength = br.ReadUInt32();
                                                liquidNode.Unknown3 = br.ReadUInt32();
                                                liquidNode.Unknown4a = br.ReadUInt16();
                                                liquidNode.Unknown4b = br.ReadUInt16();
                                                liquidNode.Unknown5 = br.ReadUInt32();
                                                liquidNode.Unknown6 = br.ReadUInt32();
                                                
                                                adtInfo.TerrainLod.LiquidNode = liquidNode;
                                                _logger.LogDebug("Parsed MLLN chunk");
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLLN chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLLI": // LOD Liquid Indices
                                            try
                                            {
                                                int vectorCount = (int)(chunkSize / (sizeof(float) * 3)); // Each Vector3 is 12 bytes
                                                var liquidIndices = new System.Numerics.Vector3[vectorCount];
                                                
                                                for (int i = 0; i < vectorCount; i++)
                                                {
                                                    liquidIndices[i] = new System.Numerics.Vector3(
                                                        br.ReadSingle(),
                                                        br.ReadSingle(),
                                                        br.ReadSingle()
                                                    );
                                                }
                                                
                                                adtInfo.TerrainLod.LiquidIndices = liquidIndices;
                                                _logger.LogDebug("Parsed MLLI chunk with {Count} liquid indices", vectorCount);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLLI chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        case "MLLV": // LOD Liquid Vertices
                                            try
                                            {
                                                int vectorCount = (int)(chunkSize / (sizeof(float) * 3)); // Each Vector3 is 12 bytes
                                                var liquidVertices = new System.Numerics.Vector3[vectorCount];
                                                
                                                for (int i = 0; i < vectorCount; i++)
                                                {
                                                    liquidVertices[i] = new System.Numerics.Vector3(
                                                        br.ReadSingle(),
                                                        br.ReadSingle(),
                                                        br.ReadSingle()
                                                    );
                                                }
                                                
                                                adtInfo.TerrainLod.LiquidVertices = liquidVertices;
                                                _logger.LogDebug("Parsed MLLV chunk with {Count} liquid vertices", vectorCount);
                                            }
                                            catch (Exception ex)
                                            {
                                                _logger.LogWarning(ex, "Error parsing MLLV chunk");
                                                ms.Position += chunkSize; // Skip the rest of the chunk
                                            }
                                            break;
                                            
                                        default:
                                            // Skip unknown ML chunks
                                            _logger.LogDebug("Skipping unknown ML chunk {ChunkId}, size: {Size}", chunkId, chunkSize);
                                            ms.Position += chunkSize;
                                            break;
                                    }
                                }
                                else
                                {
                                    // Skip other unknown chunks
                                    _logger.LogDebug("Skipping unknown chunk {ChunkId}, size: {Size}", chunkId, chunkSize);
                                    try
                                    {
                                        ms.Position += chunkSize;
                                    }
                                    catch (IOException)
                                    {
                                        // If we can't seek, we've likely reached the end of the file
                                        _logger.LogDebug("Unable to seek {ChunkSize} bytes forward, ending parsing", chunkSize);
                                        break;
                                    }
                                }
                                break;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Error processing chunk {ChunkId} with size {Size}", chunkId, chunkSize);
                        // Try to skip this chunk and continue
                        try
                        {
                            ms.Position += chunkSize;
                        }
                        catch
                        {
                            // If we can't recover, just break out of the loop
                            break;
                        }
                    }
                }
            }
            
            // Handle case of a standard monolithic ADT that should contain all data
            if (isMonolithicAdt)
            {
                _logger.LogDebug("Processing monolithic ADT with all data embedded");
            }

            // At the end of the method, update the adtInfo with our findings
            adtInfo.UsesFileDataId = isUsingFileDataId;
            
            // If we have model or WMO placements with FileDataIDs, ensure we have those references
            EnsureFileDataIdReferences(adtInfo);
            
            return adtInfo;
        }

        /// <summary>
        /// Ensures FileDataID references are properly extracted from placements.
        /// </summary>
        /// <param name="adtInfo">The ADT info to update.</param>
        private void EnsureFileDataIdReferences(AdtInfo adtInfo)
        {
            // Collect FileDataIDs from model placements
            var modelFileDataIds = new HashSet<uint>();
            foreach (var placement in adtInfo.ModelPlacementDetails)
            {
                if (placement.NameIdIsFileDataId && placement.NameId > 0)
                {
                    modelFileDataIds.Add(placement.NameId);
                    adtInfo.ReferencedModels.Add(placement.NameId);
                }
            }
            
            // Add to Properties for ModelFileDataIds
            if (modelFileDataIds.Count > 0)
            {
                if (!adtInfo.Properties.TryGetValue("ModelFileDataIds", out var modelFileDataIdsObj))
                {
                    adtInfo.Properties["ModelFileDataIds"] = new List<uint>();
                    modelFileDataIdsObj = adtInfo.Properties["ModelFileDataIds"];
                }
                
                if (modelFileDataIdsObj is List<uint> existingModelFileDataIds)
                {
                    foreach (var id in modelFileDataIds)
                    {
                        if (!existingModelFileDataIds.Contains(id))
                        {
                            existingModelFileDataIds.Add(id);
                        }
                    }
                }
            }
            
            // Collect FileDataIDs from WMO placements
            var wmoFileDataIds = new HashSet<uint>();
            foreach (var placement in adtInfo.WmoPlacementDetails)
            {
                if (placement.NameIdIsFileDataId && placement.NameId > 0)
                {
                    wmoFileDataIds.Add(placement.NameId);
                    adtInfo.ReferencedWmos.Add(placement.NameId);
                }
            }
            
            // Add to Properties for WmoFileDataIds
            if (wmoFileDataIds.Count > 0)
            {
                if (!adtInfo.Properties.TryGetValue("WmoFileDataIds", out var wmoFileDataIdsObj))
                {
                    adtInfo.Properties["WmoFileDataIds"] = new List<uint>();
                    wmoFileDataIdsObj = adtInfo.Properties["WmoFileDataIds"];
                }
                
                if (wmoFileDataIdsObj is List<uint> existingWmoFileDataIds)
                {
                    foreach (var id in wmoFileDataIds)
                    {
                        if (!existingWmoFileDataIds.Contains(id))
                        {
                            existingWmoFileDataIds.Add(id);
                        }
                    }
                }
            }
            
            // If we found any FileDataIDs, mark the ADT as using FileDataIDs
            if (modelFileDataIds.Count > 0 || wmoFileDataIds.Count > 0)
            {
                adtInfo.UsesFileDataId = true;
            }
        }

        private string ReadNullTerminatedString(BinaryReader br)
        {
            var bytes = new List<byte>();
            byte b;
            
            try
            {
                while ((b = br.ReadByte()) != 0)
                {
                    bytes.Add(b);
                    
                    // Safety check to prevent infinite loops
                    if (bytes.Count > 1000) // Max string length
                    {
                        _logger.LogWarning("String exceeded maximum length of 1000 bytes, truncating");
                        break;
                    }
                }
            }
            catch (EndOfStreamException)
            {
                // End of stream reached before null terminator
                _logger.LogDebug("End of stream reached while reading null-terminated string");
                // We'll use what we have so far
            }
            
            // If we read no bytes but didn't get an exception, return empty string
            if (bytes.Count == 0)
            {
                return string.Empty;
            }
            
            // Convert bytes to string
            try
            {
                return Encoding.ASCII.GetString(bytes.ToArray());
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error converting bytes to string");
                return string.Empty;
            }
        }

        /// <summary>
        /// Converts ADT info to an analysis result.
        /// </summary>
        /// <param name="adtInfo">The ADT info to convert.</param>
        /// <param name="filePath">The path to the ADT file.</param>
        /// <returns>An ADT analysis result.</returns>
        private AdtAnalysisResult ConvertToAdtAnalysisResult(AdtInfo adtInfo, string filePath)
        {
            // Log the raw ADT info we're converting
            _logger.LogDebug("Converting ADT info to analysis result - File: {FilePath}, Model Placements: {ModelCount}, WMO Placements: {WmoCount}, Textures: {TextureCount}",
                filePath, adtInfo.ModelPlacementDetails.Count, adtInfo.WmoPlacementDetails.Count, adtInfo.TextureNames.Count);

            var result = new AdtAnalysisResult
            {
                FileName = Path.GetFileName(filePath),
                FilePath = filePath,
                AdtVersion = (uint)adtInfo.Version,
                UsesFileDataId = adtInfo.UsesFileDataId,
                Header = new AdtHeader
                {
                    Flags = adtInfo.Flags,
                    TextureLayerCount = adtInfo.TextureNames.Count,
                    TerrainChunkCount = (int)adtInfo.TerrainChunks,
                    ModelReferenceCount = adtInfo.ModelNames.Count,
                    WmoReferenceCount = adtInfo.WmoNames.Count,
                    ModelPlacementCount = adtInfo.ModelPlacementDetails.Count,
                    WmoPlacementCount = adtInfo.WmoPlacementDetails.Count,
                    // Assuming DoodadReferenceCount should be 0 if not tracking doodads separately
                    DoodadReferenceCount = 0 
                }
            };

            // Extract coordinates from filename
            var match = CoordinateRegex.Match(result.FileName);
            if (match.Success)
            {
                if (int.TryParse(match.Groups[1].Value, out int x))
                    result.XCoord = x;

                if (int.TryParse(match.Groups[2].Value, out int y))
                    result.YCoord = y;
            }

            // Process texture references
            foreach (var textureName in adtInfo.TextureNames)
            {
                if (string.IsNullOrEmpty(textureName) || textureName.StartsWith("<unknown", StringComparison.OrdinalIgnoreCase))
                    continue;

                var textureReference = new FileReference
                {
                    OriginalPath = textureName,
                    NormalizedPath = Utilities.PathUtility.NormalizePath(textureName),
                    Type = FileReferenceType.Texture
                };

                // Avoid duplicates
                if (!result.TextureReferences.Any(t => t.NormalizedPath == textureReference.NormalizedPath))
                {
                    result.TextureReferences.Add(textureReference);
                }
            }

            // Log the number of texture references we found and added
            _logger.LogDebug("Processed texture references - Total textures from ADT: {TextureCount}, Added to result: {AddedCount}", 
                adtInfo.TextureNames.Count, result.TextureReferences.Count);

            // Process texture FileDataIDs (only if ADT uses FileDataIDs)
            if (adtInfo.UsesFileDataId)
            {
                // FileDataIDs for textures would be stored in Properties if they exist
                if (adtInfo.Properties.TryGetValue("TextureFileDataIds", out var textureFileDataIdsObj) && 
                    textureFileDataIdsObj is List<uint> textureFileDataIds)
                {
                    foreach (var textureFileDataId in textureFileDataIds)
                    {
                        if (textureFileDataId == 0)
                            continue;

                        var textureReference = new FileReference
                        {
                            OriginalPath = $"<FileDataID:{textureFileDataId}>",
                            NormalizedPath = $"<FileDataID:{textureFileDataId}>",
                            Type = FileReferenceType.Texture,
                            FileDataId = textureFileDataId,
                            UsesFileDataId = true
                        };

                        // Avoid duplicates
                        if (!result.TextureReferences.Any(t => t.FileDataId == textureReference.FileDataId && t.UsesFileDataId))
                        {
                            result.TextureReferences.Add(textureReference);
                        }
                    }
                }
            }

            // Process model references (paths)
            foreach (var modelName in adtInfo.ModelNames)
            {
                if (string.IsNullOrEmpty(modelName) || modelName.StartsWith("<unknown", StringComparison.OrdinalIgnoreCase))
                    continue;

                var modelReference = new FileReference
                {
                    OriginalPath = modelName,
                    NormalizedPath = Utilities.PathUtility.NormalizePath(modelName),
                    Type = FileReferenceType.Model
                };

                // Avoid duplicates
                if (!result.ModelReferences.Any(m => m.NormalizedPath == modelReference.NormalizedPath))
                {
                    result.ModelReferences.Add(modelReference);
                }
            }

            // Process model FileDataIDs (only if ADT uses FileDataIDs)
            if (adtInfo.UsesFileDataId)
            {
                // FileDataIDs for models would be stored in Properties if they exist
                if (adtInfo.Properties.TryGetValue("ModelFileDataIds", out var modelFileDataIdsObj) && 
                    modelFileDataIdsObj is List<uint> modelFileDataIds)
                {
                    foreach (var modelFileDataId in modelFileDataIds)
                    {
                        if (modelFileDataId == 0)
                            continue;
                            
                        var modelReference = new FileReference
                        {
                            OriginalPath = $"<FileDataID:{modelFileDataId}>",
                            NormalizedPath = $"<FileDataID:{modelFileDataId}>",
                            Type = FileReferenceType.Model,
                            FileDataId = modelFileDataId,
                            UsesFileDataId = true
                        };

                        // Avoid duplicates
                        if (!result.ModelReferences.Any(m => m.FileDataId == modelFileDataId && m.UsesFileDataId))
                        {
                            result.ModelReferences.Add(modelReference);
                        }
                    }
                }
            }

            // Log the number of model references we found and added
            _logger.LogDebug("Processed model references - From ADT: {ModelCount}, Added to result: {AddedCount}",
                adtInfo.ModelNames.Count, result.ModelReferences.Count);

            // Process WMO references (paths)
            foreach (var wmoName in adtInfo.WmoNames)
            {
                if (string.IsNullOrEmpty(wmoName) || wmoName.StartsWith("<unknown", StringComparison.OrdinalIgnoreCase))
                    continue;

                var wmoReference = new FileReference
                {
                    OriginalPath = wmoName,
                    NormalizedPath = Utilities.PathUtility.NormalizePath(wmoName),
                    Type = FileReferenceType.Wmo
                };

                // Avoid duplicates
                if (!result.WmoReferences.Any(w => w.NormalizedPath == wmoReference.NormalizedPath))
                {
                    result.WmoReferences.Add(wmoReference);
                }
            }

            // Process WMO FileDataIDs (only if ADT uses FileDataIDs)
            if (adtInfo.UsesFileDataId)
            {
                // FileDataIDs for WMOs would be stored in Properties if they exist
                if (adtInfo.Properties.TryGetValue("WmoFileDataIds", out var wmoFileDataIdsObj) && 
                    wmoFileDataIdsObj is List<uint> wmoFileDataIds)
                {
                    foreach (var wmoFileDataId in wmoFileDataIds)
                    {
                        if (wmoFileDataId == 0)
                            continue;

                        var wmoReference = new FileReference
                        {
                            OriginalPath = $"<FileDataID:{wmoFileDataId}>",
                            NormalizedPath = $"<FileDataID:{wmoFileDataId}>",
                            Type = FileReferenceType.Wmo,
                            FileDataId = wmoFileDataId,
                            UsesFileDataId = true
                        };

                        // Avoid duplicates
                        if (!result.WmoReferences.Any(w => w.FileDataId == wmoReference.FileDataId && w.UsesFileDataId))
                        {
                            result.WmoReferences.Add(wmoReference);
                        }
                    }
                }
            }
            
            // Generate references from placements with FileDataIDs if they don't already exist
            EnsurePlacementBasedReferences(result);

            // Log the number of WMO references we found and added
            _logger.LogDebug("Processed WMO references - From ADT: {WmoCount}, Added to result: {AddedCount}",
                adtInfo.WmoNames.Count, result.WmoReferences.Count);

            // Process model placements
            foreach (var placementInfo in adtInfo.ModelPlacementDetails)
            {
                try 
                {
                    string? modelName = null;
                    bool usesFileDataId = false;
                    uint fileDataId = 0;

                    // Determine the model name based on whether the NameId is a FileDataID or an index
                    if (adtInfo.UsesFileDataId)
                    {
                        usesFileDataId = true;
                        fileDataId = placementInfo.NameId;
                        
                        // Try to find a reference with this FileDataID
                        var modelRef = result.ModelReferences.FirstOrDefault(m => m.FileDataId == fileDataId && m.UsesFileDataId);
                        if (modelRef != null)
                        {
                            modelName = modelRef.OriginalPath;
                        }
                        else
                        {
                            modelName = $"<FileDataID:{fileDataId}>";
                            _logger.LogDebug("Found model placement with FileDataID {FileDataId} but no corresponding model reference", fileDataId);
                        }
                    }
                    else
                    {
                        // Legacy format - NameId is an index into the model names array
                        if (placementInfo.NameId < adtInfo.ModelNames.Count)
                        {
                            modelName = adtInfo.ModelNames[(int)placementInfo.NameId];
                        }
                        else
                        {
                            modelName = $"<unknown model index {placementInfo.NameId}>";
                            _logger.LogWarning("Model index {NameId} out of range (0-{ModelCount})", placementInfo.NameId, adtInfo.ModelNames.Count - 1);
                        }
                    }

                    // Create the model placement
                    var modelPlacement = new ModelPlacement
                    {
                        UniqueId = (int)placementInfo.UniqueId,
                        NameId = (int)placementInfo.NameId,
                        Name = modelName,
                        Position = placementInfo.Position,
                        Rotation = placementInfo.Rotation,
                        Scale = placementInfo.Scale,
                        Flags = placementInfo.Flags,
                        FileDataId = fileDataId,
                        UsesFileDataId = usesFileDataId
                    };

                    // Add to result
                    result.ModelPlacements.Add(modelPlacement);

                    // Track unique ID
                    result.UniqueIds.Add((int)placementInfo.UniqueId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing model placement: {Error}", ex.Message);
                    result.Errors.Add($"Error processing model placement: {ex.Message}");
                }
            }

            // Log the number of model placements we processed
            _logger.LogDebug("Processed model placements - From ADT: {PlacementCount}, Added to result: {AddedCount}",
                adtInfo.ModelPlacementDetails.Count, result.ModelPlacements.Count);

            // Process WMO placements
            foreach (var placementInfo in adtInfo.WmoPlacementDetails)
            {
                try
                {
                    string? wmoName = null;
                    bool usesFileDataId = false;
                    uint fileDataId = 0;

                    // Determine the WMO name based on whether the NameId is a FileDataID or an index
                    if (adtInfo.UsesFileDataId)
                    {
                        usesFileDataId = true;
                        fileDataId = placementInfo.NameId;
                        
                        // Try to find a reference with this FileDataID
                        var wmoRef = result.WmoReferences.FirstOrDefault(w => w.FileDataId == fileDataId && w.UsesFileDataId);
                        if (wmoRef != null)
                        {
                            wmoName = wmoRef.OriginalPath;
                        }
                        else
                        {
                            wmoName = $"<FileDataID:{fileDataId}>";
                            _logger.LogDebug("Found WMO placement with FileDataID {FileDataId} but no corresponding WMO reference", fileDataId);
                        }
                    }
                    else
                    {
                        // Legacy format - NameId is an index into the WMO names array
                        if (placementInfo.NameId < adtInfo.WmoNames.Count)
                        {
                            wmoName = adtInfo.WmoNames[(int)placementInfo.NameId];
                        }
                        else
                        {
                            wmoName = $"<unknown WMO index {placementInfo.NameId}>";
                            _logger.LogWarning("WMO index {NameId} out of range (0-{WmoCount})", placementInfo.NameId, adtInfo.WmoNames.Count - 1);
                        }
                    }

                    // Create the WMO placement
                    var wmoPlacement = new WmoPlacement
                    {
                        UniqueId = (int)placementInfo.UniqueId,
                        NameId = (int)placementInfo.NameId,
                        Name = wmoName,
                        Position = placementInfo.Position,
                        Rotation = placementInfo.Rotation,
                        DoodadSet = placementInfo.DoodadSet,
                        NameSet = placementInfo.NameSet,
                        Flags = placementInfo.Flags,
                        Scale = placementInfo.Scale,
                        FileDataId = fileDataId,
                        UsesFileDataId = usesFileDataId
                    };

                    // Add to result
                    result.WmoPlacements.Add(wmoPlacement);

                    // Track unique ID
                    result.UniqueIds.Add((int)placementInfo.UniqueId);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing WMO placement: {Error}", ex.Message);
                    result.Errors.Add($"Error processing WMO placement: {ex.Message}");
                }
            }

            // Log the number of WMO placements we processed
            _logger.LogDebug("Processed WMO placements - From ADT: {PlacementCount}, Added to result: {AddedCount}",
                adtInfo.WmoPlacementDetails.Count, result.WmoPlacements.Count);

            // Copy TerrainLod data if it exists
            if (adtInfo.TerrainLod != null)
            {
                result.TerrainLod = adtInfo.TerrainLod;
                _logger.LogDebug("Copied TerrainLod data to analysis result");
            }

            // Process terrain chunks
            // ... existing code ...

            _logger.LogInformation("Converted ADT info to analysis result - File: {FilePath}, Models: {ModelCount}, WMOs: {WmoCount}, Textures: {TextureCount}, Chunks: {ChunkCount}",
                filePath, result.ModelPlacements.Count, result.WmoPlacements.Count, result.TextureReferences.Count, result.TerrainChunks.Count);

            return result;
        }

        /// <summary>
        /// Extracts X and Y coordinates from the filename of an ADT file.
        /// </summary>
        /// <param name="result">The AdtAnalysisResult to update with coordinates.</param>
        private void ExtractCoordinatesFromFilename(AdtAnalysisResult result)
        {
            if (string.IsNullOrEmpty(result.FileName))
                return;

            var match = CoordinateRegex.Match(result.FileName);
            if (match.Success)
            {
                if (int.TryParse(match.Groups[1].Value, out int x))
                    result.XCoord = x;

                if (int.TryParse(match.Groups[2].Value, out int y))
                    result.YCoord = y;
            }
        }

        /// <summary>
        /// Checks if a Vector3 has invalid values (NaN, Infinity, or unreasonably large/small)
        /// </summary>
        private bool IsInvalidVector(Vector3 vector)
        {
            // Check for NaN or Infinity
            if (float.IsNaN(vector.X) || float.IsInfinity(vector.X) ||
                float.IsNaN(vector.Y) || float.IsInfinity(vector.Y) ||
                float.IsNaN(vector.Z) || float.IsInfinity(vector.Z))
            {
                return true;
            }
            
            // Check for unreasonably large values
            if (Math.Abs(vector.X) > 100000 || Math.Abs(vector.Y) > 100000 || Math.Abs(vector.Z) > 100000)
            {
                return true;
            }
            
            return false;
        }

        /// <summary>
        /// Generates references from placements with FileDataIDs if they don't already exist.
        /// </summary>
        /// <param name="result">The AdtAnalysisResult to update.</param>
        private void EnsurePlacementBasedReferences(AdtAnalysisResult result)
        {
            // Collect FileDataIDs from model placements
            var modelFileDataIds = new HashSet<uint>();
            foreach (var placement in result.ModelPlacements)
            {
                if (placement.UsesFileDataId && placement.FileDataId > 0)
                {
                    modelFileDataIds.Add(placement.FileDataId);
                }
            }
            
            // Create model references from FileDataIDs
            foreach (var fileDataId in modelFileDataIds)
            {
                // Check if a reference with this FileDataID already exists
                if (!result.ModelReferences.Any(r => r.FileDataId == fileDataId && r.UsesFileDataId))
                {
                    // Create a new reference
                    var modelReference = new FileReference
                    {
                        OriginalPath = $"<FileDataID:{fileDataId}>",
                        NormalizedPath = $"<FileDataID:{fileDataId}>",
                        Type = FileReferenceType.Model,
                        FileDataId = fileDataId,
                        UsesFileDataId = true
                    };
                    
                    result.ModelReferences.Add(modelReference);
                    _logger.LogDebug("Created model reference from placement FileDataID: {FileDataID}", fileDataId);
                }
            }
            
            // Collect FileDataIDs from WMO placements
            var wmoFileDataIds = new HashSet<uint>();
            foreach (var placement in result.WmoPlacements)
            {
                if (placement.UsesFileDataId && placement.FileDataId > 0)
                {
                    wmoFileDataIds.Add(placement.FileDataId);
                }
            }
            
            // Create WMO references from FileDataIDs
            foreach (var fileDataId in wmoFileDataIds)
            {
                // Check if a reference with this FileDataID already exists
                if (!result.WmoReferences.Any(r => r.FileDataId == fileDataId && r.UsesFileDataId))
                {
                    // Create a new reference
                    var wmoReference = new FileReference
                    {
                        OriginalPath = $"<FileDataID:{fileDataId}>",
                        NormalizedPath = $"<FileDataID:{fileDataId}>",
                        Type = FileReferenceType.Wmo,
                        FileDataId = fileDataId,
                        UsesFileDataId = true
                    };
                    
                    result.WmoReferences.Add(wmoReference);
                    _logger.LogDebug("Created WMO reference from placement FileDataID: {FileDataID}", fileDataId);
                }
            }
            
            // If we found any FileDataIDs, mark the ADT as using FileDataIDs
            if (modelFileDataIds.Count > 0 || wmoFileDataIds.Count > 0)
            {
                result.UsesFileDataId = true;
            }
        }
    }
}