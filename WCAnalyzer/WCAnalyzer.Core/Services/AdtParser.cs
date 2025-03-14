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
                var result = ConvertToAdtAnalysisResult(adtInfo);
                
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
            
            // Check if this is a tex0 or obj0 file based on file size and initial bytes
            bool isSpecialFile = false;
            if (fileData.Length < 1024) // Small files are likely not standard ADT files
            {
                isSpecialFile = true;
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
                                
                            case "MDDF": // Model placements chunk
                                _logger.LogDebug("Found MDDF chunk, size: {Size}", chunkSize);
                                int modelPlacementCount = (int)(chunkSize / 36); // Each entry is 36 bytes
                                adtInfo.ModelPlacements = modelPlacementCount;
                                
                                // Read actual model placement data
                                for (int i = 0; i < modelPlacementCount; i++)
                                {
                                    var placement = new ModelPlacementInfo();
                                    
                                    // Read basic info
                                    placement.NameId = (uint)br.ReadInt32();
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
                                    
                                    // Read scale
                                    placement.Scale = br.ReadSingle();
                                    
                                    // Read flags
                                    placement.Flags = br.ReadInt32();
                                    
                                    // Check if the flags indicate that NameId is a FileDataID
                                    bool nameIdIsFileDataId = (placement.Flags & 0x40) != 0; // 0x40 is MDDFFlags.NameIdIsFiledataId
                                    placement.NameIdIsFileDataId = nameIdIsFileDataId;
                                    
                                    // Validate data to ensure it's reasonable
                                    bool isValid = true;
                                    
                                    // Check for unreasonably large or small values
                                    if (float.IsNaN(placement.Scale) || float.IsInfinity(placement.Scale) || 
                                        placement.Scale < 0.01f || placement.Scale > 100.0f)
                                    {
                                        _logger.LogWarning("Skipping model placement with invalid scale: {Scale}", placement.Scale);
                                        isValid = false;
                                    }
                                    
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
                                        
                                        // Log if this is using a FileDataID
                                        if (nameIdIsFileDataId)
                                        {
                                            _logger.LogDebug("Model placement using FileDataID: {FileDataID}", placement.NameId);
                                        }
                                    }
                                }
                                
                                _logger.LogDebug("Read {Count} model placements", adtInfo.ModelPlacementDetails.Count);
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
                                        }
                                    }
                                }
                                
                                _logger.LogDebug("Read {Count} WMO placements", adtInfo.WmoPlacementDetails.Count);
                                break;
                            
                            default:
                                // Skip unknown chunks
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
            
            return adtInfo;
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
        /// Converts an AdtInfo object to an AdtAnalysisResult object.
        /// </summary>
        /// <param name="adtInfo">The AdtInfo object to convert.</param>
        /// <returns>An AdtAnalysisResult object.</returns>
        private AdtAnalysisResult ConvertToAdtAnalysisResult(AdtInfo adtInfo)
        {
            var result = new AdtAnalysisResult
            {
                FilePath = adtInfo.FilePath ?? string.Empty,
                FileName = adtInfo.FileName ?? string.Empty,
                AdtVersion = (uint)adtInfo.Version
            };

            // Extract coordinates from filename
            ExtractCoordinatesFromFilename(result);

            // Set header information
            result.Header = new AdtHeader
            {
                Flags = adtInfo.Flags,
                TerrainChunkCount = adtInfo.TerrainChunks,
                TextureLayerCount = adtInfo.TextureNames.Count,
                ModelReferenceCount = adtInfo.ModelNames.Count,
                WmoReferenceCount = adtInfo.WmoNames.Count,
                ModelPlacementCount = adtInfo.ModelPlacements,
                WmoPlacementCount = adtInfo.WmoPlacements
            };

            // Convert texture names to texture references
            foreach (var textureName in adtInfo.TextureNames)
            {
                result.TextureReferences.Add(new FileReference
                {
                    OriginalPath = textureName,
                    NormalizedPath = textureName.ToLowerInvariant(),
                    Type = FileReferenceType.Texture
                });
            }

            // Convert model names to model references
            foreach (var modelName in adtInfo.ModelNames)
            {
                result.ModelReferences.Add(new FileReference
                {
                    OriginalPath = modelName,
                    NormalizedPath = modelName.ToLowerInvariant(),
                    Type = FileReferenceType.Model
                });
            }

            // Convert WMO names to WMO references
            foreach (var wmoName in adtInfo.WmoNames)
            {
                result.WmoReferences.Add(new FileReference
                {
                    OriginalPath = wmoName,
                    NormalizedPath = wmoName.ToLowerInvariant(),
                    Type = FileReferenceType.Wmo
                });
            }

            // Create terrain chunks with actual data
            for (int i = 0; i < adtInfo.TerrainChunkDetails.Count; i++)
            {
                var chunkInfo = adtInfo.TerrainChunkDetails[i];
                var chunk = new TerrainChunk
                {
                    AreaId = chunkInfo.AreaId,
                    Flags = (uint)chunkInfo.Flags,
                    Position = new System.Drawing.Point(i % 16, i / 16), // Assuming standard 16x16 grid layout
                    WorldPosition = new Vector3(chunkInfo.X, chunkInfo.Y, chunkInfo.Z),
                    Heights = new float[145], // 9x9 heightmap grid vertices plus padding (9*9*1 = 81)
                    Normals = new Vector3[145], // Same as heights
                    VertexColors = new List<Vector3>(),
                    AlphaMaps = new List<byte[]>(),
                    LiquidLevel = 0.0f
                };

                // Copy height data if available
                if (chunkInfo.Heights != null && chunkInfo.Heights.Length > 0)
                {
                    // Copy the actual height values
                    Array.Copy(chunkInfo.Heights, chunk.Heights, Math.Min(chunkInfo.Heights.Length, chunk.Heights.Length));
                    _logger.LogDebug("Copied {Count} height values for chunk {Index}", 
                        Math.Min(chunkInfo.Heights.Length, chunk.Heights.Length), i);
                }

                // Copy normal data if available
                if (chunkInfo.Normals != null && chunkInfo.Normals.Length > 0)
                {
                    // Copy the actual normal values
                    Array.Copy(chunkInfo.Normals, chunk.Normals, Math.Min(chunkInfo.Normals.Length, chunk.Normals.Length));
                    _logger.LogDebug("Copied {Count} normal values for chunk {Index}", 
                        Math.Min(chunkInfo.Normals.Length, chunk.Normals.Length), i);
                }

                // Process texture layers if available
                if (chunkInfo.TextureLayers != null && chunkInfo.TextureLayers.Count > 0)
                {
                    foreach (var layerInfo in chunkInfo.TextureLayers)
                    {
                        var textureLayer = new TextureLayer
                        {
                            TextureId = (int)layerInfo.TextureId,
                            Flags = (uint)layerInfo.Flags,
                            EffectId = layerInfo.EffectId,
                            AlphaMapOffset = (int)layerInfo.OffsetInMCAL,
                            AlphaMapSize = 0 // Will be calculated if alpha map data is available
                        };

                        // Add texture name if available
                        if (textureLayer.TextureId >= 0 && textureLayer.TextureId < result.TextureReferences.Count)
                        {
                            textureLayer.TextureName = result.TextureReferences[textureLayer.TextureId].OriginalPath;
                        }
                        else
                        {
                            textureLayer.TextureName = $"<unknown texture {textureLayer.TextureId}>";
                        }

                        chunk.TextureLayers.Add(textureLayer);
                    }
                    _logger.LogDebug("Added {Count} texture layers for chunk {Index}", chunk.TextureLayers.Count, i);
                }
                // If no texture layers were found but we have texture references, add a default layer
                else if (result.TextureReferences.Count > 0 && chunk.TextureLayers.Count == 0)
                {
                    var baseLayer = new TextureLayer
                    {
                        TextureId = 0,
                        Flags = 0,
                        EffectId = 0,
                        AlphaMapOffset = 0,
                        AlphaMapSize = 0,
                        TextureName = result.TextureReferences[0].OriginalPath
                    };
                    chunk.TextureLayers.Add(baseLayer);
                }
                // If no texture references at all, add a placeholder
                else if (chunk.TextureLayers.Count == 0)
                {
                    var baseLayer = new TextureLayer
                    {
                        TextureId = 0,
                        Flags = 0,
                        EffectId = 0,
                        AlphaMapOffset = 0,
                        AlphaMapSize = 0,
                        TextureName = "<unknown texture>"
                    };
                    chunk.TextureLayers.Add(baseLayer);
                }

                // Process alpha map data if available
                if (chunkInfo.AlphaMapData != null && chunkInfo.AlphaMapData.Length > 0)
                {
                    // For simplicity, just add the entire alpha map data as a single entry
                    chunk.AlphaMaps.Add(chunkInfo.AlphaMapData);
                    _logger.LogDebug("Added {Size} bytes of alpha map data for chunk {Index}", 
                        chunkInfo.AlphaMapData.Length, i);
                }

                // Process shadow map data if available
                if (chunkInfo.ShadowMapData != null && chunkInfo.ShadowMapData.Length > 0)
                {
                    chunk.ShadowMap = chunkInfo.ShadowMapData;
                    _logger.LogDebug("Added {Size} bytes of shadow map data for chunk {Index}", 
                        chunkInfo.ShadowMapData.Length, i);
                }

                // Create empty DoodadRefs list
                chunk.DoodadRefs = new List<int>();

                result.TerrainChunks.Add(chunk);
            }

            // Add model placements with actual data
            foreach (var placementInfo in adtInfo.ModelPlacementDetails)
            {
                string modelName;
                
                if (placementInfo.NameIdIsFileDataId)
                {
                    // For FileDataID, just use an empty name as we don't have a way to resolve it yet
                    modelName = string.Empty;
                    _logger.LogDebug("Model placement uses FileDataID: {FileDataID}", placementInfo.NameId);
                }
                else
                {
                    // Use the index to look up the model name
                    int modelIndex = (int)placementInfo.NameId; // Cast uint to int for indexing
                    modelName = modelIndex >= 0 && modelIndex < adtInfo.ModelNames.Count 
                        ? adtInfo.ModelNames[modelIndex] 
                        : string.Empty;
                }
                    
                var modelPlacement = new ModelPlacement
                {
                    UniqueId = placementInfo.UniqueId,
                    NameId = (int)placementInfo.NameId, // Cast uint to int for the NameId property
                    Name = modelName,
                    Position = new Vector3(placementInfo.Position.X, placementInfo.Position.Y, placementInfo.Position.Z),
                    Rotation = new Vector3(placementInfo.Rotation.X, placementInfo.Rotation.Y, placementInfo.Rotation.Z),
                    Scale = placementInfo.Scale,
                    Flags = placementInfo.Flags
                };
                
                // Set FileDataID information if applicable
                if (placementInfo.NameIdIsFileDataId)
                {
                    modelPlacement.FileDataId = placementInfo.NameId;
                    modelPlacement.UsesFileDataId = true;
                }
                
                result.ModelPlacements.Add(modelPlacement);
                
                // Add unique ID to the result's unique IDs set
                if (placementInfo.UniqueId != 0)
                {
                    result.UniqueIds.Add(placementInfo.UniqueId);
                }
            }

            // Add WMO placements with actual data
            foreach (var placementInfo in adtInfo.WmoPlacementDetails)
            {
                string wmoName;
                
                if (placementInfo.NameIdIsFileDataId)
                {
                    // For FileDataID, just use an empty name as we don't have a way to resolve it yet
                    wmoName = string.Empty;
                    _logger.LogDebug("WMO placement uses FileDataID: {FileDataID}", placementInfo.NameId);
                }
                else
                {
                    // Use the index to look up the WMO name
                    int wmoIndex = (int)placementInfo.NameId; // Cast to int for indexing
                    wmoName = wmoIndex >= 0 && wmoIndex < adtInfo.WmoNames.Count 
                        ? adtInfo.WmoNames[wmoIndex] 
                        : string.Empty;
                }

                // Add more debug information to trace the name resolution    
                if (string.IsNullOrEmpty(wmoName) && !placementInfo.NameIdIsFileDataId)
                {
                    _logger.LogWarning("Could not resolve WMO name for NameId={NameId}, WmoNames count={Count}",
                        placementInfo.NameId, adtInfo.WmoNames.Count);
                }
                else if (!placementInfo.NameIdIsFileDataId)
                {
                    _logger.LogDebug("Resolved WMO name: {Name} for NameId={NameId}",
                        wmoName, placementInfo.NameId);
                }
                    
                var wmoPlacement = new WmoPlacement
                {
                    UniqueId = placementInfo.UniqueId,
                    NameId = (int)placementInfo.NameId, // Cast uint to int for the NameId property
                    Name = wmoName,
                    Position = new Vector3(placementInfo.Position.X, placementInfo.Position.Y, placementInfo.Position.Z),
                    Rotation = new Vector3(placementInfo.Rotation.X, placementInfo.Rotation.Y, placementInfo.Rotation.Z),
                    Flags = (int)placementInfo.Flags, // Cast ushort to int for the Flags property
                    DoodadSet = placementInfo.DoodadSet,
                    NameSet = placementInfo.NameSet,
                    Scale = placementInfo.Scale
                };
                
                // Set FileDataID information if applicable
                if (placementInfo.NameIdIsFileDataId)
                {
                    wmoPlacement.FileDataId = placementInfo.NameId;
                    wmoPlacement.UsesFileDataId = true;
                }
                
                result.WmoPlacements.Add(wmoPlacement);
                
                // Add unique ID to the result's unique IDs set
                if (placementInfo.UniqueId != 0)
                {
                    result.UniqueIds.Add(placementInfo.UniqueId);
                }
            }

            // Add any remaining unique IDs not already included
            foreach (var uniqueId in adtInfo.UniqueIds)
            {
                if (uniqueId != 0 && !result.UniqueIds.Contains(uniqueId))
                {
                    result.UniqueIds.Add(uniqueId);
                }
            }

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
            if (match.Success && match.Groups.Count >= 3)
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
    }
}