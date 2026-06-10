using System;
using System.Collections.Generic;
using System.IO;
using System.Linq; // Added for Reverse
using System.Text;
using WoWToolbox.Core.Models;

namespace WoWToolbox.Core.WMO
{
    public static class WmoRootLoader
    {
        // Helper to read all null-terminated strings from a byte array
        // Adapted from MOTX.ReadStrings
        private static List<string> ReadNullTerminatedStrings(byte[] data)
        {
            var result = new List<string>();
            if (data == null || data.Length == 0) return result;
            int start = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == 0)
                {
                    if (i > start)
                        result.Add(Encoding.UTF8.GetString(data, start, i - start));
                    start = i + 1;
                }
            }
            // Handle potential trailing string without null terminator (though MOGN should be properly terminated)
            if (start < data.Length)
                result.Add(Encoding.UTF8.GetString(data, start, data.Length - start));
            return result;
        }

        // Returns the group count from MOHD and the list of group file names from MOGN
        public static (int groupCount, List<string> groupNames) LoadGroupInfo(string rootWmoPath)
        {
#if false
            var groupNames = new List<string>();
            int groupCount = -1; // Initialize to -1 to indicate MOHD not found or invalid
            byte[]? mognData = null;
            bool isV14 = false;
            long momoStart = -1;
            long momoSize = -1;

            try
            {
                using var stream = File.OpenRead(rootWmoPath);
                using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
                long fileLen = stream.Length;

                // Log file header for debugging
                Console.WriteLine($"[DEBUG][WmoRootLoader] === File Header Dump (first 32 bytes) ===");
                byte[] headerBytes = new byte[Math.Min(32, fileLen)];
                stream.Position = 0;
                int bytesRead = stream.Read(headerBytes, 0, headerBytes.Length);
                Console.WriteLine($"Offset  | {string.Join(" ", Enumerable.Range(0, 16).Select(i => $"{i:X2}"))}");
                for (int i = 0; i < bytesRead; i += 16)
                {
                    int lineLength = Math.Min(16, bytesRead - i);
                    var line = headerBytes.Skip(i).Take(lineLength).ToArray();
                    Console.Write($"{i:X6} | {string.Join(" ", line.Select(b => $"{b:X2}"))}");
                    Console.Write(new string(' ', 3 * (16 - lineLength))); // Pad short lines
                    Console.Write(" | ");
                    Console.WriteLine(new string(line.Select(b => b >= 32 && b < 127 ? (char)b : '.').ToArray()));
                }
                stream.Position = 0;

                // --- Step 1: Detect version and find MOHD/MOGN chunks ---
                uint version = 0;
                long versionChunkPos = -1;
                int groupCount = -1;
                
                Console.WriteLine("\n[DEBUG][WmoRootLoader] Starting chunk scan to detect version and find MOHD/MOGN chunks...");
                Console.WriteLine($"[DEBUG][WmoRootLoader] File size: {fileLen} bytes (0x{fileLen:X8})");
                
                // Read first 4 bytes to detect version
                stream.Position = 0;
                byte[] first4Bytes = new byte[4];
                int bytesRead = stream.Read(first4Bytes, 0, 4);
                
                // In WoW files, chunk headers are always reversed (big-endian FourCC)
                Array.Reverse(first4Bytes);
                string first4AsString = Encoding.ASCII.GetString(first4Bytes);
                
                // Check for valid chunk IDs (reversed in file, so we reverse them back)
                if (first4AsString == "MVER")
                {
                    Console.WriteLine("[DEBUG][WmoRootLoader] Detected WMO v17+ file (MVER chunk)");
                }
                else if (first4AsString == "MPHD") // MPHD chunk (v14)
                {
                    Console.WriteLine("[DEBUG][WmoRootLoader] Detected WMO v14 file (MPHD chunk)");
                    Console.WriteLine("[DEBUG][WmoRootLoader] Detected WMO v14 file (MPHD chunk in big-endian)");
                    isBigEndian = true;
                }
                else
                {
                    Console.WriteLine($"[WARN][WmoRootLoader] Unknown file format, first 4 bytes: {BitConverter.ToString(first4Bytes)} ('{first4AsString}')");
                    // Try to find MVER or MPHD in first 1KB
                    byte[] headerSearch = new byte[1024];
                    stream.Position = 0;
                    int headerBytesRead = stream.Read(headerSearch, 0, 1024);
                    string headerAsString = Encoding.ASCII.GetString(headerSearch, 0, headerBytesRead);
                    
                    if (headerAsString.Contains("MVER"))
                    {
                        int pos = headerAsString.IndexOf("MVER");
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Found MVER at offset {pos}, assuming WMO v17+ (little-endian)");
                        isBigEndian = false;
                        stream.Position = pos;
                    }
                    else if (headerAsString.Contains("REVM"))
                    {
                        int pos = headerAsString.IndexOf("REVM");
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Found MVER (big-endian) at offset {pos}, assuming WMO v17+ (big-endian)");
                        isBigEndian = true;
                        stream.Position = pos;
                    }
                    else if (headerAsString.Contains("MPHD"))
                    {
                        int pos = headerAsString.IndexOf("MPHD");
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Found MPHD at offset {pos}, assuming WMO v14 (big-endian)");
                        isBigEndian = true;
                        stream.Position = pos;
                    }
                    else if (headerAsString.Contains("DHPM"))
                    {
                        int pos = headerAsString.IndexOf("DHPM");
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Found MPHD (little-endian) at offset {pos}, assuming WMO v14 (little-endian)");
                        isBigEndian = false;
                        stream.Position = pos;
                    }
                    else
                    {
                        Console.WriteLine("[WARN][WmoRootLoader] Could not identify WMO version, will try to scan chunks...");
                        stream.Position = 0; // Reset to start
                    }
                }
                
                // Now scan through chunks
                while (stream.Position + 8 <= fileLen)
                {
                    long chunkStart = stream.Position;
                    
                    // Read chunk ID (4 bytes)
                    byte[] chunkIdBytes = new byte[4];
                    bytesRead = stream.Read(chunkIdBytes, 0, 4);
                    if (bytesRead < 4)
                    {
                        Console.WriteLine($"[WARN][WmoRootLoader] Couldn't read chunk ID at position 0x{chunkStart:X8}");
                        break;
                    }
                    
                    // Read chunk size (4 bytes)
                    byte[] sizeBytes = new byte[4];
                    bytesRead = stream.Read(sizeBytes, 0, 4);
                    if (bytesRead < 4)
                    {
                        Console.WriteLine($"[WARN][WmoRootLoader] Couldn't read chunk size at position 0x{stream.Position-4:X8}");
                        break;
                    }
                    
                    // Handle endianness for chunk size
                    uint chunkSize;
                    if (isBigEndian)
                    {
                        Array.Reverse(sizeBytes);
                        chunkSize = BitConverter.ToUInt32(sizeBytes, 0);
                    }
                    else
                    {
                        chunkSize = BitConverter.ToUInt32(sizeBytes, 0);
                    }
                    
                    // Get chunk ID as string (handle endianness)
                    string chunkId;
                    if (isBigEndian)
                    {
                        Array.Reverse(chunkIdBytes);
                        chunkId = Encoding.ASCII.GetString(chunkIdBytes);
                        Array.Reverse(chunkIdBytes); // Restore original for display
                    }
                    else
                    {
                        chunkId = Encoding.ASCII.GetString(chunkIdBytes);
                    }
                    
                    // Log chunk info
                    Console.WriteLine($"[DEBUG][WmoRootLoader] Chunk '{chunkId}' at 0x{chunkStart:X8}, size: {chunkSize} (0x{chunkSize:X8})");
                    Console.WriteLine($"[DEBUG][WmoRootLoader]   Header: {BitConverter.ToString(chunkIdBytes)} {BitConverter.ToString(sizeBytes)}");
                    
                    switch (chunkId)
                    {
                        case "MVER":
                            // For v17+, read version
                            if (chunkSize >= 4)
                            {
                                version = reader.ReadUInt32();
                                Console.WriteLine($"[DEBUG][WmoRootLoader]   WMO Version: {version}");
                                versionChunkPos = chunkStart;
                                
                                // Log the full MVER chunk data for debugging
                                Console.WriteLine($"[DEBUG][WmoRootLoader]   MVER chunk data (hex): {BitConverter.ToString(reader.ReadBytes((int)chunkSize - 4)).Replace("-", " ")}");
                                stream.Position -= (chunkSize - 4); // Rewind to start of data
                            }
                            break;
                            
                        case "MOHD":
                            // Read group count (first 4 bytes of MOHD)
                            if (chunkSize >= 4)
                            {
                                // Log the full MOHD chunk data for debugging
                                byte[] mohdData = reader.ReadBytes((int)chunkSize);
                                Console.WriteLine($"[DEBUG][WmoRootLoader]   MOHD chunk data (hex): {BitConverter.ToString(mohdData).Replace("-", " ")}");
                                
                                // Parse group count (first 4 bytes)
                                groupCount = BitConverter.ToInt32(mohdData, 0);
                                Console.WriteLine($"[DEBUG][WmoRootLoader]   Found MOHD chunk with group count: {groupCount}");
                                
                                // Log more MOHD fields if available
                                if (chunkSize >= 0x40) // MOHD is typically 0x40 bytes in v17+
                                {
                                    int nPortals = BitConverter.ToInt32(mohdData, 4);
                                    int nLights = BitConverter.ToInt32(mohdData, 8);
                                    int nDoodadNames = BitConverter.ToInt32(mohdData, 12);
                                    int nDoodadDefs = BitConverter.ToInt32(mohdData, 16);
                                    int nDoodadSets = BitConverter.ToInt32(mohdData, 20);
                                    
                                    Console.WriteLine($"[DEBUG][WmoRootLoader]   MOHD details - Portals: {nPortals}, Lights: {nLights}, DoodadNames: {nDoodadNames}, DoodadDefs: {nDoodadDefs}, DoodadSets: {nDoodadSets}");
                                }
                                
                                // If we're in v14 mode but found MOHD, we might need to switch to v17
                                if (isV14)
                                {
                                    Console.WriteLine("[WARN][WmoRootLoader] Found MOHD chunk but we're in v14 mode. This might indicate a detection issue.");
                                    isV14 = false; // Switch to v17 mode
                                }
                                
                                // Don't skip the data, we already read it
                                continue;
                            }
                            break;
                            
                        case "MOGN":
                            // Read group names
                            mognData = reader.ReadBytes((int)chunkSize);
                            Console.WriteLine($"[DEBUG][WmoRootLoader]   Found MOGN chunk with {mognData.Length} bytes");
                            
                            // Log first 32 bytes of MOGN data
                            int logLength = Math.Min(32, mognData.Length);
                            Console.WriteLine($"[DEBUG][WmoRootLoader]   MOGN start (hex): {BitConverter.ToString(mognData, 0, logLength).Replace("-", " ")}");
                            
                            // Try to parse group names
                            try
                            {
                                var groupNameList = ReadNullTerminatedStrings(mognData);
                                Console.WriteLine($"[DEBUG][WmoRootLoader]   Parsed {groupNameList.Count} group names from MOGN");
                                if (groupNameList.Count > 0)
                                    Console.WriteLine($"[DEBUG][WmoRootLoader]   First group name: '{groupNameList[0]}'");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[WARN][WmoRootLoader] Failed to parse MOGN data: {ex.Message}");
                            }
                            
                            continue; // Already read the data
                            
                        case "MPHD":
                            // For v14, this is the header chunk
                            if (chunkSize >= 4)
                            {
                                isV14 = true;
                                Console.WriteLine("[DEBUG][WmoRootLoader]   Found MPHD chunk (v14 header)");
                                
                                // Log MPHD data
                                byte[] mphdData = reader.ReadBytes((int)chunkSize);
                                Console.WriteLine($"[DEBUG][WmoRootLoader]   MPHD chunk data (hex): {BitConverter.ToString(mphdData).Replace("-", " ")}");
                                
                                // In v14, group count is in MOHD which is a sub-chunk of MOMT
                                // We'll need to read MOMT to find MOHD
                                if (chunkSize >= 0x40) // MPHD is 0x40 bytes in v14
                                {
                                    // Already read the data, so just continue
                                    continue;
                                }
                            }
                            break;
                            
                        case "MOMT":
                            // In v14, MOHD is a sub-chunk of MOMT
                            if (isV14 && chunkSize > 0)
                            {
                                Console.WriteLine($"[DEBUG][WmoRootLoader]   Found MOMT chunk (v14) at 0x{chunkStart:X8}, size: {chunkSize} (0x{chunkSize:X8})");
                                long momtEnd = stream.Position + chunkSize;
                                
                                // Look for MOHD sub-chunk
                                while (stream.Position + 8 <= momtEnd)
                                {
                                    long subChunkStart = stream.Position;
                                    string subChunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                                    uint subChunkSize = reader.ReadUInt32();
                                    
                                    Console.WriteLine($"[DEBUG][WmoRootLoader]     Sub-chunk '{subChunkId}' at 0x{subChunkStart:X8}, size: {subChunkSize} (0x{subChunkSize:X8})");
                                    
                                    if (subChunkId == "MOHD" && subChunkSize >= 4)
                                    {
                                        // Read MOHD data
                                        byte[] mohdData = reader.ReadBytes((int)subChunkSize);
                                        Console.WriteLine($"[DEBUG][WmoRootLoader]     MOHD sub-chunk data (hex): {BitConverter.ToString(mohdData).Replace("-", " ")}");
                                        
                                        // Parse group count (first 4 bytes)
                                        groupCount = BitConverter.ToInt32(mohdData, 0);
                                        Console.WriteLine($"[DEBUG][WmoRootLoader]     Found MOHD sub-chunk with group count: {groupCount}");
                                        
                                        // Skip to end of MOHD sub-chunk
                                        stream.Position = subChunkStart + 8 + subChunkSize;
                                        break;
                                    }
                                    
                                    // Skip to next sub-chunk
                                    long skipAmount = (long)subChunkSize;
                                    if (stream.Position + skipAmount > momtEnd)
                                        skipAmount = momtEnd - stream.Position;
                                        
                                    if (skipAmount > 0)
                                        stream.Position += skipAmount;
                                }
                                
                                // Skip to end of MOMT chunk if needed
                                if (stream.Position < momtEnd)
                                    stream.Position = momtEnd;
                                    
                                continue;
                            }
                            break;
                            
                        default:
                            // For other chunks, just log their existence and skip them
                            Console.WriteLine($"[DEBUG][WmoRootLoader]   Skipping chunk '{chunkId}' (0x{chunkSize:X8} bytes)");
                            break;
                        }
                    }
                    
                    // Skip to next chunk (chunk data starts at chunkStart + 8)
                    long nextChunkPos = chunkStart + 8 + chunkSize;
                    if (nextChunkPos > stream.Length)
                    {
                        Console.WriteLine($"[WARN][WmoRootLoader] Invalid chunk size {chunkSize}, would go past end of file");
                        break;
                    }
                    stream.Position = nextChunkPos;
                }
                
                // If we found a group count, return it
                if (groupCount >= 0)
                {
                    Console.WriteLine($"[DEBUG][WmoRootLoader] Returning group count: {groupCount}");
                    return (groupCount, groupNames);
                }
                
                Console.WriteLine("[WARN][WmoRootLoader] No MOHD chunk found or couldn't read group count");
                return (-1, groupNames);

                // --- Step 2: Parse root chunks (v14: inside MOMO, v17+: top-level) ---
                stream.Position = 0;
                Console.WriteLine($"[DEBUG][WmoRootLoader] ===== Starting chunk parsing =====");
                Console.WriteLine($"[DEBUG][WmoRootLoader] File position: {stream.Position}, Length: {stream.Length}");
                Console.WriteLine($"[DEBUG][WmoRootLoader] isV14 = {isV14}");
                
                if (isV14)
                {
                    bool momoProcessedSuccessfully = false; // ADDED Flag
                    // --- v14: look for MOMO container ---
                    while (stream.Position + 8 <= fileLen)
                    {
                        long chunkStart = stream.Position;
                        var chunkIdBytes = reader.ReadBytes(4);
                        if (chunkIdBytes.Length < 4) break;
                        uint chunkId = BitConverter.ToUInt32(chunkIdBytes, 0);
                        string chunkIdStr = WoWToolbox.Core.WMO.FourCC.ToString(chunkIdBytes);
                        uint chunkSize = reader.ReadUInt32();
                        long chunkDataPos = stream.Position;
                        long chunkEnd = chunkDataPos + chunkSize;
                        if (chunkIdStr == "MOMO" || chunkIdStr == "OMOM")
                        {
                            Console.WriteLine($"[DEBUG][WmoRootLoader] v14: Found MOMO chunk at 0x{chunkStart:X}. Size: {chunkSize}. Data starts at 0x{chunkDataPos:X}, Ends at 0x{chunkEnd:X}");
                            bool foundMohdInMomo = false;
                            bool foundMognInMomo = false;
                            long momoStartPos = chunkDataPos; // Remember start for safety
                            stream.Position = momoStartPos;  // Ensure we start reading MOMO data
                            Console.WriteLine($"[DEBUG][WmoRootLoader] v14: Starting MOMO sub-chunk scan at 0x{stream.Position:X}");

                            // Loop through sub-chunks within MOMO
                            while (stream.Position + 8 <= chunkEnd)
                            {
                                long subChunkStart = stream.Position;
                                byte[] subChunkIdBytes = reader.ReadBytes(4);
                                uint subChunkId = BitConverter.ToUInt32(subChunkIdBytes, 0);
                                uint subChunkSize = reader.ReadUInt32();
                                long subChunkDataPos = stream.Position;
                                long subChunkEnd = subChunkDataPos + subChunkSize;

                                // Log sub-chunk details using hex format for ID
                                Console.WriteLine($"[DEBUG][WmoRootLoader][MOMO-InnerLoop] Found SubChunk ID: 0x{subChunkId:X8}, Size: {subChunkSize}, Start: {subChunkStart}, DataPos: {subChunkDataPos}, End: {subChunkEnd}");

                                if (subChunkId == 0x4D4F4844) // 'MOHD'
                                {
                                    Console.WriteLine("[DEBUG][WmoRootLoader][MOMO-InnerLoop] Found MOHD inside MOMO.");
                                    // FIX: Read nGroups directly from the stream at the start of the chunk's data
                                    reader.BaseStream.Position = subChunkDataPos;
                                    groupCount = (int)reader.ReadUInt32(); // Read the nGroups field (4 bytes) - ADDED CAST
                                    Console.WriteLine($"[DEBUG][WmoRootLoader][MOMO-InnerLoop] Read MOHD nGroups: {groupCount}");
                                    foundMohdInMomo = true;
                                    // Position will be advanced by reader, ensure it's correct before next read or skip
                                }
                                else if (subChunkId == 0x4E474F4D) // 'MOGN'
                                {
                                    Console.WriteLine($"[DEBUG][WmoRootLoader]     v14: Found MOGN inside MOMO at 0x{subChunkStart:X}");
                                    mognData = reader.ReadBytes((int)subChunkSize);
                                    foundMognInMomo = true;
                                    Console.WriteLine($"[DEBUG][WmoRootLoader]       v14: Read MOGN, Data Size = {mognData.Length}");
                                }
                                else
                                {
                                    // FIX: Log unknown chunk ID using hex format
                                    Console.WriteLine($"[DEBUG][WmoRootLoader][MOMO-InnerLoop] Skipping unknown SubChunk ID: 0x{subChunkId:X8}");
                                }

                                // Important: Advance stream position to the end of the current sub-chunk
                                stream.Position = subChunkEnd;
                                // FIX: Log using hex format for ID
                                Console.WriteLine($"[DEBUG][WmoRootLoader][MOMO-InnerLoop] Advanced stream position to: {stream.Position} after processing sub-chunk 0x{subChunkId:X8}");

                                // Safety break if position doesn't advance or goes too far (shouldn't happen with correct logic)
                                if (stream.Position >= chunkEnd || stream.Position < subChunkStart) // Use < subChunkStart for safety
                                {
                                    // FIX: Log using hex format for ID
                                    Console.WriteLine($"[ERROR][WmoRootLoader][MOMO-InnerLoop] Stream position invalid after processing sub-chunk 0x{subChunkId:X8}. Pos: {stream.Position}, SubChunkStart: {subChunkStart}, SubChunkEnd: {subChunkEnd}, MomoEnd: {chunkEnd}. Breaking inner loop.");
                                    break;
                                }

                                // Optimization: Exit if we found both needed chunks
                                if (foundMohdInMomo && foundMognInMomo)
                                {
                                    Console.WriteLine("[DEBUG][WmoRootLoader]     v14: Found both MOHD and MOGN in MOMO. Exiting sub-chunk loop.");
                                    break;
                                }
                            }
                            Console.WriteLine($"[DEBUG][WmoRootLoader] v14: Finished MOMO sub-chunk scan. Final position: 0x{stream.Position:X}. MOMO End: 0x{chunkEnd:X}");

                            if (foundMohdInMomo && foundMognInMomo)
                            {
                                momoProcessedSuccessfully = true;
                            }
                            else
                            {
                                 Console.WriteLine($"[WARN][WmoRootLoader] v14: Did not find both MOHD ({foundMohdInMomo}) and MOGN ({foundMognInMomo}) inside MOMO chunk.");
                            }

                            // Original fix: Ensure we are positioned *after* the entire MOMO chunk before breaking the outer loop
                            stream.Position = chunkEnd;
                            Console.WriteLine($"[DEBUG][WmoRootLoader] v14: Set stream position to end of MOMO (0x{stream.Position:X}) before break.");
                            Console.WriteLine($"[DEBUG][WmoRootLoader] v14: Finished processing MOMO chunk. Breaking OUTER loop. Success Flag: {momoProcessedSuccessfully}");
                            break; // Exit the main chunk loop since we found MOMO
                        }
                        stream.Position = chunkEnd; // Advance if current chunk wasn't MOMO
                    }

                    // --- Check v14 MOMO processing result --- 
                    if (momoProcessedSuccessfully)
                    {
                        // Process MOGN Data if found
                        if (mognData != null) 
                        {
                             groupNames = ReadNullTerminatedStrings(mognData);
                             Console.WriteLine($"[DEBUG][WmoRootLoader] Parsed {groupNames.Count} names from MOGN data.");
                        }
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Returning post-MOMO loop (Success): count={groupCount}, names={groupNames.Count}");
                        Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: V14_MOMO_SUCCESS !!!!!!");
                        return (groupCount, groupNames); // SUCCESSFUL RETURN
                    }
                    else
                    {
                        // This path is reached if the loop finished without finding/processing MOMO correctly
                        Console.WriteLine("[ERR][WmoRootLoader] Failed to find or successfully process MOMO chunk content in v14 file.");
                         Console.WriteLine($"[DEBUG][WmoRootLoader] Returning post-MOMO loop (Failure): count={-1}, names=0");
                        Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: V14_MOMO_FAILURE !!!!!!");
                        return (-1, new List<string>()); // FAILURE RETURN (v14)
                    }
                    else
                    {
                        // --- v17+ path ---
                    Console.WriteLine("[DEBUG][WmoRootLoader] Starting v17+ chunk scan...");
                    stream.Position = 0;
                    var v17Chunks = SimpleChunkReader.ReadAllChunks(stream);
                    Console.WriteLine($"[DEBUG][WmoRootLoader] Found {v17Chunks.Count} top-level chunks in v17 WMO");
                    
                    // Log all chunks for debugging
                    foreach (var chunk in v17Chunks)
                    {
                        string dataPreview = chunk.Data.Length > 0 
                            ? $"Data[0]={chunk.Data[0]:X2}..." 
                            : "No data";
                        Console.WriteLine($"  - Chunk '{chunk.Id}' at 0x{chunk.Offset:X8}, size={chunk.Size} ({dataPreview})");
                    }
                    
                            var mohdChunk = v17Chunks.FirstOrDefault(c => c.Id == "MOHD");
                        if (mohdChunk != null)
                        {
                            Console.WriteLine($"[DEBUG][WmoRootLoader] v17+: Found MOHD chunk at offset 0x{mohdChunk.Offset:X8}, size: {mohdChunk.Size} bytes");
                            
                            if (mohdChunk.Data != null)
                            {
                                Console.WriteLine($"[DEBUG][WmoRootLoader] v17+: MOHD data length: {mohdChunk.Data.Length} bytes");
                                if (mohdChunk.Data.Length >= 4)
                                {
                                    groupCount = BitConverter.ToInt32(mohdChunk.Data, 0);
                                    Console.WriteLine($"[DEBUG][WmoRootLoader] v17+: Found MOHD chunk with group count: {groupCount}");
                                }
                                else
                                {
                                    Console.WriteLine($"[WARN][WmoRootLoader] v17+: MOHD chunk data too small: {mohdChunk.Data.Length} bytes (need at least 4)");
                                }
                            }
                            else
                            {
                                Console.WriteLine("[WARN][WmoRootLoader] v17+: MOHD chunk data is null");
                            }
                        }
                        else
                        {
                            Console.WriteLine("[WARN][WmoRootLoader] v17+: MOHD chunk not found in top-level chunks");
                            Console.WriteLine("[DEBUG][WmoRootLoader] Available chunk IDs: " + string.Join(", ", v17Chunks.Select(c => c.Id).Distinct()));
                        }        
                                // Log more MOHD fields if available
                                if (chunkSize >= 0x40) // MOHD is typically 0x40 bytes in v17+
                                {
                                    int nPortals = BitConverter.ToInt32(mohdData, 4);
                                    int nLights = BitConverter.ToInt32(mohdData, 8);
{{ ... }}
                        return (-1, new List<string>()); // FAILURE RETURN (v14)
                    }
                    else
                    {
                        // --- v17+ path ---
                    Console.WriteLine("[DEBUG][WmoRootLoader] ===== Starting v17+ chunk scan =====");
                    Console.WriteLine($"[DEBUG][WmoRootLoader] Stream position before ReadAllChunks: {stream.Position}");
                    Console.WriteLine($"[DEBUG][WmoRootLoader] Stream length: {stream.Length}");
                    
                    // Log first 32 bytes of the stream
                    long originalPosition = stream.Position;
                    byte[] headerBytes = new byte[32];
                    int headerBytesRead = stream.Read(headerBytes, 0, headerBytes.Length);
                    stream.Position = originalPosition;
                    
                    Console.WriteLine($"[DEBUG][WmoRootLoader] First {headerBytesRead} bytes of stream (hex): {BitConverter.ToString(headerBytes, 0, headerBytesRead).Replace("-", " ")}");
                    Console.WriteLine($"[DEBUG][WmoRootLoader] First {headerBytesRead} bytes of stream (ASCII): {new string(headerBytes.Take(headerBytesRead).Select(b => b >= 32 && b < 127 ? (char)b : '.').ToArray())}");
                    
                    var v17Chunks = SimpleChunkReader.ReadAllChunks(stream);
                    Console.WriteLine($"[DEBUG][WmoRootLoader] Found {v17Chunks.Count} top-level chunks in v17+ WMO");
                                    using var br2 = new BinaryReader(ms);
                                    uint nTextures = br2.ReadUInt32();
                                    groupCount = br2.ReadInt32();
                                    Console.WriteLine($"[DEBUG][WmoRootLoader] (v17) MOHD nTextures={nTextures}, groupCount={groupCount}");
                                    
                                    // Dump MOHD data for debugging
                                    Console.WriteLine($"[DEBUG][WmoRootLoader] MOHD data: {BitConverter.ToString(mohd.Data.Take(32).ToArray())}...");
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"[ERR][WmoRootLoader] Error reading MOHD chunk: {ex.Message}");
                                }
                            }
                            else
                            {
                                Console.WriteLine($"[WARN][WmoRootLoader] MOHD chunk size {mohd.Size} is too small (need at least 8 bytes).");
                            }
                        }
                        else
                        {
                            Console.WriteLine("[ERR][WmoRootLoader] v17+: MOHD chunk not found in top-level chunks.");
                        }

                        var mogn = v17Chunks.FirstOrDefault(c => c.Id == "MOGN");
                        if (mogn != null)
                        {
                            mognData = mogn.Data;
                            Console.WriteLine($"[DEBUG][WmoRootLoader] (v17) MOGN found at 0x{mogn.Offset:X8}, size={mogn.Size}");
                            
                            // Dump start of MOGN data
                            int dumpLength = Math.Min(32, mognData.Length);
                            Console.WriteLine($"[DEBUG][WmoRootLoader] MOGN data start: {BitConverter.ToString(mognData.Take(dumpLength).ToArray())}...");
                        }
                        else
                        {
                            Console.WriteLine("[WARN][WmoRootLoader] v17+: MOGN chunk not found in top-level chunks.");
                        }

                        if (groupCount == -1 || mognData == null)
                        {
                            Console.WriteLine($"[ERR][WmoRootLoader] v17+: Failed to retrieve MOHD (found={mohd != null}) or MOGN (found={mogn != null}).");
                            Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: V17_PLUS_MISSING_CHUNKS !!!!!!");
                            return (-1, new List<string>());
                        }
                        else
                        {
                        Console.WriteLine("[DEBUG][WmoRootLoader] v17+: Successfully found MOHD & MOGN via SimpleChunkReader.");
                    }

                    // --- v17+ Post-processing check ---
                    if (groupCount == -1)
                    {
                        Console.WriteLine("[ERR][WmoRootLoader] v17+: MOHD chunk not found or group count could not be read.");
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Returning from v17+ path (MOHD not found): count={-1}, names=0");
                        Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: V17_PLUS_MOHD_FAILURE !!!!!!");
                        return (-1, groupNames); // FAILURE RETURN (v17+)
                    }
                    if (mognData != null)
                    {
                        try {
                            groupNames = ReadNullTerminatedStrings(mognData);
                            Console.WriteLine($"[DEBUG][WmoRootLoader] v17+: Parsed {groupNames.Count} names from MOGN data.");
                            if (groupNames.Count != groupCount)
                            {
                                Console.WriteLine($"[WARN][WmoRootLoader] v17+ Mismatch: MOHD group count ({groupCount}) != MOGN name count ({groupNames.Count}).");
                            }
                            Console.WriteLine($"[DEBUG][WmoRootLoader] Returning v17+ path (Success): count={groupCount}, names={groupNames.Count}");
                            Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: V17_PLUS_SUCCESS !!!!!!");
                            return (groupCount, groupNames); // SUCCESSFUL RETURN (v17+)
                        } catch (Exception ex) {
                            Console.WriteLine($"[ERR][WmoRootLoader] v17+ Exception during ReadNullTerminatedStrings: {ex.Message}");
                            Console.WriteLine($"[DEBUG][WmoRootLoader] Returning from v17+ path (String read error): count={-1}, names=0");
                            Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: V17_PLUS_STRING_READ_ERROR !!!!!!");
                            return (-1, new List<string>()); // FAILURE RETURN (v17+)
                        }
                    }
                    else
                    {
                        Console.WriteLine("[ERR][WmoRootLoader] v17+: MOGN chunk not found. Cannot determine group file names.");
                        groupNames.Clear();
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Returning from v17+ path (MOGN not found): count={groupCount}, names=0");
                        Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: V17_PLUS_MOGN_NOT_FOUND !!!!!!");
                        return (groupCount, groupNames); // Return count but no names
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR][WmoRootLoader] Failed to load group info from {rootWmoPath}: {ex.Message}");
                Console.WriteLine($"[DEBUG][WmoRootLoader] EXCEPTION CAUGHT: {ex.GetType().Name} - {ex.Message}"); 
                Console.WriteLine($"[DEBUG][WmoRootLoader] StackTrace: {ex.StackTrace}"); 
                groupNames.Clear(); // Ensure empty list on error
                Console.WriteLine($"[DEBUG][WmoRootLoader] Returning from CATCH block: count={-1}, names=0"); 
                Console.WriteLine("!!!!!! LoadGroupInfo EXIT PATH: EXCEPTION !!!!!!");
                return (-1, groupNames); // Return count -1 and empty list on error
            }

            #else
            // TEMP stub to keep build compiling while detailed implementation is refactored
            return (-1, new List<string>());
#endif
            // This final return should ideally not be reached if all paths return within try/catch
            // Console.WriteLine($"[DEBUG][WmoRootLoader] Returning NORMALLY at end: count={groupCount}, names={groupNames.Count}"); 
            // return (groupCount, groupNames);
        }

        // Utility: List all top-level (and MOMO sub-) chunks in a root WMO file
        public static void ListChunks(string filePath)
        {
            using var stream = File.OpenRead(filePath);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            long fileLen = stream.Length;
            Console.WriteLine($"[RootChunkLister] Listing top-level chunks in: {filePath}");
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = WoWToolbox.Core.WMO.FourCC.ToString(chunkIdBytes);
                uint chunkSize = reader.ReadUInt32();
                long chunkDataPos = stream.Position;
                long chunkEnd = chunkDataPos + chunkSize;
                Console.WriteLine($"  Offset 0x{chunkStart:X6}: Chunk '{chunkIdStr}' Size: {chunkSize} (0x{chunkSize:X})");
                if (chunkIdStr == "MOMO")
                {
                    // List subchunks inside MOMO
                    long momoEnd = chunkDataPos + chunkSize;
                    long savePos = stream.Position;
                    Console.WriteLine($"    [MOMO] Listing subchunks:");
                    while (stream.Position + 8 <= momoEnd)
                    {
                        long subChunkStart = stream.Position;
                        var subChunkIdBytes = reader.ReadBytes(4);
                        if (subChunkIdBytes.Length < 4) break;
                        string subChunkIdStr = new string(subChunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                        uint subChunkSize = reader.ReadUInt32();
                        long subChunkDataPos = stream.Position;
                        long subChunkEnd = subChunkDataPos + subChunkSize;
                        Console.WriteLine($"      Offset 0x{subChunkStart:X6}: Subchunk '{subChunkIdStr}' Size: {subChunkSize} (0x{subChunkSize:X})");
                        stream.Position = subChunkEnd;
                    }
                    stream.Position = savePos; // Restore after MOMO
                }
                stream.Position = chunkEnd;
            }
            Console.WriteLine($"[RootChunkLister] End of chunk listing at offset 0x{stream.Position:X6}");
        }

        /// <summary>
        /// Loads the unified WMO texturing model (textures, materials, group names, group info) from a root WMO file (v14 or v17+).
        /// </summary>
        public static (WmoTextureBlock textures, WmoMaterialBlock materials, WmoGroupNameBlock groupNames, WmoGroupInfoBlock groupInfo)
            LoadTexturingModel(string rootWmoPath)
        {
            byte[] motxData = null;
            List<MOMT> momtList = new();
            byte[] mognData = null;
            List<MOGI> mogiList = new();

            using var stream = File.OpenRead(rootWmoPath);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
            long fileLen = stream.Length;

            // --- Step 1: Detect version ---
            uint version = 0;
            stream.Position = 0;
            while (stream.Position + 8 <= fileLen)
            {
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                uint chunkId = BitConverter.ToUInt32(chunkIdBytes, 0);
                uint chunkSize = reader.ReadUInt32();
                long chunkEnd = stream.Position + chunkSize;
                if (chunkId == 0x5245564D) // 'MVER'
                {
                    if (chunkSize == 4)
                        version = reader.ReadUInt32();
                    stream.Position = chunkEnd;
                    break;
                }
                else
                {
                    stream.Position = chunkEnd;
                }
            }
            bool isV14 = (version == 14);

            // --- Step 2: Scan for relevant chunks ---
            stream.Position = 0;
            if (isV14)
            {
                // v14: scan MOMO subchunks
                while (stream.Position + 8 <= fileLen)
                {
                    var chunkIdBytes = reader.ReadBytes(4);
                    if (chunkIdBytes.Length < 4) break;
                    uint chunkId = BitConverter.ToUInt32(chunkIdBytes, 0);
                    uint chunkSize = reader.ReadUInt32();
                    long chunkDataPos = stream.Position;
                    long chunkEnd = chunkDataPos + chunkSize;
                    if (chunkId == 0x4F4D4F4D) // 'MOMO'
                    {
                        long momoEnd = chunkDataPos + chunkSize;
                        while (stream.Position + 8 <= momoEnd)
                        {
                            var subChunkIdBytes = reader.ReadBytes(4);
                            if (subChunkIdBytes.Length < 4) break;
                            uint subChunkId = BitConverter.ToUInt32(subChunkIdBytes, 0);
                            uint subChunkSize = reader.ReadUInt32();
                            long subChunkDataPos = stream.Position;
                            long subChunkEnd = subChunkDataPos + subChunkSize;
                            if (subChunkId == 0x4D4F5458) // 'MOTX'
                                motxData = reader.ReadBytes((int)subChunkSize);
                            else if (subChunkId == 0x4D4F4D54) // 'MOMT'
                            {
                                int count = (int)(subChunkSize / 64);
                                for (int i = 0; i < count; i++)
                                    momtList.Add(MOMT.FromBinaryReader(reader));
                            }
                            else if (subChunkId == 0x4E474F4D) // 'MOGN'
                                mognData = reader.ReadBytes((int)subChunkSize);
                            else if (subChunkId == 0x4D4F4749) // 'MOGI'
                            {
                                int count = (int)(subChunkSize / 32);
                                for (int i = 0; i < count; i++)
                                    mogiList.Add(MOGI.FromBinaryReader(reader));
                            }
                            else
                                stream.Position = subChunkEnd;
                            if (stream.Position < subChunkEnd)
                                stream.Position = subChunkEnd;
                        }
                        break; // Only one MOMO
                    }
                    else
                        stream.Position = chunkEnd;
                }
            }
            else
            {
                // v17+: scan top-level chunks
                while (stream.Position + 8 <= fileLen)
                {
                    var chunkIdBytes = reader.ReadBytes(4);
                    if (chunkIdBytes.Length < 4) break;
                    uint chunkId = BitConverter.ToUInt32(chunkIdBytes, 0);
                    uint chunkSize = reader.ReadUInt32();
                    long chunkDataPos = stream.Position;
                    long chunkEnd = chunkDataPos + chunkSize;
                    if (chunkId == 0x4D4F5458) // 'MOTX'
                        motxData = reader.ReadBytes((int)chunkSize);
                    else if (chunkId == 0x4D4F4D54) // 'MOMT'
                    {
                        int count = (int)(chunkSize / 64);
                        for (int i = 0; i < count; i++)
                            momtList.Add(MOMT.FromBinaryReader(reader));
                    }
                    else if (chunkId == 0x4E474F4D) // 'MOGN'
                        mognData = reader.ReadBytes((int)chunkSize);
                    else if (chunkId == 0x4D4F4749) // 'MOGI'
                    {
                        int count = (int)(chunkSize / 32);
                        for (int i = 0; i < count; i++)
                            mogiList.Add(MOGI.FromBinaryReader(reader));
                    }
                    else
                        stream.Position = chunkEnd;
                    if (stream.Position < chunkEnd)
                        stream.Position = chunkEnd;
                }
            }

            // --- Step 3: Build model ---
            var textures = motxData != null ? WmoTexturingModelFactory.FromMotx(motxData) : new WmoTextureBlock();
            var materials = WmoTexturingModelFactory.FromMomt(momtList);
            var groupNames = mognData != null ? WmoTexturingModelFactory.FromMogn(mognData) : new WmoGroupNameBlock();
            var groupInfo = WmoTexturingModelFactory.FromMogi(mogiList);
            return (textures, materials, groupNames, groupInfo);
        }
    }
} 