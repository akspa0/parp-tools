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

                // --- Step 1: Detect version ---
                uint version = 0;
                long versionChunkPos = -1;
                stream.Position = 0;
                while (stream.Position + 8 <= fileLen)
                {
                    long chunkStart = stream.Position;
                    var chunkIdBytes = reader.ReadBytes(4);
                    if (chunkIdBytes.Length < 4) break;
                    uint chunkId = BitConverter.ToUInt32(chunkIdBytes, 0);
                    string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                    uint chunkSize = reader.ReadUInt32();
                    long chunkDataPos = stream.Position;
                    long chunkEnd = chunkDataPos + chunkSize;
                    // DEBUG: Print all chunk headers as parsed during version detection
                    Console.WriteLine($"[DEBUG][WmoRootLoader] (version detect) Chunk at 0x{chunkStart:X}: '{chunkIdStr}' Size: {chunkSize} (0x{chunkSize:X})");
                    Console.WriteLine($"[DEBUG][WmoRootLoader]   Raw chunkIdBytes: {BitConverter.ToString(chunkIdBytes)}  chunkId (hex): 0x{chunkId:X8}");
                    // NOTE: WoW chunk IDs are stored as 4 ASCII bytes in little-endian order in the file.
                    // For example, 'MVER' is stored as 'REVM' (0x5245564D), 'MOMO' as 'OMOM' (0x4F4D4F4D), 'MOHD' as 'DHOM' (0x4D4F4844), 'MOGN' as 'NGOM' (0x4E474F4D).
                    if (chunkId == 0x5245564D) // 'MVER' (little-endian: 'REVM')
                    {
                        versionChunkPos = chunkStart;
                        if (chunkSize == 4)
                        {
                            version = reader.ReadUInt32();
                            isV14 = (version == 14);
                            Console.WriteLine($"[DEBUG][WmoRootLoader] Detected MVER: {version} (isV14={isV14})");
                        }
                        else
                        {
                            Console.WriteLine($"[WARN][WmoRootLoader] MVER chunk size {chunkSize} is unexpected (expected 4). Skipping version read.");
                        }
                        stream.Position = chunkEnd;
                        break; // Found version, can proceed
                    }
                    else
                    {
                        stream.Position = chunkEnd;
                    }
                }

                // --- Step 2: Parse root chunks (v14: inside MOMO, v17+: top-level) ---
                stream.Position = 0;
                Console.WriteLine($"!!!!!! Pre-check isV14 = {isV14} !!!!!!");
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
                        string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                        uint chunkSize = reader.ReadUInt32();
                        long chunkDataPos = stream.Position;
                        long chunkEnd = chunkDataPos + chunkSize;
                        if (chunkId == 0x4F4D4F4D) // 'MOMO' (little-endian: 'OMOM')
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
                }
                else
                {
                    bool foundMohd_v17 = false;
                    bool foundMogn_v17 = false;
                    // --- v17+: parse top-level chunks as before ---
                    while (stream.Position + 8 <= fileLen)
                    {
                        long chunkStart = stream.Position;
                        var chunkIdBytes = reader.ReadBytes(4);
                        if (chunkIdBytes.Length < 4) break;
                        uint chunkId = BitConverter.ToUInt32(chunkIdBytes, 0);
                        string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                        uint chunkSize = reader.ReadUInt32();
                        long chunkDataPos = stream.Position;
                        long chunkEnd = chunkDataPos + chunkSize;
                        if (chunkEnd > fileLen)
                        {
                            Console.WriteLine($"[ERR][WmoRootLoader] Chunk '{chunkIdStr}' (0x{chunkId:X8}) at 0x{chunkStart:X} claims size {chunkSize}, which exceeds file length {fileLen}. Stopping parse.");
                            break;
                        }
                        if (chunkId == 0x4D4F4844) // 'MOHD' (little-endian: 'DHOM')
                        {
                            if (chunkSize >= 8)
                            {
                                stream.Position += 4; // Skip nTextures
                                groupCount = reader.ReadInt32();
                                Console.WriteLine($"[DEBUG][WmoRootLoader] Found MOHD, Group Count: {groupCount}");
                                stream.Position = chunkEnd;
                                foundMohd_v17 = true; // Mark v17 MOHD found
                            }
                            else
                            {
                                Console.WriteLine($"[WARN][WmoRootLoader] MOHD chunk size {chunkSize} is too small. Cannot read group count.");
                                stream.Position = chunkEnd;
                            }
                        }
                        else if (chunkId == 0x4E474F4D) // 'MOGN' (little-endian: 'NGOM')
                        {
                            Console.WriteLine($"[DEBUG][WmoRootLoader] Found MOGN, Size: {chunkSize}");
                            mognData = reader.ReadBytes((int)chunkSize);
                            foundMogn_v17 = true; // Mark v17 MOGN found
                            stream.Position = chunkEnd;
                        }
                        else
                        {
                            stream.Position = chunkEnd;
                        }
                         // Break v17 loop if both found
                         if (foundMohd_v17 && foundMogn_v17)
                         {
                            Console.WriteLine("[DEBUG][WmoRootLoader] v17+: Found both MOHD and MOGN. Exiting loop.");
                            break;
                         }
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
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
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