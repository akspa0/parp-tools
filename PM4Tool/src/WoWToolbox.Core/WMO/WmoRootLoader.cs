using System;
using System.Collections.Generic;
using System.IO;
using System.Linq; // Added for Reverse
using System.Text;

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

            try
            {
                using var stream = File.OpenRead(rootWmoPath);
                using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
                long fileLen = stream.Length;

                while (stream.Position + 8 <= fileLen) // Need 8 bytes for chunk header
                {
                    long chunkStart = stream.Position;
                    var chunkIdBytes = reader.ReadBytes(4);
                    if (chunkIdBytes.Length < 4) break;
                    // Read chunk ID without reversing for direct comparison with uint
                    uint chunkId = BitConverter.ToUInt32(chunkIdBytes, 0);
                    string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray()); // For logging
                    uint chunkSize = reader.ReadUInt32();
                    long chunkDataPos = stream.Position;
                    long chunkEnd = chunkDataPos + chunkSize;

                    if (chunkEnd > fileLen)
                    {
                        Console.WriteLine($"[ERR][WmoRootLoader] Chunk '{chunkIdStr}' (0x{chunkId:X8}) at 0x{chunkStart:X} claims size {chunkSize}, which exceeds file length {fileLen}. Stopping parse.");
                        break;
                    }

                    if (chunkId == 0x4D4F4844) // 'MOHD' (little-endian)
                    {
                        if (chunkSize >= 8) // Need at least nTextures and nGroups fields
                        {
                            stream.Position += 4; // Skip nTextures
                            groupCount = reader.ReadInt32();
                            Console.WriteLine($"[DEBUG][WmoRootLoader] Found MOHD, Group Count: {groupCount}");
                            // Don't break, continue searching for MOGN
                            stream.Position = chunkEnd; // Move to the end of MOHD chunk
                        }
                        else
                        {
                             Console.WriteLine($"[WARN][WmoRootLoader] MOHD chunk size {chunkSize} is too small. Cannot read group count.");
                             stream.Position = chunkEnd;
                        }
                    }
                    else if (chunkId == 0x4D4F474E) // 'MOGN' (little-endian)
                    {
                        Console.WriteLine($"[DEBUG][WmoRootLoader] Found MOGN, Size: {chunkSize}");
                        mognData = reader.ReadBytes((int)chunkSize);
                        // Can break here if we assume MOGN appears after MOHD, or let it continue
                        // Let's break for efficiency once both MOHD and MOGN are found (if MOHD was found)
                        if (groupCount != -1) break;
                        stream.Position = chunkEnd; // Ensure position is correct if we didn't break
                    }
                    else
                    {
                        // Skip other chunks
                        stream.Position = chunkEnd;
                    }
                }

                // Post-processing
                if (groupCount == -1)
                {
                    Console.WriteLine("[ERR][WmoRootLoader] MOHD chunk not found or group count could not be read.");
                    return (-1, groupNames); // Return count -1 and empty list
                }

                if (mognData != null)
                {
                    groupNames = ReadNullTerminatedStrings(mognData);
                    Console.WriteLine($"[DEBUG][WmoRootLoader] Extracted {groupNames.Count} names from MOGN.");

                    if (groupNames.Count != groupCount)
                    {
                        Console.WriteLine($"[WARN][WmoRootLoader] Mismatch: MOHD group count ({groupCount}) != MOGN name count ({groupNames.Count}). Using names from MOGN.");
                    }
                }
                else
                {
                    Console.WriteLine("[ERR][WmoRootLoader] MOGN chunk not found. Cannot determine group file names.");
                    groupNames.Clear();
                    // Return the count from MOHD, but an empty list for names
                    return (groupCount, groupNames);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR][WmoRootLoader] Failed to load group info from {rootWmoPath}: {ex.Message}");
                groupNames.Clear(); // Ensure empty list on error
                return (-1, groupNames); // Return count -1 and empty list on error
            }

            return (groupCount, groupNames);
        }
    }
} 