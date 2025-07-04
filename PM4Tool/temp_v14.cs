using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Numerics;
using WoWToolbox.Core.WMO; // For WmoGroupMesh, WmoRootLoader, etc.
using Warcraft.NET.Files.BLP;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Png;

namespace WoWToolbox.WmoV14Converter
{
    public static class WmoV14ToV17Converter
    {
        // Persistent log file path (timestamped)
        private static readonly string LogFilePath = $"conversion-{DateTime.Now:yyyyMMdd-HHmmss}.log";
        private static readonly object LogLock = new object();
        private static bool LogFileInitialized = false;

        public static void Log(string message)
        {
            string line = $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}";
            Console.WriteLine(line);
            lock (LogLock)
            {
                if (!LogFileInitialized)
                {
                    File.WriteAllText(LogFilePath, "# WoWToolbox v14 Converter Log\n");
                    LogFileInitialized = true;
                }
                File.AppendAllText(LogFilePath, line + "\n");
            }
        }
        public static void LogException(string context, Exception ex)
        {
            Log($"ERROR in {context}: {ex.Message}\n{ex.StackTrace}");
        }
        public static string GetLogFilePath() => LogFilePath;

        /// <summary>
        /// Parse the input v14 WMO and export the first group's geometry as a Wavefront OBJ file using a v14-specific parser.
        /// </summary>
        public static void ExportFirstGroupAsObj(string inputWmo, string outputObj)
        {
            Log($"[OBJ] Opening v14 WMO: {inputWmo}");
            using var stream = File.OpenRead(inputWmo);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
            long fileLen = stream.Length;
            stream.Position = 0;

            // --- Step 1: Log all top-level chunk IDs, offsets, and sizes ---
            var chunkHeaders = new List<(string id, long offset, uint size)>();
            stream.Position = 0;
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = Encoding.ASCII.GetString(chunkIdBytes.Reverse().ToArray());
                uint chunkSize = reader.ReadUInt32();
                Log($"[WMO] Top-level chunk: ID='{chunkIdStr}' Offset=0x{chunkStart:X} Size={chunkSize}");
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
                if (chunkIdStr == "MOMO")
                {
                    // Parse subchunks inside MOMO
                    long momoDataStart = chunkStart + 8;
                    stream.Position = momoDataStart;
                    byte[] momoData = reader.ReadBytes((int)chunkSize);
                    using var momoStream = new MemoryStream(momoData);
                    using var momoReader = new BinaryReader(momoStream);
                    while (momoStream.Position + 8 <= momoStream.Length)
                    {
                        long subChunkStart = momoStream.Position;
                        var subChunkIdBytes = momoReader.ReadBytes(4);
                        if (subChunkIdBytes.Length < 4) break;
                        string subChunkIdStr = Encoding.ASCII.GetString(subChunkIdBytes.Reverse().ToArray());
                        uint subChunkSize = momoReader.ReadUInt32();
                        Log($"[WMO][MOMO] Subchunk: ID='{subChunkIdStr}' Offset=0x{subChunkStart:X} Size={subChunkSize}");
                        // Optionally: add to chunkHeaders or process as needed
                        momoStream.Position = subChunkStart + 8 + subChunkSize;
                    }
                    // After parsing MOMO, move stream to the end of the chunk
                    stream.Position = chunkStart + 8 + chunkSize;
                }
                else
                {
                    stream.Position = chunkStart + 8 + chunkSize;
                }
            }

            // --- Step 2: Find the first group region (MOGP header + subchunks) ---
            int mver = 0;
            long? mogpOffset = null;
            long? nextMogpOffset = null;
            for (int i = 0; i < chunkHeaders.Count; i++)
            {
                var (chunkIdStr, chunkStart, chunkSize) = chunkHeaders[i];
                if (chunkIdStr == "MVER")
                {
                    stream.Position = chunkStart + 8;
                    mver = reader.ReadInt32();
                }
                if (chunkIdStr == "MOGP" && mogpOffset == null)
                {
                    mogpOffset = chunkStart;
                    // Find the next MOGP chunk (if any)
                    for (int j = i + 1; j < chunkHeaders.Count; j++)
                    {
                        if (chunkHeaders[j].id == "MOGP")
                        {
                            nextMogpOffset = chunkHeaders[j].offset;
                            break;
                        }
                    }
                }
            }
            if (mver != 14)
            {
                Log($"[OBJ] Not a v14 WMO (MVER={mver})");
                return;
            }
            if (mogpOffset == null)
            {
                Log("[OBJ] No MOGP group found in WMO. Top-level chunks found:");
                foreach (var (id, offset, size) in chunkHeaders)
                    Log($"  ID='{id}' Offset=0x{offset:X} Size={size}");
                return;
            }
            long groupStart = mogpOffset.Value;
            long groupEnd = nextMogpOffset ?? fileLen;
            long groupLen = groupEnd - groupStart;
            stream.Position = groupStart;
            byte[] groupData = reader.ReadBytes((int)groupLen);
            Log($"[WMO] Extracted group region: start=0x{groupStart:X}, end=0x{groupEnd:X}, len={groupLen}");

            // --- Step 3: Print hex dump of first 64 bytes of groupData ---
            Log("[DEBUG] Hex dump of first 64 bytes of groupData:");
            int dumpLen = Math.Min(64, groupData.Length);
            for (int i = 0; i < dumpLen; i += 16)
            {
                var line = groupData.Skip(i).Take(Math.Min(16, dumpLen - i)).ToArray();
                Log($"  {i:X4}: ");
                foreach (var b in line) Log($"{b:X2} ");
                Log($"  ");
                foreach (var b in line) Log(((char)(char.IsControl((char)b) ? '.' : (char)b)).ToString());
                Log($"");
            }

            // --- Step 4: Parse the group using the v14-specific parser ---
            WmoGroupMesh mesh = null;
            try
            {
                mesh = LoadFromV14GroupChunk(groupData);
                Log("[OBJ] Parsed group using v14-specific parser");
            }
            catch (Exception ex)
            {
                Log($"[OBJ] ERROR: Failed to parse group with v14-specific parser: {ex.Message}");
                return;
            }

            // --- Step 5: Export OBJ using Core's SaveToObj ---
            try
            {
                WmoGroupMesh.SaveToObj(mesh, outputObj);
                Log($"[OBJ] Wrote OBJ file: {outputObj}");
            }
            catch (Exception ex)
            {
                Log($"[OBJ] ERROR: Failed to export OBJ: {ex.Message}");
            }
            Log($"");
        }

        /// <summary>
        /// Parse a v14 group chunk and return a populated WmoGroupMesh.
        /// </summary>
        public static WmoGroupMesh LoadFromV14GroupChunk(byte[] groupData)
        {
            var mesh = new WmoGroupMesh();
            using var ms = new MemoryStream(groupData);
            using var br = new BinaryReader(ms);
            long groupLen = ms.Length;
            ms.Position = 0;
            // --- Skip the MOGP header (0x40 bytes for v14) ---
            const int MOGP_HEADER_SIZE = 0x40;
            if (groupLen < MOGP_HEADER_SIZE)
            {
                Log($"[v14] ERROR: Group data too small for MOGP header (len={groupLen})");
                return mesh;
            }
            var mogpHeader = br.ReadBytes(MOGP_HEADER_SIZE);
            // --- Scan forward for the first valid subchunk header ---
            string[] validSubchunks = { "MOPY", "MOVT", "MONR", "MOTV", "MOVI", "MOBA", "MLIQ", "MOCV", "MOTX", "MOGN", "MOMT", "MOLV", "MOIN", "MODR", "MOBN", "MOBR", "MOCV", "MOLM", "MOLD" };
            int scanOffset = MOGP_HEADER_SIZE;
            int foundOffset = -1;
            string foundId = "";
            for (int off = MOGP_HEADER_SIZE; off < MOGP_HEADER_SIZE + 256 && off + 4 <= groupLen; off += 4)
            {
                ms.Position = off;
                var idBytes = br.ReadBytes(4);
                if (idBytes.Length < 4) break;
                string id = Encoding.ASCII.GetString(idBytes.Reverse().ToArray());
                if (validSubchunks.Contains(id))
                {
                    foundOffset = off;
                    foundId = id;
                    break;
                }
            }
            if (foundOffset == -1)
            {
                Log($"[v14] ERROR: No valid subchunk header found within 256 bytes after MOGP header. Aborting parse.");
                return mesh;
            }
            Log($"[v14] Found first subchunk '{foundId}' at offset 0x{foundOffset:X}");
            ms.Position = foundOffset;
            // Map of subchunk name to (offset, size, data)
            var subchunks = new Dictionary<string, (long offset, uint size, byte[] data)>();
            var subchunkList = new List<(string id, long offset, uint size)>();
            int chunkIdx = 0;
            while (ms.Position + 8 <= groupLen)
            {
                // --- End-of-region checks ---
                long bytesLeft = groupLen - ms.Position;
                if (bytesLeft < 8)
                {
                    Log($"[v14][END] Fewer than 8 bytes left in group region (bytesLeft={bytesLeft}). Ending subchunk parse cleanly.");
                    break;
                }
                // Peek next 4 bytes
                long peekPos = ms.Position;
                var peekBytes = br.ReadBytes(4);
                ms.Position = peekPos; // reset
                bool allZero = peekBytes.All(b => b == 0);
                bool allNonPrintable = peekBytes.All(b => b < 0x20 || b > 0x7E);
                if (allZero || allNonPrintable)
                {
                    Log($"[v14][END] Next 4 bytes at 0x{peekPos:X} are all zero or non-printable (likely padding). Ending subchunk parse cleanly.");
                    break;
                }
                long subChunkStart = ms.Position;
                var subChunkIdBytes = br.ReadBytes(4);
                if (subChunkIdBytes.Length < 4) break;
                string subChunkIdStr = Encoding.ASCII.GetString(subChunkIdBytes.Reverse().ToArray());
                uint subChunkSize = br.ReadUInt32();
                long subChunkDataPos = ms.Position;
                long subChunkEnd = subChunkDataPos + subChunkSize;
                // Stop if reading past group region
                if (subChunkEnd > groupLen)
                {
                    Log($"[v14] WARNING: Chunk '{subChunkIdStr}' at 0x{subChunkStart:X} has invalid size {subChunkSize} (would end at 0x{subChunkEnd:X}, group len 0x{groupLen:X}). Stopping subchunk parse.");
                    break;
                }
                Log($"[v14] Subchunk #{chunkIdx}: ID='{subChunkIdStr}' Offset=0x{subChunkStart:X} Size={subChunkSize}");
                subchunkList.Add((subChunkIdStr, subChunkStart, subChunkSize));
                byte[] subChunkData = br.ReadBytes((int)subChunkSize);
                if (subChunkData.Length != subChunkSize)
                {
                    Log($"[v14] ERROR: Could not read {subChunkSize} bytes for chunk '{subChunkIdStr}' at 0x{subChunkStart:X}. Only got {subChunkData.Length} bytes. Aborting parse.");
                    break;
                }
                subchunks[subChunkIdStr] = (subChunkStart, subChunkSize, subChunkData);
                // Special logging for MOVI and MOPY
                if (subChunkIdStr == "MOVI" || subChunkIdStr == "MOPY")
                {
                    Log($"[v14][DIAG] First 32 bytes of {subChunkIdStr}:");
                    for (int i = 0; i < Math.Min(32, subChunkData.Length); i += 16)
                    {
                        var line = subChunkData.Skip(i).Take(Math.Min(16, subChunkData.Length - i)).ToArray();
                        Log($"  {i:X4}: ");
                        foreach (var b in line) Log($"{b:X2} ");
                        Log($"  ");
                        foreach (var b in line) Log(((char)(char.IsControl((char)b) ? '.' : (char)b)).ToString());
                        Log($"");
                    }
                }
                ms.Position = subChunkEnd;
                // CONDITIONAL padding: only skip if next 4 bytes are all zero or all non-printable
                long afterSubChunk = ms.Position;
                if (ms.Position + 4 <= groupLen) {
                    var padPeekBytes = br.ReadBytes(4);
                    ms.Position = afterSubChunk; // reset
                    bool padAllZero = padPeekBytes.All(b => b == 0);
                    bool padAllNonPrintable = padPeekBytes.All(b => b < 0x20 || b > 0x7E);
                    if (padAllZero || padAllNonPrintable) {
                        Log($"[v14][PAD-ODDITY] Conditional padding detected after chunk '{subChunkIdStr}' at 0x{afterSubChunk:X}: next 4 bytes are all zero or non-printable. Skipping 4 bytes. This may indicate a hidden flag or undocumented structure in the WMO format.");
                        ms.Position += 4;
                    } else {
                        Log($"[v14][PAD-ODDITY] No padding after chunk '{subChunkIdStr}' at 0x{afterSubChunk:X}: next subchunk starts immediately. This may indicate a hidden flag or undocumented structure in the WMO format.");
                    }
                }
                // Dynamic chunk header search if next 4 bytes are not a valid chunk ID
                long searchStart = ms.Position;
                if (ms.Position + 4 <= groupLen)
                {
                    ms.Position = searchStart;
                    var nextIdBytes = br.ReadBytes(4);
                    ms.Position = searchStart; // reset
                    string nextId = Encoding.ASCII.GetString(nextIdBytes.Reverse().ToArray());
                    if (!validSubchunks.Contains(nextId))
                    {
                        // Only attempt realignment if enough bytes left
                        if (ms.Position + 4 + 1 <= groupLen)
                        {
                            Log($"[v14][REALIGN] Next 4 bytes at 0x{searchStart:X} do not match a valid chunk ID ('{nextId}'). Scanning forward for valid chunk header...");
                            bool found = false;
                            for (int scan = 1; scan <= 16 && searchStart + scan + 4 <= groupLen; scan++)
                            {
                                ms.Position = searchStart + scan;
                                var scanIdBytes = br.ReadBytes(4);
                                ms.Position = searchStart + scan; // reset
                                string scanId = Encoding.ASCII.GetString(scanIdBytes.Reverse().ToArray());
                                if (validSubchunks.Contains(scanId))
                                {
                                    Log($"[v14][REALIGN] Found valid chunk ID '{scanId}' at 0x{searchStart + scan:X} (skipped {scan} byte(s)). Realigning.");
                                    ms.Position = searchStart + scan;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                            {
                                ms.Position = searchStart;
                                Log($"[v14][REALIGN] No valid chunk header found within 16 bytes after 0x{searchStart:X}. Continuing as before.");
                            }
                        }
                    }
                }
                chunkIdx++;
            }
            // Print summary of all subchunks found
            Log("[v14] Subchunks found in group:");
            foreach (var (id, offset, size) in subchunkList)
                Log($"  ID='{id}' Offset=0x{offset:X} Size={size}");
            // Log group end summary
            if (ms.Position >= groupLen)
                Log($"[v14][END] Reached end of group region at 0x{ms.Position:X} (groupLen=0x{groupLen:X})");
            else
                Log($"[v14][END] Stopped subchunk parse at 0x{ms.Position:X} (groupLen=0x{groupLen:X})");
            // Parse geometry subchunks
            // MOVT: Vertices
            if (subchunks.TryGetValue("MOVT", out var movt))
            {
                int vertCount = movt.data.Length / 12;
                mesh.Vertices.Clear();
                using var vms = new MemoryStream(movt.data);
                using var vbr = new BinaryReader(vms);
                for (int v = 0; v < vertCount; v++)
                {
                    float x = vbr.ReadSingle();
                    float z = vbr.ReadSingle();
                    float y = -vbr.ReadSingle(); // WoW Y/Z swap
                    mesh.Vertices.Add(new WmoVertex { Position = new System.Numerics.Vector3(x, y, z) });
                }
                Log($"[v14] Parsed {vertCount} vertices from MOVT");
            }
            else
            {
                Log("[v14] No MOVT (vertices) found in group");
            }
            
            // MOTV: Texture coordinates
            if (subchunks.TryGetValue("MOTV", out var motv))
            {
                int uvCount = motv.data.Length / 8; // Each UV is 2 floats (8 bytes)
                Log($"[v14][UVs] Found {uvCount} texture coordinates in MOTV chunk");
                
                // If we have vertices but no UVs assigned yet, create temporary storage
                if (mesh.Vertices.Count > 0)
                {
                    // Read all UVs
                    var uvs = new List<Vector2>(uvCount);
                    using var uvms = new MemoryStream(motv.data);
                    using var uvbr = new BinaryReader(uvms);
                    
                    for (int u = 0; u < uvCount; u++)
                    {
                        float u1 = uvbr.ReadSingle();
                        float v1 = uvbr.ReadSingle();
                        uvs.Add(new Vector2(u1, v1));
                        
                        if (u < 10 || u > uvCount - 10) // Log first and last 10 UVs
                        {
                            Log($"[v14][UVs] UV[{u}]: ({u1:F6}, {v1:F6})");
                        }
                    }
                    
                    // Assign UVs to vertices (assuming 1:1 mapping)
                    int assignCount = Math.Min(mesh.Vertices.Count, uvs.Count);
                    for (int i = 0; i < assignCount; i++)
                    {
                        var vertex = mesh.Vertices[i];
                        vertex.UV = uvs[i];
                        mesh.Vertices[i] = vertex; // Struct reassignment
                    }
                    
                    Log($"[v14][UVs] Assigned {assignCount} UVs to vertices");
                    if (assignCount < mesh.Vertices.Count)
                    {
                        Log($"[v14][UVs] WARNING: Not enough UVs for all vertices. Some vertices will have default (0,0) UVs.");
                    }
                }
            }
            else
            {
                Log("[v14][UVs] No MOTV (texture coordinates) found in group");
            }
            
            // MOPY: Material indices and flags
            List<byte> materialIds = new();
            Dictionary<ushort, int> flagCounts = new(); // To count occurrences of each flag value
            
            if (subchunks.TryGetValue("MOPY", out var mopy))
            {
                int faceCount = mopy.data.Length / 2;
                using var mms = new MemoryStream(mopy.data);
                using var mbr = new BinaryReader(mms);
                for (int f = 0; f < faceCount; f++)
                {
                    byte flags = mbr.ReadByte();
                    byte matId = mbr.ReadByte();
                    materialIds.Add(matId);
                    
                    // For detailed flag analysis
                    ushort combinedFlags = flags;
                    if (!flagCounts.ContainsKey(combinedFlags))
                        flagCounts[combinedFlags] = 0;
                    flagCounts[combinedFlags]++;
                }
                Log($"[v14] Parsed {faceCount} faces from MOPY");
                
                // Log flag statistics
                Log($"[v14][FLAGS] Triangle flag statistics:");
                foreach (var kvp in flagCounts.OrderBy(kvp => kvp.Key))
                {
                    string binaryFlags = Convert.ToString(kvp.Key, 2).PadLeft(8, '0');
                    Log($"[v14][FLAGS] Flag value 0x{kvp.Key:X2} ({binaryFlags}): {kvp.Value} triangles");
                }
            }
            else
            {
                Log("[v14] No MOPY (material indices) found in group");
            }
            
            // MOVI: Indices
            if (subchunks.TryGetValue("MOVI", out var movi))
            {
                int indexCount = movi.data.Length / 2;
                using var ims = new MemoryStream(movi.data);
                using var ibr = new BinaryReader(ims);
                var indices = new List<ushort>(indexCount);
                for (int i = 0; i < indexCount; i++)
                    indices.Add(ibr.ReadUInt16());
                // Each triangle is 3 indices
                int triCount = indices.Count / 3;
                mesh.Triangles.Clear();
                for (int t = 0; t < triCount; t++)
                {
                    int i0 = indices[t * 3 + 0];
                    int i1 = indices[t * 3 + 1];
                    int i2 = indices[t * 3 + 2];
                    byte matId = (materialIds.Count > t) ? materialIds[t] : (byte)0;
                    
                    mesh.Triangles.Add(new WmoTriangle { 
                        Index0 = i0, 
                        Index1 = i1, 
                        Index2 = i2, 
                        MaterialId = matId,
                        Flags = 0
                    });
                }
                Log($"[v14] Parsed {triCount} triangles from MOVI");
            }
            else if (materialIds.Count > 0 && mesh.Vertices.Count > 0)
            {
                // Reference implementation: faces are sequential triples of vertices
                int faceCount = materialIds.Count;
                int maxTri = mesh.Vertices.Count / 3;
                int triCount = Math.Min(faceCount, maxTri);
                mesh.Triangles.Clear();
                for (int t = 0; t < triCount; t++)
                {
                    int i0 = t * 3 + 0;
                    int i1 = t * 3 + 1;
                    int i2 = t * 3 + 2;
                    if (i2 >= mesh.Vertices.Count) break;
                    byte matId = materialIds[t];
                    
                    mesh.Triangles.Add(new WmoTriangle { 
                        Index0 = i0, 
                        Index1 = i1, 
                        Index2 = i2, 
                        MaterialId = matId,
                        Flags = 0
                    });
                }
                Log($"[v14] Assembled {triCount} triangles from sequential vertex triples (reference impl)");
            }
            else
            {
                Log("[v14] WARNING: No MOVI (indices) found in group and cannot assemble triangles from MOPY/materials. No triangles will be assembled.");
            }
            
            // Count triangles by flag type
            var trianglesByFlag = mesh.Triangles
                .GroupBy(t => t.Flags)
                .OrderBy(g => g.Key)
                .ToDictionary(g => g.Key, g => g.Count());
            
            Log($"[v14][ANALYSIS] Triangle count by flag value:");
            foreach (var kvp in trianglesByFlag)
            {
                string binaryFlags = Convert.ToString(kvp.Key, 2).PadLeft(8, '0');
                Log($"[v14][ANALYSIS] Flag 0x{kvp.Key:X2} ({binaryFlags}): {kvp.Value} triangles");
            }
            
            // Log triangle count
            Log($"[v14] Assembled {mesh.Triangles.Count} triangles in group");
            return mesh;
        }

        public static void ExportAllGroupsAsObj(string inputWmo, string outputPrefix)
        {
            // Extract and convert textures before exporting meshes
            var outputDir = Path.GetDirectoryName(outputPrefix);
            var textureDirName = "textures"; // Relative folder for textures
            var textureDir = Path.Combine(outputDir, textureDirName);
            
            // Extract textures to the texture directory next to the output files
            ExtractAndConvertTextures(inputWmo, "test_data/053_textures", textureDir);
            Log($"[OBJ] Opening v14 WMO: {inputWmo}");
            using var stream = File.OpenRead(inputWmo);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
            long fileLen = stream.Length;
            stream.Position = 0;

            // --- Step 1: Log all top-level chunk IDs, offsets, and sizes ---
            var chunkHeaders = new List<(string id, long offset, uint size)>();
            
            // --- Setup MOTX/MOMT recording helpers ---
            long motxOffset = -1, momtOffset = -1;
            uint motxSize = 0, momtSize = 0;
            void RecordMotx(long offset, uint size) { if (motxOffset == -1) { motxOffset = offset; motxSize = size; } }
            void RecordMomt(long offset, uint size) { if (momtOffset == -1) { momtOffset = offset; momtSize = size; } }
            
            stream.Position = 0;
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = Encoding.ASCII.GetString(chunkIdBytes.Reverse().ToArray());
                Log($"[DEBUG] Raw chunk ID bytes: {BitConverter.ToString(chunkIdBytes)} String: '{chunkIdStr}'");
                uint chunkSize = reader.ReadUInt32();
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
                Log($"[WMO] Top-level chunk: ID='{chunkIdStr}' Offset=0x{chunkStart:X} Size={chunkSize}");
                if (chunkIdStr == "MOMO")
                {
                    // Parse subchunks inside MOMO
                    long momoDataStart = chunkStart + 8;
                    stream.Position = momoDataStart;
                    byte[] momoData = reader.ReadBytes((int)chunkSize);
                    using var momoStream = new MemoryStream(momoData);
                    using var momoReader = new BinaryReader(momoStream);
                    while (momoStream.Position + 8 <= momoStream.Length)
                    {
                        long subChunkStart = momoStream.Position;
                        var subChunkIdBytes = momoReader.ReadBytes(4);
                        if (subChunkIdBytes.Length < 4) break;
                        string subChunkIdStr = Encoding.ASCII.GetString(subChunkIdBytes.Reverse().ToArray());
                        uint subChunkSize = momoReader.ReadUInt32();
                        if (subChunkIdStr == "MOTX") RecordMotx(momoDataStart + subChunkStart + 8, subChunkSize);
                        if (subChunkIdStr == "MOMT") RecordMomt(momoDataStart + subChunkStart + 8, subChunkSize);
                        momoStream.Position = subChunkStart + 8 + subChunkSize;
                    }
                }
                else
                {
                    stream.Position = chunkStart + 8 + chunkSize;
                }
            }

            // --- Step 2: Find all MOGP group regions ---
            int mver = 0;
            for (int i = 0; i < chunkHeaders.Count; i++)
            {
                var (chunkIdStr, chunkStart, chunkSize) = chunkHeaders[i];
                if (chunkIdStr == "MVER")
                {
                    stream.Position = chunkStart + 8;
                    mver = reader.ReadInt32();
                }
            }
            if (mver != 14)
            {
                Log($"[OBJ] Not a v14 WMO (MVER={mver})");
                return;
            }
            // Find all MOGP group regions
            var groupRegions = new List<(int groupIdx, long start, long end)>();
            for (int i = 0, groupIdx = 0; i < chunkHeaders.Count; i++)
            {
                var (chunkIdStr, chunkStart, chunkSize) = chunkHeaders[i];
                if (chunkIdStr == "MOGP")
                {
                    long groupStart = chunkStart;
                    long groupEnd = fileLen;
                    // Find next MOGP or EOF
                    for (int j = i + 1; j < chunkHeaders.Count; j++)
                    {
                        if (chunkHeaders[j].id == "MOGP")
                        {
                            groupEnd = chunkHeaders[j].offset;
                            break;
                        }
                    }
                    groupRegions.Add((groupIdx++, groupStart, groupEnd));
                }
            }
            Log($"[OBJ] Found {groupRegions.Count} group(s) in WMO");

            // --- Step 3: Build materialId-to-PNG mapping from MOTX/MOMT ---
            // (This logic is adapted from ExtractAndConvertTextures)
            // Find MOTX and MOMT chunks
            foreach (var (chunkIdStr, chunkStart, chunkSize) in chunkHeaders)
            {
                if (chunkIdStr == "MOTX") RecordMotx(chunkStart + 8, chunkSize);
                if (chunkIdStr == "MOMT") RecordMomt(chunkStart + 8, chunkSize);
                if (chunkIdStr == "MOMO")
                {
                    long momoDataStart = chunkStart + 8;
                    stream.Position = momoDataStart;
                    byte[] momoData = reader.ReadBytes((int)chunkSize);
                    using var momoStream = new MemoryStream(momoData);
                    using var momoReader = new BinaryReader(momoStream);
                    while (momoStream.Position + 8 <= momoStream.Length)
                    {
                        long subChunkStart = momoStream.Position;
                        var subChunkIdBytes = momoReader.ReadBytes(4);
                        if (subChunkIdBytes.Length < 4) break;
                        string subChunkIdStr = Encoding.ASCII.GetString(subChunkIdBytes.Reverse().ToArray());
                        uint subChunkSize = momoReader.ReadUInt32();
                        if (subChunkIdStr == "MOTX") RecordMotx(momoDataStart + subChunkStart + 8, subChunkSize);
                        if (subChunkIdStr == "MOMT") RecordMomt(momoDataStart + subChunkStart + 8, subChunkSize);
                        momoStream.Position = subChunkStart + 8 + subChunkSize;
                    }
                }
            }
            Dictionary<byte, string> materialIdToPng = new();
            if (motxOffset != -1 && momtOffset != -1)
            {
                stream.Position = motxOffset;
                byte[] motxData = reader.ReadBytes((int)motxSize);
                stream.Position = momtOffset;
                int nMaterials = (int)(momtSize / 44);
                for (int i = 0; i < nMaterials && i < 256; i++) // byte MaterialId max 255
                {
                    stream.Position = momtOffset + i * 44 + 0xC;
                    int texOffset = reader.ReadInt32();
                    string tex = null;
                    if (texOffset >= 0 && texOffset < motxData.Length)
                    {
                        int end = texOffset;
                        while (end < motxData.Length && motxData[end] != 0) end++;
                        tex = Encoding.ASCII.GetString(motxData, texOffset, end - texOffset);
                    }
                    if (!string.IsNullOrWhiteSpace(tex))
                    {
                        // PNG filename is just the base name with .png, as in ExtractAndConvertTextures
                        string pngFile = Path.GetFileNameWithoutExtension(tex) + ".png";
                        // Store just the relative path from MTL to PNG
                        materialIdToPng[(byte)i] = Path.Combine(textureDirName, pngFile);
                    }
                }
            }

            int exported = 0;
            foreach (var (groupIdx, groupStart, groupEnd) in groupRegions)
            {
                long groupLen = groupEnd - groupStart;
                stream.Position = groupStart;
                byte[] groupData = reader.ReadBytes((int)groupLen);
                string outFile = $"{outputPrefix}-group-{groupIdx:D3}.obj";
                string mtlFile = Path.ChangeExtension(outFile, ".mtl");
                Log($"[OBJ] Exporting group {groupIdx} (offset=0x{groupStart:X}, len={groupLen}) to {outFile} and {mtlFile}");
                try
                {
                    var mesh = LoadFromV14GroupChunk(groupData);
                    WmoGroupMesh.SaveToObjAndMtl(mesh, outFile, mtlFile, materialIdToPng);
                    Log($"[OBJ] Wrote OBJ file: {outFile} and MTL file: {mtlFile}");
                    
                    exported++;
                }
                catch (Exception ex)
                {
                    Log($"[OBJ] ERROR: Failed to export group {groupIdx}: {ex.Message}");
                }
            }
            Log($"[OBJ] Exported {exported}/{groupRegions.Count} group(s) to OBJ+MTL files with prefix '{outputPrefix}-group-'.");
        }

        public static void ExportMergedGroupsAsObj(string inputWmo, string outputObj)
        {
            Log($"[INFO] Converting all groups in {inputWmo} to a single merged OBJ: {outputObj}");
            // Extract and convert textures before exporting meshes
            ExtractAndConvertTextures(inputWmo, "test_data/053_textures", "output/053_textures_png");
            
            using var stream = File.OpenRead(inputWmo);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
            long fileLen = stream.Length;
            stream.Position = 0;

            // --- Step 1: Log all top-level chunk IDs, offsets, and sizes ---
            var chunkHeaders = new List<(string id, long offset, uint size)>();
            
            stream.Position = 0;
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = Encoding.ASCII.GetString(chunkIdBytes.Reverse().ToArray());
                uint chunkSize = reader.ReadUInt32();
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
                Log($"[WMO] Top-level chunk: ID='{chunkIdStr}' Offset=0x{chunkStart:X} Size={chunkSize}");
                stream.Position = chunkStart + 8 + chunkSize;
            }

            // --- Step 2: Find all MOGP group regions ---
            int mver = 0;
            for (int i = 0; i < chunkHeaders.Count; i++)
            {
                var (chunkIdStr, chunkStart, chunkSize) = chunkHeaders[i];
                if (chunkIdStr == "MVER")
                {
                    stream.Position = chunkStart + 8;
                    mver = reader.ReadInt32();
                }
            }
            if (mver != 14)
            {
                Log($"[OBJ] Not a v14 WMO (MVER={mver})");
                return;
            }
            // Find all MOGP group regions
            var groupRegions = new List<(int groupIdx, long start, long end)>();
            for (int i = 0, groupIdx = 0; i < chunkHeaders.Count; i++)
            {
                var (chunkIdStr, chunkStart, chunkSize) = chunkHeaders[i];
                if (chunkIdStr == "MOGP")
                {
                    long groupStart = chunkStart;
                    long groupEnd = fileLen;
                    // Find next MOGP or EOF
                    for (int j = i + 1; j < chunkHeaders.Count; j++)
                    {
                        if (chunkHeaders[j].id == "MOGP")
                        {
                            groupEnd = chunkHeaders[j].offset;
                            break;
                        }
                    }
                    groupRegions.Add((groupIdx++, groupStart, groupEnd));
                }
            }
            Log($"[OBJ] Found {groupRegions.Count} group(s) in WMO");
            
            // --- Step 3: Build materialId-to-PNG mapping from MOTX/MOMT ---
            long motxOffset = -1, momtOffset = -1;
            uint motxSize = 0, momtSize = 0;
            
            foreach (var (chunkIdStr, chunkStart, chunkSize) in chunkHeaders)
            {
                if (chunkIdStr == "MOTX") { motxOffset = chunkStart + 8; motxSize = chunkSize; }
                if (chunkIdStr == "MOMT") { momtOffset = chunkStart + 8; momtSize = chunkSize; }
            }
            
            Dictionary<byte, string> materialIdToPng = new();
            if (motxOffset != -1 && momtOffset != -1)
            {
                stream.Position = motxOffset;
                byte[] motxData = reader.ReadBytes((int)motxSize);
                stream.Position = momtOffset;
                int nMaterials = (int)(momtSize / 44);
                for (int i = 0; i < nMaterials && i < 256; i++) // byte MaterialId max 255
                {
                    stream.Position = momtOffset + i * 44 + 0xC;
                    int texOffset = reader.ReadInt32();
                    string tex = null;
                    if (texOffset >= 0 && texOffset < motxData.Length)
                    {
                        int end = texOffset;
                        while (end < motxData.Length && motxData[end] != 0) end++;
                        tex = Encoding.ASCII.GetString(motxData, texOffset, end - texOffset);
                    }
                    if (!string.IsNullOrWhiteSpace(tex))
                    {
                        // PNG filename is just the base name with .png, as in ExtractAndConvertTextures
                        string pngFile = Path.GetFileNameWithoutExtension(tex) + ".png";
                        // Use fixed path for materials
                        materialIdToPng[(byte)i] = Path.Combine("053_textures_png", pngFile);
                    }
                }
            }
            
            // --- Step 4: Parse and merge all groups ---
            var mergedMesh = new WoWToolbox.Core.WMO.WmoGroupMesh();
            int totalVertices = 0, totalTriangles = 0, mergedGroups = 0;
            foreach (var (groupIdx, groupStart, groupEnd) in groupRegions)
            {
                long groupLen = groupEnd - groupStart;
                stream.Position = groupStart;
                byte[] groupData = reader.ReadBytes((int)groupLen);
                try
                {
                    var mesh = LoadFromV14GroupChunk(groupData);
                    if (mesh.Vertices.Count == 0 || mesh.Triangles.Count == 0)
                    {
                        Log($"[OBJ] Skipping empty group {groupIdx} (vertices: {mesh.Vertices.Count}, triangles: {mesh.Triangles.Count})");
                        continue;
                    }
                    
                    int vertOffset = mergedMesh.Vertices.Count;
                    mergedMesh.Vertices.AddRange(mesh.Vertices);
                    
                    foreach (var tri in mesh.Triangles)
                    {
                        mergedMesh.Triangles.Add(new WoWToolbox.Core.WMO.WmoTriangle
                        {
                            Index0 = tri.Index0 + vertOffset,
                            Index1 = tri.Index1 + vertOffset,
                            Index2 = tri.Index2 + vertOffset,
                            MaterialId = tri.MaterialId,
                            Flags = 0 // Ignore flags
                        });
                    }
                    mergedGroups++;
                    totalVertices += mesh.Vertices.Count;
                    totalTriangles += mesh.Triangles.Count;
                }
                catch (Exception ex)
                {
                    Log($"[OBJ] ERROR: Failed to parse/merge group {groupIdx}: {ex.Message}\n{ex.StackTrace}");
                }
            }
            Log($"[OBJ] Merged {mergedGroups} group(s): {totalVertices} vertices, {totalTriangles} triangles");
            Log($"[OBJ] Attempting to write merged OBJ file: {outputObj}");
            
            if (mergedMesh.Vertices.Count == 0 || mergedMesh.Triangles.Count == 0)
            {
                Log($"[OBJ][WARN] Merged mesh is empty (vertices: {mergedMesh.Vertices.Count}, triangles: {mergedMesh.Triangles.Count}). Cannot export.");
                return;
            }
            
            // --- Step 5: Export merged mesh as OBJ+MTL ---
            string mtlFile = Path.ChangeExtension(outputObj, ".mtl");
            try
            {
                WoWToolbox.Core.WMO.WmoGroupMesh.SaveToObjAndMtl(mergedMesh, outputObj, mtlFile, materialIdToPng);
                Log($"[OBJ] Wrote merged OBJ file: {outputObj} and MTL file: {mtlFile}");
            }
            catch (Exception ex)
            {
                Log($"[OBJ] ERROR: Failed to export merged OBJ+MTL: {ex.Message}\n{ex.StackTrace}");
            }
        }

        public static void ExtractAndConvertTextures(string inputWmo, string blpRoot, string pngOutDir)
        {
            Log($"[TEX][START] ExtractAndConvertTextures: inputWmo={inputWmo}, blpRoot={blpRoot}, pngOutDir={pngOutDir}");
            if (!Directory.Exists(pngOutDir)) Directory.CreateDirectory(pngOutDir);
            using var stream = File.OpenRead(inputWmo);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
            long fileLen = stream.Length;
            stream.Position = 0;
            // Log all top-level chunk IDs, offsets, and sizes
            Log($"[TEX][SCAN] Scanning all top-level chunks in {inputWmo}");
            var chunkHeaders = new List<(string id, long offset, uint size)>();
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = Encoding.ASCII.GetString(chunkIdBytes.Reverse().ToArray());
                Log($"[DEBUG] Raw chunk ID bytes: {BitConverter.ToString(chunkIdBytes)} String: '{chunkIdStr}'");
                uint chunkSize = reader.ReadUInt32();
                Log($"[WMO] Top-level chunk: ID='{chunkIdStr}' Offset=0x{chunkStart:X} Size={chunkSize}");
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
                Log($"[TEX][CHUNK] ID='{chunkIdStr}' Offset=0x{chunkStart:X} Size={chunkSize}");
                // If not MOTX or MOMT, log first 32 bytes as hex dump for unknowns
                if (chunkIdStr != "MOTX" && chunkIdStr != "MOMT") {
                    long savePos = stream.Position;
                    stream.Position = chunkStart + 8;
                    byte[] dump = reader.ReadBytes((int)Math.Min(32, chunkSize));
                    var hex = string.Join(" ", dump.Select(b => b.ToString("X2")));
                    Log($"[TEX][CHUNK][DUMP] ID='{chunkIdStr}' First 32 bytes: {hex}");
                    stream.Position = savePos;
                }
                if (chunkIdStr == "MOMO")
                {
                    // Parse subchunks inside MOMO
                    long momoDataStart = chunkStart + 8;
                    stream.Position = momoDataStart;
                    byte[] momoData = reader.ReadBytes((int)chunkSize);
                    using var momoStream = new MemoryStream(momoData);
                    using var momoReader = new BinaryReader(momoStream);
                    while (momoStream.Position + 8 <= momoStream.Length)
                    {
                        long subChunkStart = momoStream.Position;
                        var subChunkIdBytes = momoReader.ReadBytes(4);
                        if (subChunkIdBytes.Length < 4) break;
                        string subChunkIdStr = Encoding.ASCII.GetString(subChunkIdBytes.Reverse().ToArray());
                        uint subChunkSize = momoReader.ReadUInt32();
                        Log($"[WMO][MOMO] Subchunk: ID='{subChunkIdStr}' Offset=0x{subChunkStart:X} Size={subChunkSize}");
                        // Optionally: add to chunkHeaders or process as needed
                        momoStream.Position = subChunkStart + 8 + subChunkSize;
                    }
                    // After parsing MOMO, move stream to the end of the chunk
                    stream.Position = chunkStart + 8 + chunkSize;
                }
                else
                {
                    stream.Position = chunkStart + 8 + chunkSize;
                }
            }
            // Find MOTX and MOMT chunks
            List<string> motxStrings = new();
            List<int> momtTextureOffsets = new();
            long motxOffset = -1, momtOffset = -1;
            uint motxSize = 0, momtSize = 0;
            // Helper to record found MOTX/MOMT
            void RecordMotx(long offset, uint size) { if (motxOffset == -1) { motxOffset = offset; motxSize = size; } }
            void RecordMomt(long offset, uint size) { if (momtOffset == -1) { momtOffset = offset; momtSize = size; } }
            foreach (var (chunkIdStr, chunkStart, chunkSize) in chunkHeaders)
            {
                if (chunkIdStr == "MOTX") RecordMotx(chunkStart + 8, chunkSize);
                if (chunkIdStr == "MOMT") RecordMomt(chunkStart + 8, chunkSize);
                if (chunkIdStr == "MOMO")
                {
                    // Parse subchunks inside MOMO
                    long momoDataStart = chunkStart + 8;
                    stream.Position = momoDataStart;
                    byte[] momoData = reader.ReadBytes((int)chunkSize);
                    using var momoStream = new MemoryStream(momoData);
                    using var momoReader = new BinaryReader(momoStream);
                    while (momoStream.Position + 8 <= momoStream.Length)
                    {
                        long subChunkStart = momoStream.Position;
                        var subChunkIdBytes = momoReader.ReadBytes(4);
                        if (subChunkIdBytes.Length < 4) break;
                        string subChunkIdStr = Encoding.ASCII.GetString(subChunkIdBytes.Reverse().ToArray());
                        uint subChunkSize = momoReader.ReadUInt32();
                        if (subChunkIdStr == "MOTX") RecordMotx(momoDataStart + subChunkStart + 8, subChunkSize);
                        if (subChunkIdStr == "MOMT") RecordMomt(momoDataStart + subChunkStart + 8, subChunkSize);
                        momoStream.Position = subChunkStart + 8 + subChunkSize;
                    }
                }
            }
            if (motxOffset == -1 || momtOffset == -1)
            {
                Log($"[TEX][WARN] MOTX or MOMT chunk not found in {inputWmo}. No textures will be processed.");
                return;
            }
            // Parse MOTX: keep as raw byte array
            stream.Position = motxOffset;
            byte[] motxData = reader.ReadBytes((int)motxSize);
            // Parse MOMT: each material is 44 bytes, texture1 offset at +0xC
            stream.Position = momtOffset;
            int nMaterials = (int)(momtSize / 44);
            List<string> referencedBlps = new();
            for (int i = 0; i < nMaterials; i++)
            {
                stream.Position = momtOffset + i * 44 + 0xC;
                int texOffset = reader.ReadInt32();
                // Extract string directly from MOTX using offset
                if (texOffset >= 0 && texOffset < motxData.Length)
                {
                    int end = texOffset;
                    while (end < motxData.Length && motxData[end] != 0) end++;
                    string tex = Encoding.ASCII.GetString(motxData, texOffset, end - texOffset);
                    if (!string.IsNullOrWhiteSpace(tex) && !referencedBlps.Contains(tex))
                        referencedBlps.Add(tex);
                }
            }
            Log($"[TEX][DEBUG] {referencedBlps.Count} unique BLPs referenced by materials (direct offset extraction):");
            foreach (var blp in referencedBlps) Log($"[TEX][REF] {blp}");
            // --- Recursive BLP search helper ---
            static string FindBlpRecursive(string blpRoot, string relPath)
            {
                // Normalize path separators
                string searchName = Path.GetFileName(relPath).ToLowerInvariant();
                var files = Directory.GetFiles(blpRoot, "*.blp", SearchOption.AllDirectories);
                foreach (var file in files)
                {
                    if (Path.GetFileName(file).ToLowerInvariant() == searchName)
                        return file;
                }
                return null;
            }
            int pngWritten = 0;
            int blpsFound = 0, blpsMissing = 0;
            foreach (var blpRelPath in referencedBlps)
            {
                string assetString = blpRelPath;
                string expectedRelPath = blpRelPath.Replace("/", "\\");
                string blpPath = Path.Combine(blpRoot, expectedRelPath);
                string foundBlpPath = null;
                if (File.Exists(blpPath))
                {
                    foundBlpPath = blpPath;
                    Log($"[TEX][BLP] Found (direct): {assetString} -> {blpPath}");
                }
                else
                {
                    foundBlpPath = FindBlpRecursive(blpRoot, blpRelPath);
                    if (foundBlpPath != null)
                        Log($"[TEX][BLP] Found (recursive): {assetString} -> {foundBlpPath}");
                    else {
                        Log($"[TEX][WARN] BLP not found: {assetString} (searched for {Path.GetFileName(blpRelPath)} recursively under {blpRoot})");
                        blpsMissing++;
                    }
                }
                if (foundBlpPath == null) continue;
                blpsFound++;
                string pngPath = Path.Combine(pngOutDir, Path.GetFileNameWithoutExtension(blpRelPath) + ".png");
                Log($"[TEX][PNG] Attempting to write PNG: {pngPath}");
                if (File.Exists(pngPath))
                {
                    Log($"[TEX] PNG already exists: {pngPath}");
                    continue;
                }
                try
                {
                    // Create the directory if it doesn't exist (for nested paths)
                    Directory.CreateDirectory(Path.GetDirectoryName(pngPath));
                    
                    byte[] blpBytes = File.ReadAllBytes(foundBlpPath);
                    var blp = new BLP(blpBytes);
                    var image = blp.GetMipMap(0); // Image<Rgba32>
                    
                    // Ensure image is not null
                    if (image != null)
                    {
                        using (var fs = File.OpenWrite(pngPath))
                        {
                            image.Save(fs, new PngEncoder());
                        }
                        Log($"[TEX] Converted {assetString} to {pngPath}");
                        pngWritten++;
                    }
                    else
                    {
                        Log($"[TEX][ERR] Failed to get image data from BLP: {foundBlpPath}");
                    }
                }
                catch (Exception ex)
                {
                    Log($"[TEX][ERR] Failed to convert {assetString}: {ex.Message}");
                }
            }
            
            // Create a fallback texture for missing materials
            var fallbackPath = Path.Combine(pngOutDir, "missing.png");
            if (!File.Exists(fallbackPath))
            {
                try
                {
                    // Create a simple 64x64 checkerboard pattern for missing textures
                    var fallbackImage = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(64, 64);
                    
                    // Fill with a checkerboard pattern
                    for (int y = 0; y < 64; y++)
                    {
                        for (int x = 0; x < 64; x++)
                        {
                            bool isEvenTile = ((x / 8) + (y / 8)) % 2 == 0;
                            if (isEvenTile)
                                fallbackImage[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(255, 0, 255, 255); // Magenta
                            else
                                fallbackImage[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(0, 0, 0, 255); // Black
                        }
                    }
                    
                    using (var fs = File.OpenWrite(fallbackPath))
                    {
                        fallbackImage.Save(fs, new PngEncoder());
                    }
                    Log($"[TEX] Created fallback texture: {fallbackPath}");
                }
                catch (Exception ex)
                {
                    Log($"[TEX][ERR] Failed to create fallback texture: {ex.Message}");
                }
            }
            
            if (referencedBlps.Count == 0)
                Log($"[TEX][FATAL] No referenced BLPs found in MOTX/MOMT. Check chunk parsing and input file integrity.");
            Log($"[TEX][SUMMARY] BLPs found: {blpsFound}, missing: {blpsMissing}, PNGs written: {pngWritten}");
        }

        /// <summary>
        /// Exports a v14 WMO as a v17 WMO using the new WmoV17Writer.
        /// </summary>
        public static void ExportAsV17Wmo(string inputWmo, string outputWmo)
        {
            Log($"[V17] Starting v14→v17 WMO export: {inputWmo} → {outputWmo}");
            using var stream = File.OpenRead(inputWmo);
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
            long fileLen = stream.Length;
            stream.Position = 0;

            // --- Step 1: Parse all top-level chunk headers ---
            var chunkHeaders = new List<(string id, long offset, uint size)>();
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = Encoding.ASCII.GetString(chunkIdBytes.Reverse().ToArray());
                uint chunkSize = reader.ReadUInt32();
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
                stream.Position = chunkStart + 8 + chunkSize;
            }

            // --- Step 2: Extract MOTX (textures) ---
            var motxHeader = chunkHeaders.Find(h => h.id == "MOTX");
            List<string> textures = new();
            if (motxHeader != default)
            {
                stream.Position = motxHeader.offset + 8;
                byte[] motxData = reader.ReadBytes((int)motxHeader.size);
                int start = 0;
                for (int i = 0; i < motxData.Length; i++)
                {
                    if (motxData[i] == 0)
                    {
                        if (i > start)
                            textures.Add(Encoding.ASCII.GetString(motxData, start, i - start));
                        start = i + 1;
                    }
                }
                if (start < motxData.Length)
                    textures.Add(Encoding.ASCII.GetString(motxData, start, motxData.Length - start));
                Log($"[V17] Extracted {textures.Count} textures from MOTX");
            }
            else
            {
                Log("[V17][WARN] No MOTX chunk found");
            }

            // --- Step 3: Extract MOMT (materials) ---
            var momtHeader = chunkHeaders.Find(h => h.id == "MOMT");
            List<WoWToolbox.Core.Models.WmoMaterial> materials = new();
            if (momtHeader != default)
            {
                stream.Position = momtHeader.offset + 8;
                int nMaterials = (int)(momtHeader.size / 64); // v17 size
                for (int i = 0; i < nMaterials; i++)
                {
                    materials.Add(new WoWToolbox.Core.Models.WmoMaterial());
                }
                Log($"[V17] Extracted {materials.Count} materials from MOMT");
            }
            else
            {
                Log("[V17][WARN] No MOMT chunk found");
            }

            // --- Step 4: Extract MOHD (header fields) ---
            var mohdHeader = chunkHeaders.Find(h => h.id == "MOHD");
            var header = new WmoRootHeader();
            if (mohdHeader != default)
            {
                stream.Position = mohdHeader.offset + 8;
                header.MaterialCount = reader.ReadUInt32();
                header.GroupCount = reader.ReadUInt32();
                header.PortalCount = reader.ReadUInt32();
                header.LightCount = reader.ReadUInt32();
                header.ModelCount = reader.ReadUInt32();
                header.DoodadCount = reader.ReadUInt32();
                header.SetCount = reader.ReadUInt32();
                header.AmbientColor = reader.ReadUInt32();
                header.AreaTableID = reader.ReadUInt32();
                header.BoundingBoxMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                header.BoundingBoxMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                header.Flags = reader.ReadUInt16();
                header.LodCount = reader.ReadUInt16();
                Log("[V17] Extracted MOHD header fields");
            }
            else
            {
                Log("[V17][WARN] No MOHD chunk found");
            }

            // --- Step 5: Build WmoRootData ---
            var rootData = new WmoRootData {
                Header = header,
                Textures = textures,
                Materials = materials,
                // TODO: GroupNames, GroupInfos, Doodads, etc.
            };

            // --- Step 6: TODO: Extract and map group data ---
            var groupData = new WmoGroupData();

            // --- Step 7: Write v17 WMO ---
            var writer = new WmoV17Writer();
            writer.WriteRoot(outputWmo, rootData);
            Log($"[V17] Wrote v17 root WMO: {outputWmo}");

            // --- Step 8: Also export OBJ+MTL and PNG textures ---
            string outputPrefix = Path.Combine(Path.GetDirectoryName(outputWmo) ?? ".", Path.GetFileNameWithoutExtension(outputWmo));
            ExportAllGroupsAsObj(inputWmo, outputPrefix + "-group");
            Log($"[OBJ/MTL] Exported OBJ+MTL to: {outputPrefix}-group-*.obj/.mtl");
            string textureDir = Path.Combine(Path.GetDirectoryName(outputWmo) ?? ".", "textures");
            ExtractAndConvertTextures(inputWmo, "test_data/053_textures", textureDir);
            Log($"[PNG] Extracted textures to: {textureDir}");
        }
    }
} 