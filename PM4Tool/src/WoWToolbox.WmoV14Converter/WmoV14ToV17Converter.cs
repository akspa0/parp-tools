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
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint chunkSize = reader.ReadUInt32();
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
                Log($"[WMO] Top-level chunk: ID='{chunkIdStr}' Offset=0x{chunkStart:X} Size={chunkSize}");
                stream.Position = chunkStart + 8 + chunkSize;
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
                string id = new string(idBytes.Reverse().Select(b => (char)b).ToArray());
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
                string subChunkIdStr = new string(subChunkIdBytes.Reverse().Select(b => (char)b).ToArray());
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
                    string nextId = new string(nextIdBytes.Reverse().Select(b => (char)b).ToArray());
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
                                string scanId = new string(scanIdBytes.Reverse().Select(b => (char)b).ToArray());
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
            // MOPY: Material indices and flags
            List<byte> materialIds = new();
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
                }
                Log($"[v14] Parsed {faceCount} faces from MOPY");
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
                    mesh.Triangles.Add(new WmoTriangle { Index0 = i0, Index1 = i1, Index2 = i2, MaterialId = matId });
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
                    mesh.Triangles.Add(new WmoTriangle { Index0 = i0, Index1 = i1, Index2 = i2, MaterialId = matId });
                }
                Log($"[v14] Assembled {triCount} triangles from sequential vertex triples (reference impl)");
            }
            else
            {
                Log("[v14] WARNING: No MOVI (indices) found in group and cannot assemble triangles from MOPY/materials. No triangles will be assembled.");
            }
            // Log triangle count
            Log($"[v14] Assembled {mesh.Triangles.Count} triangles in group");
            // TODO: Parse MOBA, MLIQ, and other subchunks as needed
            return mesh;
        }

        public static void ExportAllGroupsAsObj(string inputWmo, string outputPrefix)
        {
            // Extract and convert textures before exporting meshes
            ExtractAndConvertTextures(inputWmo, "test_data/053_textures", "output/053_textures_png");
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
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint chunkSize = reader.ReadUInt32();
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
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
            int exported = 0;
            foreach (var (groupIdx, groupStart, groupEnd) in groupRegions)
            {
                long groupLen = groupEnd - groupStart;
                stream.Position = groupStart;
                byte[] groupData = reader.ReadBytes((int)groupLen);
                string outFile = $"{outputPrefix}-group-{groupIdx:D3}.obj";
                Log($"[OBJ] Exporting group {groupIdx} (offset=0x{groupStart:X}, len={groupLen}) to {outFile}");
                try
                {
                    var mesh = LoadFromV14GroupChunk(groupData);
                    WmoGroupMesh.SaveToObj(mesh, outFile);
                    Log($"[OBJ] Wrote OBJ file: {outFile}");
                    exported++;
                }
                catch (Exception ex)
                {
                    Log($"[OBJ] ERROR: Failed to export group {groupIdx}: {ex.Message}");
                }
            }
            Log($"[OBJ] Exported {exported}/{groupRegions.Count} group(s) to OBJ files with prefix '{outputPrefix}-group-'.");
        }

        public static void ExportMergedGroupsAsObj(string inputWmo, string outputObj)
        {
            // Extract and convert textures before exporting meshes
            ExtractAndConvertTextures(inputWmo, "test_data/053_textures", "output/053_textures_png");
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
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint chunkSize = reader.ReadUInt32();
                chunkHeaders.Add((chunkIdStr, chunkStart, chunkSize));
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
            // --- Step 3: Parse and merge all groups ---
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
                    int vertOffset = mergedMesh.Vertices.Count;
                    // Merge vertices
                    mergedMesh.Vertices.AddRange(mesh.Vertices);
                    // Merge triangles (adjust indices)
                    foreach (var tri in mesh.Triangles)
                    {
                        mergedMesh.Triangles.Add(new WoWToolbox.Core.WMO.WmoTriangle
                        {
                            Index0 = tri.Index0 + vertOffset,
                            Index1 = tri.Index1 + vertOffset,
                            Index2 = tri.Index2 + vertOffset,
                            MaterialId = tri.MaterialId
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
                Log($"[OBJ][WARN] Merged mesh is empty (vertices: {mergedMesh.Vertices.Count}, triangles: {mergedMesh.Triangles.Count}). Writing file anyway for debugging.");
            }
            // --- Step 4: Export merged mesh as OBJ ---
            try
            {
                WoWToolbox.Core.WMO.WmoGroupMesh.SaveToObj(mergedMesh, outputObj);
                Log($"[OBJ] Wrote merged OBJ file: {outputObj}");
            }
            catch (Exception ex)
            {
                Log($"[OBJ] ERROR: Failed to export merged OBJ: {ex.Message}\n{ex.StackTrace}");
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
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint chunkSize = reader.ReadUInt32();
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
                stream.Position = chunkStart + 8 + chunkSize;
            }
            // Find MOTX and MOMT chunks
            List<string> motxStrings = new();
            List<int> momtTextureOffsets = new();
            long motxOffset = -1, momtOffset = -1;
            uint motxSize = 0, momtSize = 0;
            foreach (var (chunkIdStr, chunkStart, chunkSize) in chunkHeaders)
            {
                if (chunkIdStr == "MOTX") { motxOffset = chunkStart + 8; motxSize = chunkSize; }
                if (chunkIdStr == "MOMT") { momtOffset = chunkStart + 8; momtSize = chunkSize; }
            }
            Log($"[TEX][DEBUG] MOTX offset={motxOffset}, size={motxSize}; MOMT offset={momtOffset}, size={momtSize}");
            if (motxOffset == -1 || momtOffset == -1)
            {
                Log($"[TEX][WARN] MOTX or MOMT chunk not found in {inputWmo}. No textures will be processed.");
                return;
            }
            // Parse MOTX: null-terminated strings
            stream.Position = motxOffset;
            byte[] motxData = reader.ReadBytes((int)motxSize);
            int idx = 0;
            while (idx < motxData.Length)
            {
                int start = idx;
                while (idx < motxData.Length && motxData[idx] != 0) idx++;
                string tex = Encoding.ASCII.GetString(motxData, start, idx - start);
                if (!string.IsNullOrWhiteSpace(tex)) motxStrings.Add(tex);
                while (idx < motxData.Length && motxData[idx] == 0) idx++; // skip padding
            }
            Log($"[TEX][DEBUG] Found {motxStrings.Count} texture paths in MOTX:");
            foreach (var s in motxStrings) Log($"[TEX][MOTX] {s}");
            // Parse MOMT: each material is 44 bytes, texture1 offset at +0xC
            stream.Position = momtOffset;
            int nMaterials = (int)(momtSize / 44);
            for (int i = 0; i < nMaterials; i++)
            {
                stream.Position = momtOffset + i * 44 + 0xC;
                int texOffset = reader.ReadInt32();
                momtTextureOffsets.Add(texOffset);
            }
            // Build set of referenced textures
            HashSet<string> referencedBlps = new();
            foreach (int offset in momtTextureOffsets)
            {
                int runningOffset = 0;
                foreach (var tex in motxStrings)
                {
                    if (runningOffset == offset)
                    {
                        referencedBlps.Add(tex);
                        break;
                    }
                    runningOffset += tex.Length + 1; // null terminator
                    while (runningOffset % 4 != 0) runningOffset++;
                }
            }
            Log($"[TEX][DEBUG] {referencedBlps.Count} unique BLPs referenced by materials:");
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
                    byte[] blpBytes = File.ReadAllBytes(foundBlpPath);
                    var blp = new BLP(blpBytes);
                    var image = blp.GetMipMap(0); // Image<Rgba32>
                    using (var fs = File.OpenWrite(pngPath))
                    {
                        image.Save(fs, new PngEncoder());
                    }
                    Log($"[TEX] Converted {assetString} to {pngPath}");
                    pngWritten++;
                }
                catch (Exception ex)
                {
                    Log($"[TEX][ERR] Failed to convert {assetString}: {ex.Message}");
                }
            }
            if (referencedBlps.Count == 0)
                Log($"[TEX][FATAL] No referenced BLPs found in MOTX/MOMT. Check chunk parsing and input file integrity.");
            Log($"[TEX][SUMMARY] BLPs found: {blpsFound}, missing: {blpsMissing}, PNGs written: {pngWritten}");
        }
    }
} 