using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Numerics;
using WoWToolbox.Core.WMO; // For WmoGroupMesh, WmoRootLoader, etc.

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
            string[] validSubchunks = { "MOPY", "MOVT", "MONR", "MOTV", "MOVI", "MOBA", "MLIQ", "MOCV", "MOTX", "MOGN", "MOMT" };
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
                // 4-byte alignment: skip padding if needed
                if (ms.Position % 4 != 0)
                {
                    long pad = 4 - (ms.Position % 4);
                    Log($"[v14] Skipping {pad} padding byte(s) after chunk '{subChunkIdStr}'");
                    ms.Position += pad;
                }
                chunkIdx++;
            }
            // Print summary of all subchunks found
            Log("[v14] Subchunks found in group:");
            foreach (var (id, offset, size) in subchunkList)
                Log($"  ID='{id}' Offset=0x{offset:X} Size={size}");
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
    }
} 