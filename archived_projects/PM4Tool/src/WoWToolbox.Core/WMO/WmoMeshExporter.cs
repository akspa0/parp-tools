using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace WoWToolbox.Core.WMO
{
    /// <summary>
    /// Provides utilities for loading WMO meshes and exporting them to OBJ format.
    /// </summary>
    public static class WmoMeshExporter
    {
        /// <summary>
        /// Loads and merges all group meshes for a WMO root file.
        /// </summary>
        /// <param name="rootWmoPath">Path to the WMO root file.</param>
        /// <returns>The merged WmoGroupMesh, or null if no valid groups were loaded.</returns>
        public static WmoGroupMesh? LoadMergedWmoMesh(string rootWmoPath)
        {
            if (!File.Exists(rootWmoPath))
                throw new FileNotFoundException($"WMO root file not found: {rootWmoPath}");

            string groupsDir = Path.GetDirectoryName(rootWmoPath) ?? ".";
            string rootBaseName = Path.GetFileNameWithoutExtension(rootWmoPath);
            var (groupCount, internalGroupNames) = WmoRootLoader.LoadGroupInfo(rootWmoPath);
            if (groupCount <= 0)
                return null;

            var groupMeshes = new List<WmoGroupMesh>();
            for (int i = 0; i < groupCount; i++)
            {
                string? groupPathToLoad = FindGroupFilePath(i, rootBaseName, groupsDir, internalGroupNames);
                if (groupPathToLoad == null)
                    continue;
                using var groupStream = File.OpenRead(groupPathToLoad);
                WmoGroupMesh mesh = WmoGroupMesh.LoadFromStream(groupStream, groupPathToLoad);
                if (mesh != null && mesh.Vertices.Count > 0 && mesh.Triangles.Count > 0)
                    groupMeshes.Add(mesh);
            }
            if (groupMeshes.Count == 0)
                return null;
            return WmoGroupMesh.MergeMeshes(groupMeshes);
        }

        /// <summary>
        /// Exports a merged WmoGroupMesh to OBJ format.
        /// </summary>
        /// <param name="mesh">The merged WmoGroupMesh to export.</param>
        /// <param name="outputPath">The output OBJ file path.</param>
        public static void SaveMergedWmoToObj(WmoGroupMesh mesh, string outputPath)
        {
            if (mesh == null) throw new ArgumentNullException(nameof(mesh));
            WmoGroupMesh.SaveToObj(mesh, outputPath);
        }

        /// <summary>
        /// Finds the group file path for a given group index.
        /// </summary>
        /// <param name="groupIndex">The group index.</param>
        /// <param name="rootBaseName">The base name of the root WMO file (without extension).</param>
        /// <param name="groupsDir">The directory containing the group files.</param>
        /// <param name="internalGroupNames">The list of internal group names from the WMO root.</param>
        /// <returns>The path to the group file, or null if not found.</returns>
        public static string? FindGroupFilePath(int groupIndex, string rootBaseName, string groupsDir, List<string> internalGroupNames)
        {
            string? groupPathToLoad = null;
            string internalName = (groupIndex < internalGroupNames.Count) ? internalGroupNames[groupIndex] : null;
            string numberedName = $"{rootBaseName}_{groupIndex:D3}.wmo";

            if (!string.IsNullOrEmpty(internalName))
            {
                string potentialInternalPath = Path.Combine(groupsDir, internalName);
                if (File.Exists(potentialInternalPath))
                    groupPathToLoad = potentialInternalPath;
            }
            if (groupPathToLoad == null)
            {
                string potentialNumberedPath = Path.Combine(groupsDir, numberedName);
                if (File.Exists(potentialNumberedPath))
                    groupPathToLoad = potentialNumberedPath;
            }
            return groupPathToLoad;
        }

        /// <summary>
        /// Loads and merges all group meshes for a WMO v14 (monolithic) file.
        /// </summary>
        /// <param name="v14WmoPath">Path to the WMO v14 file.</param>
        /// <returns>The merged WmoGroupMesh, or null if no valid groups were loaded.</returns>
        public static WmoGroupMesh? LoadMergedWmoMeshV14(string v14WmoPath)
        {
            if (!File.Exists(v14WmoPath))
                throw new FileNotFoundException($"WMO v14 file not found: {v14WmoPath}");

            using var stream = File.OpenRead(v14WmoPath);
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            long fileLen = stream.Length;

            // Prepare debug log (move declaration here for full method scope)
            string debugLogPath = Path.Combine(AppContext.BaseDirectory, "output", "wmo_v14", Path.GetFileNameWithoutExtension(v14WmoPath) + "_detailed_momo_debug.txt");
            Directory.CreateDirectory(Path.GetDirectoryName(debugLogPath)!);
            using var debugLog = new StreamWriter(debugLogPath, false);

            // Dump a full chunk map of the file for debugging
            long fileScanPos = 0;
            debugLog.WriteLine("[CHUNKMAP] Full file chunk map:");
            while (fileScanPos + 8 <= fileLen)
            {
                stream.Position = fileScanPos;
                var idBytes = reader.ReadBytes(4);
                if (idBytes.Length < 4) break;
                string idStr = new string(idBytes.Reverse().Select(b => (char)b).ToArray());
                uint size = reader.ReadUInt32();
                long dataPos = stream.Position;
                long nextChunk = dataPos + size;
                debugLog.WriteLine($"[CHUNKMAP] {idStr} at 0x{fileScanPos:X} size {size} (0x{size:X}) next=0x{nextChunk:X}");
                fileScanPos = nextChunk;
            }
            stream.Position = 0; // Reset for normal logic

            // Step 1: Find MOMO chunk
            stream.Position = 0;
            long momoStart = -1, momoEnd = -1;
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint chunkSize = reader.ReadUInt32();
                long chunkDataPos = stream.Position;
                long chunkEnd = chunkDataPos + chunkSize;
                debugLog.WriteLine($"[DEBUG] Top-level chunk: {chunkIdStr} at 0x{chunkStart:X} size {chunkSize} (0x{chunkSize:X})");
                if (chunkIdStr == "MOMO")
                {
                    momoStart = chunkDataPos;
                    momoEnd = chunkEnd;
                    break;
                }
                stream.Position = chunkEnd;
            }
            if (momoStart < 0 || momoEnd < 0)
                throw new InvalidDataException("MOMO chunk not found in v14 WMO file.");

            // Step 2: Scan for MOGP chunks at the top level after MOMO
            var groupMeshes = new List<WmoGroupMesh>();
            stream.Position = momoEnd;
            long filePos = momoEnd;
            WmoGroupMesh? currentMesh = null;
            List<MOPY>? currentMopy = null;
            int groupIndex = 0;
            bool dumpedFirstMogp = false;
            while (filePos + 8 <= fileLen)
            {
                stream.Position = filePos;
                var idBytes = reader.ReadBytes(4);
                if (idBytes.Length < 4) break;
                string idStr = new string(idBytes.Reverse().Select(b => (char)b).ToArray());
                uint size = reader.ReadUInt32();
                long dataPos = stream.Position;
                long nextChunk = dataPos + size;
                debugLog.WriteLine($"[CHUNK] {idStr} at 0x{filePos:X} size {size} (0x{size:X}) next=0x{nextChunk:X}");
                if (idStr == "MOGP")
                {
                    if (currentMesh != null)
                    {
                        debugLog.WriteLine($"[GROUP] Parsed group {groupIndex}: {currentMesh.Vertices.Count} vertices, {currentMesh.Triangles.Count} triangles");
                        groupMeshes.Add(currentMesh);
                        groupIndex++;
                    }
                    currentMesh = new WmoGroupMesh();
                    currentMopy = null;
                    if (!dumpedFirstMogp) {
                        stream.Position = dataPos;
                        var mogpBytes = reader.ReadBytes((int)Math.Min(512, fileLen - dataPos));
                        debugLog.WriteLine($"[HEXDUMP] First 512 bytes after first MOGP at 0x{dataPos:X}:");
                        debugLog.WriteLine(BitConverter.ToString(mogpBytes).Replace("-", " "));
                        // Powerflush: Try different header skips and log the next 64 bytes as ASCII/hex
                        int[] headerSkips = new int[] { 0x40, 0x80, 0xC0, 0x100 };
                        foreach (var skip in headerSkips) {
                            long testOffset = dataPos + skip;
                            stream.Position = testOffset;
                            var testBytes = reader.ReadBytes(64);
                            string ascii = System.Text.Encoding.ASCII.GetString(testBytes).Replace("\0", ".");
                            string hex = BitConverter.ToString(testBytes).Replace("-", " ");
                            debugLog.WriteLine($"[POWERFLUSH] After skipping 0x{skip:X} bytes (offset 0x{testOffset:X}):");
                            debugLog.WriteLine($"[POWERFLUSH] ASCII: {ascii}");
                            debugLog.WriteLine($"[POWERFLUSH] HEX:   {hex}");
                        }
                        dumpedFirstMogp = true;
                    }
                    // Parse all sub-chunks within the MOGP data region
                    long mogpDataStart = dataPos;
                    long mogpHeaderSize = 0x40; // 64 bytes, typical for v14 MOGP
                    long subChunkStart = mogpDataStart + mogpHeaderSize;
                    long mogpDataEnd = dataPos + size;
                    debugLog.WriteLine($"[DEBUG] Skipping MOGP header of {mogpHeaderSize} bytes, sub-chunks start at 0x{subChunkStart:X}");
                    debugLog.WriteLine($"[DEBUG] mogpDataStart=0x{mogpDataStart:X}, mogpHeaderSize=0x{mogpHeaderSize:X}, subChunkStart=0x{subChunkStart:X}, mogpDataEnd=0x{mogpDataEnd:X}");
                    // Log the first 32 bytes at subChunkStart for alignment check
                    stream.Position = subChunkStart;
                    var subChunkPeek = reader.ReadBytes(32);
                    string subChunkPeekAscii = System.Text.Encoding.ASCII.GetString(subChunkPeek).Replace("\0", ".");
                    string subChunkPeekHex = BitConverter.ToString(subChunkPeek).Replace("-", " ");
                    debugLog.WriteLine($"[DEBUG] First 32 bytes at subChunkStart (0x{subChunkStart:X}):");
                    debugLog.WriteLine($"[DEBUG] ASCII: {subChunkPeekAscii}");
                    debugLog.WriteLine($"[DEBUG] HEX:   {subChunkPeekHex}");
                    long subPos = subChunkStart;
                    while (subPos + 8 <= mogpDataEnd)
                    {
                        stream.Position = subPos;
                        var subIdBytes = reader.ReadBytes(4);
                        if (subIdBytes.Length < 4) break;
                        string subIdStr = new string(subIdBytes.Reverse().Select(b => (char)b).ToArray());
                        uint subSize = reader.ReadUInt32();
                        long subDataPos = stream.Position;
                        long subNext = subDataPos + subSize;
                        debugLog.WriteLine($"[SUBCHUNK] {subIdStr} at 0x{subDataPos:X} size {subSize} (0x{subSize:X})");
                        if (subIdStr == "MOVT")
                        {
                            stream.Position = subDataPos;
                            int vtxCount = (int)(subSize / (sizeof(float) * 3));
                            currentMesh.Vertices = MOVT.ReadArray(reader, vtxCount)
                                .Select(pos => new WmoVertex { Position = pos }).ToList();
                            debugLog.WriteLine($"[SUBCHUNK] MOVT: {vtxCount} vertices");
                        }
                        else if (subIdStr == "MONR")
                        {
                            stream.Position = subDataPos;
                            int nrmCount = (int)(subSize / (sizeof(float) * 3));
                            var normals = MONR.ReadArray(reader, nrmCount);
                            for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, normals.Count); i++)
                            {
                                var v = currentMesh.Vertices[i];
                                v.Normal = normals[i];
                                currentMesh.Vertices[i] = v;
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MONR: {nrmCount} normals");
                        }
                        else if (subIdStr == "MOTV")
                        {
                            stream.Position = subDataPos;
                            int uvCount = (int)(subSize / (sizeof(float) * 2));
                            var uvs = MOTV.ReadArray(reader, uvCount);
                            for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, uvs.Count); i++)
                            {
                                var v = currentMesh.Vertices[i];
                                v.UV = uvs[i];
                                currentMesh.Vertices[i] = v;
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MOTV: {uvCount} uvs");
                        }
                        else if (subIdStr == "MOPY")
                        {
                            stream.Position = subDataPos;
                            int mopyCount = (int)(subSize / 2);
                            currentMopy = MOPY.ReadArray(reader, mopyCount);
                            debugLog.WriteLine($"[SUBCHUNK] MOPY: {mopyCount} entries");
                        }
                        else if (subIdStr == "MOIN")
                        {
                            if (currentMopy != null)
                            {
                                int nFaces = currentMopy.Count;
                                for (int i = 0; i < nFaces; i++)
                                {
                                    currentMesh.Triangles.Add(new WmoTriangle
                                    {
                                        Index0 = (ushort)(i * 3 + 0),
                                        Index1 = (ushort)(i * 3 + 1),
                                        Index2 = (ushort)(i * 3 + 2),
                                        MaterialId = currentMopy[i].MaterialId,
                                        Flags = currentMopy[i].Flags
                                    });
                                }
                                debugLog.WriteLine($"[SUBCHUNK] MOIN: {nFaces} faces (sequential indices)");
                            }
                        }
                        else if (subIdStr == "MOVI")
                        {
                            stream.Position = subDataPos;
                            int idxCount = (int)(subSize / sizeof(ushort));
                            var indices = MOVI.ReadArray(reader, idxCount);
                            if (currentMopy != null)
                            {
                                for (int i = 0; i + 2 < indices.Count && i / 3 < currentMopy.Count; i += 3)
                                {
                                    currentMesh.Triangles.Add(new WmoTriangle
                                    {
                                        Index0 = indices[i],
                                        Index1 = indices[i + 1],
                                        Index2 = indices[i + 2],
                                        MaterialId = currentMopy[i / 3].MaterialId,
                                        Flags = currentMopy[i / 3].Flags
                                    });
                                }
                                debugLog.WriteLine($"[SUBCHUNK] MOVI: {indices.Count / 3} faces from {indices.Count} indices");
                            }
                        }
                        subPos = subNext;
                    }
                }
                else if (currentMesh != null)
                {
                    debugLog.WriteLine($"[SUBCHUNK] {idStr} at 0x{dataPos:X} size {size} (0x{size:X})");
                    if (idStr == "MOVT")
                    {
                        stream.Position = dataPos;
                        int vtxCount = (int)(size / (sizeof(float) * 3));
                        currentMesh.Vertices = MOVT.ReadArray(reader, vtxCount)
                            .Select(pos => new WmoVertex { Position = pos }).ToList();
                        debugLog.WriteLine($"[SUBCHUNK] MOVT: {vtxCount} vertices");
                    }
                    else if (idStr == "MONR")
                    {
                        stream.Position = dataPos;
                        int nrmCount = (int)(size / (sizeof(float) * 3));
                        var normals = MONR.ReadArray(reader, nrmCount);
                        for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, normals.Count); i++)
                        {
                            var v = currentMesh.Vertices[i];
                            v.Normal = normals[i];
                            currentMesh.Vertices[i] = v;
                        }
                        debugLog.WriteLine($"[SUBCHUNK] MONR: {nrmCount} normals");
                    }
                    else if (idStr == "MOTV")
                    {
                        stream.Position = dataPos;
                        int uvCount = (int)(size / (sizeof(float) * 2));
                        var uvs = MOTV.ReadArray(reader, uvCount);
                        for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, uvs.Count); i++)
                        {
                            var v = currentMesh.Vertices[i];
                            v.UV = uvs[i];
                            currentMesh.Vertices[i] = v;
                        }
                        debugLog.WriteLine($"[SUBCHUNK] MOTV: {uvCount} uvs");
                    }
                    else if (idStr == "MOPY")
                    {
                        stream.Position = dataPos;
                        int mopyCount = (int)(size / 2);
                        currentMopy = MOPY.ReadArray(reader, mopyCount);
                        debugLog.WriteLine($"[SUBCHUNK] MOPY: {mopyCount} entries");
                    }
                    else if (idStr == "MOIN")
                    {
                        if (currentMopy != null)
                        {
                            int nFaces = currentMopy.Count;
                            for (int i = 0; i < nFaces; i++)
                            {
                                currentMesh.Triangles.Add(new WmoTriangle
                                {
                                    Index0 = (ushort)(i * 3 + 0),
                                    Index1 = (ushort)(i * 3 + 1),
                                    Index2 = (ushort)(i * 3 + 2),
                                    MaterialId = currentMopy[i].MaterialId,
                                    Flags = currentMopy[i].Flags
                                });
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MOIN: {nFaces} faces (sequential indices)");
                        }
                    }
                    else if (idStr == "MOVI")
                    {
                        stream.Position = dataPos;
                        int idxCount = (int)(size / sizeof(ushort));
                        var indices = MOVI.ReadArray(reader, idxCount);
                        if (currentMopy != null)
                        {
                            for (int i = 0; i + 2 < indices.Count && i / 3 < currentMopy.Count; i += 3)
                            {
                                currentMesh.Triangles.Add(new WmoTriangle
                                {
                                    Index0 = indices[i],
                                    Index1 = indices[i + 1],
                                    Index2 = indices[i + 2],
                                    MaterialId = currentMopy[i / 3].MaterialId,
                                    Flags = currentMopy[i / 3].Flags
                                });
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MOVI: {indices.Count / 3} faces from {indices.Count} indices");
                        }
                    }
                }
                filePos = nextChunk;
            }
            if (currentMesh != null)
            {
                debugLog.WriteLine($"[GROUP] Parsed group {groupIndex}: {currentMesh.Vertices.Count} vertices, {currentMesh.Triangles.Count} triangles");
                groupMeshes.Add(currentMesh);
            }
            if (groupMeshes.Count == 0)
            {
                debugLog.WriteLine("[DEBUG] No group meshes found, returning null");
                return null;
            }
            // Export the first group as Ironforge_053.obj, others as Ironforge_053_000.obj, etc.
            try
            {
                string outputDir = Path.Combine(AppContext.BaseDirectory, "output", "wmo_v14");
                Directory.CreateDirectory(outputDir);
                for (int i = 0; i < groupMeshes.Count; i++)
                {
                    var mesh = groupMeshes[i];
                    string objPath;
                    if (i == 0)
                        objPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(v14WmoPath) + ".obj");
                    else
                        objPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(v14WmoPath) + $"_{(i-1):D3}.obj");
                    WmoGroupMesh.SaveToObj(mesh, objPath);
                    debugLog.WriteLine($"[INFO] Exported group {i}: {mesh.Vertices.Count} vertices, {mesh.Triangles.Count} faces to {Path.GetFullPath(objPath)}");
                }
                // Log directory contents after export
                try {
                    var files = Directory.GetFiles(outputDir, Path.GetFileNameWithoutExtension(v14WmoPath) + "*.obj");
                    debugLog.WriteLine($"[INFO] OBJ files in output directory:");
                    foreach (var file in files) debugLog.WriteLine($"[INFO]   {file}");
                } catch (Exception ex) {
                    debugLog.WriteLine($"[ERROR] Could not list OBJ files: {ex.Message}");
                }
                debugLog.WriteLine($"[INFO] Total groups exported: {groupMeshes.Count}");
            }
            catch (Exception ex)
            {
                debugLog.WriteLine($"[ERROR] Exception exporting group OBJs: {ex.Message}");
            }
            // Optionally return the first group mesh for compatibility
            return groupMeshes[0];
        }

        /// <summary>
        /// Loads geometry from all MOMO chunks in a v14 WMO file, merging all geometry sets.
        /// </summary>
        /// <param name="v14WmoPath">Path to the WMO v14 file.</param>
        /// <returns>The constructed WmoGroupMesh, or null if no geometry found.</returns>
        public static WmoGroupMesh? LoadAllMomoChunksV14Mesh(string v14WmoPath)
        {
            if (!File.Exists(v14WmoPath))
                throw new FileNotFoundException($"WMO v14 file not found: {v14WmoPath}");

            Console.WriteLine($"[DEBUG] Starting v14 export for: {v14WmoPath}");
            // Prepare debug log (move declaration here for full method scope)
            string debugLogPath = Path.Combine(AppContext.BaseDirectory, "output", "wmo_v14", Path.GetFileNameWithoutExtension(v14WmoPath) + "_detailed_momo_debug.txt");
            Directory.CreateDirectory(Path.GetDirectoryName(debugLogPath)!);
            using var debugLog = new StreamWriter(debugLogPath, false);
            debugLog.WriteLine($"[DEBUG] Starting v14 export for: {v14WmoPath}");

            using var stream = File.OpenRead(v14WmoPath);
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            long fileLen = stream.Length;

            // Dump a full chunk map of the file for debugging
            long fileScanPos = 0;
            debugLog.WriteLine("[CHUNKMAP] Full file chunk map:");
            while (fileScanPos + 8 <= fileLen)
            {
                stream.Position = fileScanPos;
                var idBytes = reader.ReadBytes(4);
                if (idBytes.Length < 4) break;
                string idStr = new string(idBytes.Reverse().Select(b => (char)b).ToArray());
                uint size = reader.ReadUInt32();
                long dataPos = stream.Position;
                long nextChunk = dataPos + size;
                debugLog.WriteLine($"[CHUNKMAP] {idStr} at 0x{fileScanPos:X} size {size} (0x{size:X}) next=0x{nextChunk:X}");
                fileScanPos = nextChunk;
            }
            stream.Position = 0; // Reset for normal logic

            // Step 1: Find MOMO chunk
            stream.Position = 0;
            long momoStart = -1, momoEnd = -1;
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint chunkSize = reader.ReadUInt32();
                long chunkDataPos = stream.Position;
                long chunkEnd = chunkDataPos + chunkSize;
                debugLog.WriteLine($"[DEBUG] Top-level chunk: {chunkIdStr} at 0x{chunkStart:X} size {chunkSize} (0x{chunkSize:X})");
                if (chunkIdStr == "MOMO")
                {
                    momoStart = chunkDataPos;
                    momoEnd = chunkEnd;
                    break;
                }
                stream.Position = chunkEnd;
            }
            if (momoStart < 0 || momoEnd < 0)
                throw new InvalidDataException("MOMO chunk not found in v14 WMO file.");

            // Step 2: Scan for MOGP chunks at the top level after MOMO
            var groupMeshes = new List<WmoGroupMesh>();
            stream.Position = momoEnd;
            long filePos = momoEnd;
            WmoGroupMesh? currentMesh = null;
            List<MOPY>? currentMopy = null;
            int groupIndex = 0;
            bool dumpedFirstMogp = false;
            while (filePos + 8 <= fileLen)
            {
                stream.Position = filePos;
                var idBytes = reader.ReadBytes(4);
                if (idBytes.Length < 4) break;
                string idStr = new string(idBytes.Reverse().Select(b => (char)b).ToArray());
                uint size = reader.ReadUInt32();
                long dataPos = stream.Position;
                long nextChunk = dataPos + size;
                debugLog.WriteLine($"[CHUNK] {idStr} at 0x{filePos:X} size {size} (0x{size:X}) next=0x{nextChunk:X}");
                if (idStr == "MOGP")
                {
                    if (currentMesh != null)
                    {
                        debugLog.WriteLine($"[GROUP] Parsed group {groupIndex}: {currentMesh.Vertices.Count} vertices, {currentMesh.Triangles.Count} triangles");
                        groupMeshes.Add(currentMesh);
                        groupIndex++;
                    }
                    currentMesh = new WmoGroupMesh();
                    currentMopy = null;
                    if (!dumpedFirstMogp) {
                        stream.Position = dataPos;
                        var mogpBytes = reader.ReadBytes((int)Math.Min(512, fileLen - dataPos));
                        debugLog.WriteLine($"[HEXDUMP] First 512 bytes after first MOGP at 0x{dataPos:X}:");
                        debugLog.WriteLine(BitConverter.ToString(mogpBytes).Replace("-", " "));
                        // Powerflush: Try different header skips and log the next 64 bytes as ASCII/hex
                        int[] headerSkips = new int[] { 0x40, 0x80, 0xC0, 0x100 };
                        foreach (var skip in headerSkips) {
                            long testOffset = dataPos + skip;
                            stream.Position = testOffset;
                            var testBytes = reader.ReadBytes(64);
                            string ascii = System.Text.Encoding.ASCII.GetString(testBytes).Replace("\0", ".");
                            string hex = BitConverter.ToString(testBytes).Replace("-", " ");
                            debugLog.WriteLine($"[POWERFLUSH] After skipping 0x{skip:X} bytes (offset 0x{testOffset:X}):");
                            debugLog.WriteLine($"[POWERFLUSH] ASCII: {ascii}");
                            debugLog.WriteLine($"[POWERFLUSH] HEX:   {hex}");
                        }
                        dumpedFirstMogp = true;
                    }
                    // Parse all sub-chunks within the MOGP data region
                    long mogpDataStart = dataPos;
                    long mogpHeaderSize = 0x40; // 64 bytes, typical for v14 MOGP
                    long subChunkStart = mogpDataStart + mogpHeaderSize;
                    long mogpDataEnd = dataPos + size;
                    debugLog.WriteLine($"[DEBUG] Skipping MOGP header of {mogpHeaderSize} bytes, sub-chunks start at 0x{subChunkStart:X}");
                    debugLog.WriteLine($"[DEBUG] mogpDataStart=0x{mogpDataStart:X}, mogpHeaderSize=0x{mogpHeaderSize:X}, subChunkStart=0x{subChunkStart:X}, mogpDataEnd=0x{mogpDataEnd:X}");
                    // Log the first 32 bytes at subChunkStart for alignment check
                    stream.Position = subChunkStart;
                    var subChunkPeek = reader.ReadBytes(32);
                    string subChunkPeekAscii = System.Text.Encoding.ASCII.GetString(subChunkPeek).Replace("\0", ".");
                    string subChunkPeekHex = BitConverter.ToString(subChunkPeek).Replace("-", " ");
                    debugLog.WriteLine($"[DEBUG] First 32 bytes at subChunkStart (0x{subChunkStart:X}):");
                    debugLog.WriteLine($"[DEBUG] ASCII: {subChunkPeekAscii}");
                    debugLog.WriteLine($"[DEBUG] HEX:   {subChunkPeekHex}");
                    long subPos = subChunkStart;
                    while (subPos + 8 <= mogpDataEnd)
                    {
                        stream.Position = subPos;
                        var subIdBytes = reader.ReadBytes(4);
                        if (subIdBytes.Length < 4) break;
                        string subIdStr = new string(subIdBytes.Reverse().Select(b => (char)b).ToArray());
                        uint subSize = reader.ReadUInt32();
                        long subDataPos = stream.Position;
                        long subNext = subDataPos + subSize;
                        debugLog.WriteLine($"[SUBCHUNK] {subIdStr} at 0x{subDataPos:X} size {subSize} (0x{subSize:X})");
                        if (subIdStr == "MOVT")
                        {
                            stream.Position = subDataPos;
                            int vtxCount = (int)(subSize / (sizeof(float) * 3));
                            currentMesh.Vertices = MOVT.ReadArray(reader, vtxCount)
                                .Select(pos => new WmoVertex { Position = pos }).ToList();
                            debugLog.WriteLine($"[SUBCHUNK] MOVT: {vtxCount} vertices");
                        }
                        else if (subIdStr == "MONR")
                        {
                            stream.Position = subDataPos;
                            int nrmCount = (int)(subSize / (sizeof(float) * 3));
                            var normals = MONR.ReadArray(reader, nrmCount);
                            for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, normals.Count); i++)
                            {
                                var v = currentMesh.Vertices[i];
                                v.Normal = normals[i];
                                currentMesh.Vertices[i] = v;
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MONR: {nrmCount} normals");
                        }
                        else if (subIdStr == "MOTV")
                        {
                            stream.Position = subDataPos;
                            int uvCount = (int)(subSize / (sizeof(float) * 2));
                            var uvs = MOTV.ReadArray(reader, uvCount);
                            for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, uvs.Count); i++)
                            {
                                var v = currentMesh.Vertices[i];
                                v.UV = uvs[i];
                                currentMesh.Vertices[i] = v;
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MOTV: {uvCount} uvs");
                        }
                        else if (subIdStr == "MOPY")
                        {
                            stream.Position = subDataPos;
                            int mopyCount = (int)(subSize / 2);
                            currentMopy = MOPY.ReadArray(reader, mopyCount);
                            debugLog.WriteLine($"[SUBCHUNK] MOPY: {mopyCount} entries");
                        }
                        else if (subIdStr == "MOIN")
                        {
                            if (currentMopy != null)
                            {
                                int nFaces = currentMopy.Count;
                                for (int i = 0; i < nFaces; i++)
                                {
                                    currentMesh.Triangles.Add(new WmoTriangle
                                    {
                                        Index0 = (ushort)(i * 3 + 0),
                                        Index1 = (ushort)(i * 3 + 1),
                                        Index2 = (ushort)(i * 3 + 2),
                                        MaterialId = currentMopy[i].MaterialId,
                                        Flags = currentMopy[i].Flags
                                    });
                                }
                                debugLog.WriteLine($"[SUBCHUNK] MOIN: {nFaces} faces (sequential indices)");
                            }
                        }
                        else if (subIdStr == "MOVI")
                        {
                            stream.Position = subDataPos;
                            int idxCount = (int)(subSize / sizeof(ushort));
                            var indices = MOVI.ReadArray(reader, idxCount);
                            if (currentMopy != null)
                            {
                                for (int i = 0; i + 2 < indices.Count && i / 3 < currentMopy.Count; i += 3)
                                {
                                    currentMesh.Triangles.Add(new WmoTriangle
                                    {
                                        Index0 = indices[i],
                                        Index1 = indices[i + 1],
                                        Index2 = indices[i + 2],
                                        MaterialId = currentMopy[i / 3].MaterialId,
                                        Flags = currentMopy[i / 3].Flags
                                    });
                                }
                                debugLog.WriteLine($"[SUBCHUNK] MOVI: {indices.Count / 3} faces from {indices.Count} indices");
                            }
                        }
                        subPos = subNext;
                    }
                }
                else if (currentMesh != null)
                {
                    debugLog.WriteLine($"[SUBCHUNK] {idStr} at 0x{dataPos:X} size {size} (0x{size:X})");
                    if (idStr == "MOVT")
                    {
                        stream.Position = dataPos;
                        int vtxCount = (int)(size / (sizeof(float) * 3));
                        currentMesh.Vertices = MOVT.ReadArray(reader, vtxCount)
                            .Select(pos => new WmoVertex { Position = pos }).ToList();
                        debugLog.WriteLine($"[SUBCHUNK] MOVT: {vtxCount} vertices");
                    }
                    else if (idStr == "MONR")
                    {
                        stream.Position = dataPos;
                        int nrmCount = (int)(size / (sizeof(float) * 3));
                        var normals = MONR.ReadArray(reader, nrmCount);
                        for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, normals.Count); i++)
                        {
                            var v = currentMesh.Vertices[i];
                            v.Normal = normals[i];
                            currentMesh.Vertices[i] = v;
                        }
                        debugLog.WriteLine($"[SUBCHUNK] MONR: {nrmCount} normals");
                    }
                    else if (idStr == "MOTV")
                    {
                        stream.Position = dataPos;
                        int uvCount = (int)(size / (sizeof(float) * 2));
                        var uvs = MOTV.ReadArray(reader, uvCount);
                        for (int i = 0; i < Math.Min(currentMesh.Vertices.Count, uvs.Count); i++)
                        {
                            var v = currentMesh.Vertices[i];
                            v.UV = uvs[i];
                            currentMesh.Vertices[i] = v;
                        }
                        debugLog.WriteLine($"[SUBCHUNK] MOTV: {uvCount} uvs");
                    }
                    else if (idStr == "MOPY")
                    {
                        stream.Position = dataPos;
                        int mopyCount = (int)(size / 2);
                        currentMopy = MOPY.ReadArray(reader, mopyCount);
                        debugLog.WriteLine($"[SUBCHUNK] MOPY: {mopyCount} entries");
                    }
                    else if (idStr == "MOIN")
                    {
                        if (currentMopy != null)
                        {
                            int nFaces = currentMopy.Count;
                            for (int i = 0; i < nFaces; i++)
                            {
                                currentMesh.Triangles.Add(new WmoTriangle
                                {
                                    Index0 = (ushort)(i * 3 + 0),
                                    Index1 = (ushort)(i * 3 + 1),
                                    Index2 = (ushort)(i * 3 + 2),
                                    MaterialId = currentMopy[i].MaterialId,
                                    Flags = currentMopy[i].Flags
                                });
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MOIN: {nFaces} faces (sequential indices)");
                        }
                    }
                    else if (idStr == "MOVI")
                    {
                        stream.Position = dataPos;
                        int idxCount = (int)(size / sizeof(ushort));
                        var indices = MOVI.ReadArray(reader, idxCount);
                        if (currentMopy != null)
                        {
                            for (int i = 0; i + 2 < indices.Count && i / 3 < currentMopy.Count; i += 3)
                            {
                                currentMesh.Triangles.Add(new WmoTriangle
                                {
                                    Index0 = indices[i],
                                    Index1 = indices[i + 1],
                                    Index2 = indices[i + 2],
                                    MaterialId = currentMopy[i / 3].MaterialId,
                                    Flags = currentMopy[i / 3].Flags
                                });
                            }
                            debugLog.WriteLine($"[SUBCHUNK] MOVI: {indices.Count / 3} faces from {indices.Count} indices");
                        }
                    }
                }
                filePos = nextChunk;
            }
            if (currentMesh != null)
            {
                debugLog.WriteLine($"[GROUP] Parsed group {groupIndex}: {currentMesh.Vertices.Count} vertices, {currentMesh.Triangles.Count} triangles");
                groupMeshes.Add(currentMesh);
            }
            if (groupMeshes.Count == 0)
            {
                debugLog.WriteLine("[DEBUG] No group meshes found, returning null");
                return null;
            }
            var mergedMesh = WmoGroupMesh.MergeMeshes(groupMeshes);
            // Debug output: log total vertices/triangles
            try
            {
                string outputDir = Path.Combine(AppContext.BaseDirectory, "output", "wmo_v14");
                Directory.CreateDirectory(outputDir);
                string debugPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(v14WmoPath) + "_mesh_debug.txt");
                using (var debug = new StreamWriter(debugPath, false))
                {
                    debug.WriteLine($"[INFO] Group meshes found: {groupMeshes.Count}");
                    debug.WriteLine($"[INFO] Total vertices: {mergedMesh.Vertices.Count}");
                    debug.WriteLine($"[INFO] Total triangles: {mergedMesh.Triangles.Count}");
                }
                Console.WriteLine($"[DEBUG] Wrote geometry sets debug file: {debugPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DEBUG] Exception writing geometry sets debug file: {ex.Message}");
            }
            Console.WriteLine("[DEBUG] Returning merged mesh from LoadAllMomoChunksV14Mesh");
            debugLog.WriteLine($"[DEBUG] Returning merged mesh from LoadAllMomoChunksV14Mesh");
            return mergedMesh;
        }
    }
} 