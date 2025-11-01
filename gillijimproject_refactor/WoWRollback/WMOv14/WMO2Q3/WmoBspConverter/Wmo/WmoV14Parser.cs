using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using WmoBspConverter.Bsp;

namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// WMO v14 parser that handles the special MOMO container structure.
    /// Based on the proven V14ChunkReader approach from old_sources.
    /// </summary>
    public class WmoV14Parser
    {
        public class ChunkInfo
        {
            public string Id { get; set; } = string.Empty;
            public uint Offset { get; set; }
            public uint Size { get; set; }
            public byte[] Data { get; set; } = Array.Empty<byte>();
        }

        public class WmoV14Data
        {
            public uint Version { get; set; }
            public List<ChunkInfo> Chunks { get; set; } = new();
            public List<string> Textures { get; set; } = new();
            public List<WmoGroupData> Groups { get; set; } = new();
            public byte[] FileBytes { get; set; } = Array.Empty<byte>();
        }

        public class WmoGroupData
        {
            public string Name { get; set; } = string.Empty;
            public List<Vector3> Vertices { get; set; } = new();
            public List<ushort> Indices { get; set; } = new();
            public List<byte> FaceMaterials { get; set; } = new();
            public List<Vector2> UVs { get; set; } = new();
            public uint Flags { get; set; }
        }

        /// <summary>
        /// Parse WMO v14 file and extract all relevant data
        /// </summary>
        public WmoV14Data ParseWmoV14(string filePath)
        {
            using var fileStream = File.OpenRead(filePath);
            return ParseWmoV14(fileStream);
        }

        /// <summary>
        /// Parse WMO v14 from stream
        /// </summary>
        public WmoV14Data ParseWmoV14(Stream stream)
        {
            // Snapshot entire file for region slicing
            byte[] fileBytes;
            using (var ms = new MemoryStream())
            {
                stream.Position = 0;
                stream.CopyTo(ms);
                fileBytes = ms.ToArray();
            }
            using var mem = new MemoryStream(fileBytes, writable: false);
            var reader = new BinaryReader(mem, Encoding.ASCII, leaveOpen: true);
            mem.Position = 0;

            var wmoData = new WmoV14Data();
            wmoData.FileBytes = fileBytes;

            // Read version first
            var versionChunk = ReadChunk(reader);
            if (versionChunk.Id != "MVER")
                throw new InvalidDataException($"Expected MVER chunk, got {versionChunk.Id}");

            wmoData.Version = BitConverter.ToUInt32(versionChunk.Data, 0);
            if (wmoData.Version != 14)
                throw new InvalidDataException($"Expected WMO v14, got v{wmoData.Version}");

            Console.WriteLine($"[DEBUG] Parsing WMO version {wmoData.Version}");

            // Read all remaining chunks
            while (mem.Position < mem.Length)
            {
                var chunk = ReadChunk(reader);
                if (chunk.Id == "MOMO")
                {
                    // Parse sub-chunks inside MOMO container with absolute base offset
                    ParseMomoContainer(chunk.Data, chunk.Offset, wmoData);
                }
                else
                {
                    wmoData.Chunks.Add(chunk);
                }
            }

            // Process parsed chunks to extract useful data
            ProcessChunks(wmoData);

            return wmoData;
        }

        private ChunkInfo ReadChunk(BinaryReader reader)
        {
            var chunkIdBytes = reader.ReadBytes(4);
            Array.Reverse(chunkIdBytes);
            var chunkId = Encoding.ASCII.GetString(chunkIdBytes);

            var chunkSize = reader.ReadUInt32();
            var dataStart = reader.BaseStream.Position;
            var data = reader.ReadBytes((int)chunkSize);

            return new ChunkInfo
            {
                Id = chunkId,
                Offset = (uint)(dataStart - 8),
                Size = chunkSize,
                Data = data
            };
        }

        private void ParseMomoContainer(byte[] momoData, uint baseOffset, WmoV14Data wmoData)
        {
            using var momoStream = new MemoryStream(momoData);
            var momoReader = new BinaryReader(momoStream, Encoding.ASCII, leaveOpen: true);

            while (momoStream.Position + 8 <= momoStream.Length)
            {
                var chunkIdBytes = momoReader.ReadBytes(4);
                Array.Reverse(chunkIdBytes);
                var chunkId = Encoding.ASCII.GetString(chunkIdBytes);

                var chunkSize = momoReader.ReadUInt32();
                var dataStart = momoStream.Position;

                if (dataStart + chunkSize > momoStream.Length)
                    break; // corrupted chunk

                var data = momoReader.ReadBytes((int)chunkSize);
                wmoData.Chunks.Add(new ChunkInfo
                {
                    Id = chunkId,
                    // Absolute offset to subchunk header (ID field) within full file
                    Offset = baseOffset + (uint)(dataStart - 8),
                    Size = chunkSize,
                    Data = data
                });

                Console.WriteLine($"[DEBUG] Found MOMO subchunk: {chunkId} ({chunkSize} bytes)");

                momoStream.Position = dataStart + chunkSize;
            }
        }

        private void ProcessChunks(WmoV14Data wmoData)
        {
            Console.WriteLine($"[DEBUG] Processing {wmoData.Chunks.Count} total chunks");

            // Extract textures (MOTX chunk)
            var textureChunk = wmoData.Chunks.FirstOrDefault(c => c.Id == "MOTX");
            if (textureChunk?.Data != null)
            {
                wmoData.Textures = ParseTextureNames(textureChunk.Data);
            }

            // Build groups by slicing file into regions: each MOGP header → next MOGP (or EOF)
            wmoData.Groups.Clear();
            var mogpIndices = wmoData.Chunks
                .Select((c, i) => new { c, i })
                .Where(x => x.c.Id == "MOGP")
                .Select(x => x.i)
                .ToList();

            for (int gi = 0; gi < mogpIndices.Count; gi++)
            {
                var start = (int)wmoData.Chunks[mogpIndices[gi]].Offset;
                var end = (gi + 1 < mogpIndices.Count)
                    ? (int)wmoData.Chunks[mogpIndices[gi + 1]].Offset
                    : wmoData.FileBytes.Length;
                if (end > wmoData.FileBytes.Length) end = wmoData.FileBytes.Length;
                if (end <= start) continue;

                var region = new byte[end - start];
                Buffer.BlockCopy(wmoData.FileBytes, start, region, 0, region.Length);

                var group = new WmoGroupData { Name = $"group_{gi}" };
                ProcessMogpRegion(region, group);
                wmoData.Groups.Add(group);
            }

            // Fallback: if no groups parsed, try any top-level MOVT/MOVI/MOPY into a single group
            if (wmoData.Groups.Count == 0)
            {
                var g = new WmoGroupData { Name = "group_0" };
                bool any = false;
                foreach (var c in wmoData.Chunks)
                {
                    if (c.Id == "MOVT") { ProcessMovtChunk(c.Data, g); any = true; }
                    else if (c.Id == "MOVI") { ProcessMoviChunk(c.Data, g); any = true; }
                    else if (c.Id == "MOPY") { ProcessMopyChunk(c.Data, g); any = true; }
                }
                if (any) wmoData.Groups.Add(g);
            }
        }

        private List<string> ParseTextureNames(byte[] textureData)
        {
            var textures = new List<string>();
            var str = Encoding.ASCII.GetString(textureData);
            var parts = str.Split('\0', StringSplitOptions.RemoveEmptyEntries);

            foreach (var texture in parts)
            {
                if (!string.IsNullOrWhiteSpace(texture))
                {
                    textures.Add(texture);
                }
            }

            return textures;
        }

        private List<string> ParseGroupNames(byte[]? groupNamesData)
        {
            if (groupNamesData == null)
                return new List<string>();

            var names = new List<string>();
            var str = Encoding.ASCII.GetString(groupNamesData);
            var parts = str.Split('\0', StringSplitOptions.RemoveEmptyEntries);

            foreach (var name in parts)
            {
                if (!string.IsNullOrWhiteSpace(name))
                {
                    names.Add(name);
                }
            }

            return names;
        }

        private void ProcessGroups(byte[] groupInfoData, List<string> groupNames, WmoV14Data wmoData)
        {
            // WMO v14 group info structure: 32 bytes per group
            // uint32_t flags, CAaBox bounding_box, int32_t nameoffset
            const int GROUP_INFO_SIZE = 32;

            var groupCount = groupInfoData.Length / GROUP_INFO_SIZE;

            for (int i = 0; i < groupCount; i++)
            {
                var offset = i * GROUP_INFO_SIZE;

                var flags = BitConverter.ToUInt32(groupInfoData, offset);
                var nameOffset = BitConverter.ToInt32(groupInfoData, offset + 28); // offset + 0x1C

                var groupName = $"group_{i}";
                if (nameOffset >= 0 && nameOffset < groupNames.Count)
                {
                    groupName = groupNames[nameOffset] ?? groupName;
                }

                var groupData = new WmoGroupData
                {
                    Name = groupName,
                    Flags = flags
                };

                wmoData.Groups.Add(groupData);
            }

            Console.WriteLine($"[DEBUG] Processed {groupCount} groups");
        }

        /// <summary>
        /// Convert WMO v14 data to BSP format
        /// </summary>
        public BspFile ConvertToBsp(WmoV14Data wmoData)
        {
            var bspFile = new BspFile();

            // Add textures
            foreach (var textureName in wmoData.Textures)
            {
                var bspTexture = new BspTexture
                {
                    Name = textureName,
                    Flags = 0
                };
                bspFile.Textures.Add(bspTexture);
            }

            // Convert groups to BSP geometry using parsed WMO data
            foreach (var group in wmoData.Groups)
            {
                Console.WriteLine($"[DEBUG] Processing group: {group.Name}");
                Console.WriteLine($"[DEBUG] Group has {group.Vertices.Count} vertices, {group.Indices.Count} indices");

                // Determine triangle indices: use MOVI if present; otherwise fall back to sequential triples using MOPY face count
                List<(ushort a, ushort b, ushort c)> tris = new List<(ushort, ushort, ushort)>();
                if (group.Indices.Count >= 3)
                {
                    for (int i = 0; i + 2 < group.Indices.Count; i += 3)
                    {
                        tris.Add((group.Indices[i], group.Indices[i + 1], group.Indices[i + 2]));
                    }
                }
                else if (group.FaceMaterials.Count > 0 && group.Vertices.Count >= 3)
                {
                    int faceCount = group.FaceMaterials.Count; // already parsed as faces, not bytes
                    int maxTris = Math.Min(faceCount, group.Vertices.Count / 3);
                    for (int t = 0; t < maxTris; t++)
                    {
                        int baseIdx = t * 3;
                        tris.Add(((ushort)baseIdx, (ushort)(baseIdx + 1), (ushort)(baseIdx + 2)));
                    }
                    Console.WriteLine($"[DEBUG] Fallback triangles assembled from sequential triples: {tris.Count}");
                }

                // For each triangle, duplicate its three vertices into the BSP Vertices list and fill MeshVertices
                foreach (var (a, b, c) in tris)
                {
                    if (a >= group.Vertices.Count || b >= group.Vertices.Count || c >= group.Vertices.Count)
                        continue;

                    var start = bspFile.Vertices.Count;

                    var p0 = group.Vertices[a];
                    var p1 = group.Vertices[b];
                    var p2 = group.Vertices[c];

                    bspFile.Vertices.Add(new BspVertex { Position = p0, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });
                    bspFile.Vertices.Add(new BspVertex { Position = p1, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });
                    bspFile.Vertices.Add(new BspVertex { Position = p2, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });

                    // Mesh indices for this triangle (relative to FirstVertex)
                    int meshStart = bspFile.MeshVertices.Count;
                    bspFile.MeshVertices.Add(0);
                    bspFile.MeshVertices.Add(1);
                    bspFile.MeshVertices.Add(2);

                    bspFile.Faces.Add(new BspFace
                    {
                        Texture = 0,
                        Effect = -1,
                        Type = 3, // mesh with MeshVertices
                        FirstVertex = start,
                        NumVertices = 3,
                        FirstMeshVertex = meshStart,
                        NumMeshVertices = 3,
                        Lightmap = -1
                    });
                }
            }

            Console.WriteLine($"[DEBUG] Converted to BSP: {bspFile.Vertices.Count} vertices, {bspFile.Faces.Count} faces, {bspFile.Textures.Count} textures");

            return bspFile;
        }

        private void ProcessMogpRegion(byte[] regionData, WmoGroupData groupData)
        {
            using var ms = new MemoryStream(regionData, writable: false);
            using var br = new BinaryReader(ms, Encoding.ASCII, leaveOpen: true);

            // Skip v14 MOGP header (0x40 bytes)
            const int MOGP_HEADER_SIZE = 0x40;
            if (ms.Length < MOGP_HEADER_SIZE) return;
            ms.Position = MOGP_HEADER_SIZE;

            // Scan forward for first valid subchunk header after MOGP header
            string[] valid = new[] { "MOPY", "MOVT", "MOVI", "MONR", "MOTV", "MOBA", "MOBN", "MOBR" };
            long firstPos = -1;
            for (long off = MOGP_HEADER_SIZE; off + 4 <= ms.Length; off++)
            {
                ms.Position = off;
                var idBytes = br.ReadBytes(4);
                if (idBytes.Length < 4) break;
                var id = Encoding.ASCII.GetString(idBytes.Reverse().ToArray());
                if (valid.Contains(id)) { firstPos = off; break; }
            }
            if (firstPos == -1) return;
            ms.Position = firstPos;

            while (ms.Position + 8 <= ms.Length)
            {
                long subStart = ms.Position;
                var idBytes = br.ReadBytes(4);
                if (idBytes.Length < 4) break;
                var id = Encoding.ASCII.GetString(idBytes.Reverse().ToArray());
                var size = br.ReadUInt32();
                long dataPos = ms.Position;
                long subEnd = dataPos + size;
                if (subEnd > ms.Length) break;

                var data = br.ReadBytes((int)size);
                switch (id)
                {
                    case "MOVT": ProcessMovtChunk(data, groupData); break;
                    case "MOVI": ProcessMoviChunk(data, groupData); break;
                    case "MOPY": ProcessMopyChunk(data, groupData); break;
                }
                ms.Position = subEnd;

                // Realign to next valid subchunk header if necessary (scan up to 16 bytes)
                if (ms.Position + 4 <= ms.Length)
                {
                    long searchStart = ms.Position;
                    ms.Position = searchStart;
                    var nextBytes = br.ReadBytes(4);
                    ms.Position = searchStart;
                    var nextId = nextBytes.Length == 4 ? Encoding.ASCII.GetString(nextBytes.Reverse().ToArray()) : string.Empty;
                    if (!valid.Contains(nextId))
                    {
                        bool found = false;
                        for (int s = 1; s <= 16 && searchStart + s + 4 <= ms.Length; s++)
                        {
                            ms.Position = searchStart + s;
                            var scanBytes = br.ReadBytes(4);
                            ms.Position = searchStart + s;
                            var scanId = scanBytes.Length == 4 ? Encoding.ASCII.GetString(scanBytes.Reverse().ToArray()) : string.Empty;
                            if (valid.Contains(scanId)) { found = true; break; }
                        }
                        if (!found) ms.Position = searchStart; // continue sequentially
                    }
                }
            }
        }

        private void ProcessMovtChunk(byte[] movtData, WmoGroupData groupData)
        {
            // MOVT: 3D vertices (12 bytes per vertex: x, y, z floats)
            const int VERTEX_SIZE = 12;
            var vertexCount = movtData.Length / VERTEX_SIZE;
            
            groupData.Vertices.Clear();
            
            for (int i = 0; i < vertexCount; i++)
            {
                var offset = i * VERTEX_SIZE;
                // v14 axis mapping aligned with PoC: x, z, -y
                var x = BitConverter.ToSingle(movtData, offset);
                var z = BitConverter.ToSingle(movtData, offset + 4);
                var y = -BitConverter.ToSingle(movtData, offset + 8);
                groupData.Vertices.Add(new Vector3(x, y, z));
            }
            
            Console.WriteLine($"[DEBUG] Extracted {vertexCount} vertices from MOVT");
        }

        private void ProcessMoviChunk(byte[] moviData, WmoGroupData groupData)
        {
            // MOVI: Face indices (2 bytes per index)
            const int INDEX_SIZE = 2;
            var indexCount = moviData.Length / INDEX_SIZE;
            
            groupData.Indices.Clear();
            
            for (int i = 0; i < indexCount; i++)
            {
                var offset = i * INDEX_SIZE;
                groupData.Indices.Add(BitConverter.ToUInt16(moviData, offset));
            }
            
            Console.WriteLine($"[DEBUG] Extracted {indexCount} indices from MOVI");
        }

        private void ProcessMopyChunk(byte[] mopyData, WmoGroupData groupData)
        {
            // MOPY: 2 bytes per face: flags, materialId
            int faceCount = mopyData.Length / 2;
            groupData.FaceMaterials.Clear();
            for (int i = 0; i < faceCount; i++)
            {
                byte flags = mopyData[i * 2 + 0];
                byte matId = mopyData[i * 2 + 1];
                groupData.FaceMaterials.Add(matId);
            }
            Console.WriteLine($"[DEBUG] Extracted {faceCount} faces from MOPY");
        }
        private void ProcessTopLevelMovtChunk(byte[] movtData, WmoGroupData? groupData)
        {
            if (groupData == null) return;

            // MOVT: 3D vertices (12 bytes per vertex: x, y, z floats)
            const int VERTEX_SIZE = 12;
            var vertexCount = movtData.Length / VERTEX_SIZE;
            
            groupData.Vertices.Clear();
            
            for (int i = 0; i < vertexCount; i++)
            {
                var offset = i * VERTEX_SIZE;
                var x = BitConverter.ToSingle(movtData, offset);
                var y = BitConverter.ToSingle(movtData, offset + 4);
                var z = BitConverter.ToSingle(movtData, offset + 8);
                
                groupData.Vertices.Add(new Vector3(x, y, z));
            }
            
            Console.WriteLine($"[DEBUG] Extracted {vertexCount} vertices from top-level MOVT");
        }

        private void ProcessTopLevelMoviChunk(byte[] moviData, WmoGroupData? groupData)
        {
            if (groupData == null) return;

            // MOVI: Face indices (2 bytes per index)
            const int INDEX_SIZE = 2;
            var indexCount = moviData.Length / INDEX_SIZE;
            
            groupData.Indices.Clear();
            
            for (int i = 0; i < indexCount; i++)
            {
                var offset = i * INDEX_SIZE;
                groupData.Indices.Add(BitConverter.ToUInt16(moviData, offset));
            }
            
            Console.WriteLine($"[DEBUG] Extracted {indexCount} indices from top-level MOVI");
        }

        private void ProcessTopLevelMopyChunk(byte[] mopyData, WmoGroupData? groupData)
        {
            if (groupData == null) return;

            // MOPY: Face material assignments (1 byte per face)
            var faceCount = mopyData.Length;
            
            groupData.FaceMaterials.Clear();
            for (int i = 0; i < faceCount; i++)
            {
                groupData.FaceMaterials.Add(mopyData[i]);
            }
            
            Console.WriteLine($"[DEBUG] Extracted {faceCount} face materials from top-level MOPY");
        }
    }
}