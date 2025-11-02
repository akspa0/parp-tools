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

        private void ProcessMobaChunk(byte[] mobaData, WmoGroupData groupData)
        {
            // MOBA: render batches. V14 uses 24 bytes per entry
            // V14 structure: lightMap(1), texture(1), boundingBox(12), startIndex(2), count(2), minIndex(2), maxIndex(2), flags(1), padding(1)
            const int ENTRY_SIZE = 24;
            int count = mobaData.Length / ENTRY_SIZE;
            if (count <= 0) return;
            
            for (int i = 0; i < count; i++)
            {
                int o = i * ENTRY_SIZE;
                byte lightMap = mobaData[o + 0];
                byte texture = mobaData[o + 1];           // Material index in v14
                // Skip bounding box (12 bytes at offset 2-13)
                ushort startIndex = BitConverter.ToUInt16(mobaData, o + 14);
                ushort numIndices = BitConverter.ToUInt16(mobaData, o + 16);
                ushort minIndex = BitConverter.ToUInt16(mobaData, o + 18);
                ushort maxIndex = BitConverter.ToUInt16(mobaData, o + 20);
                byte flags = mobaData[o + 22];
                // byte 23 is padding

                groupData.Batches.Add(new MobaBatch
                {
                    FirstFace = startIndex,               // Index into MOVI
                    NumFaces = numIndices,                // Number of MOVI indices (not triangles!)
                    FirstVertex = minIndex,
                    LastVertex = maxIndex,
                    Flags = flags,
                    MaterialId = texture,                 // V14 uses 'texture' field as material ID
                    PossibleBox2Z = 0                     // Not present in v14
                });
            }
            Console.WriteLine($"[DEBUG] Extracted {count} MOBA batches (v14 format: 24 bytes each)");
            
            // Log first few batches for debugging with hex dump
            for (int i = 0; i < Math.Min(count, 3); i++)
            {
                var b = groupData.Batches[groupData.Batches.Count - count + i];
                int o = i * ENTRY_SIZE;
                string hex = BitConverter.ToString(mobaData, o, Math.Min(24, mobaData.Length - o));
                Console.WriteLine($"[DEBUG]   Batch {i}: lightMap={mobaData[o]}, texture={mobaData[o+1]}, startIdx={b.FirstFace}, count={b.NumFaces}, mat={b.MaterialId}");
                Console.WriteLine($"[DEBUG]     Raw bytes: {hex}");
            }
        }

        private void ProcessMotvChunk(byte[] motvData, WmoGroupData groupData)
        {
            // MOTV: primary UV set; 8 bytes per vertex (u,v)
            const int UV_SIZE = 8;
            int count = motvData.Length / UV_SIZE;
            if (count <= 0) return;
            groupData.UVs.Clear();
            for (int i = 0; i < count; i++)
            {
                int o = i * UV_SIZE;
                float u = BitConverter.ToSingle(motvData, o + 0);
                float v = BitConverter.ToSingle(motvData, o + 4);
                groupData.UVs.Add(new Vector2(u, v));
            }
            Console.WriteLine($"[DEBUG] Extracted {count} UVs from MOTV");
        }

        private List<WmoMaterial> ParseMomtMaterials(byte[] momtData, List<string> textures, Dictionary<uint, string> texOffsetToName)
        {
            // V14 MOMT structure is 44 bytes (not 64!)
            // Has 'version' field at start, NO shader field, NO texture_3/color_2/flags_2/runTimeData
            const int ENTRY_SIZE = 44;
            var materials = new List<WmoMaterial>();
            
            for (int off = 0; off + ENTRY_SIZE <= momtData.Length; off += ENTRY_SIZE)
            {
                var mat = new WmoMaterial();
                
                // V14 structure: version(4), flags(4), blendMode(4), texture_1(4), sidnColor(4), frameSidnColor(4), texture_2(4), diffColor(4), ground_type(4), padding(8)
                uint version = BitConverter.ToUInt32(momtData, off + 0x00);  // V14 only!
                mat.Flags = BitConverter.ToUInt32(momtData, off + 0x04);
                mat.Shader = 0;  // NO shader field in v14
                mat.BlendMode = BitConverter.ToUInt32(momtData, off + 0x08);
                mat.Texture1Offset = BitConverter.ToUInt32(momtData, off + 0x0C);
                mat.EmissiveColor = BitConverter.ToUInt32(momtData, off + 0x10);  // sidnColor
                mat.Flags2 = BitConverter.ToUInt32(momtData, off + 0x14);         // frameSidnColor (runtime)
                mat.Texture2Offset = BitConverter.ToUInt32(momtData, off + 0x18);
                mat.DiffuseColor = BitConverter.ToUInt32(momtData, off + 0x1C);
                mat.GroundType = BitConverter.ToUInt32(momtData, off + 0x20);
                // 0x24-0x2B is padding in v14
                mat.Texture3Offset = 0;  // Not present in v14
                mat.AmbientColor = 0;
                mat.SpecularColor = 0;
                mat.Color2 = 0;
                
                materials.Add(mat);
            }
            
            Console.WriteLine($"[DEBUG] Parsed {materials.Count} materials from MOMT (v14 format: 44 bytes each)");
            
            // Log material details for debugging
            for (int i = 0; i < Math.Min(materials.Count, 5); i++)
            {
                var m = materials[i];
                string tex1Name = texOffsetToName.TryGetValue(m.Texture1Offset, out var t1) ? t1 : $"offset_{m.Texture1Offset}";
                Console.WriteLine($"[DEBUG]   Mat {i}: Flags=0x{m.Flags:X8}, Blend={m.BlendMode}, Tex1={tex1Name}");
            }
            if (materials.Count > 5)
                Console.WriteLine($"[DEBUG]   ... and {materials.Count - 5} more materials");
            
            return materials;
        }

        private List<uint> ParseMomtTextureIndices(byte[] momtData, List<string> textures, Dictionary<uint, string> texOffsetToName)
        {
            const int ENTRY_SIZE = 44;  // V14 uses 44 bytes, not 64
            var list = new List<uint>();
            for (int off = 0; off + ENTRY_SIZE <= momtData.Length; off += ENTRY_SIZE)
            {
                // V14: version(4), flags(4), blendMode(4), texture_1(4)...
                // Try texture1 at +0x0C as MOTX offset
                uint texture1Offset = BitConverter.ToUInt32(momtData, off + 0x0C);
                if (texOffsetToName.TryGetValue(texture1Offset, out var name))
                {
                    int find = textures.FindIndex(t => string.Equals(t, name, StringComparison.OrdinalIgnoreCase));
                    if (find >= 0) { list.Add((uint)find); continue; }
                }
                // Fallback: treat as index
                if (texture1Offset < (uint)textures.Count)
                {
                    list.Add(texture1Offset);
                    continue;
                }
                // Give up: 0
                list.Add(0);
            }
            return list;
        }

        private (List<string> Names, Dictionary<uint, string> Offsets) ParseTextureTable(byte[] motxData)
        {
            var names = new List<string>();
            var offsets = new Dictionary<uint, string>();
            int pos = 0;
            while (pos < motxData.Length)
            {
                int start = pos;
                // find null terminator
                while (pos < motxData.Length && motxData[pos] != 0) pos++;
                var name = Encoding.ASCII.GetString(motxData, start, pos - start);
                if (!string.IsNullOrWhiteSpace(name))
                {
                    names.Add(name);
                    offsets[(uint)start] = name;
                }
                // skip null
                pos++;
            }
            return (names, offsets);
        }

        private Dictionary<int, string> ParseStringBlockWithOffsets(byte[] data)
        {
            var dict = new Dictionary<int, string>();
            int pos = 0;
            while (pos < data.Length)
            {
                int start = pos;
                while (pos < data.Length && data[pos] != 0) pos++;
                string s = Encoding.ASCII.GetString(data, start, pos - start);
                dict[start] = s;
                pos++; // skip null
            }
            return dict;
        }

        public class WmoV14Data
        {
            public uint Version { get; set; }
            public List<ChunkInfo> Chunks { get; set; } = new();
            public List<string> Textures { get; set; } = new();
            public Dictionary<uint, string> TextureOffsetToName { get; set; } = new();
            public List<uint> MaterialTextureIndices { get; set; } = new();
            public List<WmoMaterial> Materials { get; set; } = new();
            public List<WmoGroupData> Groups { get; set; } = new();
            public byte[] FileBytes { get; set; } = Array.Empty<byte>();
            public Dictionary<int, string> GroupNameMap { get; set; } = new();
            public List<int> GroupNameIndices { get; set; } = new();
        }

        public class WmoGroupData
        {
            public string Name { get; set; } = string.Empty;
            public List<Vector3> Vertices { get; set; } = new();
            public List<ushort> Indices { get; set; } = new();
            public List<byte> FaceFlags { get; set; } = new();
            public List<byte> FaceMaterials { get; set; } = new();
            public List<Vector2> UVs { get; set; } = new();
            public uint Flags { get; set; }
            public List<MobaBatch> Batches { get; set; } = new();
            public List<ushort> FaceOrder { get; set; } = new();
        }

        public struct MobaBatch
        {
            public uint FirstFace;
            public ushort NumFaces;
            public ushort FirstVertex;
            public ushort LastVertex;
            public byte Flags;
            public byte MaterialId;
            public ushort PossibleBox2Z;
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
                var (names, offsets) = ParseTextureTable(textureChunk.Data);
                wmoData.Textures = names;
                wmoData.TextureOffsetToName = offsets;
            }
            // Extract materials (MOMT chunk → full material data + texture indices)
            var momtChunk = wmoData.Chunks.FirstOrDefault(c => c.Id == "MOMT");
            if (momtChunk?.Data != null)
            {
                wmoData.Materials = ParseMomtMaterials(momtChunk.Data, wmoData.Textures, wmoData.TextureOffsetToName);
                wmoData.MaterialTextureIndices = ParseMomtTextureIndices(momtChunk.Data, wmoData.Textures, wmoData.TextureOffsetToName);
            }

            // Extract group names (MOGN/MOGI)
            var mognChunk = wmoData.Chunks.FirstOrDefault(c => c.Id == "MOGN");
            if (mognChunk?.Data != null)
            {
                wmoData.GroupNameMap = ParseStringBlockWithOffsets(mognChunk.Data);
            }
            var mogiChunk = wmoData.Chunks.FirstOrDefault(c => c.Id == "MOGI");
            if (mogiChunk?.Data != null)
            {
                const int REC = 32;
                int count = mogiChunk.Data.Length / REC;
                for (int i = 0; i < count; i++)
                {
                    int ofs = i * REC;
                    int nameIdx = BitConverter.ToInt32(mogiChunk.Data, ofs + 28);
                    wmoData.GroupNameIndices.Add(nameIdx);
                }
            }

            // Build groups: use each MOGP chunk's data buffer directly (headerless slice already captured by ReadChunk)
            wmoData.Groups.Clear();
            var mogpChunks = wmoData.Chunks
                .Select((c, i) => new { c, i })
                .Where(x => x.c.Id == "MOGP")
                .ToList();

            for (int gi = 0; gi < mogpChunks.Count; gi++)
            {
                var region = mogpChunks[gi].c.Data; // starts at MOGP data (header), length = chunk size
                if (region == null || region.Length < 64) continue;

                var group = new WmoGroupData { Name = $"group_{gi}" };
                if (gi < wmoData.GroupNameIndices.Count)
                {
                    int nameOfs = wmoData.GroupNameIndices[gi];
                    if (nameOfs >= 0 && wmoData.GroupNameMap.TryGetValue(nameOfs, out var nm) && !string.IsNullOrWhiteSpace(nm))
                        group.Name = nm;
                }
                ProcessMogpRegion(region, group);
                Console.WriteLine($"[GROUP] {group.Name}: MOVT verts={group.Vertices.Count}, MOVI idx={group.Indices.Count}, MOPY faces={group.FaceMaterials.Count}");
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

                // Determine triangle indices: prefer MOVI; fallback to sequential triples if MOVI is absent
                List<(ushort a, ushort b, ushort c)> tris = new List<(ushort, ushort, ushort)>();
                if (group.Indices.Count >= 3)
                {
                    int usable = group.Indices.Count - (group.Indices.Count % 3);
                    if (usable != group.Indices.Count)
                    {
                        Console.WriteLine($"[WARN] Indices not multiple of 3 in group '{group.Name}': {group.Indices.Count} -> using {usable}");
                    }
                    for (int i = 0; i + 2 < usable; i += 3)
                    {
                        tris.Add((group.Indices[i], group.Indices[i + 1], group.Indices[i + 2]));
                    }
                }
                else if (group.FaceMaterials.Count > 0 && group.Vertices.Count >= 3)
                {
                    int faceCount = group.FaceMaterials.Count;
                    int maxTris = Math.Min(faceCount, group.Vertices.Count / 3);
                    for (int t = 0; t < maxTris; t++)
                    {
                        int baseIdx = t * 3;
                        tris.Add(((ushort)baseIdx, (ushort)(baseIdx + 1), (ushort)(baseIdx + 2)));
                    }
                    Console.WriteLine($"[WARN] MOVI indices missing in group '{group.Name}'; used sequential fallback: {tris.Count} tris");
                }
                else
                {
                    Console.WriteLine($"[WARN] Group '{group.Name}' lacks MOVI and insufficient data for fallback; skipping triangles");
                    continue;
                }

                // For each triangle, duplicate its three vertices into the BSP Vertices list and fill MeshVertices
                int triIndex = 0;
                foreach (var (a, b, c) in tris)
                {
                    if (a >= group.Vertices.Count || b >= group.Vertices.Count || c >= group.Vertices.Count)
                        continue;

                    var start = bspFile.Vertices.Count;

                    var p0 = group.Vertices[a];
                    var p1 = group.Vertices[b];
                    var p2 = group.Vertices[c];

                    // Compute face normal and skip degenerates
                    var e1 = p1 - p0;
                    var e2 = p2 - p0;
                    var n = Vector3.Cross(e1, e2);
                    var len = n.Length();
                    if (len < 1e-6f) {
                        triIndex++;
                        continue; // degenerate triangle
                    }
                    n /= len;

                    // Get material ID for this triangle from FaceMaterials
                    int materialId = 0;
                    if (triIndex < group.FaceMaterials.Count)
                    {
                        materialId = group.FaceMaterials[triIndex];  // FaceMaterials is List<byte>
                        // Clamp to valid texture range
                        materialId = Math.Max(0, Math.Min(materialId, bspFile.Textures.Count - 1));
                    }

                    bspFile.Vertices.Add(new BspVertex { Position = p0, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = n, Color = new byte[] { 255, 255, 255, 255 } });
                    bspFile.Vertices.Add(new BspVertex { Position = p1, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = n, Color = new byte[] { 255, 255, 255, 255 } });
                    bspFile.Vertices.Add(new BspVertex { Position = p2, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = n, Color = new byte[] { 255, 255, 255, 255 } });

                    // Mesh indices for this triangle (relative to FirstVertex)
                    int meshStart = bspFile.MeshVertices.Count;
                    bspFile.MeshVertices.Add(0);
                    bspFile.MeshVertices.Add(1);
                    bspFile.MeshVertices.Add(2);

                    bspFile.Faces.Add(new BspFace
                    {
                        Texture = materialId,  // Use correct material from MOBA/MOPY
                        Effect = -1,
                        Type = 3, // mesh with MeshVertices
                        FirstVertex = start,
                        NumVertices = 3,
                        FirstMeshVertex = meshStart,
                        NumMeshVertices = 3,
                        Lightmap = -1,
                        Normal = n
                    });
                    
                    triIndex++;
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

            // fresh containers per group region
            groupData.Vertices.Clear();
            groupData.Indices.Clear();
            groupData.FaceFlags.Clear();
            groupData.FaceMaterials.Clear();
            groupData.UVs.Clear();
            groupData.Batches.Clear();

            // Find first plausible sub-chunk header after the variable-size MOGP header.
            var valid = new HashSet<string>{"MOVT","MOVI","MOIN","MONR","MOTV","MOPY","MOBA","MOBN","MOBR"};
            long foundPos = -1;
            long searchStart = MOGP_HEADER_SIZE;
            long searchLimit = Math.Min(ms.Length, searchStart + 1024); // scan up to 1KB past header
            for (long off = searchStart; off + 8 <= searchLimit; off++)
            {
                ms.Position = off;
                var idBytes = br.ReadBytes(4);
                if (idBytes.Length < 4) break;
                string sid = Encoding.ASCII.GetString(idBytes.Reverse().ToArray());
                if (!valid.Contains(sid)) continue;
                uint ssz = br.ReadUInt32();
                long next = off + 8 + ssz;
                if (next <= ms.Length) { foundPos = off; break; }
            }
            if (foundPos < 0) foundPos = MOGP_HEADER_SIZE; // fallback
            ms.Position = foundPos;

            bool seenMOVT = false, seenMOVI = false, seenMOPY = false;
            bool indicesFromMoin = false;
            int movtVerts = 0, moviIdx = 0, mopyFaces = 0;

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
                    case "MOVT":
                        ProcessMovtChunk(data, groupData);
                        seenMOVT = true;
                        movtVerts = data.Length / 12;
                        break;
                    case "MOVI":
                        ProcessMoviChunk(data, groupData);
                        seenMOVI = true;
                        moviIdx = data.Length / 2;
                        break;
                    case "MOIN":
                        if (!seenMOVI)
                        {
                            // Append all MOIN segments when MOVI is absent
                            ProcessMoinChunk(data, groupData);
                            indicesFromMoin = true;
                            moviIdx = groupData.Indices.Count;
                        }
                        break;
                    case "MOTV":
                        ProcessMotvChunk(data, groupData);
                        break;
                    case "MOPY":
                        ProcessMopyChunk(data, groupData);
                        seenMOPY = true;
                        mopyFaces = data.Length / 2;
                        break;
                    case "MOBA":
                        ProcessMobaChunk(data, groupData);
                        break;
                    case "MOBR":
                        ProcessMobrChunk(data, groupData);
                        break;
                }
                ms.Position = subEnd;
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
                // Store raw WMO coordinates; downstream map generator applies WMO->Q3 transform
                var x = BitConverter.ToSingle(movtData, offset);
                var y = BitConverter.ToSingle(movtData, offset + 4);
                var z = BitConverter.ToSingle(movtData, offset + 8);
                groupData.Vertices.Add(new Vector3(x, y, z));
            }
            
            Console.WriteLine($"[DEBUG] Extracted {vertexCount} vertices from MOVT");
        }

        private void ProcessMoviChunk(byte[] moviData, WmoGroupData groupData)
        {
            // MOVI: Face indices (2 bytes per index)
            const int INDEX_SIZE = 2;
            var indexCount = moviData.Length / INDEX_SIZE;
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
            groupData.FaceFlags.Clear();
            groupData.FaceMaterials.Clear();
            for (int i = 0; i < faceCount; i++)
            {
                byte flags = mopyData[i * 2 + 0];
                byte matId = mopyData[i * 2 + 1];
                groupData.FaceFlags.Add(flags);
                groupData.FaceMaterials.Add(matId);
            }
            Console.WriteLine($"[DEBUG] Extracted {faceCount} faces from MOPY");
            
            // Log first few MOPY entries for debugging
            for (int i = 0; i < Math.Min(faceCount, 5); i++)
            {
                Console.WriteLine($"[DEBUG]   MOPY {i}: flags=0x{mopyData[i*2]:X2}, mat={mopyData[i*2+1]}");
            }
        }
        private void ProcessMobrChunk(byte[] mobrData, WmoGroupData groupData)
        {
            // MOBR: face order mapping (uint16 entries)
            int count = mobrData.Length / 2;
            groupData.FaceOrder.Clear();
            for (int i = 0; i < count; i++)
            {
                groupData.FaceOrder.Add(BitConverter.ToUInt16(mobrData, i * 2));
            }
            Console.WriteLine($"[DEBUG] Extracted {count} entries from MOBR");
        }
        private void ProcessMoinChunk(byte[] moinData, WmoGroupData groupData)
        {
            // Treat MOIN as an index buffer like MOVI (2 bytes per index). Some v14 files use MOIN instead of MOVI.
            const int INDEX_SIZE = 2;
            if (moinData.Length % INDEX_SIZE != 0)
            {
                Console.WriteLine("[WARN] MOIN size not divisible by 2; skipping.");
                return;
            }
            int indexCount = moinData.Length / INDEX_SIZE;
            for (int i = 0; i < indexCount; i++)
            {
                int o = i * INDEX_SIZE;
                groupData.Indices.Add(BitConverter.ToUInt16(moinData, o));
            }
            Console.WriteLine($"[DEBUG] Extracted {indexCount} indices from MOIN");
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
            groupData.FaceFlags.Clear();
            for (int i = 0; i < faceCount; i++)
            {
                groupData.FaceFlags.Add(0);
                groupData.FaceMaterials.Add(mopyData[i]);
            }
            
            Console.WriteLine($"[DEBUG] Extracted {faceCount} face materials from top-level MOPY");
        }
    }
}