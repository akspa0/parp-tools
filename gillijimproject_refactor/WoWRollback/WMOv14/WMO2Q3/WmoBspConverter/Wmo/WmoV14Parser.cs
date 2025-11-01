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
                    // Parse sub-chunks inside MOMO container
                    ParseMomoContainer(chunk.Data, wmoData);
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

        private void ParseMomoContainer(byte[] momoData, WmoV14Data wmoData)
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
                    Offset = (uint)dataStart,
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

                // Process MOGP chunk data for this group if available
                var mogpChunk = wmoData.Chunks.FirstOrDefault(c => c.Id == "MOGP");
                if (mogpChunk?.Data != null)
                {
                    ProcessMogpChunk(mogpChunk.Data, groupData);
                }

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

                // For each triangle, duplicate its three vertices into the BSP Vertices list
                for (int i = 0; i + 2 < group.Indices.Count; i += 3)
                {
                    var idx0 = group.Indices[i];
                    var idx1 = group.Indices[i + 1];
                    var idx2 = group.Indices[i + 2];

                    if (idx0 >= group.Vertices.Count || idx1 >= group.Vertices.Count || idx2 >= group.Vertices.Count)
                        continue;

                    var start = bspFile.Vertices.Count;

                    // Duplicate three vertices for this face
                    var p0 = group.Vertices[idx0];
                    var p1 = group.Vertices[idx1];
                    var p2 = group.Vertices[idx2];

                    bspFile.Vertices.Add(new BspVertex { Position = p0, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });
                    bspFile.Vertices.Add(new BspVertex { Position = p1, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });
                    bspFile.Vertices.Add(new BspVertex { Position = p2, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });

                    bspFile.Faces.Add(new BspFace
                    {
                        Texture = 0,
                        Effect = -1,
                        Type = 1, // polygon
                        FirstVertex = start,
                        NumVertices = 3,
                        FirstMeshVertex = 0,
                        NumMeshVertices = 0,
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

                // Optional padding skip: if next 4 bytes are zero/non-printable, skip 4
                if (ms.Position + 4 <= ms.Length)
                {
                    var peek = br.ReadBytes(4);
                    bool allZero = peek.All(b => b == 0);
                    bool nonPrintable = peek.All(b => b < 0x20 || b > 0x7E);
                    if (allZero || nonPrintable)
                    {
                        // skip
                    }
                    else
                    {
                        // rewind if actually the next id
                        ms.Position -= 4;
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
            // MOPY: Face material assignments (1 byte per face)
            var faceCount = mopyData.Length;
            
            groupData.FaceMaterials.Clear();
            for (int i = 0; i < faceCount; i++)
            {
                groupData.FaceMaterials.Add(mopyData[i]);
            }
            
            Console.WriteLine($"[DEBUG] Extracted {faceCount} face materials from MOPY");
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