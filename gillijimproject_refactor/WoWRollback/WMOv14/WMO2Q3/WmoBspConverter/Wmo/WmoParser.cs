using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// Unified WMO parser that auto-detects version and parses v14, v16, or v17 files.
    /// Outputs WmoV14Parser.WmoV14Data for compatibility with all downstream code.
    /// </summary>
    public class WmoParser
    {
        private readonly bool _verbose;

        public WmoParser(bool verbose = false)
        {
            _verbose = verbose;
        }

        /// <summary>
        /// Parse a WMO file of any supported version, returning unified data structure.
        /// </summary>
        public WmoV14Parser.WmoV14Data Parse(string wmoPath)
        {
            if (!File.Exists(wmoPath))
                throw new FileNotFoundException($"WMO file not found: {wmoPath}");

            var version = DetectVersion(wmoPath);
            
            if (_verbose)
                Console.WriteLine($"[INFO] Detected WMO version: {version}");

            return version switch
            {
                14 => ParseV14(wmoPath),
                16 or 17 => ParseV16V17(wmoPath),
                _ => throw new NotSupportedException($"Unsupported WMO version: {version}")
            };
        }

        /// <summary>
        /// Detect WMO version from the MVER chunk.
        /// </summary>
        private int DetectVersion(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            // Read first chunk
            var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
            var chunkSize = reader.ReadUInt32();

            // v14 files may start with MOMO container
            if (chunkId == "OMOM" || chunkId == "MOMO")
            {
                // v14 with MOMO wrapper - parse inside
                // First 4 bytes inside MOMO should be MVER reversed
                var innerChunk = Encoding.ASCII.GetString(reader.ReadBytes(4));
                if (innerChunk == "REVM" || innerChunk == "MVER")
                {
                    reader.ReadUInt32(); // size
                    return (int)reader.ReadUInt32();
                }
                return 14; // Assume v14 if has MOMO
            }
            
            // Standard MVER at start (v16/v17, or unwrapped v14)
            if (chunkId == "REVM" || chunkId == "MVER")
            {
                return (int)reader.ReadUInt32();
            }

            throw new InvalidDataException($"Invalid WMO file - expected MVER or MOMO, got: {chunkId}");
        }

        /// <summary>
        /// Parse v14 WMO using the existing specialized parser.
        /// </summary>
        private WmoV14Parser.WmoV14Data ParseV14(string path)
        {
            if (_verbose)
                Console.WriteLine("[INFO] Using v14 parser (MOMO container format)");
            
            var parser = new WmoV14Parser();
            return parser.ParseWmoV14(path);
        }

        /// <summary>
        /// Parse v16/v17 WMO (root + group files) and convert to WmoV14Data structure.
        /// </summary>
        private WmoV14Parser.WmoV14Data ParseV16V17(string path)
        {
            if (_verbose)
                Console.WriteLine("[INFO] Using v16/v17 parser (split group files)");

            var result = new WmoV14Parser.WmoV14Data();
            
            // Parse root file
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            var textures = new List<string>();
            var groupInfos = new List<GroupInfoV17>();
            byte[] textureData = null;
            byte[] groupNameData = null;
            uint groupCount = 0;
            uint materialCount = 0;

            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                var chunkSize = reader.ReadUInt32();
                var chunkEnd = reader.BaseStream.Position + chunkSize;

                switch (chunkId)
                {
                    case "REVM": // MVER reversed
                        result.Version = reader.ReadUInt32();
                        break;

                    case "DHOM": // MOHD reversed
                        materialCount = reader.ReadUInt32();
                        groupCount = reader.ReadUInt32();
                        // Portal count read here - skip it, version already set from MVER
                        reader.BaseStream.Position = chunkEnd; // Skip rest of header
                        break;

                    case "XTOM": // MOTX reversed
                        textureData = reader.ReadBytes((int)chunkSize);
                        textures.AddRange(ParseTextureStrings(textureData));
                        break;

                    case "TMOM": // MOMT reversed
                        // Parse materials (64 bytes each in v17) - skip for now
                        // We mainly need texture names from MOTX
                        break;

                    case "NGOM": // MOGN reversed
                        groupNameData = reader.ReadBytes((int)chunkSize);
                        break;

                    case "IGOM": // MOGI reversed
                        int giCount = (int)(chunkSize / 32);
                        for (int i = 0; i < giCount; i++)
                        {
                            groupInfos.Add(new GroupInfoV17
                            {
                                Flags = reader.ReadUInt32(),
                                BoundingBoxMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                                BoundingBoxMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                                NameOffset = reader.ReadInt32()
                            });
                        }
                        break;

                    case "TPOM": // MOPT reversed (Portals)
                        // V17 Portal Defs usually standard 20 bytes
                        // Use existing logic from WmoV14Parser if compatible, or reimplement inline
                        // For now, read raw bytes and pass to a helper, or just implement basic parsing
                        {
                            var portalData = reader.ReadBytes((int)chunkSize);
                            ProcessMoptChunkV17(portalData, result);
                        }
                        break;

                    case "RPOM": // MOPR reversed (Portal Refs)
                        {
                            // V17 Portal Refs usually 8 bytes?
                            var refData = reader.ReadBytes((int)chunkSize);
                            ProcessMoprChunkV17(refData, result);
                        }
                        break;
                }

                reader.BaseStream.Position = chunkEnd;
            }

            result.Textures = textures;
            result.Groups = new List<WmoV14Parser.WmoGroupData>();

            // Parse group files
            var baseName = Path.GetFileNameWithoutExtension(path);
            var directory = Path.GetDirectoryName(path) ?? ".";

            for (int i = 0; i < (int)groupCount; i++)
            {
                var groupPath = Path.Combine(directory, $"{baseName}_{i:D3}.wmo");
                
                if (File.Exists(groupPath))
                {
                    var groupData = ParseGroupFileV17(groupPath, i, groupNameData, groupInfos);
                    result.Groups.Add(groupData);
                    
                    if (_verbose && groupData.Vertices.Count > 0)
                        Console.WriteLine($"[GROUP] {i}: {groupData.Vertices.Count} verts, {groupData.Indices.Count / 3} faces");
                }
                else
                {
                    if (_verbose)
                        Console.WriteLine($"[WARN] Group file not found: {groupPath}");
                }
            }

            if (_verbose)
                Console.WriteLine($"[INFO] Parsed {result.Groups.Count} groups, {result.Textures.Count} textures");

            return result;
        }

        private List<string> ParseTextureStrings(byte[] data)
        {
            var result = new List<string>();
            var current = new StringBuilder();
            
            foreach (var b in data)
            {
                if (b == 0)
                {
                    if (current.Length > 0)
                    {
                        result.Add(current.ToString());
                        current.Clear();
                    }
                }
                else
                {
                    current.Append((char)b);
                }
            }
            
            return result;
        }

        private void ProcessMoptChunkV17(byte[] data, WmoV14Parser.WmoV14Data wmoData)
        {
            // MOPT: Portal Definitions
            // V17 seems to use 20 bytes per entry: StartGrp(2), EndGrp(2), VertexCount(2), FirstVertex(2), Plane(12)
            // Or potentially 64 bytes?
            // Let's assume 20 bytes based on WmoV14Parser findings.
            
            int entrySize = 20;
            if (data.Length % 20 != 0)
            {
                // Fallback guessing
                if (data.Length % 64 == 0) entrySize = 64; // v17 might use 64 bytes?
                else if (data.Length % 32 == 0) entrySize = 32;
            }

            int count = data.Length / entrySize;
            wmoData.Portals.Clear();

            using var reader = new BinaryReader(new MemoryStream(data));
            for (int i = 0; i < count; i++)
            {
                var def = new WmoV14Parser.MoptDef();
                if (entrySize >= 20)
                {
                    def.StartGroup = reader.ReadUInt16();
                    def.EndGroup = reader.ReadUInt16();
                    def.VertexCount = reader.ReadUInt16();
                    def.FirstVertex = reader.ReadUInt16();
                    
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    def.Plane = new Vector4(x, y, z, 0); // D is implicit or calculated usually, or missing in 12-byte plane
                    
                    // Skip remaining padding if any
                    if (entrySize > 20) reader.ReadBytes(entrySize - 20);
                }
                wmoData.Portals.Add(def);
            }
        }

        private void ProcessMoprChunkV17(byte[] data, WmoV14Parser.WmoV14Data wmoData)
        {
            // MOPR: Portal References
            // Standard 8 bytes: PortalIndex(2), GroupIndex(2), Side(2), Reserved(2)
            int entrySize = 8;
            int count = data.Length / entrySize;
            wmoData.PortalRefs.Clear();

            using var reader = new BinaryReader(new MemoryStream(data));
            for (int i = 0; i < count; i++)
            {
                var r = new WmoV14Parser.MoprRef();
                r.PortalIndex = reader.ReadUInt16();
                r.GroupIndex = reader.ReadUInt16();
                int side = reader.ReadInt16();
                // Skip reserved
                reader.ReadUInt16();
                
                r.Sign = side;
                wmoData.PortalRefs.Add(r);
            }
        }

        private WmoV14Parser.WmoGroupData ParseGroupFileV17(string path, int groupIndex, 
            byte[] groupNameData, List<GroupInfoV17> groupInfos)
        {
            var group = new WmoV14Parser.WmoGroupData
            {
                Vertices = new List<Vector3>(),
                Normals = new List<Vector3>(),
                UVs = new List<Vector2>(),
                Indices = new List<ushort>(),
                FaceMaterials = new List<byte>()
            };

            // Get group name if available
            if (groupIndex < groupInfos.Count && groupNameData != null)
            {
                var nameOffset = groupInfos[groupIndex].NameOffset;
                if (nameOffset >= 0 && nameOffset < groupNameData.Length)
                {
                    var sb = new StringBuilder();
                    for (int i = nameOffset; i < groupNameData.Length && groupNameData[i] != 0; i++)
                        sb.Append((char)groupNameData[i]);
                    group.Name = sb.ToString();
                }
            }

            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);

            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                var chunkSize = reader.ReadUInt32();
                var chunkEnd = reader.BaseStream.Position + chunkSize;

                switch (chunkId)
                {
                    case "REVM": // MVER
                        reader.ReadUInt32(); // version
                        break;

                    case "PGOM": // MOGP - group header containing subchunks
                        ParseMogpV17(reader, chunkSize, group);
                        break;
                }

                reader.BaseStream.Position = chunkEnd;
            }

            return group;
        }

        private void ParseMogpV17(BinaryReader reader, uint chunkSize, WmoV14Parser.WmoGroupData group)
        {
            var startPos = reader.BaseStream.Position;
            
            // Read MOGP header (68 bytes in v17)
            var nameOffset = reader.ReadUInt32();
            var descOffset = reader.ReadUInt32();
            var flags = reader.ReadUInt32();
            group.Flags = flags; // Store flags (needed for exterior/interior distinction)
            var bbMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            var bbMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            var portalStart = reader.ReadUInt16();
            var portalCount = reader.ReadUInt16();
            var transBatchCount = reader.ReadUInt16();
            var intBatchCount = reader.ReadUInt16();
            var extBatchCount = reader.ReadUInt16();
            var padding = reader.ReadUInt16();
            var fogIndices = reader.ReadBytes(4);
            var liquidType = reader.ReadUInt32();
            var groupId = reader.ReadUInt32();
            var unused1 = reader.ReadUInt32();
            var unused2 = reader.ReadUInt32();

            // Parse subchunks
            var subchunkEnd = startPos + chunkSize;
            while (reader.BaseStream.Position < subchunkEnd - 8)
            {
                if (reader.BaseStream.Position + 8 > reader.BaseStream.Length)
                    break;

                var subId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                var subSize = reader.ReadUInt32();
                var subEnd = reader.BaseStream.Position + subSize;

                switch (subId)
                {
                    case "TVOM": // MOVT - vertices
                        int vertCount = (int)(subSize / 12);
                        for (int i = 0; i < vertCount; i++)
                        {
                            group.Vertices.Add(new Vector3(
                                reader.ReadSingle(),
                                reader.ReadSingle(),
                                reader.ReadSingle()));
                        }
                        break;

                    case "RNOM": // MONR - normals
                        int normCount = (int)(subSize / 12);
                        for (int i = 0; i < normCount; i++)
                        {
                            group.Normals.Add(new Vector3(
                                reader.ReadSingle(),
                                reader.ReadSingle(),
                                reader.ReadSingle()));
                        }
                        break;

                    case "VTOM": // MOTV - UVs
                        int uvCount = (int)(subSize / 8);
                        for (int i = 0; i < uvCount; i++)
                        {
                            group.UVs.Add(new Vector2(
                                reader.ReadSingle(),
                                reader.ReadSingle()));
                        }
                        break;

                    case "IVOM": // MOVI - indices
                        int idxCount = (int)(subSize / 2);
                        for (int i = 0; i < idxCount; i++)
                        {
                            group.Indices.Add(reader.ReadUInt16());
                        }
                        break;

                    case "YPOM": // MOPY - face materials (2 bytes per face)
                        int faceCount = (int)(subSize / 2);
                        for (int i = 0; i < faceCount; i++)
                        {
                            var flags2 = reader.ReadByte();
                            var matId = reader.ReadByte();
                            group.FaceMaterials.Add(matId);
                        }
                        break;

                    case "ABOM": // MOBA - batches (24 bytes each in v17)
                        int batchCount = (int)(subSize / 24);
                        for (int i = 0; i < batchCount; i++)
                        {
                            // Skip bounding box (12 bytes) for v17
                            reader.ReadBytes(12);
                            var startIndex = reader.ReadUInt32();
                            var count = reader.ReadUInt16();
                            var minIndex = reader.ReadUInt16();
                            var maxIndex = reader.ReadUInt16();
                            var materialId = reader.ReadByte();
                            var unknown = reader.ReadByte();
                            
                            // Could use batch info to map materials more accurately
                        }
                        break;
                }

                reader.BaseStream.Position = subEnd;
            }
        }

        private struct GroupInfoV17
        {
            public uint Flags;
            public Vector3 BoundingBoxMin;
            public Vector3 BoundingBoxMax;
            public int NameOffset;
        }
    }
}
