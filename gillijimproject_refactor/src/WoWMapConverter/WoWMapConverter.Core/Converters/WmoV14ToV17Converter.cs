using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts WMO v14 (Alpha) to WMO v17 (LK 3.3.5) format.
/// Handles monolithic Alpha WMO → split root + group files.
/// </summary>
public class WmoV14ToV17Converter
{
    /// <summary>
    /// Convert a v14 WMO file to v17 format.
    /// </summary>
    public List<string> Convert(string inputPath, string outputPath)
    {
        Console.WriteLine($"[INFO] Converting {Path.GetFileName(inputPath)} to v17...");
        
        using var fs = new FileStream(inputPath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);

        var data = ParseWmoV14(reader);
        
        // Ensure output directory exists (including group files)
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        WriteRootFile(data, outputPath);
        
        // Write group files
        WriteGroupFiles(data, outputPath);

        Console.WriteLine($"[SUCCESS] Converted to v17: {outputPath}");
        
        return data.Textures;
    }

    private WmoV14Data ParseWmoV14(BinaryReader reader)
    {
        var data = new WmoV14Data();

        // Read MVER
        var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (magic != "REVM") // Reversed
            throw new InvalidDataException($"Expected MVER, got {magic}");
        
        var size = reader.ReadUInt32();
        data.Version = reader.ReadUInt32();
        
        if (data.Version != 14)
            throw new InvalidDataException($"Expected WMO v14, got v{data.Version}");

        // Read MOMO container (contains header chunks like MOHD, MOTX, MOMT, etc.)
        magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (magic != "OMOM") // Reversed MOMO
            throw new InvalidDataException($"Expected MOMO container, got {magic}");
        
        var momoSize = reader.ReadUInt32();
        var momoEnd = reader.BaseStream.Position + momoSize;

        // Parse chunks within MOMO (header data)
        while (reader.BaseStream.Position < momoEnd)
        {
            var chunkMagic = new string(reader.ReadChars(4).Reverse().ToArray());
            var chunkSize = reader.ReadUInt32();
            var chunkEnd = reader.BaseStream.Position + chunkSize;

            switch (chunkMagic)
            {
                case "MOHD":
                    ParseMohd(reader, data);
                    break;
                case "MOTX":
                    ParseMotx(reader, chunkSize, data);
                    break;
                case "MOMT":
                    ParseMomt(reader, chunkSize, data);
                    break;
                case "MOGN":
                    ParseMogn(reader, chunkSize, data);
                    break;
                case "MOGI":
                    ParseMogi(reader, chunkSize, data);
                    break;
                case "MODS":
                    ParseMods(reader, chunkSize, data);
                    break;
                case "MODN":
                    data.DoodadNamesRaw = reader.ReadBytes((int)chunkSize);
                    break;
                case "MODD":
                    ParseModd(reader, chunkSize, data);
                    break;
                case "MOPV":
                    ParseMopv(reader, chunkSize, data);
                    break;
                case "MOPT":
                    ParseMopt(reader, chunkSize, data);
                    break;
                case "MOPR":
                    ParseMopr(reader, chunkSize, data);
                    break;
                case "MOLT":
                    ParseMolt(reader, chunkSize, data);
                    break;
                default:
                    // Skip unknown chunks
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        // Alpha WMO: MOGP group chunks are OUTSIDE the MOMO container
        // Continue reading until end of file
        while (reader.BaseStream.Position < reader.BaseStream.Length - 8)
        {
            try
            {
                var chunkMagic = new string(reader.ReadChars(4).Reverse().ToArray());
                var chunkSize = reader.ReadUInt32();
                var chunkEnd = reader.BaseStream.Position + chunkSize;

                if (chunkMagic == "MOGP")
                {
                    ParseMogp(reader, chunkSize, data);
                }
                // else skip unknown chunk

                reader.BaseStream.Position = chunkEnd;
            }
            catch
            {
                break; // End of file or parse error
            }
        }

        Console.WriteLine($"[DEBUG] Parsed v14 WMO: {data.Groups.Count} groups, {data.Materials.Count} materials, {data.Textures.Count} textures");
        
        // Resolve material specific textures
        ResolveMaterialTextures(data);
        
        // Recalculate bounds (Group and Global) to fix missing MOHD/MOGI boxes
        RecalculateBounds(data);
        
        return data;
    }

    private void RecalculateBounds(WmoV14Data data)
    {
        var globalMin = new Vector3(float.MaxValue);
        var globalMax = new Vector3(float.MinValue);
        
        for (int i = 0; i < data.Groups.Count; i++)
        {
            var group = data.Groups[i];
            
            // Calculate Group Bounds from Vertices
            var groupMin = new Vector3(float.MaxValue);
            var groupMax = new Vector3(float.MinValue);
            
            if (group.Vertices != null && group.Vertices.Count > 0)
            {
                foreach (var v in group.Vertices)
                {
                    groupMin = Vector3.Min(groupMin, v);
                    groupMax = Vector3.Max(groupMax, v);
                }
            }
            else
            {
                // Fallback to existing bounds if no verts (unlikely for valid group)
                groupMin = group.BoundsMin;
                groupMax = group.BoundsMax;
            }

            // Update Group Data Bounds (for MOGP)
            group.BoundsMin = groupMin;
            group.BoundsMax = groupMax;
            
            // Update MOGI Bounds
            if (i < data.GroupInfos.Count)
            {
                var info = data.GroupInfos[i];
                info.BoundsMin = groupMin;
                info.BoundsMax = groupMax;
                data.GroupInfos[i] = info; // Struct update
            }
            
            // Update Global Bounds
            globalMin = Vector3.Min(globalMin, groupMin);
            globalMax = Vector3.Max(globalMax, groupMax);
        }
        
        // Update MOHD Bounds
        data.BoundsMin = globalMin;
        data.BoundsMax = globalMax;
        
        Console.WriteLine($"[DEBUG] Recalculated Bounds: {data.BoundsMin} - {data.BoundsMax}");
    }

    private void ResolveMaterialTextures(WmoV14Data data)
    {
        if (data.MotxRaw.Length == 0) return;
        
        for (int i = 0; i < data.Materials.Count; i++)
        {
            var mat = data.Materials[i];
            mat.Texture1Name = GetStringFromOffset(data.MotxRaw, mat.Texture1Offset);
            mat.Texture2Name = GetStringFromOffset(data.MotxRaw, mat.Texture2Offset);
            mat.Texture2Name = GetStringFromOffset(data.MotxRaw, mat.Texture2Offset);
            mat.Texture3Name = GetStringFromOffset(data.MotxRaw, mat.Texture3Offset);
            
            Console.WriteLine($"[DEBUG] Material {i}: Tex1='{mat.Texture1Name}' Tex2='{mat.Texture2Name}'");
            
            data.Materials[i] = mat; // update struct in list
        }
        
        Console.WriteLine("[DEBUG] Texture List (MOTX):");
        for(int t=0; t<data.Textures.Count; t++)
             Console.WriteLine($"  [{t}] {data.Textures[t]}");
    }

    private string GetStringFromOffset(byte[] table, uint offset)
    {
        if (offset >= table.Length) return string.Empty;
        int end = (int)offset;
        while (end < table.Length && table[end] != 0) end++;
        if (end > offset)
            return Encoding.UTF8.GetString(table, (int)offset, end - (int)offset);
        return string.Empty;
    }

    private void AddTextureToMotx(string texturePath, List<byte> builder, Dictionary<string, uint> offsetMap)
    {
        // Align to 1 byte (packed) - add padding if needed? 
        // For now, no padding.
        
        uint offset = (uint)builder.Count;
        offsetMap[texturePath] = offset;
        
        var fullPath = texturePath.Replace('/', '\\'); // Force backslashes
        var bytes = Encoding.UTF8.GetBytes(fullPath);
        builder.AddRange(bytes);
        builder.Add(0); // Null terminator
    }

    private void ParseMohd(BinaryReader reader, WmoV14Data data)
    {
        data.MaterialCount = reader.ReadUInt32();
        data.GroupCount = reader.ReadUInt32();
        data.PortalCount = reader.ReadUInt32();
        data.LightCount = reader.ReadUInt32();
        data.DoodadNameCount = reader.ReadUInt32();
        data.DoodadDefCount = reader.ReadUInt32();
        data.DoodadSetCount = reader.ReadUInt32();
        data.AmbientColor = reader.ReadUInt32();
        data.WmoId = reader.ReadUInt32();
        data.BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        data.BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        data.Flags = reader.ReadUInt32();
    }

    private void ParseMotx(BinaryReader reader, uint size, WmoV14Data data)
    {
        var bytes = reader.ReadBytes((int)size);
        data.MotxRaw = bytes;
        data.Textures = ParseStringTable(bytes);
    }

    private void ParseMomt(BinaryReader reader, uint size, WmoV14Data data)
    {
        // v14 MOMT is 44 bytes per material
        int count = (int)(size / 44);
        data.Materials = new List<WmoMaterial>(count);

        for (int i = 0; i < count; i++)
        {
            var mat = new WmoMaterial
            {
                Flags = reader.ReadUInt32(),
                Shader = reader.ReadUInt32(),
                BlendMode = reader.ReadUInt32(),
                Texture1Offset = reader.ReadUInt32(),
                EmissiveColor = reader.ReadUInt32(),
                FrameEmissiveColor = reader.ReadUInt32(),
                Texture2Offset = reader.ReadUInt32(),
                DiffuseColor = reader.ReadUInt32(),
                GroundType = reader.ReadUInt32(),
                Texture3Offset = reader.ReadUInt32(),
                Color2 = reader.ReadUInt32()
            };
            data.Materials.Add(mat);
        }
    }

    private void ParseMogn(BinaryReader reader, uint size, WmoV14Data data)
    {
        var bytes = reader.ReadBytes((int)size);
        data.GroupNames = ParseStringTable(bytes);
    }

    private void ParseMogi(BinaryReader reader, uint size, WmoV14Data data)
    {
        // v14 MOGI is 40 bytes per group (vs 32 in v17)
        // v14 layout: offset(4) + size(4) + flags(4) + bbox(24) + nameoffset(4) = 40
        // v17 layout: flags(4) + bbox(24) + nameoffset(4) = 32
        int count = (int)(size / 40);
        data.GroupInfos = new List<WmoGroupInfo>(count);

        for (int i = 0; i < count; i++)
        {
            // Skip v14-only fields (offset and size of the embedded group data)
            reader.ReadUInt32(); // offset - not needed for v17 separate files
            reader.ReadUInt32(); // size - not needed for v17 separate files
            
            var info = new WmoGroupInfo
            {
                Flags = reader.ReadUInt32(),
                BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                NameOffset = reader.ReadInt32()
            };
            data.GroupInfos.Add(info);
        }
    }

    private void ParseMods(BinaryReader reader, uint size, WmoV14Data data)
    {
        // MODS: Doodad sets, 32 bytes each
        int count = (int)(size / 32);
        data.DoodadSets = new List<WmoDoodadSet>(count);

        for (int i = 0; i < count; i++)
        {
            var nameBytes = reader.ReadBytes(20);
            int nullIdx = Array.IndexOf(nameBytes, (byte)0);
            var name = nullIdx >= 0 
                ? Encoding.ASCII.GetString(nameBytes, 0, nullIdx) 
                : Encoding.ASCII.GetString(nameBytes);
            
            var set = new WmoDoodadSet
            {
                Name = name,
                StartIndex = reader.ReadUInt32(),
                Count = reader.ReadUInt32()
            };
            reader.ReadUInt32(); // padding
            data.DoodadSets.Add(set);
        }
    }

    private void ParseModd(BinaryReader reader, uint size, WmoV14Data data)
    {
        // MODD: Doodad definitions, 40 bytes each (0x28 from Ghidra)
        int count = (int)(size / 40);
        data.DoodadDefs = new List<WmoDoodadDef>(count);

        for (int i = 0; i < count; i++)
        {
            var def = new WmoDoodadDef
            {
                NameIndex = reader.ReadUInt32() & 0xFFFFFF, // 24-bit index
                Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Orientation = new Quaternion(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Scale = reader.ReadSingle(),
                Color = reader.ReadUInt32()
            };
            data.DoodadDefs.Add(def);
        }
    }

    private void ParseMopv(BinaryReader reader, uint size, WmoV14Data data)
    {
        // MOPV: Portal vertices, 12 bytes each (Vector3)
        int count = (int)(size / 12);
        data.PortalVertices = new List<Vector3>(count);

        for (int i = 0; i < count; i++)
        {
            data.PortalVertices.Add(new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()));
        }
    }

    private void ParseMopt(BinaryReader reader, uint size, WmoV14Data data)
    {
        // MOPT: Portal info, 20 bytes each
        int count = (int)(size / 20);
        data.Portals = new List<WmoPortal>(count);

        for (int i = 0; i < count; i++)
        {
            var portal = new WmoPortal
            {
                StartVertex = reader.ReadUInt16(),
                Count = reader.ReadUInt16(),
                PlaneA = reader.ReadSingle(),
                PlaneB = reader.ReadSingle(),
                PlaneC = reader.ReadSingle(),
                PlaneD = reader.ReadSingle()
            };
            data.Portals.Add(portal);
        }
    }

    private void ParseMopr(BinaryReader reader, uint size, WmoV14Data data)
    {
        // MOPR: Portal refs, 8 bytes each
        int count = (int)(size / 8);
        data.PortalRefs = new List<WmoPortalRef>(count);

        for (int i = 0; i < count; i++)
        {
            var pref = new WmoPortalRef
            {
                PortalIndex = reader.ReadUInt16(),
                GroupIndex = reader.ReadUInt16(),
                Side = reader.ReadInt16()
            };
            reader.ReadUInt16(); // padding
            data.PortalRefs.Add(pref);
        }
    }

    private void ParseMolt(BinaryReader reader, uint size, WmoV14Data data)
    {
        // MOLT: Lights, 32 bytes each in v14 (no quaternion rotation)
        int count = (int)(size / 32);
        data.Lights = new List<WmoLight>(count);

        for (int i = 0; i < count; i++)
        {
            var light = new WmoLight
            {
                Type = reader.ReadByte(),
                UseAtten = reader.ReadByte() != 0
            };
            reader.ReadBytes(2); // padding
            light.Color = reader.ReadUInt32();
            light.Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            light.Intensity = reader.ReadSingle();
            light.AttenStart = reader.ReadSingle();
            light.AttenEnd = reader.ReadSingle();
            data.Lights.Add(light);
        }
    }

    private void ParseMogp(BinaryReader reader, uint size, WmoV14Data data)
    {
        var group = new WmoGroupData();
        var startPos = reader.BaseStream.Position;
        var endPos = startPos + size;

        // Alpha MOGP header is 128 bytes (0x80) based on Ghidra analysis
        // Standard fields (44 bytes)
        group.NameOffset = reader.ReadUInt32();           // +0x00
        group.DescriptiveNameOffset = reader.ReadUInt32(); // +0x04
        group.Flags = reader.ReadUInt32();                // +0x08
        group.BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()); // +0x0C - +0x17
        group.BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()); // +0x18 - +0x23
        group.PortalStart = reader.ReadUInt16();          // +0x24
        group.PortalCount = reader.ReadUInt16();          // +0x26
        group.TransBatchCount = reader.ReadUInt16();      // +0x28
        group.IntBatchCount = reader.ReadUInt16();        // +0x2A
        group.ExtBatchCount = reader.ReadUInt16();        // +0x2C
        reader.ReadUInt16();                              // +0x2E padding
        
        // Skip remaining header bytes to reach offset 0x80 where subchunks start
        // We've read 48 bytes (0x30), need to skip to 0x80
        reader.ReadBytes(0x80 - 0x30); // Skip 80 bytes (fog indices, batch info, etc.)


        // Parse sub-chunks
        while (reader.BaseStream.Position < endPos - 8)
        {
            var chunkMagic = new string(reader.ReadChars(4).Reverse().ToArray());
            var chunkSize = reader.ReadUInt32();
            var chunkEnd = reader.BaseStream.Position + chunkSize;

            switch (chunkMagic)
            {
                case "MOPY":
                    // v14 MOPY is 4 bytes per face
                    int faceCount = (int)(chunkSize / 4);
                    Console.WriteLine($"[DEBUG] MOPY: {chunkSize} bytes, {faceCount} faces");
                    // group.FaceMaterials = new List<byte>(faceCount); // Already init in struct
                    
                    // Histogram for debugging geometry assignment
                    var matCounts = new Dictionary<byte, int>();
                    
                    for (int i = 0; i < faceCount; i++)
                    {
                        var flag = reader.ReadByte(); 
                        byte matId = reader.ReadByte(); 
                        
                        // Track counts
                        if (!matCounts.ContainsKey(matId)) matCounts[matId] = 0;
                        matCounts[matId]++;

                        // REVERTED remapping for diagnosis - trusting native
                        group.FaceMaterials.Add(matId); 
                        
                        reader.ReadBytes(2); // padding
                    }
                    
                    Console.WriteLine("[DEBUG] MOPY Material Histogram:");
                    foreach(var kvp in matCounts.OrderBy(k => k.Key))
                        Console.WriteLine($"  Mat {kvp.Key}: {kvp.Value} faces");
                    break;

                case "MOVT":
                    int vertCount = (int)(chunkSize / 12);
                    Console.WriteLine($"[DEBUG] MOVT: {chunkSize} bytes, {vertCount} vertices (Stride: 12)");
                    group.Vertices = new List<Vector3>(vertCount);
                    for (int i = 0; i < vertCount; i++)
                    {
                        group.Vertices.Add(new Vector3(
                            reader.ReadSingle(),
                            reader.ReadSingle(),
                            reader.ReadSingle()));
                    }
                    break;

                case "MOIN": // v14 uses MOIN, not MOVI
                    int idxCount = (int)(chunkSize / 2);
                    Console.WriteLine($"[DEBUG] MOIN: {chunkSize} bytes, {idxCount} indices (Stride: 2)");
                    
                    // Sanity check: 0.5.3 might use uint32?
                    if (chunkSize % 2 != 0) Console.WriteLine("[ERROR] MOIN size not divisible by 2!");
                    
                    group.Indices = new List<ushort>(idxCount);
                    for (int i = 0; i < idxCount; i++)
                    {
                        group.Indices.Add(reader.ReadUInt16());
                    }
                    break;

                case "MOTV":
                    int uvCount = (int)(chunkSize / 8);
                    group.UVs = new List<Vector2>(uvCount);
                    for (int i = 0; i < uvCount; i++)
                    {
                        var u = reader.ReadSingle();
                        var v = reader.ReadSingle();
                        
                        // FIX: Flip V coordinate for correct mapping
                        group.UVs.Add(new Vector2(u, 1.0f - v));
                    }
                    break;

                case "MOBA":
                    // v14 MOBA: 24 bytes per batch
                    // Confirmed from Ghidra analysis of wowclient.exe (0.5.3):
                    // Offset 0x00: unknown byte
                    // Offset 0x01: Material ID (byte)
                    // Offset 0x02-0x0D: BBox (6 x int16)
                    // Offset 0x0E: StartIndex (ushort!) - NOT uint32!
                    // Offset 0x10: IndexCount (ushort)
                    // Offset 0x12-0x15: 4 unknown bytes (possibly vertex min/max)
                    // Offset 0x16: Flags (byte)
                    // Offset 0x17: unknown byte
                    int batchCount = (int)(chunkSize / 24);
                    Console.WriteLine($"[DEBUG] MOBA chunk: {chunkSize} bytes, {batchCount} batches (v14 format)");
                    
                    for (int b = 0; b < batchCount; b++)
                    {
                        byte unknown0 = reader.ReadByte(); // 0x00
                        byte matId = reader.ReadByte();    // 0x01 - Material ID!
                        
                        // BBox: 6 x int16 (0x02-0x0D)
                        short bx = reader.ReadInt16();
                        short by = reader.ReadInt16();
                        short bz = reader.ReadInt16();
                        short tx = reader.ReadInt16();
                        short ty = reader.ReadInt16();
                        short tz = reader.ReadInt16();
                        
                        // Index data
                        ushort startIndex = reader.ReadUInt16(); // 0x0E - ushort, NOT uint!
                        ushort indexCount = reader.ReadUInt16(); // 0x10
                        
                        // Unknown (possibly vertex range)
                        ushort unknown1 = reader.ReadUInt16(); // 0x12
                        ushort unknown2 = reader.ReadUInt16(); // 0x14
                        
                        byte flags = reader.ReadByte();        // 0x16
                        byte unknown3 = reader.ReadByte();     // 0x17
                        
                        Console.WriteLine($"[DEBUG] v14 Batch {b}: Mat={matId}, Start={startIndex}, Count={indexCount}, Flags={flags}");
                        
                        // Store parsed batch
                        group.Batches.Add(new WmoBatch
                        {
                            MaterialId = matId,
                            FirstFace = (uint)(startIndex / 3),
                            NumFaces = (ushort)(indexCount / 3),
                            FirstVertex = unknown1, // May be vertex start
                            LastVertex = unknown2,  // May be vertex end
                            Flags = flags
                        });
                    }
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        // Only rebuild batches if we didn't parse them from MOBA
        if (group.Batches.Count == 0)
        {
            Console.WriteLine("[DEBUG] No native MOBA batches, rebuilding from MOPY...");
            RebuildBatches(group);
        }
        else
        {
            Console.WriteLine($"[DEBUG] Using {group.Batches.Count} native MOBA batches");
        }

        data.Groups.Add(group);
    }
    
    private void RebuildBatches(WmoGroupData group)
    {
        // 1. Group indices by Material ID
        // Note: Indices list is flat triangles (v1, v2, v3, v1, v2, v3...)
        // MOPY list corresponds to triangles (t1, t2...)
        
        var facesByMat = new Dictionary<byte, List<ushort>>();
        int triangleCount = group.FaceMaterials.Count;
        
        for (int i = 0; i < triangleCount; i++)
        {
            byte matId = group.FaceMaterials[i];
            
            // Skip "hole" material 255? 
            // In WMOs, 255 often means hidden/collision only. 
            // v17 batches usually don't include them for rendering.
            if (matId == 255) 
                continue; 

            if (!facesByMat.ContainsKey(matId)) 
                facesByMat[matId] = new List<ushort>();
            
            // Add the 3 indices for this triangle
            if (i * 3 + 2 < group.Indices.Count)
            {
                facesByMat[matId].Add(group.Indices[i*3]);
                facesByMat[matId].Add(group.Indices[i*3+1]);
                facesByMat[matId].Add(group.Indices[i*3+2]);
            }
        }
        
        // 2. Rebuild Batches
        group.Indices.Clear(); // Clear old indices, we will rebuild sorted
        group.FaceMaterials.Clear(); // Rebuild MOPY to match index order
        group.Batches = new List<WmoBatch>();
        
        uint currentFaceStart = 0;
        
        foreach (var kvp in facesByMat.OrderBy(k => k.Key))
        {
            byte matId = kvp.Key;
            var indices = kvp.Value;
            var numFaces = (ushort)(indices.Count / 3);
            
            var batch = new WmoBatch();
            batch.MaterialId = matId;
            batch.FirstFace = currentFaceStart;
            batch.NumFaces = numFaces;
            batch.Flags = 0; // Default flags
            
            // Rebuild Indices and MOPY
            ushort minV = ushort.MaxValue, maxV = 0;
            for (int i = 0; i < indices.Count; i += 3)
            {
                 // Add 3 indices for triangle
                 ushort v1 = indices[i];
                 ushort v2 = indices[i+1];
                 ushort v3 = indices[i+2];
                 
                 group.Indices.Add(v1);
                 group.Indices.Add(v2);
                 group.Indices.Add(v3);
                 
                 // Update min/max vertex
                 if (v1 < minV) minV = v1;
                 if (v1 > maxV) maxV = v1;
                 if (v2 < minV) minV = v2;
                 if (v2 > maxV) maxV = v2;
                 if (v3 < minV) minV = v3;
                 if (v3 > maxV) maxV = v3;
                 
                 // Add material entry for this face (triangle)
                 group.FaceMaterials.Add(matId);
            }
            
            batch.FirstVertex = minV;
            batch.LastVertex = maxV;
            
            // Create "Infinite" Bounding Box (-30000 to +30000) to safely prevent culling
            // Using logic: minX, minY, minZ, maxX, maxY, maxZ (shorts)
            batch.BoundingBoxRaw = new byte[12];
            using (var ms = new MemoryStream(batch.BoundingBoxRaw))
            using (var bw = new BinaryWriter(ms))
            {
                short safeMin = -30000;
                short safeMax = 30000;
                bw.Write(safeMin); bw.Write(safeMin); bw.Write(safeMin);
                bw.Write(safeMax); bw.Write(safeMax); bw.Write(safeMax);
            }
            
            group.Batches.Add(batch);
            currentFaceStart += numFaces;
            
            Console.WriteLine($"[DEBUG] Rebuilt Batch Mat={matId}: Faces={numFaces}, Verts=[{minV}-{maxV}]");
        }
    }

    private void WriteRootFile(WmoV14Data data, string outputPath)
    {
        var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
        Directory.CreateDirectory(outputDir);

        using var stream = File.Create(outputPath);
        using var writer = new BinaryWriter(stream);

        // MVER (version 17)
        WriteChunk(writer, "MVER", w => w.Write((uint)17));

        // MOHD
        WriteChunk(writer, "MOHD", w =>
        {
            // Debug: Print ambient color components
            byte ambR = (byte)(data.AmbientColor & 0xFF);
            byte ambG = (byte)((data.AmbientColor >> 8) & 0xFF);
            byte ambB = (byte)((data.AmbientColor >> 16) & 0xFF);
            byte ambA = (byte)((data.AmbientColor >> 24) & 0xFF);
            Console.WriteLine($"[DEBUG] MOHD AmbientColor: R={ambR} G={ambG} B={ambB} A={ambA} (Raw: 0x{data.AmbientColor:X8})");
            
            w.Write(data.MaterialCount);
            w.Write((uint)data.Groups.Count);
            w.Write((uint)data.Portals.Count);   // portals
            w.Write((uint)data.Lights.Count);    // lights
            w.Write((uint)data.DoodadDefs.Count); // models (doodad count, per wowdev: "nModels: number of M2 models imported")
            w.Write((uint)data.DoodadDefs.Count); // doodads (doodad definitions)
            w.Write((uint)Math.Max(1, data.DoodadSets.Count)); // doodad sets (at least 1)
            w.Write(data.AmbientColor);
            w.Write(data.WmoId);
            WriteVector3(w, data.BoundsMin);
            WriteVector3(w, data.BoundsMax);
            w.Write((ushort)data.Flags);
            w.Write((ushort)0); // LOD
        });

        // Write MOTX and update Material offsets
        var newOffsets = new Dictionary<string, uint>();
        var motxBuilder = new List<byte>();
        
        foreach (var mat in data.Materials)
        {
            // Ensure unique strings are added
            if (!string.IsNullOrEmpty(mat.Texture1Name) && !newOffsets.ContainsKey(mat.Texture1Name))
                AddTextureToMotx(mat.Texture1Name, motxBuilder, newOffsets);
            if (!string.IsNullOrEmpty(mat.Texture2Name) && !newOffsets.ContainsKey(mat.Texture2Name))
                AddTextureToMotx(mat.Texture2Name, motxBuilder, newOffsets);
            if (!string.IsNullOrEmpty(mat.Texture3Name) && !newOffsets.ContainsKey(mat.Texture3Name))
                AddTextureToMotx(mat.Texture3Name, motxBuilder, newOffsets);
        }

        // Add any unused textures from original list just in case
        foreach (var tex in data.Textures)
        {
             if (!newOffsets.ContainsKey(tex))
                AddTextureToMotx(tex, motxBuilder, newOffsets);
        }

        // MOTX
        WriteChunk(writer, "MOTX", w =>
        {
            w.Write(motxBuilder.ToArray());
        });

        // MOMT (expand to v17 64-byte format)
        WriteChunk(writer, "MOMT", w =>
        {
            foreach (var mat in data.Materials)
            {
                uint off1 = !string.IsNullOrEmpty(mat.Texture1Name) && newOffsets.ContainsKey(mat.Texture1Name) ? newOffsets[mat.Texture1Name] : 0;
                uint off2 = !string.IsNullOrEmpty(mat.Texture2Name) && newOffsets.ContainsKey(mat.Texture2Name) ? newOffsets[mat.Texture2Name] : 0;
                uint off3 = !string.IsNullOrEmpty(mat.Texture3Name) && newOffsets.ContainsKey(mat.Texture3Name) ? newOffsets[mat.Texture3Name] : 0;
                
                w.Write(mat.Flags);
                w.Write(mat.Shader);
                w.Write(mat.BlendMode);
                w.Write(off1); // Texture1Offset
                w.Write(mat.EmissiveColor);
                w.Write(mat.FrameEmissiveColor);
                w.Write(off2); // Texture2Offset
                w.Write(mat.DiffuseColor);
                w.Write(mat.GroundType);
                w.Write(off3); // Texture3Offset
                w.Write(mat.Color2);
                w.Write((uint)0); // flags2
                w.Write(new byte[16]); // runtime data
            }
        });

        // MOGN
        WriteChunk(writer, "MOGN", w =>
        {
            foreach (var name in data.GroupNames)
            {
                var bytes = Encoding.UTF8.GetBytes(name);
                w.Write(bytes);
                w.Write((byte)0);
            }
        });

        // MOGI (v17 is 32 bytes)
        WriteChunk(writer, "MOGI", w =>
        {
            int gi = 0;
            foreach (var info in data.GroupInfos)
            {
                // Debug: print group flags
                bool isInterior = (info.Flags & 0x2000) != 0;
                bool isExterior = (info.Flags & 0x8) != 0;
                Console.WriteLine($"[DEBUG] MOGI Group {gi}: Flags=0x{info.Flags:X8} Interior={isInterior} Exterior={isExterior}");
                
                w.Write(info.Flags);
                WriteVector3(w, info.BoundsMin);
                WriteVector3(w, info.BoundsMax);
                w.Write(info.NameOffset);
                gi++;
            }
        });

        // MOSB (skybox - empty string padded to 4)
        WriteChunk(writer, "MOSB", w =>
        {
            w.Write(new byte[4]); // Empty null-terminated string, padded
        });

        // MOPV (portal vertices)
        WriteChunk(writer, "MOPV", w =>
        {
            foreach (var v in data.PortalVertices)
                WriteVector3(w, v);
        });

        // MOPT (portal info, 20 bytes each)
        WriteChunk(writer, "MOPT", w =>
        {
            foreach (var p in data.Portals)
            {
                w.Write(p.StartVertex);
                w.Write(p.Count);
                w.Write(p.PlaneA);
                w.Write(p.PlaneB);
                w.Write(p.PlaneC);
                w.Write(p.PlaneD);
            }
        });

        // MOPR (portal refs, 8 bytes each)
        WriteChunk(writer, "MOPR", w =>
        {
            foreach (var pr in data.PortalRefs)
            {
                w.Write(pr.PortalIndex);
                w.Write(pr.GroupIndex);
                w.Write(pr.Side);
                w.Write((ushort)0); // padding
            }
        });

        // MOVV (visible vertices - empty)
        WriteChunk(writer, "MOVV", w => { }); // 0 bytes

        // MOVB (visible blocks - empty)
        WriteChunk(writer, "MOVB", w => { }); // 0 bytes

        // MOLT (lights - v17 has 48 bytes per light with quaternion rotation)
        WriteChunk(writer, "MOLT", w =>
        {
            foreach (var lt in data.Lights)
            {
                w.Write(lt.Type);
                w.Write(lt.UseAtten ? (byte)1 : (byte)0);
                w.Write((ushort)0); // padding
                w.Write(lt.Color);
                WriteVector3(w, lt.Position);
                w.Write(lt.Intensity);
                // v17 adds quaternion rotation (16 bytes) - use identity
                w.Write(0f); w.Write(0f); w.Write(0f); w.Write(1f);
                w.Write(lt.AttenStart);
                w.Write(lt.AttenEnd);
            }
        });

        // MODS (doodad sets)
        WriteChunk(writer, "MODS", w =>
        {
            if (data.DoodadSets.Count > 0)
            {
                foreach (var set in data.DoodadSets)
                {
                    var nameBytes = Encoding.UTF8.GetBytes(set.Name ?? "Set_$DefaultGlobal");
                    w.Write(nameBytes);
                    w.Write(new byte[20 - Math.Min(nameBytes.Length, 20)]);
                    w.Write(set.StartIndex);
                    w.Write(set.Count);
                    w.Write((uint)0); // padding
                }
            }
            else
            {
                // Write minimal default set
                var setName = Encoding.UTF8.GetBytes("Set_$DefaultGlobal");
                w.Write(setName);
                w.Write(new byte[20 - setName.Length]);
                w.Write((uint)0); // firstIndex
                w.Write((uint)0); // count
                w.Write((uint)0); // padding
            }
        });

        // MODN (doodad names - raw string table)
        WriteChunk(writer, "MODN", w =>
        {
            if (data.DoodadNamesRaw.Length > 0)
                w.Write(data.DoodadNamesRaw);
        });

        // MODD (doodad defs - 40 bytes each)
        WriteChunk(writer, "MODD", w =>
        {
            foreach (var dd in data.DoodadDefs)
            {
                w.Write(dd.NameIndex); // 24-bit index + 8-bit flags
                WriteVector3(w, dd.Position);
                w.Write(dd.Orientation.X);
                w.Write(dd.Orientation.Y);
                w.Write(dd.Orientation.Z);
                w.Write(dd.Orientation.W);
                w.Write(dd.Scale);
                w.Write(dd.Color);
            }
        });

        // MFOG (fog - one default entry required)
        WriteChunk(writer, "MFOG", w =>
        {
            // Default fog entry (48 bytes)
            w.Write((uint)0); // flags
            w.Write(0f); w.Write(0f); w.Write(0f); // pos
            w.Write(0f); // smaller_radius
            w.Write(0f); // larger_radius
            // Fog[0] - normal fog
            w.Write(444.4445f); // end
            w.Write(0.25f); // start_scalar
            w.Write((uint)0); // color (BGRA)
            // Fog[1] - underwater fog
            w.Write(222.2222f); // end
            w.Write(-0.5f); // start_scalar
            w.Write((uint)0); // color (BGRA)
        });

        Console.WriteLine($"[DEBUG] Wrote v17 root: {data.Groups.Count} groups, {data.Materials.Count} materials, {data.DoodadDefs.Count} doodads");
    }

    private void WriteGroupFiles(WmoV14Data data, string rootPath)
    {
        var baseName = Path.GetFileNameWithoutExtension(rootPath);
        var outputDir = Path.GetDirectoryName(rootPath) ?? ".";

        for (int i = 0; i < data.Groups.Count; i++)
        {
            var group = data.Groups[i];
            var groupPath = Path.Combine(outputDir, $"{baseName}_{i:D3}.wmo");
            WriteGroupFile(group, groupPath, i);
        }
    }

    private void WriteGroupFile(WmoGroupData group, string outputPath, int groupIndex)
    {
        using var stream = File.Create(outputPath);
        using var writer = new BinaryWriter(stream);

        // MVER
        WriteChunk(writer, "MVER", w => w.Write((uint)17));

        // MOGP (header + subchunks)
        var mogpStart = writer.BaseStream.Position;
        writer.Write(Encoding.ASCII.GetBytes("PGOM")); // Reversed
        var sizePos = writer.BaseStream.Position;
        writer.Write((uint)0); // Size placeholder

        var mogpDataStart = writer.BaseStream.Position;

        // MOGP header (68 bytes per noggit-red wmo_group_header)
        // Fix flags based on what we are about to write
        // 0x1    = has_bsp_tree (MOBN/MOBR) - We generate default
        // 0x4    = has_vertex_color (MOCV) - We generate/preserve
        // 0x200  = has_light (MOLR) - We DON'T generate (Clear)
        // 0x400  = has_MPBV etc - We DON'T generate (Clear)
        // 0x800  = has_doodads (MODR) - We preserve if present
        // 0x1000 = has_liquid (MLIQ) - We preserve if present

        var fixedFlags = group.Flags;
        fixedFlags |= 0x1u; // MOBN
        fixedFlags |= 0x4u; // MOCV
        fixedFlags &= ~0x200u; // No MOLR
        fixedFlags &= ~0x400u; // No MPBV
        
        // MODR
        if (group.DoodadRefs != null && group.DoodadRefs.Count > 0)
            fixedFlags |= 0x800u;
        else
            fixedFlags &= ~0x800u;

        // MLIQ
        if (group.LiquidData != null && group.LiquidData.Length > 0)
            fixedFlags |= 0x1000u;
        else
            fixedFlags &= ~0x1000u;

        
        writer.Write(group.NameOffset);                      // +0x00 (4)
        writer.Write(group.DescriptiveNameOffset);           // +0x04 (4)
        writer.Write(fixedFlags);                            // +0x08 (4)
        WriteVector3(writer, group.BoundsMin);               // +0x0C (12)
        WriteVector3(writer, group.BoundsMax);               // +0x18 (12)
        writer.Write(group.PortalStart);                     // +0x24 (2)
        writer.Write(group.PortalCount);                     // +0x26 (2)
        writer.Write(group.TransBatchCount);                 // +0x28 (2)
        writer.Write(group.IntBatchCount);                   // +0x2A (2)
        writer.Write(group.ExtBatchCount);                   // +0x2C (2)
        writer.Write((ushort)0);                             // +0x2E (2)
        writer.Write(new byte[4]);                           // +0x30 (4) fogs
        writer.Write((uint)0);                               // +0x34 (4) liquid
        writer.Write((uint)0);                               // +0x38 (4) id
        writer.Write((int)0);                                // +0x3C (4) unk2
        writer.Write((int)0);                                // +0x40 (4) unk3

        // STRICT CHUNK ORDER (Noggit/Standard 3.3.5 Requirement):
        // MOPY -> MOVI -> MOVT -> MONR -> MOTV -> MOBA -> [Optional: MOLR, MODR, MOBN, MOCV, MLIQ]

        // 1. MOPY (Materials)
        // FIX: Verify face count against indices to prevent "exploded" geometry (reading past MOVI)
        int validFaceCount = group.Indices.Count / 3;
        WriteSubChunk(writer, "MOPY", w =>
        {
            for (int i = 0; i < validFaceCount; i++)
            {
                byte matId = (i < group.FaceMaterials.Count) ? group.FaceMaterials[i] : (byte)0;
                w.Write((byte)0); // flags
                w.Write(matId);
            }
        });

        // 2. MOVI (Indices) - Restored from MOIN
        WriteSubChunk(writer, "MOVI", w =>
        {
            foreach (var idx in group.Indices)
                w.Write(idx);
        });

        // 3. MOVT (Vertices)
        WriteSubChunk(writer, "MOVT", w =>
        {
            foreach (var v in group.Vertices)
                WriteVector3(w, v);
        });

        // 4. MONR (Normals)
        WriteSubChunk(writer, "MONR", w =>
        {
            var normals = GenerateNormals(group);
            foreach (var n in normals)
                WriteVector3(w, n);
        });

        // 5. MOTV (UVs)
        WriteSubChunk(writer, "MOTV", w =>
        {
            foreach (var uv in group.UVs)
            {
                w.Write(uv.X);
                w.Write(uv.Y);
            }
        });
        
        // 6. MOLV - Removed (Not in Noggit/Standard 3.3.5)

        // 7. MOBA (Batches)
        WriteSubChunk(writer, "MOBA", w =>
        {
            if (group.Batches != null)
            {
                foreach (var b in group.Batches)
                {
                    // 24 bytes per batch (v17)
                    if (b.BoundingBoxRaw != null && b.BoundingBoxRaw.Length == 12)
                        w.Write(b.BoundingBoxRaw);
                    else
                        w.Write(new byte[12]); // Empty box
                    
                    w.Write((uint)(b.FirstFace * 3)); // v17 expects Index Start (Offset), not Face Index
                    w.Write((ushort)(b.NumFaces * 3)); // v17 expects Index Count, not Face Count
                    w.Write(b.FirstVertex);
                    w.Write(b.LastVertex);
                    w.Write(b.Flags);
                    w.Write(b.MaterialId);
                }
            }
        });

        // --- Optional Chunks (Flag Dependent) ---

        // 8. MOLR (Light Refs) - Skipped (Flag 0x200 cleared)

        // 9. MODR (Doodad Refs) - checking 0x800
        if ((fixedFlags & 0x800) != 0)
        {
             WriteSubChunk(writer, "MODR", w =>
             {
                 if (group.DoodadRefs != null)
                 {
                     foreach(var dr in group.DoodadRefs) w.Write(dr);
                 }
             });
        }

        // 10. MOBN + MOBR (BSP Tree) - Flag 0x1
        // Writing dummy leaf node
        WriteSubChunk(writer, "MOBN", w =>
        {
            w.Write((short)4); // 4 = Leaf
            w.Write((short)0); // flags?
            w.Write((short)group.Indices.Count / 3); // nFaces
            w.Write((int)0); // faceStart
            w.Write(0f); // planeDist
        });
        
        WriteSubChunk(writer, "MOBR", w =>
        {
            // Face indices for BSP (0, 1, 2...)
            for (ushort f = 0; f < group.Indices.Count / 3; f++)
                w.Write(f);
        });

        // 11. MPBV (Portals) - Skipped (Flag 0x400 cleared)

        // 12. MOCV (Vertex Colors) - Flag 0x4
        WriteSubChunk(writer, "MOCV", w =>
        {
            if (group.VertexColors != null && group.VertexColors.Count > 0)
            {
                 foreach (var c in group.VertexColors)
                     w.Write(c);
            }
            else
            {
                // Write white for every vertex if missing
                for (int v = 0; v < group.Vertices.Count; v++)
                    w.Write(0xFFFFFFFF); // BGRA White
            }
        });

        // 13. MLIQ (Liquid) - Flag 0x1000
        if ((fixedFlags & 0x1000) != 0)
        {
            WriteSubChunk(writer, "MLIQ", w =>
            {
                 if (group.LiquidData != null) w.Write(group.LiquidData);
            });
        }

        // Update MOGP size
        var endPos = writer.BaseStream.Position;
        var mogpSize = (uint)(endPos - mogpDataStart);
        writer.BaseStream.Position = sizePos;
        writer.Write(mogpSize);
    }



    private List<string> ParseStringTable(byte[] data)
    {
        var result = new List<string>();
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                {
                    var str = Encoding.UTF8.GetString(data, start, i - start);
                    if (!string.IsNullOrEmpty(str))
                        result.Add(str);
                }
                start = i + 1;
            }
        }
        return result;
    }

    private List<Vector3> GenerateNormals(WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i < normals.Length; i++)
            normals[i] = Vector3.Zero;

        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            var i0 = group.Indices[i];
            var i1 = group.Indices[i + 1];
            var i2 = group.Indices[i + 2];

            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                continue;

            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var normal = Vector3.Normalize(Vector3.Cross(e1, e2));

            normals[i0] += normal;
            normals[i1] += normal;
            normals[i2] += normal;
        }

        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitY).ToList();
    }

    private void WriteChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData)
    {
        var reversed = new string(chunkId.Reverse().ToArray());
        writer.Write(Encoding.ASCII.GetBytes(reversed));
        var sizePos = writer.BaseStream.Position;
        writer.Write((uint)0);
        var dataStart = writer.BaseStream.Position;
        writeData(writer);
        var dataEnd = writer.BaseStream.Position;
        writer.BaseStream.Position = sizePos;
        writer.Write((uint)(dataEnd - dataStart));
        writer.BaseStream.Position = dataEnd;
    }

    private void WriteSubChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData)
        => WriteChunk(writer, chunkId, writeData);

    private void WriteVector3(BinaryWriter writer, Vector3 v)
    {
        writer.Write(v.X);
        writer.Write(v.Y);
        writer.Write(v.Z);
    }

    #region Data Structures

    public class WmoV14Data
    {
        public uint Version;
        public uint MaterialCount, GroupCount, PortalCount, LightCount;
        public uint DoodadNameCount, DoodadDefCount, DoodadSetCount;
        public uint AmbientColor, WmoId, Flags;
        public Vector3 BoundsMin, BoundsMax;
        public List<string> Textures = new();
        public List<WmoMaterial> Materials = new();
        
        // Temp storage for raw MOTX to resolve offsets
        public byte[] MotxRaw = Array.Empty<byte>();
        public List<string> GroupNames = new();
        public List<WmoGroupInfo> GroupInfos = new();
        public List<WmoGroupData> Groups = new();
        
        // Doodad data
        public List<WmoDoodadSet> DoodadSets = new();
        public byte[] DoodadNamesRaw = Array.Empty<byte>();
        public List<WmoDoodadDef> DoodadDefs = new();
        
        // Portal data
        public List<Vector3> PortalVertices = new();
        public List<WmoPortal> Portals = new();
        public List<WmoPortalRef> PortalRefs = new();
        
        // Light data
        public List<WmoLight> Lights = new();
    }

    public struct WmoMaterial
    {
        public uint Flags, Shader, BlendMode;
        public uint Texture1Offset, EmissiveColor, FrameEmissiveColor;
        public uint Texture2Offset, DiffuseColor, GroundType;
        public uint Texture3Offset, Color2;
        
        // Resolved names
        public string Texture1Name, Texture2Name, Texture3Name;
    }

    public struct WmoGroupInfo
    {
        public uint Flags;
        public Vector3 BoundsMin, BoundsMax;
        public int NameOffset;
    }

    public class WmoGroupData
    {
        public uint NameOffset, DescriptiveNameOffset, Flags;
        public Vector3 BoundsMin, BoundsMax;
        public ushort PortalStart, PortalCount;
        public ushort TransBatchCount, IntBatchCount, ExtBatchCount;
        public List<Vector3> Vertices = new();
        public List<ushort> Indices = new();
        public List<Vector2> UVs = new();
        public List<byte> FaceMaterials = new();
        public List<WmoBatch> Batches = new();
        
        // New features for parity with Legacy Exporter
        public List<ushort> DoodadRefs = new(); // MODR
        public List<uint> VertexColors = new(); // MOCV (BGRA)
        public byte[] LiquidData = Array.Empty<byte>(); // MLIQ
    }

    public struct WmoBatch
    {
        public byte[] BoundingBoxRaw; // 12 bytes (2x 3x int16)
        public uint FirstFace;
        public ushort NumFaces, FirstVertex, LastVertex;
        public byte Flags, MaterialId;
    }

    public struct WmoDoodadSet
    {
        public string Name;
        public uint StartIndex;
        public uint Count;
    }

    public struct WmoDoodadDef
    {
        public uint NameIndex;
        public Vector3 Position;
        public Quaternion Orientation;
        public float Scale;
        public uint Color;
    }

    public struct WmoPortal
    {
        public ushort StartVertex;
        public ushort Count;
        public float PlaneA, PlaneB, PlaneC, PlaneD;
    }

    public struct WmoPortalRef
    {
        public ushort PortalIndex;
        public ushort GroupIndex;
        public short Side;
    }

    public struct WmoLight
    {
        public byte Type;
        public bool UseAtten;
        public uint Color;
        public Vector3 Position;
        public float Intensity;
        public float AttenStart, AttenEnd;
    }

    #endregion
}
