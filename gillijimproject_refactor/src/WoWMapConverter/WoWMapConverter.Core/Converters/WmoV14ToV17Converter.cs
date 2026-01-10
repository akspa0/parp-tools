using System.Numerics;
using System.Text;
using WoWMapConverter.Core.Services;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts WMO v14 (Alpha) to WMO v17 (LK 3.3.5) format.
/// Handles monolithic Alpha WMO → split root + group files.
/// </summary>
public class WmoV14ToV17Converter
{
    /// <summary>
    /// Convert a v14 WMO file to v17 format.
    /// Automatically handles per-asset MPQ archives (.wmo.MPQ).
    /// </summary>
    public List<string> Convert(string inputPath, string outputPath)
    {
        Console.WriteLine($"[INFO] Converting {Path.GetFileName(inputPath)} to v17...");
        
        // Try reading from file directly or from MPQ archive
        var data = AlphaMpqReader.ReadWithMpqFallback(inputPath);
        if (data == null)
            throw new FileNotFoundException($"WMO not found (checked direct and MPQ): {inputPath}");
        
        return ConvertFromBytes(data, outputPath);
    }
    
    /// <summary>
    /// Convert WMO v14 data from bytes to v17 format.
    /// </summary>
    public List<string> ConvertFromBytes(byte[] wmoData, string outputPath)
    {
        using var ms = new MemoryStream(wmoData);
        using var reader = new BinaryReader(ms);

        var data = ParseWmoV14Internal(reader);
        
        // Ensure output directory exists (including group files)
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        WriteRootFile(data, outputPath);
        
        // Write group files
        WriteGroupFiles(data, outputPath);

        Console.WriteLine($"[SUCCESS] Converted to v17: {outputPath}");
        
        return data.Textures;
    }

    /// <summary>
    /// Parse WMO v14 file and return structured data for analysis.
    /// </summary>
    public WmoV14Data ParseWmoV14(string inputPath)
    {
        var data = AlphaMpqReader.ReadWithMpqFallback(inputPath);
        if (data == null)
            throw new FileNotFoundException($"WMO not found (checked direct and MPQ): {inputPath}");
        
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        var wmoData = ParseWmoV14Internal(reader);
        
        // Populate group names from MOGN/MOGI
        for (int i = 0; i < wmoData.Groups.Count && i < wmoData.GroupInfos.Count; i++)
        {
            var nameOfs = wmoData.GroupInfos[i].NameOffset;
            wmoData.Groups[i].Name = GetGroupName(wmoData.GroupNames, nameOfs) ?? $"group_{i}";
        }
        
        return wmoData;
    }
    
    private string? GetGroupName(List<string> groupNames, int nameOffset)
    {
        // MOGN is a string table; nameOffset is a byte offset into the packed strings
        // For now, use index-based lookup (simplified)
        if (nameOffset >= 0 && nameOffset < groupNames.Count)
            return groupNames[nameOffset];
        return null;
    }

    private WmoV14Data ParseWmoV14Internal(BinaryReader reader)
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
        
        // Synchronize flags between MOGP/MOGI (Critical for visibility)
        SynchronizeFlags(data);
        
        // Sort/Partition batches (Trans vs Opaque) and update counts
        // v14 batches might be mixed; v17 requires Trans first, then Opaque
        SortBatches(data);
        
        return data;
    }
    
    private void SortBatches(WmoV14Data data)
    {
        for (int i = 0; i < data.Groups.Count; i++)
        {
            var group = data.Groups[i];
            if (group.Batches == null || group.Batches.Count == 0) continue;
            
            var transBatches = new List<WmoBatch>();
            var opaqueBatches = new List<WmoBatch>();
            
            foreach (var batch in group.Batches)
            {
                // Check material blend mode
                // BlendMode 0=Opaque, 1=AlphaKey (Cutout) -> Treat as Opaque for sorting
                // BlendMode 2+=AlphaBlend/Additive/etc -> Treat as Transparent
                bool isTrans = false;
                if (batch.MaterialId < data.Materials.Count)
                {
                    var mat = data.Materials[batch.MaterialId];
                    if (mat.BlendMode >= 2)
                        isTrans = true;
                }
                
                if (isTrans)
                    transBatches.Add(batch);
                else
                    opaqueBatches.Add(batch);
            }
            
            // Reconstruct list: Trans first, then Opaque
            // (Standard 3.3.5 convention: Trans batches come first in the list)
            group.Batches.Clear();
            group.Batches.AddRange(transBatches);
            group.Batches.AddRange(opaqueBatches);
            
            // Update counts based on group type
            // Determine if Interior or Exterior based on Flags
            bool isInterior = (group.Flags & 0x2000) != 0;
            
            group.TransBatchCount = (ushort)transBatches.Count;
            if (isInterior)
            {
                group.IntBatchCount = (ushort)opaqueBatches.Count;
                group.ExtBatchCount = 0;
            }
            else
            {
                group.IntBatchCount = 0;
                group.ExtBatchCount = (ushort)opaqueBatches.Count;
            }
            
            // Debug log
            // Console.WriteLine($"[DEBUG] Sorted Batches Group {i}: {group.TransBatchCount} Trans, {group.IntBatchCount} Int, {group.ExtBatchCount} Ext");
        }
    }

    private void SynchronizeFlags(WmoV14Data data)
    {
        for (int i = 0; i < data.Groups.Count; i++)
        {
            var group = data.Groups[i];
            
            // Calculate v17 flags (Interior/Exterior/Lighting/etc.)
            // exact same logic as used for writing, but now applied to BOTH MOGI and MOGP
            uint v17Flags = CalculateV17GroupFlags(group);
            
            // Update MOGP Flags
            group.Flags = v17Flags;
            
            // Update MOGI Flags (if available)
            if (i < data.GroupInfos.Count)
            {
                var info = data.GroupInfos[i];
                info.Flags = v17Flags;
                data.GroupInfos[i] = info;
            }
            
            // Log changes for debugging
            bool isInterior = (v17Flags & 0x2000) != 0;
            bool isExterior = (v17Flags & 0x8) != 0;
            // Console.WriteLine($"[DEBUG] Group {i} Flags Synced: 0x{v17Flags:X8} (Int={isInterior}, Ext={isExterior})");
        }
    }
    
    private uint CalculateV17GroupFlags(WmoGroupData group)
    {
        var fixedFlags = group.Flags;
        
        // 0x1 = has_bsp_tree (MOBN/MOBR) - We always generate BSP
        fixedFlags |= 0x1u; 
        
        // INTERIOR heuristic: If NOT pure Exterior (0x8), then set Interior (0x2000)
        // v14 often lacks 0x2000/0x8 explicitly set for what we consider "interior"
        bool isExterior = (fixedFlags & 0x8) != 0;
        bool isInterior = !isExterior;
        
        if (isInterior)
        {
            fixedFlags |= 0x2000u; // Set Interior flag
        }
        
        // MOCV - Set for ALL groups that have vertex colors
        if (group.VertexColors != null && group.VertexColors.Count > 0)
        {
            fixedFlags |= 0x4u;
        }
        else
        {
            fixedFlags &= ~0x4u;
        }
        
        // Clear unsupported/generated-on-write flags
        fixedFlags &= ~0x200u; // No MOLR
        fixedFlags &= ~0x400u; // No MPBV
        
        // 0x40 = exterior_lit - Set for exterior groups
        if (isExterior)
        {
            fixedFlags |= 0x40u;
        }
        
        // MODR (doodads)
        if (group.DoodadRefs != null && group.DoodadRefs.Count > 0)
            fixedFlags |= 0x800u;
        else
            fixedFlags &= ~0x800u;

        // MLIQ (liquid)
        if (group.LiquidData != null && group.LiquidData.Length > 0)
            fixedFlags |= 0x1000u;
        else
            fixedFlags &= ~0x1000u;
            
        return fixedFlags;
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
            
            // Debug: Print parsed DiffuseColor
            byte b = (byte)(mat.DiffuseColor & 0xFF);
            byte g = (byte)((mat.DiffuseColor >> 8) & 0xFF);
            byte r = (byte)((mat.DiffuseColor >> 16) & 0xFF);
            byte a = (byte)((mat.DiffuseColor >> 24) & 0xFF);
            Console.WriteLine($"[DEBUG] Material {i} DiffuseColor: R={r} G={g} B={b} A={a} (Raw: 0x{mat.DiffuseColor:X8})");
            
            // DiffuseColor preserved from v14 (black = 0xFF000000)
            
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
                        
                        // FIX TENTATIVE: Users reported upside down textures.
                        // Standard WoW is D3D-style (0,0 top-left). Noggit/GL is 0,0 bottom-left.
                        // 1.0 - v is usually correct for conversion.
                        // If it's STILL upside down, maybe v14 is ALREADY GL-style? (Unlikely)
                        // Or maybe we shouldn't flip?
                        // Let's KEEP the flip for now, as removing it would flip them back if they were correct.
                        // If "Upside Down" means "Vertically Mirrored", then 1-v fixes it.
                        // If they are mirrored NOW, then we should remove it.
                        // Let's try removing it ONLY if requested.
                        // User Request: "looks like textures are upside down" (implying CURRENT state is wrong).
                        // Current state has `1.0 - v`. So `1.0 - v` makes it upside down?
                        // Let's try REMOVING the flip.
                        group.UVs.Add(new Vector2(u, v));
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
                        // FIX: v14 StartIndex is 32-bit (0x0E-0x11)
                        // This was the cause of "drop-outs" (High word read as count = 0)
                        uint startIndex = reader.ReadUInt32();   // 0x0E
                        ushort indexCount = reader.ReadUInt16(); // 0x12
                        
                        // Unknown (possibly vertex range or padding) -> Usually 0 in v14
                        ushort unknown1 = reader.ReadUInt16();   // 0x14
                        
                        byte flags = reader.ReadByte();          // 0x16
                        byte unknown2 = reader.ReadByte();       // 0x17
                        
                        // Console.WriteLine($"[DEBUG] v14 Batch {b}: Mat={matId}, Start={startIndex}, Count={indexCount}, Flags={flags}");
                        
                        // Store parsed batch
                        group.Batches.Add(new WmoBatch
                        {
                            MaterialId = matId,
                            FirstFace = startIndex / 3, // StartIndex is an index into indices array (triplets?) No, usually direct index.
                            // Convert to Face Index: v14 seems to use index offsets.
                            // However, WmoBatch expects FirstFace (i.e. index / 3).
                            // Let's assume startIndex is index offset.
                            NumFaces = (ushort)(indexCount / 3),
                            FirstVertex = 0, // Will be recalculated
                            LastVertex = 0,  // Will be recalculated
                            Flags = flags
                        });
                    }
                    break;
                
                case "MOCV":
                    // v14 MOCV: 4 bytes per vertex (BGRA)
                    int colorCount = (int)(chunkSize / 4);
                    Console.WriteLine($"[DEBUG] MOCV: {chunkSize} bytes, {colorCount} vertex colors (v14 format)");
                    group.VertexColors = new List<uint>(colorCount);
                    for (int i = 0; i < colorCount; i++)
                    {
                        group.VertexColors.Add(reader.ReadUInt32());
                    }
                    // Debug: Print first few colors
                    for (int i = 0; i < Math.Min(3, colorCount); i++)
                    {
                        uint c = group.VertexColors[i];
                        Console.WriteLine($"  Parsed Color[{i}]: B={c & 0xFF} G={(c >> 8) & 0xFF} R={(c >> 16) & 0xFF} A={(c >> 24) & 0xFF}");
                    }
                    break;
                    
                case "MOLV":
                    // v14 MOLV: Lightmap UVs per face-vertex (C2Vector = 8 bytes)
                    int uvLmCount = (int)(chunkSize / 8);
                    Console.WriteLine($"[DEBUG] MOLV: {chunkSize} bytes, {uvLmCount} lightmap UVs");
                    group.LightmapUVs = new List<Vector2>(uvLmCount);
                    for (int i = 0; i < uvLmCount; i++)
                    {
                        group.LightmapUVs.Add(new Vector2(
                            reader.ReadSingle(),
                            reader.ReadSingle()));
                    }
                    break;
                    
                case "MOLM":
                    // v14 MOLM: Lightmap Info - structure unclear, logging raw
                    Console.WriteLine($"[WARNING] Found MOLM (Lightmap Info) chunk ({chunkSize} bytes) - Parsing...");
                    // According to wiki, MOLM is per-batch lightmap metadata
                    // Try to parse common structures: could be (uint32 offset, uint16 width, uint16 height) = 8 bytes
                    int lmInfoCount = (int)(chunkSize / 8);
                    group.LightmapInfos = new List<LightmapInfo>(lmInfoCount);
                    for (int i = 0; i < lmInfoCount; i++)
                    {
                        group.LightmapInfos.Add(new LightmapInfo
                        {
                            DataOffset = reader.ReadUInt32(),
                            Width = reader.ReadUInt16(),
                            Height = reader.ReadUInt16()
                        });
                    }
                    Console.WriteLine($"[DEBUG] Parsed {lmInfoCount} lightmap info entries");
                    break;
                    
                case "MOLD":
                    // v14 MOLD: Raw lightmap pixel data
                    Console.WriteLine($"[DEBUG] MOLD: {chunkSize} bytes of lightmap pixel data");
                    group.LightmapData = reader.ReadBytes((int)chunkSize);
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
            // Calculate bounding boxes AND fix potentially invalid vertex ranges (v14 doesn't have unknown_box)
            RecalculateBatchData(group);
        }
        
        // v14 FIX: Validate and regenerate indices if needed
        // v14 MOVT contains nFaces*3 vertices (sequential per face, not indexed)
        // If MOIN is empty/invalid, generate sequential indices (0,1,2, 3,4,5...)
        int expectedIndices = group.FaceMaterials.Count * 3;
        if (group.Indices.Count != expectedIndices)
        {
            Console.WriteLine($"[FIX] v14 Index Mismatch: MOIN has {group.Indices.Count} indices but expected {expectedIndices} (nFaces={group.FaceMaterials.Count} * 3)");
            Console.WriteLine($"      Regenerating sequential indices 0,1,2, 3,4,5, ... {expectedIndices-1}");
            group.Indices = new List<ushort>(expectedIndices);
            for (int i = 0; i < expectedIndices; i++)
            {
                group.Indices.Add((ushort)i);
            }
        }
        
        // DISABLED: Mesh repair breaks batch data (minIndex/maxIndex become invalid)
        // TODO: Need to recalculate batch bounds and unknown_box after vertex welding
        // RepairMesh(group);

        data.Groups.Add(group);
    }
    
    /// <summary>
    /// Repair mesh geometry by welding nearby vertices and removing degenerate triangles.
    /// </summary>
    private void RepairMesh(WmoGroupData group)
    {
        const float WELD_EPSILON = 0.0001f; // Distance threshold for welding
        const float AREA_EPSILON = 0.00001f; // Minimum triangle area
        
        int originalVertexCount = group.Vertices.Count;
        int originalFaceCount = group.FaceMaterials.Count;
        
        // ===== Phase 1: Vertex Welding =====
        // Build spatial hash for fast vertex lookup
        var vertexMap = new Dictionary<int, int>(); // old index -> new index
        var newVertices = new List<Vector3>();
        var newUVs = new List<Vector2>();
        var newColors = new List<uint>();
        var newLightmapUVs = new List<Vector2>();
        
        for (int i = 0; i < group.Vertices.Count; i++)
        {
            var v = group.Vertices[i];
            int foundIndex = -1;
            
            // Search for existing vertex within epsilon (requires BOTH position AND UV match)
            // This preserves intentional seams between facade/shell geometry
            var currentUV = i < group.UVs.Count ? group.UVs[i] : new Vector2(0, 0);
            
            for (int j = 0; j < newVertices.Count; j++)
            {
                var existing = newVertices[j];
                float dx = Math.Abs(v.X - existing.X);
                float dy = Math.Abs(v.Y - existing.Y);
                float dz = Math.Abs(v.Z - existing.Z);
                
                // Check position match
                if (dx < WELD_EPSILON && dy < WELD_EPSILON && dz < WELD_EPSILON)
                {
                    // Also check UV match to preserve texture seams
                    var existingUV = j < newUVs.Count ? newUVs[j] : new Vector2(0, 0);
                    float du = Math.Abs(currentUV.X - existingUV.X);
                    float dv = Math.Abs(currentUV.Y - existingUV.Y);
                    
                    if (du < WELD_EPSILON && dv < WELD_EPSILON)
                    {
                        foundIndex = j;
                        break;
                    }
                }
            }
            
            if (foundIndex >= 0)
            {
                // Map to existing vertex
                vertexMap[i] = foundIndex;
            }
            else
            {
                // Add new vertex
                vertexMap[i] = newVertices.Count;
                newVertices.Add(v);
                
                // Also copy associated data
                if (i < group.UVs.Count)
                    newUVs.Add(group.UVs[i]);
                if (i < group.VertexColors.Count)
                    newColors.Add(group.VertexColors[i]);
                if (i < group.LightmapUVs.Count)
                    newLightmapUVs.Add(group.LightmapUVs[i]);
            }
        }
        
        // Remap indices
        for (int i = 0; i < group.Indices.Count; i++)
        {
            int oldIdx = group.Indices[i];
            if (vertexMap.TryGetValue(oldIdx, out int newIdx))
            {
                group.Indices[i] = (ushort)newIdx;
            }
        }
        
        // Replace vertex data
        group.Vertices = newVertices;
        if (newUVs.Count > 0) group.UVs = newUVs;
        if (newColors.Count > 0) group.VertexColors = newColors;
        if (newLightmapUVs.Count > 0) group.LightmapUVs = newLightmapUVs;
        
        int weldedCount = originalVertexCount - newVertices.Count;
        
        // ===== Phase 2: Degenerate Triangle Removal =====
        var validFaces = new List<int>(); // indices of valid face triplets
        int degenerateCount = 0;
        
        for (int faceIdx = 0; faceIdx < group.FaceMaterials.Count; faceIdx++)
        {
            int baseIdx = faceIdx * 3;
            if (baseIdx + 2 >= group.Indices.Count)
            {
                degenerateCount++;
                continue;
            }
            
            int i0 = group.Indices[baseIdx];
            int i1 = group.Indices[baseIdx + 1];
            int i2 = group.Indices[baseIdx + 2];
            
            // Check for duplicate indices
            if (i0 == i1 || i1 == i2 || i0 == i2)
            {
                degenerateCount++;
                continue;
            }
            
            // Check for valid vertex indices
            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
            {
                degenerateCount++;
                continue;
            }
            
            // Calculate triangle area using cross product
            var v0 = group.Vertices[i0];
            var v1 = group.Vertices[i1];
            var v2 = group.Vertices[i2];
            
            var e1 = new Vector3(v1.X - v0.X, v1.Y - v0.Y, v1.Z - v0.Z);
            var e2 = new Vector3(v2.X - v0.X, v2.Y - v0.Y, v2.Z - v0.Z);
            
            // Cross product
            var cross = new Vector3(
                e1.Y * e2.Z - e1.Z * e2.Y,
                e1.Z * e2.X - e1.X * e2.Z,
                e1.X * e2.Y - e1.Y * e2.X
            );
            
            float areaSq = cross.X * cross.X + cross.Y * cross.Y + cross.Z * cross.Z;
            
            if (areaSq < AREA_EPSILON)
            {
                degenerateCount++;
                continue;
            }
            
            validFaces.Add(faceIdx);
        }
        
        // Rebuild indices and materials with only valid faces
        if (degenerateCount > 0)
        {
            var newIndices = new List<ushort>();
            var newMaterials = new List<byte>();
            
            foreach (int faceIdx in validFaces)
            {
                int baseIdx = faceIdx * 3;
                newIndices.Add(group.Indices[baseIdx]);
                newIndices.Add(group.Indices[baseIdx + 1]);
                newIndices.Add(group.Indices[baseIdx + 2]);
                newMaterials.Add(group.FaceMaterials[faceIdx]);
            }
            
            group.Indices = newIndices;
            group.FaceMaterials = newMaterials;
        }
        
        // Report results
        if (weldedCount > 0 || degenerateCount > 0)
        {
            Console.WriteLine($"[MESH REPAIR] Welded {weldedCount} vertices ({originalVertexCount} → {newVertices.Count})");
            Console.WriteLine($"[MESH REPAIR] Removed {degenerateCount} degenerate faces ({originalFaceCount} → {validFaces.Count})");
        }
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
            
            // Calculate actual bounding box from vertices in this batch
            // unknown_box: bx,by,bz (min) and tx,ty,tz (max) as int16
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            
            for (int i = 0; i < indices.Count; i++)
            {
                int vIdx = indices[i];
                if (vIdx < group.Vertices.Count)
                {
                    var v = group.Vertices[vIdx];
                    if (v.X < minX) minX = v.X;
                    if (v.Y < minY) minY = v.Y;
                    if (v.Z < minZ) minZ = v.Z;
                    if (v.X > maxX) maxX = v.X;
                    if (v.Y > maxY) maxY = v.Y;
                    if (v.Z > maxZ) maxZ = v.Z;
                }
            }
            
            // Convert to int16, rounding away from zero
            short bx = (short)Math.Floor(minX);
            short by = (short)Math.Floor(minY);
            short bz = (short)Math.Floor(minZ);
            short tx = (short)Math.Ceiling(maxX);
            short ty = (short)Math.Ceiling(maxY);
            short tz = (short)Math.Ceiling(maxZ);
            
            batch.BoundingBoxRaw = new byte[12];
            using (var ms = new MemoryStream(batch.BoundingBoxRaw))
            using (var bw = new BinaryWriter(ms))
            {
                bw.Write(bx); bw.Write(by); bw.Write(bz);
                bw.Write(tx); bw.Write(ty); bw.Write(tz);
            }
            
            group.Batches.Add(batch);
            currentFaceStart += numFaces;
            
            Console.WriteLine($"[DEBUG] Rebuilt Batch Mat={matId}: Faces={numFaces}, Verts=[{minV}-{maxV}], Box=[{bx},{by},{bz}]-[{tx},{ty},{tz}]");
        }
    }
    
    /// <summary>
    /// Calculate bounding boxes (unknown_box) for batches parsed from v14.
    /// v14 doesn't have the bx,by,bz,tx,ty,tz fields that v17 requires.
    /// </summary>
    /// <summary>
    /// Recalculate batch data (Vertex Start/End, Bounding Box) from actual geometry.
    /// v14 MOBA "unknown" fields (0x12, 0x14) are NOT reliable vertex ranges.
    /// We must scan the index buffer to get true min/max vertices to prevent culling issues.
    /// </summary>
    private void RecalculateBatchData(WmoGroupData group)
    {
        for (int batchIdx = 0; batchIdx < group.Batches.Count; batchIdx++)
        {
            var batch = group.Batches[batchIdx];
            
            // Calculate true min/max vertex index and bounding box from the index buffer
            // FirstFace and NumFaces are reliable (from MOBA 0x0E/0x10)
            
            uint faceStart = batch.FirstFace;
            uint numFaces = batch.NumFaces;
            uint indexStart = faceStart * 3;
            uint indexCount = (uint)(numFaces * 3);
            
            if (indexStart + indexCount > group.Indices.Count)
            {
                Console.WriteLine($"[WARNING] Batch {batchIdx} indices out of range! Start={indexStart} Count={indexCount} Total={group.Indices.Count}");
                continue;
            }

            ushort minV = ushort.MaxValue;
            ushort maxV = 0;
            
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
            
            bool hasValidIndices = false;

            for (int i = 0; i < indexCount; i++)
            {
                ushort vIdx = group.Indices[(int)(indexStart + i)];
                
                // Track min/max vertex index
                if (vIdx < minV) minV = vIdx;
                if (vIdx > maxV) maxV = vIdx;
                
                // Track bounding box
                if (vIdx < group.Vertices.Count)
                {
                    var vert = group.Vertices[vIdx];
                    if (vert.X < minX) minX = vert.X;
                    if (vert.Y < minY) minY = vert.Y;
                    if (vert.Z < minZ) minZ = vert.Z;
                    if (vert.X > maxX) maxX = vert.X;
                    if (vert.Y > maxY) maxY = vert.Y;
                    if (vert.Z > maxZ) maxZ = vert.Z;
                    hasValidIndices = true;
                }
            }
            
            if (!hasValidIndices)
            {
                Console.WriteLine($"[WARNING] Batch {batchIdx} has no valid indices/vertices!");
                continue;
            }

            // Update batch with TRUE vertex range
            // This is critical for incorrect culling when v14 inputs are garbage
            if (batch.FirstVertex != minV || batch.LastVertex != maxV)
            {
                // Console.WriteLine($"[FIX] Batch {batchIdx} Vertex Range: Old={batch.FirstVertex}-{batch.LastVertex} -> New={minV}-{maxV}");
                batch.FirstVertex = minV;
                batch.LastVertex = maxV;
            }

            // Recalculate Bounding Box (always, don't trust v14)
            // Convert to int16, rounding away from zero (conservative bounds)
            short bx = (short)Math.Floor(minX);
            short by = (short)Math.Floor(minY);
            short bz = (short)Math.Floor(minZ);
            short tx = (short)Math.Ceiling(maxX);
            short ty = (short)Math.Ceiling(maxY);
            short tz = (short)Math.Ceiling(maxZ);
            
            batch.BoundingBoxRaw = new byte[12];
            using (var ms = new MemoryStream(batch.BoundingBoxRaw))
            using (var bw = new BinaryWriter(ms))
            {
                bw.Write(bx); bw.Write(by); bw.Write(bz);
                bw.Write(tx); bw.Write(ty); bw.Write(tz);
            }
            
            // Update the batch in the list (struct copy)
            group.Batches[batchIdx] = batch;
            
            // Console.WriteLine($"[DEBUG] Recalculated Batch {batchIdx} Box: [{bx},{by},{bz}]-[{tx},{ty},{tz}]");
        }
    }

    /// <summary>
    /// Convert v14 lightmap data (MOLM/MOLD/MOLV) to v17 vertex colors (MOCV).
    /// If no lightmap data exists, generates neutral gray vertex colors.
    /// </summary>
    private void GenerateMocvFromLightmaps(WmoGroupData group)
    {
        int vertexCount = group.Vertices.Count;
        
        // If we already have valid MOCV from parsing, check if it's usable
        if (group.VertexColors.Count == vertexCount && group.VertexColors.Count > 0)
        {
            // Calculate average luminosity to detect placeholder/dark colors
            double avgLum = 0;
            foreach (var c in group.VertexColors)
            {
                byte b = (byte)(c & 0xFF);
                byte g = (byte)((c >> 8) & 0xFF);
                byte r = (byte)((c >> 16) & 0xFF);
                avgLum += (r + g + b) / 3.0;
            }
            avgLum /= group.VertexColors.Count;
            
            Console.WriteLine($"[DEBUG] MOCV Stats: Avg Luminosity = {avgLum:F2}");
            
            if (avgLum >= 10) // Valid colors, keep them
            {
                Console.WriteLine("[FIX] MOCV: Valid colors detected. Forced all vertex alphas to 0xFF (Opaque).");
                for (int i = 0; i < group.VertexColors.Count; i++)
                {
                    group.VertexColors[i] = (group.VertexColors[i] & 0x00FFFFFF) | 0xFF000000;
                }
                return;
            }
            else
            {
                Console.WriteLine("[FIX] MOCV Detected as Dark/Placeholder (AvgLum < 10). Regenerating...");
            }
        }
        
        // Check if we have lightmap data to sample
        if (group.LightmapData.Length > 0 && group.LightmapUVs.Count > 0 && group.LightmapInfos.Count > 0)
        {
            Console.WriteLine($"[DEBUG] Converting lightmap to MOCV: {group.LightmapInfos.Count} lightmaps, {group.LightmapUVs.Count} UVs, {group.LightmapData.Length} bytes pixel data");
            
            // Generate MOCV by sampling lightmap textures
            group.VertexColors = new List<uint>(vertexCount);
            
            // Simple approach: For each vertex, find corresponding UV and sample lightmap
            // MOLV is per-face-vertex (3 per triangle), we need to map to unique vertices
            var vertexColorAccum = new Dictionary<int, List<uint>>();
            
            // MOLD format assumption: BGRA 32-bit per pixel packed in each lightmap
            for (int faceIdx = 0; faceIdx < group.FaceMaterials.Count && faceIdx * 3 + 2 < group.Indices.Count; faceIdx++)
            {
                // Get lightmap info for this face (use first if only one)
                int lmIdx = Math.Min(faceIdx, group.LightmapInfos.Count - 1);
                var lmInfo = group.LightmapInfos[lmIdx];
                
                if (lmInfo.Width == 0 || lmInfo.Height == 0) continue;
                
                for (int c = 0; c < 3; c++) // 3 vertices per face
                {
                    int uvIdx = faceIdx * 3 + c;
                    int vertIdx = group.Indices[faceIdx * 3 + c];
                    
                    if (uvIdx >= group.LightmapUVs.Count) continue;
                    
                    var uv = group.LightmapUVs[uvIdx];
                    
                    // Clamp UV to [0,1]
                    float u = Math.Clamp(uv.X, 0f, 1f);
                    float v = Math.Clamp(uv.Y, 0f, 1f);
                    
                    // Calculate pixel coordinates
                    int px = (int)(u * (lmInfo.Width - 1));
                    int py = (int)(v * (lmInfo.Height - 1));
                    
                    // Calculate offset in MOLD data (4 bytes per pixel)
                    int pixelOffset = (int)(lmInfo.DataOffset + (py * lmInfo.Width + px) * 4);
                    
                    if (pixelOffset + 4 <= group.LightmapData.Length)
                    {
                        uint color = BitConverter.ToUInt32(group.LightmapData, pixelOffset);
                        
                        if (!vertexColorAccum.ContainsKey(vertIdx))
                            vertexColorAccum[vertIdx] = new List<uint>();
                        vertexColorAccum[vertIdx].Add(color);
                    }
                }
            }
            
            // Average sampled colors per vertex
            for (int i = 0; i < vertexCount; i++)
            {
                if (vertexColorAccum.TryGetValue(i, out var colors) && colors.Count > 0)
                {
                    int avgB = 0, avgG = 0, avgR = 0;
                    foreach (var c in colors)
                    {
                        avgB += (int)(c & 0xFF);
                        avgG += (int)((c >> 8) & 0xFF);
                        avgR += (int)((c >> 16) & 0xFF);
                    }
                    avgB /= colors.Count;
                    avgG /= colors.Count;
                    avgR /= colors.Count;
                    
                    group.VertexColors.Add((uint)(avgB | (avgG << 8) | (avgR << 16) | 0xFF000000));
                }
                else
                {
                    // No lightmap sample, use neutral gray
                    group.VertexColors.Add(0xFF808080);
                }
            }
            
            Console.WriteLine($"[DEBUG] Generated {group.VertexColors.Count} vertex colors from lightmap sampling");
            return;
        }
        
        // Fallback: Generate neutral gray MOCV
        Console.WriteLine($"[FIX] No lightmap data available. Generating neutral gray MOCV ({vertexCount} vertices)");
        group.VertexColors = new List<uint>(vertexCount);
        for (int i = 0; i < vertexCount; i++)
        {
            // Neutral gray (RGB=128) with full alpha
            group.VertexColors.Add(0xFF808080);
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
                
                // FIX: Set F_UNLIT flag (0x1) to disable lighting calculations
                // v14 WMOs use lightmaps (MOLM/MOLD) which don't exist in v17/3.3.5
                // Without MOCV vertex colors, groups render black unless F_UNLIT is set
                uint fixedFlags = mat.Flags | 0x1u; // F_UNLIT = disable lighting
                
                w.Write(fixedFlags);
                w.Write(mat.Shader);
                w.Write(mat.BlendMode);
                w.Write(off1); // Texture1Offset
                w.Write(mat.EmissiveColor);
                w.Write(mat.FrameEmissiveColor);
                w.Write(off2); // Texture2Offset
                
                // FIX: v14 materials often have black DiffuseColor which causes black rendering
                // Replace with neutral gray if the color is too dark
                uint diffuseColor = mat.DiffuseColor;
                byte b = (byte)(diffuseColor & 0xFF);
                byte g = (byte)((diffuseColor >> 8) & 0xFF);
                byte r = (byte)((diffuseColor >> 16) & 0xFF);
                int luminance = (r + g + b) / 3;
                if (luminance < 32) // Too dark
                {
                    diffuseColor = 0xFF808080; // Neutral gray
                    Console.WriteLine($"[FIX] Material DiffuseColor was too dark (L={luminance}), set to neutral gray");
                }
                w.Write(diffuseColor);
                
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

        // MODN (doodad names - raw string table with MDX→M2 remapping)
        WriteChunk(writer, "MODN", w =>
        {
            if (data.DoodadNamesRaw.Length > 0)
            {
                // Parse string table and remap .mdx → .m2
                var names = ParseStringTable(data.DoodadNamesRaw);
                var remappedBuilder = new MemoryStream();
                var remappedWriter = new BinaryWriter(remappedBuilder, Encoding.UTF8);
                
                int mdxCount = 0;
                foreach (var name in names)
                {
                    var remapped = name;
                    if (name.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                    {
                        // Remap .mdx to .m2 for LK compatibility
                        remapped = name.Substring(0, name.Length - 4) + ".m2";
                        mdxCount++;
                    }
                    
                    // Write null-terminated string
                    remappedWriter.Write(Encoding.UTF8.GetBytes(remapped));
                    remappedWriter.Write((byte)0);
                }
                
                if (mdxCount > 0)
                    Console.WriteLine($"[DEBUG] Remapped {mdxCount} MDX doodad paths to M2");
                
                w.Write(remappedBuilder.ToArray());
            }
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
            
            // DISABLED: Made textures and geometry worse
            // ValidateAndRepairGroupData(group, i);
            
            // Convert v14 lightmap data to MOCV vertex colors before writing
            GenerateMocvFromLightmaps(group);
            
            var groupPath = Path.Combine(outputDir, $"{baseName}_{i:D3}.wmo");
            WriteGroupFile(group, groupPath, i);
        }
    }
    
    /// <summary>
    /// Validate and repair group data to prevent drop-outs and rendering issues.
    /// Ensures UV count matches vertex count, indices are in range, etc.
    /// </summary>
    private void ValidateAndRepairGroupData(WmoGroupData group, int groupIndex)
    {
        int vertexCount = group.Vertices.Count;
        int indexCount = group.Indices.Count;
        int faceCount = group.FaceMaterials.Count;
        
        // 0. Flip UV V-coordinate (v14 uses different texture coordinate system than v17)
        // This fixes inverted textures where top/bottom are swapped
        for (int i = 0; i < group.UVs.Count; i++)
        {
            var uv = group.UVs[i];
            group.UVs[i] = new Vector2(uv.X, 1.0f - uv.Y); // Flip V
        }
        
        // 1. Ensure UV count matches vertex count
        if (group.UVs.Count != vertexCount)
        {
            Console.WriteLine($"[REPAIR] Group {groupIndex}: UV count mismatch ({group.UVs.Count} UVs vs {vertexCount} vertices)");
            
            // Rebuild UVs from existing data or generate default
            var newUVs = new List<Vector2>(vertexCount);
            for (int i = 0; i < vertexCount; i++)
            {
                if (i < group.UVs.Count)
                    newUVs.Add(group.UVs[i]);
                else
                    newUVs.Add(new Vector2(0, 0)); // Default UV
            }
            group.UVs = newUVs;
            Console.WriteLine($"[REPAIR] Group {groupIndex}: Fixed UV count to {group.UVs.Count}");
        }
        
        // 2. Validate indices are in range
        int outOfRangeCount = 0;
        for (int i = 0; i < indexCount; i++)
        {
            if (group.Indices[i] >= vertexCount)
            {
                outOfRangeCount++;
                group.Indices[i] = 0; // Reset to 0 to prevent crash
            }
        }
        if (outOfRangeCount > 0)
        {
            Console.WriteLine($"[REPAIR] Group {groupIndex}: Fixed {outOfRangeCount} out-of-range indices");
        }
        
        // 3. Ensure face count matches index count / 3
        int expectedFaces = indexCount / 3;
        if (faceCount != expectedFaces)
        {
            Console.WriteLine($"[REPAIR] Group {groupIndex}: Face count mismatch ({faceCount} faces vs {expectedFaces} expected)");
            
            // Rebuild face materials
            var newMaterials = new List<byte>(expectedFaces);
            for (int i = 0; i < expectedFaces; i++)
            {
                if (i < group.FaceMaterials.Count)
                    newMaterials.Add(group.FaceMaterials[i]);
                else
                    newMaterials.Add(0); // Default material
            }
            group.FaceMaterials = newMaterials;
            Console.WriteLine($"[REPAIR] Group {groupIndex}: Fixed face count to {group.FaceMaterials.Count}");
        }
        
        // 4. Always rebuild batches from face materials to ensure consistency
        Console.WriteLine($"[REPAIR] Group {groupIndex}: Rebuilding batches for consistency");
        RebuildBatches(group);
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
        // Flags are now PRE-CALCULATED via SynchronizeFlags() to ensure MOGI/MOGP consistency.
        // We just use group.Flags directly.

        var fixedFlags = group.Flags;
        
        // Debug output for confirmation
        bool isInterior = (fixedFlags & 0x2000) != 0;
        bool isExterior = (fixedFlags & 0x8) != 0;
        Console.WriteLine($"[DEBUG] Writing Group Flags: 0x{fixedFlags:X8} (Int={isInterior}, Ext={isExterior})");

        
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
            // Valid MOBN Node (16 bytes)
            // Offset   Type    Name        Description
            // 0x00     uint16  Flags       4 = Leaf, 0 = Branch/Node
            // 0x02     int16   NegChild    Index of negative child node (or -1/0 if leaf)
            // 0x04     int16   PosChild    Index of positive child node (or -1/0 if leaf)
            // 0x06     uint16  nFaces      Number of faces in this node (0 if branch)
            // 0x08     uint32  FaceStart   Index into MOBR array (first face index)
            // 0x0C     float   PlaneDist   Distance to splitting plane
            
            w.Write((ushort)4); // Flags: 4 = Leaf
            w.Write((short)-1); // NegChild: -1 (Leaf/NoChild) - Noggit defines Flag_NoChild = 0xFFFF
            w.Write((short)-1); // PosChild: -1 (Leaf/NoChild)
            w.Write((ushort)(group.Indices.Count / 3)); // nFaces: All faces in this single leaf
            w.Write((uint)0);   // FaceStart: Start at 0 in MOBR
            w.Write(0f);        // PlaneDist: 0 for leaf
        });
        
        WriteSubChunk(writer, "MOBR", w =>
        {
            // Face indices for BSP (0, 1, 2...)
            for (ushort f = 0; f < group.Indices.Count / 3; f++)
                w.Write(f);
        });

        // 10b. MPBP, MPBI, MPBG - Only required if flag 0x400 is set.
        // Input has 0x2007 (No 0x400), so we must NOT write these.


        // 11. MPBV (Portals) - Skipped (Flag 0x400 cleared)

        // 12. MOCV (Vertex Colors) - Flag 0x4
        if ((fixedFlags & 0x4) != 0)
        {
            WriteSubChunk(writer, "MOCV", w =>
            {
                Console.WriteLine($"[DEBUG] MOCV: Writing {group.VertexColors.Count} parsed vertex colors");
                foreach (var c in group.VertexColors)
                    w.Write(c);
            });
        }
        else
        {
            Console.WriteLine("[DEBUG] MOCV: SKIPPED (no vertex colors parsed)");
        }

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
        public string? Name; // Group name from MOGN
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
        
        // v14 Lightmap data (to be converted to MOCV)
        public List<Vector2> LightmapUVs = new(); // MOLV - per-face-vertex UVs
        public byte[] LightmapData = Array.Empty<byte>(); // MOLD - raw lightmap pixels
        public List<LightmapInfo> LightmapInfos = new(); // MOLM - lightmap metadata
    }
    
    // v14 Lightmap info structure
    public struct LightmapInfo
    {
        public uint DataOffset;  // Offset into MOLD data
        public ushort Width;     // Lightmap texture width
        public ushort Height;    // Lightmap texture height
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
