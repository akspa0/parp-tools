using System.Numerics;
using System.Text;
using WoWMapConverter.Core.Services;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Extended Converter for WMO v14/v15 hybrids or variants.
/// Implements stricter checks and corrected output logic based on Ghidra verification.
/// </summary>
public class WmoV14ToV17ExtendedConverter
{
    public List<string> Convert(string inputPath, string outputPath)
    {
        Console.WriteLine($"[INFO] Converting {Path.GetFileName(inputPath)} to v17 (Extended Mode)...");
        var data = AlphaMpqReader.ReadWithMpqFallback(inputPath);
        if (data == null)
            throw new FileNotFoundException($"WMO not found: {inputPath}");
        return ConvertFromBytes(data, outputPath);
    }
    
    public List<string> ConvertFromBytes(byte[] wmoData, string outputPath)
    {
        using var ms = new MemoryStream(wmoData);
        using var reader = new BinaryReader(ms);

        var data = ParseWmoV14(reader);
        
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        WriteRootFile(data, outputPath);
        
        WriteGroupFiles(data, outputPath);

        Console.WriteLine($"[SUCCESS] Converted to v17: {outputPath}");
        
        return data.Textures;
    }

    private WmoV14ToV17Converter.WmoV14Data ParseWmoV14(BinaryReader reader)
    {
        var data = new WmoV14ToV17Converter.WmoV14Data();

        // Read MVER
        var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (magic != "REVM") 
            throw new InvalidDataException($"Expected MVER, got {magic}");
        
        var size = reader.ReadUInt32();
        data.Version = reader.ReadUInt32();
        
        // Extended mode might be more lenient or handle v15?
        if (data.Version != 14)
             Console.WriteLine($"[WARN] WMO Version is {data.Version} (Expected 14). Proceeding in Extended Mode.");

        // Read MOMO container 
        magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (magic != "OMOM") 
            throw new InvalidDataException($"Expected MOMO container, got {magic}");
        
        var momoSize = reader.ReadUInt32();
        var momoEnd = reader.BaseStream.Position + momoSize;

        // Parse chunks within MOMO
        while (reader.BaseStream.Position < momoEnd)
        {
            var chunkMagic = new string(reader.ReadChars(4).Reverse().ToArray());
            var chunkSize = reader.ReadUInt32();
            var chunkEnd = reader.BaseStream.Position + chunkSize;

            switch (chunkMagic)
            {
                case "MOHD": ParseMohd(reader, data); break;
                case "MOTX": ParseMotx(reader, chunkSize, data); break;
                case "MOMT": ParseMomt(reader, chunkSize, data); break;
                case "MOGN": ParseMogn(reader, chunkSize, data); break;
                case "MOGI": ParseMogi(reader, chunkSize, data); break;
                case "MODS": ParseMods(reader, chunkSize, data); break;
                case "MODN": data.DoodadNamesRaw = reader.ReadBytes((int)chunkSize); break;
                case "MODD": ParseModd(reader, chunkSize, data); break;
                case "MOPV": ParseMopv(reader, chunkSize, data); break;
                case "MOPT": ParseMopt(reader, chunkSize, data); break;
                case "MOPR": ParseMopr(reader, chunkSize, data); break;
                case "MOLT": ParseMolt(reader, chunkSize, data); break;
                default: break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        // MOGP groups (Outside MOMO)
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

                reader.BaseStream.Position = chunkEnd;
            }
            catch
            {
                break;
            }
        }
        
        // Resolve material specific textures
        ResolveMaterialTextures(data);
        
        // Recalculate bounds 
        RecalculateBounds(data);

        return data;
    }

    // --- Copied Helpers ---

    private void ParseMohd(BinaryReader reader, WmoV14ToV17Converter.WmoV14Data data)
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

    private void ParseMotx(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        var bytes = reader.ReadBytes((int)size);
        data.MotxRaw = bytes;
        data.Textures = ParseStringTable(bytes);
    }

    private void ParseMomt(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 44);
        data.Materials = new List<WmoV14ToV17Converter.WmoMaterial>(count);
        for (int i = 0; i < count; i++)
        {
            data.Materials.Add(new WmoV14ToV17Converter.WmoMaterial
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
            });
        }
    }

    private void ParseMogn(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        var bytes = reader.ReadBytes((int)size);
        data.GroupNames = ParseStringTable(bytes);
    }
    
    private void ParseMogi(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 40);
        data.GroupInfos = new List<WmoV14ToV17Converter.WmoGroupInfo>(count);
        for (int i = 0; i < count; i++)
        {
            reader.ReadUInt32(); // offset
            reader.ReadUInt32(); // size
            data.GroupInfos.Add(new WmoV14ToV17Converter.WmoGroupInfo
            {
                Flags = reader.ReadUInt32(),
                BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                NameOffset = reader.ReadInt32()
            });
        }
    }

    private void ParseMods(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 32);
        data.DoodadSets = new List<WmoV14ToV17Converter.WmoDoodadSet>(count);
        for (int i = 0; i < count; i++)
        {
            var nameBytes = reader.ReadBytes(20);
            int nullIdx = Array.IndexOf(nameBytes, (byte)0);
            var name = nullIdx >= 0 ? Encoding.ASCII.GetString(nameBytes, 0, nullIdx) : Encoding.ASCII.GetString(nameBytes);
            
            var set = new WmoV14ToV17Converter.WmoDoodadSet
            {
                Name = name,
                StartIndex = reader.ReadUInt32(),
                Count = reader.ReadUInt32()
            };
            reader.ReadUInt32(); // padding
            data.DoodadSets.Add(set);
        }
    }

    private void ParseModd(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 40);
        data.DoodadDefs = new List<WmoV14ToV17Converter.WmoDoodadDef>(count);
        for (int i = 0; i < count; i++)
        {
             var def = new WmoV14ToV17Converter.WmoDoodadDef
            {
                NameIndex = reader.ReadUInt32() & 0xFFFFFF,
                Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Orientation = new Quaternion(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Scale = reader.ReadSingle(),
                Color = reader.ReadUInt32()
            };
            data.DoodadDefs.Add(def);
        }
    }

    private void ParseMopv(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 12);
        data.PortalVertices = new List<Vector3>(count);
        for (int i = 0; i < count; i++)
            data.PortalVertices.Add(new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()));
    }

    private void ParseMopt(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 20);
        data.Portals = new List<WmoV14ToV17Converter.WmoPortal>(count);
        for (int i = 0; i < count; i++)
        {
            data.Portals.Add(new WmoV14ToV17Converter.WmoPortal
            {
                StartVertex = reader.ReadUInt16(),
                Count = reader.ReadUInt16(),
                PlaneA = reader.ReadSingle(), PlaneB = reader.ReadSingle(), PlaneC = reader.ReadSingle(), PlaneD = reader.ReadSingle()
            });
        }
    }

    private void ParseMopr(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 8);
        data.PortalRefs = new List<WmoV14ToV17Converter.WmoPortalRef>(count);
        for (int i = 0; i < count; i++)
        {
            var pref = new WmoV14ToV17Converter.WmoPortalRef
            {
                PortalIndex = reader.ReadUInt16(),
                GroupIndex = reader.ReadUInt16(),
                Side = reader.ReadInt16()
            };
            reader.ReadUInt16(); // padding
            data.PortalRefs.Add(pref);
        }
    }

    private void ParseMolt(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        int count = (int)(size / 32);
        data.Lights = new List<WmoV14ToV17Converter.WmoLight>(count);
        for (int i = 0; i < count; i++)
        {
            var light = new WmoV14ToV17Converter.WmoLight
            {
                Type = reader.ReadByte(),
                UseAtten = reader.ReadByte() != 0
            };
            reader.ReadBytes(2); 
            light.Color = reader.ReadUInt32();
            light.Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
            light.Intensity = reader.ReadSingle();
            light.AttenStart = reader.ReadSingle();
            light.AttenEnd = reader.ReadSingle();
            data.Lights.Add(light);
        }
    }

    private void ParseMogp(BinaryReader reader, uint size, WmoV14ToV17Converter.WmoV14Data data)
    {
        var group = new WmoV14ToV17Converter.WmoGroupData();
        var startPos = reader.BaseStream.Position;
        var endPos = startPos + size;

        group.NameOffset = reader.ReadUInt32();
        group.DescriptiveNameOffset = reader.ReadUInt32();
        group.Flags = reader.ReadUInt32();
        group.BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        group.BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        group.PortalStart = reader.ReadUInt16();
        group.PortalCount = reader.ReadUInt16();
        group.TransBatchCount = reader.ReadUInt16();
        group.IntBatchCount = reader.ReadUInt16();
        group.ExtBatchCount = reader.ReadUInt16();
        reader.ReadUInt16(); 

        reader.ReadBytes(0x80 - 0x30); // Skip header

        while (reader.BaseStream.Position < endPos - 8)
        {
            var chunkMagic = new string(reader.ReadChars(4).Reverse().ToArray());
            var chunkSize = reader.ReadUInt32();
            var chunkEnd = reader.BaseStream.Position + chunkSize;

            switch (chunkMagic)
            {
                case "MOPY":
                    int faceCount = (int)(chunkSize / 4);
                    // Use a Dictionary to handle potentially out of order material IDs in MOPY vs Batches
                    for (int i = 0; i < faceCount; i++)
                    {
                        reader.ReadByte(); 
                        group.FaceMaterials.Add(reader.ReadByte()); 
                        reader.ReadBytes(2); 
                    }
                    if (group.FaceMaterials.Count == 0) Console.WriteLine("[WARN] MOPY empty!");
                    break;
                case "MOVT":
                    int nVerts = (int)(chunkSize / 12);
                    group.Vertices = new List<Vector3>(nVerts);
                    for(int i=0; i<nVerts; i++) group.Vertices.Add(new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()));
                    break;
                case "MOIN":
                    int nIdx = (int)(chunkSize / 2);
                    group.Indices = new List<ushort>(nIdx);
                    for(int i=0; i<nIdx; i++) group.Indices.Add(reader.ReadUInt16());
                    break;
                case "MOTV":
                    int nUv = (int)(chunkSize / 8);
                    group.UVs = new List<Vector2>(nUv);
                     for (int i = 0; i < nUv; i++) group.UVs.Add(new Vector2(reader.ReadSingle(), 1.0f - reader.ReadSingle()));
                    break;
                case "MOBA":
                    int nBatches = (int)(chunkSize / 24);
                    for (int b = 0; b < nBatches; b++)
                    {
                        reader.ReadByte(); // 0x00
                        byte matId = reader.ReadByte(); // 0x01
                        byte[] bbox = reader.ReadBytes(12);
                        ushort startIndex = reader.ReadUInt16();
                        ushort count = reader.ReadUInt16();
                        ushort u1 = reader.ReadUInt16();
                        ushort u2 = reader.ReadUInt16();
                        byte flags = reader.ReadByte();
                        reader.ReadByte();

                        group.Batches.Add(new WmoV14ToV17Converter.WmoBatch
                        {
                            MaterialId = matId,
                            FirstFace = (uint)(startIndex / 3),
                            NumFaces = (ushort)(count / 3),
                            FirstVertex = u1,
                            LastVertex = u2,
                            Flags = flags,
                            BoundingBoxRaw = bbox
                        });
                    }
                    break;
                case "MOCV":
                    int nColors = (int)(chunkSize / 4);
                    group.VertexColors = new List<uint>(nColors);
                    for(int i=0; i<nColors; i++) group.VertexColors.Add(reader.ReadUInt32());
                    break;
            }
            reader.BaseStream.Position = chunkEnd;
        }

        if (group.Batches.Count == 0 && group.Indices.Count > 0)
        {
             // Fallback Rebuild if MOBA missing
             RebuildBatches(group);
        }

        data.Groups.Add(group);
    }

    private void RebuildBatches(WmoV14ToV17Converter.WmoGroupData group)
    {
        // Simple rebuild logic
        var facesByMat = new Dictionary<byte, List<ushort>>();
        for (int i = 0; i < group.FaceMaterials.Count; i++)
        {
             byte matId = group.FaceMaterials[i];
             if (matId == 255) continue;
             if (!facesByMat.ContainsKey(matId)) facesByMat[matId] = new List<ushort>();
             if (i*3+2 < group.Indices.Count)
             {
                 facesByMat[matId].Add(group.Indices[i*3]);
                 facesByMat[matId].Add(group.Indices[i*3+1]);
                 facesByMat[matId].Add(group.Indices[i*3+2]);
             }
        }
        group.Indices.Clear();
        group.FaceMaterials.Clear();
        group.Batches = new List<WmoV14ToV17Converter.WmoBatch>();
        
        uint currentFace = 0;
        foreach (var kvp in facesByMat.OrderBy(k => k.Key))
        {
             var batch = new WmoV14ToV17Converter.WmoBatch { MaterialId = kvp.Key, FirstFace = currentFace, NumFaces = (ushort)(kvp.Value.Count / 3) };
             ushort min=ushort.MaxValue, max=0;
             foreach(var idx in kvp.Value) {
                 if (idx < min) min = idx;
                 if (idx > max) max = idx;
                 group.Indices.Add(idx);
             }
             batch.FirstVertex = min;
             batch.LastVertex = max;
             // Add MOPY
             for(int i=0; i < batch.NumFaces; i++) group.FaceMaterials.Add(batch.MaterialId);
             
             // Box
             batch.BoundingBoxRaw = new byte[12]; // Dummy
             group.Batches.Add(batch);
             currentFace += batch.NumFaces;
        }
    }

    private void ResolveMaterialTextures(WmoV14ToV17Converter.WmoV14Data data)
    {
        if (data.MotxRaw.Length == 0) return;
        for (int i = 0; i < data.Materials.Count; i++)
        {
            var mat = data.Materials[i];
            mat.Texture1Name = GetStringFromOffset(data.MotxRaw, mat.Texture1Offset);
            mat.Texture2Name = GetStringFromOffset(data.MotxRaw, mat.Texture2Offset);
            mat.Texture3Name = GetStringFromOffset(data.MotxRaw, mat.Texture3Offset);
            data.Materials[i] = mat;
        }
    }

    private void RecalculateBounds(WmoV14ToV17Converter.WmoV14Data data)
    {
        var globalMin = new Vector3(float.MaxValue);
        var globalMax = new Vector3(float.MinValue);
        for (int i = 0; i < data.Groups.Count; i++)
        {
            var g = data.Groups[i];
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            if (g.Vertices != null && g.Vertices.Count > 0)
            {
                foreach(var v in g.Vertices) { min = Vector3.Min(min, v); max = Vector3.Max(max, v); }
            }
            else { min = g.BoundsMin; max = g.BoundsMax; }
            g.BoundsMin = min; g.BoundsMax = max;
            if (i < data.GroupInfos.Count) { 
                var inf = data.GroupInfos[i]; inf.BoundsMin = min; inf.BoundsMax = max; data.GroupInfos[i] = inf;
            }
            globalMin = Vector3.Min(globalMin, min);
            globalMax = Vector3.Max(globalMax, max);
        }
        data.BoundsMin = globalMin;
        data.BoundsMax = globalMax;
    }

    private List<string> ParseStringTable(byte[] data)
    {
        var result = new List<string>();
        int start = 0;
        for (int i = 0; i < data.Length; i++) {
            if (data[i] == 0) {
                if (i > start) result.Add(Encoding.UTF8.GetString(data, start, i - start));
                start = i + 1;
            }
        }
        return result;
    }
    
    private string GetStringFromOffset(byte[] table, uint offset)
    {
        if (offset >= table.Length) return string.Empty;
        int end = (int)offset;
        while (end < table.Length && table[end] != 0) end++;
        return end > offset ? Encoding.UTF8.GetString(table, (int)offset, end - (int)offset) : string.Empty;
    }

    // --- Writing Logic (Same as before) ---
    private void WriteRootFile(WmoV14ToV17Converter.WmoV14Data data, string outputPath)
    {
        var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
        using var stream = File.Create(outputPath);
        using var writer = new BinaryWriter(stream);

        WriteChunk(writer, "MVER", w => w.Write((uint)17));

        WriteChunk(writer, "MOHD", w =>
        {
            w.Write(data.MaterialCount);
            w.Write((uint)data.Groups.Count);
            w.Write((uint)data.Portals.Count);
            w.Write((uint)data.Lights.Count);
            w.Write((uint)data.DoodadDefs.Count);
            w.Write((uint)data.DoodadDefs.Count);
            w.Write((uint)Math.Max(1, data.DoodadSets.Count));
            w.Write(data.AmbientColor);
            w.Write(data.WmoId);
            WriteVector3(w, data.BoundsMin);
            WriteVector3(w, data.BoundsMax);
            w.Write((ushort)data.Flags);
            w.Write((ushort)0);
        });

        // Write MOTX
        var newOffsets = new Dictionary<string, uint>();
        var motxBuilder = new List<byte>();
        foreach (var mat in data.Materials)
        {
            if (!newOffsets.ContainsKey(mat.Texture1Name)) AddTextureToMotx(mat.Texture1Name, motxBuilder, newOffsets);
            if (!string.IsNullOrEmpty(mat.Texture2Name) && !newOffsets.ContainsKey(mat.Texture2Name)) AddTextureToMotx(mat.Texture2Name, motxBuilder, newOffsets);
            if (!string.IsNullOrEmpty(mat.Texture3Name) && !newOffsets.ContainsKey(mat.Texture3Name)) AddTextureToMotx(mat.Texture3Name, motxBuilder, newOffsets);
        }
        WriteChunk(writer, "MOTX", w => w.Write(motxBuilder.ToArray()));

        // MOMT
        WriteChunk(writer, "MOMT", w =>
        {
            foreach (var mat in data.Materials)
            {
                w.Write(mat.Flags);
                w.Write(mat.Shader);
                w.Write(mat.BlendMode);
                w.Write(newOffsets.ContainsKey(mat.Texture1Name) ? newOffsets[mat.Texture1Name] : 0);
                w.Write(mat.EmissiveColor);
                w.Write(mat.FrameEmissiveColor);
                w.Write(!string.IsNullOrEmpty(mat.Texture2Name) ? newOffsets[mat.Texture2Name] : 0);
                w.Write(mat.DiffuseColor);
                w.Write(mat.GroundType);
                w.Write(!string.IsNullOrEmpty(mat.Texture3Name) ? newOffsets[mat.Texture3Name] : 0);
                w.Write(mat.Color2);
                w.Write(0); // RuntimeData
                w.Write(0);
                w.Write(0);
                w.Write(0); 
            }
        });

        // MOGN
        var groupNameOffsets = new List<int>();
        var mognBuilder = new List<byte>();
        foreach (var g in data.GroupInfos) 
        {
             groupNameOffsets.Add(mognBuilder.Count);
             var nameBytes = Encoding.UTF8.GetBytes("Group" + groupNameOffsets.Count);
             mognBuilder.AddRange(nameBytes);
             mognBuilder.Add(0);
        }
        WriteChunk(writer, "MOGN", w => w.Write(mognBuilder.ToArray()));

        // MOGI
        WriteChunk(writer, "MOGI", w =>
        {
            for(int i=0; i<data.GroupInfos.Count; i++)
            {
                var info = data.GroupInfos[i];
                w.Write(info.Flags);
                WriteVector3(w, info.BoundsMin);
                WriteVector3(w, info.BoundsMax);
                w.Write(groupNameOffsets.Count > i ? groupNameOffsets[i] : -1);
            }
        });

        // Write MODS, MODN, MODD
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
                    w.Write((uint)0);
                }
            }
            else
            {
                var setName = Encoding.UTF8.GetBytes("Set_$DefaultGlobal");
                w.Write(setName);
                w.Write(new byte[20 - setName.Length]);
                w.Write((uint)0); w.Write((uint)0); w.Write((uint)0);
            }
        });

        WriteChunk(writer, "MODN", w =>
        {
             if (data.DoodadNamesRaw.Length > 0)
             {
                 var str = Encoding.UTF8.GetString(data.DoodadNamesRaw).Replace(".mdx", ".m2").Replace(".MDX", ".m2");
                 w.Write(Encoding.UTF8.GetBytes(str));
             }
        });

        WriteChunk(writer, "MODD", w =>
        {
            foreach (var dd in data.DoodadDefs)
            {
                w.Write(dd.NameIndex);
                WriteVector3(w, dd.Position);
                w.Write(dd.Orientation.X); w.Write(dd.Orientation.Y); w.Write(dd.Orientation.Z); w.Write(dd.Orientation.W);
                w.Write(dd.Scale);
                w.Write(dd.Color);
            }
        });
    }

    private void WriteGroupFiles(WmoV14ToV17Converter.WmoV14Data data, string outputPath)
    {
        string baseName = Path.GetFileNameWithoutExtension(outputPath);
        string outputDir = Path.GetDirectoryName(outputPath) ?? ".";

        for (int i = 0; i < data.Groups.Count; i++)
        {
            var group = data.Groups[i];
            string groupFileName = $"{baseName}_{i:D3}.wmo";
            string groupPath = Path.Combine(outputDir, groupFileName);

            using var stream = File.Create(groupPath);
            using var writer = new BinaryWriter(stream);

            WriteChunk(writer, "MVER", w => w.Write((uint)17));

            // MOGP
            var mogpStart = writer.BaseStream.Position;
            writer.Write(Encoding.ASCII.GetBytes("PGOM")); 
            var sizePos = writer.BaseStream.Position;
            writer.Write((uint)0); 
            var mogpDataStart = writer.BaseStream.Position;

            writer.Write(group.NameOffset);
            writer.Write(group.DescriptiveNameOffset);
            
            uint fixedFlags = group.Flags;
            bool isExterior = (fixedFlags & 0x8) != 0;
            if (!isExterior) fixedFlags |= 0x2000;

            writer.Write(fixedFlags);
            WriteVector3(writer, group.BoundsMin);
            WriteVector3(writer, group.BoundsMax);
            
            writer.Write(group.PortalStart);
            writer.Write(group.PortalCount);
            writer.Write(group.TransBatchCount);
            writer.Write(group.IntBatchCount);
            writer.Write(group.ExtBatchCount);
            writer.Write((ushort)0); 

            WriteSubChunk(writer, "MOPY", w =>
            {
                for(int f=0; f < group.FaceMaterials.Count; f++) 
                {
                     w.Write((byte)0); 
                     w.Write(group.FaceMaterials[f]); 
                }
            });

            WriteSubChunk(writer, "MOVI", w => { foreach (var idx in group.Indices) w.Write(idx); });

            WriteSubChunk(writer, "MOVT", w => { foreach (var v in group.Vertices) WriteVector3(w, v); });

            WriteSubChunk(writer, "MONR", w => 
            {
                var norms = GenerateNormals(group);
                foreach(var n in norms) WriteVector3(w, n);
            });

            WriteSubChunk(writer, "MOTV", w => { foreach (var uv in group.UVs) { w.Write(uv.X); w.Write(uv.Y); } });

            WriteSubChunk(writer, "MOBA", w =>
            {
                foreach (var b in group.Batches)
                {
                   if (b.BoundingBoxRaw != null && b.BoundingBoxRaw.Length == 12) w.Write(b.BoundingBoxRaw);
                   else w.Write(new byte[12]);
                   
                   w.Write((uint)(b.FirstFace * 3)); 
                   w.Write((ushort)(b.NumFaces * 3)); 
                   w.Write(b.FirstVertex);
                   w.Write(b.LastVertex);
                   w.Write(b.Flags);
                   w.Write(b.MaterialId);
                }
            });

            if ((fixedFlags & 0x800) != 0 && group.DoodadRefs != null)
            {
                WriteSubChunk(writer, "MODR", w => { foreach(var d in group.DoodadRefs) w.Write(d); });
            }

            // MOBN / MOBR (BSP) - conditional on 0x1! (Fixed for Extended)
            if ((fixedFlags & 0x1) != 0)
            {
                WriteSubChunk(writer, "MOBN", w =>
                {
                    w.Write((ushort)4);      // Flags (0x4 = Leaf)
                    w.Write((short)-1);      // NegChild
                    w.Write((short)-1);      // PosChild
                    w.Write((ushort)(group.Indices.Count / 3)); 
                    w.Write((uint)0);        
                    w.Write(0f);             
                });
                
                WriteSubChunk(writer, "MOBR", w =>
                {
                    for (ushort f = 0; f < group.Indices.Count / 3; f++) w.Write(f);
                });
            }

            if ((fixedFlags & 0x4) != 0 && group.VertexColors != null)
            {
                 WriteSubChunk(writer, "MOCV", w => { foreach(var c in group.VertexColors) w.Write(c); });
            }

            var endMogp = writer.BaseStream.Position;
            writer.BaseStream.Position = sizePos;
            writer.Write((uint)(endMogp - mogpDataStart));
            writer.BaseStream.Position = endMogp;
        }
    }

    private void WriteChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData)
    {
        var reversed = new string(chunkId.Reverse().ToArray());
        writer.Write(Encoding.ASCII.GetBytes(reversed));
        var sizePos = writer.BaseStream.Position;
        writer.Write((uint)0);
        var start = writer.BaseStream.Position;
        writeData(writer);
        var end = writer.BaseStream.Position;
        writer.BaseStream.Position = sizePos;
        writer.Write((uint)(end - start));
        writer.BaseStream.Position = end;
    }
    
    private void WriteSubChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData) => WriteChunk(writer, chunkId, writeData);
    
    private void WriteVector3(BinaryWriter w, Vector3 v) { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); }

    private void AddTextureToMotx(string texturePath, List<byte> builder, Dictionary<string, uint> offsetMap)
    {
        uint offset = (uint)builder.Count;
        offsetMap[texturePath] = offset;
        var bytes = Encoding.UTF8.GetBytes(texturePath.Replace('/', '\\'));
        builder.AddRange(bytes);
        builder.Add(0);
    }
    
     private List<Vector3> GenerateNormals(WmoV14ToV17Converter.WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i < normals.Length; i++) normals[i] = Vector3.Zero;

        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            var i0 = group.Indices[i];
            var i1 = group.Indices[i + 1];
            var i2 = group.Indices[i + 2];
            if (i0>=normals.Length||i1>=normals.Length||i2>=normals.Length) continue;
            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var n = Vector3.Normalize(Vector3.Cross(e1, e2));
            normals[i0]+=n; normals[i1]+=n; normals[i2]+=n;
        }
        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitY).ToList();
    }
}
