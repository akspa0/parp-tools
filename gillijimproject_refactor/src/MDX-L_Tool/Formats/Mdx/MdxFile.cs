using System.Text;

namespace MdxLTool.Formats.Mdx;

/// <summary>
/// MDX file reader/writer.
/// Based on Ghidra analysis of MDLFileRead (0x0078b660) and related functions.
/// </summary>
public class MdxFile
{
    public uint Version { get; set; }
    public MdlModel Model { get; set; } = new();
    public string ModelName => Model.Name;
    public List<MdlSequence> Sequences { get; } = new();
    public List<uint> GlobalSequences { get; } = new();
    public List<MdlMaterial> Materials { get; } = new();
    public List<MdlTexture> Textures { get; } = new();
    public List<MdlGeoset> Geosets { get; } = new();
    public List<MdlBone> Bones { get; } = new();
    public List<C3Vector> PivotPoints { get; } = new();
    public List<MdlAttachment> Attachments { get; } = new();
    public List<MdlCamera> Cameras { get; } = new();

    /// <summary>Load MDX binary file</summary>
    public static MdxFile Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        return Load(br);
    }

    /// <summary>Load MDX from BinaryReader</summary>
    public static MdxFile Load(BinaryReader br)
    {
        var mdx = new MdxFile();

        // Verify magic
        uint magic = br.ReadUInt32();
        if (magic != MdxHeaders.MAGIC)
            throw new InvalidDataException($"Invalid MDX magic: 0x{magic:X8} (expected 0x{MdxHeaders.MAGIC:X8})");

        // Read chunks until EOF
        while (br.BaseStream.Position < br.BaseStream.Length)
        {
            if (br.BaseStream.Length - br.BaseStream.Position < 8) break;

            string tag = ReadTag(br);
            uint size = br.ReadUInt32();
            long chunkEnd = br.BaseStream.Position + size;

            Console.WriteLine($"  Found chunk: {tag} ({size} bytes)");
            switch (tag)
            {
                case MdxHeaders.VERS:
                    mdx.Version = br.ReadUInt32();
                    break;

                case MdxHeaders.MODL:
                    mdx.Model = ReadModl(br, size);
                    break;

                case MdxHeaders.SEQS:
                    ReadSeqs(br, size, mdx.Sequences);
                    break;

                case MdxHeaders.GLBS:
                    ReadGlbs(br, size, mdx.GlobalSequences);
                    break;

                case MdxHeaders.MTLS:
                    ReadMtls(br, size, mdx.Materials);
                    break;

                case MdxHeaders.TEXS:
                    ReadTexs(br, size, mdx.Textures);
                    break;

                case MdxHeaders.GEOS:
                    ReadGeos(br, size, mdx.Geosets);
                    break;

                case MdxHeaders.BONE:
                    ReadBone(br, size, mdx.Bones);
                    break;

                case MdxHeaders.PIVT:
                    ReadPivt(br, size, mdx.PivotPoints);
                    break;

                case MdxHeaders.ATCH:
                    ReadAtch(br, size, mdx.Attachments);
                    break;

                case MdxHeaders.CAMS:
                    ReadCams(br, size, mdx.Cameras);
                    break;

                default:
                    // Unknown chunk - skip it safely
                    break;
            }

            // Ensure we're at chunk end to prevent issues with partial reads or unknown tags
            if (br.BaseStream.Position > chunkEnd)
            {
                // This shouldn't happen with correct logic but good for debugging
            }
            br.BaseStream.Position = chunkEnd;
        }

        // Link pivots to bones
        for (int i = 0; i < mdx.Bones.Count && i < mdx.PivotPoints.Count; i++)
        {
            mdx.Bones[i].Pivot = mdx.PivotPoints[i];
        }

        return mdx;
    }

    static string ReadTag(BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(4);
        return Encoding.ASCII.GetString(bytes);
    }

    static string ReadFixedString(BinaryReader br, int length)
    {
        byte[] bytes = br.ReadBytes(length);
        int end = Array.IndexOf(bytes, (byte)0);
        if (end < 0) end = length;
        return Encoding.ASCII.GetString(bytes, 0, end);
    }

    /// <summary>
    /// ReadModl - based on Ghidra decompilation of ReadBinModelGlobals @ 0x007b2800
    /// Structure: Name(0x50) + AnimFile(0x104) + Radius + MinBounds + MaxBounds + Flags(byte) + BlendTime
    /// Total size must be exactly 0x175 (373) bytes.
    /// </summary>
    static MdlModel ReadModl(BinaryReader br, uint size)
    {
        var model = new MdlModel();
        
        // Name: 0x50 (80) bytes fixed string
        model.Name = ReadFixedString(br, 0x50);
        
        // AnimationFile: 0x104 (260) bytes fixed string  
        model.AnimationFile = ReadFixedString(br, 0x104);
        
        // Radius (float) - comes BEFORE bounds in Alpha format!
        float radius = br.ReadSingle();
        
        // MinBounds (float[3])
        var min = ReadC3Vector(br);
        
        // MaxBounds (float[3])
        var max = ReadC3Vector(br);
        
        model.Bounds = new CMdlBounds
        {
            Extent = new CAaBox { Min = min, Max = max },
            Radius = radius
        };
        
        // Flags: 1 byte
        model.Flags = br.ReadByte();
        
        // BlendTime: uint32
        model.BlendTime = br.ReadUInt32();
        
        return model;
    }

    static CMdlBounds ReadBounds(BinaryReader br)
    {
        return new CMdlBounds
        {
            Extent = new CAaBox
            {
                Min = ReadC3Vector(br),
                Max = ReadC3Vector(br)
            },
            Radius = br.ReadSingle()
        };
    }

    static C3Vector ReadC3Vector(BinaryReader br)
    {
        return new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
    }

    static C2Vector ReadC2Vector(BinaryReader br)
    {
        return new C2Vector(br.ReadSingle(), br.ReadSingle());
    }

    /// <summary>
    /// ReadSeqs - based on Ghidra decompilation of ReadBinSequences
    /// Chunk format: count(uint32) + count * 0x8c sequence records
    /// Record: Name(0x50) + TimeStart + TimeEnd + MoveSpeed + Flags + MinBounds + MaxBounds + Frequency + Radius + Replay.Start + Replay.End + BlendTime
    /// </summary>
    static void ReadSeqs(BinaryReader br, uint size, List<MdlSequence> sequences)
    {
        uint count = br.ReadUInt32();
        for (uint i = 0; i < count; i++)
        {
            var seq = new MdlSequence();
            seq.Name = ReadFixedString(br, 0x50);                           // 0x00-0x4F
            seq.Time = new CiRange { Start = br.ReadInt32(), End = br.ReadInt32() }; // 0x50-0x57
            seq.MoveSpeed = br.ReadSingle();                                // 0x58
            seq.Flags = br.ReadUInt32();                                    // 0x5C
            var min = ReadC3Vector(br);                                     // 0x60-0x6B
            var max = ReadC3Vector(br);                                     // 0x6C-0x77
            seq.Frequency = br.ReadSingle();                                // 0x78 (Rarity in MDL)
            float radius = br.ReadSingle();                                 // 0x7C
            seq.Bounds = new CMdlBounds
            {
                Extent = new CAaBox { Min = min, Max = max },
                Radius = radius
            };
            seq.Replay = new CiRange { Start = br.ReadInt32(), End = br.ReadInt32() }; // 0x80-0x87
            seq.BlendTime = br.ReadUInt32();                                // 0x88-0x8B
            sequences.Add(seq);
        }
    }

    static void ReadGlbs(BinaryReader br, uint size, List<uint> globals)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
            globals.Add(br.ReadUInt32());
    }

    static void ReadMtls(BinaryReader br, uint size, List<MdlMaterial> materials)
    {
        if (size < 8) return;
        uint count = br.ReadUInt32();
        br.ReadUInt32(); // Padding/Unknown
        
        for (uint i = 0; i < count; i++)
        {
            uint matSize = br.ReadUInt32();
            long matEnd = br.BaseStream.Position - 4 + matSize;
            
            var mat = new MdlMaterial();
            mat.PriorityPlane = br.ReadInt32();
            uint layerCount = br.ReadUInt32();
            
            for (uint j = 0; j < layerCount; j++)
            {
                uint layerSize = br.ReadUInt32();
                long layerEnd = br.BaseStream.Position - 4 + layerSize;
                
                var layer = new MdlTexLayer();
                layer.BlendMode = (MdlTexOp)br.ReadUInt32();
                layer.Flags = (MdlGeoFlags)br.ReadUInt32();
                layer.TextureId = br.ReadInt32();
                layer.TransformId = br.ReadInt32();
                layer.CoordId = br.ReadInt32();
                layer.StaticAlpha = br.ReadSingle();
                
                // Skip animation tracks in layer
                br.BaseStream.Position = layerEnd;
                mat.Layers.Add(layer);
            }
            
            br.BaseStream.Position = matEnd;
            materials.Add(mat);
        }
    }

    static void ReadTexs(BinaryReader br, uint size, List<MdlTexture> textures)
    {
        uint entrySize = 268; // 0x10C
        uint count = size / entrySize;
        for (uint i = 0; i < count; i++)
        {
            var tex = new MdlTexture();
            tex.ReplaceableId = br.ReadUInt32();
            tex.Path = ReadFixedString(br, 0x104);
            tex.Flags = br.ReadUInt32();
            textures.Add(tex);
        }
    }

    static void ReadGeos(BinaryReader br, uint size, List<MdlGeoset> geosets)
    {
        uint count = br.ReadUInt32();
        for (uint i = 0; i < count; i++)
        {
            uint geoSize = br.ReadUInt32();
            long geoEnd = br.BaseStream.Position - 4 + geoSize;

            var geo = new MdlGeoset();

            // Read geoset sub-chunks
            while (br.BaseStream.Position < geoEnd)
            {
                if (geoEnd - br.BaseStream.Position < 8) break;

                string subTag = ReadTag(br);
                int subCount = br.ReadInt32();
                
                switch (subTag)
                {
                    case MdxHeaders.VRTX:
                        for (int k = 0; k < subCount; k++)
                            geo.Vertices.Add(ReadC3Vector(br));
                        break;

                    case MdxHeaders.NRMS:
                        for (int k = 0; k < subCount; k++)
                            geo.Normals.Add(ReadC3Vector(br));
                        break;

                    case MdxHeaders.PTYP:
                        br.BaseStream.Position += subCount * 1; // Primitive type is 1 byte
                        break;

                    case MdxHeaders.PCNT:
                        br.BaseStream.Position += subCount * 4; // Primitive count is 4 bytes
                        break;

                    case MdxHeaders.PVTX:
                        for (int k = 0; k < subCount; k++)
                            geo.Indices.Add(br.ReadUInt16());
                        break;

                    case MdxHeaders.GNDX:
                        for (int k = 0; k < subCount; k++)
                            geo.VertexGroups.Add(br.ReadByte());
                        break;

                    case MdxHeaders.MTGC:
                        for (int k = 0; k < subCount; k++)
                            geo.MatrixGroups.Add(br.ReadUInt32());
                        break;

                    case MdxHeaders.MATS:
                        for (int k = 0; k < subCount; k++)
                            geo.MatrixIndices.Add(br.ReadUInt32());
                        break;

                    case MdxHeaders.UVAS:
                        {
                            // In 0.5.3, UVAS contains the UV data directly for N maps.
                            // subCount here is the number of texture coordinate maps.
                            int numMaps = subCount;
                            // For 0.5.3, we expect vertex count to match VRTX count.
                            int vertexCount = geo.Vertices.Count;
                            for (int m = 0; m < numMaps; m++)
                            {
                                for (int k = 0; k < vertexCount; k++)
                                {
                                    if (m == 0) // Only take the first map for now
                                        geo.TexCoords.Add(ReadC2Vector(br));
                                    else
                                        br.BaseStream.Position += 8;
                                }
                            }
                        }
                        break;

                    case MdxHeaders.UVBS:
                        for (int k = 0; k < subCount; k++)
                            geo.TexCoords.Add(ReadC2Vector(br));
                        break;

                    case MdxHeaders.BIDX:
                    case MdxHeaders.BWGT:
                        br.BaseStream.Position += (long)subCount * 4; // 4 indices/weights per vertex
                        break;

                    case MdxHeaders.ATSQ:
                        br.BaseStream.Position += (long)subCount * 4; // Found in 0.5.3 geosets
                        break;

                    default:
                        // Log unknown tag for debugging without hanging
                        Console.WriteLine($"      Skipping unknown GEOS sub-tag: {subTag} ({subCount} elements)");
                        int elementSize = GuessElementSize(subTag);
                        if (elementSize > 0)
                            br.BaseStream.Position += (long)subCount * elementSize;
                        else
                        {
                            br.BaseStream.Position = geoEnd;
                            goto NextGeoset;
                        }
                        break;
                }
            }

            // Read trailing geoset data (MaterialID, SelectionGroup, Flags, Bounds)
            if (br.BaseStream.Position + 0x1C <= geoEnd)
            {
                geo.MaterialId = br.ReadInt32();
                geo.SelectionGroup = br.ReadUInt32();
                geo.Flags = br.ReadUInt32();
                geo.Bounds = ReadBounds(br);
            }

        NextGeoset:
            geosets.Add(geo);
            br.BaseStream.Position = geoEnd;
        }
    }

    static void ReadBone(BinaryReader br, uint size, List<MdlBone> bones)
    {
        uint count = br.ReadUInt32();
        for (uint i = 0; i < count; i++)
        {
            var bone = new MdlBone();
            ReadNode(br, bone);

            // Optional geoset info after node
            if (br.BaseStream.Position + 8 <= br.BaseStream.Length)
            {
                bone.GeosetId = br.ReadInt32();
                bone.GeosetAnimId = br.ReadInt32();
            }
            bones.Add(bone);
        }
    }

    static void ReadNode(BinaryReader br, MdlBone node)
    {
        uint nodeSize = br.ReadUInt32();
        long nodeEnd = br.BaseStream.Position - 4 + nodeSize;

        node.Name = ReadFixedString(br, 0x50);
        node.ObjectId = br.ReadInt32();
        node.ParentId = br.ReadInt32();
        node.Flags = br.ReadUInt32();

        // Read animation tracks (KGTR, KGRT, KGSC)
        while (br.BaseStream.Position < nodeEnd)
        {
            if (nodeEnd - br.BaseStream.Position < 8) break;

            string tag = ReadTag(br);
            uint keyCount = br.ReadUInt32();
            uint interp = br.ReadUInt32();   // 0=None, 1=Linear, 2=Hermite, 3=Bezier
            uint globalSeqId = br.ReadUInt32();

            int keySize = (interp <= 1) ? 16 : 40; // Hermite/Bezier have tangents
            br.BaseStream.Position += keyCount * (uint)keySize;
        }

        br.BaseStream.Position = nodeEnd;
    }

    static void ReadPivt(BinaryReader br, uint size, List<C3Vector> pivots)
    {
        int count = (int)(size / 12);
        for (int i = 0; i < count; i++)
            pivots.Add(ReadC3Vector(br));
    }

    static void ReadAtch(BinaryReader br, uint size, List<MdlAttachment> attachments)
    {
        uint count = br.ReadUInt32();
        br.ReadUInt32(); // Unknown/padding

        for (uint i = 0; i < count; i++)
        {
            uint nodeSize = br.ReadUInt32();
            long nodeEnd = br.BaseStream.Position - 4 + nodeSize;

            var atch = new MdlAttachment();
            atch.Name = ReadFixedString(br, 0x50);
            atch.ObjectId = br.ReadInt32();
            atch.ParentId = br.ReadInt32();
            uint flags = br.ReadUInt32(); 

            // Skip tracks until we reach the part of the struct after the node
            br.BaseStream.Position = nodeEnd;

            // In 0.5.3, Path and AttachmentId might follow GenObject within a larger struct
            // but usually they are separate or following immediately.
            // Let's check remaining space in the chunk if possible, but ATCH is a list.
            // Based on Ghidra, ReadBinAttachment reads Path(0x104) and ID(4).
            
            if (br.BaseStream.Position + 0x108 <= br.BaseStream.Length)
            {
                 // atch.Path = ReadFixedString(br, 0x104);
                 // atch.AttachmentId = br.ReadUInt32();
            }

            attachments.Add(atch);
        }
    }

    static void ReadCams(BinaryReader br, uint size, List<MdlCamera> cameras)
    {
        uint count = br.ReadUInt32();
        br.ReadUInt32(); // Unknown/padding

        for (uint i = 0; i < count; i++)
        {
            uint camSize = br.ReadUInt32();
            long camEnd = br.BaseStream.Position - 4 + camSize;

            var cam = new MdlCamera();
            cam.Name = ReadFixedString(br, 0x50);
            cam.Position = ReadC3Vector(br);
            cam.FieldOfView = br.ReadSingle();
            cam.FarClip = br.ReadSingle();
            cam.NearClip = br.ReadSingle();
            
            cam.Target.Name = ReadFixedString(br, 0x50);
            cam.Target.Position = ReadC3Vector(br);

            // Skip tracks until camEnd
            br.BaseStream.Position = camEnd;
            cameras.Add(cam);
        }
    }

    static void ReadLite(BinaryReader br, uint size, List<MdlLight> lights)
    {
        uint count = br.ReadUInt32();
        for (uint i = 0; i < count; i++)
        {
            var light = new MdlLight();
            uint nodeSize = br.ReadUInt32();
            long nodeEnd = br.BaseStream.Position - 4 + nodeSize;

            ReadNode(br, light);

            if (br.BaseStream.Position + 0x20 <= nodeEnd)
            {
                light.Type = br.ReadInt32();
                light.AttenuationStart = br.ReadSingle();
                light.AttenuationEnd = br.ReadSingle();
                light.Color = ReadC3Vector(br);
                light.Intensity = br.ReadSingle();
                light.AmbientColor = ReadC3Vector(br);
                light.AmbientIntensity = br.ReadSingle();
            }

            br.BaseStream.Position = nodeEnd;
            lights.Add(light);
        }
    }

    static void ReadHelp(BinaryReader br, uint size, List<MdlBone> helpers)
    {
        uint count = br.ReadUInt32();
        for (uint i = 0; i < count; i++)
        {
            var helper = new MdlBone();
            uint nodeSize = br.ReadUInt32();
            long nodeEnd = br.BaseStream.Position - 4 + nodeSize;
            
            ReadNode(br, helper);
            br.BaseStream.Position = nodeEnd;
            helpers.Add(helper);
        }
    }

    static int GuessElementSize(string tag) => tag switch
    {
        MdxHeaders.VRTX => 12,
        MdxHeaders.NRMS => 12,
        MdxHeaders.UVBS => 8,
        MdxHeaders.PTYP => 1, // Primitive Type is 1 byte in 0.5.3
        MdxHeaders.PCNT => 4, // Primitive Count is 4 bytes in 0.5.3
        MdxHeaders.PVTX => 2,
        MdxHeaders.GNDX => 1,
        MdxHeaders.MTGC => 4,
        MdxHeaders.MATS => 4,
        "BIDX" => 4, // 4 indices per vertex
        "BWGT" => 4, // 4 weights per vertex
        "ATSQ" => 4,
        _ => 0
    };

    /// <summary>Save as MDL text format</summary>
    public void SaveMdl(string path)
    {
        using var sw = new StreamWriter(path);
        var writer = new MdlWriter();
        writer.Write(this, sw);
    }
}
