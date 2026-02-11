using System.Text;
using MdxLTool.Formats.Obj;

namespace MdxLTool.Formats.Mdx;

/// <summary>
/// MDX file reader/writer.
/// Based on Ghidra analysis of MDLFileRead (0x0078b660) and related functions.
/// </summary>
public class MdxFile
{
    public uint Version { get; set; }
    public MdlModel Model { get; set; } = new();
    public byte[]? RawData { get; set; }
    public string ModelName 
    { 
        get => Model.Name;
        set => Model.Name = value;
    }
    public List<MdlSequence> Sequences { get; } = new();
    public List<uint> GlobalSequences { get; } = new();
    public List<MdlMaterial> Materials { get; } = new();
    public List<MdlTexture> Textures { get; } = new();
    public List<MdlGeoset> Geosets { get; } = new();
    public List<MdlBone> Bones { get; } = new();
    public List<C3Vector> PivotPoints { get; } = new();
    public List<MdlAttachment> Attachments { get; } = new();
    public List<MdlCamera> Cameras { get; } = new();
    public List<MdlGeosetAnimation> GeosetAnimations { get; } = new();
    public List<MdlParticleEmitter2> ParticleEmitters2 { get; } = new();
    public List<MdlRibbonEmitter> RibbonEmitters { get; } = new();

    /// <summary>Load MDX binary file</summary>
    public static MdxFile Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        return Load(br);
    }

    public static MdxFile Load(Stream stream)
    {
        using var br = new BinaryReader(stream, Encoding.Default, leaveOpen: true);
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

            // Console.WriteLine($"  Found chunk: {tag} ({size} bytes)");
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
                    ReadGeosets(br, size, mdx.Geosets, mdx.Version);
                    break;

                case MdxHeaders.ATSQ:
                    ReadAtsq(br, size, mdx.GeosetAnimations);
                    break;

                case "PRE2":
                    ReadPre2(br, size, mdx.ParticleEmitters2, mdx.PivotPoints);
                    break;

                case "RIBB":
                    ReadRibb(br, size, mdx.RibbonEmitters, mdx.PivotPoints);
                    break;

                case MdxHeaders.BONE:
                case MdxHeaders.LITE:
                case MdxHeaders.HELP:
                case MdxHeaders.ATCH:
                case MdxHeaders.PIVT:
                case MdxHeaders.CAMS:
                case MdxHeaders.EVTS:
                case MdxHeaders.CLID:
                case "PREM":
                    // Known chunks we don't fully parse geometry for yet
                    br.BaseStream.Position = chunkEnd;
                    break;

                default:
                    br.BaseStream.Position = chunkEnd;
                    break;
            }

            // Ensure we are at the end of the chunk
            if (br.BaseStream.Position != chunkEnd)
            {
                br.BaseStream.Position = chunkEnd;
            }
        }

        return mdx;
    }

    public void SaveMdl(string path)
    {
        var writer = new MdlWriter();
        using var sw = new StreamWriter(path);
        writer.Write(this, sw);
    }

    public void SaveObj(string path, bool split = false, Dictionary<int, string>? exportedTextures = null)
    {
        var writer = new ObjWriter();
        if (exportedTextures != null)
            writer.ExportedTextures = exportedTextures;
        writer.Write(this, path, split);
    }

    // --- Chunk Readers ---

    static string ReadTag(BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(4);
        return Encoding.ASCII.GetString(bytes);
    }

    static string ReadFixedString(BinaryReader br, int length)
    {
        byte[] bytes = br.ReadBytes(length);
        int nullIdx = Array.IndexOf(bytes, (byte)0);
        if (nullIdx >= 0)
            return Encoding.ASCII.GetString(bytes, 0, nullIdx);
        return Encoding.ASCII.GetString(bytes);
    }

    static uint ReadVers(BinaryReader br, uint size)
    {
        uint version = br.ReadUInt32();
        Console.WriteLine($"MDX Version: {version}");
        return version;
    }

    static MdlModel ReadModl(BinaryReader br, uint size)
    {
        var model = new MdlModel();
        model.Name = ReadFixedString(br, 0x50);
        
        var bounds = new CMdlBounds();
        bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        model.Bounds = bounds;
        
        model.BlendTime = br.ReadUInt32();
        return model;
    }

    static void ReadSeqs(BinaryReader br, uint size, List<MdlSequence> sequences)
    {
        uint count = size / 132;
        for (uint i = 0; i < count; i++)
        {
            var seq = new MdlSequence();
            seq.Name = ReadFixedString(br, 0x50);
            
            var time = new CiRange();
            time.Start = br.ReadInt32();
            time.End = br.ReadInt32();
            seq.Time = time;
            
            seq.MoveSpeed = br.ReadSingle();
            seq.Flags = br.ReadUInt32();
            
            seq.Frequency = br.ReadSingle();
            var replay = new CiRange();
            replay.Start = br.ReadInt32();
            replay.End = br.ReadInt32();
            seq.Replay = replay;
            
            // Extent
            br.ReadBytes(28); 
            sequences.Add(seq);
        }
    }

    static void ReadGlbs(BinaryReader br, uint size, List<uint> globals)
    {
        uint count = size / 4;
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

    static void ReadGeosets(BinaryReader br, uint size, List<MdlGeoset> geosets, uint version)
    {
        long end = br.BaseStream.Position + size;
        
        // Alpha 0.5.3 seems to have a uint32 count here
        if (size >= 4)
        {
            uint possibleCount = br.ReadUInt32();
            if (possibleCount > 100) // Likely a size, not a count
            {
                br.BaseStream.Position -= 4;
            }
            else
            {
            }
        }

        while (br.BaseStream.Position < end)
        {
            if (end - br.BaseStream.Position < 4) break;

            uint geosetSize = br.ReadUInt32();
            long geosetEnd = br.BaseStream.Position - 4 + geosetSize;
            
            var geoset = ReadGeoset(br, geosetSize - 4, version);
            geosets.Add(geoset);
            
            br.BaseStream.Position = geosetEnd;
        }
    }

    static void ReadAtsq(BinaryReader br, uint size, List<MdlGeosetAnimation> animations)
    {
        long end = br.BaseStream.Position + size;
        
        while (br.BaseStream.Position < end)
        {
            if (end - br.BaseStream.Position < 8) break;
            
            uint animSize = br.ReadUInt32();
            long animEnd = br.BaseStream.Position - 4 + animSize;
            
            var anim = ReadAtsqEntry(br, animSize - 4);
            animations.Add(anim);
            
            br.BaseStream.Position = animEnd;
        }
    }

    static MdlGeosetAnimation ReadAtsqEntry(BinaryReader br, uint size)
    {
        var anim = new MdlGeosetAnimation();
        long animEnd = br.BaseStream.Position + size;
        
        // Read header
        anim.GeosetId = br.ReadUInt32();
        anim.DefaultAlpha = br.ReadSingle();
        anim.DefaultColor = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        anim.Unknown = br.ReadUInt32();
        
        // Read sub-chunks
        while (br.BaseStream.Position < animEnd - 8)
        {
            string tag = ReadTag(br);
            uint subSize = br.ReadUInt32();
            long subEnd = br.BaseStream.Position + subSize;
            
            switch (tag)
            {
                case "KGAO":
                    ReadAlphaKeys(br, anim);
                    break;
                case "KGAC":
                    ReadColorKeys(br, anim);
                    break;
                default:
                    br.BaseStream.Position = subEnd;
                    break;
            }
            
            if (br.BaseStream.Position != subEnd)
                br.BaseStream.Position = subEnd;
        }
        
        return anim;
    }

    static void ReadAlphaKeys(BinaryReader br, MdlGeosetAnimation anim)
    {
        uint keyCount = br.ReadUInt32();
        anim.AlphaInterpolation = (MdlAnimInterpolation)br.ReadUInt32();
        anim.AlphaGlobalSeqId = br.ReadInt32();
        
        for (uint i = 0; i < keyCount; i++)
        {
            var key = new MdlAnimKey<float>
            {
                Time = br.ReadInt32(),
                Value = br.ReadSingle()
            };
            
            if (anim.AlphaInterpolation >= MdlAnimInterpolation.Hermite)
            {
                key.TangentIn = br.ReadSingle();
                key.TangentOut = br.ReadSingle();
            }
            
            anim.AlphaKeys.Add(key);
        }
    }

    static void ReadColorKeys(BinaryReader br, MdlGeosetAnimation anim)
    {
        uint keyCount = br.ReadUInt32();
        anim.ColorInterpolation = (MdlAnimInterpolation)br.ReadUInt32();
        anim.ColorGlobalSeqId = br.ReadInt32();
        
        for (uint i = 0; i < keyCount; i++)
        {
            var key = new MdlAnimKey<C3Color>
            {
                Time = br.ReadInt32(),
                Value = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle())
            };
            
            if (anim.ColorInterpolation >= MdlAnimInterpolation.Hermite)
            {
                key.ColorTangentIn = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                key.ColorTangentOut = new C3Color(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            }
            
            anim.ColorKeys.Add(key);
        }
    }

    static int _geosetDebugIndex = 0;

    static MdlGeoset ReadGeoset(BinaryReader br, uint size, uint mdxVersion)
    {
        var geo = new MdlGeoset();
        long geoEnd = br.BaseStream.Position + size;
        int gIdx = _geosetDebugIndex++;
        var tagsFound = new List<string>();

        while (br.BaseStream.Position < geoEnd)
        {
            if (geoEnd - br.BaseStream.Position < 8) break;

            // Peek at next 4 bytes — if they don't form a valid geoset sub-chunk tag,
            // this is the non-tagged footer (MaterialId, bounds, etc.), not another chunk.
            long preTagPos = br.BaseStream.Position;
            string tag = ReadTag(br);
            if (!IsValidGeosetTag(tag))
            {
                br.BaseStream.Position = preTagPos; // Rewind — footer starts here
                break;
            }
            uint count = br.ReadUInt32();
            long tagPos = preTagPos;
            tagsFound.Add($"{tag}({count})");

            switch (tag)
            {
                case "VRTX":
                    for (int i = 0; i < count; i++)
                        geo.Vertices.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    break;
                case "NRMS":
                    for (int i = 0; i < count; i++)
                        geo.Normals.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    break;
                case "PTYP":
                {
                    // In Alpha 0.5.3, PTYP count is number of primitive groups.
                    // Each entry is a uint32 primitive type (4 = triangles).
                    // Peek ahead to validate: if reading count*4 bytes lands us on a valid tag, that's correct.
                    long afterRead4 = br.BaseStream.Position + count * 4;
                    long afterRead1 = br.BaseStream.Position + count; // 1 byte per entry?
                    
                    // Check what tag follows each size assumption
                    bool valid4 = false, valid1 = false;
                    if (afterRead4 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead4;
                        string nextTag4 = (afterRead4 + 4 <= geoEnd) ? Encoding.ASCII.GetString(br.ReadBytes(4)) : "";
                        valid4 = IsValidGeosetTag(nextTag4);
                        br.BaseStream.Position = save;
                    }
                    if (afterRead1 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead1;
                        string nextTag1 = (afterRead1 + 4 <= geoEnd) ? Encoding.ASCII.GetString(br.ReadBytes(4)) : "";
                        valid1 = IsValidGeosetTag(nextTag1);
                        br.BaseStream.Position = save;
                    }
                    
                    if (valid4)
                    {
                        br.ReadBytes((int)count * 4);
                    }
                    else if (valid1)
                    {
                        Console.WriteLine($"      [PTYP] Using 1-byte elements (count={count}) — next tag valid at +{count}");
                        br.ReadBytes((int)count);
                    }
                    else
                    {
                        // Fallback: try 4-byte
                        Console.WriteLine($"      [PTYP] Neither 1-byte nor 4-byte gives valid next tag. Defaulting to 4-byte (count={count})");
                        br.ReadBytes((int)count * 4);
                    }
                    break;
                }
                case "PCNT":
                {
                    // Same ambiguity — check if 4 bytes or some other size per element
                    long afterRead4 = br.BaseStream.Position + count * 4;
                    long afterRead1 = br.BaseStream.Position + count;
                    
                    bool valid4 = false, valid1 = false;
                    if (afterRead4 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead4;
                        string nextTag4 = Encoding.ASCII.GetString(br.ReadBytes(4));
                        valid4 = IsValidGeosetTag(nextTag4);
                        br.BaseStream.Position = save;
                    }
                    if (afterRead1 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead1;
                        string nextTag1 = Encoding.ASCII.GetString(br.ReadBytes(4));
                        valid1 = IsValidGeosetTag(nextTag1);
                        br.BaseStream.Position = save;
                    }
                    
                    if (valid4)
                    {
                        br.ReadBytes((int)count * 4);
                    }
                    else if (valid1)
                    {
                        Console.WriteLine($"      [PCNT] Using 1-byte elements (count={count}) — next tag valid at +{count}");
                        br.ReadBytes((int)count);
                    }
                    else
                    {
                        Console.WriteLine($"      [PCNT] Neither 1-byte nor 4-byte gives valid next tag. Defaulting to 4-byte (count={count})");
                        br.ReadBytes((int)count * 4);
                    }
                    break;
                }
                case "PVTX":
                    for (int i = 0; i < count; i++)
                        geo.Indices.Add(br.ReadUInt16());
                    break;
                case "GNDX":
                    br.ReadBytes((int)count); 
                    break;
                case "MTGC":
                    br.ReadBytes((int)count * 4); 
                    break;
                case "MATS":
                    // MATS = Matrix Indices (bone matrix refs for skinning), NOT MaterialId!
                    // MaterialId is in the non-tagged footer after all sub-chunks.
                    for (int i = 0; i < count; i++)
                        geo.MatrixIndices.Add(br.ReadUInt32());
                    break;
                case "TVER":
                    br.ReadBytes((int)count * 4); 
                    break;
                case "UVAS":
                    if (mdxVersion == 1300) // Alpha 0.5.3 Special Case
                    {
                        // Alpha 0.5.3 Optimization: 
                        // If Version is 1300, UVAS block seemingly contains the data directly 
                        // if Count is 1 (which means 1 UV set).
                        // The data length corresponds to the number of vertices.
                        // We must read this data to maintain alignment.
                        int nVerts = geo.Vertices.Count;
                        if (nVerts > 0)
                        {
                            for (int k = 0; k < nVerts; k++)
                                geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
                        }
                    }
                    // If not version 1300, it's a standard container and we continue to inner chunks (UVBS)
                    break;
                case "UVBS":
                    for (int i = 0; i < count; i++)
                        geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
                    break;
                case "BIDX":
                {
                    // Bone Indices — could be 1 byte or 4 bytes per entry depending on version.
                    // Use peek-ahead validation (same pattern as PTYP/PCNT).
                    long afterRead4 = br.BaseStream.Position + count * 4;
                    long afterRead1 = br.BaseStream.Position + count;
                    
                    bool valid4 = false, valid1 = false;
                    if (afterRead4 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead4;
                        string nextTag4 = Encoding.ASCII.GetString(br.ReadBytes(4));
                        valid4 = IsValidGeosetTag(nextTag4);
                        br.BaseStream.Position = save;
                    }
                    if (afterRead1 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead1;
                        string nextTag1 = Encoding.ASCII.GetString(br.ReadBytes(4));
                        valid1 = IsValidGeosetTag(nextTag1);
                        br.BaseStream.Position = save;
                    }
                    
                    if (valid4)
                        br.ReadBytes((int)count * 4);
                    else if (valid1)
                        br.ReadBytes((int)count);
                    else
                        br.ReadBytes((int)count * 4); // fallback to 4-byte
                    break;
                }
                case "BWGT":
                {
                    // Bone Weights — could be 1 byte or 4 bytes per entry.
                    // Use peek-ahead validation.
                    long afterRead4 = br.BaseStream.Position + count * 4;
                    long afterRead1 = br.BaseStream.Position + count;
                    
                    bool valid4 = false, valid1 = false;
                    if (afterRead4 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead4;
                        string nextTag4 = Encoding.ASCII.GetString(br.ReadBytes(4));
                        valid4 = IsValidGeosetTag(nextTag4);
                        br.BaseStream.Position = save;
                    }
                    if (afterRead1 + 4 <= geoEnd)
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = afterRead1;
                        string nextTag1 = Encoding.ASCII.GetString(br.ReadBytes(4));
                        valid1 = IsValidGeosetTag(nextTag1);
                        br.BaseStream.Position = save;
                    }
                    
                    if (valid4)
                        br.ReadBytes((int)count * 4);
                    else if (valid1)
                        br.ReadBytes((int)count);
                    else
                    {
                        // BWGT is often the last tagged chunk before the footer.
                        // If neither stride lands on a valid tag, consume remaining to geoEnd minus footer.
                        long footerSize = Math.Min(geoEnd - br.BaseStream.Position, 12); // MaterialId+SelectionGroup+Flags minimum
                        long dataToRead = geoEnd - br.BaseStream.Position - footerSize;
                        if (dataToRead > 0 && dataToRead <= count * 4)
                            br.ReadBytes((int)dataToRead);
                        else
                            br.ReadBytes((int)count * 4); // fallback
                    }
                    break;
                }

                default:
                    // Smart Seek for Alignment/Padding Recovery
                    long currentPos = br.BaseStream.Position - 8; // Start of the unknown tag
                    long limit = Math.Min(currentPos + 64, geoEnd);
                    
                    // Hex dump the mystery bytes for diagnosis
                    {
                        long save = br.BaseStream.Position;
                        br.BaseStream.Position = currentPos;
                        int dumpLen = (int)Math.Min(20, geoEnd - currentPos);
                        byte[] dump = br.ReadBytes(dumpLen);
                        br.BaseStream.Position = save;
                        string hex = BitConverter.ToString(dump).Replace("-", " ");
                        string ascii = new string(dump.Select(b => b >= 32 && b < 127 ? (char)b : '.').ToArray());
                        Console.WriteLine($"      [GEOS#{gIdx}] Unknown tag '{tag}' (0x{(uint)(tag[0])|(uint)(tag[1])<<8|(uint)(tag[2])<<16|(uint)(tag[3])<<24:X8}) count=0x{count:X8} at pos {currentPos}");
                        Console.WriteLine($"      [GEOS#{gIdx}] Hex: {hex}");
                        Console.WriteLine($"      [GEOS#{gIdx}] Asc: {ascii}");
                    }
                    
                    // Start scan from 1 byte ahead
                    br.BaseStream.Position = currentPos + 1; 
                    bool recovered = false;
                    
                    while (br.BaseStream.Position < limit - 4)
                    {
                        long p = br.BaseStream.Position;
                        byte[] tagBytes = br.ReadBytes(4);
                        br.BaseStream.Position = p; // Rewind
                        
                        string possibleTag = Encoding.ASCII.GetString(tagBytes);
                        if (IsValidGeosetTag(possibleTag))
                        {
                             Console.WriteLine($"      [GEOS#{gIdx}] RECOVERY: Skipped {p - currentPos} bytes → '{possibleTag}' at pos {p}");
                             br.BaseStream.Position = p; // Align to new tag
                             recovered = true;
                             break;
                        }
                        br.BaseStream.Position = p + 1;
                    }

                    if (!recovered)
                    {
                        Console.WriteLine($"      [GEOS#{gIdx}] WARN: Recovery scan failed for tag '{tag}' at {currentPos}");
                        br.BaseStream.Position = currentPos + 8;
                    }
                    break;
            }
        }

        // After all tagged sub-chunks, read non-tagged footer data
        // Standard WC3/Alpha MDX geoset footer: MaterialId(4) + SelectionGroup(4) + SelectionFlags(4)
        // + BoundsRadius(4) + BoundsMin(12) + BoundsMax(12) + NumSeqExtents(4) + SeqExtents[N](28 each)
        long remaining = geoEnd - br.BaseStream.Position;
        if (remaining >= 12)
        {
            geo.MaterialId = br.ReadInt32();
            geo.SelectionGroup = br.ReadUInt32();
            geo.Flags = br.ReadUInt32();
        }

        return geo;
    }

    static bool IsValidGeosetTag(string tag)
    {
        return tag == "VRTX" || tag == "NRMS" || tag == "PTYP" || tag == "PCNT" || 
               tag == "PVTX" || tag == "GNDX" || tag == "MTGC" || tag == "MATS" || 
               tag == "TVER" || tag == "UVAS" || tag == "UVBS" || tag == "BIDX" ||
               tag == "BWGT";
    }

    /// <summary>
    /// Read a Node structure (shared base for bones, emitters, lights, etc.)
    /// Returns (name, objectId, parentId, flags, pivotPosition).
    /// The reader is positioned after the node (including any animation sub-chunks).
    /// </summary>
    static (string name, int objectId, int parentId, uint flags) ReadNode(BinaryReader br)
    {
        uint nodeSize = br.ReadUInt32();
        long nodeEnd = br.BaseStream.Position - 4 + nodeSize;

        string name = ReadFixedString(br, 0x50);
        int objectId = br.ReadInt32();
        int parentId = br.ReadInt32();
        uint flags = br.ReadUInt32();

        // Skip animation sub-chunks (KGTR, KGRT, KGSC) within the node
        br.BaseStream.Position = nodeEnd;

        return (name, objectId, parentId, flags);
    }

    /// <summary>Parse PRE2 chunk — Particle Emitter 2 entries</summary>
    static void ReadPre2(BinaryReader br, uint chunkSize, List<MdlParticleEmitter2> emitters, List<C3Vector> pivots)
    {
        long chunkEnd = br.BaseStream.Position + chunkSize;
        uint count = br.ReadUInt32();

        for (uint i = 0; i < count && br.BaseStream.Position < chunkEnd; i++)
        {
            long emitterStart = br.BaseStream.Position;
            uint emitterSize = br.ReadUInt32();
            long emitterEnd = emitterStart + emitterSize;

            try
            {
                var emitter = new MdlParticleEmitter2();
                var node = ReadNode(br);
                emitter.Name = node.name;
                emitter.ObjectId = node.objectId;
                emitter.ParentId = node.parentId;
                emitter.Flags = node.flags;

                // Set pivot from PIVT chunk if available
                if (node.objectId >= 0 && node.objectId < pivots.Count)
                    emitter.Position = pivots[node.objectId];

                // Emitter content size (redundant)
                if (emitterEnd - br.BaseStream.Position < 4) { br.BaseStream.Position = emitterEnd; continue; }
                br.ReadUInt32(); // content size

                // Read emitter properties (matching reference implementation order)
                if (emitterEnd - br.BaseStream.Position < 172) { br.BaseStream.Position = emitterEnd; continue; }

                br.ReadUInt32(); // EmitterType (0=Base, 1=Plane, 2=Sphere, 3=Spline)
                emitter.Speed = br.ReadSingle();
                emitter.Variation = br.ReadSingle();
                emitter.Latitude = br.ReadSingle();
                br.ReadSingle(); // Longitude
                emitter.Gravity = br.ReadSingle();
                br.ReadSingle(); // ZSource
                emitter.Lifespan = br.ReadSingle();
                emitter.EmissionRate = br.ReadSingle();
                emitter.Length = br.ReadSingle();
                emitter.Width = br.ReadSingle();
                emitter.Rows = br.ReadInt32();
                emitter.Columns = br.ReadInt32();
                emitter.HeadOrTail = br.ReadInt32(); // +1 in ref impl (0→Head=1, 1→Tail=2, 2→Both=3)
                emitter.TailLength = br.ReadSingle();
                emitter.Time = br.ReadSingle();

                // 3 segment colors (RGB floats)
                for (int c = 0; c < 3; c++)
                    emitter.SegmentColor[c] = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());

                // 3 segment alphas (bytes)
                emitter.SegmentAlpha[0] = br.ReadByte();
                emitter.SegmentAlpha[1] = br.ReadByte();
                emitter.SegmentAlpha[2] = br.ReadByte();

                // ParticleScaling[3], LifeSpanUVAnim[3], DecayUVAnim[3], TailUVAnim[3], TailDecayUVAnim[3]
                // = 15 floats
                emitter.SegmentScaling[0] = br.ReadSingle();
                emitter.SegmentScaling[1] = br.ReadSingle();
                emitter.SegmentScaling[2] = br.ReadSingle();
                br.ReadBytes(12 * 4); // skip 12 UV anim floats

                emitter.FilterMode = (ParticleFilterMode)br.ReadInt32();
                emitter.TextureId = br.ReadInt32();
                emitter.PriorityPlane = br.ReadInt32();
                emitter.ReplaceableId = br.ReadUInt32();

                // Skip remaining fields (GeometryModel, RecursionModel, TwinkleFps, etc.)
                // and animation sub-chunks — just jump to end
                emitters.Add(emitter);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PRE2] Failed to parse emitter {i}: {ex.Message}");
            }

            br.BaseStream.Position = emitterEnd;
        }

        br.BaseStream.Position = chunkEnd;
        Console.WriteLine($"[PRE2] Parsed {emitters.Count} particle emitters");
    }

    /// <summary>Parse RIBB chunk — Ribbon Emitter entries</summary>
    static void ReadRibb(BinaryReader br, uint chunkSize, List<MdlRibbonEmitter> emitters, List<C3Vector> pivots)
    {
        long chunkEnd = br.BaseStream.Position + chunkSize;
        uint count = br.ReadUInt32();

        for (uint i = 0; i < count && br.BaseStream.Position < chunkEnd; i++)
        {
            long emitterStart = br.BaseStream.Position;
            uint emitterSize = br.ReadUInt32();
            long emitterEnd = emitterStart + emitterSize;

            try
            {
                var emitter = new MdlRibbonEmitter();
                var node = ReadNode(br);
                emitter.Name = node.name;
                emitter.ObjectId = node.objectId;
                emitter.ParentId = node.parentId;
                emitter.Flags = node.flags;

                if (node.objectId >= 0 && node.objectId < pivots.Count)
                    emitter.Position = pivots[node.objectId];

                // Emitter content size (redundant)
                if (emitterEnd - br.BaseStream.Position < 4) { br.BaseStream.Position = emitterEnd; continue; }
                br.ReadUInt32();

                if (emitterEnd - br.BaseStream.Position < 40) { br.BaseStream.Position = emitterEnd; continue; }

                emitter.HeightAbove = br.ReadSingle();
                emitter.HeightBelow = br.ReadSingle();
                emitter.Alpha = br.ReadSingle();
                emitter.Color = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                emitter.Lifespan = br.ReadSingle();
                emitter.TextureSlot = br.ReadInt32();
                emitter.EmissionRate = br.ReadInt32();
                emitter.Rows = br.ReadInt32();
                emitter.Columns = br.ReadInt32();
                emitter.MaterialId = br.ReadInt32();
                emitter.Gravity = br.ReadSingle();

                emitters.Add(emitter);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[RIBB] Failed to parse emitter {i}: {ex.Message}");
            }

            br.BaseStream.Position = emitterEnd;
        }

        br.BaseStream.Position = chunkEnd;
        Console.WriteLine($"[RIBB] Parsed {emitters.Count} ribbon emitters");
    }
}
