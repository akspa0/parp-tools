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

                case MdxHeaders.BONE:
                case MdxHeaders.LITE:
                case MdxHeaders.HELP:
                case MdxHeaders.ATCH:
                case MdxHeaders.PIVT:
                case MdxHeaders.CAMS:
                case MdxHeaders.EVTS:
                case MdxHeaders.CLID:
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

    static MdlGeoset ReadGeoset(BinaryReader br, uint size, uint mdxVersion)
    {
        var geo = new MdlGeoset();
        long geoEnd = br.BaseStream.Position + size;

        while (br.BaseStream.Position < geoEnd)
        {
            if (geoEnd - br.BaseStream.Position < 8) break;

            string tag = ReadTag(br);
            uint count = br.ReadUInt32();


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
                    // Was 1 byte, but alignment suggests 4 bytes (standard uint32)
                    // If count is 1 (triangles), we want 4 bytes.
                    br.ReadBytes((int)count * 4); 
                    break;
                case "PCNT":
                    // Primitive Counts
                    br.ReadBytes((int)count * 4); 
                    break;
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
                    if (count > 0)
                    {
                        geo.MaterialId = br.ReadInt32();
                        if (count > 1) br.ReadBytes((int)(count - 1) * 4);
                    }
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
                    // Bone Indices (Alpha legacy?)
                    // Determine element size based on remaining bytes in geoset
                    long bidxRem = geoEnd - br.BaseStream.Position;
                    if (count > 0 && bidxRem >= count)
                    {
                        // Check if 4 bytes per element matches remaining exactly
                        if (bidxRem == count * 4)
                        {
                            br.ReadBytes((int)count * 4);
                        }
                        else if (bidxRem == count) // 1 byte per element
                        {
                            br.ReadBytes((int)count);
                        }
                        else
                        {
                            // Ambiguous. Default to 1 (safe?) or 4?
                            // Or just Smart Seek the next tag?
                            // If we are at end, consuming 'remaining' is safest.
                                br.ReadBytes((int)bidxRem);
                        }
                    }
                    break;

                default:
                    // Smart Seek for Alignment/Padding Recovery
                    // Alpha 0.5.3 often puts padding bytes between chunks (e.g. 8 bytes after UVAS data).
                    // Instead of aborting, we scan forward a short distance to find the next valid tag.
                    long currentPos = br.BaseStream.Position - 8; // Start of the unknown tag
                    long limit = Math.Min(currentPos + 64, geoEnd);
                    
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
                             Console.WriteLine($"      [RECOVERY] Skipped {p - currentPos} bytes. Resuming at valid tag '{possibleTag}' (Pos: {p}).");
                             br.BaseStream.Position = p; // Align to new tag
                             recovered = true;
                             break;
                        }
                        br.BaseStream.Position = p + 1;
                    }

                    if (!recovered)
                    {
                        Console.WriteLine($"      [WARN] Unknown GEOS sub-tag: {tag} at {currentPos}. Scan failed.");
                        // Restore position to continue blindly or let loop finish
                        br.BaseStream.Position = currentPos + 8;
                    }
                    break;
            }
        }

        return geo;
    }

    static bool IsValidGeosetTag(string tag)
    {
        return tag == "VRTX" || tag == "NRMS" || tag == "PTYP" || tag == "PCNT" || 
               tag == "PVTX" || tag == "GNDX" || tag == "MTGC" || tag == "MATS" || 
               tag == "TVER" || tag == "UVAS" || tag == "UVBS" || tag == "BIDX";
    }
}
