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
        
        // Ghidra-verified: GEOS chunk body starts with uint32 geosetCount
        uint geosetCount = br.ReadUInt32();
        Console.WriteLine($"  [GEOS] geosetCount={geosetCount}, chunkSize={size}, pos={br.BaseStream.Position}");

        // Sanity check: if geosetCount looks like a size (> 255), it's probably not a count
        // and the format doesn't have the count field here
        if (geosetCount > 255)
        {
            Console.WriteLine($"  [GEOS] geosetCount={geosetCount} looks like a size, rewinding");
            br.BaseStream.Position -= 4;
            geosetCount = uint.MaxValue; // use while loop instead
        }

        uint g = 0;
        while (br.BaseStream.Position < end && g < geosetCount)
        {
            if (end - br.BaseStream.Position < 4) break;

            uint geosetSize = br.ReadUInt32();
            long geosetEnd = br.BaseStream.Position - 4 + geosetSize;
            if (g < 2) Console.WriteLine($"  [GEOS] Geoset[{g}]: size={geosetSize}, end={geosetEnd}, posAfterSize={br.BaseStream.Position}");
            
            var geoset = ReadGeoset(br, geosetSize - 4, version, g < 2);
            geosets.Add(geoset);
            
            br.BaseStream.Position = geosetEnd;
            g++;
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

    static MdlGeoset ReadGeoset(BinaryReader br, uint size, uint mdxVersion, bool debug = false)
    {
        var geo = new MdlGeoset();
        long geoEnd = br.BaseStream.Position + size;

        while (br.BaseStream.Position < geoEnd)
        {
            if (geoEnd - br.BaseStream.Position < 8) break;

            // Peek at next 4 bytes to check if it's a valid sub-chunk tag.
            // If not, we've reached the trailing standalone fields (materialId, etc.)
            long peekPos = br.BaseStream.Position;
            string peekTag = ReadTag(br);
            br.BaseStream.Position = peekPos; // rewind
            if (!IsValidGeosetTag(peekTag))
            {
                if (debug) { Console.WriteLine($"    [PEEK] Non-tag at pos={peekPos}: '{peekTag}' (0x{(peekPos < br.BaseStream.Length - 4 ? BitConverter.ToString(br.ReadBytes(4)).Replace("-","") : "EOF")})"); br.BaseStream.Position = peekPos; }
                break; // exit loop to read trailing fields
            }

            long subPos = br.BaseStream.Position;
            string tag = ReadTag(br);
            uint count = br.ReadUInt32();
            long afterHeader = br.BaseStream.Position;
            if (debug) Console.WriteLine($"    [{tag}] count={count} at pos={subPos} dataStart={afterHeader}");
            switch (tag)
            {
                case "VRTX":
                    for (int i = 0; i < count; i++)
                        geo.Vertices.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    if (debug) Console.WriteLine($"      VRTX read {count} verts, pos now={br.BaseStream.Position}");
                    break;
                case "NRMS":
                    for (int i = 0; i < count; i++)
                        geo.Normals.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    if (debug) Console.WriteLine($"      NRMS read {count} norms, pos now={br.BaseStream.Position}");
                    break;
                case "PTYP":
                    br.ReadBytes((int)count * 4);
                    if (debug) Console.WriteLine($"      PTYP skipped {count*4} bytes, pos now={br.BaseStream.Position}");
                    break;
                case "PCNT":
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
                    // Bone matrix indices, NOT material ID.
                    // materialId is a standalone trailing field after all sub-chunks.
                    br.ReadBytes((int)count * 4);
                    break;
                case "TVER":
                    br.ReadBytes((int)count * 4);
                    break;
                case "UVAS":
                {
                    // Alpha 0.5.3 (v1300): count = number of UV sets (typically 1)
                    // Data follows inline: numVertices UV pairs per set
                    int nVerts = geo.Vertices.Count;
                    if (nVerts > 0)
                    {
                        for (int k = 0; k < nVerts; k++)
                            geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
                        // Skip additional UV sets beyond the first
                        int extraSets = (int)count - 1;
                        if (extraSets > 0)
                            br.ReadBytes(extraSets * nVerts * 8);
                    }
                    if (debug) Console.WriteLine($"      UVAS count={count} nVerts={nVerts}, pos now={br.BaseStream.Position}");
                    break;
                }
                case "UVBS":
                    for (int i = 0; i < count; i++)
                        geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
                    break;
                case "BIDX":
                    br.ReadBytes((int)count * 4);
                    break;
                case "BWGT":
                    br.ReadBytes((int)count * 4);
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

        // Per wowdev wiki MDLGEOSETSECTION: after all sub-chunks come standalone fields:
        //   uint32 materialId, uint32 selectionGroup, uint32 flags,
        //   CMdlBounds bounds, uint32 numSeqBounds, CMdlBounds seqBounds[numSeqBounds]
        // Read these if there's enough data remaining.
        long remaining = geoEnd - br.BaseStream.Position;
        if (debug) Console.WriteLine($"    [TRAILING] pos={br.BaseStream.Position}, remaining={remaining}, geoEnd={geoEnd}");
        if (remaining >= 12) // at least materialId + selectionGroup + flags
        {
            geo.MaterialId = br.ReadInt32();
            if (debug) Console.WriteLine($"    [TRAILING] materialId={geo.MaterialId}, selGroup={br.BaseStream.Position}, flags next");
            geo.SelectionGroup = br.ReadUInt32();
            geo.Flags = br.ReadUInt32();
            
            // CMdlBounds = CAaBox(6 floats) + float radius = 28 bytes
            remaining = geoEnd - br.BaseStream.Position;
            if (remaining >= 28)
            {
                br.ReadBytes(28); // bounds (min3 + max3 + radius) — skip for now
                
                remaining = geoEnd - br.BaseStream.Position;
                if (remaining >= 4)
                {
                    uint numSeqBounds = br.ReadUInt32();
                    long seqBoundsBytes = numSeqBounds * 28L;
                    remaining = geoEnd - br.BaseStream.Position;
                    if (seqBoundsBytes <= remaining)
                        br.ReadBytes((int)seqBoundsBytes);
                }
            }
        }

        // Seek to exact geoset end in case of any alignment issues
        br.BaseStream.Position = geoEnd;

        return geo;
    }

    static bool IsValidGeosetTag(string tag)
    {
        return tag == "VRTX" || tag == "NRMS" || tag == "PTYP" || tag == "PCNT" || 
               tag == "PVTX" || tag == "GNDX" || tag == "MTGC" || tag == "MATS" || 
               tag == "TVER" || tag == "UVAS" || tag == "UVBS" || tag == "BIDX" || tag == "BWGT";
    }
}
