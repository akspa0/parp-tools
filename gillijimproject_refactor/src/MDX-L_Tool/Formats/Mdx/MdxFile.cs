using System.Text;
using MdxLTool.Formats.Obj;

namespace MdxLTool.Formats.Mdx;

/// <summary>
/// MDX file reader/writer.
/// Based on Ghidra analysis of MDLFileRead (0x0078b660) and related functions.
/// </summary>
public class MdxFile
{
    /// <summary>When false (default), suppresses diagnostic Console.WriteLine output for performance.</summary>
    public static bool Verbose { get; set; } = false;

    public uint Version { get; set; }
    public MdlModel Model { get; set; } = new();
    public byte[]? RawData { get; set; }
    public string ModelName 
    { 
        get => Model.Name;
        set => Model.Name = value;
    }

    static bool LooksLikeLegacyNamedSeqRecord(BinaryReader br, long recordStart, uint entrySize, long chunkEnd)
    {
        if (recordStart < 0 || recordStart + entrySize > chunkEnd)
            return false;

        long save = br.BaseStream.Position;
        try
        {
            br.BaseStream.Position = recordStart + 0x50;
            uint startTime = br.ReadUInt32();
            uint endTime = br.ReadUInt32();
            float moveSpeed = br.ReadSingle();

            bool intervalLooksRight = endTime >= startTime && (endTime - startTime) <= 0x0FFFFFFF;
            bool moveSpeedLooksRight = !float.IsNaN(moveSpeed) && !float.IsInfinity(moveSpeed) && moveSpeed >= 0f && moveSpeed < 10000f;
            return intervalLooksRight && moveSpeedLooksRight;
        }
        catch
        {
            return false;
        }
        finally
        {
            br.BaseStream.Position = save;
        }
    }

    static void ParseLegacyNamedSeqRecord(BinaryReader br, uint entrySize, List<MdlSequence> sequences)
    {
        long entryStart = br.BaseStream.Position;

        var seq = new MdlSequence();
        seq.Name = ReadFixedString(br, 0x50);

        var time = new CiRange
        {
            Start = br.ReadInt32(),
            End = br.ReadInt32()
        };
        seq.Time = time;

        seq.MoveSpeed = br.ReadSingle();
        seq.Flags = br.ReadUInt32();
        seq.Frequency = br.ReadSingle();

        if (entrySize == 128)
        {
            uint syncPoint = br.ReadUInt32();
            seq.Replay = new CiRange { Start = unchecked((int)syncPoint), End = 0 };
        }
        else if (entrySize == 132)
        {
            uint syncPoint = br.ReadUInt32();
            seq.Replay = new CiRange { Start = unchecked((int)syncPoint), End = 0 };
        }
        else
        {
            int replayStart = br.ReadInt32();
            int replayEnd = br.ReadInt32();
            seq.Replay = new CiRange { Start = replayStart, End = replayEnd };
            if (entrySize >= 140)
                seq.BlendTime = br.ReadUInt32();
        }

        var bounds = new CMdlBounds();
        if (entrySize == 128)
        {
            bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            bounds.Radius = br.ReadSingle();
        }
        else
        {
            bounds.Radius = br.ReadSingle();
            bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        }
        seq.Bounds = bounds;

        br.BaseStream.Position = entryStart + entrySize;
        sequences.Add(seq);
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
        string currentTag = "<magic>";
        long currentChunkOffset = br.BaseStream.Position;
        uint currentChunkSize = 0;
        var chunkTrail = new List<string>(32);

        try
        {
            // Verify magic
            uint magic = br.ReadUInt32();
            if (magic != MdxHeaders.MAGIC)
                throw new InvalidDataException($"Invalid MDX magic: 0x{magic:X8} (expected 0x{MdxHeaders.MAGIC:X8})");

            // Read chunks until EOF
            while (br.BaseStream.Position < br.BaseStream.Length)
            {
                if (br.BaseStream.Length - br.BaseStream.Position < 8) break;

                currentChunkOffset = br.BaseStream.Position;
                string tag = ReadTag(br);
                currentTag = tag;
                uint size = br.ReadUInt32();
                currentChunkSize = size;
                long payloadStart = br.BaseStream.Position;
                long remaining = br.BaseStream.Length - payloadStart;
                if (size > remaining)
                    throw new InvalidDataException($"Chunk '{tag}' overruns stream: size=0x{size:X}, remaining=0x{remaining:X}");

                long chunkEnd = payloadStart + size;
                chunkTrail.Add($"{tag}@0x{currentChunkOffset:X}/0x{size:X}");

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
                        ReadSeqs(br, size, mdx.Sequences, mdx.Version);
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

                    case MdxHeaders.PIVT:
                        ReadPivt(br, size, mdx.PivotPoints);
                        break;

                    case MdxHeaders.BONE:
                        ReadBone(br, size, mdx.Bones, mdx.PivotPoints);
                        break;

                    case MdxHeaders.HELP:
                        ReadHelp(br, size, mdx.Bones, mdx.PivotPoints);
                        break;

                    case MdxHeaders.LITE:
                    case MdxHeaders.ATCH:
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
                if (br.BaseStream.Position > chunkEnd)
                    throw new InvalidDataException($"Chunk '{tag}' parser overread: pos=0x{br.BaseStream.Position:X}, end=0x{chunkEnd:X}");

                if (br.BaseStream.Position != chunkEnd)
                    br.BaseStream.Position = chunkEnd;
            }
        }
        catch (Exception ex) when (ex is not InvalidDataException)
        {
            int trailStart = Math.Max(0, chunkTrail.Count - 8);
            string trail = chunkTrail.Count > 0
                ? string.Join(" -> ", chunkTrail.GetRange(trailStart, chunkTrail.Count - trailStart))
                : "<none>";
            throw new InvalidDataException(
                $"MDX parse failed at tag '{currentTag}' offset=0x{currentChunkOffset:X} size=0x{currentChunkSize:X}. Chunk trail: {trail}. {ex.Message}",
                ex);
        }

        // Deferred pivot assignment: PIVT chunk typically comes AFTER BONE/HELP in MDX files,
        // so pivots may not be available when bones are first parsed.
        for (int i = 0; i < mdx.Bones.Count; i++)
        {
            var bone = mdx.Bones[i];
            if (bone.ObjectId >= 0 && bone.ObjectId < mdx.PivotPoints.Count)
                bone.Pivot = mdx.PivotPoints[bone.ObjectId];
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
        if (Verbose) Console.WriteLine($"MDX Version: {version}");
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

    static void ReadSeqs(BinaryReader br, uint size, List<MdlSequence> sequences, uint mdxVersion)
    {
        long startPos = br.BaseStream.Position;

        // 0.9.0 deep-dive contract: SEQS uses
        //   uint32 count + count * 0x8C records
        // and should hard-fail when this framing is violated.
        if (size >= 4)
        {
            uint count = br.ReadUInt32();
            uint remaining = size - 4;
            const uint seqRecordSize090 = 0x8C;
            if (count > 0 && remaining == count * seqRecordSize090)
            {
                long seqDataStart = startPos + 4;
                long seqChunkEnd = startPos + size;

                // Some files are framed as count + N * 0x8C but still use name-prefixed
                // records instead of strict numeric 0.9.0 layout. Detect this by sampling
                // record heads and parse as named records when sane.
                uint nameSampleCount = Math.Min(count, 2u);
                bool countedNamedLooksSane = true;
                for (uint i = 0; i < nameSampleCount; i++)
                {
                    long recordStart = seqDataStart + i * seqRecordSize090;
                    if (!LooksLikeLegacyNamedSeqRecord(br, recordStart, seqRecordSize090, seqChunkEnd))
                    {
                        countedNamedLooksSane = false;
                        break;
                    }
                }

                if (countedNamedLooksSane)
                {
                    br.BaseStream.Position = seqDataStart;
                    for (uint i = 0; i < count; i++)
                        ParseCountedNamedSeqRecord8C(br, sequences);
                    return;
                }

                uint sanitySampleCount = Math.Min(count, 2u);
                bool seq090LooksSane = true;
                for (uint i = 0; i < sanitySampleCount; i++)
                {
                    long recordStart = seqDataStart + i * seqRecordSize090;
                    if (!LooksLikeSeq090Record(br, recordStart, seqChunkEnd))
                    {
                        seq090LooksSane = false;
                        break;
                    }
                }

                br.BaseStream.Position = seqDataStart;
                if (seq090LooksSane)
                {
                    for (uint i = 0; i < count; i++)
                    {
                        long entryStart = br.BaseStream.Position;
                        var seq = new MdlSequence();

                        // Recovered map (0x8C record):
                        // 0x00 animId, 0x04 subId, 0x08..0x0F zero,
                        // 0x50 startTime, 0x54 endTime, 0x58 moveSpeed,
                        // 0x5C flags, 0x60..0x6B bbox min, 0x6C..0x77 bbox max,
                        // 0x78 blendTime(float/unknown), 0x7C playbackSpeed, 0x80 frequency.
                        uint animId = br.ReadUInt32();
                        br.ReadUInt32(); // subId
                        br.ReadBytes(8); // 0x08..0x0F reserved/zero

                        // 0x10..0x4F are not decoded in this pass.
                        br.ReadBytes(0x40);

                        uint intervalStart = br.ReadUInt32();
                        uint intervalEnd = br.ReadUInt32();
                        seq.Time = new CiRange
                        {
                            Start = unchecked((int)intervalStart),
                            End = unchecked((int)intervalEnd)
                        };

                        seq.MoveSpeed = br.ReadSingle();
                        seq.Flags = br.ReadUInt32();

                        var bounds = new CMdlBounds();
                        bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                        bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                        seq.Bounds = bounds;

                        br.ReadSingle(); // blendTime-like float at 0x78
                        br.ReadUInt32(); // playbackSpeed-like value at 0x7C
                        uint frequencyLike = br.ReadUInt32();
                        seq.Frequency = frequencyLike;

                        br.ReadUInt32(); // 0x84 pad/unk
                        br.ReadUInt32(); // 0x88 pad/unk

                        seq.Name = $"Seq_{animId}";
                        sequences.Add(seq);

                        br.BaseStream.Position = entryStart + seqRecordSize090;
                    }
                    return;
                }
            }

            br.BaseStream.Position = startPos;
        }

        // WoW Alpha SEQS variants observed in the wild:
        //   uint32 count + count * entrySize
        // where entrySize is commonly 140 (Alpha), but 136/132 variants also exist.
        // Core fields always begin with:
        //   name[80], intervalStart, intervalEnd, moveSpeed, flags, frequency
        if (size >= 4)
        {
            int count = br.ReadInt32();
            if (count > 0)
            {
                uint remaining = size - 4;
                if (remaining % (uint)count == 0)
                {
                    uint entrySize = remaining / (uint)count;
                    if (entrySize is 128 or 132 or 136 or 140)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            long entryStart = br.BaseStream.Position;

                            var seq = new MdlSequence();
                            seq.Name = ReadFixedString(br, 0x50);

                            uint intervalStart = br.ReadUInt32();
                            uint intervalEnd = br.ReadUInt32();
                            seq.Time = new CiRange
                            {
                                Start = unchecked((int)intervalStart),
                                End = unchecked((int)intervalEnd)
                            };

                            seq.MoveSpeed = br.ReadSingle();
                            seq.Flags = br.ReadUInt32();
                            seq.Frequency = br.ReadSingle();

                            // Variants differ in the next metadata fields:
                            // 132: syncPoint (uint)
                            // 136: replayStart/replayEnd (2x int)
                            // 140: replayStart/replayEnd + blendTime (int)
                            if (entrySize == 128)
                            {
                                uint syncPoint = br.ReadUInt32();
                                seq.Replay = new CiRange { Start = unchecked((int)syncPoint), End = 0 };
                            }
                            else if (entrySize == 132)
                            {
                                uint syncPoint = br.ReadUInt32();
                                seq.Replay = new CiRange { Start = unchecked((int)syncPoint), End = 0 };
                            }
                            else
                            {
                                int replayStart = br.ReadInt32();
                                int replayEnd = br.ReadInt32();
                                seq.Replay = new CiRange { Start = replayStart, End = replayEnd };
                                if (entrySize == 140)
                                    seq.BlendTime = br.ReadUInt32();
                            }

                            var bounds = new CMdlBounds();
                            if (entrySize == 128)
                            {
                                bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                                bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                                bounds.Radius = br.ReadSingle();
                            }
                            else
                            {
                                bounds.Radius = br.ReadSingle();
                                bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                                bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                            }
                            seq.Bounds = bounds;

                            // Ensure alignment to declared entry size even if there are extra unknown bytes
                            br.BaseStream.Position = entryStart + entrySize;

                            sequences.Add(seq);
                        }
                        return;
                    }
                }
            }
        }

        // Fallback legacy parsing: raw array with no leading count.
        // Seen variants use 132/136/140-byte records, sometimes with tail padding.
        br.BaseStream.Position = startPos;
        long seqChunkEndFallback = startPos + size;
        uint[] rawEntrySizes = { 140, 136, 132, 128 };
        foreach (uint entrySize in rawEntrySizes)
        {
            if (size < entrySize)
                continue;

            uint remainder = size % entrySize;
            if (remainder > 12)
                continue;

            uint legacyCount = size / entrySize;
            if (legacyCount == 0)
                continue;

            uint sanitySampleCount = Math.Min(legacyCount, 2u);
            bool looksSane = true;
            for (uint i = 0; i < sanitySampleCount; i++)
            {
                long recordStart = startPos + i * entrySize;
                if (!LooksLikeLegacyNamedSeqRecord(br, recordStart, entrySize, seqChunkEndFallback))
                {
                    looksSane = false;
                    break;
                }
            }

            if (!looksSane)
                continue;

            br.BaseStream.Position = startPos;
            for (uint i = 0; i < legacyCount; i++)
                ParseLegacyNamedSeqRecord(br, entrySize, sequences);
            return;
        }

        // Final compatibility fallback: historical raw 132-byte parsing.
        br.BaseStream.Position = startPos;
        uint legacyCount132 = size / 132;
        for (uint i = 0; i < legacyCount132; i++)
            ParseLegacyNamedSeqRecord(br, 132, sequences);
    }

    static void ParseCountedNamedSeqRecord8C(BinaryReader br, List<MdlSequence> sequences)
    {
        long entryStart = br.BaseStream.Position;
        const uint entrySize = 0x8C;

        var seq = new MdlSequence();
        seq.Name = ReadFixedString(br, 0x50);

        uint intervalStart = br.ReadUInt32();
        uint intervalEnd = br.ReadUInt32();
        seq.Time = new CiRange
        {
            Start = unchecked((int)intervalStart),
            End = unchecked((int)intervalEnd)
        };

        seq.MoveSpeed = br.ReadSingle();
        seq.Flags = br.ReadUInt32();

        var bounds = new CMdlBounds();
        bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        seq.Bounds = bounds;

        br.ReadSingle(); // 0x78 blend/playback-like float (unknown)
        br.ReadUInt32(); // 0x7C playback-like value

        int replayStart = br.ReadInt32();  // 0x80
        int replayEnd = br.ReadInt32();    // 0x84
        seq.Replay = new CiRange { Start = replayStart, End = replayEnd };

        seq.BlendTime = br.ReadUInt32(); // 0x88
        seq.Frequency = 1.0f;

        br.BaseStream.Position = entryStart + entrySize;
        sequences.Add(seq);
    }

    static bool LooksLikeSeq090Record(BinaryReader br, long recordStart, long chunkEnd)
    {
        const int seqRecordSize090 = 0x8C;
        if (recordStart < 0 || recordStart + seqRecordSize090 > chunkEnd)
            return false;

        long save = br.BaseStream.Position;
        try
        {
            // Legacy 0x8C records are often name-based (ASCII in first 0x50 bytes).
            // Strict 0.9.0 records are numeric-heavy; reject obvious name payloads.
            br.BaseStream.Position = recordStart;
            byte[] head = br.ReadBytes(0x20);
            int printable = head.Count(b => b >= 32 && b <= 126);
            if (printable >= 10)
                return false;

            br.BaseStream.Position = recordStart + 0x08;
            uint reserved0 = br.ReadUInt32();
            uint reserved1 = br.ReadUInt32();

            br.BaseStream.Position = recordStart + 0x50;
            uint startTime = br.ReadUInt32();
            uint endTime = br.ReadUInt32();
            float moveSpeed = br.ReadSingle();

            bool reservedLooksRight = reserved0 == 0 && reserved1 == 0;
            bool intervalLooksRight = endTime >= startTime && (endTime - startTime) <= 0x0FFFFFFF;
            bool moveSpeedLooksRight = !float.IsNaN(moveSpeed) && !float.IsInfinity(moveSpeed) && moveSpeed >= 0f && moveSpeed < 10000f;

            return reservedLooksRight && intervalLooksRight && moveSpeedLooksRight;
        }
        catch
        {
            return false;
        }
        finally
        {
            br.BaseStream.Position = save;
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
        // TEXS entry size varies by version:
        //   0.5.3 / WC3: 264 bytes (0x108) = ReplaceableId(4) + Path(0x100) + Flags(4)
        //   0.8.0+:      268 bytes (0x10C) = ReplaceableId(4) + Path(0x104) + Flags(4)
        // Auto-detect from chunk size divisibility.
        uint entrySize;
        int pathLen;
        if (size % 268 == 0)
        {
            entrySize = 268;
            pathLen = 0x104;
        }
        else if (size % 264 == 0)
        {
            entrySize = 264;
            pathLen = 0x100;
        }
        else
        {
            throw new InvalidDataException($"Invalid TEXS size 0x{size:X}: expected divisibility by 0x108 or 0x10C.");
        }

        uint count = size / entrySize;
        for (uint i = 0; i < count; i++)
        {
            var tex = new MdlTexture();
            tex.ReplaceableId = br.ReadUInt32();
            tex.Path = ReadFixedString(br, pathLen);
            tex.Flags = br.ReadUInt32();
            textures.Add(tex);
        }
    }

    static void ReadGeosets(BinaryReader br, uint size, List<MdlGeoset> geosets, uint version)
    {
        long start = br.BaseStream.Position;
        int initialCount = geosets.Count;

        // Ported compatibility path from wow-mdx-viewer:
        // - v1500 uses a different GEOS layout than v1300/v1400.
        // - v1300/v1400 share the classic tagged geoset layout.
        // If strict parse fails, fall back to legacy adaptive parser below.
        if (version == 1500)
        {
            try
            {
                ReadGeosetsPortedV1500(br, size, geosets);
                return;
            }
            catch (Exception ex) when (ex is InvalidDataException || ex is EndOfStreamException || ex is ArgumentOutOfRangeException)
            {
                if (Verbose) Console.WriteLine($"[GEOS] v1500 strict parse failed, using legacy fallback: {ex.Message}");
                br.BaseStream.Position = start;
                if (geosets.Count > initialCount)
                    geosets.RemoveRange(initialCount, geosets.Count - initialCount);
            }
        }
        else if (version >= 1300)
        {
            try
            {
                ReadGeosetsPortedV1300(br, size, geosets);
                return;
            }
            catch (Exception ex) when (ex is InvalidDataException || ex is EndOfStreamException || ex is ArgumentOutOfRangeException)
            {
                if (Verbose) Console.WriteLine($"[GEOS] v1300/v1400 strict parse failed, using legacy fallback: {ex.Message}");
                br.BaseStream.Position = start;
                if (geosets.Count > initialCount)
                    geosets.RemoveRange(initialCount, geosets.Count - initialCount);
            }
        }

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

            long geosetStart = br.BaseStream.Position;
            uint geosetSize = br.ReadUInt32();
            bool payloadSizedEntry = version >= 1400;
            long geosetEnd = payloadSizedEntry
                ? geosetStart + 4 + geosetSize
                : geosetStart + geosetSize;

            // Transitional 0.9.x files can present GEOS in two layouts:
            // 1) size-prefixed geoset entries (expected by this loop)
            // 2) non-size-prefixed payload beginning with a count/other dword
            // If the prefixed framing looks invalid, fall back to a bounded single-geoset parse.
            bool looksLikePrefixedEntry = geosetSize >= 12 && geosetEnd > geosetStart && geosetEnd <= end;
            if (looksLikePrefixedEntry)
            {
                long save = br.BaseStream.Position;
                br.BaseStream.Position = geosetStart + 4;
                string firstSubTag = ReadTag(br);
                looksLikePrefixedEntry = IsValidGeosetTag(firstSubTag);
                br.BaseStream.Position = save;
            }
            if (!looksLikePrefixedEntry)
            {
                br.BaseStream.Position = geosetStart;
                uint remaining = checked((uint)(end - geosetStart));
                var fallback = ReadGeoset(br, remaining, version);
                geosets.Add(fallback);

                if (br.BaseStream.Position > end)
                    throw new InvalidDataException(
                        $"Geoset section overran read buffer in fallback parse: pos=0x{br.BaseStream.Position:X}, geosEnd=0x{end:X}.");

                br.BaseStream.Position = end;
                break;
            }

            uint geosetPayloadSize = payloadSizedEntry
                ? geosetSize
                : geosetSize - 4;
            long savePos = br.BaseStream.Position;
            br.BaseStream.Position = geosetStart + 4;
            string firstTag = ReadTag(br);
            br.BaseStream.Position = geosetStart + 4;
            if (!IsValidGeosetTag(firstTag) && geosetSize >= 16)
            {
                uint possibleHeader = br.ReadUInt32();
                string secondTag = ReadTag(br);
                if (possibleHeader <= 0x100000 && IsValidGeosetTag(secondTag))
                {
                    // Some transitional prefixed entries include an extra count/header dword.
                    // Skip it before handing payload to ReadGeoset.
                    if (geosetPayloadSize >= 4)
                        geosetPayloadSize -= 4;
                }
                else
                {
                    br.BaseStream.Position = savePos;
                }
            }
            else
            {
                br.BaseStream.Position = savePos;
            }
            
            var geoset = ReadGeoset(br, geosetPayloadSize, version);
            geosets.Add(geoset);

            if (br.BaseStream.Position > geosetEnd)
                throw new InvalidDataException(
                    $"Geoset section overran read buffer: pos=0x{br.BaseStream.Position:X}, geosetEnd=0x{geosetEnd:X}.");
            
            br.BaseStream.Position = geosetEnd;
        }
    }

    static void ReadGeosetsPortedV1300(BinaryReader br, uint size, List<MdlGeoset> geosets)
    {
        long chunkStart = br.BaseStream.Position;
        long chunkEnd = chunkStart + size;

        if (chunkEnd - br.BaseStream.Position < 4)
            throw new InvalidDataException("GEOS(v1300): missing geoset count.");

        int geosetCount = br.ReadInt32();
        if (geosetCount < 0 || geosetCount > 100000)
            throw new InvalidDataException($"GEOS(v1300): invalid geoset count {geosetCount}.");

        for (int gi = 0; gi < geosetCount; gi++)
        {
            long geosetStart = br.BaseStream.Position;
            if (chunkEnd - geosetStart < 4)
                throw new InvalidDataException($"GEOS(v1300): truncated geoset header at index {gi}.");

            uint geosetSize = br.ReadUInt32();
            long geosetEnd = geosetStart + geosetSize;
            if (geosetEnd > chunkEnd || geosetEnd <= geosetStart)
                throw new InvalidDataException($"GEOS(v1300): invalid geoset size 0x{geosetSize:X} at index {gi}.");

            var geo = new MdlGeoset();

            ExpectTag(br, "VRTX", "GEOS(v1300): expected VRTX");
            int vertexCount = br.ReadInt32();
            if (vertexCount < 0) throw new InvalidDataException("GEOS(v1300): negative VRTX count.");
            for (int i = 0; i < vertexCount; i++)
                geo.Vertices.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));

            ExpectTag(br, "NRMS", "GEOS(v1300): expected NRMS");
            int normalCount = br.ReadInt32();
            if (normalCount < 0) throw new InvalidDataException("GEOS(v1300): negative NRMS count.");
            for (int i = 0; i < normalCount; i++)
                geo.Normals.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));

            // Optional UVAS block in alpha-era files: direct UV arrays (no UVBS wrapper).
            if (TryReadTag(br, "UVAS"))
            {
                int textureChunkCount = br.ReadInt32();
                if (textureChunkCount < 0) throw new InvalidDataException("GEOS(v1300): negative UVAS count.");
                int tvertCount = geo.Vertices.Count * 2;

                for (int uvSet = 0; uvSet < textureChunkCount; uvSet++)
                {
                    for (int uv = 0; uv < tvertCount / 2; uv++)
                    {
                        float u = br.ReadSingle();
                        float v = br.ReadSingle();
                        if (uvSet == 0)
                            geo.TexCoords.Add(new C2Vector(u, v));
                    }
                }
            }

            ExpectTag(br, "PTYP", "GEOS(v1300): expected PTYP");
            int primitiveCount = br.ReadInt32();
            if (primitiveCount < 0) throw new InvalidDataException("GEOS(v1300): negative PTYP count.");
            for (int i = 0; i < primitiveCount; i++)
            {
                byte primitiveType = br.ReadByte();
                if (primitiveType != 4)
                    throw new InvalidDataException($"GEOS(v1300): unsupported primitive type {primitiveType}.");
            }

            ExpectTag(br, "PCNT", "GEOS(v1300): expected PCNT");
            int faceGroupCount = br.ReadInt32();
            if (faceGroupCount < 0) throw new InvalidDataException("GEOS(v1300): negative PCNT count.");
            for (int i = 0; i < faceGroupCount; i++)
                br.ReadInt32();

            ExpectTag(br, "PVTX", "GEOS(v1300): expected PVTX");
            int indexCount = br.ReadInt32();
            if (indexCount < 0) throw new InvalidDataException("GEOS(v1300): negative PVTX count.");
            for (int i = 0; i < indexCount; i++)
                geo.Indices.Add(br.ReadUInt16());

            ExpectTag(br, "GNDX", "GEOS(v1300): expected GNDX");
            int vertexGroupCount = br.ReadInt32();
            if (vertexGroupCount < 0) throw new InvalidDataException("GEOS(v1300): negative GNDX count.");
            for (int i = 0; i < vertexGroupCount; i++)
                geo.VertexGroups.Add(br.ReadByte());

            ExpectTag(br, "MTGC", "GEOS(v1300): expected MTGC");
            int matrixGroupCount = br.ReadInt32();
            if (matrixGroupCount < 0) throw new InvalidDataException("GEOS(v1300): negative MTGC count.");
            for (int i = 0; i < matrixGroupCount; i++)
                geo.MatrixGroups.Add((uint)br.ReadInt32());

            ExpectTag(br, "MATS", "GEOS(v1300): expected MATS");
            int matrixIndexCount = br.ReadInt32();
            if (matrixIndexCount < 0) throw new InvalidDataException("GEOS(v1300): negative MATS count.");
            for (int i = 0; i < matrixIndexCount; i++)
                geo.MatrixIndices.Add((uint)br.ReadInt32());

            // Optional explicit UVBS block in some variants.
            if (TryReadTag(br, "UVBS"))
            {
                int uvCount = br.ReadInt32();
                if (uvCount < 0) throw new InvalidDataException("GEOS(v1300): negative UVBS count.");
                if (geo.TexCoords.Count == 0)
                {
                    for (int i = 0; i < uvCount; i++)
                        geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
                }
                else
                {
                    br.ReadBytes(uvCount * 8);
                }
            }

            ExpectTag(br, "BIDX", "GEOS(v1300): expected BIDX");
            int boneIndexCount = br.ReadInt32();
            if (boneIndexCount < 0) throw new InvalidDataException("GEOS(v1300): negative BIDX count.");
            for (int i = 0; i < boneIndexCount; i++)
                br.ReadUInt32();

            ExpectTag(br, "BWGT", "GEOS(v1300): expected BWGT");
            int boneWeightCount = br.ReadInt32();
            if (boneWeightCount < 0) throw new InvalidDataException("GEOS(v1300): negative BWGT count.");
            for (int i = 0; i < boneWeightCount; i++)
                br.ReadUInt32();

            geo.MaterialId = br.ReadInt32();
            geo.SelectionGroup = unchecked((uint)br.ReadInt32());
            geo.Flags = unchecked((uint)br.ReadInt32());
            geo.Bounds = ReadBoundsRadiusMinMax(br);

            int geosetAnimCount = br.ReadInt32();
            if (geosetAnimCount < 0) throw new InvalidDataException("GEOS(v1300): negative geosetAnimCount.");
            for (int i = 0; i < geosetAnimCount; i++)
                geo.AnimExtents.Add(ReadBoundsRadiusMinMax(br));

            geosets.Add(geo);
            br.BaseStream.Position = geosetEnd;
        }

        br.BaseStream.Position = chunkEnd;
    }

    static void ReadGeosetsPortedV1500(BinaryReader br, uint size, List<MdlGeoset> geosets)
    {
        long chunkStart = br.BaseStream.Position;
        long chunkEnd = chunkStart + size;

        if (chunkEnd - br.BaseStream.Position < 4)
            throw new InvalidDataException("GEOS(v1500): missing geoset count.");

        int geosetCount = br.ReadInt32();
        if (geosetCount < 0 || geosetCount > 100000)
            throw new InvalidDataException($"GEOS(v1500): invalid geoset count {geosetCount}.");

        var vertexCounts = new List<int>(geosetCount);

        // Pass 1: fixed-size geoset headers
        for (int i = 0; i < geosetCount; i++)
        {
            var geo = new MdlGeoset();

            geo.MaterialId = br.ReadInt32();
            br.ReadSingle(); br.ReadSingle(); br.ReadSingle(); // bounds center

            var bounds = new CMdlBounds();
            bounds.Radius = br.ReadSingle();
            geo.Bounds = bounds;

            geo.SelectionGroup = unchecked((uint)br.ReadInt32());
            br.ReadInt32(); // geoset index
            geo.Flags = unchecked((uint)br.ReadInt32());

            ExpectTag(br, "PVTX", "GEOS(v1500): expected PVTX header");
            int vertexCount = br.ReadInt32();
            if (vertexCount < 0) throw new InvalidDataException("GEOS(v1500): negative vertexCount.");

            ExpectTag(br, "PTYP", "GEOS(v1500): expected PTYP header");
            br.ReadInt32(); // primitiveTypeCount

            ExpectTag(br, "PVTX", "GEOS(v1500): expected PVTX primitive header");
            br.ReadInt32(); // primitiveVertexCount

            br.ReadBytes(8); // padding

            vertexCounts.Add(vertexCount);
            geosets.Add(geo);
        }

        // Pass 2: packed vertex and index data
        for (int i = 0; i < geosetCount; i++)
        {
            var geo = geosets[geosets.Count - geosetCount + i];
            int vertexCount = vertexCounts[i];

            var boneLookup = new List<byte[]>();
            var boneLookupIndex = new Dictionary<string, byte>(StringComparer.Ordinal);

            for (int v = 0; v < vertexCount; v++)
            {
                geo.Vertices.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                br.ReadUInt32(); // BoneWeights

                byte b0 = br.ReadByte();
                byte b1 = br.ReadByte();
                byte b2 = br.ReadByte();
                byte b3 = br.ReadByte();
                string key = $"{b0},{b1},{b2},{b3}";

                geo.Normals.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
                br.ReadBytes(8); // unused TVertex

                if (!boneLookupIndex.TryGetValue(key, out byte groupIndex))
                {
                    groupIndex = checked((byte)boneLookup.Count);
                    boneLookupIndex[key] = groupIndex;
                    boneLookup.Add(new[] { b0, b1, b2, b3 });
                }
                geo.VertexGroups.Add(groupIndex);
            }

            foreach (var weights in boneLookup)
            {
                int len = weights.Length;
                while (len > 1 && weights[len - 1] == 0)
                    len--;

                geo.MatrixGroups.Add((uint)len);
                for (int j = 0; j < len; j++)
                    geo.MatrixIndices.Add(weights[j]);
            }

            br.ReadInt32(); // primitive type
            br.ReadInt32(); // unknown

            ushort numPrimVertices = br.ReadUInt16();
            br.ReadUInt16(); // minVertex
            br.ReadUInt16(); // maxVertex
            br.ReadUInt16(); // padding

            for (int j = 0; j < numPrimVertices; j++)
                geo.Indices.Add(br.ReadUInt16());

            int rem = numPrimVertices % 8;
            if (rem != 0)
                br.ReadBytes(2 * (8 - rem));
        }

        br.BaseStream.Position = chunkEnd;
    }

    static CMdlBounds ReadBoundsRadiusMinMax(BinaryReader br)
    {
        var bounds = new CMdlBounds();
        bounds.Radius = br.ReadSingle();
        bounds.Extent.Min = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        bounds.Extent.Max = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        return bounds;
    }

    static bool TryReadTag(BinaryReader br, string expected)
    {
        long save = br.BaseStream.Position;
        string actual = ReadTag(br);
        if (actual == expected)
            return true;

        br.BaseStream.Position = save;
        return false;
    }

    static void ExpectTag(BinaryReader br, string expected, string errorPrefix)
    {
        string actual = ReadTag(br);
        if (actual != expected)
            throw new InvalidDataException($"{errorPrefix}: got '{actual}'.");
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
                        if (Verbose) Console.WriteLine($"      [PTYP] Using 1-byte elements (count={count}) — next tag valid at +{count}");
                        br.ReadBytes((int)count);
                    }
                    else
                    {
                        // Fallback: try 4-byte
                        if (Verbose) Console.WriteLine($"      [PTYP] Neither 1-byte nor 4-byte gives valid next tag. Defaulting to 4-byte (count={count})");
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
                        if (Verbose) Console.WriteLine($"      [PCNT] Using 1-byte elements (count={count}) — next tag valid at +{count}");
                        br.ReadBytes((int)count);
                    }
                    else
                    {
                        if (Verbose) Console.WriteLine($"      [PCNT] Neither 1-byte nor 4-byte gives valid next tag. Defaulting to 4-byte (count={count})");
                        br.ReadBytes((int)count * 4);
                    }
                    break;
                }
                case "PVTX":
                    for (int i = 0; i < count; i++)
                        geo.Indices.Add(br.ReadUInt16());
                    break;
                case "GNDX":
                    for (int i = 0; i < count; i++)
                        geo.VertexGroups.Add(br.ReadByte());
                    break;
                case "MTGC":
                    for (int i = 0; i < count; i++)
                        geo.MatrixGroups.Add(br.ReadUInt32());
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
                {
                    // UVAS = UV Animation Sets container. 'count' = number of UV sets (typically 1).
                    // In standard WC3 MDX, UVAS contains nested UVBS sub-chunks.
                    // In Alpha 0.5.3 (v1300), UVAS may contain UV data directly (no UVBS wrapper).
                    // Detect by peeking: if next 4 bytes are "UVBS", parse as nested; otherwise read directly.
                    long uvasDataStart = br.BaseStream.Position;
                    bool hasNestedUvbs = false;
                    if (br.BaseStream.Position + 4 <= geoEnd)
                    {
                        long peekPos = br.BaseStream.Position;
                        string peekTag = Encoding.ASCII.GetString(br.ReadBytes(4));
                        br.BaseStream.Position = peekPos; // rewind
                        hasNestedUvbs = (peekTag == "UVBS");
                    }

                    if (hasNestedUvbs)
                    {
                        // Standard format: UVBS sub-chunk(s) follow — let the main loop handle them
                        if (Verbose) Console.WriteLine($"      [UVAS] Nested UVBS detected (count={count})");
                    }
                    else
                    {
                        // Alpha 0.5.3 direct UV data: read nVerts UV pairs
                        int nVerts = geo.Vertices.Count;
                        if (nVerts > 0)
                        {
                            for (int k = 0; k < nVerts; k++)
                                geo.TexCoords.Add(new C2Vector(br.ReadSingle(), br.ReadSingle()));
                            if (Verbose) Console.WriteLine($"      [UVAS] Direct UV data: {nVerts} UVs for {nVerts} verts");
                        }
                    }
                    break;
                }
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
                        if (Verbose)
                        {
                            Console.WriteLine($"      [GEOS#{gIdx}] Unknown tag '{tag}' (0x{(uint)(tag[0])|(uint)(tag[1])<<8|(uint)(tag[2])<<16|(uint)(tag[3])<<24:X8}) count=0x{count:X8} at pos {currentPos}");
                            Console.WriteLine($"      [GEOS#{gIdx}] Hex: {hex}");
                            Console.WriteLine($"      [GEOS#{gIdx}] Asc: {ascii}");
                        }
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
                             if (Verbose) Console.WriteLine($"      [GEOS#{gIdx}] RECOVERY: Skipped {p - currentPos} bytes → '{possibleTag}' at pos {p}");
                             br.BaseStream.Position = p; // Align to new tag
                             recovered = true;
                             break;
                        }
                        br.BaseStream.Position = p + 1;
                    }

                    if (!recovered)
                    {
                        if (Verbose) Console.WriteLine($"      [GEOS#{gIdx}] WARN: Recovery scan failed for tag '{tag}' at {currentPos}");
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

    /// <summary>
    /// Read a Node structure with animation tracks (KGTR, KGRT, KGSC).
    /// Used by BONE and HELP parsers that need animation data.
    /// </summary>
    static (string name, int objectId, int parentId, uint flags,
            MdlAnimTrack<C3Vector>? translation, MdlAnimTrack<C4Quaternion>? rotation, MdlAnimTrack<C3Vector>? scaling)
        ReadNodeWithTracks(BinaryReader br)
    {
        uint nodeSize = br.ReadUInt32();
        long nodeEnd = br.BaseStream.Position - 4 + nodeSize;

        string name = ReadFixedString(br, 0x50);
        int objectId = br.ReadInt32();
        int parentId = br.ReadInt32();
        uint flags = br.ReadUInt32();

        MdlAnimTrack<C3Vector>? translation = null;
        MdlAnimTrack<C4Quaternion>? rotation = null;
        MdlAnimTrack<C3Vector>? scaling = null;

        // Parse animation sub-chunks within the node
        while (br.BaseStream.Position + 8 <= nodeEnd)
        {
            long subPos = br.BaseStream.Position;
            string subTag = ReadTag(br);
            uint subCount = br.ReadUInt32();

            switch (subTag)
            {
                case "KGTR": // Translation track (vec3)
                    translation = ReadVec3Track(br, subCount);
                    break;
                case "KGRT": // Rotation track (quaternion)
                    rotation = ReadQuatTrack(br, subCount);
                    break;
                case "KGSC": // Scaling track (vec3)
                    scaling = ReadVec3Track(br, subCount);
                    break;
                default:
                    // Unknown sub-chunk — skip to end of node
                    br.BaseStream.Position = nodeEnd;
                    return (name, objectId, parentId, flags, translation, rotation, scaling);
            }
        }

        br.BaseStream.Position = nodeEnd;
        return (name, objectId, parentId, flags, translation, rotation, scaling);
    }

    /// <summary>Read a vec3 animation track (KGTR or KGSC)</summary>
    static MdlAnimTrack<C3Vector> ReadVec3Track(BinaryReader br, uint keyCount)
    {
        var track = new MdlAnimTrack<C3Vector>();
        track.InterpolationType = (MdlTrackType)br.ReadUInt32();
        track.GlobalSeqId = br.ReadInt32();

        bool hasTangents = track.InterpolationType == MdlTrackType.Hermite ||
                           track.InterpolationType == MdlTrackType.Bezier;

        for (uint i = 0; i < keyCount; i++)
        {
            var key = new MdlTrackKey<C3Vector>();
            key.Frame = br.ReadInt32();
            key.Value = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            if (hasTangents)
            {
                key.InTan = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                key.OutTan = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            }
            track.Keys.Add(key);
        }

        return track;
    }

    /// <summary>Read a quaternion animation track (KGRT) — uses C4QuaternionCompressed (8 bytes per quat)</summary>
    static MdlAnimTrack<C4Quaternion> ReadQuatTrack(BinaryReader br, uint keyCount)
    {
        var track = new MdlAnimTrack<C4Quaternion>();
        track.InterpolationType = (MdlTrackType)br.ReadUInt32();
        track.GlobalSeqId = br.ReadInt32();

        bool hasTangents = track.InterpolationType == MdlTrackType.Hermite ||
                           track.InterpolationType == MdlTrackType.Bezier;

        for (uint i = 0; i < keyCount; i++)
        {
            var key = new MdlTrackKey<C4Quaternion>();
            key.Frame = br.ReadInt32();
            // KGRT keys are C4QuaternionCompressed (64-bit packed), not float4
            // Linear stride: 0x0C (4 time + 8 compressed quat)
            // Hermite/Bezier stride: 0x1C (4 time + 8*3 compressed quats)
            var compressed = new C4QuaternionCompressed
            {
                Data0 = br.ReadUInt32(),
                Data1 = br.ReadUInt32()
            };
            key.Value = compressed.Decompress();
            if (hasTangents)
            {
                var inTanC = new C4QuaternionCompressed { Data0 = br.ReadUInt32(), Data1 = br.ReadUInt32() };
                var outTanC = new C4QuaternionCompressed { Data0 = br.ReadUInt32(), Data1 = br.ReadUInt32() };
                key.InTan = inTanC.Decompress();
                key.OutTan = outTanC.Decompress();
            }
            track.Keys.Add(key);
        }

        return track;
    }

    /// <summary>Parse PIVT chunk — pivot points for bones/nodes</summary>
    static void ReadPivt(BinaryReader br, uint size, List<C3Vector> pivots)
    {
        if (size % 12 != 0)
            throw new InvalidDataException($"Invalid PIVT size 0x{size:X}: expected multiple of 12.");

        uint count = size / 12; // 3 floats × 4 bytes
        for (uint i = 0; i < count; i++)
            pivots.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
        if (Verbose) Console.WriteLine($"[PIVT] Parsed {pivots.Count} pivot points");
    }

    /// <summary>Parse BONE chunk — skeleton bones with animation tracks</summary>
    static void ReadBone(BinaryReader br, uint chunkSize, List<MdlBone> bones, List<C3Vector> pivots)
    {
        long chunkEnd = br.BaseStream.Position + chunkSize;
        uint count = br.ReadUInt32();

        for (uint i = 0; i < count && br.BaseStream.Position < chunkEnd; i++)
        {
            long boneStart = br.BaseStream.Position;
            try
            {
                var node = ReadNodeWithTracks(br);
                var bone = new MdlBone
                {
                    Name = node.name,
                    ObjectId = node.objectId,
                    ParentId = node.parentId,
                    Flags = node.flags,
                    TranslationTrack = node.translation,
                    RotationTrack = node.rotation,
                    ScalingTrack = node.scaling
                };

                // Bone-specific fields after the node
                if (br.BaseStream.Position + 8 <= chunkEnd)
                {
                    bone.GeosetId = br.ReadInt32();
                    bone.GeosetAnimId = br.ReadInt32();
                }

                bones.Add(bone);
            }
            catch (Exception ex)
            {
                if (Verbose) Console.WriteLine($"[BONE] Failed to parse bone {i}: {ex.Message}");
                // Try to recover by scanning for next bone or end of chunk
                break;
            }
        }

        br.BaseStream.Position = chunkEnd;
        if (Verbose) Console.WriteLine($"[BONE] Parsed {bones.Count} bones");
    }

    /// <summary>Parse HELP chunk — helper nodes (participate in skeleton hierarchy like bones)</summary>
    static void ReadHelp(BinaryReader br, uint chunkSize, List<MdlBone> bones, List<C3Vector> pivots)
    {
        long chunkEnd = br.BaseStream.Position + chunkSize;
        uint count = br.ReadUInt32();

        for (uint i = 0; i < count && br.BaseStream.Position < chunkEnd; i++)
        {
            try
            {
                var node = ReadNodeWithTracks(br);
                var bone = new MdlBone
                {
                    Name = node.name,
                    ObjectId = node.objectId,
                    ParentId = node.parentId,
                    Flags = node.flags,
                    TranslationTrack = node.translation,
                    RotationTrack = node.rotation,
                    ScalingTrack = node.scaling,
                    GeosetId = -1,
                    GeosetAnimId = -1
                };

                bones.Add(bone);
            }
            catch (Exception ex)
            {
                if (Verbose) Console.WriteLine($"[HELP] Failed to parse helper {i}: {ex.Message}");
                break;
            }
        }

        br.BaseStream.Position = chunkEnd;
        if (Verbose) Console.WriteLine($"[HELP] Parsed helpers, total bones now: {bones.Count}");
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

                // Emitter content size (redundant in known files)
                if (emitterEnd - br.BaseStream.Position < 4) { br.BaseStream.Position = emitterEnd; continue; }
                br.ReadUInt32();

                // Ported PRE2 scalar layout from wow-mdx-viewer parser.
                br.ReadInt32(); // EmitterType
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
                emitter.HeadOrTail = br.ReadInt32();
                emitter.TailLength = br.ReadSingle();
                emitter.Time = br.ReadSingle();

                for (int c = 0; c < 3; c++)
                    emitter.SegmentColor[c] = new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());

                emitter.SegmentAlpha[0] = br.ReadByte();
                emitter.SegmentAlpha[1] = br.ReadByte();
                emitter.SegmentAlpha[2] = br.ReadByte();

                emitter.SegmentScaling[0] = br.ReadSingle();
                emitter.SegmentScaling[1] = br.ReadSingle();
                emitter.SegmentScaling[2] = br.ReadSingle();
                br.ReadBytes(12 * 4); // remaining UV animation/scaling floats

                emitter.FilterMode = (ParticleFilterMode)br.ReadInt32();
                emitter.TextureId = br.ReadInt32();
                emitter.PriorityPlane = br.ReadInt32();
                emitter.ReplaceableId = br.ReadUInt32();

                br.ReadBytes(0x104); // GeometryModel
                br.ReadBytes(0x104); // RecursionModel

                br.ReadSingle(); // TwinkleFps
                br.ReadSingle(); // TwinkleOnOff
                br.ReadSingle(); // TwinkleScale min
                br.ReadSingle(); // TwinkleScale max
                br.ReadSingle(); // IvelScale
                br.ReadBytes(6 * 4); // Tumble[6]
                br.ReadSingle(); // Drag
                br.ReadSingle(); // Spin
                br.ReadBytes(3 * 4); // WindVector
                br.ReadSingle(); // WindTime

                for (int j = 0; j < 2; j++)
                {
                    br.ReadSingle(); // FollowSpeed
                    br.ReadSingle(); // FollowScale
                }

                int splineCount = br.ReadInt32();
                if (splineCount < 0)
                    throw new InvalidDataException($"[PRE2] Invalid spline count {splineCount}.");
                br.ReadBytes(splineCount * 12);

                emitter.Squirt = br.ReadInt32();

                // Optional animation sub-chunks at tail.
                while (br.BaseStream.Position + 4 <= emitterEnd)
                {
                    string keyword = ReadTag(br);
                    switch (keyword)
                    {
                        case "KP2S":
                        case "KP2R":
                        case "KP2G":
                        case "KP2W":
                        case "KP2N":
                        case "KVIS":
                        case "KP2E":
                        case "KP2L":
                        case "KPLN":
                        case "KLIF":
                        case "KP2Z":
                            SkipAnimVector(br, 4);
                            break;
                        default:
                            br.BaseStream.Position = emitterEnd;
                            break;
                    }
                }

                emitters.Add(emitter);
            }
            catch (Exception ex)
            {
                if (Verbose) Console.WriteLine($"[PRE2] Failed to parse emitter {i}: {ex.Message}");
            }

            br.BaseStream.Position = emitterEnd;
        }

        br.BaseStream.Position = chunkEnd;
        if (Verbose) Console.WriteLine($"[PRE2] Parsed {emitters.Count} particle emitters");
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

                // Optional animation sub-chunks at tail.
                while (br.BaseStream.Position + 4 <= emitterEnd)
                {
                    string keyword = ReadTag(br);
                    switch (keyword)
                    {
                        case "KVIS":
                        case "KRHA":
                        case "KRHB":
                        case "KRAL":
                            SkipAnimVector(br, 4);
                            break;
                        case "KRTX":
                            SkipAnimVector(br, 4); // INT1 uses int32 values
                            break;
                        case "KRCO":
                            SkipAnimVector(br, 12);
                            break;
                        default:
                            br.BaseStream.Position = emitterEnd;
                            break;
                    }
                }

                emitters.Add(emitter);
            }
            catch (Exception ex)
            {
                if (Verbose) Console.WriteLine($"[RIBB] Failed to parse emitter {i}: {ex.Message}");
            }

            br.BaseStream.Position = emitterEnd;
        }

        br.BaseStream.Position = chunkEnd;
        if (Verbose) Console.WriteLine($"[RIBB] Parsed {emitters.Count} ribbon emitters");
    }

    static void SkipAnimVector(BinaryReader br, int valueByteSize)
    {
        int keyCount = br.ReadInt32();
        int interpolation = br.ReadInt32();
        br.ReadInt32(); // globalSeqId

        if (keyCount < 0)
            throw new InvalidDataException($"Invalid animation key count {keyCount}.");

        bool hasTangents = interpolation == 2 || interpolation == 3;
        for (int i = 0; i < keyCount; i++)
        {
            br.ReadInt32(); // frame
            br.ReadBytes(valueByteSize);
            if (hasTangents)
            {
                br.ReadBytes(valueByteSize);
                br.ReadBytes(valueByteSize);
            }
        }
    }
}
