using GillijimProject.Utilities;
using GillijimProject.WowFiles.Terrain;
using GillijimProject.WowFiles.Objects;

namespace GillijimProject.WowFiles;

public sealed class Wdt
{
    private readonly FileStream _fs;
    private readonly ChunkHeader _main;
    public IReadOnlyList<MainAlpha.Entry> MainEntries { get; }

    private Wdt(FileStream fs, ChunkHeader main, List<MainAlpha.Entry> entries)
    { _fs = fs; _main = main; MainEntries = entries; }

    public static Wdt Load(string path)
    {
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        // Locate MAIN by scanning inside Alpha container
        var main = WowChunkedFormat.FindMain(fs);
        var parsed = MainAlpha.Parse(fs, main);
        return new Wdt(fs, main, parsed.Entries.ToList());
    }

    public MhdrAlpha GetMhdrFor(MainAlpha.Entry entry)
    {
        Util.Assert(entry.HasMhdr, "Entry has no MHDR");
        long mhdrHeader = entry.MhdrOffset; // MhdrOffset is absolute in Alpha format
        
        // Validate offset is within file bounds
        if (mhdrHeader < 0 || mhdrHeader + 8 > _fs.Length)
        {
            throw new InvalidDataException($"MHDR offset 0x{mhdrHeader:X8} is out of bounds (file size: {_fs.Length:N0})");
        }
        
        var mhdrChunk = Chunk.ReadHeader(_fs, mhdrHeader);
        Util.Assert(mhdrChunk.Tag == Tags.MHDR, "Expected MHDR at MAIN entry offset");
        return MhdrAlpha.Parse(_fs, mhdrChunk.PayloadOffset);
    }

    // Contextual helper to read MCIN from a specific MAIN entry
    public McinAlpha GetMcinFor(MainAlpha.Entry entry, out long tileStart)
    {
        Util.Assert(entry.HasMhdr, "Entry has no MHDR");
        long mhdrHeader = entry.MhdrOffset; // MhdrOffset is absolute in Alpha format
        
        // Validate offset is within file bounds
        if (mhdrHeader < 0 || mhdrHeader + 8 > _fs.Length)
        {
            throw new InvalidDataException($"MHDR offset 0x{mhdrHeader:X8} is out of bounds (file size: {_fs.Length:N0})");
        }
        
        var mhdrChunk = Chunk.ReadHeader(_fs, mhdrHeader);
        Util.Assert(mhdrChunk.Tag == Tags.MHDR, "Expected MHDR at MAIN entry offset");
        var mhdr = MhdrAlpha.Parse(_fs, mhdrChunk.PayloadOffset);

        long mcinHeader = mhdrChunk.PayloadOffset + mhdr.McinRelOffset;
        var mcinChunk = Chunk.ReadHeader(_fs, mcinHeader);
        Util.Assert(mcinChunk.Tag == Tags.MCIN, "Expected MCIN via MHDR rel offset");

        tileStart = entry.MhdrOffset; // MhdrOffset is absolute, use as tileStart base
        var mcin = McinAlpha.Parse(_fs, mcinChunk.PayloadOffset, tileStart);
        return mcin;
    }

    public McinAlpha GetMcinFor(MhdrAlpha mhdr, long mhdrPayloadBase, out long tileStart)
    {
        long mcinHeader = mhdrPayloadBase + mhdr.McinRelOffset;
        var mcinChunk = Chunk.ReadHeader(_fs, mcinHeader);
        Util.Assert(mcinChunk.Tag == Tags.MCIN, "Expected MCIN via MHDR rel offset");
        tileStart = mhdrPayloadBase - 8; // MHDR header begins 8 bytes before payload
        return McinAlpha.Parse(_fs, mcinChunk.PayloadOffset, tileStart);
    }

    public McnkAlpha ReadMcnkHeader(long absoluteOffset)
    {
        return McnkAlpha.ReadHeader(_fs, absoluteOffset);
    }
    
    public McvtAlphaReader ReadMcvt(long absoluteOffset)
    {
        return McvtAlphaReader.ReadFrom(_fs, absoluteOffset);
    }

    // New parser methods for integrated chunk types
    
    /// <summary>
    /// Parse MCLY (Alpha layer data) chunk at the specified offset
    /// </summary>
    public MclyAlpha ReadMcly(long absoluteOffset)
    {
        return MclyAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MTEX (texture filenames) chunk at the specified offset
    /// </summary>
    public MtexAlpha ReadMtex(long absoluteOffset)
    {
        return MtexAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MODF (WMO placement) chunk at the specified offset
    /// </summary>
    public ModfAlpha ReadModf(long absoluteOffset)
    {
        return ModfAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MDDF (doodad placement) chunk at the specified offset
    /// </summary>
    public MddfAlpha ReadMddf(long absoluteOffset)
    {
        return MddfAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MCRF (cross-references) chunk at the specified offset
    /// </summary>
    public McrfAlpha ReadMcrf(long absoluteOffset)
    {
        return McrfAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MMID (model indices) chunk at the specified offset
    /// </summary>
    public MmidAlpha ReadMmid(long absoluteOffset)
    {
        return MmidAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MDNM (doodad filenames) chunk at the specified offset
    /// </summary>
    public MdnmAlpha ReadMdnm(long absoluteOffset)
    {
        return MdnmAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MONM (WMO filenames) chunk at the specified offset
    /// </summary>
    public MonmAlpha ReadMonm(long absoluteOffset)
    {
        return MonmAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MMDX (M2 model filenames) chunk at the specified offset
    /// </summary>
    public MmdxAlpha ReadMmdx(long absoluteOffset)
    {
        return MmdxAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Parse MWMO (WMO filenames) chunk at the specified offset
    /// </summary>
    public MwmoAlpha ReadMwmo(long absoluteOffset)
    {
        return MwmoAlpha.Parse(_fs, absoluteOffset);
    }
    
    /// <summary>
    /// Scan for and parse all chunks of a specific type in the WDT file
    /// </summary>
    public List<T> FindAndParseChunks<T>(uint chunkTag, Func<long, T> parser)
    {
        var results = new List<T>();
        var buffer = new byte[8];
        long position = 0;
        
        _fs.Seek(0, SeekOrigin.Begin);
        
        while (position < _fs.Length - 8)
        {
            _fs.Seek(position, SeekOrigin.Begin);
            int read = _fs.Read(buffer, 0, 8);
            if (read < 8) break;
            
            uint tag = BitConverter.ToUInt32(buffer, 0);
            uint size = BitConverter.ToUInt32(buffer, 4);
            
            if (tag == chunkTag && size > 0 && size < _fs.Length - position)
            {
                try
                {
                    var parsed = parser(position);
                    results.Add(parsed);
                    position += 8 + size;
                    
                    // Align to 4-byte boundary
                    if (position % 4 != 0)
                        position += 4 - (position % 4);
                }
                catch
                {
                    position++;
                }
            }
            else
            {
                position++;
            }
        }
        
        return results;
    }
}
