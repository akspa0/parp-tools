using GillijimProject.Utilities;

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
}
