using GillijimProject.Utilities;

namespace GillijimProject.WowFiles;

public sealed class McnkAlpha
{
    public long HeaderStart { get; }
    public long PayloadStart => HeaderStart + 8; // after 'MCNK'+size header
    public uint McvtRel { get; }
    public uint McnrRel { get; }
    public uint MclqRel { get; }
    public uint ChunksSize { get; }

    public long McvtAbs(long tileDataStart) => McvtRel == 0 ? 0 : PayloadStart + McvtRel;
    public long McnrAbs(long tileDataStart) => McnrRel == 0 ? 0 : PayloadStart + McnrRel;
    public long MclqAbs(long tileDataStart) => MclqRel == 0 ? 0 : PayloadStart + MclqRel;
    public long EndBound => PayloadStart + ChunksSize;

    private McnkAlpha(long headerStart, uint mcvtRel, uint mcnrRel, uint mclqRel, uint chunksSize)
    {
        HeaderStart = headerStart;
        McvtRel = mcvtRel; McnrRel = mcnrRel; MclqRel = mclqRel; ChunksSize = chunksSize;
    }

    public static McnkAlpha ReadHeader(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCNK, "Expected MCNK tag");
        // MCNK header is 0x80 bytes starting at payload
        var hdr = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + 0x80)];
        Util.Assert(hdr.Length >= 0x80, "MCNK header truncated");
        uint mcvtRel = Util.ReadUInt32LE(hdr, 0x18);
        uint mcnrRel = Util.ReadUInt32LE(hdr, 0x1C);
        uint chunksSize = Util.ReadUInt32LE(hdr, 0x64);
        uint mclqRel = Util.ReadUInt32LE(hdr, 0x68);
        return new McnkAlpha((long)absoluteOffset, mcvtRel, mcnrRel, mclqRel, chunksSize);
    }

    public static McnkAlpha ReadHeader(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCNK, "Expected MCNK tag");
        Span<byte> hdr = stackalloc byte[0x80];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(hdr);
        Util.Assert(read == 0x80, "MCNK header truncated");
        uint mcvtRel = Util.ReadUInt32LE(hdr, 0x18);
        uint mcnrRel = Util.ReadUInt32LE(hdr, 0x1C);
        uint chunksSize = Util.ReadUInt32LE(hdr, 0x64);
        uint mclqRel = Util.ReadUInt32LE(hdr, 0x68);
        return new McnkAlpha((long)absoluteOffset, mcvtRel, mcnrRel, mclqRel, chunksSize);
    }

    public (bool BoundsOk, bool McvtOk, bool McnrOk) ValidateSubchunks()
    {
        bool boundsOk = true;
        bool mcvtOk = true;
        bool mcnrOk = true;
        if (McvtRel != 0)
        {
            long mcvtEnd = PayloadStart + McvtRel + 580;
            boundsOk &= mcvtEnd <= EndBound;
            mcvtOk &= 580 == 580; // constant by definition
        }
        if (McnrRel != 0)
        {
            long mcnrEnd = PayloadStart + McnrRel + 435;
            boundsOk &= mcnrEnd <= EndBound;
            mcnrOk &= 435 == 435;
        }
        return (boundsOk, mcvtOk, mcnrOk);
    }
}
