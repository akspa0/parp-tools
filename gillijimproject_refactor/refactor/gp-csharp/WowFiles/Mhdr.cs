using GillijimProject.Utilities;

namespace GillijimProject.WowFiles;

public sealed class MhdrAlpha
{
    public uint McinRelOffset { get; }

    private MhdrAlpha(uint mcinRel) => McinRelOffset = mcinRel;

    public static MhdrAlpha Parse(ReadOnlySpan<byte> data, long payloadOffset)
    {
        uint mcinRel = Util.ReadUInt32LE(data, (int)payloadOffset);
        return new MhdrAlpha(mcinRel);
    }

    public static MhdrAlpha Parse(Stream s, long payloadOffset)
    {
        Span<byte> buf = stackalloc byte[4];
        s.Seek(payloadOffset, SeekOrigin.Begin);
        int read = s.Read(buf);
        Util.Assert(read == 4, "Failed to read MHDR.mcinRel");
        uint mcinRel = Util.ReadUInt32LE(buf, 0);
        return new MhdrAlpha(mcinRel);
    }
}
