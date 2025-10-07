using System.IO;

namespace AlphaWDTReader.ReferencePort;

public readonly struct McnkHeader
{
    public const int Size = 0x80; // Alpha MCNK header size in bytes

    public readonly int McvtRel;   // at 0x18
    public readonly int McnrRel;   // at 0x1C
    public readonly int ChunksSize; // at 0x64
    public readonly int MclqRel;   // at 0x68 (Alpha liquids region start, if present)

    private McnkHeader(int mcvtRel, int mcnrRel, int chunksSize, int mclqRel)
    {
        McvtRel = mcvtRel;
        McnrRel = mcnrRel;
        ChunksSize = chunksSize;
        MclqRel = mclqRel;
    }

    public static bool TryRead(BinaryReader br, out McnkHeader header)
    {
        header = default;
        var bytes = br.ReadBytes(Size);
        if (bytes.Length != Size) return false;
        int mcvtRel = BitConverter.ToInt32(bytes, 0x18);
        int mcnrRel = BitConverter.ToInt32(bytes, 0x1C);
        int chunksSize = BitConverter.ToInt32(bytes, 0x64);
        int mclqRel = BitConverter.ToInt32(bytes, 0x68);
        header = new McnkHeader(mcvtRel, mcnrRel, chunksSize, mclqRel);
        return true;
    }
}
