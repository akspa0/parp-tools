namespace AlphaWDTReader.IO;

public static class AlphaFourCC
{
    public const uint MVER = 0x5245564Du; // 'MVER'
    public const uint MPHD = 0x4448504Du; // 'MPHD'
    public const uint MAIN = 0x4E49414Du; // 'MAIN'
    public const uint MDNM = 0x4D4E444Du; // 'MDNM'
    public const uint MONM = 0x4D4E4F4Du; // 'MONM'
    public const uint MODF = 0x46444F4Du; // 'MODF'

    // Per-tile/MCNK related
    public const uint MHDR = 0x5244484Du; // 'MHDR'
    public const uint MCIN = 0x4E49434Du; // 'MCIN'
    public const uint MCNK = 0x4B4E434Du; // 'MCNK'
    public const uint MCVT = 0x5456434Du; // 'MCVT'
    public const uint MCNR = 0x524E434Du; // 'MCNR'
    public const uint MCLQ = 0x514C434Du; // 'MCLQ'

    public static uint ReadFourCC(BinaryReader br)
    {
        var bytes = br.ReadBytes(4);
        if (bytes.Length < 4) throw new EndOfStreamException();
        return BitConverter.ToUInt32(bytes, 0);
    }

    public static bool Matches(uint got, uint tag)
    {
        return got == tag || Reverse(got) == tag;
    }

    public static uint Reverse(uint v)
    {
        // swap endianness of 4 bytes
        return ((v & 0x000000FFu) << 24) |
               ((v & 0x0000FF00u) << 8)  |
               ((v & 0x00FF0000u) >> 8)  |
               ((v & 0xFF000000u) >> 24);
    }

    public static string ToString(uint fourcc)
    {
        var b = BitConverter.GetBytes(fourcc);
        return System.Text.Encoding.ASCII.GetString(b);
    }
}
