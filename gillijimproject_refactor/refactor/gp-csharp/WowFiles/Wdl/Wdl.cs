using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Wdl;

public sealed class WdlAlpha
{
    public ReadOnlyMemory<byte> RawData { get; }
    public uint Size { get; }
    
    private WdlAlpha(ReadOnlyMemory<byte> rawData, uint size)
    {
        RawData = rawData;
        Size = size;
    }
    
    public static WdlAlpha Load(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        var buffer = new byte[fs.Length];
        int read = fs.Read(buffer);
        Util.Assert(read == fs.Length, $"Failed to read WDL file, expected {fs.Length} bytes");
        
        return new WdlAlpha(buffer, (uint)fs.Length);
    }
    
    public static WdlAlpha Parse(Stream s, long absoluteOffset, uint size)
    {
        var buffer = new byte[size];
        s.Seek(absoluteOffset, SeekOrigin.Begin);
        int read = s.Read(buffer);
        Util.Assert(read == size, $"Failed to read WDL data, expected {size} bytes");
        
        return new WdlAlpha(buffer, size);
    }
}
