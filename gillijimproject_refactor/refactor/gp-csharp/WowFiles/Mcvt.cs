using GillijimProject.Utilities;

namespace GillijimProject.WowFiles;

public sealed class McvtAlphaReader
{
    public const int ExpectedSize = 145 * 4; // 145 float values
    
    private readonly float[] _heights;
    
    private McvtAlphaReader(float[] heights)
    {
        _heights = heights;
    }
    
    public static McvtAlphaReader ReadFrom(Stream s, long absoluteOffset)
    {
        s.Seek(absoluteOffset, SeekOrigin.Begin);
        
        // Read chunk header
        Span<byte> header = stackalloc byte[8];
        int read = s.Read(header);
        Util.Assert(read == 8, "Failed to read MCVT header");
        
        uint tag = Util.ReadUInt32LE(header, 0);
        uint size = Util.ReadUInt32LE(header, 4);
        
        Util.Assert(tag == Tags.MCVT, "Expected MCVT tag");
        Util.Assert(size == ExpectedSize, $"Expected MCVT size {ExpectedSize}, got {size}");
        
        // Read height data
        var buffer = new byte[ExpectedSize];
        read = s.Read(buffer);
        Util.Assert(read == ExpectedSize, "Failed to read MCVT data");
        
        var heights = new float[145];
        for (int i = 0; i < 145; i++)
        {
            heights[i] = BitConverter.ToSingle(buffer, i * 4);
        }
        
        return new McvtAlphaReader(heights);
    }
    
    public float[] GetHeights() => (float[])_heights.Clone();
}
