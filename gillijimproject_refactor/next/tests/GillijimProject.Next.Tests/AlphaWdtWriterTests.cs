using System;
using System.IO;
using System.Text;
using Xunit;
using GillijimProject.Next.Core.IO;

namespace GillijimProject.Next.Tests;

public class AlphaWdtWriterTests
{
    [Fact]
    public void GeneratesWdtWithEmbeddedOffsets()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), "AlphaWdtWriterTests_" + Guid.NewGuid());
        Directory.CreateDirectory(tempRoot);
        try
        {
            var map = "TestMap";
            // Create two synthetic ADTs with 'RDHM' marker at different offsets
            var adt1 = new byte[32];
            Encoding.ASCII.GetBytes("RDHM").CopyTo(adt1, 0); // MHDR at 0
            var adt2 = new byte[40];
            Encoding.ASCII.GetBytes("RDHM").CopyTo(adt2, 8); // MHDR at 8

            File.WriteAllBytes(Path.Combine(tempRoot, "TestMap_1_2.adt"), adt1);
            File.WriteAllBytes(Path.Combine(tempRoot, "TestMap_3_4.adt"), adt2);

            var outWdt = Path.Combine(tempRoot, "TestMap.wdt");
            var res = AlphaWdtWriter.GenerateFromLkAdts(map, tempRoot, outWdt, new AlphaWdtWriter.Options());

            Assert.True(File.Exists(outWdt));
            Assert.Equal(4096, res.TilesTotal);
            Assert.Equal(2, res.TilesEmbedded);

            using var fs = File.OpenRead(outWdt);
            // Read MVER
            ReadChunk(fs, out var fourcc, out var size, out var dataStart);
            Assert.Equal("REVM", fourcc); // reversed 'MVER'
            Assert.Equal(4, size);

            // MPHD
            ReadChunk(fs, out fourcc, out size, out dataStart);
            Assert.Equal("DHPM", fourcc);
            Assert.Equal(16, size);

            // MAIN
            long mainHeader = fs.Position;
            ReadChunk(fs, out fourcc, out size, out long mainDataStart);
            Assert.Equal("NIAM", fourcc);
            Assert.Equal(4096 * 16, size);

            // MDNM present by default
            ReadChunk(fs, out fourcc, out size, out dataStart);
            Assert.Equal("MNDM", fourcc);
            Assert.Equal(0, size);

            // Validate two MAIN entries are populated
            fs.Seek(mainDataStart + ( (2 * 64 + 1) * 16), SeekOrigin.Begin); // y=2,x=1 -> idx 129
            var off129 = ReadInt(fs);
            var size129 = ReadInt(fs);
            Assert.True(off129 > 0);
            Assert.True(size129 >= adt1.Length);

            fs.Seek(mainDataStart + ( (4 * 64 + 3) * 16), SeekOrigin.Begin); // y=4,x=3 -> idx 259
            var off259 = ReadInt(fs);
            var size259 = ReadInt(fs);
            Assert.True(off259 > 0);
            Assert.True(size259 >= adt2.Length);

            // Verify markers at absolute offsets
            fs.Seek(off129, SeekOrigin.Begin);
            var tag1 = ReadFourCC(fs);
            Assert.True(tag1 is "RDHM" or "MHDR");

            fs.Seek(off259, SeekOrigin.Begin);
            var tag2 = ReadFourCC(fs);
            Assert.True(tag2 is "RDHM" or "MHDR");
        }
        finally
        {
            try { Directory.Delete(tempRoot, recursive: true); } catch { }
        }
    }

    private static void ReadChunk(FileStream fs, out string reversedFourCC, out int size, out long dataStart)
    {
        var hdr = new byte[8];
        if (fs.Read(hdr, 0, 8) != 8) throw new EndOfStreamException();
        reversedFourCC = Encoding.ASCII.GetString(hdr, 0, 4);
        size = BitConverter.ToInt32(hdr, 4);
        dataStart = fs.Position;
        fs.Seek(size + (size % 2 == 1 ? 1 : 0), SeekOrigin.Current);
    }

    private static int ReadInt(FileStream fs)
    {
        Span<byte> b = stackalloc byte[4];
        if (fs.Read(b) != 4) throw new EndOfStreamException();
        return BitConverter.ToInt32(b);
    }

    private static string ReadFourCC(FileStream fs)
    {
        Span<byte> b = stackalloc byte[4];
        if (fs.Read(b) != 4) throw new EndOfStreamException();
        return Encoding.ASCII.GetString(b);
    }
}
