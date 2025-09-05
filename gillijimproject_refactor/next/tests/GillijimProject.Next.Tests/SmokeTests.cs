using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.IO;
using Xunit;

namespace GillijimProject.Next.Tests;

public class SmokeTests
{
    [Fact]
    public void CanInstantiateDomainTypes()
    {
        var wdt = new WdtAlpha("sample.wdt");
        var adt = new AdtAlpha("map_00_00.adt");
        var lk = new AdtLk("map_00_00");
        Assert.NotNull(wdt);
        Assert.NotNull(adt);
        Assert.NotNull(lk);
    }

    [Fact]
    public void ParsesMinimalWdlWithSingleMareTile()
    {
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "WdlSmoke_" + System.Guid.NewGuid());
        System.IO.Directory.CreateDirectory(tempRoot);
        var wdlPath = System.IO.Path.Combine(tempRoot, "TestMap.wdl");

        try
        {
            using (var fs = new System.IO.FileStream(wdlPath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None))
            using (var bw = new System.IO.BinaryWriter(fs, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                // 1) Write one MARE chunk first and remember absolute offset
                long marePos = fs.Position;
                bw.Write(System.Text.Encoding.ASCII.GetBytes("MARE"));
                bw.Write((uint)((17 * 17 + 16 * 16) * 2)); // size = 1090

                // Outer 17x17 = 289 int16
                for (int j = 0; j < 17; j++)
                {
                    for (int i = 0; i < 17; i++)
                    {
                        bw.Write((short)100);
                    }
                }
                // Inner 16x16 = 256 int16
                for (int j = 0; j < 16; j++)
                {
                    for (int i = 0; i < 16; i++)
                    {
                        bw.Write((short)200);
                    }
                }

                // 2) Write MAOF with a single offset entry (y=0,x=0)
                bw.Write(System.Text.Encoding.ASCII.GetBytes("MAOF"));
                bw.Write((uint)4); // size of one uint32 offset
                bw.Write((uint)marePos);
            }

            // Parse and validate
            var wdl = AlphaReader.ParseWdl(wdlPath);
            Assert.NotNull(wdl);
            Assert.NotNull(wdl.Tiles[0, 0]);
            var tile = wdl.Tiles[0, 0]!;
            Assert.Equal(100, tile.Height17[0, 0]);
            Assert.Equal(200, tile.Height16[0, 0]);
        }
        finally
        {
            try { System.IO.Directory.Delete(tempRoot, recursive: true); } catch { }
        }
    }

    [Fact]
    public void ParsesWdlWithReversedFourCCs()
    {
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "WdlSmokeRev_" + System.Guid.NewGuid());
        System.IO.Directory.CreateDirectory(tempRoot);
        var wdlPath = System.IO.Path.Combine(tempRoot, "TestMapRev.wdl");

        try
        {
            using (var fs = new System.IO.FileStream(wdlPath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None))
            using (var bw = new System.IO.BinaryWriter(fs, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                // Write REVM (MVER reversed) version 18
                bw.Write(System.Text.Encoding.ASCII.GetBytes("REVM"));
                bw.Write((uint)4);
                bw.Write(18);

                // Write ERAM (MARE reversed) and remember offset
                long marePos = fs.Position;
                bw.Write(System.Text.Encoding.ASCII.GetBytes("ERAM"));
                bw.Write((uint)((17 * 17 + 16 * 16) * 2));
                for (int j = 0; j < 17; j++)
                    for (int i = 0; i < 17; i++)
                        bw.Write((short)123);
                for (int j = 0; j < 16; j++)
                    for (int i = 0; i < 16; i++)
                        bw.Write((short)234);

                // Write FOAM (MAOF reversed) with a single offset
                bw.Write(System.Text.Encoding.ASCII.GetBytes("FOAM"));
                bw.Write((uint)4);
                bw.Write((uint)marePos);
            }

            var wdl = AlphaReader.ParseWdl(wdlPath);
            Assert.NotNull(wdl);
            Assert.NotNull(wdl.Tiles[0, 0]);
            var tile = wdl.Tiles[0, 0]!;
            Assert.Equal(123, tile.Height17[0, 0]);
            Assert.Equal(234, tile.Height16[0, 0]);
        }
        finally
        {
            try { System.IO.Directory.Delete(tempRoot, recursive: true); } catch { }
        }
    }

    [Fact]
    public void ParsesRealWdlFixtureOrSkip()
    {
        var projectDir = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.AppContext.BaseDirectory, "..", "..", ".."));
        var testDataRoot = System.IO.Path.GetFullPath(System.IO.Path.Combine(projectDir, "..", "test_data"));
        if (!System.IO.Directory.Exists(testDataRoot))
        {
            System.Console.WriteLine("[skip] test_data not found; skipping WDL fixture parsing test.");
            return;
        }

        string? wdlPath = null;
        try
        {
            foreach (var p in System.IO.Directory.EnumerateFiles(testDataRoot, "*.wdl", System.IO.SearchOption.AllDirectories))
            {
                wdlPath = p;
                break;
            }
        }
        catch { /* ignore */ }

        if (string.IsNullOrEmpty(wdlPath) || !System.IO.File.Exists(wdlPath))
        {
            System.Console.WriteLine("[skip] no .wdl fixture found under test_data; skipping.");
            return;
        }

        var wdl = AlphaReader.ParseWdl(wdlPath);
        Assert.NotNull(wdl);

        bool any = false;
        for (int y = 0; y < 64 && !any; y++)
        {
            for (int x = 0; x < 64 && !any; x++)
            {
                if (wdl.Tiles[y, x] is not null) any = true;
            }
        }

        Assert.True(any, "At least one tile should be present in a real WDL.");
    }

    [Fact]
    public void ParsesWdlWithMahoHolesNormal()
    {
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "WdlMaho_" + System.Guid.NewGuid());
        System.IO.Directory.CreateDirectory(tempRoot);
        var wdlPath = System.IO.Path.Combine(tempRoot, "TestMapMaho.wdl");

        try
        {
            using (var fs = new System.IO.FileStream(wdlPath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None))
            using (var bw = new System.IO.BinaryWriter(fs, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                // Write MARE and remember offset
                long marePos = fs.Position;
                bw.Write(System.Text.Encoding.ASCII.GetBytes("MARE"));
                bw.Write((uint)((17 * 17 + 16 * 16) * 2)); // 1090 bytes
                for (int j = 0; j < 17; j++) for (int i = 0; i < 17; i++) bw.Write((short)10);
                for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) bw.Write((short)20);

                // Immediately following MAHO (16 x ushort)
                bw.Write(System.Text.Encoding.ASCII.GetBytes("MAHO"));
                bw.Write((uint)(16 * 2));
                var rows = new ushort[16];
                rows[3] = (ushort)(1 << 7); // set bit at (y=3, x=7)
                foreach (var r in rows) bw.Write(r);

                // MAOF with single offset
                bw.Write(System.Text.Encoding.ASCII.GetBytes("MAOF"));
                bw.Write((uint)4);
                bw.Write((uint)marePos);
            }

            var wdl = AlphaReader.ParseWdl(wdlPath);
            Assert.NotNull(wdl);
            var tile = Assert.IsType<WdlTile>(wdl.Tiles[0, 0]);
            Assert.True(tile.IsHole(3, 7));
            Assert.False(tile.IsHole(0, 0));
        }
        finally
        {
            try { System.IO.Directory.Delete(tempRoot, recursive: true); } catch { }
        }
    }

    [Fact]
    public void ParsesWdlWithMahoHolesReversedFourCCs()
    {
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "WdlMahoRev_" + System.Guid.NewGuid());
        System.IO.Directory.CreateDirectory(tempRoot);
        var wdlPath = System.IO.Path.Combine(tempRoot, "TestMapMahoRev.wdl");

        try
        {
            using (var fs = new System.IO.FileStream(wdlPath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None))
            using (var bw = new System.IO.BinaryWriter(fs, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                // Optional REVM (MVER reversed)
                bw.Write(System.Text.Encoding.ASCII.GetBytes("REVM"));
                bw.Write((uint)4);
                bw.Write(18);

                // ERAM (MARE reversed)
                long marePos = fs.Position;
                bw.Write(System.Text.Encoding.ASCII.GetBytes("ERAM"));
                bw.Write((uint)((17 * 17 + 16 * 16) * 2));
                for (int j = 0; j < 17; j++) for (int i = 0; i < 17; i++) bw.Write((short)11);
                for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) bw.Write((short)22);

                // OHAM (MAHO reversed)
                bw.Write(System.Text.Encoding.ASCII.GetBytes("OHAM"));
                bw.Write((uint)(16 * 2));
                var rows = new ushort[16];
                rows[5] = (ushort)(1 << 9); // set bit at (y=5, x=9)
                foreach (var r in rows) bw.Write(r);

                // FOAM (MAOF reversed)
                bw.Write(System.Text.Encoding.ASCII.GetBytes("FOAM"));
                bw.Write((uint)4);
                bw.Write((uint)marePos);
            }

            var wdl = AlphaReader.ParseWdl(wdlPath);
            Assert.NotNull(wdl);
            var tile = Assert.IsType<WdlTile>(wdl.Tiles[0, 0]);
            Assert.True(tile.IsHole(5, 9));
            Assert.False(tile.IsHole(2, 2));
            Assert.Equal(11, tile.Height17[0, 0]);
            Assert.Equal(22, tile.Height16[0, 0]);
        }
        finally
        {
            try { System.IO.Directory.Delete(tempRoot, recursive: true); } catch { }
        }
    }

    [Fact]
    public void ParsesWdlWithMissingMahoDefaultsToZero()
    {
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "WdlNoMaho_" + System.Guid.NewGuid());
        System.IO.Directory.CreateDirectory(tempRoot);
        var wdlPath = System.IO.Path.Combine(tempRoot, "TestMapNoMaho.wdl");

        try
        {
            using (var fs = new System.IO.FileStream(wdlPath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None))
            using (var bw = new System.IO.BinaryWriter(fs, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                // MARE payload with heights
                long marePos = fs.Position;
                bw.Write(System.Text.Encoding.ASCII.GetBytes("MARE"));
                uint expected = (uint)((17 * 17 + 16 * 16) * 2);
                bw.Write(expected);
                for (int j = 0; j < 17; j++) for (int i = 0; i < 17; i++) bw.Write((short)7);
                for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) bw.Write((short)9);
                // No MAHO emitted

                // MAOF with single offset
                bw.Write(System.Text.Encoding.ASCII.GetBytes("MAOF"));
                bw.Write((uint)4);
                bw.Write((uint)marePos);
            }

            var wdl = AlphaReader.ParseWdl(wdlPath);
            Assert.NotNull(wdl);
            var tile = Assert.IsType<WdlTile>(wdl.Tiles[0, 0]);
            // All holes should be false by default
            Assert.False(tile.IsHole(0, 0));
            Assert.False(tile.IsHole(15, 15));
        }
        finally
        {
            try { System.IO.Directory.Delete(tempRoot, recursive: true); } catch { }
        }
    }

    [Fact]
    public void ParsesWdlWithOddSizedChunksAndPadding()
    {
        var tempRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "WdlOdd_" + System.Guid.NewGuid());
        System.IO.Directory.CreateDirectory(tempRoot);
        var wdlPath = System.IO.Path.Combine(tempRoot, "TestMapOdd.wdl");

        try
        {
            using (var fs = new System.IO.FileStream(wdlPath, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None))
            using (var bw = new System.IO.BinaryWriter(fs, System.Text.Encoding.ASCII, leaveOpen: true))
            {
                // ERAM (reversed MARE) with odd size
                long marePos = fs.Position;
                bw.Write(System.Text.Encoding.ASCII.GetBytes("ERAM"));
                uint expected = (uint)((17 * 17 + 16 * 16) * 2); // 1090
                uint oddSize = expected + 1; // 1091
                bw.Write(oddSize);
                // Write the 1090 bytes of heights
                for (int j = 0; j < 17; j++) for (int i = 0; i < 17; i++) bw.Write((short)12);
                for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) bw.Write((short)34);
                // One extra payload byte to reach 1091
                bw.Write((byte)0xCD);
                // Writer pad to align (since size is odd), so that next chunk starts on even boundary
                bw.Write((byte)0x00);

                // OHAM (reversed MAHO) with odd size (33)
                bw.Write(System.Text.Encoding.ASCII.GetBytes("OHAM"));
                uint mahoOdd = 33u;
                bw.Write(mahoOdd);
                // 32 bytes of 16 rows
                var rows = new ushort[16];
                rows[1] = (ushort)(1 << 2); // (1,2)
                foreach (var r in rows) bw.Write(r);
                // extra payload byte to make 33
                bw.Write((byte)0xAB);
                // align pad for odd size
                bw.Write((byte)0x00);

                // FOAM (reversed MAOF) with one offset
                bw.Write(System.Text.Encoding.ASCII.GetBytes("FOAM"));
                bw.Write((uint)4);
                bw.Write((uint)marePos);
            }

            var wdl = AlphaReader.ParseWdl(wdlPath);
            Assert.NotNull(wdl);
            var tile = Assert.IsType<WdlTile>(wdl.Tiles[0, 0]);
            Assert.Equal(12, tile.Height17[0, 0]);
            Assert.Equal(34, tile.Height16[0, 0]);
            Assert.True(tile.IsHole(1, 2));
        }
        finally
        {
            try { System.IO.Directory.Delete(tempRoot, recursive: true); } catch { }
        }
    }
}
