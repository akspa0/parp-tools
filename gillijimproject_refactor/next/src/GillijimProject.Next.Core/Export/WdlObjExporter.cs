using System;
using System.Globalization;
using System.IO;
using System.Text;
using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Export;

public static class WdlObjExporter
{
    private const double CellWorldSize = 533.3333333333 / 16.0; // yards per height cell in ADT terms
    public sealed record ExportOptions(
        double Scale = 1.0,
        bool SkipHoles = true,
        bool NormalizeWorld = true,
        double HeightScale = 1.0
    );

    public sealed record ExportStats(
        int TilesExported,
        int VerticesWritten,
        int FacesWritten
    );

    public static ExportStats ExportPerTile(Wdl wdl, string tilesDir, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        Directory.CreateDirectory(tilesDir);

        int tiles = 0, verts = 0, faces = 0;
        double xyScale = (options.NormalizeWorld ? CellWorldSize : 1.0) * options.Scale;
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var t = wdl.Tiles[y, x];
                if (t is null) continue;

                string path = Path.Combine(tilesDir, $"tile_{x}_{y}.obj");
                using var sw = new StreamWriter(File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read), Encoding.UTF8);
                var s = WriteTileObj(sw, t, baseX: 0, baseZ: 0, xyScale, options.HeightScale, options.SkipHoles, name: $"tile_{x}_{y}");
                tiles++;
                verts += s.VerticesWritten;
                faces += s.FacesWritten;
            }
        }
        return new ExportStats(tiles, verts, faces);
    }

    public static ExportStats ExportMerged(Wdl wdl, string mergedObjPath, ExportOptions? options = null)
    {
        options ??= new ExportOptions();
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(mergedObjPath)) ?? ".");

        int tiles = 0, verts = 0, faces = 0;
        using var sw = new StreamWriter(File.Open(mergedObjPath, FileMode.Create, FileAccess.Write, FileShare.Read), Encoding.UTF8);
        sw.WriteLine("# merged WDL mesh");

        int vertexOffset = 0;
        double xyScale = (options.NormalizeWorld ? CellWorldSize : 1.0) * options.Scale;
        for (int ty = 0; ty < 64; ty++)
        {
            for (int tx = 0; tx < 64; tx++)
            {
                var t = wdl.Tiles[ty, tx];
                if (t is null) continue;

                // Offset each tile in X/Z by 16 cells * xyScale to avoid overlap
                double baseX = tx * 16 * xyScale;
                double baseZ = ty * 16 * xyScale;

                var s = WriteTileObj(sw, t, baseX, baseZ, xyScale, options.HeightScale, options.SkipHoles, name: $"tile_{tx}_{ty}", vertexOffset: vertexOffset);
                tiles++;
                verts += s.VerticesWritten;
                faces += s.FacesWritten;
                vertexOffset += s.VerticesWritten;
            }
        }

        return new ExportStats(tiles, verts, faces);
    }

    private static ExportStats WriteTileObj(TextWriter sw, WdlTile tile, double baseX, double baseZ, double xyScale, double heightScale, bool skipHoles, string name, int vertexOffset = 0)
    {
        int vertsBefore = 0, faces = 0;
        sw.WriteLine($"o {name}");

        // Vertices: 17x17 grid, OBJ with X/Z swapped: X=j, Z=i, Y=height
        for (int j = 0; j <= 16; j++)
        {
            for (int i = 0; i <= 16; i++)
            {
                double x = baseZ + j * xyScale;
                double y = tile.Height17[j, i] * heightScale;
                double z = baseX + i * xyScale;
                sw.WriteLine(FormattableString.Invariant($"v {x:R} {y:R} {z:R}"));
                vertsBefore++;
            }
        }

        // Faces: 16x16 cells, two triangles per cell
        // Indexing: local index idx(i,j) = j*17 + i + 1 + vertexOffset
        static int Idx(int i, int j) => j * 17 + i + 1;
        for (int j = 0; j < 16; j++)
        {
            for (int i = 0; i < 16; i++)
            {
                if (skipHoles && tile.IsHole(j, i)) continue;
                int v00 = vertexOffset + Idx(i, j);
                int v10 = vertexOffset + Idx(i + 1, j);
                int v01 = vertexOffset + Idx(i, j + 1);
                int v11 = vertexOffset + Idx(i + 1, j + 1);
                // Triangle 1: v00, v10, v11
                sw.WriteLine($"f {v00} {v10} {v11}");
                // Triangle 2: v00, v11, v01
                sw.WriteLine($"f {v00} {v11} {v01}");
                faces += 2;
            }
        }

        return new ExportStats(TilesExported: 1, VerticesWritten: vertsBefore, FacesWritten: faces);
    }
}
