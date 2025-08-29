// docs/AlphaWDTReader/snippets/lit_reader.cs
// Purpose: Minimal reader for legacy LIT files (Alpha-era lights) to assist analysis.
// Output a list of records suitable for CSV export.

using System;
using System.Collections.Generic;
using System.IO;

namespace Snippets
{
    public sealed class LitRecord
    {
        public int LightId;
        public float X, Y, Z;
        public byte R, G, B;
        public float Radius;
    }

    public static class LitReader
    {
        // Very conservative parser: attempts to read fixed-size records.
        // Adjust sizes/fields as more structure is confirmed from real files.
        public static List<LitRecord> Read(string path)
        {
            var list = new List<LitRecord>();
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            // Heuristic: try to read until EOF with a simple layout
            // [i32 id][f32 x][f32 y][f32 z][u8 r][u8 g][u8 b][u8 pad][f32 radius]
            while (fs.Position + 4 + 4*3 + 4 + 4 <= fs.Length)
            {
                long start = fs.Position;
                try
                {
                    int id = br.ReadInt32();
                    float x = br.ReadSingle();
                    float y = br.ReadSingle();
                    float z = br.ReadSingle();
                    byte r = br.ReadByte();
                    byte g = br.ReadByte();
                    byte b = br.ReadByte();
                    br.ReadByte(); // pad
                    float radius = br.ReadSingle();

                    list.Add(new LitRecord { LightId = id, X = x, Y = y, Z = z, R = r, G = g, B = b, Radius = radius });
                }
                catch
                {
                    // If parsing fails, step forward one byte to resync and continue.
                    fs.Position = start + 1;
                }
            }
            return list;
        }
    }
}
