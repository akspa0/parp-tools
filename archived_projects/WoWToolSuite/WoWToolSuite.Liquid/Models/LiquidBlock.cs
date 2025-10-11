using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;

namespace WowToolSuite.Liquid.Models
{
    public class LiquidBlock
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public Vector2 Coord { get; set; }
        public ushort[] Data { get; set; } = new ushort[80];
        public float[][] Heights { get; set; } = new float[4][] { new float[4], new float[4], new float[4], new float[4] };
        public string? SourceFile { get; set; }

        public float GlobalX => Coord.X;
        public float GlobalY => Coord.Y;
        public float GlobalZ => Vertices.Count > 0 ? Vertices[0].Z : 0;
        public float MinHeight => Vertices.Count > 0 ? Vertices.Min(v => v.Z) : 0;
        public float MaxHeight => Vertices.Count > 0 ? Vertices.Max(v => v.Z) : 0;
        public ushort LiquidType { get; set; }

        public LiquidBlock()
        {
        }

        public LiquidBlock(float x, float y, float z, float minHeight, float maxHeight, int liquidType)
        {
            Coord = new Vector2(x, y);
            LiquidType = (ushort)liquidType;
            // Initialize vertices with a default value
            for (int i = 0; i < 16; i++)
            {
                Vertices.Add(new Vector3(x, y, z));
            }
            // Initialize heights
            for (int i = 0; i < 4; i++)
            {
                Heights[i] = new float[4];
                for (int j = 0; j < 4; j++)
                {
                    Heights[i][j] = z;
                }
            }
        }
    }
} 