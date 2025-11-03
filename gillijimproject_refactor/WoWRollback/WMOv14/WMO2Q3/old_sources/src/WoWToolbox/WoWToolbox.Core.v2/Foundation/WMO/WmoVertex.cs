using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WoWToolbox.Core.v2.Foundation.WMO
{
    public struct WmoVertex
    {
        public Vector3 Position { get; set; }
        public Vector3 Normal { get; set; }
        public Vector2 UV { get; set; }

        public static List<WmoVertex> FromV14(byte[] movtData)
        {
            var vertices = new List<WmoVertex>();
            using var stream = new MemoryStream(movtData);
            using var reader = new BinaryReader(stream);
            int count = movtData.Length / 12;

            for (int i = 0; i < count; i++)
            {
                var vertex = new WmoVertex
                {
                    Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle())
                };
                vertices.Add(vertex);
            }

            return vertices;
        }
    }
}
