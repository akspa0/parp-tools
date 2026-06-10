using System.Collections.Generic;
using System.IO;

namespace WoWToolbox.Core.v2.Foundation.WMO
{
    public struct WmoFace
    {
        public ushort A { get; set; }
        public ushort B { get; set; }
        public ushort C { get; set; }

        public static List<WmoFace> FromV14(byte[] moviData)
        {
            var faces = new List<WmoFace>();
            using var stream = new MemoryStream(moviData);
            using var reader = new BinaryReader(stream);
            int count = moviData.Length / 2;

            for (int i = 0; i < count / 3; i++)
            {
                var face = new WmoFace
                {
                    A = reader.ReadUInt16(),
                    B = reader.ReadUInt16(),
                    C = reader.ReadUInt16()
                };
                faces.Add(face);
            }

            return faces;
        }
    }
}
