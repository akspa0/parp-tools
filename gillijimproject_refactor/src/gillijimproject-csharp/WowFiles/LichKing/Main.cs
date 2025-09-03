using System.Collections.Generic;
using System.Linq;

namespace GillijimProject.WowFiles.LichKing
{
    public class Main : Chunk
    {
        public Main(MhdrOffset[] mhdrOffsets) : base("MAIN", GenerateData(mhdrOffsets).Length, GenerateData(mhdrOffsets))
        {
        }

        private static byte[] GenerateData(MhdrOffset[] mhdrOffsets)
        {
            var data = new List<byte>();
            foreach (var offset in mhdrOffsets)
            {
                data.AddRange(offset.GetBytes());
            }
            return data.ToArray();
        }
    }

    public struct MhdrOffset
    {
        public int Flags;
        public int Offset;
        public int Size;
        public int Unknown;

        public byte[] GetBytes()
        {
            var bytes = new List<byte>();
            bytes.AddRange(System.BitConverter.GetBytes(Flags));
            bytes.AddRange(System.BitConverter.GetBytes(Offset));
            bytes.AddRange(System.BitConverter.GetBytes(Size));
            bytes.AddRange(System.BitConverter.GetBytes(Unknown));
            return bytes.ToArray();
        }
    }
}
