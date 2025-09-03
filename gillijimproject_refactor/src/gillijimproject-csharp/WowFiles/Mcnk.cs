using System;

namespace GillijimProject.WowFiles
{
    public abstract class Mcnk : Chunk
    {
        protected Mcnk() : base("MCNK", 0, System.Array.Empty<byte>()) { }

        protected Mcnk(byte[] adtFile, int offsetInFile) : base(adtFile, offsetInFile)
        {
        }

        // [PORT] Provide a forwarding constructor used by Alpha/LK wrappers assembling MCNK payloads in memory.
        protected Mcnk(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData)
        {
        }

        public abstract byte[] GetPayload();

        public new byte[] GetWholeChunk()
        {
            var payload = GetPayload();
            var wrapper = new Chunk("MCNK", payload.Length, payload);
            return wrapper.GetWholeChunk();
        }
    }
}
