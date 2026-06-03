using System.Buffers.Binary;
using System.Text;

namespace WowViewer.Core.IO.M2Chunked;

public sealed class M2ChunkedChunkWalker(BinaryReader reader)
{
    private readonly BinaryReader _reader = reader ?? throw new ArgumentNullException(nameof(reader));

    public IReadOnlyList<M2ChunkedChunkHeader> Walk()
    {
        Stream stream = _reader.BaseStream;
        if (!stream.CanSeek)
            throw new ArgumentException("Chunked MDX walking requires a seekable stream.", nameof(reader));

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            if (stream.Length < sizeof(uint))
                return [];

            Span<byte> magicBytes = stackalloc byte[sizeof(uint)];
            stream.ReadExactly(magicBytes);
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(magicBytes);
            if (magic != MdxMagic.Mdlx)
                return [];

            List<M2ChunkedChunkHeader> chunks = [];
            while (stream.Position <= stream.Length - 8)
            {
                long chunkOffset = stream.Position;
                string fourCC = ReadFourCc(_reader);
                uint declaredSize = _reader.ReadUInt32();
                long payloadStart = stream.Position;
                long remaining = stream.Length - payloadStart;
                bool isTruncated = false;
                uint readableSize = declaredSize;
                if (declaredSize > remaining)
                {
                    readableSize = checked((uint)Math.Max(0, remaining));
                    isTruncated = true;
                    Console.Error.WriteLine(
                        $"[M2Chunked] Truncated chunk '{fourCC}' at 0x{chunkOffset:X}: declared=0x{declaredSize:X}, remaining=0x{remaining:X}.");
                }

                chunks.Add(new M2ChunkedChunkHeader(fourCC, readableSize, chunkOffset, isTruncated));
                stream.Position = payloadStart + readableSize;
                if (isTruncated)
                    break;
            }

            return chunks;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadFourCc(BinaryReader reader)
    {
        byte[] bytes = reader.ReadBytes(4);
        if (bytes.Length != 4)
            throw new EndOfStreamException("Unexpected end of stream while reading MDX FourCC.");

        return Encoding.ASCII.GetString(bytes);
    }
}
