using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;

namespace WoWToolbox.Core.v2.Foundation.WMO.V17
{
    /// <summary>
    /// Generic chunk walker for post-alpha (v17+) WMO files. Compared to v14 the fourCC bytes are
    /// written in normal order so we can read them directly.
    /// </summary>
    public static class V17ChunkReader
    {
        public sealed record ChunkInfo(string Id, uint Offset, uint Size, byte[] Data);

        public static List<ChunkInfo> ReadAllChunks(Stream stream, ILogger? logger = null)
        {
            if (!stream.CanSeek) throw new ArgumentException("Stream must be seekable", nameof(stream));

            var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            var chunks = new List<ChunkInfo>();
            stream.Position = 0;
            while (stream.Position + 8 <= stream.Length)
            {
                string id = Encoding.ASCII.GetString(reader.ReadBytes(4));
                uint size = reader.ReadUInt32();
                long dataStart = stream.Position;
                if (dataStart + size > stream.Length)
                    break; // corrupted

                byte[] data = reader.ReadBytes((int)size);
                chunks.Add(new ChunkInfo(id, (uint)(dataStart - 8), size, data));
                logger?.LogDebug("Chunk {Id} at 0x{Offset:X8} size {Size}", id, dataStart - 8, size);
            }
            return chunks;
        }
    }
}
