using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using System.Buffers.Binary;
using System.Text;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14
{
    /// <summary>
    /// Very small, initial helper that walks a v14 WMO binary and extracts all top-level chunks.
    /// This is a stub to unblock further detailed per-chunk parsing.
    /// </summary>
    public static class V14ChunkReader
    {
        public sealed record ChunkInfo(string Id, uint Offset, uint Size, byte[] Data);

        /// <summary>
        /// Reads every chunk (id+size+payload) from the supplied <paramref name="stream"/>.
        /// </summary>
        public static List<ChunkInfo> ReadAllChunks(Stream stream, ILogger? logger = null)
        {
            if (!stream.CanSeek) throw new ArgumentException("Stream must be seekable", nameof(stream));

            var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            var chunks = new List<ChunkInfo>();
            stream.Position = 0;
            while (stream.Position < stream.Length)
            {
                if (stream.Length - stream.Position < 8) break; // safety

                // chunk IDs in WMOs are little-endian fourCCs reversed in legacy files.
                var idBytes = reader.ReadBytes(4);
                Array.Reverse(idBytes);
                string id = Encoding.ASCII.GetString(idBytes);

                uint size = reader.ReadUInt32();
                long dataStart = stream.Position;
                if (dataStart + size > stream.Length)
                {
                    // corrupted; stop parsing to avoid OOM
                    break;
                }

                byte[] data = reader.ReadBytes((int)size);

                if (id == "MOMO")
                {
                    // Parse sub-chunks inside MOMO so callers can find MOHD, MOGN, etc.
                    using var ms = new MemoryStream(data);
                    var subReader = new BinaryReader(ms);
                    while (ms.Position + 8 <= ms.Length)
                    {
                        var subIdBytes = subReader.ReadBytes(4);
                        Array.Reverse(subIdBytes);
                        string subId = Encoding.ASCII.GetString(subIdBytes);
                        uint subSize = subReader.ReadUInt32();
                        long subDataStart = ms.Position;
                        if (subDataStart + subSize > ms.Length) break; // corrupted
                        byte[] subData = subReader.ReadBytes((int)subSize);
                        chunks.Add(new ChunkInfo(subId, (uint)(dataStart - 8 + subDataStart), subSize, subData));
                        ms.Position = subDataStart + subSize;
                    }
                    // Optionally still add MOMO chunk itself if needed—currently omitted to avoid duplicates
                }
                else
                {
                    chunks.Add(new ChunkInfo(id, (uint)(dataStart - 8), size, data));
                }

                logger?.LogDebug("Chunk {Id} at 0x{Offset:X8} size {Size}", id, dataStart - 8, size);

                // Extra diagnostics: after MOVT dump next 256 bytes and show next four IDs
                if (id == "MOVT" && logger != null)
                {
                    long saved = stream.Position;
                    int dumpLen = (int)Math.Min(256, stream.Length - stream.Position);
                    byte[] peek = reader.ReadBytes(dumpLen);
                    logger.LogDebug("After MOVT dump ({Len} bytes): {Hex}", dumpLen, BitConverter.ToString(peek).Replace("-", ""));

                    // Attempt to read next four chunk IDs
                    var ids = new List<string>();
                    int offset = 0;
                    while (ids.Count < 4 && offset + 8 <= peek.Length)
                    {
                        var idBytesPeek = peek.AsSpan(offset, 4).ToArray();
                        Array.Reverse(idBytesPeek);
                        ids.Add(Encoding.ASCII.GetString(idBytesPeek));
                        offset += 8 + (int)BinaryPrimitives.ReadUInt32LittleEndian(peek.AsSpan(offset + 4, 4));
                    }
                    logger.LogDebug("Next chunk IDs after MOVT: {Ids}", string.Join(",", ids));
                    stream.Position = saved; // restore position
                }
            }
            return chunks;
        }
    }
}
