using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Interfaces;
using ArcaneFileParser.Core.Formats.WMO.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO
{
    public class WMOFile
    {
        public uint Version { get; private set; }
        public Dictionary<string, IChunk> Chunks { get; private set; }
        public bool IsAlphaFormat => Version == 14;

        public WMOFile()
        {
            Chunks = new Dictionary<string, IChunk>();
        }

        public void Load(string filename)
        {
            using (var reader = new BinaryReader(File.OpenRead(filename)))
            {
                // Read MVER chunk first
                var mver = ChunkFactory.CreateChunk(reader) as MVER;
                if (mver == null)
                {
                    throw new InvalidDataException("File is not a valid WMO file (missing MVER chunk)");
                }
                Version = mver.Version;
                Chunks["MVER"] = mver;

                // For v14, expect a MOMO container chunk next
                if (IsAlphaFormat)
                {
                    var momo = ChunkFactory.CreateChunk(reader) as MOMO;
                    if (momo == null)
                    {
                        throw new InvalidDataException("Alpha WMO file is missing MOMO chunk");
                    }
                    Chunks["MOMO"] = momo;

                    // Copy chunks from MOMO to top level for consistent access
                    foreach (var chunk in (momo as ContainerChunkBase).SubChunks)
                    {
                        Chunks[chunk.Key] = chunk.Value;
                    }
                }
                else
                {
                    // For v17, read chunks until end of file
                    while (reader.BaseStream.Position < reader.BaseStream.Length)
                    {
                        var chunk = ChunkFactory.CreateChunk(reader);
                        if (chunk != null)
                        {
                            Chunks[chunk.ChunkId] = chunk;
                        }
                    }
                }
            }
        }

        public void LoadGroup(string filename)
        {
            if (IsAlphaFormat)
            {
                throw new InvalidOperationException("Alpha WMO files do not have separate group files");
            }

            using (var reader = new BinaryReader(File.OpenRead(filename)))
            {
                // Read MVER chunk first
                var mver = ChunkFactory.CreateChunk(reader) as MVER;
                if (mver == null || mver.Version != Version)
                {
                    throw new InvalidDataException("Invalid or mismatched WMO group file version");
                }

                // Read group chunks
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    var chunk = ChunkFactory.CreateChunk(reader);
                    if (chunk != null)
                    {
                        Chunks[chunk.ChunkId] = chunk;
                    }
                }
            }
        }

        public T GetChunk<T>(string chunkId) where T : class, IChunk
        {
            if (Chunks.TryGetValue(chunkId, out IChunk chunk))
            {
                return chunk as T;
            }
            return null;
        }
    }
} 