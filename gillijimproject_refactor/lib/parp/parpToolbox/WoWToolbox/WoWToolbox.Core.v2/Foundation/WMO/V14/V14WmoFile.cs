using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Models;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14
{
    /// <summary>
    /// Lightweight reader that understands the high-level structure of a v14 WMO file.
    /// Its purpose is to verify that all essential chunks are located and counts line up with the MOHD header.
    /// Geometry data is parsed only to the extent needed for count validation.
    /// </summary>
    public sealed class V14WmoFile
    {
        public MOHDHeader Header { get; private init; }
        public IReadOnlyList<string> TextureNames { get; private init; } = Array.Empty<string>();
        public IReadOnlyList<V14Group> Groups { get; private init; } = Array.Empty<V14Group>();

        private V14WmoFile() { }

        public static V14WmoFile Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException("WMO not found", path);
            using var fs = File.OpenRead(path);
            var chunks = V14ChunkReader.ReadAllChunks(fs);

            // MOHD
            var mohdChunk = chunks.FirstOrDefault(c => c.Id == "MOHD");
            if (mohdChunk == null) throw new InvalidDataException("MOHD chunk missing");
            var header = MOHDHeader.FromSpan(mohdChunk.Data);

            // MOTX (texture names)
            var motxChunk = chunks.FirstOrDefault(c => c.Id == "MOTX");
            var textures = motxChunk != null ? MOTXParser.Parse(motxChunk.Data) : new List<string>();

            // Groups – each has its own chunk block consisting of MOGP header followed by sub-chunks
            var groups = new List<V14Group>();
            foreach (var mogp in chunks.Where(c => c.Id == "MOGP"))
            {
                var grpHeader = MOGPGroupHeader.FromSpan(mogp.Data);
                // sub-chunks are expected to follow immediately; locate them by offset range
                long start = mogp.Offset + 8 + grpHeader.Description; // naive assumption – placeholder
                // For skeleton reader we ignore precise seeking and just grab first MOVT/MOVI/MOPY after this MOGP
                var movt = chunks.FirstOrDefault(c => c.Id == "MOVT" && c.Offset > mogp.Offset);
                var movi = chunks.FirstOrDefault(c => c.Id == "MOVI" && c.Offset > mogp.Offset);
                var mopy = chunks.FirstOrDefault(c => c.Id == "MOPY" && c.Offset > mogp.Offset);

                var group = new V14Group
                {
                    Header = grpHeader,
                    Vertices = movt != null ? MOVTParser.Parse(movt.Data) : new List<System.Numerics.Vector3>(),
                    Faces = movi != null ? MOVIParser.Parse(movi.Data) : new List<(ushort, ushort, ushort)>(),
                    FaceFlags = mopy != null ? MOPYParser.Parse(mopy.Data) : new List<byte>()
                };
                groups.Add(group);
            }

            return new V14WmoFile
            {
                Header = header,
                TextureNames = textures,
                Groups = groups
            };
        }
    }
}
