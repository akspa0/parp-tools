using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Models; // reuse structures
using WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers; // reuse existing parsers (identical layout)
using System.Numerics;

namespace WoWToolbox.Core.v2.Foundation.WMO.V17
{
    /// <summary>
    /// Skeleton high-level reader for WMO version 17.  It locates essential chunks and parses enough
    /// geometry data to verify counts.  The intent is to flesh out additional chunk handling iteratively
    /// while allowing tests to load real v17 files today.
    /// </summary>
    public sealed class V17WmoFile
    {
        public V14.Models.MOHDHeader Header { get; private init; }
        public IReadOnlyList<string> TextureNames { get; private init; } = Array.Empty<string>();
        public IReadOnlyList<V17Group> Groups { get; private init; } = Array.Empty<V17Group>();

        private V17WmoFile() { }

        public static V17WmoFile Load(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException("WMO not found", path);
            using var fs = File.OpenRead(path);
            var chunks = V17ChunkReader.ReadAllChunks(fs);

            // MOHD
            var mohdChunk = chunks.FirstOrDefault(c => c.Id == "MOHD");
            if (mohdChunk == null) throw new InvalidDataException("MOHD chunk missing");
            var header = V14.Models.MOHDHeader.FromSpan(mohdChunk.Data);

            // MOTX – texture names
            var motxChunk = chunks.FirstOrDefault(c => c.Id == "MOTX");
            var textures = motxChunk != null ? V14.Parsers.MOTXParser.Parse(motxChunk.Data) : new List<string>();

            // Groups – locate each MOGP header and immediately following MOVT/MOVI/MOPY blocks
            var groups = new List<V17Group>();
            foreach (var mogp in chunks.Where(c => c.Id == "MOGP"))
            {
                var grpHeader = V14.Models.MOGPGroupHeader.FromSpan(mogp.Data);

                // naive association: first MOVT/MOVI/MOPY chunks that occur after this header and before the next MOGP
                long thisIndex = mogp.Offset;
                var nextMogp = chunks.Where(c => c.Id == "MOGP" && c.Offset > thisIndex).OrderBy(c => c.Offset).FirstOrDefault();
                long boundary = nextMogp?.Offset ?? long.MaxValue;

                var movt = chunks.FirstOrDefault(c => c.Id == "MOVT" && c.Offset > thisIndex && c.Offset < boundary);
                var movi = chunks.FirstOrDefault(c => c.Id == "MOVI" && c.Offset > thisIndex && c.Offset < boundary);
                var mopy = chunks.FirstOrDefault(c => c.Id == "MOPY" && c.Offset > thisIndex && c.Offset < boundary);

                var group = new V17Group
                {
                    Header = grpHeader,
                    Vertices = movt != null ? MOVTParser.Parse(movt.Data) : new List<Vector3>(),
                    Faces = movi != null ? MOVIParser.Parse(movi.Data) : new List<(ushort, ushort, ushort)>(),
                    FaceFlags = mopy != null ? MOPYParser.Parse(mopy.Data) : new List<byte>()
                };
                groups.Add(group);
            }

            return new V17WmoFile
            {
                Header = header,
                TextureNames = textures,
                Groups = groups
            };
        }
    }
}
