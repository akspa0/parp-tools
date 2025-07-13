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

            // Shared geometry at root (v17 may use MOVV/MOVB or legacy MOVT/MOVI)
            var rootMovv = chunks.FirstOrDefault(c => c.Id == "MOVV");
            var rootMovb = chunks.FirstOrDefault(c => c.Id == "MOVB");
            var rootMovt = chunks.FirstOrDefault(c => c.Id == "MOVT");
            var rootMovi = chunks.FirstOrDefault(c => c.Id == "MOVI");

            List<Vector3> sharedVertices;
            if (rootMovv != null && rootMovv.Size > 0)
                sharedVertices = V14.Parsers.MOVTParser.Parse(rootMovv.Data); // same 3-float layout
            else if (rootMovt != null)
                sharedVertices = V14.Parsers.MOVTParser.Parse(rootMovt.Data);
            else
                sharedVertices = new();

            List<(ushort, ushort, ushort)> sharedIndices;
            if (rootMovb != null && rootMovb.Size > 0)
                sharedIndices = ParseMovb(rootMovb.Data);
            else if (rootMovi != null)
                sharedIndices = V14.Parsers.MOVIParser.Parse(rootMovi.Data);
            else
                sharedIndices = new();

            bool hasShared = sharedVertices.Count > 0 && sharedIndices.Count > 0;

            // Groups – load from external files listed by MOGI
            var groups = new List<V17Group>();
            var mogiChunk = chunks.FirstOrDefault(c => c.Id == "MOGI");
            int groupCount = mogiChunk != null ? (int)(mogiChunk.Size / 32) : 0;

            string dir = Path.GetDirectoryName(path)!;
            string baseName = Path.GetFileNameWithoutExtension(path)!;

            for (int idx = 0; idx < groupCount; idx++)
            {
                string ext = Path.Combine(dir, $"{baseName}_{idx:D3}.wmo");
                if (!File.Exists(ext))
                    ext = Path.Combine(dir, $"{baseName}_{idx:D4}.wmo");
                if (!File.Exists(ext))
                    continue;

                using var gs = File.OpenRead(ext);
                groups.Add(V17GroupLoader.Load(gs, sharedVertices, sharedIndices));
                List<(ushort, ushort, ushort)> flist;

                if (hasShared && extHeader.VertexCount > 0 && extHeader.FirstVertex < sharedVertices.Count)
                {
                    int s = (int)extHeader.FirstVertex;
                    int cnt = (int)Math.Min(extHeader.VertexCount, sharedVertices.Count - s);
                    vlist = sharedVertices.GetRange(s, cnt);
                }
                else
                {
                    var extMovt = extChunks.FirstOrDefault(c => c.Id == "MOVT");
                    vlist = extMovt != null ? V14.Parsers.MOVTParser.Parse(extMovt.Data) : new();
                }

                if (hasShared && extHeader.IndexCount > 0 && extHeader.FirstIndex < sharedIndices.Count)
                {
                    int s = (int)extHeader.FirstIndex;
                    int cnt = (int)Math.Min(extHeader.IndexCount, sharedIndices.Count - s);
                    flist = sharedIndices.GetRange(s, cnt);
                }
                else
                {
                    var extMovi = extChunks.FirstOrDefault(c => c.Id == "MOVI");
                    flist = extMovi != null ? V14.Parsers.MOVIParser.Parse(extMovi.Data) : new();
                }

                groups.Add(new V17Group
                {
                    Header = extHeader,
                    Vertices = vlist,
                    Faces = flist,
                    FaceFlags = new List<byte>()
                });
            }

            return new V17WmoFile
            {
                Header = header,
                TextureNames = textures,
                Groups = groups
            };
        }

        private static List<(ushort, ushort, ushort)> ParseMovb(byte[] data)
        {
            var list = new List<(ushort, ushort, ushort)>();
            for (int i = 0; i + 5 < data.Length; i += 6)
            {
                ushort a = BitConverter.ToUInt16(data, i);
                ushort b = BitConverter.ToUInt16(data, i + 2);
                ushort c = BitConverter.ToUInt16(data, i + 4);
                list.Add((a, b, c));
            }
            return list;
        }
    }
}
