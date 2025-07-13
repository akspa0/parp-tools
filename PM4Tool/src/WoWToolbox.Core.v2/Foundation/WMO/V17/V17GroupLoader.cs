using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using Microsoft.Extensions.Logging;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Models;
using WoWToolbox.Core.v2.Foundation.WMO.V17.Parsers;

namespace WoWToolbox.Core.v2.Foundation.WMO.V17
{
    public static class V17GroupLoader
    {
        public static V17Group Load(Stream stream, IReadOnlyList<Vector3> sharedVertices, IReadOnlyList<(ushort, ushort, ushort)> sharedFaces, ILogger? logger = null)
        {
            var chunks = V17ChunkReader.ReadAllChunks(stream, logger);

            var mogpChunk = chunks.FirstOrDefault(c => c.Id == "MOGP") ?? throw new InvalidDataException("MOGP missing in group file");
            var header = MOGPGroupHeader.FromSpan(mogpChunk.Data);

            // Vertices & faces
            List<Vector3> vertices;
            List<(ushort, ushort, ushort)> faces;

            if (header.NVertices == 0)
            {
                // Use shared root geometry slice
                vertices = sharedVertices.Skip((int)header.FirstVertex).Take((int)header.VertexCount).ToList();
                faces = sharedFaces.Skip((int)(header.FirstFace / 3)).Take((int)(header.FaceCount / 3)).ToList();
            }
            else
            {
                // Local MOVT / MOVI inside group
                var movt = chunks.FirstOrDefault(c => c.Id == "MOVT") ?? throw new InvalidDataException("MOVT missing");
                vertices = new List<Vector3>((int)(movt.Size / 12));
                using (var br = new BinaryReader(new MemoryStream(movt.Data)))
                {
                    while (br.BaseStream.Position < br.BaseStream.Length)
                    {
                        vertices.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
                    }
                }

                var movi = chunks.FirstOrDefault(c => c.Id == "MOVI") ?? throw new InvalidDataException("MOVI missing");
                faces = new List<(ushort, ushort, ushort)>((int)(movi.Size / 6));
                for (int i = 0; i + 5 < movi.Data.Length; i += 6)
                {
                    faces.Add((BitConverter.ToUInt16(movi.Data, i), BitConverter.ToUInt16(movi.Data, i + 2), BitConverter.ToUInt16(movi.Data, i + 4)));
                }
            }

            // Face flags / materials
            var faceFlags = new List<byte>();
            var mopy = chunks.FirstOrDefault(c => c.Id == "MOPY");
            if (mopy != null)
            {
                var entries = MOPYParser.Parse(mopy.Data);
                faceFlags = entries.Select(e => e.Flags).ToList();
            }
            else
            {
                faceFlags = Enumerable.Repeat((byte)0, faces.Count).ToList();
            }

            return new V17Group
            {
                Header = header,
                Vertices = vertices,
                Faces = faces,
                FaceFlags = faceFlags,
                Flags = header.Flags,
                NameIndex = header.NameOfs
            };
        }
    }
}
