using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Extensions.Logging;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Models;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers;
using WoWToolbox.Core.v2.Foundation.WMO.V17.Chunks;

namespace WoWToolbox.Core.v2.Foundation.WMO.V17
{
    /// <summary>
    /// High-level loader for a root ("main") WMO v17 file.  It walks all chunks via <see cref="V17ChunkReader"/>
    /// and materialises strongly-typed C# records defined under <c>Foundation.WMO.V17.Chunks</c>.
    /// At this early stage we only expose the data required by downstream tooling (OBJ exporter, tests).
    /// Extra chunks (portals, fog, liquids, doodads) will be surfaced in later passes.
    /// </summary>
    public static class V17RootLoader
    {
        public sealed record V17Root(
            uint Version,
            MOHDHeader Header,
            IReadOnlyList<string> TextureNames,
            IReadOnlyList<MOMTMaterial> Materials,
            IReadOnlyList<MOGIEntry> GroupInfo,
            IReadOnlyList<uint>? GroupFileDataIds,
            IReadOnlyList<V17Group> Groups);

        /// <summary>
        /// Parse a v17 WMO root file from a stream.  Caller owns the stream lifetime.
        /// </summary>
        public static V17Root Load(Stream stream, ILogger? logger = null)
        {
            var chunks = V17ChunkReader.ReadAllChunks(stream, logger);

            // MVER – ensure version 17
            var mverChunk = chunks.FirstOrDefault(c => c.Id == "MVER")
                ?? throw new InvalidDataException("MVER chunk missing");
            if (mverChunk.Size != 4) throw new InvalidDataException("MVER size != 4");
            uint version = BitConverter.ToUInt32(mverChunk.Data, 0);
            if (version != 17)
                throw new InvalidDataException($"Unsupported WMO version {version}, expected 17.");

            // MOHD – root header counts, flags
            var mohdChunk = chunks.FirstOrDefault(c => c.Id == "MOHD")
                ?? throw new InvalidDataException("MOHD chunk missing");
            var header = MOHDHeader.FromSpan(mohdChunk.Data);

            // MOTX – texture names (optional in rare cases)
            var motxChunk = chunks.FirstOrDefault(c => c.Id == "MOTX");
            var textureNames = motxChunk != null ? MOTXParser.Parse(motxChunk.Data) : new List<string>();

            // MOMT – materials (optional for purely collision-only WMOs)
            var momtChunk = chunks.FirstOrDefault(c => c.Id == "MOMT");
            var materials = new List<MOMTMaterial>();
            if (momtChunk != null)
            {
                int count = (int)(momtChunk.Size / 64);
                using var br = new BinaryReader(new MemoryStream(momtChunk.Data));
                for (int i = 0; i < count; i++)
                {
                    materials.Add(new MOMTMaterial(
                        br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(),
                        br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(),
                        br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32(), br.ReadUInt32()));
                }
            }

            // MOGI – per-group info
            var mogiChunk = chunks.FirstOrDefault(c => c.Id == "MOGI");
            var groupInfo = new List<MOGIEntry>();
            if (mogiChunk != null)
            {
                int count = (int)(mogiChunk.Size / 32);
                using var br = new BinaryReader(new MemoryStream(mogiChunk.Data));
                for (int i = 0; i < count; i++)
                {
                    uint flags = br.ReadUInt32();
                    var bb1 = new System.Numerics.Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                    var bb2 = new System.Numerics.Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                    int nameIdx = br.ReadInt32();
                    groupInfo.Add(new MOGIEntry(flags, bb1, bb2, nameIdx));
                }
            }

            // GFID – external group fileDataIDs (rare outside CASC context)
            var gfidChunk = chunks.FirstOrDefault(c => c.Id == "GFID");
            IReadOnlyList<uint>? gfid = null;
            if (gfidChunk != null)
            {
                int cnt = (int)(gfidChunk.Size / 4);
                var arr = new uint[cnt];
                Buffer.BlockCopy(gfidChunk.Data, 0, arr, 0, (int)gfidChunk.Size);
                gfid = arr;
            }

            // For now we defer group loading – caller can use V17GroupLoader per external file.
            var groups = new List<V17Group>();

            return new V17Root(version, header, textureNames, materials, groupInfo, gfid, groups);
        }
    }
}
