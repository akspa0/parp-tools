using ParpToolbox.Formats.PM4;
using System.Collections.Generic;
using System.Linq;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace ParpToolbox.Services.PM4.Core
{
    /// <summary>
    /// Service for accessing raw PM4 chunk data directly from a Pm4Scene object.
    /// </summary>
    public class Pm4ChunkAccessService : IPm4ChunkAccessService
    {
        public IEnumerable<MsurChunk.Entry> GetMsurChunks(Pm4Scene scene)
        {
            return scene.Surfaces;
        }

        public IEnumerable<System.Numerics.Vector3> GetMsvtVertices(Pm4Scene scene)
        {
            return scene.Vertices;
        }

        public IEnumerable<MslkEntry> GetMslkChunks(Pm4Scene scene)
        {
            return scene.Links;
        }
    }
}
