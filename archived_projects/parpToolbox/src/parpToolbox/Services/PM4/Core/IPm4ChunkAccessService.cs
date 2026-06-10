using ParpToolbox.Formats.PM4;
using System.Collections.Generic;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace ParpToolbox.Services.PM4.Core
{
    /// <summary>
    /// Defines a service for accessing raw PM4 chunk data directly from a Pm4Scene object.
    /// </summary>
    public interface IPm4ChunkAccessService
    {
        /// <summary>
        /// Retrieves the raw MSUR chunk data.
        /// </summary>
        /// <param name="scene">The loaded PM4 scene object.</param>
        /// <returns>An enumerable of the raw surface data chunks.</returns>
        IEnumerable<MsurChunk.Entry> GetMsurChunks(Pm4Scene scene);

        /// <summary>
        /// Retrieves the raw MSVT chunk data.
        /// </summary>
        /// <param name="scene">The loaded PM4 scene object.</param>
        /// <returns>An enumerable of the raw vertex data chunks.</returns>
        IEnumerable<System.Numerics.Vector3> GetMsvtVertices(Pm4Scene scene);

        /// <summary>
        /// Retrieves the raw MSLK chunk data.
        /// </summary>
        /// <param name="scene">The loaded PM4 scene object.</param>
        /// <returns>An enumerable of the raw surface link data chunks.</returns>
        IEnumerable<MslkEntry> GetMslkChunks(Pm4Scene scene);
    }
}
