using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Models.PD4
{
    /// <summary>
    /// Lightweight representation of a parsed PD4 file. The structure will evolve as real parsing logic is implemented.
    /// </summary>
    public class Pd4Model
    {
        /// <summary>
        /// Gets or sets the collection of terrain tiles extracted from the PD4.
        /// </summary>
        public IReadOnlyList<Vector3> Vertices { get; init; } = new List<Vector3>();

        /// <summary>
        /// Optional raw chunk data for advanced consumers. Not populated until full PD4 parsing exists.
        /// </summary>
        public byte[]? RawData { get; init; }
    }
}
