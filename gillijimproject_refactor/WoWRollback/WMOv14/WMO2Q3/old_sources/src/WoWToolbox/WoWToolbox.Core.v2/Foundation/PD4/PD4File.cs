using System.IO;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.v2.Foundation.PM4; // reuse chunk properties and helpers
using Warcraft.NET.Attribute;
using WoWToolbox.Core.v2.Foundation.PD4.Chunks;

namespace WoWToolbox.Core.v2.Foundation.PD4
{
    /// <summary>
    /// Minimal PD4 file wrapper. PD4 appears to share the same chunk set as PM4 but scoped to a single WMO.
    /// We simply inherit from <see cref="PM4File"/> so that existing utilities expecting PM4File can operate
    /// on PD4 files without modification.
    /// </summary>
    public class PD4File : PM4File, IBinarySerializable
    {
        /// <summary>
        /// Gets or sets the MCRC chunk (PD4 checksum placeholder).
        /// </summary>
        [ChunkOptional]
        public MCRCChunk? MCRC { get; set; }

        public PD4File() { }
        public PD4File(byte[] data) : base(data) { }

        /// <summary>
        /// Loads a PD4 file from disk.
        /// </summary>
        public static new PD4File FromFile(string path)
        {
            return new PD4File(File.ReadAllBytes(path));
        }
    }
}
