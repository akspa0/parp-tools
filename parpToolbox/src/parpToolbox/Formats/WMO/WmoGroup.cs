using System.Collections.Generic;
using System.Numerics;

namespace ParpToolbox.Formats.WMO
{
    /// <summary>
    /// Immutable geometry container for a single WMO group.
    /// </summary>
    public sealed class WmoGroup
    {
        public string Name { get; }
        public IReadOnlyList<Vector3> Vertices { get; }
        public IReadOnlyList<(ushort, ushort, ushort)> Faces { get; }
        public IReadOnlyList<byte> FaceMaterialIds { get; }

        public WmoGroup(string name,
                         IReadOnlyList<Vector3> vertices,
                         IReadOnlyList<(ushort, ushort, ushort)> faces,
                         IReadOnlyList<byte>? faceMaterialIds = null)
        {
            Name = name;
            Vertices = vertices;
            Faces = faces;
            FaceMaterialIds = faceMaterialIds ?? new List<byte>();
        }
    }
}
