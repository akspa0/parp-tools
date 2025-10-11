using System.Collections.Generic;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers;

namespace WoWToolbox.Core.v2.Foundation.WMO.V17
{
    public class V17Group
    {
        public V14.Models.MOGPGroupHeader Header { get; init; }
        public List<Vector3> Vertices { get; init; } = new();
        public List<(ushort A, ushort B, ushort C)> Faces { get; init; } = new();
        public List<byte> FaceFlags { get; init; } = new();
    }
}
