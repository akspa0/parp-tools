using System.Collections.Generic;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Models;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Models
{
    public class V14Group
    {
        public MOGPGroupHeader Header { get; init; }
        public List<Vector3> Vertices { get; init; } = new();
        public List<(ushort A, ushort B, ushort C)> Faces { get; init; } = new();
        public List<byte> FaceFlags { get; init; } = new();
    }
}
