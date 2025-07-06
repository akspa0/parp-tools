using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Foundation.WMO
{
    public class WmoGroupMesh
    {
        public List<WmoVertex> Vertices { get; set; } = new List<WmoVertex>();
        public List<WmoFace> Indices { get; set; } = new List<WmoFace>();
    }
}
