using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Foundation.WMO
{
    public class WmoGroupMesh
    {
        public List<WmoVertex> Vertices { get; set; } = new List<WmoVertex>();
        public List<WmoFace> Indices { get; set; } = new List<WmoFace>();
        public List<byte> RenderFlags { get; set; } = new List<byte>();
        public List<string> TextureNames { get; set; } = new List<string>();
        public List<ushort> FaceMaterialIds { get; set; } = new List<ushort>();
    }
}
