using System.Collections.Generic;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public interface ICoordinateService
    {
        Vector3 FromMsvtVertex(MSVT_Vertex vertex);
        Vector3 FromMsvtVertexSimple(MSVT_Vertex vertex);
        Vector3 FromMsvtVertexSimple(WoWToolbox.Core.v2.Foundation.PM4.Chunks.MsvtVertex vertex);
        Vector3 FromMscnVertex(MSCN_Vertex vertex);
        Vector3 FromMscnVertex(System.Numerics.Vector3 vertex);
        Vector3 FromMspvVertex(MSPV_Vertex vertex);
        Vector3 FromMspvVertex(WoWToolbox.Core.v2.Models.PM4.Chunks.C3Vector vertex);
        Vector3 FromMspvVertex(Warcraft.NET.Files.Structures.C3Vector vertex);
        Vector3 FromMprlEntry(MPRL_Entry entry);
        Vector3 FromMprlEntry(WoWToolbox.Core.v2.Foundation.PM4.Chunks.MprlEntry entry);
        List<Vector3> ComputeVertexNormals(IList<Vector3> vertices, IList<int> indices);
    }
}
