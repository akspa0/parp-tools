using WoWToolbox.Core.v2.Models.PM4;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public interface IRenderMeshBuilder
    {
        RenderMesh BuildFromPm4(PM4File pm4File);
    }
}
