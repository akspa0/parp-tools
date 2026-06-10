using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.Export
{
    public interface IObjExporter
    {
        void Export(RenderMesh mesh, string filePath);
    }
}
