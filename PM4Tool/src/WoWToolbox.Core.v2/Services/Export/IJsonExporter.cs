namespace WoWToolbox.Core.v2.Services.Export
{
    public interface IJsonExporter
    {
        void Export(object data, string filePath);
    }
}
