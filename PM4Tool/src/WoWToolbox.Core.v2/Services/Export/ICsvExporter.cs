using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Services.Export
{
    public interface ICsvExporter
    {
        void Export<T>(IEnumerable<T> data, string filePath);
    }
}
