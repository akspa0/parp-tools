using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public interface IPm4BatchProcessor
    {
        BatchProcessResult Process(string pm4FilePath);
    }
}
