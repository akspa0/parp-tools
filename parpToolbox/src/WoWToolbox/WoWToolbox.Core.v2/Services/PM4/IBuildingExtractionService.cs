using System.Collections.Generic;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public interface IBuildingExtractionService
    {
        IEnumerable<BuildingFragment> ExtractBuildings(PM4File pm4File);
    }
}
