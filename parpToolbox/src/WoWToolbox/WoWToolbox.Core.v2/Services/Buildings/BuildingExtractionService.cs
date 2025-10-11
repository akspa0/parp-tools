using System.Collections.Generic;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.Buildings;

public class BuildingExtractionService : IBuildingExtractionService
{
    public List<BuildingFragment> ExtractBuildings(PM4File pm4File)
    {
        // This is a placeholder implementation.
        // In a real scenario, this method would analyze the PM4 file chunks
        // to identify and extract geometry corresponding to buildings.
        return new List<BuildingFragment>();
    }
}
