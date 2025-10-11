using System.Collections.Generic;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.Buildings
{
    /// <summary>
    /// Provides functionality to extract and export building model geometry.
    /// </summary>
    public interface IBuildingExtractionService
    {
        /// <summary>
        /// Extracts building fragments from a given PM4 file.
        /// </summary>
        /// <param name="pm4File">The PM4 file to extract building fragments from.</param>
        /// <returns>A list of extracted building fragments.</returns>
        List<BuildingFragment> ExtractBuildings(PM4File pm4File);
    }
}
