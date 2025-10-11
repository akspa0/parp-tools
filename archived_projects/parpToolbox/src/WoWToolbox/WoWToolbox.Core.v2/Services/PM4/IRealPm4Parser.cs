using System.Collections.Generic;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Parses a <see cref="PM4File"/> into higher-level navigation objects suitable for correlation with WMO geometry.
    /// </summary>
    public interface IRealPm4Parser
    {
        /// <summary>
        /// Extracts individual navigation mesh objects from the supplied PM4 file.
        /// </summary>
        /// <param name="pm4">A loaded <see cref="PM4File"/>.</param>
        /// <returns>Collection of <see cref="IndividualNavigationObject"/> objects. May be empty if no valid geometry.</returns>
        IEnumerable<IndividualNavigationObject> ParseIndividualObjects(PM4File pm4);
    }
}
