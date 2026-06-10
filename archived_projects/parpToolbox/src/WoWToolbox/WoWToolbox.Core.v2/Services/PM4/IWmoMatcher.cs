using System.Collections.Generic;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public interface IWmoMatcher
    {
        IEnumerable<WmoMatchResult> Match(IEnumerable<BuildingFragment> fragments);
    }
}
