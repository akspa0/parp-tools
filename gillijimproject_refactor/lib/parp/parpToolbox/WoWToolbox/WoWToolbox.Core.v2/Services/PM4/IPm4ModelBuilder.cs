using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.Data;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public interface IPm4ModelBuilder
    {
        CompleteWMOModel Build(PM4File pm4File);
    }
}
