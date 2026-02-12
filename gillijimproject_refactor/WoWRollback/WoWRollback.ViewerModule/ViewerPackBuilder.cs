using System;
using System.Collections.Generic;
using WoWRollback.Core.Services.Viewer;

namespace WoWRollback.ViewerModule;

public class ViewerPackBuilder
{
    public (int TilesWritten, int MapsWritten, string DefaultMap) Build(
        string sessionRoot,
        string minimapRoot,
        string outRoot,
        IReadOnlyList<string> versionsFilter,
        HashSet<string> mapsFilter,
        string label,
        Func<string, IMinimapProvider?>? providerFactory)
    {
        // Stub implementation to satisfy build
        return (0, 0, mapsFilter?.Count > 0 ? System.Linq.Enumerable.First(mapsFilter) : "");
    }

    public void HarvestFromConvertedAdts(
        List<(string Version, string Map, string MapDir)> inputs,
        string? communityListfile,
        string? lkListfile,
        string overlaysRoot)
    {
        // Stub implementation
    }
}
