using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Models.PM4;

public class BatchProcessResult
{
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public List<BuildingFragment> BuildingFragments { get; set; } = new List<BuildingFragment>();
    public List<WmoMatchResult> WmoMatches { get; set; } = new List<WmoMatchResult>();
}
