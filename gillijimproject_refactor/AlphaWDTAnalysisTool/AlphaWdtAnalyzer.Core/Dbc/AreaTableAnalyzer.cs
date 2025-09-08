using System;
using System.Collections.Generic;

namespace AlphaWdtAnalyzer.Core.Dbc;

public sealed class AreaRow
{
    public int Id { get; init; }
    public string Name { get; init; } = string.Empty;
}

public sealed class AreaCrosswalk
{
    public List<(AreaRow Alpha, AreaRow Lk, string Match)> Matches { get; } = new();
    public List<AreaRow> UnmatchedAlpha { get; } = new();
    public List<AreaRow> UnmatchedLk { get; } = new();
}

public static class AreaTableAnalyzer
{
    // [DEPRECATED] This analyzer previously depended on a raw DBC reader. We now use DBCD.
    // Use the DBCD-backed export/mapping pipeline instead of calling this method.
    public static AreaCrosswalk Compare(string alphaAreaDbc, string lkAreaDbc)
    {
        throw new NotSupportedException("AreaTableAnalyzer.Compare is deprecated. Use the DBCD-backed export/mapping pipeline.");
    }
}
