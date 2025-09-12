using System.Collections.Generic;
using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Services;

/// <summary>
/// Analyzes unique IDs across MDDF/MODF entries.
/// </summary>
public static class UniqueIdAnalyzer
{
    public static UniqueIdReport Analyze(IEnumerable<string> adtPaths)
    {
        // TODO: Implement real analysis
        return new UniqueIdReport(
            TotalEntries: 0,
            MissingAssets: 0,
            DuplicateIds: 0,
            Notes: "Analysis not yet implemented"
        );
    }
}
