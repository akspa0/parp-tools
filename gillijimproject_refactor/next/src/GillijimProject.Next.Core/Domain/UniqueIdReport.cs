namespace GillijimProject.Next.Core.Domain;

/// <summary>
/// Summary of UniqueID analysis for an ADT set.
/// </summary>
public sealed record UniqueIdReport(
    int TotalEntries,
    int MissingAssets,
    int DuplicateIds,
    string Notes
);
