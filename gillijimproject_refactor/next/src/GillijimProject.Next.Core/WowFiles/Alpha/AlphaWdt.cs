using System.Collections.Generic;

namespace GillijimProject.Next.Core.WowFiles.Alpha;

/// <summary>
/// Minimal domain model for Alpha-era WDT, sufficient for building LK WDT and discovering ADTs.
/// </summary>
public sealed record AlphaWdt(
    string Path,
    bool WmoBased,
    IReadOnlyList<int> AdtOffsets,
    IReadOnlyList<string> MdnmFiles,
    IReadOnlyList<string> MonmFiles
);
