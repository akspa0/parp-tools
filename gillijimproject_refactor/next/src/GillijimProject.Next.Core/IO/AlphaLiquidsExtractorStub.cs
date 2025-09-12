using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.Domain.Liquids;

namespace GillijimProject.Next.Core.IO;

/// <summary>
/// Temporary placeholder extractor that returns an empty set (no liquids).
/// Replace with a real Alpha ADT reader.
/// </summary>
public sealed class AlphaLiquidsExtractorStub : IAlphaLiquidsExtractor
{
    public MclqData?[] Extract(AdtAlpha adt) => new MclqData?[256];
}
