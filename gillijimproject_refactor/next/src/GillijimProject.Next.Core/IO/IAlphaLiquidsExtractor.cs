using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.Domain.Liquids;

namespace GillijimProject.Next.Core.IO;

/// <summary>
/// Abstraction for extracting Alpha-era MCLQ data from an ADT. Returns a 16x16 (256) grid
/// of per-MCNK MCLQ payloads; null entries indicate no liquids for that chunk.
/// </summary>
public interface IAlphaLiquidsExtractor
{
    MclqData?[] Extract(AdtAlpha adt);
}
