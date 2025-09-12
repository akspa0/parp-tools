using System.Collections.Generic;
using GillijimProject.Next.Core.Domain;
using GillijimProject.Next.Core.Domain.Liquids;
using GillijimProject.Next.Core.Transform.Liquids;
using GillijimProject.Next.Core.IO;

namespace GillijimProject.Next.Core.Transform;

/// <summary>
/// Alpha → LK conversion pipeline entrypoint.
/// </summary>
public static class AlphaToLkConverter
{
    /// <summary>
    /// Converts an Alpha WDT and its ADTs into LK ADTs with AreaID translation.
    /// </summary>
    public static IEnumerable<AdtLk> Convert(WdtAlpha wdt, IEnumerable<AdtAlpha> adts, Services.AreaIdTranslator translator, LiquidsOptions liquids, IAlphaLiquidsExtractor extractor)
    {
        foreach (var adt in adts)
        {
            var mclqs = extractor.Extract(adt);
            var output = new AdtLk($"Converted:{adt.Path}")
            {
                Mh2oByChunk = new Mh2oChunk?[256]
            };

            if (mclqs is not null)
            {
                int count = mclqs.Length < output.Mh2oByChunk.Length ? mclqs.Length : output.Mh2oByChunk.Length;
                for (int i = 0; i < count; i++)
                {
                    var m = mclqs[i];
                    if (m is null) continue;
                    output.Mh2oByChunk[i] = LiquidsConverter.MclqToMh2o(m, liquids);
                }
            }

            yield return output;
        }
    }
}
