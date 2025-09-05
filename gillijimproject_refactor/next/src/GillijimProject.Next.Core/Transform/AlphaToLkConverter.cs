using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Transform;

/// <summary>
/// Alpha → LK conversion pipeline entrypoint.
/// </summary>
public static class AlphaToLkConverter
{
    /// <summary>
    /// Converts an Alpha WDT and its ADTs into LK ADTs with AreaID translation.
    /// </summary>
    public static IEnumerable<AdtLk> Convert(WdtAlpha wdt, IEnumerable<AdtAlpha> adts, Services.AreaIdTranslator translator)
    {
        foreach (var adt in adts)
        {
            // TODO: Apply translator and real mapping logic
            yield return new AdtLk($"Converted:{adt.Path}");
        }
    }
}
