using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.IO;

/// <summary>
/// Readers for Alpha-era WDT/ADT inputs.
/// </summary>
public static class AlphaReader
{
    /// <summary>
    /// Parses an Alpha WDT file into a minimal model.
    /// </summary>
    public static WdtAlpha ParseWdt(string alphaWdtPath)
    {
        // TODO: Implement real parsing based on existing ported readers.
        return new WdtAlpha(alphaWdtPath);
    }

    /// <summary>
    /// Parses an Alpha ADT file into a minimal model.
    /// </summary>
    public static AdtAlpha ParseAdt(string alphaAdtPath)
    {
        // TODO: Implement real parsing based on existing ported readers.
        return new AdtAlpha(alphaAdtPath);
    }
}
