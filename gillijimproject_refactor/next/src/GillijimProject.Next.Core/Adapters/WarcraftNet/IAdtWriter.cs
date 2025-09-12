using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Adapters.WarcraftNet;

/// <summary>
/// Contract for writing LK ADT files.
/// </summary>
public interface IAdtWriter
{
    void Write(AdtLk adt, string outputPath);
}
