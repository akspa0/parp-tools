using System.IO;
using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Adapters.WarcraftNet;

/// <summary>
/// Writes LK ADT files using Warcraft.NET. Stub implementation for scaffolding.
/// </summary>
public sealed class WarcraftNetAdtWriter : IAdtWriter
{
    public void Write(AdtLk adt, string outputPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        // TODO: Map AdtLk to Warcraft.NET structures and serialize v18 ADT.
        File.WriteAllText(outputPath, $"// Placeholder for {adt.Name} (LK ADT v18)\n");
    }
}
