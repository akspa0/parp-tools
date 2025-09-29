using System.IO;
using System.Threading.Tasks;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Generates per-tile minimap PNGs from source imagery (BLP or pre-rendered raster).
/// </summary>
public sealed class MinimapComposer
{
    /// <summary>
    /// Composes a minimap image for a tile and writes it to <paramref name="destinationPath"/>.
    /// </summary>
    /// <remarks>
    /// // TODO(PORT): Decode BLP via lib/wow.tools.local adapter and render onto a 512x512 canvas.
    /// </remarks>
    public Task ComposeAsync(Stream source, string destinationPath, ViewerOptions options)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
        // Placeholder implementation writes nothing yet.
        return Task.CompletedTask;
    }
}
