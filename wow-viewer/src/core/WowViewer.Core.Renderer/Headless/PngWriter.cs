using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WowViewer.Core.Renderer.Headless;

public static class PngWriter
{
    public static void WritePng(string path, byte[] rgba, int width, int height)
    {
        string dir = Path.GetDirectoryName(path)!;
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        using var image = Image.LoadPixelData<Rgba32>(rgba, width, height);
        image.SaveAsPng(path);
    }
}
