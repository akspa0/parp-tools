using System;
using System.IO;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace WoWRollback.MinimapModule.Services;

/// <summary>
/// Converts BLP texture files to PNG format.
/// </summary>
public static class BlpConverter
{
    /// <summary>
    /// Load a BLP file and return as ImageSharp image.
    /// </summary>
    public static Image<Rgba32>? LoadBlp(string blpPath)
    {
        if (!File.Exists(blpPath))
            return null;

        using var stream = File.OpenRead(blpPath);
        return LoadBlp(stream);
    }

    /// <summary>
    /// Load a BLP from stream and return as ImageSharp image.
    /// </summary>
    public static Image<Rgba32>? LoadBlp(Stream blpStream)
    {
        try
        {
            using var blp = new BlpFile(blpStream);
            return blp.GetImage(0); // Get highest resolution mipmap
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// Convert BLP file to PNG.
    /// </summary>
    public static bool ConvertToPng(string blpPath, string pngPath)
    {
        using var image = LoadBlp(blpPath);
        if (image == null)
            return false;

        Directory.CreateDirectory(Path.GetDirectoryName(pngPath)!);
        image.SaveAsPng(pngPath);
        return true;
    }

    /// <summary>
    /// Convert BLP stream to PNG file.
    /// </summary>
    public static bool ConvertToPng(Stream blpStream, string pngPath)
    {
        using var image = LoadBlp(blpStream);
        if (image == null)
            return false;

        Directory.CreateDirectory(Path.GetDirectoryName(pngPath)!);
        image.SaveAsPng(pngPath);
        return true;
    }
}
