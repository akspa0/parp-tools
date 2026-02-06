using System;
using System.IO;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using MdxLTool.Formats.Mdx;

namespace MdxLTool.Services;

/// <summary>
/// Service for resolving and converting BLP textures.
/// </summary>
public class TextureService
{
    private readonly NativeMpqService _mpqService;

    public TextureService(NativeMpqService mpqService)
    {
        _mpqService = mpqService;
    }

    /// <summary>
    /// Attempts to resolve a texture path and convert it to PNG.
    /// </summary>
    /// <param name="texturePath">The virtual path from the MDX (e.g., "Creature\Dragon\Dragon.blp")</param>
    /// <param name="modelRoot">Path to the directory containing the model (for local lookup)</param>
    /// <param name="outputDir">Target directory for the converted PNG</param>
    /// <returns>The relative path to the converted PNG, or null if failed</returns>
    public string? ExportTexture(string texturePath, string modelRoot, string outputDir)
    {
        if (string.IsNullOrWhiteSpace(texturePath)) return null;

        // 1. Try local lookup (same folder as MDX)
        var fileName = Path.GetFileName(texturePath);
        var localPath = Path.Combine(modelRoot, fileName);
        if (File.Exists(localPath))
        {
            return ConvertBlpToPng(localPath, outputDir);
        }

        // 2. Try MPQ lookup
        if (_mpqService.FileExists(texturePath))
        {
            var data = _mpqService.ReadFile(texturePath);
            if (data != null)
            {
                return ConvertBlpToPng(data, fileName, outputDir);
            }
        }

        // 3. Try variations (some models reference .tga or have no extension in 0.5.3?)
        var blpPath = Path.ChangeExtension(texturePath, ".blp");
        if (blpPath != texturePath && _mpqService.FileExists(blpPath))
        {
            var data = _mpqService.ReadFile(blpPath);
            if (data != null)
            {
                return ConvertBlpToPng(data, Path.GetFileName(blpPath), outputDir);
            }
        }

        return null;
    }

    public string? ConvertBlpToPng(string blpPath, string outputDir)
    {
        try
        {
            using var fs = File.OpenRead(blpPath);
            return ConvertBlpToPng(fs, Path.GetFileName(blpPath), outputDir);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to read local BLP {blpPath}: {ex.Message}");
            return null;
        }
    }

    public string? ConvertBlpToPng(byte[] data, string fileName, string outputDir)
    {
        using var ms = new MemoryStream(data);
        return ConvertBlpToPng(ms, fileName, outputDir);
    }

    private string? ConvertBlpToPng(Stream stream, string fileName, string outputDir)
    {
        try
        {
            using var blp = new BlpFile(stream);
            // Alpha textures are usually small, 0 index mipmap is fine
            var bmp = blp.GetBitmap(0);
            
            var pngName = Path.ChangeExtension(fileName, ".png");
            var outputPath = Path.Combine(outputDir, pngName);
            
            Directory.CreateDirectory(outputDir);

            // Using ImageSharp to save as PNG
            // We need to convert System.Drawing.Bitmap data to ImageSharp
            var rect = new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height);
            var data = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, 
                System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            
            try {
                var bytes = new byte[data.Stride * data.Height];
                System.Runtime.InteropServices.Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);
                
                using var image = Image.LoadPixelData<Bgra32>(bytes, bmp.Width, bmp.Height);
                image.SaveAsPng(outputPath);
            }
            finally {
                bmp.UnlockBits(data);
            }

            return pngName;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] Failed to convert {fileName} to PNG: {ex.Message}");
            return null;
        }
    }
}
