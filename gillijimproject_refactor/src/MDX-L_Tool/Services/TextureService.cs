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
    private readonly DbcService _dbcService;

    public TextureService(NativeMpqService mpqService, DbcService dbcService)
    {
        _mpqService = mpqService;
        _dbcService = dbcService;
    }

    /// <summary>
    /// Attempts to resolve a texture and convert it to PNG.
    /// </summary>
    public string? ExportTexture(MdlTexture texture, string modelName, string modelPath, string modelRoot, string outputDir)
    {
        string? texturePath = texture.Path;
        
        // Handle ReplaceableId
        if (texture.ReplaceableId != 0 && string.IsNullOrWhiteSpace(texturePath))
        {
            texturePath = ResolveReplaceablePath(texture.ReplaceableId, modelName, modelPath);
            if (texturePath != null)
                Console.WriteLine($"  Resolved ReplaceableId {texture.ReplaceableId} to: \"{texturePath}\"");
        }

        if (string.IsNullOrWhiteSpace(texturePath)) return null;

        // 1. Try local lookup (same folder as MDX)
        var fileName = Path.GetFileName(texturePath);
        var localPath = Path.Combine(modelRoot, fileName);
        if (File.Exists(localPath))
        {
            return ConvertBlpToPng(localPath, outputDir);
        }

        // 2. Try MPQ lookup
        var mpqNormalized = texturePath.Replace('/', '\\');
        if (_mpqService.FileExists(mpqNormalized))
        {
            var data = _mpqService.ReadFile(mpqNormalized);
            if (data != null)
            {
                return ConvertBlpToPng(data, fileName, outputDir);
            }
        }

        // 3. Try variations (e.g. .blp extension)
        var blpPath = Path.ChangeExtension(mpqNormalized, ".blp");
        if (blpPath != mpqNormalized && _mpqService.FileExists(blpPath))
        {
            var data = _mpqService.ReadFile(blpPath);
            if (data != null)
            {
                return ConvertBlpToPng(data, Path.GetFileName(blpPath), outputDir);
            }
        }

        return null;
    }

    private string? ResolveReplaceablePath(uint replaceableId, string modelName, string modelPath)
    {
        // 1. Try DBC lookup
        var variations = _dbcService.GetVariations(modelPath);
        
        // Mapping convention:
        // ID 11, 12... -> Variation index 0, 1...
        // ID 1, 2...   -> Variation index 0, 1... (for some models)
        int index = -1;
        if (replaceableId >= 11) index = (int)(replaceableId - 11);
        else if (replaceableId >= 1) index = (int)(replaceableId - 1);

        if (index >= 0 && index < variations.Count)
        {
            var varName = variations[index];
            if (!varName.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                varName += ".blp";
            return varName;
        }

        // 2. Fallbacks
        switch (replaceableId)
        {
            case 11:
            case 1:
                return $"{modelName}Skin.blp";
            default:
                return null;
        }
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
