using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SereniaBLPLib;

namespace WmoBspConverter.Wmo;

/// <summary>
/// Converts WMO BLP textures to Q3-compatible TGA format with resizing.
/// Q3 requires power-of-two textures, max 512x512 for safe compatibility.
/// </summary>
public class WmoTextureConverter
{
    private const int MAX_TEXTURE_SIZE = 512;  // Q3 safe maximum
    private const int MIN_TEXTURE_SIZE = 8;    // Minimum reasonable size
    
    /// <summary>
    /// Convert all WMO textures to TGA format in the output directory.
    /// Returns mapping from original WMO paths to Q3-style relative paths.
    /// </summary>
    public Dictionary<string, string> ConvertTextures(
        List<string> wmoTextures,
        string sourceWmoDir,
        string outputDir,
        bool verbose = false)
    {
        var mapping = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        
        // Create textures output directory
        var texturesDir = Path.Combine(outputDir, "textures", "wmo");
        Directory.CreateDirectory(texturesDir);
        
        foreach (var texPath in wmoTextures)
        {
            if (string.IsNullOrWhiteSpace(texPath)) continue;
            
            // Clean up the WMO texture path
            var cleanPath = texPath.Replace('/', '\\').Trim();
            
            // Generate Q3-style output path
            var baseName = Path.GetFileNameWithoutExtension(cleanPath).ToLowerInvariant();
            var q3RelPath = $"textures/wmo/{baseName}.tga";
            var outputPath = Path.Combine(texturesDir, $"{baseName}.tga");
            
            // Track mapping regardless of conversion success
            mapping[texPath] = q3RelPath;
            
            if (File.Exists(outputPath))
            {
                if (verbose) Console.WriteLine($"[SKIP] Texture already exists: {q3RelPath}");
                continue;
            }
            
            // Try to find and convert the BLP
            var blpPath = FindBlpFile(cleanPath, sourceWmoDir);
            if (blpPath == null)
            {
                Console.WriteLine($"[WARN] Texture not found: {cleanPath}");
                continue;
            }
            
            try
            {
                ConvertBlpToTga(blpPath, outputPath, verbose);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to convert {cleanPath}: {ex.Message}");
            }
        }
        
        Console.WriteLine($"[INFO] Converted {mapping.Count} textures to TGA");
        return mapping;
    }
    
    /// <summary>
    /// Convert a single BLP file to TGA with power-of-two resizing.
    /// </summary>
    public void ConvertBlpToTga(string blpPath, string outputPath, bool verbose = false)
    {
        using var blpStream = File.OpenRead(blpPath);
        using var blp = new BlpFile(blpStream);
        
        // GetImage returns ImageSharp Image<Rgba32> directly
        using var image = blp.GetImage(0); // Mip level 0
        
        int newWidth = ClampToPowerOfTwo(image.Width);
        int newHeight = ClampToPowerOfTwo(image.Height);
        
        if (newWidth != image.Width || newHeight != image.Height)
        {
            if (verbose)
                Console.WriteLine($"[RESIZE] {Path.GetFileName(blpPath)}: {image.Width}x{image.Height} → {newWidth}x{newHeight}");
            
            image.Mutate(x => x.Resize(newWidth, newHeight));
        }
        
        // Save as TGA
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        image.SaveAsTga(outputPath);
        
        if (verbose)
            Console.WriteLine($"[OK] {Path.GetFileName(blpPath)} → {Path.GetFileName(outputPath)}");
    }
    
    /// <summary>
    /// Find the BLP file by searching common locations.
    /// </summary>
    private string? FindBlpFile(string texturePath, string sourceDir)
    {
        // Try direct path
        var direct = Path.Combine(sourceDir, texturePath);
        if (File.Exists(direct)) return direct;
        
        // Try with .blp extension if not present
        if (!texturePath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
        {
            var withExt = Path.Combine(sourceDir, texturePath + ".blp");
            if (File.Exists(withExt)) return withExt;
        }
        
        // Try searching parent directories (up to 5 levels)
        var current = new DirectoryInfo(sourceDir);
        for (int i = 0; i < 5 && current != null; i++)
        {
            var candidate = Path.Combine(current.FullName, texturePath);
            if (File.Exists(candidate)) return candidate;
            
            // Also try with .blp extension
            if (!texturePath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
            {
                candidate = Path.Combine(current.FullName, texturePath + ".blp");
                if (File.Exists(candidate)) return candidate;
            }
            
            current = current.Parent;
        }
        
        // Try filename-only search in source directory tree
        var fileName = Path.GetFileName(texturePath);
        if (!fileName.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
            fileName += ".blp";
            
        try
        {
            var found = Directory.EnumerateFiles(sourceDir, fileName, SearchOption.AllDirectories)
                .FirstOrDefault();
            if (found != null) return found;
        }
        catch { /* Ignore search errors */ }
        
        return null;
    }
    
    /// <summary>
    /// Clamp dimension to power-of-two within Q3 limits.
    /// </summary>
    private int ClampToPowerOfTwo(int value)
    {
        // Find nearest power of two
        int pot = MIN_TEXTURE_SIZE;
        while (pot < value && pot < MAX_TEXTURE_SIZE)
        {
            pot *= 2;
        }
        
        // Clamp to max
        return Math.Min(pot, MAX_TEXTURE_SIZE);
    }
}
