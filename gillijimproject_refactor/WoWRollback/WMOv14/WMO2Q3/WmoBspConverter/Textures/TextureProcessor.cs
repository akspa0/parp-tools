using System;
using System.IO;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Tga;

using System.Collections.Generic;
using System.Linq;
using WoWFormatLib.FileReaders;

namespace WmoBspConverter.Textures
{
    /// <summary>
    /// Enhanced texture processor that handles BLP files and converts them for Quake 3 usage.
    /// Uses WoWFormatLib BLPReader for proper WoW texture format support.
    /// </summary>
    public class TextureProcessor
    {
        private readonly string _textureOutputDir;
        private readonly bool _extractTextures;
        
        public TextureProcessor(string outputDir, bool extractTextures = true)
        {
            _textureOutputDir = Path.Combine(outputDir, "textures", "wmo");
            _extractTextures = extractTextures;
            
            if (_extractTextures)
            {
                Directory.CreateDirectory(_textureOutputDir);
            }
        }

        /// <summary>
        /// Process a list of texture names and convert BLP files to Q3-compatible formats.
        /// </summary>
        /// <param name="textureNames">List of texture paths from WMO MOTX chunk</param>
        /// <returns>List of shader names and optional extracted texture files</returns>
        public async Task<List<TextureInfo>> ProcessTexturesAsync(List<string> textureNames)
        {
            var textureInfos = new List<TextureInfo>();
            
            foreach (var textureName in textureNames)
            {
                try
                {
                    var textureInfo = await ProcessTextureAsync(textureName);
                    textureInfos.Add(textureInfo);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] Failed to process texture '{textureName}': {ex.Message}");
                    // Add fallback texture
                    textureInfos.Add(new TextureInfo
                    {
                        OriginalName = textureName,
                        ShaderName = "textures/wmo/default",
                        IsValid = false,
                        Error = ex.Message
                    });
                }
            }

            return textureInfos;
        }

        private async Task<TextureInfo> ProcessTextureAsync(string texturePath)
        {
            var info = new TextureInfo
            {
                OriginalName = texturePath,
                ShaderName = ConvertTexturePathToShader(texturePath),
                IsValid = true
            };

            // If BLP extraction is disabled, just return the shader name
            if (!_extractTextures)
            {
                return info;
            }

            // Try to load and convert the BLP file
            var convertedTexture = await ConvertBLPToQ3Format(texturePath);
            if (convertedTexture != null)
            {
                info.OutputPath = convertedTexture.OutputPath;
                info.Width = convertedTexture.Width;
                info.Height = convertedTexture.Height;
                info.Format = convertedTexture.Format;
            }
            else
            {
                // Fallback to placeholder if BLP conversion fails
                var placeholderResult = await CreatePlaceholderTextureAsync(texturePath);
                if (placeholderResult != null)
                {
                    info.OutputPath = placeholderResult.OutputPath;
                    info.Width = placeholderResult.Width;
                    info.Height = placeholderResult.Height;
                    info.Format = placeholderResult.Format;
                    info.Error = "Used placeholder - BLP conversion failed";
                }
                else
                {
                    info.IsValid = false;
                    info.Error = "Failed to create placeholder texture";
                }
            }

            return info;
        }

        /// <summary>
        /// Convert BLP texture to Q3-compatible TGA format using WoWFormatLib.
        /// </summary>
        private async Task<TextureInfo?> ConvertBLPToQ3Format(string texturePath)
        {
            if (string.IsNullOrEmpty(texturePath))
                return null;

            // Clean up the texture path
            var cleanPath = texturePath.Replace("\\", "/");
            var fileName = Path.GetFileNameWithoutExtension(cleanPath);
            var fileNameLower = fileName.ToLowerInvariant();
            var outputFileName = $"{fileNameLower}.tga";
            var outputPath = Path.Combine(_textureOutputDir, outputFileName);

            // Skip if already processed
            if (File.Exists(outputPath))
            {
                var imageInfo = await Image.IdentifyAsync(outputPath);
                return new TextureInfo
                {
                    OriginalName = texturePath,
                    OutputPath = outputPath,
                    ShaderName = ConvertTexturePathToShader(texturePath),
                    Width = imageInfo?.Width ?? 0,
                    Height = imageInfo?.Height ?? 0,
                    Format = "TGA"
                };
            }

            try
            {
                // Try to load the actual BLP file using WoWFormatLib
                // Note: This requires access to WoW data files or file data IDs
                var blpImage = await LoadBLPTextureAsync(texturePath);
                
                if (blpImage != null)
                {
                    await blpImage.SaveAsTgaAsync(outputPath, new TgaEncoder
                    {
                        BitsPerPixel = TgaBitsPerPixel.Pixel24,
                        Compression = TgaCompression.None
                    });
                    return new TextureInfo
                    {
                        OriginalName = texturePath,
                        OutputPath = outputPath,
                        ShaderName = ConvertTexturePathToShader(texturePath),
                        Width = blpImage.Width,
                        Height = blpImage.Height,
                        Format = "TGA"
                    };
                }
                
                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DEBUG] Failed to convert texture '{texturePath}': {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Load BLP texture using WoWFormatLib BLPReader.
        /// </summary>
        private Task<Image<Rgba32>?> LoadBLPTextureAsync(string texturePath)
        {
            try
            {
                // For now, return null since we don't have file data IDs
                // In a full implementation, this would resolve the texture path to a file data ID
                // and use BLPReader to load the actual texture data
                
                // Placeholder: try to find a BLP file with the same name
                var blpPath = FindBLPFile(texturePath);
                if (!string.IsNullOrEmpty(blpPath) && File.Exists(blpPath))
                {
                    using var stream = File.OpenRead(blpPath);
                    var reader = new BLPReader();
                    reader.LoadBLP(stream);
                    return Task.FromResult(reader.bmp?.CloneAs<Rgba32>());
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DEBUG] BLP loading failed for '{texturePath}': {ex.Message}");
            }
            
            return Task.FromResult<Image<Rgba32>?>(null);
        }

        /// <summary>
        /// Find a BLP file corresponding to the texture path.
        /// Maps WMO texture paths to local filesystem locations.
        /// </summary>
        private string? FindBLPFile(string texturePath)
        {
            if (string.IsNullOrEmpty(texturePath))
                return null;
                
            // Convert texture path to potential BLP file location
            var cleanPath = texturePath.Replace("\\", "/").Replace("Dungeons/Textures", "Dungeons/Textures");
            var fileName = Path.GetFileNameWithoutExtension(cleanPath);
            
            Console.WriteLine($"[DEBUG] Looking for BLP file: {fileName}");
            
            // Extract directory information from the texture path
            var pathParts = cleanPath.Split('/');
            var subfolder = "";
            if (pathParts.Length > 2)
            {
                subfolder = pathParts[pathParts.Length - 2]; // e.g., "ceiling", "walls", etc.
            }
            
            // Only try .blp extension first (most common)
            var extensions = new[] { ".blp", ".BLP" };
            
            foreach (var ext in extensions)
            {
                var candidate = $"{fileName}{ext}";
                
                // Try the most specific path first - exact directory match
                if (!string.IsNullOrEmpty(subfolder))
                {
                    // Check parent directory first (..\test_data)
                    var exactPath = Path.Combine("..", "test_data", "Dungeons", "Textures", subfolder.ToLower(), candidate);
                    Console.WriteLine($"[DEBUG] Checking exact path: {exactPath}");
                    if (File.Exists(exactPath))
                    {
                        Console.WriteLine($"[INFO] Found BLP file: {exactPath}");
                        return Path.GetFullPath(exactPath); // Return full path to ensure consistency
                    }
                    
                    // Try with original case
                    exactPath = Path.Combine("..", "test_data", "Dungeons", "Textures", subfolder, candidate);
                    Console.WriteLine($"[DEBUG] Checking exact path: {exactPath}");
                    if (File.Exists(exactPath))
                    {
                        Console.WriteLine($"[INFO] Found BLP file: {exactPath}");
                        return Path.GetFullPath(exactPath);
                    }
                }
                
                // Try other common directories
                var searchLocations = new[]
                {
                    // Test data structures in parent directory
                    Path.Combine("..", "test_data", "Dungeons", "Textures", "trim", candidate),
                    Path.Combine("..", "test_data", "Dungeons", "Textures", "walls", candidate),
                    Path.Combine("..", "test_data", "Dungeons", "Textures", "floor", candidate),
                    Path.Combine("..", "test_data", "Dungeons", "Textures", "roof", candidate),
                    Path.Combine("..", "test_data", "Dungeons", "Textures", "ceiling", candidate),
                    Path.Combine("..", "test_data", "Dungeons", "Textures", "brick", candidate),
                    Path.Combine("..", "test_data", "Dungeons", "Textures", candidate),
                    
                    // Local test data structures
                    Path.Combine("test_data", "Dungeons", "Textures", "trim", candidate),
                    Path.Combine("test_data", "Dungeons", "Textures", "walls", candidate),
                    Path.Combine("test_data", "Dungeons", "Textures", "floor", candidate),
                    Path.Combine("test_data", "Dungeons", "Textures", "roof", candidate),
                    Path.Combine("test_data", "Dungeons", "Textures", "ceiling", candidate),
                    Path.Combine("test_data", "Dungeons", "Textures", "brick", candidate),
                    Path.Combine("test_data", "Dungeons", "Textures", candidate),
                    
                    // Current directory and common patterns
                    candidate,
                    Path.Combine("textures", candidate),
                    Path.Combine("Textures", candidate)
                };
                
                foreach (var location in searchLocations)
                {
                    Console.WriteLine($"[DEBUG] Checking: {location}");
                    if (File.Exists(location))
                    {
                        Console.WriteLine($"[INFO] Found BLP file: {location}");
                        return Path.GetFullPath(location);
                    }
                }
            }
            
            // Last resort: try to find any BLP file with similar name in the entire test_data directory
            try
            {
                var testDataDir = Path.Combine("..", "test_data");
                if (Directory.Exists(testDataDir))
                {
                    var blpFiles = Directory.GetFiles(testDataDir, $"{fileName}.*", SearchOption.AllDirectories);
                    if (blpFiles.Length > 0)
                    {
                        var foundFile = blpFiles[0];
                        Console.WriteLine($"[INFO] Found BLP file via recursive search: {foundFile}");
                        return foundFile;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[DEBUG] Recursive search failed: {ex.Message}");
            }
            
            Console.WriteLine($"[WARN] Could not find BLP file for: {texturePath}");
            return null;
        }

        /// <summary>
        /// Create a placeholder texture when actual BLP conversion fails.
        /// </summary>
        private async Task<TextureInfo?> CreatePlaceholderTextureAsync(string texturePath)
        {
            try
            {
                var width = 256;
                var height = 256;
                var image = new Image<Rgba32>(width, height);

                var background = new Rgba32(128, 128, 128, 255);
                var borderColor = new Rgba32(255, 0, 0, 255);

                // Fill background
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
                    {
                        image[x, y] = background;
                    }
                }

                // Draw border
                for (int x = 0; x < width; x++)
                {
                    image[x, 0] = borderColor;
                    image[x, height - 1] = borderColor;
                }
                for (int y = 0; y < height; y++)
                {
                    image[0, y] = borderColor;
                    image[width - 1, y] = borderColor;
                }

                var cleanPath = texturePath.Replace("\\", "/");
                var fileName = Path.GetFileNameWithoutExtension(cleanPath).ToLowerInvariant();
                var outputFileName = $"{fileName}_placeholder.tga";
                var outputPath = Path.Combine(_textureOutputDir, outputFileName);
                
                await image.SaveAsTgaAsync(outputPath, new TgaEncoder
                {
                    BitsPerPixel = TgaBitsPerPixel.Pixel24,
                    Compression = TgaCompression.None
                });
                
                return new TextureInfo
                {
                    OriginalName = texturePath,
                    OutputPath = outputPath,
                    ShaderName = ConvertTexturePathToShader(texturePath),
                    Width = width,
                    Height = height,
                    Format = "TGA"
                };
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Convert WMO texture path to Quake 3 shader name.
        /// </summary>
        private string ConvertTexturePathToShader(string texturePath)
        {
            if (string.IsNullOrEmpty(texturePath))
                return "wmo/default";

            // Normalize path
            var cleanPath = texturePath.Replace("\\", "/");

            // Return wmo/<file> without "textures/" prefix - GtkRadiant adds it automatically
            var fileNameNoExt = Path.GetFileNameWithoutExtension(cleanPath).ToLowerInvariant();
            return $"wmo/{fileNameNoExt}";
        }

        /// <summary>
        /// Generate shader scripts for Quake 3 based on WMO material properties.
        /// </summary>
        public void GenerateShaderScripts(List<WmoBspConverter.Wmo.WmoMaterial> materials, List<TextureInfo> textures)
        {
            var shaderDir = Path.Combine(_textureOutputDir, "shaders");
            Directory.CreateDirectory(shaderDir);

            var shaderScript = GenerateShaderScript(materials, textures);
            var shaderPath = Path.Combine(shaderDir, "wmo_textures.shader");
            
            File.WriteAllText(shaderPath, shaderScript);
            Console.WriteLine($"[INFO] Generated shader script: {shaderPath}");
        }

        /// <summary>
        /// Generate a simple shader script that maps each processed texture to itself (no extension in map path).
        /// </summary>
        public void GenerateShaderScripts(List<TextureInfo> textures)
        {
            var shaderDir = Path.Combine(_textureOutputDir, "shaders");
            Directory.CreateDirectory(shaderDir);

            var lines = new List<string>();
            lines.Add("// Simple WMO texture shaders (auto-generated)");
            lines.Add("");
            foreach (var tex in textures)
            {
                // Use the shader name (textures/...) without extension in the map directive
                lines.Add(tex.ShaderName);
                lines.Add("{");
                lines.Add("    { ");
                lines.Add($"        map {tex.ShaderName}");
                lines.Add("        rgbGen vertex");
                lines.Add("    }");
                lines.Add("}");
                lines.Add("");
            }

            var shaderPath = Path.Combine(shaderDir, "wmo_textures.shader");
            File.WriteAllText(shaderPath, string.Join("\n", lines));
            Console.WriteLine($"[INFO] Generated shader script: {shaderPath}");
        }

        /// <summary>
        /// Generate the actual shader script content.
        /// </summary>
        private string GenerateShaderScript(List<WmoBspConverter.Wmo.WmoMaterial> materials, List<TextureInfo> textures)
        {
            var shaderLines = new List<string>();
            shaderLines.Add("// Generated shader script for WMO textures");
            shaderLines.Add("// This script maps WMO materials to Quake 3 shader properties");
            shaderLines.Add("");

            for (int i = 0; i < materials.Count && i < textures.Count; i++)
            {
                var material = materials[i];
                var texture = textures[i];
                
                shaderLines.Add($"// Original texture: {texture.OriginalName}");
                shaderLines.Add($"// Material flags: 0x{material.Flags:X8}, Shader: {material.Shader}, BlendMode: {material.BlendMode}");
                shaderLines.Add("{");
                shaderLines.Add($"    q3map_lightmapSize 256 256");
                
                // Map WMO material properties to Q3 shader flags
                if ((material.Flags & 0x1) != 0) // F_UNLIT
                {
                    shaderLines.Add("    q3map_sunlight");
                }
                
                if ((material.Flags & 0x2) != 0) // F_UNFOGGED
                {
                    shaderLines.Add("    q3map_nofog");
                }
                
                if ((material.Flags & 0x4) != 0) // F_UNCULLED
                {
                    shaderLines.Add("    cull twosided");
                }
                
                // Blend mode handling
                switch (material.BlendMode)
                {
                    case 0: // Opaque
                        shaderLines.Add("    blendFunc GL_ONE GL_ZERO");
                        break;
                    case 1: // Alpha test
                        shaderLines.Add("    blendFunc GL_SRC_ALPHA GL_ONE_MINUS_SRC_ALPHA");
                        break;
                    case 2: // Additive
                        shaderLines.Add("    blendFunc GL_SRC_ALPHA GL_ONE");
                        break;
                    case 3: // Modulate
                        shaderLines.Add("    blendFunc GL_ZERO GL_SRC_COLOR");
                        break;
                }
                
                // Use shader path without extension; Q3 will load .tga/.jpg as available
                shaderLines.Add($"    map {texture.ShaderName}");
                shaderLines.Add("}");
                shaderLines.Add("");
            }

            return string.Join("\n", shaderLines);
        }
    }

    /// <summary>
    /// Information about a processed texture.
    /// </summary>
    public class TextureInfo
    {
        public string OriginalName { get; set; } = "";
        public string ShaderName { get; set; } = "";
        public string? OutputPath { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public string Format { get; set; } = "";
        public bool IsValid { get; set; } = true;
        public string? Error { get; set; }
    }
}