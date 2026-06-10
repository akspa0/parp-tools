using System;
using System.Collections.Generic;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Warcraft.NET.Files.BLP;
using BLPFile = Warcraft.NET.Files.BLP.BLP;

namespace WoWToolbox.Core.v2.Foundation.WMO
{
    /// <summary>
    /// Utility to locate referenced BLP textures inside the same directory tree as the source WMO
    /// and extract them as PNG using ImageSharp. Designed for unit-test usage; not a full asset DB.
    /// </summary>
    public static class WmoTextureExtractor
    {
        /// <param name="wmoPath">Full path of the original WMO file (used to resolve sibling BLPs).</param>
        /// <param name="textureNames">Strings from MOTX table (e.g. "Textures\\Baked\\Ironforge_Stone01.blp").</param>
        /// <param name="outputDir">Directory where PNGs should be written.</param>
        /// <returns>List of PNG filenames actually written (relative to <paramref name="outputDir"/>).</returns>
        // Backwards-compat alias used by converter
        public static List<string> Extract(IReadOnlyList<string> textureNames, string sourceDir, string outputDir)
        {
            // Fake WMO path assembled from first texture to reuse internal logic; only directory matters.
            string dummyWmo = Path.Combine(sourceDir, "dummy.wmo");
            return ExtractTextures(dummyWmo, textureNames, outputDir);
        }

        public static List<string> ExtractTextures(string wmoPath, IReadOnlyList<string> textureNames, string outputDir)
        {
            var written = new List<string>();
            if (textureNames.Count == 0) return written;
            string wmoDir = Path.GetDirectoryName(wmoPath)!;
            Directory.CreateDirectory(outputDir);

            foreach (string tex in textureNames)
            {
                string clean = tex.Replace("/", Path.DirectorySeparatorChar.ToString())
                                  .Replace("\\", Path.DirectorySeparatorChar.ToString());
                string candidate = Path.Combine(wmoDir, clean);
                if (!File.Exists(candidate))
                    continue; // texture not shipped in sample dataset

                byte[] blpBytes = File.ReadAllBytes(candidate);
                var blp = new BLPFile(blpBytes);
                using var img = blp.GetMipMap(0);
                string pngName = Path.GetFileNameWithoutExtension(clean) + ".png";
                string pngPath = Path.Combine(outputDir, pngName);
                img.Save(pngPath);
                written.Add(pngName);
            }
            return written;
        }
    }
}
