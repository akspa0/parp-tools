using System;
using System.Threading.Tasks;
using System.IO;
using WoWToolbox.Core.v2.Services.WMO.Legacy;

namespace WoWToolbox.Core.v2.Services.WMO
{
    /// <summary>
    /// Temporary compatibility wrapper to satisfy existing tests that reference the legacy
    /// <c>WmoV14Converter</c> class name. Internally delegates to <see cref="FullV14Converter"/>.
    /// TODO: Remove once tests are updated to target the new API.
    /// </summary>
    public class WmoV14Converter
    {
        private readonly FullV14Converter _inner = new();

        public void ConvertToV17(string inputWmoPath, string outputWmoPath, string? textureOutputDir = null)
        {
            var outDir = Path.GetDirectoryName(outputWmoPath) ?? Path.GetTempPath();
            var task = _inner.ConvertAsync(inputWmoPath, outDir);
            task.GetAwaiter().GetResult();

            if (!File.Exists(task.Result.ConvertedWmoPath))
                throw new InvalidOperationException("Conversion failed – output file not produced.");

            // Move/rename resulting file to requested location
            File.Copy(task.Result.ConvertedWmoPath, outputWmoPath, overwrite:true);

            // Copy textures if requested
            if (textureOutputDir != null && task.Result.TexturePaths.Count > 0)
            {
                Directory.CreateDirectory(textureOutputDir);
                foreach (var tex in task.Result.TexturePaths)
                {
                    var dest = Path.Combine(textureOutputDir, Path.GetFileName(tex));
                    File.Copy(tex, dest, true);
                }
            }
        }

        public string ExportFirstGroupAsObj(string inputWmoPath)
        {
            return _inner.ExportFirstGroupAsObj(inputWmoPath);
        }

        public byte[] ConvertToV17(byte[] v14Data) => _inner.ConvertToV17(v14Data);

        public Task<WmoConversionResult> ConvertAsync(string inputWmoPath, string outputDir) => _inner.ConvertAsync(inputWmoPath, outputDir);
    }
}
