using System;
using System.IO;
using GillijimProject.WowFiles.LichKing;

namespace WoWRollback.LkToAlphaModule.Writers
{
    public static class LkAdtWriter
    {
        public static bool Write(AdtLk adt, string outputPath, bool validate = true)
        {
            if (adt == null) throw new ArgumentNullException(nameof(adt));
            if (string.IsNullOrWhiteSpace(outputPath)) throw new ArgumentException("Output path is required", nameof(outputPath));

            // Ensure output directory exists when a directory is provided
            if (Directory.Exists(outputPath))
            {
                // AdtLk.ToFile handles directory targets; no further action
            }
            else
            {
                var dir = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
            }

            if (validate && !adt.ValidateIntegrity())
            {
                // Attempt to self-correct via internal update path by re-serializing through ToFile
                // AdtLk.UpdateOrCreateMhdrAndMcin() is invoked by ToFile() when integrity fails
            }

            adt.ToFile(outputPath);

            return !validate || adt.ValidateIntegrity();
        }
    }
}
