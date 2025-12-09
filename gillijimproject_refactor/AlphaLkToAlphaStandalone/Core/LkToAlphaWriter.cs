using System;
using System.IO;
using GillijimProject.WowFiles;

namespace AlphaLkToAlphaStandalone.Core
{
    internal static class LkToAlphaWriter
    {
        public static void Plan(LkToAlphaConversionOptions options, int matchedAdtCount)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));

            var planPath = Path.Combine(options.AlphaOutputDirectory, "alpha_plan.txt");

            try
            {
                Directory.CreateDirectory(options.AlphaOutputDirectory);

                var lines = new[]
                {
                    "Alpha LK→Alpha conversion plan (scaffold)",
                    $"MapName: {options.MapName}",
                    $"LK root: {options.LkRootDirectory}",
                    $"LK map dir: {options.LkMapDirectory}",
                    $"Planned Alpha WDT: " + Path.Combine(options.AlphaOutputDirectory, options.MapName + ".wdt"),
                    $"Planned Alpha ADT count (from matched LK ADTs): {matchedAdtCount}",
                    "",
                    "NOTE: This file is a scaffold-only plan; a stub Alpha WDT may be written separately based on tile occupancy."
                };

                File.WriteAllLines(planPath, lines);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Warning: failed to write alpha_plan.txt: {ex.Message}");
            }
        }

        public static void WriteAlphaWdtStub(LkToAlphaConversionOptions options)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));

            var tilesCsvPath = Path.Combine(options.DiagnosticsDirectory, "tiles_summary.csv");
            if (!File.Exists(tilesCsvPath))
            {
                Console.Error.WriteLine($"Warning: tiles_summary.csv not found at {tilesCsvPath}; skipping Alpha WDT stub.");
                return;
            }

            try
            {
                var tiles = new bool[64, 64];
                bool first = true;
                foreach (var line in File.ReadLines(tilesCsvPath))
                {
                    if (first)
                    {
                        first = false;
                        continue;
                    }

                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    var parts = line.Split(',');
                    if (parts.Length < 3)
                    {
                        continue;
                    }

                    if (!int.TryParse(parts[0], out var x) || !int.TryParse(parts[1], out var y))
                    {
                        continue;
                    }

                    if (!int.TryParse(parts[2], out var present))
                    {
                        continue;
                    }

                    if (x < 0 || x >= 64 || y < 0 || y >= 64)
                    {
                        continue;
                    }

                    tiles[x, y] = present != 0;
                }

                var mverData = new byte[4];
                BitConverter.GetBytes(18).CopyTo(mverData, 0);
                var mver = new Chunk(ChunkHeaders.MVER, mverData.Length, mverData);

                var mphdData = new byte[128];
                var mphd = new Mphd(ChunkHeaders.MPHD, mphdData.Length, mphdData);

                const int entrySize = 16;
                var mainData = new byte[64 * 64 * entrySize];
                int nextOffset = 0x1000;
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        int index = y * 64 + x;
                        int baseOffset = index * entrySize;
                        uint offsetValue = 0;
                        if (tiles[x, y])
                        {
                            offsetValue = (uint)nextOffset;
                            nextOffset += 0x1000;
                        }

                        BitConverter.GetBytes(offsetValue).CopyTo(mainData, baseOffset + 0);
                    }
                }

                var main = new Chunk(ChunkHeaders.MAIN, mainData.Length, mainData);

                var mverBytes = mver.GetWholeChunk();
                var mphdBytes = mphd.GetWholeChunk();
                var mainBytes = main.GetWholeChunk();

                var totalLength = mverBytes.Length + mphdBytes.Length + mainBytes.Length;
                var buffer = new byte[totalLength];
                Buffer.BlockCopy(mverBytes, 0, buffer, 0, mverBytes.Length);
                Buffer.BlockCopy(mphdBytes, 0, buffer, mverBytes.Length, mphdBytes.Length);
                Buffer.BlockCopy(mainBytes, 0, buffer, mverBytes.Length + mphdBytes.Length, mainBytes.Length);

                Directory.CreateDirectory(options.AlphaOutputDirectory);
                var outPath = Path.Combine(options.AlphaOutputDirectory, options.MapName + ".wdt");
                File.WriteAllBytes(outPath, buffer);
                Console.WriteLine($"Wrote stub Alpha WDT to {outPath}");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Warning: failed to write stub Alpha WDT: {ex.Message}");
            }
        }
    }
}
