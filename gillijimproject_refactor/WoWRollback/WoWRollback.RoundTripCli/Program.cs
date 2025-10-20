using System;
using System.IO;
using WoWRollback.LkToAlphaModule.Validators;

namespace WoWRollback.RoundTripCli
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            try
            {
                if (args.Length == 0 || Array.IndexOf(args, "--help") >= 0)
                {
                    PrintHelp();
                    return 0;
                }

                string? wdtAlpha = GetArgValue(args, "--wdt-alpha");
                string? adtLk = GetArgValue(args, "--adt-lk");
                string? outDir = GetArgValue(args, "--out");

                if (string.IsNullOrWhiteSpace(outDir))
                {
                    Console.Error.WriteLine("--out <dir> is required");
                    return 2;
                }

                Directory.CreateDirectory(outDir);

                var options = new WoWRollback.LkToAlphaModule.LkToAlphaOptions 
                { 
                    VerboseLogging = false,  // Disable verbose logging for cleaner output
                    UseManagedBuilders = false 
                };

                if (!string.IsNullOrWhiteSpace(wdtAlpha))
                {
                    if (!File.Exists(wdtAlpha))
                    {
                        Console.Error.WriteLine($"Alpha WDT not found: {wdtAlpha}");
                        return 3;
                    }

                    var result = RoundTripValidator.ValidateRoundTrip(wdtAlpha, outDir, options, writeIntermediates: true);
                    PrintResult(result);
                    return 0;
                }
                else if (!string.IsNullOrWhiteSpace(adtLk))
                {
                    if (!File.Exists(adtLk))
                    {
                        Console.Error.WriteLine($"LK ADT not found: {adtLk}");
                        return 3;
                    }

                    var result = RoundTripValidator.ValidateRoundTripFromLkAdt(adtLk, outDir, options);
                    PrintResult(result);
                    return 0;
                }

                Console.Error.WriteLine("Provide either --wdt-alpha <path> or --adt-lk <path>");
                return 2;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"RoundTrip failed: {ex.Message}\n{ex.StackTrace}");
                return 1;
            }
        }

        private static string? GetArgValue(string[] args, string name)
        {
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                {
                    return args[i + 1];
                }
            }
            return null;
        }

        private static void PrintHelp()
        {
            Console.WriteLine("WoWRollback RoundTrip CLI");
            Console.WriteLine("Usage:");
            Console.WriteLine("  roundtrip --wdt-alpha <path> --out <dir>");
            Console.WriteLine("  roundtrip --adt-lk <path> --out <dir>");
        }

        private static void PrintResult(RoundTripValidator.ValidationResult result)
        {
            Console.WriteLine(result.Success
                ? "[RoundTrip] SUCCESS"
                : "[RoundTrip] FAILURE");
            if (!string.IsNullOrWhiteSpace(result.ErrorMessage))
            {
                Console.WriteLine(result.ErrorMessage);
            }
            Console.WriteLine($"TilesProcessed={result.TilesProcessed} TilesMismatched={result.TilesMismatched} BytesDifferent={result.BytesDifferent}");
        }
    }
}
