using System;
using System.IO;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.Core
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: WoWToolbox.Core <path_to_pm4_file>");
                return;
            }

            try
            {
                var pm4File = PM4File.FromFile(args[0]);
                
                Console.WriteLine($"PM4 File Version: {pm4File.MVER?.Version ?? 0}");
                
                if (pm4File.MSLK != null)
                {
                    Console.WriteLine($"\nMSLK Chunk: {pm4File.MSLK}");
                    // Console.WriteLine($"Version 48 Format: {pm4File.MSLK.ValidateVersion48Format()}"); // Removed invalid call
                    
                    foreach (var entry in pm4File.MSLK.Entries)
                    {
                        Console.WriteLine($"  {entry}");
                    }
                }
                else
                {
                    Console.WriteLine("\nNo MSLK chunk found");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parsing PM4 file: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
    }
} 