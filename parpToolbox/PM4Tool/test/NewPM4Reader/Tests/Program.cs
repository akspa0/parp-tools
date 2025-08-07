using System;

namespace NewPM4Reader.Tests
{
    /// <summary>
    /// Test program for PM4 chunk implementations
    /// </summary>
    class Program
    {
        /// <summary>
        /// Main entry point
        /// </summary>
        static void Main(string[] args)
        {
            Console.WriteLine("PM4 Chunk Test Program");
            Console.WriteLine("=====================");
            Console.WriteLine();
            
            try
            {
                // Run all chunk tests
                ChunkTests.RunAllTests();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            
            Console.WriteLine();
            Console.WriteLine("Tests completed. Press any key to exit...");
            Console.ReadKey();
        }
    }
} 