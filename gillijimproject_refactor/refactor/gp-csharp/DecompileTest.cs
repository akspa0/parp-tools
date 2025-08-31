using GillijimProject.WowFiles;
using System.Diagnostics;

public static class DecompileTest
{
    public static void RunTest(Options options)
    {
        Console.WriteLine($"Starting decompile/recompile test for: {options.FilePath}");
        Console.WriteLine();
        
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            // Step 1: Parse the original file
            Console.WriteLine("Step 1: Parsing original WDT file...");
            var originalData = File.ReadAllBytes(options.FilePath);
            var wdt = WdtContainer.LoadComplete(options.FilePath);
            
            Console.WriteLine($"✓ Parsed WDT with {wdt.AllChunks.Count} chunks");
            if (options.Verbose)
            {
                var chunkCounts = wdt.AllChunks.GroupBy(c => c.Tag)
                    .ToDictionary(g => g.Key, g => g.Count());
                
                Console.WriteLine("Chunk distribution:");
                long totalParsedSize = 0;
                foreach (var kvp in chunkCounts.OrderByDescending(x => x.Value))
                {
                    var tagBytes = BitConverter.GetBytes(kvp.Key);
                    // Reverse bytes for correct display (little-endian to big-endian)
                    var tagStr = $"{(char)tagBytes[3]}{(char)tagBytes[2]}{(char)tagBytes[1]}{(char)tagBytes[0]}";
                    
                    // Calculate size for this chunk type
                    var chunksOfType = wdt.AllChunks.Where(c => c.Tag == kvp.Key);
                    var typeSize = chunksOfType.Sum(c => c.ToBytes().Length);
                    totalParsedSize += typeSize;
                    
                    Console.WriteLine($"  {tagStr}: {kvp.Value} chunks ({typeSize:N0} bytes)");
                }
                
                Console.WriteLine($"\nTotal parsed chunk data: {totalParsedSize:N0} bytes");
                Console.WriteLine($"Original file size: {originalData.Length:N0} bytes");
                Console.WriteLine($"Missing data: {originalData.Length - totalParsedSize:N0} bytes ({((originalData.Length - totalParsedSize) * 100.0 / originalData.Length):F1}%)");
            }
            
            // Step 2: Recompile to bytes
            Console.WriteLine("\nStep 2: Recompiling to binary data...");
            var recompiledData = wdt.ToBytes();
            
            Console.WriteLine($"✓ Recompiled to {recompiledData.Length:N0} bytes");
            
            // Step 3: Compare byte-for-byte
            Console.WriteLine("\nStep 3: Validating byte-for-byte accuracy...");
            
            if (originalData.Length != recompiledData.Length)
            {
                Console.WriteLine($"❌ Size mismatch: Original={originalData.Length:N0}, Recompiled={recompiledData.Length:N0}");
                return;
            }
            
            bool identical = originalData.SequenceEqual(recompiledData);
            
            if (identical)
            {
                Console.WriteLine("✅ Perfect match! Decompile/recompile preserves 100% data fidelity");
            }
            else
            {
                Console.WriteLine("❌ Data mismatch detected");
                
                // Find first difference
                for (int i = 0; i < originalData.Length; i++)
                {
                    if (originalData[i] != recompiledData[i])
                    {
                        Console.WriteLine($"First difference at byte {i:N0} (0x{i:X8}):");
                        Console.WriteLine($"  Original: 0x{originalData[i]:X2}");
                        Console.WriteLine($"  Recompiled: 0x{recompiledData[i]:X2}");
                        break;
                    }
                }
            }
            
            stopwatch.Stop();
            Console.WriteLine($"\nTest completed in {stopwatch.ElapsedMilliseconds:N0}ms");
            
            // Optional: Write recompiled file for manual inspection
            if (options.OutputDirectory != null)
            {
                var outputPath = Path.Combine(options.OutputDirectory, "recompiled_" + Path.GetFileName(options.FilePath));
                File.WriteAllBytes(outputPath, recompiledData);
                Console.WriteLine($"Recompiled file written to: {outputPath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Test failed with exception: {ex.Message}");
            if (options.Verbose)
            {
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}
