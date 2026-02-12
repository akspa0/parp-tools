using System;
using System.IO;

namespace WoWRollback.PM4Module;

/// <summary>
/// Quick test to verify ADT merger output sizes.
/// </summary>
public static class TestMerger
{
    public static void RunTest(string testDataDir, string outputDir)
    {
        var patcher = new AdtPatcher();
        
        // Find a tile with all 3 files
        var rootAdt = Path.Combine(testDataDir, "development_0_0.adt");
        var obj0Adt = Path.Combine(testDataDir, "development_0_0_obj0.adt");
        var tex0Adt = Path.Combine(testDataDir, "development_0_0_tex0.adt");
        
        if (!File.Exists(rootAdt) || !File.Exists(obj0Adt) || !File.Exists(tex0Adt))
        {
            Console.WriteLine("[ERROR] Test files not found");
            return;
        }
        
        var rootSize = new FileInfo(rootAdt).Length;
        var obj0Size = new FileInfo(obj0Adt).Length;
        var tex0Size = new FileInfo(tex0Adt).Length;
        var totalInput = rootSize + obj0Size + tex0Size;
        
        Console.WriteLine("=== Input Files ===");
        Console.WriteLine($"Root: {rootSize:N0} bytes");
        Console.WriteLine($"Obj0: {obj0Size:N0} bytes");
        Console.WriteLine($"Tex0: {tex0Size:N0} bytes");
        Console.WriteLine($"Total: {totalInput:N0} bytes");
        Console.WriteLine();
        
        // Merge
        Console.WriteLine("=== Merging ===");
        var merged = patcher.MergeSplitAdt(rootAdt, obj0Adt, tex0Adt);
        
        Console.WriteLine();
        Console.WriteLine("=== Output ===");
        Console.WriteLine($"Merged: {merged.Length:N0} bytes");
        Console.WriteLine($"Ratio: {(double)merged.Length / totalInput:P1}");
        
        // Write output
        Directory.CreateDirectory(outputDir);
        var outputPath = Path.Combine(outputDir, "development_0_0_merged.adt");
        File.WriteAllBytes(outputPath, merged);
        Console.WriteLine($"Written to: {outputPath}");
        
        // Verify chunk structure
        Console.WriteLine();
        Console.WriteLine("=== Chunk Analysis ===");
        AnalyzeChunks(merged);
    }
    
    private static void AnalyzeChunks(byte[] data)
    {
        int pos = 0;
        var chunks = new Dictionary<string, (int count, long totalSize)>();
        
        while (pos < data.Length - 8)
        {
            var sig = System.Text.Encoding.ASCII.GetString(data, pos, 4);
            var size = BitConverter.ToInt32(data, pos + 4);
            
            if (size < 0 || pos + 8 + size > data.Length)
            {
                Console.WriteLine($"[WARN] Invalid chunk at {pos}: {sig} size={size}");
                break;
            }
            
            // Reverse sig for display
            var readableSig = new string(sig.Reverse().ToArray());
            
            if (!chunks.ContainsKey(readableSig))
                chunks[readableSig] = (0, 0);
            
            var (count, total) = chunks[readableSig];
            chunks[readableSig] = (count + 1, total + size);
            
            pos += 8 + size;
        }
        
        foreach (var kvp in chunks.OrderBy(k => k.Key))
        {
            Console.WriteLine($"  {kvp.Key}: {kvp.Value.count} chunk(s), {kvp.Value.totalSize:N0} bytes");
        }
        
        Console.WriteLine($"\nTotal chunks: {chunks.Values.Sum(v => v.count)}");
    }
}
