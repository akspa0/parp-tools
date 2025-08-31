using System.Text.Json;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.IO;

public class Program
{
    public static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: ChunkParser <file> [--output <dir>]");
            Console.WriteLine("  Scans file for ALL chunk types and exports complete inventory");
            return;
        }

        string filePath = args[0];
        string? outputDir = null;
        
        for (int i = 1; i < args.Length; i++)
        {
            if (args[i] == "--output" && i + 1 < args.Length)
            {
                outputDir = args[++i];
            }
        }

        var scanner = new UniversalChunkScanner();
        var result = scanner.ScanFile(filePath);
        
        // Create output directory
        var fileName = Path.GetFileNameWithoutExtension(filePath);
        var timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        outputDir ??= Path.Combine("chunk_analysis", $"{fileName}_{timestamp}");
        Directory.CreateDirectory(outputDir);
        
        // Export results
        var jsonPath = Path.Combine(outputDir, "chunks.json");
        var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(result, jsonOptions));
        
        var summaryPath = Path.Combine(outputDir, "summary.txt");
        WriteSummary(result, summaryPath);
        
        Console.WriteLine($"Chunk analysis complete:");
        Console.WriteLine($"  File: {filePath} ({result.FileSizeBytes:N0} bytes)");
        Console.WriteLine($"  Chunks found: {result.Chunks.Count}");
        Console.WriteLine($"  Unique types: {result.ChunkTypes.Count}");
        Console.WriteLine($"  JSON: {jsonPath}");
        Console.WriteLine($"  Summary: {summaryPath}");
    }
    
    private static void WriteSummary(ChunkScanResult result, string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine($"File: {result.FileName}");
        writer.WriteLine($"Size: {result.FileSizeBytes:N0} bytes");
        writer.WriteLine($"Scan time: {result.ScanTime}");
        writer.WriteLine($"Total chunks: {result.Chunks.Count}");
        writer.WriteLine();
        
        writer.WriteLine("Chunk type summary:");
        foreach (var type in result.ChunkTypes.OrderByDescending(t => t.Count))
        {
            writer.WriteLine($"  {type.FourCC}: {type.Count} chunks, {type.TotalSize:N0} bytes");
        }
        
        writer.WriteLine();
        writer.WriteLine("All chunks (offset, size, type):");
        foreach (var chunk in result.Chunks.OrderBy(c => c.Offset))
        {
            writer.WriteLine($"  0x{chunk.Offset:X8}: {chunk.Size,8} bytes - {chunk.FourCC}");
        }
    }
}

public class UniversalChunkScanner
{
    public ChunkScanResult ScanFile(string filePath)
    {
        var startTime = DateTime.Now;
        var result = new ChunkScanResult
        {
            FileName = Path.GetFileName(filePath),
            FileSizeBytes = new FileInfo(filePath).Length,
            ScanTime = startTime
        };
        
        var fileInfo = new FileInfo(filePath);
        long fileSize = fileInfo.Length;
        int numThreads = Environment.ProcessorCount;
        long chunkSize = fileSize / numThreads;
        
        Console.WriteLine($"Scanning {fileSize:N0} byte file with {numThreads} threads...");
        
        var chunks = new ConcurrentBag<ChunkInfo>();
        var tasks = new List<Task>();
        
        for (int i = 0; i < numThreads; i++)
        {
            long startPos = i * chunkSize;
            long endPos = (i == numThreads - 1) ? fileSize : (i + 1) * chunkSize;
            
            tasks.Add(Task.Run(() => ScanFileSegment(filePath, startPos, endPos, chunks, i)));
        }
        
        Task.WaitAll(tasks.ToArray());
        
        result.Chunks = chunks.OrderBy(c => c.Offset).ToList();
        
        // Calculate statistics
        var typeGroups = result.Chunks.GroupBy(c => c.FourCC).ToList();
        result.ChunkTypes = typeGroups.Select(g => new ChunkTypeInfo
        {
            FourCC = g.Key,
            Count = g.Count(),
            TotalSize = g.Sum(c => (long)c.Size)
        }).ToList();
        
        Console.WriteLine($"Scan complete: {result.Chunks.Count} chunks found");
        return result;
    }
    
    private static void ScanFileSegment(string filePath, long startPos, long endPos, ConcurrentBag<ChunkInfo> chunks, int threadId)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var buffer = new byte[8];
        long position = startPos;
        
        // Align to potential chunk boundary
        if (position > 0)
        {
            position = AlignToPotentialChunk(fs, position, buffer);
        }
        
        long lastProgress = 0;
        
        while (position < endPos - 8 && position < fs.Length - 8)
        {
            fs.Seek(position, SeekOrigin.Begin);
            int read = fs.Read(buffer, 0, 8);
            if (read < 8) break;
            
            uint tag = BitConverter.ToUInt32(buffer, 0);
            uint size = BitConverter.ToUInt32(buffer, 4);
            
            if (IsValidChunkTag(tag) && IsValidChunkSize(size, fs.Length - position))
            {
                var chunk = new ChunkInfo
                {
                    Offset = position,
                    Size = size,
                    FourCC = FourCCToString(tag),
                    Tag = tag
                };
                
                chunks.Add(chunk);
                
                // Skip to next potential chunk
                position += 8 + size;
                
                // Align to 4-byte boundary
                if (position % 4 != 0)
                    position += 4 - (position % 4);
            }
            else
            {
                position++;
            }
            
            // Progress for this thread
            if (position - lastProgress > 50 * 1024 * 1024) // Every 50MB
            {
                double progress = (double)(position - startPos) / (endPos - startPos) * 100;
                Console.WriteLine($"  Thread {threadId}: {progress:F1}% complete");
                lastProgress = position;
            }
        }
    }
    
    private static long AlignToPotentialChunk(FileStream fs, long position, byte[] buffer)
    {
        // Look backwards for a potential chunk boundary
        long searchStart = Math.Max(0, position - 1024);
        fs.Seek(searchStart, SeekOrigin.Begin);
        
        while (searchStart < position)
        {
            int read = fs.Read(buffer, 0, 8);
            if (read < 8) break;
            
            uint tag = BitConverter.ToUInt32(buffer, 0);
            if (IsValidChunkTag(tag))
            {
                return searchStart;
            }
            searchStart++;
        }
        
        return position;
    }
    
    private static bool IsValidChunkTag(uint tag)
    {
        var fourCC = FourCCToString(tag);
        
        // Known WoW chunk types - using documentation format (M-first)
        // File stores them little-endian reversed, so we reverse when reading
        var validChunks = new HashSet<string>
        {
            // Core WDT chunks
            "MVER", "MPHD", "MAIN", "MHDR", "MCIN", "MCNK", "MCVT", "MCNR", "MCLQ", "MCAL", "MCRF",
            // Object chunks  
            "MDDF", "MODF", "MMDX", "MMID", "MWMO", "MWID",
            // Alpha chunks
            "MDNM", "MONM",
            // Water chunks
            "MH2O", "MCLV",
            // Texture chunks
            "MTEX", "MCLY", "MCSH", "MCCV"
        };
        
        return validChunks.Contains(fourCC);
    }
    
    private static bool IsValidChunkSize(uint size, long remainingBytes)
    {
        // Reasonable size limits
        return size > 0 && size < remainingBytes && size < 100 * 1024 * 1024; // Max 100MB per chunk
    }
    
    private static string FourCCToString(uint tag)
    {
        var bytes = BitConverter.GetBytes(tag);
        // Reverse bytes to match documentation format (M-first)
        Array.Reverse(bytes);
        return System.Text.Encoding.ASCII.GetString(bytes);
    }
}

public class ChunkScanResult
{
    public string FileName { get; set; } = "";
    public long FileSizeBytes { get; set; }
    public DateTime ScanTime { get; set; }
    public List<ChunkInfo> Chunks { get; set; } = new();
    public List<ChunkTypeInfo> ChunkTypes { get; set; } = new();
}

public class ChunkInfo
{
    public long Offset { get; set; }
    public uint Size { get; set; }
    public string FourCC { get; set; } = "";
    public uint Tag { get; set; }
}

public class ChunkTypeInfo
{
    public string FourCC { get; set; } = "";
    public int Count { get; set; }
    public long TotalSize { get; set; }
}
