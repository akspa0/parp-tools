using Xunit;
using WoWToolbox.Core.WMO;
using WoWToolbox.Core; // For WmoRootLoader
using WoWToolbox.Core.Models; // Added for MeshData
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Runtime.InteropServices;
using WoWToolbox.Tests;

namespace WoWToolbox.Tests.WMO
{
    public class WmoGroupMeshTests
    {
        // Define paths relative to test execution
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(Assembly.GetExecutingAssembly().Location, "../../../../../../", "test_data")); // Navigating up 6 levels from bin/Debug/netX.X
        private static string TestOutputDir 
        {
            get
            {
                var dir = OutputLocator.Central("WmoGroupMeshTests");
                Console.WriteLine($"[TEST] TestOutputDir: {dir}");
                if (!Directory.Exists(dir))
                {
                    Console.WriteLine($"[TEST] Creating directory: {dir}");
                    Directory.CreateDirectory(dir);
                }
                return dir;
            }
        }

        // Example WMO and group index
        // private const string TestWmoRootFile = "wmo/Dungeon/Ulduar/Ulduar_Entrance.wmo"; // Example, replace with a known good test file
        // private const string TestWmoRootFile = "335_wmo/World/wmo/Dungeon/Ulduar/Ulduar_Raid.wmo"; // Using provided existing file
        private const string TestWmoRootFile = "335_wmo\\World\\wmo\\Northrend\\Buildings\\IronDwarf\\ND_IronDwarf_LargeBuilding\\ND_IronDwarf_LargeBuilding.wmo"; // Corrected path with extra dir level
        // private const int TestGroupIndex = 0;

        public WmoGroupMeshTests()
        {
            // Ensure output directory exists
            Console.WriteLine($"[DEBUG] Assembly Location: {Assembly.GetExecutingAssembly().Location}");
            Console.WriteLine($"[DEBUG] Calculated TestOutputDir: {TestOutputDir}");
            Directory.CreateDirectory(TestOutputDir);
        }
        
        private static bool HasMagicNumber(byte[] buffer, int offset, byte[] magic)
        {
            if (offset + magic.Length > buffer.Length)
                return false;
                
            for (int i = 0; i < magic.Length; i++)
            {
                if (buffer[offset + i] != magic[i])
                    return false;
            }
            return true;
        }

        [Fact]
        public void LoadAndExportAllWmoGroups_ShouldCreateMergedObjFile()
        {
            // Set up console output redirection to capture all output
            var originalOut = Console.Out;
            var originalError = Console.Error;
            
            string logFilePath = Path.Combine(TestOutputDir, "test_output.log");
            Console.WriteLine($"[TEST] Writing test output to: {logFilePath}");
            
            try
            {
                using (var fileStream = new FileStream(logFilePath, FileMode.Create, FileAccess.Write, FileShare.Read))
                using (var writer = new StreamWriter(fileStream, Encoding.UTF8))
                using (var logWriter = new MultiTextWriter(originalOut, writer))
                {
                    Console.SetOut(logWriter);
                    Console.SetError(logWriter);
                    
                    try
                    {
                        // Log test start and environment info
                        Console.WriteLine($"[TEST] ===================================================");
                        Console.WriteLine($"[TEST] Starting test: {nameof(LoadAndExportAllWmoGroups_ShouldCreateMergedObjFile)}");
                        Console.WriteLine($"[TEST] Current directory: {Directory.GetCurrentDirectory()}");
                        Console.WriteLine($"[TEST] Test started at: {DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}");
                        Console.WriteLine($"[TEST] Test output directory: {TestOutputDir}");
                        Console.WriteLine($"[TEST] Runtime: {RuntimeInformation.FrameworkDescription}");
                        Console.WriteLine($"[TEST] OS: {RuntimeInformation.OSDescription}");
                        Console.WriteLine($"[TEST] ===================================================");
                        
                        // Get the test WMO file path
                        string wmoFilePath = Path.Combine("test_data", "335_wmo", "World", "wmo", "Northrend", "Buildings", "IronDwarf", "ND_IronDwarf_LargeBuilding", "ND_IronDwarf_LargeBuilding.wmo");
                        string fullWmoPath = Path.GetFullPath(wmoFilePath);
                        
                        Console.WriteLine($"[TEST] WMO file path: {fullWmoPath}");
                        Console.WriteLine($"[TEST] File exists: {File.Exists(fullWmoPath)}");
                        
                        if (File.Exists(fullWmoPath))
                        {
                            // Read the first 64 bytes of the file to check the header
                            byte[] headerBytes = new byte[64];
                            using (var file = File.OpenRead(fullWmoPath))
                            {
                                int bytesRead = file.Read(headerBytes, 0, headerBytes.Length);
                                Console.WriteLine($"[TEST] Read {bytesRead} bytes from WMO file");
                                
                                // Log the header bytes in hex
                                string hexHeader = BitConverter.ToString(headerBytes).Replace("-", " ");
                                Console.WriteLine($"[TEST] WMO file header (hex): {hexHeader}");
                                
                                // Try to read as ASCII for the first 4 bytes (should be 'MVER' or similar)
                                string headerText = System.Text.Encoding.ASCII.GetString(headerBytes, 0, 4);
                                Console.WriteLine($"[TEST] WMO file header (text): {headerText}");
                                
                                // Check for MVER chunk (WMO v17+)
                                if (headerText == "MVER")
                                {
                                    Console.WriteLine("[TEST] Detected WMO v17+ format");
                                    uint version = BitConverter.ToUInt32(headerBytes, 8);
                                    Console.WriteLine($"[TEST] WMO version: {version}");
                                }
                                // Check for MPHD chunk (WMO v14)
                                else if (headerText == "DHPM") // MPHD in little-endian
                                {
                                    Console.WriteLine("[TEST] Detected WMO v14 format (little-endian MPHD chunk)");
                                }
                                else
                                {
                                    Console.WriteLine("[TEST] WARNING: Unknown WMO format, header doesn't match expected patterns");
                                }
                            }
                        }
                        else
                        {
                            string error = $"[TEST] ERROR: WMO file not found at: {fullWmoPath}";
                            Console.WriteLine(error);
                            throw new FileNotFoundException(error);
                        }
                        
                        // Now run the actual test
                        Console.WriteLine("\n[TEST] ===== Running the actual test implementation =====");
                        LoadAndExportAllWmoGroups_ShouldCreateMergedObjFile_Impl();
                        
                        Console.WriteLine("\n[TEST] Test completed successfully");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"\n[TEST] Test failed with exception: {ex}");
                        Console.WriteLine($"[TEST] Exception type: {ex.GetType().FullName}");
                        Console.WriteLine($"[TEST] Stack trace:\n{ex.StackTrace}");
                        
                        // Log inner exception if present
                        if (ex.InnerException != null)
                        {
                            Console.WriteLine("\n[TEST] Inner exception:");
                            Console.WriteLine($"[TEST] {ex.InnerException.GetType().FullName}: {ex.InnerException.Message}");
                            Console.WriteLine($"[TEST] Inner stack trace:\n{ex.InnerException.StackTrace}");
                        }
                        
                        throw;
                    }
                    finally
                    {
                        Console.WriteLine($"[TEST] Test completed at: {DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}");
                        Console.WriteLine($"[TEST] Log file: {logFilePath}");
                        
                        // Ensure all output is written to the file
                        writer.Flush();
                        fileStream.Flush(true);
                    }
                }
            }
            finally
            {
                // Restore original console output
                Console.SetOut(originalOut);
                Console.SetError(originalError);
                
                // Let the user know where to find the log file
                Console.WriteLine($"Test output logged to: {Path.GetFullPath(logFilePath)}");
            }
        }
        
        // Helper class to write to multiple text writers
        private class MultiTextWriter : TextWriter
        {
            private readonly TextWriter[] _writers;
            
            public MultiTextWriter(params TextWriter[] writers)
            {
                _writers = writers;
            }
            
            public override Encoding Encoding => Encoding.UTF8;
            
            public override void Write(char value)
            {
                foreach (var writer in _writers)
                    writer.Write(value);
            }
            
            public override void Write(string? value)
            {
                foreach (var writer in _writers)
                    writer.Write(value);
            }
            
            public override void WriteLine(string? value)
            {
                foreach (var writer in _writers)
                    writer.WriteLine(value);
            }
            
            public override void Flush()
            {
                foreach (var writer in _writers)
                    writer.Flush();
                base.Flush();
            }
            
            protected override void Dispose(bool disposing)
            {
                if (disposing)
                {
                    foreach (var writer in _writers)
                        writer.Dispose();
                }
                base.Dispose(disposing);
            }
        }
        
        private void LoadAndExportAllWmoGroups_ShouldCreateMergedObjFile_Impl()
        {
            // Arrange
            string rootWmoPath = Path.Combine(TestDataRoot, TestWmoRootFile);
            string outputMergedObjPath = Path.Combine(TestOutputDir, Path.GetFileNameWithoutExtension(rootWmoPath) + "_merged.obj");

            Console.WriteLine($"[TEST] ===== Starting WMO Load and Export Test =====");
            Console.WriteLine($"[TEST] WMO Path: {rootWmoPath}");
            Console.WriteLine($"[TEST] Output Path: {outputMergedObjPath}");
            Console.WriteLine($"[TEST] Working Directory: {Directory.GetCurrentDirectory()}");

            // Log file info for debugging
            var fileInfo = new FileInfo(rootWmoPath);
            Console.WriteLine($"[TEST] File exists: {fileInfo.Exists}");
            Console.WriteLine($"[TEST] File size: {fileInfo.Length} bytes");
            Console.WriteLine($"[TEST] File attributes: {fileInfo.Attributes}");
            Console.WriteLine($"[TEST] Last write time: {fileInfo.LastWriteTime}");

            if (!fileInfo.Exists)
            {
                throw new FileNotFoundException($"WMO file not found: {rootWmoPath}");
            }

            // First, run the diagnostic to understand the file structure
            Console.WriteLine("\n[TEST] ===== Running File Diagnostics =====");
            RunFileDiagnostics(rootWmoPath);
            
            // Add additional diagnostics for the WMO file
            Console.WriteLine("\n[TEST] ===== Additional WMO File Analysis =====");
            AnalyzeWmoFile(rootWmoPath);
            
            // Then proceed with the original test
            // Dump first 8KB of file for analysis (larger to catch more chunks)
            using (var fs = File.OpenRead(rootWmoPath))
            {
                byte[] header = new byte[8192];
                int bytesToRead = (int)Math.Min(header.Length, fs.Length);
                int bytesRead = fs.Read(header, 0, bytesToRead);
                
                // Dump file info
                Console.WriteLine("\n[DEBUG] ===== FILE HEADER ANALYSIS =====");
                Console.WriteLine($"File: {rootWmoPath}");
                Console.WriteLine($"Size: {fileInfo.Length} bytes");
                
                // Check file type and version with more robust detection
                string fileType = "Unknown";
                uint version = 0;
                
                // Look for MVER chunk (WMO v17+)
                int mverPos = FindChunk(header, "MVER");
                int mverSize = mverPos >= 0 ? (int)BitConverter.ToUInt32(header, mverPos + 4) : -1;
                
                // Look for MPHD chunk (WMO v14)
                int mphdPos = FindChunk(header, "MPHD");
                
                // Look for MOHD chunk (both versions)
                int mohdPos = FindChunk(header, "MOHD");
                
                // Analyze findings
                if (mverPos >= 0 && mverSize > 0 && mverSize < 100)
                {
                    fileType = "WMO v17+";
                    version = BitConverter.ToUInt32(header, mverPos + 8); // Version is first DWORD after MVER header
                    Console.WriteLine($"[DEBUG] Detected {fileType} file (version {version}) at offset 0x{mverPos:X}");
                }
                else if (mphdPos >= 0)
                {
                    fileType = "WMO v14";
                    version = 14;
                    Console.WriteLine($"[DEBUG] Detected {fileType} file (MPHD at 0x{mphdPos:X})");
                }
                else if (mohdPos >= 0)
                {
                    fileType = "WMO (MOHD found but version unknown)";
                    Console.WriteLine($"[DEBUG] {fileType} (MOHD at 0x{mohdPos:X})");
                }
                else
                {
                    // Try to find any known chunk IDs in the first 4KB
                    string firstBytes = BitConverter.ToString(header, 0, Math.Min(4096, bytesRead)).Replace("-", "");
                    var knownChunks = new[] { 
                        "4D564552", // MVER
                        "4D4F4844", // MOHD
                        "4D4F474E", // MOGN
                        "4D4F4749", // MOGI
                        "4D4F4D54", // MOMT
                        "4D4F5458", // MOTX
                        "4D4F4750", // MOGP
                        "4D4F5059", // MOPY
                        "4D4F5649", // MOVI
                        "4D4F5654", // MOVT
                        "4D4F4E52", // MONR
                        "4D4F5456", // MOTV
                        "4D4F4241"  // MOBA
                    };
                    
                    var foundChunks = knownChunks
                        .Select(c => new { Id = c, Pos = firstBytes.IndexOf(c) })
                        .Where(x => x.Pos >= 0)
                        .OrderBy(x => x.Pos)
                        .Select(x => $"0x{x.Id} at offset 0x{x.Pos/2:X}")
                        .ToArray();
                    
                    if (foundChunks.Any())
                    {
                        Console.WriteLine("[WARN] Unknown file type but found known chunks:");
                        foreach (var chunk in foundChunks.Take(10)) // Limit to first 10 to avoid spam
                            Console.WriteLine($"  - {chunk}");
                        if (foundChunks.Length > 10)
                            Console.WriteLine($"  - ... and {foundChunks.Length - 10} more");
                    }
                    else
                    {
                        Console.WriteLine("[WARN] Unknown file type, no known chunks found in first 4KB");
                    }
                    
                    // Try to detect endianness by looking for 'MVER' or 'REVM' in first 1KB
                    string ascii = Encoding.ASCII.GetString(header, 0, Math.Min(1024, bytesRead));
                    if (ascii.Contains("MVER"))
                    {
                        fileType = "WMO v17+ (detected by MVER)";
                        Console.WriteLine($"[DEBUG] {fileType}");
                    }
                    else if (ascii.Contains("REVM")) // MVER in big-endian
                    {
                        fileType = "WMO v17+ (big-endian MVER)";
                        Console.WriteLine($"[DEBUG] {fileType}");
                    }
                    else if (ascii.Contains("DHPM")) // MPHD in little-endian
                    {
                        fileType = "WMO v14 (detected by MPHD)";
                        Console.WriteLine($"[DEBUG] {fileType}");
                    }
                }
                
                // Dump first 512 bytes in hex/ASCII
                Console.WriteLine("\n[DEBUG] ===== HEX/ASCII DUMP (First 512 bytes) =====");
                Console.WriteLine("Offset  00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F  0123456789ABCDEF");
                Console.WriteLine("------- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  ----------------");
                
                int bytesToDump = Math.Min(512, bytesRead);
                for (int i = 0; i < bytesToDump; i += 16)
                {
                    int lineLength = Math.Min(16, bytesToDump - i);
                    if (lineLength <= 0) break;
                    
                    // Offset
                    Console.Write($"{i:X6}  ");
                    
                    // Hex bytes (first 8 bytes)
                    for (int j = 0; j < 8; j++)
                        Console.Write(j < lineLength ? $"{header[i + j]:X2} " : "   ");
                    
                    // Extra space between first 8 and last 8 bytes
                    Console.Write(" ");
                    
                    // Hex bytes (last 8 bytes)
                    for (int j = 8; j < 16; j++)
                        Console.Write(j < lineLength ? $"{header[i + j]:X2} " : "   ");
                    
                    // Extra space before ASCII
                    Console.Write(" ");
                    
                    // ASCII representation
                    for (int j = 0; j < lineLength; j++)
                    {
                        byte b = header[i + j];
                        Console.Write(b >= 32 && b < 127 ? (char)b : '.');
                    }
                    Console.WriteLine();
                }
                
                // If we found MOHD, dump its contents
                if (mohdPos >= 0 && mohdPos + 8 < bytesRead)
                {
                    Console.WriteLine("\n[DEBUG] ===== MOHD CHUNK DETAILS =====");
                    uint mohdSize = BitConverter.ToUInt32(header, mohdPos + 4);
                    Console.WriteLine($"MOHD chunk at 0x{mohdPos:X}, size: {mohdSize} bytes");
                    
                    // Dump MOHD header (first 64 bytes or full chunk, whichever is smaller)
                    int mohdDataStart = mohdPos + 8;
                    int mohdDataEnd = Math.Min(mohdDataStart + (int)mohdSize, bytesRead);
                    int mohdDumpSize = Math.Min(64, mohdDataEnd - mohdDataStart);
                    
                    if (mohdDumpSize > 0)
                    {
                        Console.WriteLine("MOHD data (hex):");
                        for (int i = 0; i < mohdDumpSize; i += 16)
                        {
                            int lineLength = Math.Min(16, mohdDumpSize - i);
                            Console.Write($"  {mohdDataStart + i:X6}  ");
                            
                            // Hex bytes
                            for (int j = 0; j < 16; j++)
                            {
                                if (j < lineLength)
                                    Console.Write($"{header[mohdDataStart + i + j]:X2} ");
                                else
                                    Console.Write("   ");
                                
                                // Extra space after 8 bytes
                                if (j == 7) Console.Write(" ");
                            }
                            
                            // ASCII representation
                            Console.Write(" ");
                            for (int j = 0; j < lineLength; j++)
                            {
                                byte b = header[mohdDataStart + i + j];
                                Console.Write(b >= 32 && b < 127 ? (char)b : '.');
                            }
                            Console.WriteLine();
                        }
                        
                        // Try to interpret the first few DWORDs in MOHD
                        if (mohdDumpSize >= 4)
                        {
                            int groupCount = BitConverter.ToInt32(header, mohdDataStart);
                            Console.WriteLine($"Group count (first DWORD): {groupCount}");
                        }
                    }
                }
                
                // Add a helper method to find chunks
                private int FindChunk(byte[] data, string chunkId)
                {
                    if (chunkId.Length != 4) return -1;
                    byte[] idBytes = Encoding.ASCII.GetBytes(chunkId);
                    
                    // Search in first 64KB
                    int searchLen = Math.Min(65536, data.Length - 4);
                    for (int i = 0; i < searchLen; i++)
                    {
                        if (data[i] == idBytes[0] && 
                            data[i+1] == idBytes[1] && 
                            data[i+2] == idBytes[2] && 
                            data[i+3] == idBytes[3])
                        {
                            return i;
                        }
                    }
                    return -1;
                }
                
                private void AnalyzeWmoFile(string filePath)
                {
                    try
                    {
                        Console.WriteLine($"[ANALYZE] Analyzing WMO file: {filePath}");
                        
                        // Read the entire file into memory for analysis
                        byte[] fileData = File.ReadAllBytes(filePath);
                        Console.WriteLine($"[ANALYZE] Read {fileData.Length} bytes from file");
                        
                        if (fileData.Length < 16)
                        {
                            Console.WriteLine("[ANALYZE] ERROR: File is too small to be a valid WMO file");
                            return;
                        }
                        
                        // Check for MVER chunk at the beginning (WMO v17+)
                        if (fileData[0] == 'M' && fileData[1] == 'V' && fileData[2] == 'E' && fileData[3] == 'R')
                        {
                            uint mverSize = BitConverter.ToUInt32(fileData, 4);
                            uint version = BitConverter.ToUInt32(fileData, 8);
                            Console.WriteLine($"[ANALYZE] Found MVER chunk (WMO v17+), size: {mverSize}, version: {version}");
                            
                            // Look for MOHD chunk
                            int mohdPos = FindChunk(fileData, "MOHD");
                            if (mohdPos > 0)
                            {
                                Console.WriteLine($"[ANALYZE] Found MOHD chunk at position: 0x{mohdPos:X}");
                                
                                // Read MOHD data (first 4 bytes after MOHD header is group count)
                                if (mohdPos + 12 < fileData.Length)
                                {
                                    uint mohdSize = BitConverter.ToUInt32(fileData, mohdPos + 4);
                                    int groupCount = BitConverter.ToInt32(fileData, mohdPos + 8);
                                    Console.WriteLine($"[ANALYZE] MOHD size: {mohdSize} bytes, group count: {groupCount}");
                                    
                                    // Dump MOHD header
                                    Console.WriteLine("[ANALYZE] MOHD header data (hex):");
                                    int dumpBytes = Math.Min(64, (int)mohdSize + 8);
                                    for (int i = 0; i < dumpBytes; i += 16)
                                    {
                                        int lineBytes = Math.Min(16, dumpBytes - i);
                                        Console.Write($"  {mohdPos + i:X6}: ");
                                        
                                        // Hex dump
                                        for (int j = 0; j < lineBytes; j++)
                                        {
                                            if (j == 8) Console.Write(" ");
                                            Console.Write($"{fileData[mohdPos + i + j]:X2} ");
                                        }
                                        
                                        // Padding for alignment
                                        for (int j = lineBytes; j < 16; j++)
                                        {
                                            if (j == 8) Console.Write(" ");
                                            Console.Write("   ");
                                        }
                                        
                                        // ASCII dump
                                        Console.Write(" ");
                                        for (int j = 0; j < lineBytes; j++)
                                        {
                                            byte b = fileData[mohdPos + i + j];
                                            Console.Write(b >= 32 && b < 127 ? (char)b : '.');
                                        }
                                        
                                        Console.WriteLine();
                                    }
                                }
                            }
                            else
                            {
                                Console.WriteLine("[ANALYZE] WARNING: MOHD chunk not found in WMO file");
                            }
                        }
                        else
                        {
                            Console.WriteLine("[ANALYZE] File does not start with MVER chunk, may be WMO v14 or corrupted");
                        }
                        
                        // Look for other important chunks
                        string[] importantChunks = { "MOGN", "MOGI", "MOTX", "MOMT", "MOUV" };
                        foreach (string chunk in importantChunks)
                        {
                            int pos = FindChunk(fileData, chunk);
                            if (pos > 0)
                            {
                                uint size = BitConverter.ToUInt32(fileData, pos + 4);
                                Console.WriteLine($"[ANALYZE] Found {chunk} chunk at 0x{pos:X}, size: {size} bytes");
                            }
                            else
                            {
                                Console.WriteLine($"[ANALYZE] WARNING: {chunk} chunk not found in WMO file");
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[ANALYZE] ERROR during WMO analysis: {ex.Message}");
                        Console.WriteLine($"[ANALYZE] Stack trace: {ex.StackTrace}");
                    }
                }
                
                // Also show first 64 bytes as a single line for easy copying
                Console.WriteLine("\n[DEBUG] First 64 bytes (hex):");
                Console.WriteLine(BitConverter.ToString(header, 0, Math.Min(64, bytesRead)).Replace("-", " "));
                
                // And as ASCII for quick visual inspection
                Console.WriteLine("\n[DEBUG] First 64 bytes (ASCII):");
                Console.WriteLine(new string(header.Take(64).Select(b => b >= 32 && b < 127 ? (char)b : '.').ToArray()));
                
                // Look for known magic numbers
                Console.WriteLine("\n[DEBUG] ===== MAGIC NUMBER SCAN =====");
                var magicNumbers = new Dictionary<string, byte[]> 
                {
                    { "MVER", new byte[] { 0x4D, 0x56, 0x45, 0x52 } },  // Version
                    { "MOHD", new byte[] { 0x4D, 0x4F, 0x48, 0x44 } },  // Header
                    { "MOGN", new byte[] { 0x4D, 0x4F, 0x47, 0x4E } },  // Group names
                    { "MOGI", new byte[] { 0x4D, 0x4F, 0x47, 0x49 } },  // Group info
                    { "MOSB", new byte[] { 0x4D, 0x4F, 0x53, 0x42 } },  // Skybox
                    { "MOPV", new byte[] { 0x4D, 0x4F, 0x50, 0x56 } },  // Portal vertices
                    { "MOPT", new byte[] { 0x4D, 0x4F, 0x50, 0x54 } },  // Portal info
                    { "MOPR", new byte[] { 0x4D, 0x4F, 0x50, 0x52 } },  // Portal references
                    { "MOVV", new byte[] { 0x4D, 0x4F, 0x56, 0x56 } },  // Visible vertices
                    { "MOVB", new byte[] { 0x4D, 0x4F, 0x56, 0x42 } },  // Visible block indices
                    { "MOLT", new byte[] { 0x4D, 0x4F, 0x4C, 0x54 } },  // Lights
                    { "MODS", new byte[] { 0x4D, 0x4F, 0x44, 0x53 } },  // Doodad sets
                    { "MODN", new byte[] { 0x4D, 0x4F, 0x44, 0x4E } },  // Doodad names
                    { "MODD", new byte[] { 0x4D, 0x4F, 0x44, 0x44 } },  // Doodad definitions
                    { "MFOG", new byte[] { 0x4D, 0x46, 0x4F, 0x47 } },  // Fog
                    { "MCVP", new byte[] { 0x4D, 0x43, 0x56, 0x50 } }   // Culling volumes
                };

                foreach (var magic in magicNumbers)
                {
                    for (int i = 0; i < bytesRead - 4; i++)
                    {
                        if (header[i] == magic.Value[0] && 
                            header[i+1] == magic.Value[1] && 
                            header[i+2] == magic.Value[2] && 
                            header[i+3] == magic.Value[3])
                        {
                            uint size = 0;
                            if (i + 8 <= bytesRead)
                            {
                                // Try little-endian first
                                size = BitConverter.ToUInt32(header, i + 4);
                                
                                // If size is suspiciously large, try big-endian
                                if (size > 0x1000000) // 16MB is a reasonable max chunk size
                                {
                                    size = (uint)((header[i+4] << 24) | (header[i+5] << 16) | (header[i+6] << 8) | header[i+7]);
                                    Console.WriteLine($"[DEBUG] Found {magic.Key} chunk at 0x{i:X8} (big-endian), size: {size} (0x{size:X8})");
                                }
                                else
                                {
                                    Console.WriteLine($"[DEBUG] Found {magic.Key} chunk at 0x{i:X8} (little-endian), size: {size} (0x{size:X8})");
                                }
                                
                                // Dump chunk header + 16 bytes of data
                                int dumpSize = Math.Min(24, bytesRead - i);
                                Console.Write($"  {i:X8}: ");
                                for (int j = 0; j < dumpSize; j++)
                                {
                                    Console.Write($"{header[i + j]:X2} ");
                                }
                                Console.WriteLine();
                                
                                // Skip ahead to find next chunk
                                i += 7 + (int)size;
                            }
                            else
                            {
                                Console.WriteLine($"[DEBUG] Found {magic.Key} chunk at 0x{i:X8} (incomplete header)");
                            }
                        }
                    }
                }
                
                // Dump first 256 bytes in hex/ASCII
                Console.WriteLine("\n[DEBUG] ===== HEX/ASCII DUMP (First 256 bytes) =====");
                Console.WriteLine("Offset  00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F  ASCII");
                Console.WriteLine("------- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --  ----------------");
                
                for (int i = 0; i < Math.Min(256, bytesRead); i += 16)
                {
                    int lineLength = Math.Min(16, bytesRead - i);
                    if (lineLength <= 0) break;
                    
                    // Offset
                    Console.Write($"{i:X7}  ");
                    
                    // Hex bytes
                    for (int j = 0; j < 16; j++)
                    {
                        if (j < lineLength)
                            Console.Write($"{header[i + j]:X2} ");
                        else
                            Console.Write("   ");
                    }
                    
                    // ASCII representation
                    Console.Write(" ");
                    for (int j = 0; j < lineLength; j++)
                    {
                        byte b = header[i + j];
                        Console.Write(b >= 32 && b < 127 ? (char)b : '.');
                    }
                    Console.WriteLine();
                }
                
                Console.WriteLine("[DEBUG] ===== END OF FILE ANALYSIS =====\n");
            }

            Assert.True(File.Exists(rootWmoPath), $"Root WMO test file not found: {rootWmoPath}");

            int groupCount = -1;
            List<string> groupNames = new List<string>();
            try
            {
                (groupCount, groupNames) = WmoRootLoader.LoadGroupInfo(rootWmoPath);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Failed to load root WMO file: {ex.Message}");
            }
            Assert.True(groupCount >= 0, $"Failed to read group count from root WMO (LoadGroupInfo returned {groupCount}).");
            Assert.True(groupCount > 0, "Root WMO reported 0 groups.");

            List<WmoGroupMesh> loadedMeshes = new List<WmoGroupMesh>();
            string rootWmoDirectory = Path.GetDirectoryName(rootWmoPath) ?? ".";

            // Act - Load all groups
            Console.WriteLine($"Root WMO has {groupCount} groups (found {groupNames.Count} names). Attempting to load...");
            for (int groupIndex = 0; groupIndex < groupCount; groupIndex++)
            {
                string groupFileName = Path.GetFileNameWithoutExtension(rootWmoPath) + $"_{groupIndex:000}.wmo";
                string groupFilePath = Path.Combine(rootWmoDirectory, groupFileName);
                if (File.Exists(groupFilePath))
                {
                    using var stream = File.OpenRead(groupFilePath);
                    var groupMesh = WmoGroupMesh.LoadFromStream(stream, groupFilePath);
                    var meshData = WmoGroupMeshToMeshData(groupMesh);
                    if (meshData.IsValid()) loadedMeshes.Add(groupMesh);
                }
            }
            var mergedMesh = WmoGroupMesh.MergeMeshes(loadedMeshes);

            Assert.NotNull(mergedMesh); // Ensure merge succeeded
            var mergedMeshData = WmoGroupMeshToMeshData(mergedMesh);
            Assert.True(mergedMeshData.Vertices.Count > 0, "Merged mesh should have vertices.");
            Assert.True(mergedMeshData.Indices.Count > 0, "Merged mesh should have indices.");

            Console.WriteLine("Saving merged mesh...");
            SaveMeshDataToObj(mergedMeshData, outputMergedObjPath);

            // Assert
            Assert.True(File.Exists(outputMergedObjPath), $"Expected merged WMO OBJ file was not created at: {outputMergedObjPath}");
            Assert.True(new FileInfo(outputMergedObjPath).Length > 100, $"Expected merged WMO OBJ file appears empty: {outputMergedObjPath}"); // Basic size check

            Console.WriteLine($"Successfully merged {loadedMeshes.Count} WMO groups and saved to {outputMergedObjPath}");
            Console.WriteLine($"  Total Vertices: {mergedMeshData.Vertices.Count}");
            Console.WriteLine($"  Total Triangles: {mergedMeshData.Indices.Count / 3}");
        }

        // --- Added Helper Method (copied from Pm4MeshExtractorTests) ---
        private static void SaveMeshDataToObj(MeshData meshData, string outputPath)
        {
            if (meshData == null)
            {
                Console.WriteLine("[WARN] MeshData is null, cannot save OBJ.");
                return;
            }

            try
            {
                 string? directoryPath = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
                {
                    Console.WriteLine($"[DEBUG][SaveMeshDataToObj] Creating directory: {directoryPath}");
                    Directory.CreateDirectory(directoryPath);
                }

                using (var writer = new StreamWriter(outputPath, false)) // Overwrite if exists
                {
                    // Set culture for consistent decimal formatting
                    CultureInfo culture = CultureInfo.InvariantCulture;

                    // Write header
                    writer.WriteLine($"# Mesh saved by WoWToolbox.Tests.WMO.WmoGroupMeshTests");
                    writer.WriteLine($"# Vertices: {meshData.Vertices.Count}");
                    writer.WriteLine($"# Triangles: {meshData.Indices.Count / 3}");
                    writer.WriteLine($"# Generated: {DateTime.Now}");
                    writer.WriteLine();

                    // Write Vertices (v x y z)
                    if (meshData.Vertices.Count > 0)
                    {
                        writer.WriteLine("# Vertex Definitions");
                        foreach (var vertex in meshData.Vertices)
                        {
                            // Format using invariant culture to ensure '.' as decimal separator
                            writer.WriteLine(string.Format(culture, "v {0} {1} {2}", vertex.X, vertex.Y, vertex.Z));
                        }
                        writer.WriteLine(); // Blank line after vertices
                    }

                    // Write Faces (f v1 v2 v3) - OBJ uses 1-based indexing!
                    if (meshData.Indices.Count > 0)
                    {
                        writer.WriteLine("# Face Definitions");
                        for (int i = 0; i < meshData.Indices.Count; i += 3)
                        {
                            // Add 1 to each index for 1-based OBJ format
                            int idx0 = meshData.Indices[i + 0] + 1;
                            int idx1 = meshData.Indices[i + 1] + 1;
                            int idx2 = meshData.Indices[i + 2] + 1;
                            writer.WriteLine($"f {idx0} {idx1} {idx2}");
                        }
                    }
                } // StreamWriter automatically flushes and closes here
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERR] Failed to save MeshData to OBJ file '{outputPath}': {ex.Message}");
                // Optionally rethrow or handle differently
                throw;
            }
        }

        private static MeshData WmoGroupMeshToMeshData(WmoGroupMesh mesh)
        {
            var md = new MeshData();
            if (mesh == null) return md;
            foreach (var v in mesh.Vertices)
                md.Vertices.Add(v.Position);
            foreach (var tri in mesh.Triangles)
            {
                md.Indices.Add(tri.Index0);
                md.Indices.Add(tri.Index1);
                md.Indices.Add(tri.Index2);
            }
            return md;
        }
    }

    public static class MeshDataExtensions
    {
        public static bool IsValid(this MeshData? meshData)
        {
            return meshData != null
                && meshData.Vertices != null
                && meshData.Indices != null
                && meshData.Vertices.Count > 0
                && meshData.Indices.Count > 0
                && meshData.Indices.Count % 3 == 0;
        }
        
        private void RunFileDiagnostics(string filePath)
        {
            Console.WriteLine("\n[DIAGNOSTICS] Running file diagnostics...");
            
            try
            {
                byte[] fileBytes = File.ReadAllBytes(filePath);
                Console.WriteLine($"[DIAGNOSTICS] File size: {fileBytes.Length} bytes (0x{fileBytes.Length:X8})");
                
                // Look for known chunk signatures in the first 64KB
                int searchLimit = Math.Min(65536, fileBytes.Length);
                Console.WriteLine($"[DIAGNOSTICS] Scanning first {searchLimit} bytes for known chunk signatures...");
                
                // Known WMO chunk signatures (4-byte ASCII)
                var knownChunks = new Dictionary<string, string>
                {
                    { "MVER", "WMO Version Chunk (v17+)" },
                    { "MOHD", "WMO Header Chunk (v17+)" },
                    { "MOGN", "WMO Group Names Chunk" },
                    { "MOGI", "WMO Group Info Chunk" },
                    { "MOSB", "WMO Skybox Chunk" },
                    { "MOPV", "WMO Portal Vertices Chunk" },
                    { "MOPT", "WMO Portal Info Chunk" },
                    { "MOPR", "WMO Portal References Chunk" },
                    { "MOVV", "WMO Visible Vertices Chunk" },
                    { "MOVB", "WMO Visible Block Info Chunk" },
                    { "MOLT", "WMO Light Info Chunk" },
                    { "MOLS", "WMO Doodad Sets Chunk" },
                    { "MOLP", "WMO Light Params Chunk" },
                    { "MOLM", "WMO Light Map Chunk" },
                    { "MODD", "WMO Doodad Defs Chunk" },
                    { "MODN", "WMO Doodad Names Chunk" },
                    { "MODS", "WMO Doodad Sets Chunk" },
                    { "MFOG", "WMO Fog Chunk" },
                    { "MCVP", "WMO Convex Volume Planes Chunk" },
                    { "GFID", "WMO Group File Data ID Chunk" },
                    { "MPVW", "WMO Portal View Chunk" },
                    { "MAID", "WMO Area ID Chunk" },
                    { "MPHD", "WMO Header (v14)" },
                    { "MTEX", "WMO Texture List Chunk" },
                    { "MMDX", "WMO M2 Model Filenames Chunk" },
                    { "MMID", "WMO M2 Model Filename Offsets Chunk" },
                    { "MWMO", "WMO WMO Model Filenames Chunk" },
                    { "MWID", "WMO WMO Model Filename Offsets Chunk" },
                    { "MDDF", "WMO M2 Placement Chunk" },
                    { "MODF", "WMO WMO Placement Chunk" }
                };
                
                // Search for all known chunks
                var foundChunks = new List<(int offset, string chunkId, string description, uint size)>();
                
                for (int i = 0; i <= searchLimit - 8; i++) // Need at least 8 bytes (4 for ID, 4 for size)
                {
                    // Check if current position matches any known chunk ID
                    foreach (var chunk in knownChunks)
                    {
                        if (i + 4 > fileBytes.Length) continue;
                        
                        // Check if the next 4 bytes match this chunk ID
                        bool match = true;
                        for (int j = 0; j < 4; j++)
                        {
                            if (fileBytes[i + j] != chunk.Key[j])
                            {
                                match = false;
                                break;
                            }
                        }
                        
                        if (match && i + 8 <= fileBytes.Length)
                        {
                            // Read the chunk size (little-endian)
                            uint size = BitConverter.ToUInt32(fileBytes, i + 4);
                            
                            // Also try big-endian if size seems too large
                            if (size > 0x1000000) // 16MB is a reasonable max chunk size
                            {
                                byte[] sizeBytes = new byte[4];
                                Array.Copy(fileBytes, i + 4, sizeBytes, 0, 4);
                                Array.Reverse(sizeBytes);
                                size = BitConverter.ToUInt32(sizeBytes, 0);
                            }
                            
                            foundChunks.Add((i, chunk.Key, chunk.Value, size));
                            i += 7; // Skip past this chunk ID and size to avoid partial matches
                            break;
                        }
                    }
                }
                
                // Sort chunks by offset
                foundChunks.Sort((a, b) => a.offset.CompareTo(b.offset));
                
                // Display found chunks
                Console.WriteLine("\n[DIAGNOSTICS] Found chunks:");
                Console.WriteLine("Offset   Chunk  Size      Description");
                Console.WriteLine(new string('-', 60));
                
                foreach (var chunk in foundChunks)
                {
                    Console.WriteLine($"0x{chunk.offset:X6}  {chunk.chunkId}   0x{chunk.size:X6}  {chunk.description}");
                    
                    // For MOHD, dump the first few bytes
                    if (chunk.chunkId == "MOHD" && chunk.offset + 12 < fileBytes.Length)
                    {
                        Console.WriteLine("  MOHD data (first 16 bytes):");
                        int dumpSize = Math.Min(16, (int)chunk.size);
                        Console.Write("  ");
                        for (int i = 0; i < dumpSize; i++)
                        {
                            if (i > 0 && i % 8 == 0) Console.Write(" ");
                            Console.Write($"{fileBytes[chunk.offset + 8 + i]:X2} ");
                        }
                        Console.WriteLine();
                        
                        // Try to interpret as group count (first 4 bytes after chunk header)
                        if (chunk.size >= 4)
                        {
                            int groupCount = BitConverter.ToInt32(fileBytes, chunk.offset + 8);
                            Console.WriteLine($"  Group count (first DWORD): {groupCount}");
                        }
                    }
                }
                
                if (foundChunks.Count == 0)
                {
                    Console.WriteLine("No known WMO chunks found in the first 64KB of the file.");
                    
                    // Dump the first 64 bytes in hex/ASCII for manual inspection
                    Console.WriteLine("\nFirst 64 bytes of file:");
                    Console.WriteLine("Offset  00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  ASCII");
                    Console.WriteLine(new string('-', 70));
                    
                    int dumpSize = Math.Min(64, fileBytes.Length);
                    for (int i = 0; i < dumpSize; i += 16)
                    {
                        int lineLength = Math.Min(16, dumpSize - i);
                        if (lineLength <= 0) break;
                        
                        // Offset
                        Console.Write($"{i:X6}  ");
                        
                        // Hex bytes (first 8)
                        for (int j = 0; j < 8; j++)
                            Console.Write(j < lineLength ? $"{fileBytes[i + j]:X2} " : "   ");
                        
                        // Extra space between first 8 and last 8 bytes
                        Console.Write(" ");
                        
                        // Hex bytes (last 8)
                        for (int j = 8; j < 16; j++)
                            Console.Write(j < lineLength ? $"{fileBytes[i + j]:X2} " : "   ");
                        
                        // ASCII representation
                        Console.Write(" ");
                        for (int j = 0; j < lineLength; j++)
                        {
                            byte b = fileBytes[i + j];
                            Console.Write(b >= 32 && b < 127 ? (char)b : '.');
                        }
                        Console.WriteLine();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Diagnostic failed: {ex.Message}");
            }
        }
    }
} 