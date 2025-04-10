using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AdtTest
{
    class Program
    {
        static async Task Main(string[] args)
        {
            string testDataDir = @"..\..\test_data\development";
            
            if (!Directory.Exists(testDataDir))
            {
                Console.WriteLine($"Test data directory not found: {testDataDir}");
                return;
            }

            // Get a sample ADT file
            string sampleFile = Directory.GetFiles(testDataDir, "*.adt").FirstOrDefault();
            if (string.IsNullOrEmpty(sampleFile))
            {
                Console.WriteLine("No ADT files found in the test data directory.");
                return;
            }

            Console.WriteLine($"Analyzing ADT file: {sampleFile}");
            await AnalyzeAdtFile(sampleFile);
        }

        static async Task AnalyzeAdtFile(string filePath)
        {
            try
            {
                byte[] fileData = await File.ReadAllBytesAsync(filePath);
                Console.WriteLine($"File size: {fileData.Length} bytes");

                // Check if the file has reversed chunk IDs
                bool hasReversedChunkIds = false;
                if (fileData.Length >= 4)
                {
                    string signature = Encoding.ASCII.GetString(fileData, 0, 4);
                    Console.WriteLine($"First 4 bytes: {signature}");
                    
                    if (signature == "REVM")
                    {
                        hasReversedChunkIds = true;
                        Console.WriteLine("File has reversed chunk IDs");
                    }
                }

                // Dump the chunk structure
                Console.WriteLine("\nChunk Structure:");
                Console.WriteLine("Offset\tID\tSize\tReversed ID");
                
                int offset = 0;
                while (offset + 8 <= fileData.Length)
                {
                    // Read chunk ID and size
                    string chunkId = Encoding.ASCII.GetString(fileData, offset, 4);
                    int chunkSize = BitConverter.ToInt32(fileData, offset + 4);
                    
                    // Calculate the reversed chunk ID
                    string reversedId = new string(chunkId.Reverse().ToArray());
                    
                    Console.WriteLine($"{offset,6}\t{chunkId}\t{chunkSize,8}\t{reversedId}");
                    
                    // Move to the next chunk
                    offset += 8 + chunkSize;
                    
                    // Safety check to prevent infinite loops
                    if (chunkSize < 0 || offset < 0)
                    {
                        Console.WriteLine("Invalid chunk size or offset detected. Stopping analysis.");
                        break;
                    }
                }

                // If we have reversed chunk IDs, create a corrected version and analyze it
                if (hasReversedChunkIds)
                {
                    Console.WriteLine("\nCreating corrected version with reversed chunk IDs...");
                    byte[] correctedData = CorrectChunkIds(fileData);
                    
                    // Dump the corrected chunk structure
                    Console.WriteLine("\nCorrected Chunk Structure:");
                    Console.WriteLine("Offset\tID\tSize\tOriginal ID");
                    
                    offset = 0;
                    while (offset + 8 <= correctedData.Length)
                    {
                        // Read chunk ID and size
                        string chunkId = Encoding.ASCII.GetString(correctedData, offset, 4);
                        int chunkSize = BitConverter.ToInt32(correctedData, offset + 4);
                        
                        // Calculate the original chunk ID
                        string originalId = new string(chunkId.Reverse().ToArray());
                        
                        Console.WriteLine($"{offset,6}\t{chunkId}\t{chunkSize,8}\t{originalId}");
                        
                        // Move to the next chunk
                        offset += 8 + chunkSize;
                        
                        // Safety check to prevent infinite loops
                        if (chunkSize < 0 || offset < 0)
                        {
                            Console.WriteLine("Invalid chunk size or offset detected. Stopping analysis.");
                            break;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error analyzing ADT file: {ex.Message}");
            }
        }

        static byte[] CorrectChunkIds(byte[] fileData)
        {
            byte[] correctedData = new byte[fileData.Length];
            Array.Copy(fileData, correctedData, fileData.Length);
            
            int offset = 0;
            while (offset + 8 <= correctedData.Length)
            {
                // Reverse the chunk ID (4 bytes)
                Array.Reverse(correctedData, offset, 4);
                
                // Read the chunk size
                int chunkSize = BitConverter.ToInt32(correctedData, offset + 4);
                
                // Move to the next chunk
                offset += 8 + chunkSize;
                
                // Safety check to prevent infinite loops
                if (chunkSize < 0 || offset < 0)
                {
                    Console.WriteLine("Invalid chunk size or offset detected. Stopping correction.");
                    break;
                }
            }
            
            return correctedData;
        }
    }
}
