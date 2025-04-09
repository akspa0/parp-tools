using System;
using System.IO;
using System.Numerics;
using System.Collections.Generic;
using System.Text;
using NewPM4Reader.PM4.Chunks;

namespace NewPM4Reader.Tests
{
    /// <summary>
    /// Tests for PM4 chunks to verify parsing and description functionality
    /// </summary>
    public static class ChunkTests
    {
        /// <summary>
        /// Runs tests for all implemented chunks
        /// </summary>
        public static void RunAllTests()
        {
            Console.WriteLine("Running PM4 Chunk Tests...");
            Console.WriteLine("==========================");
            
            TestDHSM();
            TestRUSM();
            TestKLSM();
            TestIVSM();
            TestTVSM();
            TestNVSM();
            TestVVSM();
            
            Console.WriteLine("All tests completed!");
        }
        
        /// <summary>
        /// Tests the DHSM chunk implementation
        /// </summary>
        private static void TestDHSM()
        {
            Console.WriteLine("\nTesting DHSM chunk...");
            
            // Create test data
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write header values
                writer.Write((uint)123);          // HeaderValue1
                writer.Write((uint)456);          // HeaderValue2
                writer.Write((uint)789);          // HeaderValue3
                
                // Write placeholder values (should be 0)
                for (int i = 0; i < 5; i++)
                {
                    writer.Write((uint)0);
                }
                
                ms.Position = 0;
                
                // Create and parse the chunk
                using (var reader = new BinaryReader(ms))
                {
                    var chunk = new DHSM(reader);
                    
                    // Verify parsing
                    Console.WriteLine($"HeaderValue1: {chunk.HeaderValue1} (Expected: 123)");
                    Console.WriteLine($"HeaderValue2: {chunk.HeaderValue2} (Expected: 456)");
                    Console.WriteLine($"HeaderValue3: {chunk.HeaderValue3} (Expected: 789)");
                    Console.WriteLine($"Placeholders all zero: {Array.TrueForAll(chunk.Placeholders, p => p == 0)}");
                    
                    // Display description
                    Console.WriteLine("\nDetailed description:");
                    Console.WriteLine(chunk.GetDetailedDescription());
                }
            }
        }
        
        /// <summary>
        /// Tests the RUSM chunk implementation
        /// </summary>
        private static void TestRUSM()
        {
            Console.WriteLine("\nTesting RUSM chunk...");
            
            // Create test data
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write entry 1
                writer.Write((byte)1);            // Flags
                writer.Write((byte)4);            // IndexCount
                writer.Write((byte)0);            // Value1
                writer.Write((byte)0);            // Padding
                writer.Write(1.0f);               // Value2
                writer.Write(2.0f);               // Value3
                writer.Write(3.0f);               // Value4
                writer.Write(4.0f);               // Value5
                writer.Write((uint)10);           // MSVIFirstIndex
                writer.Write((uint)0);            // Value6
                writer.Write((uint)0);            // Value7
                
                // Write entry 2
                writer.Write((byte)2);            // Flags
                writer.Write((byte)6);            // IndexCount
                writer.Write((byte)0);            // Value1
                writer.Write((byte)0);            // Padding
                writer.Write(5.0f);               // Value2
                writer.Write(6.0f);               // Value3
                writer.Write(7.0f);               // Value4
                writer.Write(8.0f);               // Value5
                writer.Write((uint)14);           // MSVIFirstIndex
                writer.Write((uint)0);            // Value6
                writer.Write((uint)0);            // Value7
                
                ms.Position = 0;
                
                // Create and parse the chunk
                using (var reader = new BinaryReader(ms))
                {
                    var chunk = new RUSM(reader);
                    
                    // Verify parsing
                    Console.WriteLine($"Entry count: {chunk.Entries.Count} (Expected: 2)");
                    
                    if (chunk.Entries.Count >= 2)
                    {
                        Console.WriteLine($"Entry[0].Flags: {chunk.Entries[0].Flags} (Expected: 1)");
                        Console.WriteLine($"Entry[0].IndexCount: {chunk.Entries[0].IndexCount} (Expected: 4)");
                        Console.WriteLine($"Entry[0].MSVIFirstIndex: {chunk.Entries[0].MSVIFirstIndex} (Expected: 10)");
                        
                        Console.WriteLine($"Entry[1].Flags: {chunk.Entries[1].Flags} (Expected: 2)");
                        Console.WriteLine($"Entry[1].IndexCount: {chunk.Entries[1].IndexCount} (Expected: 6)");
                        Console.WriteLine($"Entry[1].MSVIFirstIndex: {chunk.Entries[1].MSVIFirstIndex} (Expected: 14)");
                    }
                    
                    // Display description
                    Console.WriteLine("\nDetailed description:");
                    Console.WriteLine(chunk.GetDetailedDescription());
                }
            }
        }
        
        /// <summary>
        /// Tests the KLSM chunk implementation
        /// </summary>
        private static void TestKLSM()
        {
            Console.WriteLine("\nTesting KLSM chunk...");
            
            // Create test data
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write entry 1
                writer.Write((byte)1);            // Flags
                writer.Write((byte)2);            // Value1
                writer.Write((ushort)0);          // Padding
                writer.Write((uint)100);          // Index
                
                // Write MSPI_first_index (24-bit) and MSPI_index_count (8-bit)
                writer.Write((byte)(10 & 0xFF));
                writer.Write((byte)((10 >> 8) & 0xFF));
                writer.Write((byte)((10 >> 16) & 0xFF));
                writer.Write((byte)5);            // MSPIIndexCount
                
                writer.Write((uint)0xFFFFFFFF);   // Value2
                writer.Write((ushort)20);         // Value3
                writer.Write((ushort)0x8000);     // Value4
                
                // Write entry 2 - with a negative index
                writer.Write((byte)2);            // Flags
                writer.Write((byte)3);            // Value1
                writer.Write((ushort)0);          // Padding
                writer.Write((uint)200);          // Index
                
                // Write MSPI_first_index as -10 (24-bit negative) and MSPI_index_count
                int negIndex = -10;
                writer.Write((byte)(negIndex & 0xFF));
                writer.Write((byte)((negIndex >> 8) & 0xFF));
                writer.Write((byte)((negIndex >> 16) & 0xFF));
                writer.Write((byte)3);            // MSPIIndexCount
                
                writer.Write((uint)0xFFFFFFFF);   // Value2
                writer.Write((ushort)30);         // Value3
                writer.Write((ushort)0x8000);     // Value4
                
                ms.Position = 0;
                
                // Create and parse the chunk
                using (var reader = new BinaryReader(ms))
                {
                    var chunk = new KLSM(reader);
                    
                    // Verify parsing
                    Console.WriteLine($"Entry count: {chunk.Entries.Count} (Expected: 2)");
                    
                    if (chunk.Entries.Count >= 2)
                    {
                        Console.WriteLine($"Entry[0].Flags: {chunk.Entries[0].Flags} (Expected: 1)");
                        Console.WriteLine($"Entry[0].Index: {chunk.Entries[0].Index} (Expected: 100)");
                        Console.WriteLine($"Entry[0].MSPIFirstIndex: {chunk.Entries[0].MSPIFirstIndex} (Expected: 10)");
                        Console.WriteLine($"Entry[0].MSPIIndexCount: {chunk.Entries[0].MSPIIndexCount} (Expected: 5)");
                        
                        Console.WriteLine($"Entry[1].Flags: {chunk.Entries[1].Flags} (Expected: 2)");
                        Console.WriteLine($"Entry[1].Index: {chunk.Entries[1].Index} (Expected: 200)");
                        Console.WriteLine($"Entry[1].MSPIFirstIndex: {chunk.Entries[1].MSPIFirstIndex} (Expected: -10)");
                        Console.WriteLine($"Entry[1].MSPIIndexCount: {chunk.Entries[1].MSPIIndexCount} (Expected: 3)");
                    }
                    
                    // Display description
                    Console.WriteLine("\nDetailed description:");
                    Console.WriteLine(chunk.GetDetailedDescription());
                }
            }
        }
        
        /// <summary>
        /// Tests the IVSM chunk implementation
        /// </summary>
        private static void TestIVSM()
        {
            Console.WriteLine("\nTesting IVSM chunk...");
            
            // Create test data
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write indices
                for (uint i = 0; i < 10; i++)
                {
                    writer.Write(i);
                }
                
                ms.Position = 0;
                
                // Create and parse the chunk
                using (var reader = new BinaryReader(ms))
                {
                    var chunk = new IVSM(reader);
                    
                    // Verify parsing
                    Console.WriteLine($"Index count: {chunk.Indices.Count} (Expected: 10)");
                    
                    for (int i = 0; i < Math.Min(5, chunk.Indices.Count); i++)
                    {
                        Console.WriteLine($"Index[{i}]: {chunk.Indices[i]} (Expected: {i})");
                    }
                    
                    // Display description
                    Console.WriteLine("\nDetailed description:");
                    Console.WriteLine(chunk.GetDetailedDescription());
                }
            }
        }
        
        /// <summary>
        /// Tests the TVSM chunk implementation
        /// </summary>
        private static void TestTVSM()
        {
            Console.WriteLine("\nTesting TVSM chunk...");
            
            // Create test data - remember, this should be in YXZ order as per documentation
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write 5 vertices in YXZ order
                for (int i = 0; i < 5; i++)
                {
                    // Y component
                    writer.Write(i * 10.0f);
                    // X component
                    writer.Write(i * 5.0f);
                    // Z component
                    writer.Write(i * 2.0f);
                }
                
                ms.Position = 0;
                
                // Create and parse the chunk
                using (var reader = new BinaryReader(ms))
                {
                    var chunk = new TVSM(reader);
                    
                    // Verify parsing
                    Console.WriteLine($"Vertex count: {chunk.RawVertices.Count} (Expected: 5)");
                    
                    // Check that the values were parsed correctly and stored in our internal format (XYZ)
                    for (int i = 0; i < Math.Min(3, chunk.RawVertices.Count); i++)
                    {
                        var vertex = chunk.RawVertices[i];
                        Console.WriteLine($"RawVertex[{i}]: X={vertex.X} (Expected: {i * 5.0f}), Y={vertex.Y} (Expected: {i * 10.0f}), Z={vertex.Z} (Expected: {i * 2.0f})");
                    }
                    
                    // Also test the transformed coordinates
                    Console.WriteLine("\nTransformed coordinates (first 3):");
                    int j = 0;
                    foreach (var transformed in chunk.TransformedVertices)
                    {
                        if (j >= 3) break;
                        
                        float expectedX = 17066.666f - (j * 10.0f);
                        float expectedY = 17066.666f - (j * 5.0f);
                        float expectedZ = (j * 2.0f) / 36.0f;
                        
                        Console.WriteLine($"TransformedVertex[{j}]: X={transformed.X:F3} (Expected: {expectedX:F3}), Y={transformed.Y:F3} (Expected: {expectedY:F3}), Z={transformed.Z:F3} (Expected: {expectedZ:F3})");
                        j++;
                    }
                    
                    // Display description
                    Console.WriteLine("\nDetailed description:");
                    Console.WriteLine(chunk.GetDetailedDescription());
                }
            }
        }
        
        /// <summary>
        /// Tests the NVSM chunk implementation
        /// </summary>
        private static void TestNVSM()
        {
            Console.WriteLine("\nTesting NVSM chunk...");
            
            // Create test data
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write 5 vectors
                for (int i = 0; i < 5; i++)
                {
                    writer.Write(i * 1.0f);       // X
                    writer.Write(i * 0.5f);       // Y
                    writer.Write(i * 0.1f);       // Z
                }
                
                ms.Position = 0;
                
                // Create and parse the chunk
                using (var reader = new BinaryReader(ms))
                {
                    var chunk = new NVSM(reader);
                    
                    // Verify parsing
                    Console.WriteLine($"Vector count: {chunk.Vectors.Count} (Expected: 5)");
                    
                    for (int i = 0; i < Math.Min(3, chunk.Vectors.Count); i++)
                    {
                        var vector = chunk.Vectors[i];
                        Console.WriteLine($"Vector[{i}]: X={vector.X} (Expected: {i * 1.0f}), Y={vector.Y} (Expected: {i * 0.5f}), Z={vector.Z} (Expected: {i * 0.1f})");
                    }
                    
                    // Display description
                    Console.WriteLine("\nDetailed description:");
                    Console.WriteLine(chunk.GetDetailedDescription());
                }
            }
        }
        
        /// <summary>
        /// Tests the VVSM chunk implementation
        /// </summary>
        private static void TestVVSM()
        {
            Console.WriteLine("\nTesting VVSM chunk...");
            
            // Create test data
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // Write 5 vertices
                for (int i = 0; i < 5; i++)
                {
                    writer.Write(i * 100.0f);     // X
                    writer.Write(i * 200.0f);     // Y
                    writer.Write(i * 300.0f);     // Z
                }
                
                ms.Position = 0;
                
                // Create and parse the chunk
                using (var reader = new BinaryReader(ms))
                {
                    var chunk = new VVSM(reader);
                    
                    // Verify parsing
                    Console.WriteLine($"Vertex count: {chunk.Vertices.Count} (Expected: 5)");
                    
                    for (int i = 0; i < Math.Min(3, chunk.Vertices.Count); i++)
                    {
                        var vertex = chunk.Vertices[i];
                        Console.WriteLine($"Vertex[{i}]: X={vertex.X} (Expected: {i * 100.0f}), Y={vertex.Y} (Expected: {i * 200.0f}), Z={vertex.Z} (Expected: {i * 300.0f})");
                    }
                    
                    // Display description
                    Console.WriteLine("\nDetailed description:");
                    Console.WriteLine(chunk.GetDetailedDescription());
                }
            }
        }
    }
} 