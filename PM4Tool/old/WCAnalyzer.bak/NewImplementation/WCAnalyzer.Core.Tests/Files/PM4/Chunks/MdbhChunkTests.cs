using NUnit.Framework;
using System;
using System.Text;
using WCAnalyzer.Core.Files.PM4.Chunks;

namespace WCAnalyzer.Core.Tests.Files.PM4.Chunks
{
    [TestFixture]
    public class MdbhChunkTests
    {
        [Test]
        public void Parse_ValidData_ParsesCorrectly()
        {
            // Arrange
            // Create a test chunk with 1 entry
            // Structure: [EntryCount (4)] + [MDBI chunk] + [MDBF chunk 1] + [MDBF chunk 2] + [MDBF chunk 3]
            
            byte[] data = new byte[64]; // Size for 1 entry with all sub-chunks
            int offset = 0;
            
            // Set entry count (1)
            BitConverter.GetBytes(1).CopyTo(data, offset);
            offset += 4;
            
            // Create MDBI chunk (sig + size + index)
            Encoding.ASCII.GetBytes("MDBI").CopyTo(data, offset);
            offset += 4;
            BitConverter.GetBytes(4).CopyTo(data, offset); // Size of data
            offset += 4;
            BitConverter.GetBytes(1).CopyTo(data, offset); // Index value
            offset += 4;
            
            // Create MDBF chunk 1
            CreateMdbfChunk(data, ref offset, "file1.m2");
            
            // Create MDBF chunk 2
            CreateMdbfChunk(data, ref offset, "file2.m2");
            
            // Create MDBF chunk 3
            CreateMdbfChunk(data, ref offset, "file3.m2");

            // Act
            var chunk = new MdbhChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetEntryCount(), Is.EqualTo(1));
            
            var entry = chunk.GetEntry(0);
            Assert.That(entry, Is.Not.Null);
            Assert.That(entry.Index, Is.EqualTo(1));
            Assert.That(entry.DestroyedFilename, Is.EqualTo("file1.m2"));
            Assert.That(entry.DamagedFilename, Is.EqualTo("file2.m2"));
            Assert.That(entry.IntactFilename, Is.EqualTo("file3.m2"));
        }

        private void CreateMdbfChunk(byte[] data, ref int offset, string filename)
        {
            // Add MDBF signature
            Encoding.ASCII.GetBytes("MDBF").CopyTo(data, offset);
            offset += 4;
            
            // Add size of filename + null terminator
            int size = filename.Length + 1;
            BitConverter.GetBytes(size).CopyTo(data, offset);
            offset += 4;
            
            // Add filename
            Encoding.ASCII.GetBytes(filename).CopyTo(data, offset);
            offset += filename.Length;
            
            // Add null terminator
            data[offset] = 0;
            offset += 1;
        }

        [Test]
        public void Parse_EmptyData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[0];

            // Act
            var chunk = new MdbhChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetEntryCount(), Is.EqualTo(0));
        }

        [Test]
        public void Parse_OnlyEntryCount_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[4];
            BitConverter.GetBytes(5).CopyTo(data, 0); // Entry count of 5 but no actual entries

            // Act
            var chunk = new MdbhChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetEntryCount(), Is.EqualTo(0)); // Should report 0 since there are no valid entries
        }

        [Test]
        public void GetEntry_OutOfRange_ThrowsException()
        {
            // Arrange
            // Create a chunk with 1 entry
            byte[] data = new byte[64]; // Size for 1 entry with all sub-chunks
            int offset = 0;
            
            // Set entry count (1)
            BitConverter.GetBytes(1).CopyTo(data, offset);
            offset += 4;
            
            // Create MDBI chunk (sig + size + index)
            Encoding.ASCII.GetBytes("MDBI").CopyTo(data, offset);
            offset += 4;
            BitConverter.GetBytes(4).CopyTo(data, offset); // Size of data
            offset += 4;
            BitConverter.GetBytes(1).CopyTo(data, offset); // Index value
            offset += 4;
            
            // Create MDBF chunks
            CreateMdbfChunk(data, ref offset, "file1.m2");
            CreateMdbfChunk(data, ref offset, "file2.m2");
            CreateMdbfChunk(data, ref offset, "file3.m2");
            
            var chunk = new MdbhChunk(data);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => chunk.GetEntry(1));
        }

        [Test]
        public void Write_ReturnsSameData()
        {
            // Arrange
            // Create a test chunk with 1 entry
            byte[] data = new byte[64]; // Size for 1 entry with all sub-chunks
            int offset = 0;
            
            // Set entry count (1)
            BitConverter.GetBytes(1).CopyTo(data, offset);
            offset += 4;
            
            // Create MDBI chunk
            Encoding.ASCII.GetBytes("MDBI").CopyTo(data, offset);
            offset += 4;
            BitConverter.GetBytes(4).CopyTo(data, offset);
            offset += 4;
            BitConverter.GetBytes(1).CopyTo(data, offset);
            offset += 4;
            
            // Create MDBF chunks
            CreateMdbfChunk(data, ref offset, "file1.m2");
            CreateMdbfChunk(data, ref offset, "file2.m2");
            CreateMdbfChunk(data, ref offset, "file3.m2");
            
            var chunk = new MdbhChunk(data);

            // Act
            byte[] result = chunk.Write();

            // Assert
            Assert.That(result, Is.EqualTo(data));
        }

        [Test]
        public void GetSignature_ReturnsMDBH()
        {
            // Arrange
            var chunk = new MdbhChunk(new byte[0]);

            // Act
            string signature = chunk.GetSignature();

            // Assert
            Assert.That(signature, Is.EqualTo("MDBH"));
        }
    }
} 