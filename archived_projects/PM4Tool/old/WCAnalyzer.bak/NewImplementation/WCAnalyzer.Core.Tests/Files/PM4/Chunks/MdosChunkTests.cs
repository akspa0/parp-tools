using NUnit.Framework;
using System;
using WCAnalyzer.Core.Files.PM4.Chunks;

namespace WCAnalyzer.Core.Tests.Files.PM4.Chunks
{
    [TestFixture]
    public class MdosChunkTests
    {
        [Test]
        public void Parse_ValidData_ParsesCorrectly()
        {
            // Arrange
            byte[] data = new byte[16]; // 2x 8-byte structures
            // First structure
            BitConverter.GetBytes(1.0f).CopyTo(data, 0);  // Field1
            BitConverter.GetBytes(2.0f).CopyTo(data, 4);  // Field2
            // Second structure
            BitConverter.GetBytes(3.0f).CopyTo(data, 8);  // Field1
            BitConverter.GetBytes(4.0f).CopyTo(data, 12); // Field2

            // Act
            var chunk = new MdosChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetEntryCount(), Is.EqualTo(2));
            
            var entry1 = chunk.GetEntry(0);
            Assert.That(entry1.Field1, Is.EqualTo(1.0f));
            Assert.That(entry1.Field2, Is.EqualTo(2.0f));
            
            var entry2 = chunk.GetEntry(1);
            Assert.That(entry2.Field1, Is.EqualTo(3.0f));
            Assert.That(entry2.Field2, Is.EqualTo(4.0f));
        }

        [Test]
        public void Parse_EmptyData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[0];

            // Act
            var chunk = new MdosChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetEntryCount(), Is.EqualTo(0));
        }

        [Test]
        public void Parse_PartialData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[4]; // Not a multiple of 8

            // Act
            var chunk = new MdosChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetEntryCount(), Is.EqualTo(0));
        }

        [Test]
        public void GetEntry_OutOfRange_ThrowsException()
        {
            // Arrange
            byte[] data = new byte[8]; // 1x 8-byte structure
            var chunk = new MdosChunk(data);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => chunk.GetEntry(1));
        }

        [Test]
        public void Write_ReturnsSameData()
        {
            // Arrange
            byte[] data = new byte[24]; // 3x 8-byte structures
            for (int i = 0; i < 3; i++)
            {
                BitConverter.GetBytes((float)(i + 1) * 1.5f).CopyTo(data, i * 8);
                BitConverter.GetBytes((float)(i + 1) * 2.5f).CopyTo(data, i * 8 + 4);
            }
            
            var chunk = new MdosChunk(data);

            // Act
            byte[] result = chunk.Write();

            // Assert
            Assert.That(result, Is.EqualTo(data));
        }

        [Test]
        public void GetSignature_ReturnsMDOS()
        {
            // Arrange
            var chunk = new MdosChunk(new byte[0]);

            // Act
            string signature = chunk.GetSignature();

            // Assert
            Assert.That(signature, Is.EqualTo("MDOS"));
        }
    }
} 