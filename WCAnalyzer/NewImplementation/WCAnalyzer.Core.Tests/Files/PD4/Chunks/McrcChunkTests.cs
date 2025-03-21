using NUnit.Framework;
using System;
using WCAnalyzer.Core.Files.PD4.Chunks;

namespace WCAnalyzer.Core.Tests.Files.PD4.Chunks
{
    [TestFixture]
    public class McrcChunkTests
    {
        [Test]
        public void Constructor_WithValue_SetsValueCorrectly()
        {
            // Arrange
            uint expectedCRC = 0x12345678;

            // Act
            var chunk = new McrcChunk(expectedCRC);

            // Assert
            Assert.That(chunk.CRC, Is.EqualTo(expectedCRC));
        }

        [Test]
        public void Constructor_WithData_ParsesCorrectly()
        {
            // Arrange
            uint expectedCRC = 0x12345678;
            byte[] data = BitConverter.GetBytes(expectedCRC);

            // Act
            var chunk = new McrcChunk(BitConverter.ToUInt32(data, 0));

            // Assert
            Assert.That(chunk.CRC, Is.EqualTo(expectedCRC));
        }

        [Test]
        public void Constructor_Default_SetsCRCToZero()
        {
            // Act
            var chunk = new McrcChunk();

            // Assert
            Assert.That(chunk.CRC, Is.EqualTo(0));
        }

        [Test]
        public void Write_ReturnsCRCAsBytes()
        {
            // Arrange
            uint expectedCRC = 0x12345678;
            var chunk = new McrcChunk(expectedCRC);
            byte[] expectedData = BitConverter.GetBytes(expectedCRC);

            // Act
            byte[] actualData = chunk.Write();

            // Assert
            Assert.That(actualData, Is.EqualTo(expectedData));
        }

        [Test]
        public void GetSize_ReturnsFourBytes()
        {
            // Arrange
            var chunk = new McrcChunk(0x12345678);

            // Act
            uint size = chunk.GetSize();

            // Assert
            Assert.That(size, Is.EqualTo(4)); // Size of uint32 is 4 bytes
        }

        [Test]
        public void GetSignature_ReturnsMCRC()
        {
            // Arrange
            var chunk = new McrcChunk();

            // Act
            string signature = chunk.GetSignature();

            // Assert
            Assert.That(signature, Is.EqualTo("MCRC"));
        }

        [Test]
        public void ToString_IncludesCRCValue()
        {
            // Arrange
            uint crc = 0x12345678;
            var chunk = new McrcChunk(crc);

            // Act
            string result = chunk.ToString();

            // Assert
            StringAssert.Contains(crc.ToString("X8"), result);
            StringAssert.Contains("MCRC", result);
        }
    }
} 