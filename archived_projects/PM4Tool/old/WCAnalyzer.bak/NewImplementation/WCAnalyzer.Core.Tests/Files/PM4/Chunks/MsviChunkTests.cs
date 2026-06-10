using NUnit.Framework;
using System;
using WCAnalyzer.Core.Files.PM4.Chunks;

namespace WCAnalyzer.Core.Tests.Files.PM4.Chunks
{
    [TestFixture]
    public class MsviChunkTests
    {
        [Test]
        public void Parse_ValidData_ParsesCorrectly()
        {
            // Arrange
            byte[] data = new byte[12]; // 3x 4-byte indices
            BitConverter.GetBytes(10).CopyTo(data, 0);
            BitConverter.GetBytes(20).CopyTo(data, 4);
            BitConverter.GetBytes(30).CopyTo(data, 8);

            // Act
            var chunk = new MsviChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetIndexCount(), Is.EqualTo(3));
            Assert.That(chunk.GetIndex(0), Is.EqualTo(10));
            Assert.That(chunk.GetIndex(1), Is.EqualTo(20));
            Assert.That(chunk.GetIndex(2), Is.EqualTo(30));
        }

        [Test]
        public void Parse_EmptyData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[0];

            // Act
            var chunk = new MsviChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetIndexCount(), Is.EqualTo(0));
        }

        [Test]
        public void Parse_PartialData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[6]; // Not a multiple of 4

            // Act
            var chunk = new MsviChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetIndexCount(), Is.EqualTo(1));
            Assert.That(chunk.GetIndex(0), Is.EqualTo(BitConverter.ToUInt32(data, 0)));
        }

        [Test]
        public void GetIndex_OutOfRange_ThrowsException()
        {
            // Arrange
            byte[] data = new byte[8]; // 2x 4-byte indices
            BitConverter.GetBytes(10).CopyTo(data, 0);
            BitConverter.GetBytes(20).CopyTo(data, 4);
            var chunk = new MsviChunk(data);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => chunk.GetIndex(2));
        }

        [Test]
        public void Write_ReturnsSameData()
        {
            // Arrange
            byte[] data = new byte[16]; // 4x 4-byte indices
            BitConverter.GetBytes(10).CopyTo(data, 0);
            BitConverter.GetBytes(20).CopyTo(data, 4);
            BitConverter.GetBytes(30).CopyTo(data, 8);
            BitConverter.GetBytes(40).CopyTo(data, 12);
            
            var chunk = new MsviChunk(data);

            // Act
            byte[] result = chunk.Write();

            // Assert
            Assert.That(result, Is.EqualTo(data));
        }

        [Test]
        public void GetSignature_ReturnsMSVI()
        {
            // Arrange
            var chunk = new MsviChunk(new byte[0]);

            // Act
            string signature = chunk.GetSignature();

            // Assert
            Assert.That(signature, Is.EqualTo("MSVI"));
        }
    }
} 