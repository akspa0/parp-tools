using NUnit.Framework;
using System;
using WCAnalyzer.Core.Files.PM4.Chunks;

namespace WCAnalyzer.Core.Tests.Files.PM4.Chunks
{
    [TestFixture]
    public class MscnChunkTests
    {
        [Test]
        public void Parse_ValidData_ParsesCorrectly()
        {
            // Arrange
            byte[] data = new byte[16]; // 4x 4-byte values
            BitConverter.GetBytes(1.0f).CopyTo(data, 0);  // Value 1
            BitConverter.GetBytes(2.0f).CopyTo(data, 4);  // Value 2
            BitConverter.GetBytes(3.0f).CopyTo(data, 8);  // Value 3
            BitConverter.GetBytes(4.0f).CopyTo(data, 12); // Value 4

            // Act
            var chunk = new MscnChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetVectorCount(), Is.EqualTo(1));
            
            var vector = chunk.GetVector(0);
            Assert.That(vector.X, Is.EqualTo(1.0f));
            Assert.That(vector.Y, Is.EqualTo(2.0f));
            Assert.That(vector.Z, Is.EqualTo(3.0f));
            Assert.That(vector.W, Is.EqualTo(4.0f));
        }

        [Test]
        public void Parse_EmptyData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[0];

            // Act
            var chunk = new MscnChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetVectorCount(), Is.EqualTo(0));
        }

        [Test]
        public void Parse_PartialData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[10]; // Not a multiple of 16

            // Act
            var chunk = new MscnChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetVectorCount(), Is.EqualTo(0));
        }

        [Test]
        public void Write_ReturnsSameData()
        {
            // Arrange
            byte[] data = new byte[32]; // 2x 4-byte vectors
            BitConverter.GetBytes(1.0f).CopyTo(data, 0);
            BitConverter.GetBytes(2.0f).CopyTo(data, 4);
            BitConverter.GetBytes(3.0f).CopyTo(data, 8);
            BitConverter.GetBytes(4.0f).CopyTo(data, 12);
            BitConverter.GetBytes(5.0f).CopyTo(data, 16);
            BitConverter.GetBytes(6.0f).CopyTo(data, 20);
            BitConverter.GetBytes(7.0f).CopyTo(data, 24);
            BitConverter.GetBytes(8.0f).CopyTo(data, 28);
            
            var chunk = new MscnChunk(data);

            // Act
            byte[] result = chunk.Write();

            // Assert
            Assert.That(result, Is.EqualTo(data));
        }

        [Test]
        public void GetSignature_ReturnsMSCN()
        {
            // Arrange
            var chunk = new MscnChunk(new byte[0]);

            // Act
            string signature = chunk.GetSignature();

            // Assert
            Assert.That(signature, Is.EqualTo("MSCN"));
        }
    }
} 