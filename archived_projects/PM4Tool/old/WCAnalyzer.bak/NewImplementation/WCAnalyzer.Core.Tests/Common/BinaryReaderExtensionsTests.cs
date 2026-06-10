using System;
using System.IO;
using System.Text;
using WCAnalyzer.Core.Common.Extensions;
using Xunit;

namespace WCAnalyzer.Core.Tests.Common
{
    public class BinaryReaderExtensionsTests
    {
        [Fact]
        public void ReadChunkSignature_ShouldReturnCorrectSignature()
        {
            // Arrange
            byte[] data = Encoding.ASCII.GetBytes("MVER");
            var memoryStream = new MemoryStream(data);
            var reader = new BinaryReader(memoryStream);

            // Act
            string signature = reader.ReadChunkSignature();

            // Assert
            Assert.Equal("MVER", signature);
        }

        [Fact]
        public void ReadChunkSignature_ShouldThrowExceptionWhenStreamIsTooShort()
        {
            // Arrange
            byte[] data = Encoding.ASCII.GetBytes("MV");
            var memoryStream = new MemoryStream(data);
            var reader = new BinaryReader(memoryStream);

            // Act & Assert
            Assert.Throws<EndOfStreamException>(() => reader.ReadChunkSignature());
        }

        [Fact]
        public void ReadNullTerminatedString_ShouldReturnCorrectString()
        {
            // Arrange
            byte[] data = Encoding.ASCII.GetBytes("Test String\0ExtraData");
            var memoryStream = new MemoryStream(data);
            var reader = new BinaryReader(memoryStream);

            // Act
            string result = reader.ReadNullTerminatedString();

            // Assert
            Assert.Equal("Test String", result);
            Assert.Equal('E', (char)reader.PeekChar()); // Next character should be 'E' from "ExtraData"
        }

        [Fact]
        public void ReadNullTerminatedString_ShouldReturnEmptyStringForImmediateNull()
        {
            // Arrange
            byte[] data = new byte[] { 0, 65, 66, 67 }; // \0ABC
            var memoryStream = new MemoryStream(data);
            var reader = new BinaryReader(memoryStream);

            // Act
            string result = reader.ReadNullTerminatedString();

            // Assert
            Assert.Equal(string.Empty, result);
            Assert.Equal('A', (char)reader.PeekChar()); // Next character should be 'A'
        }

        [Fact]
        public void ReadPaddedString_ShouldReturnCorrectString()
        {
            // Arrange
            byte[] data = Encoding.ASCII.GetBytes("Test\0\0\0\0");
            var memoryStream = new MemoryStream(data);
            var reader = new BinaryReader(memoryStream);

            // Act
            string result = reader.ReadPaddedString(8);

            // Assert
            Assert.Equal("Test", result);
            Assert.Equal(8, memoryStream.Position); // Should have read all 8 bytes
        }

        [Fact]
        public void ReadPaddedString_ShouldTruncateWhenStringExceedsLength()
        {
            // Arrange
            byte[] data = Encoding.ASCII.GetBytes("TestString123");
            var memoryStream = new MemoryStream(data);
            var reader = new BinaryReader(memoryStream);

            // Act
            string result = reader.ReadPaddedString(4);

            // Assert
            Assert.Equal("Test", result);
            Assert.Equal(4, memoryStream.Position); // Should have read exactly 4 bytes
        }
    }
} 