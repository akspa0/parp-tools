using System;
using System.IO;
using System.Text;
using WCAnalyzer.Core.Common.Extensions;
using Xunit;

namespace WCAnalyzer.Core.Tests.Common
{
    public class BinaryWriterExtensionsTests
    {
        [Fact]
        public void WriteChunkSignature_ShouldWriteCorrectSignature()
        {
            // Arrange
            var memoryStream = new MemoryStream();
            var writer = new BinaryWriter(memoryStream);

            // Act
            writer.WriteChunkSignature("MVER");
            
            // Reset the position to read
            memoryStream.Position = 0;
            var reader = new BinaryReader(memoryStream);
            
            // Assert
            Assert.Equal("MVER", Encoding.ASCII.GetString(reader.ReadBytes(4)));
        }

        [Fact]
        public void WriteChunkSignature_ShouldThrowExceptionWhenSignatureIsInvalid()
        {
            // Arrange
            var memoryStream = new MemoryStream();
            var writer = new BinaryWriter(memoryStream);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => writer.WriteChunkSignature("MV")); // Too short
            Assert.Throws<ArgumentException>(() => writer.WriteChunkSignature("MVERXX")); // Too long
            Assert.Throws<ArgumentNullException>(() => writer.WriteChunkSignature(null)); // Null
        }

        [Fact]
        public void WriteNullTerminatedString_ShouldWriteCorrectString()
        {
            // Arrange
            var memoryStream = new MemoryStream();
            var writer = new BinaryWriter(memoryStream);

            // Act
            writer.WriteNullTerminatedString("Test String");
            
            // Reset the position to read
            memoryStream.Position = 0;
            var reader = new BinaryReader(memoryStream);
            
            // Assert
            var stringBytes = reader.ReadBytes((int)memoryStream.Length);
            Assert.Equal("Test String\0", Encoding.ASCII.GetString(stringBytes));
        }

        [Fact]
        public void WriteNullTerminatedString_ShouldHandleEmptyString()
        {
            // Arrange
            var memoryStream = new MemoryStream();
            var writer = new BinaryWriter(memoryStream);

            // Act
            writer.WriteNullTerminatedString(string.Empty);
            
            // Reset the position to read
            memoryStream.Position = 0;
            var reader = new BinaryReader(memoryStream);
            
            // Assert
            var stringBytes = reader.ReadBytes((int)memoryStream.Length);
            Assert.Equal("\0", Encoding.ASCII.GetString(stringBytes));
        }

        [Fact]
        public void WritePaddedString_ShouldWriteCorrectString()
        {
            // Arrange
            var memoryStream = new MemoryStream();
            var writer = new BinaryWriter(memoryStream);

            // Act
            writer.WritePaddedString("Test", 8);
            
            // Reset the position to read
            memoryStream.Position = 0;
            var reader = new BinaryReader(memoryStream);
            
            // Assert
            var stringBytes = reader.ReadBytes(8);
            Assert.Equal("Test\0\0\0\0", Encoding.ASCII.GetString(stringBytes));
        }

        [Fact]
        public void WritePaddedString_ShouldTruncateWhenStringExceedsLength()
        {
            // Arrange
            var memoryStream = new MemoryStream();
            var writer = new BinaryWriter(memoryStream);

            // Act
            writer.WritePaddedString("TestString123", 4);
            
            // Reset the position to read
            memoryStream.Position = 0;
            var reader = new BinaryReader(memoryStream);
            
            // Assert
            var stringBytes = reader.ReadBytes(4);
            Assert.Equal("Test", Encoding.ASCII.GetString(stringBytes));
        }
    }
} 