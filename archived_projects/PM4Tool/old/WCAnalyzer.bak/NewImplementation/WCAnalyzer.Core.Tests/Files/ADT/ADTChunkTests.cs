using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using Moq;
using WCAnalyzer.Core.Files.ADT;
using WCAnalyzer.Core.Files.Interfaces;
using Xunit;

namespace WCAnalyzer.Core.Tests.Files.ADT
{
    public class ADTChunkTests
    {
        private class TestADTChunk : ADTChunk
        {
            public TestADTChunk(string signature, byte[] data, ILogger? logger = null) 
                : base(signature, data, logger)
            {
            }

            public override bool Parse()
            {
                // Simple implementation for testing
                if (Data == null || Data.Length < 4)
                {
                    Errors.Add("Data is null or too short");
                    return false;
                }
                
                // Just read a test value from the data
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    try
                    {
                        TestValue = reader.ReadInt32();
                        return true;
                    }
                    catch (Exception ex)
                    {
                        Errors.Add($"Error parsing chunk: {ex.Message}");
                        return false;
                    }
                }
            }

            public int TestValue { get; private set; }
        }

        [Fact]
        public void Constructor_ShouldInitializeProperties()
        {
            // Arrange
            string signature = "TEST";
            byte[] data = new byte[] { 1, 2, 3, 4 };
            var logger = new Mock<ILogger>();

            // Act
            var chunk = new TestADTChunk(signature, data, logger.Object);

            // Assert
            Assert.Equal(signature, chunk.Signature);
            Assert.Equal(data.Length, chunk.Size);
            Assert.Empty(chunk.Errors);
        }

        [Fact]
        public void Parse_ShouldReturnTrueForValidData()
        {
            // Arrange
            string signature = "TEST";
            byte[] data = BitConverter.GetBytes(12345); // Int32 value
            var chunk = new TestADTChunk(signature, data);

            // Act
            bool result = chunk.Parse();

            // Assert
            Assert.True(result);
            Assert.Equal(12345, chunk.TestValue);
            Assert.Empty(chunk.Errors);
        }

        [Fact]
        public void Parse_ShouldReturnFalseForInvalidData()
        {
            // Arrange
            string signature = "TEST";
            byte[] data = new byte[] { 1 }; // Too short for Int32
            var chunk = new TestADTChunk(signature, data);

            // Act
            bool result = chunk.Parse();

            // Assert
            Assert.False(result);
            Assert.Single(chunk.Errors);
            Assert.Contains("Data is null or too short", chunk.Errors);
        }

        [Fact]
        public void Parse_ShouldLogErrorsWhenLoggerIsProvided()
        {
            // Arrange
            string signature = "TEST";
            byte[] data = new byte[] { 1 }; // Too short for Int32
            var loggerMock = new Mock<ILogger>();
            var chunk = new TestADTChunk(signature, data, loggerMock.Object);

            // Act
            bool result = chunk.Parse();

            // Assert
            Assert.False(result);
            Assert.Single(chunk.Errors);
            loggerMock.Verify(
                x => x.Log(
                    It.IsAny<LogLevel>(),
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Data is null or too short")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }

        [Fact]
        public void ToString_ShouldReturnCorrectString()
        {
            // Arrange
            string signature = "TEST";
            byte[] data = new byte[] { 1, 2, 3, 4 };
            var chunk = new TestADTChunk(signature, data);

            // Act
            string result = chunk.ToString();

            // Assert
            Assert.Contains(signature, result);
            Assert.Contains(data.Length.ToString(), result);
        }
    }
} 