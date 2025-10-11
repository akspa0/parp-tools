using System;
using System.IO;
using Microsoft.Extensions.Logging;
using Moq;
using WCAnalyzer.Core.Files.ADT;
using WCAnalyzer.Core.Files.Interfaces;
using Xunit;

namespace WCAnalyzer.Core.Tests.Files.ADT
{
    public class ADTChunkFactoryTests
    {
        [Fact]
        public void CreateChunk_ShouldReturnMVERChunkForMVERSignature()
        {
            // Arrange
            string signature = "MVER";
            byte[] data = BitConverter.GetBytes(18); // Version 18
            var loggerMock = new Mock<ILogger>();
            var factory = new ADTChunkFactory(loggerMock.Object);

            // Act
            IChunk chunk = factory.CreateChunk(signature, data);

            // Assert
            Assert.IsType<MverChunk>(chunk);
            Assert.Equal(signature, chunk.Signature);
            Assert.Equal(data.Length, chunk.Size);
        }

        [Fact]
        public void CreateChunk_ShouldReturnMHDRChunkForMHDRSignature()
        {
            // Arrange
            string signature = "MHDR";
            byte[] data = new byte[64]; // Just some dummy data
            var loggerMock = new Mock<ILogger>();
            var factory = new ADTChunkFactory(loggerMock.Object);

            // Act
            IChunk chunk = factory.CreateChunk(signature, data);

            // Assert
            Assert.IsType<MhdrChunk>(chunk);
            Assert.Equal(signature, chunk.Signature);
            Assert.Equal(data.Length, chunk.Size);
        }

        [Fact]
        public void CreateChunk_ShouldReturnUnknownChunkForUnrecognizedSignature()
        {
            // Arrange
            string signature = "XXXX"; // Unknown signature
            byte[] data = new byte[4];
            var loggerMock = new Mock<ILogger>();
            var factory = new ADTChunkFactory(loggerMock.Object);

            // Act
            IChunk chunk = factory.CreateChunk(signature, data);

            // Assert
            Assert.IsType<UnknownADTChunk>(chunk);
            Assert.Equal(signature, chunk.Signature);
            Assert.Equal(data.Length, chunk.Size);
        }

        [Fact]
        public void CreateChunk_ShouldLogMessageForUnknownChunk()
        {
            // Arrange
            string signature = "XXXX"; // Unknown signature
            byte[] data = new byte[4];
            var loggerMock = new Mock<ILogger>();
            var factory = new ADTChunkFactory(loggerMock.Object);

            // Act
            factory.CreateChunk(signature, data);

            // Assert
            loggerMock.Verify(
                x => x.Log(
                    It.IsAny<LogLevel>(),
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Unknown ADT chunk type")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }

        [Fact]
        public void CreateChunk_ShouldHandleExceptionsDuringChunkCreation()
        {
            // Arrange
            string signature = "MVER";
            byte[] data = null; // This will cause an exception
            var loggerMock = new Mock<ILogger>();
            var factory = new ADTChunkFactory(loggerMock.Object);

            // Act
            IChunk chunk = factory.CreateChunk(signature, data);

            // Assert
            Assert.IsType<UnknownADTChunk>(chunk);
            Assert.Equal(signature, chunk.Signature);
            Assert.NotEmpty(chunk.Errors);
            loggerMock.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Error creating chunk")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }
    }
} 