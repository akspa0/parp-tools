using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging;
using Moq;
using WCAnalyzer.Core.Files.ADT;
using WCAnalyzer.Core.Files.Interfaces;
using Xunit;

namespace WCAnalyzer.Core.Tests.Files.ADT
{
    public class ADTFileTests
    {
        private byte[] CreateMockADTFileData()
        {
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                // MVER chunk
                writer.Write(System.Text.Encoding.ASCII.GetBytes("MVER"));
                writer.Write(4); // Size
                writer.Write(18); // Version 18
                
                // MHDR chunk
                writer.Write(System.Text.Encoding.ASCII.GetBytes("MHDR"));
                writer.Write(64); // Size
                
                // Write some dummy MHDR data
                for (int i = 0; i < 16; i++)
                {
                    writer.Write((int)i);
                }
                
                return ms.ToArray();
            }
        }
        
        [Fact]
        public void Parse_ShouldReadAllChunks()
        {
            // Arrange
            byte[] fileData = CreateMockADTFileData();
            var loggerMock = new Mock<ILogger>();
            var factoryMock = new Mock<IChunkFactory>();
            
            // Set up the factory to return mock chunks
            factoryMock.Setup(f => f.CreateChunk("MVER", It.IsAny<byte[]>()))
                .Returns(new MverChunk("MVER", new byte[] { 18, 0, 0, 0 }, loggerMock.Object));
            
            factoryMock.Setup(f => f.CreateChunk("MHDR", It.IsAny<byte[]>()))
                .Returns(new MhdrChunk("MHDR", new byte[64], loggerMock.Object));
            
            var adtFile = new ADTFile(factoryMock.Object, loggerMock.Object);

            // Act
            bool result = adtFile.Parse(fileData);

            // Assert
            Assert.True(result);
            Assert.Equal(2, adtFile.Chunks.Count);
            Assert.Contains(adtFile.Chunks, c => c.Signature == "MVER");
            Assert.Contains(adtFile.Chunks, c => c.Signature == "MHDR");
            Assert.Empty(adtFile.GetErrors());
        }
        
        [Fact]
        public void Parse_ShouldReturnFalseWhenFileDataIsNull()
        {
            // Arrange
            byte[] fileData = null;
            var loggerMock = new Mock<ILogger>();
            var factoryMock = new Mock<IChunkFactory>();
            var adtFile = new ADTFile(factoryMock.Object, loggerMock.Object);

            // Act
            bool result = adtFile.Parse(fileData);

            // Assert
            Assert.False(result);
            Assert.NotEmpty(adtFile.GetErrors());
            Assert.Contains("File data is null", adtFile.GetErrors().First());
        }
        
        [Fact]
        public void Parse_ShouldReturnFalseWhenFileDataIsTooShort()
        {
            // Arrange
            byte[] fileData = new byte[4]; // Too short for a valid chunk
            var loggerMock = new Mock<ILogger>();
            var factoryMock = new Mock<IChunkFactory>();
            var adtFile = new ADTFile(factoryMock.Object, loggerMock.Object);

            // Act
            bool result = adtFile.Parse(fileData);

            // Assert
            Assert.False(result);
            Assert.NotEmpty(adtFile.GetErrors());
            Assert.Contains("File data is too short", adtFile.GetErrors().First());
        }
        
        [Fact]
        public void GetVersionChunk_ShouldReturnMverChunk()
        {
            // Arrange
            byte[] fileData = CreateMockADTFileData();
            var loggerMock = new Mock<ILogger>();
            var adtFile = new ADTFile(new ADTChunkFactory(loggerMock.Object), loggerMock.Object);
            
            // Act
            adtFile.Parse(fileData);
            var versionChunk = adtFile.VersionChunk;

            // Assert
            Assert.NotNull(versionChunk);
            Assert.Equal("MVER", versionChunk.Signature);
            Assert.Equal(18, versionChunk.Version);
        }
        
        [Fact]
        public void GetHeaderChunk_ShouldReturnMhdrChunk()
        {
            // Arrange
            byte[] fileData = CreateMockADTFileData();
            var loggerMock = new Mock<ILogger>();
            var adtFile = new ADTFile(new ADTChunkFactory(loggerMock.Object), loggerMock.Object);
            
            // Act
            adtFile.Parse(fileData);
            var headerChunk = adtFile.HeaderChunk;

            // Assert
            Assert.NotNull(headerChunk);
            Assert.Equal("MHDR", headerChunk.Signature);
        }
        
        [Fact]
        public void Parse_ShouldHandleExceptionsDuringParsing()
        {
            // Arrange
            byte[] fileData = CreateMockADTFileData();
            var loggerMock = new Mock<ILogger>();
            var factoryMock = new Mock<IChunkFactory>();
            
            // Set up the factory to throw an exception
            factoryMock.Setup(f => f.CreateChunk(It.IsAny<string>(), It.IsAny<byte[]>()))
                .Throws(new Exception("Test exception"));
            
            var adtFile = new ADTFile(factoryMock.Object, loggerMock.Object);

            // Act
            bool result = adtFile.Parse(fileData);

            // Assert
            Assert.False(result);
            Assert.NotEmpty(adtFile.GetErrors());
            Assert.Contains("Error parsing file: Test exception", adtFile.GetErrors().First());
        }
    }
} 