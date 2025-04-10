using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using WCAnalyzer.Core.Files.ADT;
using WCAnalyzer.Core.Services;
using Xunit;

namespace WCAnalyzer.Core.Tests.Services
{
    public class FileAnalyzerServiceTests
    {
        [Fact]
        public void AnalyzeFile_ShouldReturnNullWhenFilePathIsEmpty()
        {
            // Arrange
            var loggerMock = new Mock<ILogger>();
            var service = new FileAnalyzerService(loggerMock.Object);

            // Act
            var result = service.AnalyzeFile(string.Empty);

            // Assert
            Assert.Null(result);
            loggerMock.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("File path is null or empty")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }

        [Fact]
        public void AnalyzeFile_ShouldReturnNullWhenFileDoesNotExist()
        {
            // Arrange
            string filePath = "nonexistent.adt";
            var loggerMock = new Mock<ILogger>();
            var service = new FileAnalyzerService(loggerMock.Object);

            // Act
            var result = service.AnalyzeFile(filePath);

            // Assert
            Assert.Null(result);
            loggerMock.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("File not found")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
        }

        [Fact]
        public void AnalyzeFile_ShouldReturnNullForUnsupportedFileType()
        {
            // Arrange
            // Create a temporary file
            string tempFile = Path.GetTempFileName();
            try
            {
                // Change extension to something unsupported
                string unsupportedFile = Path.ChangeExtension(tempFile, ".xyz");
                File.Move(tempFile, unsupportedFile);

                var loggerMock = new Mock<ILogger>();
                var service = new FileAnalyzerService(loggerMock.Object);

                // Act
                var result = service.AnalyzeFile(unsupportedFile);

                // Assert
                Assert.Null(result);
                loggerMock.Verify(
                    x => x.Log(
                        LogLevel.Error,
                        It.IsAny<EventId>(),
                        It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Unsupported file format")),
                        It.IsAny<Exception>(),
                        It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                    Times.Once);

                // Clean up
                File.Delete(unsupportedFile);
            }
            catch
            {
                // Clean up if an exception occurs
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
                throw;
            }
        }

        [Fact]
        public void AnalyzeADTFile_ShouldReturnParsedADTFile()
        {
            // Arrange
            // Create a mock ADT file
            string tempFile = Path.GetTempFileName();
            try
            {
                string adtFile = Path.ChangeExtension(tempFile, ".adt");
                File.Move(tempFile, adtFile);

                // Write some minimal ADT file content
                using (var fs = new FileStream(adtFile, FileMode.Create))
                using (var writer = new BinaryWriter(fs))
                {
                    // MVER chunk
                    writer.Write(System.Text.Encoding.ASCII.GetBytes("MVER"));
                    writer.Write(4); // Size
                    writer.Write(18); // Version 18
                }

                var loggerMock = new Mock<ILogger>();
                var service = new FileAnalyzerService(loggerMock.Object);

                // Act
                var result = service.AnalyzeFile(adtFile);

                // Assert
                Assert.NotNull(result);
                Assert.IsType<ADTFile>(result);
                Assert.Equal(1, ((ADTFile)result).Chunks.Count);

                // Clean up
                File.Delete(adtFile);
            }
            catch
            {
                // Clean up if an exception occurs
                if (File.Exists(tempFile))
                    File.Delete(tempFile);
                throw;
            }
        }

        [Fact]
        public async Task AnalyzeADTFilesAsync_ShouldProcessMultipleFiles()
        {
            // Arrange
            string[] tempFiles = new string[3];
            string[] adtFiles = new string[3];
            
            try
            {
                // Create three temporary ADT files
                for (int i = 0; i < 3; i++)
                {
                    tempFiles[i] = Path.GetTempFileName();
                    adtFiles[i] = Path.ChangeExtension(tempFiles[i], ".adt");
                    File.Move(tempFiles[i], adtFiles[i]);

                    // Write some minimal ADT file content
                    using (var fs = new FileStream(adtFiles[i], FileMode.Create))
                    using (var writer = new BinaryWriter(fs))
                    {
                        // MVER chunk
                        writer.Write(System.Text.Encoding.ASCII.GetBytes("MVER"));
                        writer.Write(4); // Size
                        writer.Write(18); // Version 18
                    }
                }

                var loggerMock = new Mock<ILogger>();
                var service = new FileAnalyzerService(loggerMock.Object);

                // Act
                var results = await service.AnalyzeADTFilesAsync(adtFiles);

                // Assert
                Assert.NotNull(results);
                Assert.Equal(3, results.Count);
                foreach (var result in results)
                {
                    Assert.NotNull(result);
                    Assert.IsType<ADTFile>(result);
                    Assert.Equal(1, result.Chunks.Count);
                }

                // Clean up
                foreach (var file in adtFiles)
                {
                    if (File.Exists(file))
                        File.Delete(file);
                }
            }
            catch
            {
                // Clean up if an exception occurs
                foreach (var file in adtFiles)
                {
                    if (File.Exists(file))
                        File.Delete(file);
                }
                foreach (var file in tempFiles)
                {
                    if (File.Exists(file))
                        File.Delete(file);
                }
                throw;
            }
        }
    }
} 