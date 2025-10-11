using NUnit.Framework;
using System;
using System.Text;
using WCAnalyzer.Core.Files.Interfaces;
using WCAnalyzer.Core.Files.PD4;
using WCAnalyzer.Core.Files.PD4.Chunks;
using Microsoft.Extensions.Logging;
using Moq;

namespace WCAnalyzer.Core.Tests.Files.PD4
{
    [TestFixture]
    public class PD4ChunkFactoryTests
    {
        private PD4ChunkFactory _factory;
        private Mock<ILogger> _loggerMock;

        [SetUp]
        public void Setup()
        {
            _loggerMock = new Mock<ILogger>();
            _factory = new PD4ChunkFactory(_loggerMock.Object);
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMverChunk()
        {
            // Arrange
            string signature = "MVER";
            byte[] data = BitConverter.GetBytes(18);

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MverChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMcrcChunk()
        {
            // Arrange
            string signature = "MCRC";
            byte[] data = BitConverter.GetBytes(0x12345678);

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<McrcChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That((chunk as McrcChunk).CRC, Is.EqualTo(0x12345678));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMshdChunk()
        {
            // Arrange
            string signature = "MSHD";
            byte[] data = new byte[16]; // Sample data

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MshdChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMspvChunk()
        {
            // Arrange
            string signature = "MSPV";
            byte[] data = new byte[24]; // Sample data for 2 vertices

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MspvChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMspiChunk()
        {
            // Arrange
            string signature = "MSPI";
            byte[] data = new byte[12]; // Sample data for 3 indices

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MspiChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMscnChunk()
        {
            // Arrange
            string signature = "MSCN";
            byte[] data = new byte[16]; // Sample data for 1 vector

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MscnChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMslkChunk()
        {
            // Arrange
            string signature = "MSLK";
            byte[] data = new byte[16]; // Sample data

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MslkChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMsvtChunk()
        {
            // Arrange
            string signature = "MSVT";
            byte[] data = new byte[32]; // Sample data

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MsvtChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMsviChunk()
        {
            // Arrange
            string signature = "MSVI";
            byte[] data = new byte[12]; // Sample data for 3 indices

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MsviChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMsurChunk()
        {
            // Arrange
            string signature = "MSUR";
            byte[] data = new byte[16]; // Sample data

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MsurChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithUnknownSignature_CreatesUnknownChunk()
        {
            // Arrange
            string signature = "XXXX";
            byte[] data = new byte[4];

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<UnknownChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithNullSignature_CreatesUnknownChunk()
        {
            // Arrange
            string signature = null;
            byte[] data = new byte[4];

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<UnknownChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo("????"));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithNullData_CreatesUnknownChunk()
        {
            // Arrange
            string signature = "MVER";
            byte[] data = null;

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<UnknownChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(new byte[0]));
        }

        [Test]
        public void CreateChunk_WithId_CreatesMverChunk()
        {
            // Arrange
            string id = "MVER";

            // Act
            IChunk chunk = _factory.CreateChunk(id);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MverChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(id));
        }

        [Test]
        public void CreateChunk_WithId_CreatesMcrcChunk()
        {
            // Arrange
            string id = "MCRC";

            // Act
            IChunk chunk = _factory.CreateChunk(id);

            // Assert
            Assert.That(chunk, Is.InstanceOf<McrcChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(id));
        }

        [Test]
        public void CreateChunk_WithUnknownId_CreatesUnknownChunk()
        {
            // Arrange
            string id = "XXXX";

            // Act
            IChunk chunk = _factory.CreateChunk(id);

            // Assert
            Assert.That(chunk, Is.InstanceOf<UnknownChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(id));
        }
    }
} 