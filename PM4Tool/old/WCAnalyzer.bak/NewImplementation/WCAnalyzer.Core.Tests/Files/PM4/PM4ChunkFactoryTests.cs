using NUnit.Framework;
using System;
using System.Text;
using WCAnalyzer.Core.Files.Interfaces;
using WCAnalyzer.Core.Files.PM4;
using WCAnalyzer.Core.Files.PM4.Chunks;
using Microsoft.Extensions.Logging;
using Moq;

namespace WCAnalyzer.Core.Tests.Files.PM4
{
    [TestFixture]
    public class PM4ChunkFactoryTests
    {
        private PM4ChunkFactory _factory;
        private Mock<ILogger> _loggerMock;

        [SetUp]
        public void Setup()
        {
            _loggerMock = new Mock<ILogger>();
            _factory = new PM4ChunkFactory(_loggerMock.Object);
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
        public void CreateChunk_WithSignatureAndData_CreatesMprlChunk()
        {
            // Arrange
            string signature = "MPRL";
            byte[] data = new byte[16]; // Sample data

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MprlChunk>());
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
        public void CreateChunk_WithSignatureAndData_CreatesMprrChunk()
        {
            // Arrange
            string signature = "MPRR";
            byte[] data = new byte[8]; // Sample data for 2 records

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MprrChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMdbhChunk()
        {
            // Arrange
            string signature = "MDBH";
            byte[] data = new byte[4]; // Minimal sample with just entry count
            BitConverter.GetBytes(0).CopyTo(data, 0); // No entries

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MdbhChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMdosChunk()
        {
            // Arrange
            string signature = "MDOS";
            byte[] data = new byte[16]; // Sample data for 2 entries

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MdosChunk>());
            Assert.That(chunk.GetSignature(), Is.EqualTo(signature));
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
        }

        [Test]
        public void CreateChunk_WithSignatureAndData_CreatesMdsfChunk()
        {
            // Arrange
            string signature = "MDSF";
            byte[] data = new byte[16]; // Sample data for 2 entries

            // Act
            IChunk chunk = _factory.CreateChunk(signature, data);

            // Assert
            Assert.That(chunk, Is.InstanceOf<MdsfChunk>());
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