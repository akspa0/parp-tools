using NUnit.Framework;
using System;
using WCAnalyzer.Core.Files.PM4.Chunks;

namespace WCAnalyzer.Core.Tests.Files.PM4.Chunks
{
    [TestFixture]
    public class MprrChunkTests
    {
        [Test]
        public void Parse_ValidData_ParsesCorrectly()
        {
            // Arrange
            byte[] data = new byte[8]; // 2x 4-byte records
            BitConverter.GetBytes((ushort)101).CopyTo(data, 0); // Value1 for record 1
            BitConverter.GetBytes((ushort)201).CopyTo(data, 2); // Value2 for record 1
            BitConverter.GetBytes((ushort)102).CopyTo(data, 4); // Value1 for record 2
            BitConverter.GetBytes((ushort)202).CopyTo(data, 6); // Value2 for record 2

            // Act
            var chunk = new MprrChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetRecordCount(), Is.EqualTo(2));
            
            var record1 = chunk.GetRecord(0);
            Assert.That(record1.Value1, Is.EqualTo(101));
            Assert.That(record1.Value2, Is.EqualTo(201));
            
            var record2 = chunk.GetRecord(1);
            Assert.That(record2.Value1, Is.EqualTo(102));
            Assert.That(record2.Value2, Is.EqualTo(202));
        }

        [Test]
        public void Parse_EmptyData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[0];

            // Act
            var chunk = new MprrChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetRecordCount(), Is.EqualTo(0));
        }

        [Test]
        public void Parse_PartialData_HandlesGracefully()
        {
            // Arrange
            byte[] data = new byte[6]; // Not a multiple of 4

            // Act
            var chunk = new MprrChunk(data);

            // Assert
            Assert.That(chunk.GetRawData(), Is.EqualTo(data));
            Assert.That(chunk.GetRecordCount(), Is.EqualTo(1));
            
            var record = chunk.GetRecord(0);
            Assert.That(record.Value1, Is.EqualTo(BitConverter.ToUInt16(data, 0)));
            Assert.That(record.Value2, Is.EqualTo(BitConverter.ToUInt16(data, 2)));
        }

        [Test]
        public void GetRecord_OutOfRange_ThrowsException()
        {
            // Arrange
            byte[] data = new byte[4]; // 1x 4-byte record
            var chunk = new MprrChunk(data);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => chunk.GetRecord(1));
        }

        [Test]
        public void Write_ReturnsSameData()
        {
            // Arrange
            byte[] data = new byte[16]; // 4x 4-byte records
            for (int i = 0; i < 4; i++)
            {
                BitConverter.GetBytes((ushort)(100 + i)).CopyTo(data, i * 4);
                BitConverter.GetBytes((ushort)(200 + i)).CopyTo(data, i * 4 + 2);
            }
            
            var chunk = new MprrChunk(data);

            // Act
            byte[] result = chunk.Write();

            // Assert
            Assert.That(result, Is.EqualTo(data));
        }

        [Test]
        public void GetSignature_ReturnsMPRR()
        {
            // Arrange
            var chunk = new MprrChunk(new byte[0]);

            // Act
            string signature = chunk.GetSignature();

            // Assert
            Assert.That(signature, Is.EqualTo("MPRR"));
        }
    }
} 