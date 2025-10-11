using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// DHSM chunk (MSHD reversed)
    /// </summary>
    public class DHSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "DHSM";
        
        /// <summary>
        /// First header value (0x00)
        /// </summary>
        public uint HeaderValue1 { get; set; }
        
        /// <summary>
        /// Second header value (0x04)
        /// </summary>
        public uint HeaderValue2 { get; set; }
        
        /// <summary>
        /// Third header value (0x08)
        /// </summary>
        public uint HeaderValue3 { get; set; }
        
        /// <summary>
        /// Placeholder values (0x0C-0x20)
        /// According to documentation, always 0 in version_48, likely placeholders.
        /// </summary>
        public uint[] Placeholders { get; } = new uint[5];
        
        /// <summary>
        /// Initializes a new instance of the <see cref="DHSM"/> class.
        /// </summary>
        public DHSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="DHSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public DHSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the DHSM data according to documentation.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            // Read header values as per documentation
            HeaderValue1 = reader.ReadUInt32();
            HeaderValue2 = reader.ReadUInt32();
            HeaderValue3 = reader.ReadUInt32();
            
            // Read 5 placeholder values (always 0 in version_48)
            for (int i = 0; i < 5; i++)
            {
                Placeholders[i] = reader.ReadUInt32();
            }
        }
        
        /// <summary>
        /// Gets a detailed description of the chunk contents.
        /// </summary>
        /// <returns>A detailed description string.</returns>
        public override string GetDetailedDescription()
        {
            if (!IsParsed)
            {
                return $"DHSM chunk: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine("DHSM header values:");
            sb.AppendLine($"  Value1 (0x00): 0x{HeaderValue1:X8} ({HeaderValue1})");
            sb.AppendLine($"  Value2 (0x04): 0x{HeaderValue2:X8} ({HeaderValue2})");
            sb.AppendLine($"  Value3 (0x08): 0x{HeaderValue3:X8} ({HeaderValue3})");
            
            sb.AppendLine("  Placeholders (0x0C-0x20):");
            for (int i = 0; i < Placeholders.Length; i++)
            {
                sb.AppendLine($"    [0x{0x0C + (i * 4):X2}]: 0x{Placeholders[i]:X8} ({Placeholders[i]})");
            }
            
            return sb.ToString();
        }
    }
} 