using System.Collections.Generic;
using System.IO;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public class MPRRChunk
    {
        public List<List<ushort>> Sequences { get; private set; } = new();

        public void Read(BinaryReader br, long size)
        {
            if (size == 0) return;

            var endPosition = br.BaseStream.Position + size;
            
            while (br.BaseStream.Position < endPosition)
            {
                var currentSequence = new List<ushort>();
                
                while (br.BaseStream.Position < endPosition)
                {
                    var value = br.ReadUInt16();
                    currentSequence.Add(value);

                    if (value == 0xFFFF)
                    {
                        break; // End of this sequence
                    }
                }
                
                if (currentSequence.Count > 0)
                {
                    Sequences.Add(currentSequence);
                }
            }
        }
    }
} 