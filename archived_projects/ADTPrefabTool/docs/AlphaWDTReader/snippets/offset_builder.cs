// docs/AlphaWDTReader/snippets/offset_builder.cs
// Purpose: 4-byte aligned block assembly with simple offset accounting.

using System;
using System.Collections.Generic;

namespace Snippets
{
    public sealed class OffsetBuilder
    {
        private int _pos;
        private readonly List<(string name,int start,int size)> _blocks = new();

        public int Position => _pos;
        public IReadOnlyList<(string name,int start,int size)> Blocks => _blocks;

        public int Align4()
        {
            int pad = (-_pos) & 3;
            _pos += pad;
            return pad;
        }

        public (int start,int size) Append(string name, int size)
        {
            Align4();
            int start = _pos;
            _pos += size;
            _blocks.Add((name,start,size));
            return (start,size);
        }
    }
}
