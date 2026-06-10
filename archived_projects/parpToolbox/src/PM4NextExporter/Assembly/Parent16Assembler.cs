using System.Collections.Generic;
using PM4NextExporter.Model;

namespace PM4NextExporter.Assembly
{
    internal sealed class Parent16Assembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            // TODO: Implement ParentId 16/16 split-based assembly
            return new List<AssembledObject>();
        }
    }
}
