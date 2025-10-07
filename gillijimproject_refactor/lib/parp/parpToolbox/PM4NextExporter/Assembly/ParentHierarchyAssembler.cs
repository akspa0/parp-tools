using System.Collections.Generic;
using PM4NextExporter.Model;

namespace PM4NextExporter.Assembly
{
    internal sealed class ParentHierarchyAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            // TODO: Implement ParentIndex_0x04-based assembly
            return new List<AssembledObject>();
        }
    }
}
