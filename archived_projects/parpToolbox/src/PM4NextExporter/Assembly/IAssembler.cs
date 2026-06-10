using System.Collections.Generic;
using PM4NextExporter.Model;

namespace PM4NextExporter.Assembly
{
    internal interface IAssembler
    {
        IEnumerable<AssembledObject> Assemble(Scene scene, Options options);
    }
}
