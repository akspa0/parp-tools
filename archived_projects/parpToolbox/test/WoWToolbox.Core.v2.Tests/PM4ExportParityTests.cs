using System.IO;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    // TODO: Port assertions from legacy WoWToolbox.Tests (OBJ SHA comparison, vertex counts, etc.)
    public class PM4ExportParityTests
    {
        [Fact(Skip = "Legacy parity checks not yet ported – placeholder test to reserve namespace.")]
        public void LegacyObjExporterParity_Placeholder()
        {
            // When porting, load PM4 sample, export OBJ via Core.v2 exporter, compare against committed golden file.
            Assert.True(true);
        }
    }
}
