using ParpToolbox.Formats.PM4;
using System.Collections.Generic;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Loads a PM4 file and adapts it into an immutable <see cref="Pm4Scene"/> representation.
    /// This will later be replaced with a real implementation ported from Core.v2.
    /// </summary>
    public interface IPm4Loader
    {
        Pm4Scene Load(string path);
    }
}
