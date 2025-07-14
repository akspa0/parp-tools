using System.Collections.Generic;
using ParpToolbox.Formats.WMO;

namespace ParpToolbox.Services.WMO
{
    /// <summary>
    /// Loads a WMO file (v14, v17 or later) and returns parsed groups plus texture names.
    /// </summary>
    public interface IWmoLoader
    {
        /// <param name="path">Local file path of .wmo root file</param>
        /// <returns>Tuple of texture names and parsed group list</returns>
        (IReadOnlyList<string> textures, IReadOnlyList<WmoGroup> groups) Load(string path, bool includeFacades = false);
    }
}
