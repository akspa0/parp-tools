using System.IO;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Abstraction for loading <see cref="PM4File"/> instances from disk. Allows unit-testing and DI.
    /// </summary>
    public interface IPm4FileLoader
    {
        /// <summary>
        /// Loads and parses a PM4 file from <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Absolute or relative path to a *.pm4 file.</param>
        /// <returns>An initialized <see cref="PM4File"/> instance.</returns>
        PM4File Load(string path);
    }
}
