using System;
using System.IO;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Concrete implementation of <see cref="IPm4FileLoader"/> that simply proxies to
    /// <see cref="PM4File.FromFile(string)"/>. Extracted to enable dependency-injection
    /// and mocking in higher-level services (e.g. <see cref="Pm4GlobalAnalyzer"/>).
    /// </summary>
    public sealed class Pm4FileLoader : IPm4FileLoader
    {
        public PM4File Load(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("PM4 path must be provided", nameof(path));
            if (!File.Exists(path))
                throw new FileNotFoundException($"PM4 file not found: {path}", path);

            return PM4File.FromFile(path);
        }
    }
}
