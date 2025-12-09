using System;

namespace AlphaLkToAlphaStandalone.Core
{
    internal sealed class LkToAlphaConversionOptions
    {
        public string LkRootDirectory { get; }
        public string LkMapDirectory { get; }
        public string AlphaOutputDirectory { get; }
        public string DiagnosticsDirectory { get; }
        public string MapName { get; }

        public LkToAlphaConversionOptions(string lkRootDirectory, string lkMapDirectory, string alphaOutputDirectory, string diagnosticsDirectory, string mapName)
        {
            LkRootDirectory = lkRootDirectory ?? throw new ArgumentNullException(nameof(lkRootDirectory));
            LkMapDirectory = lkMapDirectory ?? throw new ArgumentNullException(nameof(lkMapDirectory));
            AlphaOutputDirectory = alphaOutputDirectory ?? throw new ArgumentNullException(nameof(alphaOutputDirectory));
            DiagnosticsDirectory = diagnosticsDirectory ?? throw new ArgumentNullException(nameof(diagnosticsDirectory));
            MapName = mapName ?? throw new ArgumentNullException(nameof(mapName));
        }
    }
}
