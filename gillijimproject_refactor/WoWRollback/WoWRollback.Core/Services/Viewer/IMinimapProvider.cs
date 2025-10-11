using System;
using System.Collections.Generic;
using System.IO;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Abstraction for providing minimap tiles from arbitrary backends (filesystem, MPQs, etc.).
/// The provider is expected to enumerate all available tiles for a specific game version.
/// </summary>
public interface IMinimapProvider
{
    /// <summary>
    /// Enumerate available tiles. The returned tuples must provide map name, X(col), Y(row), and a stream opener.
    /// The opener must return a readable stream positioned at 0 and the caller will dispose it.
    /// </summary>
    IEnumerable<(string Map, int X, int Y, Func<Stream> Open)> Enumerate();
}
