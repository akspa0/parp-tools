using System;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4
{
    /// <inheritdoc/>
    public sealed class Pm4Adapter : IPm4Adapter
    {
        public Pm4Scene Load(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("PM4 path must be provided", nameof(path));

            // TODO: Implement real PM4 parser using lifted chunk classes.
            // For now we simply throw so callers know it is incomplete.
            throw new NotImplementedException("PM4 parsing not yet implemented – awaiting chunk port.");
        }
    }
}
