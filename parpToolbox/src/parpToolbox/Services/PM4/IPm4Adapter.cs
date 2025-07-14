using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Converts a PM4 file on disk to an in-memory <see cref="Pm4Scene"/> domain model.
    /// </summary>
    public interface IPm4Adapter
    {
        Pm4Scene Load(string path);
    }
}
