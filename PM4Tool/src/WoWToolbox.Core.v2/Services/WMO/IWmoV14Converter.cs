using System;

namespace WoWToolbox.Core.v2.Services.WMO
{
    /// <summary>
    /// Service contract for converting legacy (pre-TBC) WMO version 14 files to the modern
    /// version 17 format used in Wrath of the Lich King and later patches.
    /// </summary>
    public interface IWmoV14Converter
    {
        /// <summary>
        /// Converts an in-memory v14 WMO byte buffer to the v17 format.
        /// </summary>
        /// <param name="v14Data">Raw bytes of a v14 WMO file.</param>
        /// <returns>Byte array containing a fully converted v17 WMO file.</returns>
        byte[] ConvertToV17(byte[] v14Data, string? textureSourceDir = null, string? textureOutputDir = null);

        /// <summary>
        /// Converts a v14 WMO file to v17 and exports the first group as OBJ. The OBJ path defaults to the
        /// project_output directory when omitted or when a directory is passed.
        /// Returns the full OBJ path written.
        /// </summary>
        string ExportFirstGroupAsObj(string wmoPath, string? objPath = null);

        /// <summary>
        /// Convenience wrapper around <see cref="ConvertToV17(byte[])"/> that works with file paths.
        /// </summary>
        /// <param name="inputPath">Absolute path to the source v14 WMO.</param>
        /// <param name="outputPath">Absolute path for the resulting v17 WMO.</param>
        void ConvertToV17(string inputPath, string outputPath, string? textureOutputDir = null);
    }
}
