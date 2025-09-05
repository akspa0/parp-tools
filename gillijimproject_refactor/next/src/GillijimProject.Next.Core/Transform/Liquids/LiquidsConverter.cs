using System;
using GillijimProject.Next.Core.Domain.Liquids;

namespace GillijimProject.Next.Core.Transform.Liquids;

/// <summary>
/// Bidirectional converter for MH2O ↔ MCLQ liquid data.
/// This file provides compile-ready stubs; implementations will follow.
/// </summary>
public static class LiquidsConverter
{
    /// <summary>
    /// Converts an MH2O chunk (LK) to an MCLQ payload (Alpha) for a single MCNK.
    /// </summary>
    /// <exception cref="NotImplementedException">Implementation pending.</exception>
    public static MclqData Mh2oToMclq(Mh2oChunk src, LiquidsOptions opts)
    {
        throw new NotImplementedException("MH2O→MCLQ conversion not implemented yet.");
    }

    /// <summary>
    /// Converts an MCLQ payload (Alpha) to an MH2O chunk (LK) for a single MCNK.
    /// </summary>
    /// <exception cref="NotImplementedException">Implementation pending.</exception>
    public static Mh2oChunk MclqToMh2o(MclqData src, LiquidsOptions opts)
    {
        throw new NotImplementedException("MCLQ→MH2O conversion not implemented yet.");
    }
}
