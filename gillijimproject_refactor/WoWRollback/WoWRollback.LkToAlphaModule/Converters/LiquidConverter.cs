using System;
using System.IO;
using GillijimProject.Utilities;
using GillijimProject.WowFiles.Alpha;

namespace WoWRollback.LkToAlphaModule.Converters;

/// <summary>
/// Handles conversion of WotLK Liquid data (MH2O) to Alpha Liquid data (MCLQ).
/// </summary>
public sealed class LiquidConverter
{
    public static byte[]?[] ConvertMh2oToMclq(byte[] mh2oData)
    {
        // Wrapper around MclqBackporter
        return MclqBackporter.ConvertMh2oToMclqs(mh2oData);
    }
}
