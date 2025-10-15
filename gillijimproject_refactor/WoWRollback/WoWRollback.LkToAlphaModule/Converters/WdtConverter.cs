namespace WoWRollback.LkToAlphaModule.Converters;

public sealed class WdtConverter
{
    public byte[] ConvertWdtMain(byte[] lkMainFlags)
    {
        // Alpha WDT: keep MAIN tile flags; strip MPHD/MWMO/MODF in writer
        return lkMainFlags;
    }
}
