using System;
using System.IO;
using Warcraft.NET; // placeholder: reference added at project level

namespace WoWRollback.LkToAlphaModule.AssetConversion;

public interface IModelConverter
{
    bool TryConvertM2ToMdx(Stream src, Stream dst, out string note);
}

public interface IWmoConverter
{
    bool TryConvertWmoToV14(Stream src, Stream dst, out string note);
}

public sealed class WarcraftNetModelConverter : IModelConverter
{
    public bool TryConvertM2ToMdx(Stream src, Stream dst, out string note)
    {
        // TODO: Implement real conversion via Warcraft.NET reader/writer.
        // For now, leave as a stub that signals not converted.
        note = "stub: not converted";
        return false;
    }
}

public sealed class WarcraftNetWmoConverter : IWmoConverter
{
    public bool TryConvertWmoToV14(Stream src, Stream dst, out string note)
    {
        // TODO: Implement real conversion via Warcraft.NET reader/writer.
        note = "stub: not converted";
        return false;
    }
}
