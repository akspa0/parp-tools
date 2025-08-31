namespace GillijimProject.WowFiles;

public sealed class MphdAlpha
{
    public static bool IsPresent(ReadOnlySpan<byte> data)
    {
        // MPHD is optional in Alpha; presence is enough for v1
        return true;
    }
}
