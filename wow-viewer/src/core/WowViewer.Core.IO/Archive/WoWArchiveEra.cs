namespace WowViewer.Core.IO.Archive;

public static class WoWArchiveEra
{
    public static string Classify(string eraTag)
    {
        return eraTag switch
        {
            "0.X" => "Alpha",
            "1.X" => "Vanilla",
            "2.X" => "TBC",
            "3.X" => "Wrath",
            "4.X" => "Cata",
            _ => eraTag,
        };
    }
}
