namespace WoWToolbox.Core.v2.Models.PM4
{
    public class WmoMatchResult
    {
        public uint BuildingId { get; set; }
        public string WmoFilePath { get; set; } = string.Empty;
        public float Confidence { get; set; }
    }
}
