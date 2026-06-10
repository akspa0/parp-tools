namespace WoWToolbox.Core.Navigation.PM4.Models
{
    /// <summary>
    /// DTO for MSLK navigation node entries.
    /// Represents hierarchical navigation structure with parent/child relationships.
    /// </summary>
    public class MslkNodeEntryDto
    {
        public int EntryIndex { get; set; }
        public byte Unk00 { get; set; }
        public byte Unk01 { get; set; }
        public ushort Unk02 { get; set; }
        public uint Unk0C { get; set; }
        public ushort Unk10 { get; set; }
        public ushort Unk12 { get; set; }
    }

    /// <summary>
    /// DTO for MSLK geometry entries.
    /// Links MSPI indices to geometry data for building extraction.
    /// </summary>
    public class MslkGeometryEntryDto
    {
        public int EntryIndex { get; set; }
        public int MspiFirstIndex { get; set; }
        public byte MspiIndexCount { get; set; }
        public byte Unk00 { get; set; }
        public byte Unk01 { get; set; }
        public ushort Unk02 { get; set; }
        public uint Unk0C { get; set; }
        public ushort Unk10 { get; set; }
        public ushort Unk12 { get; set; }
    }

    /// <summary>
    /// Groups MSLK nodes and geometry for building analysis.
    /// Used in hierarchical building extraction processes.
    /// </summary>
    public class MslkGroupDto
    {
        public List<MslkNodeEntryDto> Nodes { get; set; } = new List<MslkNodeEntryDto>();
        public List<MslkGeometryEntryDto> Geometry { get; set; } = new List<MslkGeometryEntryDto>();
    }
} 