// DTOs for PM4 File Structure
using System.Collections.Generic;
using System.Numerics; // Required for Vector3
using WoWToolbox.Core.Navigation.PM4.Chunks; // Base chunk types if needed
using WoWToolbox.Core.Vectors; // For Vec3Short etc if used

namespace WoWToolbox.FileDumper.DTOs
{
    // Top-level DTO for the entire PM4 file
    public class Pm4FileDto
    {
        public MverChunkDto? MVER { get; set; }
        public MshdChunkDto? MSHD { get; set; }
        public MprlChunkDto? MPRL { get; set; }
        public MslkChunkDto? MSLK { get; set; }
        public MsvtChunkDto? MSVT { get; set; }
        public MspvChunkDto? MSPV { get; set; }
        public MspiChunkDto? MSPI { get; set; }
        public MsviChunkDto? MSVI { get; set; }
        public MsurChunkDto? MSUR { get; set; }
        public MdsfChunkDto? MDSF { get; set; }
        public MdosChunkDto? MDOS { get; set; }
        public MscnChunkDto? MSCN { get; set; }
        public MdbhChunkDto? MDBH { get; set; }
        public McrrChunkDto? MCRR { get; set; } // From Pd4? Check if relevant here or only in Core PM4File
        public MprrChunkDto? MPRR { get; set; } // From Pd4? Check if relevant here
        // Add other chunks if present in Core.PM4File (MCNK is usually ADT)
        public List<Vector3> ExteriorVertices { get; set; } = new();
    }

    // --- DTOs for individual chunk types ---

    // Example: MVER Chunk DTO
    public class MverChunkDto
    {
        public uint Version { get; set; }
    }

    // Example: MSHD Chunk DTO
    public class MshdChunkDto
    {
        public uint Flags { get; set; }
        public uint OffsetMCIN { get; set; }
        public uint OffsetMTEX { get; set; }
        public uint OffsetMMDX { get; set; }
        public uint OffsetMMID { get; set; }
        public uint OffsetMWMO { get; set; }
        public uint OffsetMWID { get; set; }
        public uint OffsetMDDF { get; set; }
        public uint OffsetMODF { get; set; }
        public uint OffsetMFBO { get; set; }
        public uint OffsetMH2O { get; set; }
        public uint OffsetMTXF { get; set; }
        public uint SizeMCIN { get; set; }
        public uint SizeMTEX { get; set; }
        public uint SizeMMDX { get; set; }
        public uint SizeMMID { get; set; }
        public uint SizeMWMO { get; set; }
        public uint SizeMWID { get; set; }
        public uint SizeMDDF { get; set; }
        public uint SizeMODF { get; set; }
        public uint SizeMFBO { get; set; }
        public uint SizeMH2O { get; set; }
        public uint SizeMTXF { get; set; }
        public uint TBD_0x78 { get; set; }
        public uint TBD_0x7C { get; set; }
        public uint TBD_0x80 { get; set; }
        public uint TBD_0x84 { get; set; }
    }

    // Example: MPRL Chunk DTO
    public class MprlEntryDto
    {
        // Based on MprlEntry definition in WoWToolbox.Core
        public Vector3 Position { get; set; }
        public uint Flags { get; set; }
    }
    public class MprlChunkDto
    {
        public List<MprlEntryDto> Entries { get; set; } = new();
    }

    // --- TODO: Define DTOs for ALL other PM4 chunks ---
    // (MSLK, MSVT, MSPV, MSPI, MSVI, MSUR, MDSF, MDOS, MSCN, MDBH, MCRR?, MPRR?)
    // These should mirror the structure and field names of the corresponding
    // classes in WoWToolbox.Core.Navigation.PM4.Chunks

    // Example (Partial): MSLK Chunk DTO
    public class MslkEntryDto
    {
        public byte Unknown_0x00 { get; set; }
        public byte Unknown_0x01 { get; set; }
        public ushort Unknown_0x02 { get; set; }
        public uint Unknown_0x04 { get; set; }
        public int MspiFirstIndex { get; set; }
        public byte MspiIndexCount { get; set; }
        public uint Unknown_0x0C { get; set; }
        public ushort Unknown_0x10 { get; set; }
        public ushort Unknown_0x12 { get; set; }
        // Add other fields if they exist in the Core MslkEntry struct/class
    }
    public class MslkChunkDto
    {
        public List<MslkEntryDto> Entries { get; set; } = new();
    }

     // Example (Partial): MSVT Chunk DTO
    public class MsvtVertexDto
    {
        // Assuming it's just a Vector3 based on PM4FileTests
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }
    public class MsvtChunkDto
    {
         public List<MsvtVertexDto> Vertices { get; set; } = new();
    }

     // Example (Partial): MSPV Chunk DTO
    public class MspvVertexDto
    {
        // Assuming it's just a Vector3 based on PM4FileTests
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }
    public class MspvChunkDto
    {
        public List<MspvVertexDto> Vertices { get; set; } = new();
    }

     // Example (Partial): MSPI Chunk DTO
    public class MspiChunkDto
    {
        public List<uint> Indices { get; set; } = new(); // Assuming uint based on PM4FileTests
    }

     // Example (Partial): MSVI Chunk DTO
    public class MsviChunkDto
    {
        public List<uint> Indices { get; set; } = new(); // Assuming uint based on PM4FileTests
    }

     // Example (Partial): MSUR Chunk DTO
    public class MsurEntryDto
    {
        // Based on MsurEntry in WoWToolbox.Core
        public byte FlagsOrUnknown_0x00 { get; set; }
        public byte IndexCount { get; set; }
        public ushort Unknown_0x02 { get; set; }
        public uint MsviFirstIndex { get; set; }
        public uint MdosIndex { get; set; }
    }
     public class MsurChunkDto
    {
        public List<MsurEntryDto> Entries { get; set; } = new();
    }

     // Example (Partial): MDSF Chunk DTO
    public class MdsfEntryDto
    {
         // Based on MdsfEntry in WoWToolbox.Core
        public uint msur_index { get; set; }
        public uint mdos_index { get; set; }
    }
    public class MdsfChunkDto
    {
        public List<MdsfEntryDto> Entries { get; set; } = new();
    }

     // Example (Partial): MDOS Chunk DTO
    public class MdosEntryDto
    {
        // Based on MdosEntry in WoWToolbox.Core
        public uint m_destructible_building_index { get; set; }
        public uint destruction_state { get; set; }
        // Add other fields if they exist
    }
    public class MdosChunkDto
    {
         public List<MdosEntryDto> Entries { get; set; } = new();
    }

     // Example (Partial): MSCN Chunk DTO
    // public class MscnVectorDataDto // REMOVED - MSCN only contains Vector3
    // {
    //     // Based on MscnVectorData in WoWToolbox.Core
    //     public float X { get; set; }
    //     public float Y { get; set; }
    //     public float Z { get; set; }
    //     public float NX { get; set; }
    //     public float NY { get; set; }
    //     public float NZ { get; set; }
    // }
     public class MscnChunkDto
    {
        // Corrected: List of System.Numerics.Vector3
        public List<Vector3> Vectors { get; set; } = new(); 
    }

     // Example (Partial): MDBH Chunk DTO
    public class MdbhEntryDto
    {
        // Based on MdbhEntry in WoWToolbox.Core
        public uint Index { get; set; }
        public string Filename { get; set; } = string.Empty;
    }
    public class MdbhChunkDto
    {
        public List<MdbhEntryDto> Entries { get; set; } = new();
    }

     // Example (Partial): MCRR Chunk DTO (If needed)
    public class McrrEntryDto
    {
        // Define based on McrrEntry in Core
    }
    public class McrrChunkDto
    {
        // public List<McrrEntryDto> Entries { get; set; } = new();
    }

     // Example (Partial): MPRR Chunk DTO (If needed)
    public class MprrEntryDto
    {
        // Define based on MprrEntry in Core
        public ushort Unknown_0x00 { get; set; }
        public ushort Unknown_0x02 { get; set; }
    }
    public class MprrChunkDto
    {
        public List<MprrEntryDto> Entries { get; set; } = new();
    }

    // Add other chunk DTOs as needed...

} 