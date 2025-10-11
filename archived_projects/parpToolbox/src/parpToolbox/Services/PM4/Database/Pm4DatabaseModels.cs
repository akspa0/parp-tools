using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Numerics;

namespace ParpToolbox.Services.PM4.Database
{
    /// <summary>
    /// Represents a PM4 file in the database with metadata.
    /// </summary>
    public class Pm4File
    {
        [Key]
        public int Id { get; set; }
        
        [Required]
        public string FileName { get; set; } = string.Empty;
        
        [Required]
        public string FilePath { get; set; } = string.Empty;
        
        public DateTime ProcessedAt { get; set; }
        
        public int TotalVertices { get; set; }
        public int TotalTriangles { get; set; }
        public int TotalSurfaces { get; set; }
        public int TotalLinks { get; set; }
        public int TotalPlacements { get; set; }
        
        // Navigation properties
        public virtual ICollection<Pm4Vertex> Vertices { get; set; } = new List<Pm4Vertex>();
        public virtual ICollection<Pm4Triangle> Triangles { get; set; } = new List<Pm4Triangle>();
        public virtual ICollection<Pm4Surface> Surfaces { get; set; } = new List<Pm4Surface>();
        public virtual ICollection<Pm4Link> Links { get; set; } = new List<Pm4Link>();
        public virtual ICollection<Pm4Placement> Placements { get; set; } = new List<Pm4Placement>();
        public virtual ICollection<Pm4SurfaceGroup> SurfaceGroups { get; set; } = new List<Pm4SurfaceGroup>();
    }
    
    /// <summary>
    /// Represents a vertex (MSVT) in the database.
    /// </summary>
    public class Pm4Vertex
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        public int GlobalIndex { get; set; }
        
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
        
        public string ChunkType { get; set; } = "MSVT"; // MSVT, MSPV, etc.
        
        // Navigation properties
        [ForeignKey("Pm4FileId")]
        public virtual Pm4File Pm4File { get; set; } = null!;
        
        public virtual ICollection<Pm4TriangleVertex> TriangleVertices { get; set; } = new List<Pm4TriangleVertex>();
    }
    
    /// <summary>
    /// Represents a triangle in the database.
    /// </summary>
    public class Pm4Triangle
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        public int GlobalIndex { get; set; }
        
        public int VertexA { get; set; }
        public int VertexB { get; set; }
        public int VertexC { get; set; }
        
        // Navigation properties
        [ForeignKey("Pm4FileId")]
        public virtual Pm4File Pm4File { get; set; } = null!;
        
        public virtual ICollection<Pm4TriangleVertex> TriangleVertices { get; set; } = new List<Pm4TriangleVertex>();
    }
    
    /// <summary>
    /// Junction table for triangle-vertex relationships.
    /// </summary>
    public class Pm4TriangleVertex
    {
        [Key]
        public int Id { get; set; }
        
        public int TriangleId { get; set; }
        public int VertexId { get; set; }
        public int VertexPosition { get; set; } // 0=A, 1=B, 2=C
        
        // Navigation properties
        [ForeignKey("TriangleId")]
        public virtual Pm4Triangle Triangle { get; set; } = null!;
        
        [ForeignKey("VertexId")]
        public virtual Pm4Vertex Vertex { get; set; } = null!;
    }
    
    /// <summary>
    /// Represents a surface (MSUR) in the database.
    /// </summary>
    public class Pm4Surface
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        public int GlobalIndex { get; set; }
        
        public int MsviFirstIndex { get; set; }
        public int IndexCount { get; set; }
        public byte GroupKey { get; set; }
        public ushort RawFlags { get; set; }
        
        // Spatial bounds for clustering
        public float BoundsMinX { get; set; }
        public float BoundsMinY { get; set; }
        public float BoundsMinZ { get; set; }
        public float BoundsMaxX { get; set; }
        public float BoundsMaxY { get; set; }
        public float BoundsMaxZ { get; set; }
        public float BoundsCenterX { get; set; }
        public float BoundsCenterY { get; set; }
        public float BoundsCenterZ { get; set; }
        
        // Navigation properties
        [ForeignKey("Pm4FileId")]
        public virtual Pm4File Pm4File { get; set; } = null!;
        
        public virtual ICollection<Pm4SurfaceGroupMember> SurfaceGroupMembers { get; set; } = new List<Pm4SurfaceGroupMember>();
    }
    
    /// <summary>
    /// Represents a link (MSLK) in the database.
    /// </summary>
    public class Pm4Link
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        public int GlobalIndex { get; set; }
        
        // MSLK fields (using reflection-safe approach)
        public uint ParentIndex { get; set; }
        public int MspiFirstIndex { get; set; }
        public int MspiIndexCount { get; set; }
        public uint ReferenceIndex { get; set; }
        
        // Raw field data for flexibility
        public string RawFieldsJson { get; set; } = string.Empty;
        
        // Navigation properties
        [ForeignKey("Pm4FileId")]
        public virtual Pm4File Pm4File { get; set; } = null!;
    }
    
    /// <summary>
    /// Represents a placement (MPRL) in the database.
    /// </summary>
    public class Pm4Placement
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        public int GlobalIndex { get; set; }
        
        public float PositionX { get; set; }
        public float PositionY { get; set; }
        public float PositionZ { get; set; }
        
        public uint Unknown4 { get; set; } // Links to MSLK.ParentIndex
        public uint Unknown6 { get; set; }
        
        // Raw field data for flexibility
        public string RawFieldsJson { get; set; } = string.Empty;
        
        // Navigation properties
        [ForeignKey("Pm4FileId")]
        public virtual Pm4File Pm4File { get; set; } = null!;
    }
    
    /// <summary>
    /// Represents a property record (MPRR) in the database.
    /// MPRR contains the true building boundaries using sentinel values (Value1=65535).
    /// </summary>
    public class Pm4Property
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        public int GlobalIndex { get; set; }
        
        /// <summary>
        /// First ushort value. When Value1=65535, acts as sentinel marker indicating object boundary.
        /// </summary>
        public ushort Value1 { get; set; }
        
        /// <summary>
        /// Second ushort value. When following a sentinel marker, identifies component type.
        /// </summary>
        public ushort Value2 { get; set; }
        
        /// <summary>
        /// True if this entry is a building boundary sentinel (Value1=65535).
        /// </summary>
        public bool IsBoundarySentinel { get; set; }
        
        // Navigation properties
        [ForeignKey(nameof(Pm4FileId))]
        public virtual Pm4File Pm4File { get; set; } = null!;
    }
    
    /// <summary>
    /// Represents a surface group (spatial clustering result).
    /// </summary>
    public class Pm4SurfaceGroup
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        public int GroupIndex { get; set; }
        
        public string GroupName { get; set; } = string.Empty;
        public string ClusteringMethod { get; set; } = string.Empty; // "Spatial", "Hierarchical", etc.
        
        // Group bounds
        public float BoundsMinX { get; set; }
        public float BoundsMinY { get; set; }
        public float BoundsMinZ { get; set; }
        public float BoundsMaxX { get; set; }
        public float BoundsMaxY { get; set; }
        public float BoundsMaxZ { get; set; }
        public float BoundsCenterX { get; set; }
        public float BoundsCenterY { get; set; }
        public float BoundsCenterZ { get; set; }
        
        public int SurfaceCount { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        
        // Navigation properties
        [ForeignKey("Pm4FileId")]
        public virtual Pm4File Pm4File { get; set; } = null!;
        
        public virtual ICollection<Pm4SurfaceGroupMember> Members { get; set; } = new List<Pm4SurfaceGroupMember>();
    }
    
    /// <summary>
    /// Junction table for surface group membership.
    /// </summary>
    public class Pm4SurfaceGroupMember
    {
        [Key]
        public int Id { get; set; }
        
        public int SurfaceGroupId { get; set; }
        public int SurfaceId { get; set; }
        
        // Navigation properties
        [ForeignKey(nameof(SurfaceGroupId))]
        public virtual Pm4SurfaceGroup SurfaceGroup { get; set; } = null!;
        
        [ForeignKey(nameof(SurfaceId))]
        public virtual Pm4Surface Surface { get; set; } = null!;
    }
    
    /// <summary>
    /// Stores raw chunk data for future-proofing as PM4 understanding evolves.
    /// Enables new tools to leverage existing raw data without re-parsing original files.
    /// </summary>
    public class Pm4RawChunk
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        
        [Required]
        public string ChunkType { get; set; } = string.Empty; // 'MSLK', 'MSUR', 'MPRL', etc.
        
        public int ChunkOffset { get; set; }     // Position in original file
        public int ChunkSize { get; set; }       // Size in bytes
        public byte[] RawData { get; set; } = Array.Empty<byte>(); // Complete raw chunk data
        
        public DateTime ParsedAt { get; set; }
        public string ParserVersion { get; set; } = string.Empty; // Version of parser used
        
        // Optional: Parsed interpretation count for analysis
        public int? InterpretedRecordCount { get; set; }
        public string? ParsingNotes { get; set; }
        
        // Navigation properties
        [ForeignKey(nameof(Pm4FileId))]
        public virtual Pm4File Pm4File { get; set; } = null!;
    }
    
    /// <summary>
    /// Decoded hierarchical container representing a complete object assembled from related surfaces, links, and placements.
    /// </summary>
    public class Pm4HierarchicalContainer
    {
        [Key]
        public int Id { get; set; }
        
        public int Pm4FileId { get; set; }
        
        // Container hierarchy identifiers
        public float ContainerX { get; set; }  // BoundsCenterX value
        public float ContainerY { get; set; }  // BoundsCenterY value  
        public float ContainerZ { get; set; }  // BoundsCenterZ value
        
        // Object composition metrics
        public int SurfaceCount { get; set; }
        public int TotalTriangles { get; set; }
        public int RelatedLinkCount { get; set; }
        public int RelatedPlacementCount { get; set; }
        
        // Object completeness score (0-3: surfaces, links, placements)
        public int CompletenessScore { get; set; }
        
        // Object classification
        public string ObjectType { get; set; } = string.Empty;  // "building", "fragment", "detail", etc.
        public bool IsCompleteObject { get; set; }  // Based on triangle count and completeness
        
        // Spatial bounds (actual calculated bounds, not encoded)
        public float BoundsMinX { get; set; }
        public float BoundsMinY { get; set; }
        public float BoundsMinZ { get; set; }
        public float BoundsMaxX { get; set; }
        public float BoundsMaxY { get; set; }
        public float BoundsMaxZ { get; set; }
        
        // Navigation properties
        [ForeignKey("Pm4FileId")]
        public virtual Pm4File Pm4File { get; set; } = null!;
        
        public virtual ICollection<Pm4HierarchicalContainerMember> Members { get; set; } = new List<Pm4HierarchicalContainerMember>();
    }
    
    /// <summary>
    /// Junction table linking hierarchical containers to their member surfaces, links, and placements.
    /// </summary>
    public class Pm4HierarchicalContainerMember
    {
        [Key]
        public int Id { get; set; }
        
        public int ContainerId { get; set; }
        
        // Member type and ID
        public string MemberType { get; set; } = string.Empty;  // "Surface", "Link", "Placement"
        public int MemberId { get; set; }  // ID from respective table
        
        // Member contribution to object
        public int TriangleContribution { get; set; }  // For surfaces: IndexCount, others: 0
        
        // Navigation properties
        [ForeignKey("ContainerId")]
        public virtual Pm4HierarchicalContainer Container { get; set; } = null!;
    }
}
