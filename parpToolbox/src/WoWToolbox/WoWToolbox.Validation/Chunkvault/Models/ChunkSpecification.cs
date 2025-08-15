using System.Collections.Generic;

namespace WoWToolbox.Validation.Chunkvault.Models
{
    /// <summary>
    /// Represents a chunk specification from the chunkvault documentation
    /// </summary>
    public class ChunkSpecification
    {
        /// <summary>
        /// The chunk identifier (e.g., "MVER", "MHDR")
        /// </summary>
        public string ChunkId { get; set; } = string.Empty;

        /// <summary>
        /// Array of supported versions for this chunk
        /// </summary>
        public List<string> SupportedVersions { get; set; } = new();

        /// <summary>
        /// Dictionary of field names and their specifications
        /// </summary>
        public List<FieldSpecification> Fields { get; set; } = new();

        /// <summary>
        /// Validation rules for this chunk
        /// </summary>
        public ChunkValidationRules? Rules { get; set; }

        /// <summary>
        /// Dependencies on other chunks
        /// </summary>
        public List<string> Dependencies { get; set; } = new();

        /// <summary>
        /// Description of the chunk's purpose and usage
        /// </summary>
        public string Description { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents a field specification within a chunk
    /// </summary>
    public class FieldSpecification
    {
        /// <summary>
        /// The data type of the field
        /// </summary>
        public string DataType { get; set; } = string.Empty;

        /// <summary>
        /// The size of the field in bytes
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Description of the field's purpose
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Version-specific information about this field
        /// </summary>
        public Dictionary<string, string> VersionNotes { get; set; } = new();
    }

    /// <summary>
    /// Represents validation rules for a chunk
    /// </summary>
    public class ChunkValidationRules
    {
        /// <summary>
        /// Size constraints for the chunk
        /// </summary>
        public ChunkSizeConstraints? Size { get; set; }

        /// <summary>
        /// Value constraints for specific fields
        /// </summary>
        public Dictionary<string, FieldValidationConstraints> FieldConstraints { get; set; } = new();

        /// <summary>
        /// Relationships with other chunks that must be validated
        /// </summary>
        public List<ChunkRelationshipConstraint> Relationships { get; set; } = new();
    }

    /// <summary>
    /// Represents size constraints for a chunk
    /// </summary>
    public class ChunkSizeConstraints
    {
        /// <summary>
        /// Minimum size in bytes
        /// </summary>
        public int? MinSize { get; set; }

        /// <summary>
        /// Maximum size in bytes
        /// </summary>
        public int? MaxSize { get; set; }

        /// <summary>
        /// Whether the size must be a multiple of a specific value
        /// </summary>
        public int? MultipleOf { get; set; }
    }

    /// <summary>
    /// Represents value constraints for a field
    /// </summary>
    public class FieldValidationConstraints
    {
        /// <summary>
        /// Minimum allowed value
        /// </summary>
        public string? MinValue { get; set; }

        /// <summary>
        /// Maximum allowed value
        /// </summary>
        public string? MaxValue { get; set; }

        /// <summary>
        /// Array of allowed values
        /// </summary>
        public List<string>? AllowedValues { get; set; }

        /// <summary>
        /// Custom validation expression
        /// </summary>
        public string? ValidationExpression { get; set; }
    }

    /// <summary>
    /// Represents relationships between chunks that must be validated
    /// </summary>
    public class ChunkRelationshipConstraint
    {
        /// <summary>
        /// The related chunk's identifier
        /// </summary>
        public string RelatedChunkId { get; set; } = string.Empty;

        /// <summary>
        /// The type of relationship
        /// </summary>
        public string RelationshipType { get; set; } = string.Empty;

        /// <summary>
        /// Description of the relationship constraint
        /// </summary>
        public string Description { get; set; } = string.Empty;
    }

    /// <summary>
    /// Types of relationships between chunks
    /// </summary>
    public enum RelationshipType
    {
        /// <summary>
        /// This chunk depends on the related chunk
        /// </summary>
        DependsOn,

        /// <summary>
        /// This chunk references data in the related chunk
        /// </summary>
        References,

        /// <summary>
        /// This chunk must appear before the related chunk
        /// </summary>
        MustPrecede,

        /// <summary>
        /// This chunk must appear after the related chunk
        /// </summary>
        MustFollow
    }
} 