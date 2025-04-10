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
        public string ChunkId { get; set; }

        /// <summary>
        /// Array of supported versions for this chunk
        /// </summary>
        public int[] SupportedVersions { get; set; }

        /// <summary>
        /// Dictionary of field names and their specifications
        /// </summary>
        public Dictionary<string, FieldSpecification> Fields { get; set; }

        /// <summary>
        /// Validation rules for this chunk
        /// </summary>
        public ValidationRules Rules { get; set; }

        /// <summary>
        /// Dependencies on other chunks
        /// </summary>
        public string[] Dependencies { get; set; }

        /// <summary>
        /// Description of the chunk's purpose and usage
        /// </summary>
        public string Description { get; set; }
    }

    /// <summary>
    /// Represents a field specification within a chunk
    /// </summary>
    public class FieldSpecification
    {
        /// <summary>
        /// The data type of the field
        /// </summary>
        public string DataType { get; set; }

        /// <summary>
        /// The size of the field in bytes
        /// </summary>
        public int Size { get; set; }

        /// <summary>
        /// Description of the field's purpose
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Version-specific information about this field
        /// </summary>
        public Dictionary<int, string> VersionNotes { get; set; }
    }

    /// <summary>
    /// Represents validation rules for a chunk
    /// </summary>
    public class ValidationRules
    {
        /// <summary>
        /// Size constraints for the chunk
        /// </summary>
        public SizeConstraints Size { get; set; }

        /// <summary>
        /// Value constraints for specific fields
        /// </summary>
        public Dictionary<string, ValueConstraints> FieldConstraints { get; set; }

        /// <summary>
        /// Relationships with other chunks that must be validated
        /// </summary>
        public RelationshipConstraints[] Relationships { get; set; }
    }

    /// <summary>
    /// Represents size constraints for a chunk
    /// </summary>
    public class SizeConstraints
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
    public class ValueConstraints
    {
        /// <summary>
        /// Minimum allowed value
        /// </summary>
        public object MinValue { get; set; }

        /// <summary>
        /// Maximum allowed value
        /// </summary>
        public object MaxValue { get; set; }

        /// <summary>
        /// Array of allowed values
        /// </summary>
        public object[] AllowedValues { get; set; }

        /// <summary>
        /// Custom validation expression
        /// </summary>
        public string ValidationExpression { get; set; }
    }

    /// <summary>
    /// Represents relationships between chunks that must be validated
    /// </summary>
    public class RelationshipConstraints
    {
        /// <summary>
        /// The related chunk's identifier
        /// </summary>
        public string RelatedChunkId { get; set; }

        /// <summary>
        /// The type of relationship
        /// </summary>
        public RelationshipType Type { get; set; }

        /// <summary>
        /// Description of the relationship constraint
        /// </summary>
        public string Description { get; set; }
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